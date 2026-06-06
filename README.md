# lob-engine: High-Frequency Limit Order Book Engine

A limit order book written in C++17 with O(1) order insertion and cancellation,
plus the pieces that make it useful for microstructure work: a Hawkes-process
order-flow simulator, the Kyle (1985) and Almgren-Chriss (2000) impact models, a
FIX 4.2 gateway, and Python bindings. It sustains over 8 million order operations
per second on a laptop.

The matching core uses the same design real exchanges run on: price-time
priority over an intrusive linked list, with a hash-map index that makes
cancellation constant-time.

```python
from lob_engine import OrderBook, HawkesSimulator

book = OrderBook(tick_size=0.01)
order_id = book.add_limit_order(side='bid', price=100.50, qty=100)
book.cancel_order(order_id)
fills = book.add_market_order(side='ask', qty=50)

print(book.best_bid(), book.best_ask(), book.mid_price())
print(book.depth(side='bid', levels=5))
print(book.order_book_imbalance(levels=3))
```

---

## Table of contents

- [Architecture](#architecture)
- [Why price-time priority and an intrusive linked list](#why-price-time-priority-and-an-intrusive-linked-list)
- [Hawkes order flow](#hawkes-order-flow)
- [Adverse selection: proof from the simulator](#adverse-selection-proof-from-the-simulator)
- [Market-impact models](#market-impact-models)
- [FIX protocol gateway](#fix-protocol-gateway)
- [Benchmarks](#benchmarks)
- [Getting started](#getting-started)
- [Python API](#python-api)
- [Project layout](#project-layout)
- [Further reading](#further-reading)

---

## Architecture

```
                         ┌──────────────────────────────────────────┐
                         │                OrderBook                  │
                         │                                           │
   add_limit_order ─────▶│   bids: std::map<int64_t, PriceLevel>     │
   add_market_order ────▶│         (best bid = highest = rbegin)     │
   cancel_order ────────▶│   asks: std::map<int64_t, PriceLevel>     │
                         │         (best ask = lowest  = begin)      │
                         │                                           │
                         │   lookup: unordered_map<id, Order*> ──┐   │
                         └───────────────────────────────────────┼───┘
                                                                 │ O(1)
        PriceLevel (one price)                                   │ cancel
        ┌───────────────────────────────────────────────┐       │
        │ head ─▶ Order ◀─▶ Order ◀─▶ Order ◀─▶ tail     │◀──────┘
        │        (intrusive doubly-linked list, FIFO)    │
        │        total_volume, order_count               │
        └───────────────────────────────────────────────┘
              ▲ orders allocated from a free-list OrderPool (no per-order malloc)

   ┌───────────────────┐   order flow   ┌───────────────────┐   λ̂, schedules
   │ HawkesSimulator   │ ─────────────▶ │ MarketSimulator   │ ─────────────▶ Kyle λ
   │ (Ogata thinning)  │   events       │ (book+fundamental)│              Almgren-Chriss
   └───────────────────┘                └───────────────────┘
                                  │
                                  ▼
                         pybind11  →  import lob_engine   (Python strategies)

   FIX 4.2 wire  ──▶  FixGateway  ──▶  OrderBook   ──▶  ExecutionReports (35=8)
   (8=...|35=D|...)   parse+route                       OrderCancelReject (35=9)
```

---

## Why price-time priority and an intrusive linked list

Price-time priority is the rule almost every lit exchange uses: fill the best
price first, and within a price, fill the oldest order first (FIFO). It is fair,
because the only way to jump the queue is to quote a better price, and it gives
traders a clear incentive to post early. Getting that rule exactly right is the
point of the engine. The rest is making it fast.

So the real question is which data structure holds the resting orders so that the
three hot operations are each O(1):

| Operation        | What it needs                                  | Our cost |
|------------------|------------------------------------------------|----------|
| add (resting)    | append to the back of a price level (FIFO)     | O(1)\*   |
| cancel           | find the order, splice it out of its level     | O(1)     |
| match (best px)  | read the best level, pop from the front        | O(1)     |

\* The asterisk: locating the price level is O(log P) in the `std::map`, where P
is the number of active price levels (small and bounded in practice). Once you
have the level, appending is O(1).

Constant-time cancellation comes from pairing two structures:

1. `std::unordered_map<order_id, Order*>` maps an order id straight to its node's
   address.
2. An intrusive doubly-linked list at each price level. The node *is* the
   `Order` (the `prev`/`next` pointers live in the struct), so once the hash map
   hands back the pointer, unlinking it is a handful of pointer writes with no
   search.

### What "intrusive" buys you over the naive version

A naive book keeps a `std::list<Order>` per level. Three problems come with that:

- Every order is its own heap allocation, because the list node wraps the order.
  At millions of orders a second, `malloc`/`free` becomes the bottleneck and the
  heap fragments.
- To cancel, you stash a `list::iterator` in the index. That works, but you are
  still paying for the per-node allocation.
- The nodes end up scattered across the heap, so walking a level misses cache
  constantly.

The intrusive version sidesteps all three:

- `prev`/`next` are fields of `Order`, so there is no wrapper node and no
  separate allocation per order.
- Orders are handed out by a free-list `OrderPool` that grabs memory in big
  blocks and recycles freed slots (the freed order's `next` pointer doubles as
  the free-list link). Once it has warmed up, the hot path never calls the system
  allocator.
- Unlinking is four pointer writes, and finding the node is a single hash lookup.

That is where the latency numbers come from (see [benchmarks](#benchmarks)):
roughly 90 ns to add a limit order, 180 ns to cancel, and north of 8M ops/sec on
a laptop-class CPU.

> **Why integer ticks?** Prices are `int64_t` ticks (dollars × 100, say), never
> floats. Floats break the one thing a matching engine cannot get wrong, namely
> comparing and ordering price levels, because a value like `1.10` has no exact
> binary representation. Integers keep the comparisons exact and keep `std::map`
> keys well-behaved.

---

## Hawkes order flow

Real order flow does not arrive as a steady, independent stream. It clusters. A
big sell prints, that nudges other participants to sell, the price moves, stops
trigger, and more selling follows. Trades beget trades. The standard way to model
this "an event now makes more events likely soon" behaviour is a self-exciting
point process, and the workhorse is the Hawkes process.

This is a 4-dimensional, mutually-exciting Hawkes process over `market_buy`,
`market_sell`, `limit_buy`, and `limit_sell`. Each type has a baseline rate
`μ_i`, and every past event raises all four future rates through an excitation
matrix `α`:

```
λ_i(t) = μ_i + Σ_j Σ_{t_k < t, type=j}  α_ij · β · exp(−β · (t − t_k))
```

Read it as: the intensity of type *i* right now is its baseline plus a decaying
bump from every past event, where a type-*j* event raises type-*i*'s rate by
`α_ij` and the bump fades on a `1/β` timescale.

- `α_ij` is the branching ratio: the expected number of type-*i* offspring that
  one type-*j* event sets off (the kernel `β·exp(−βt)` integrates to 1).
- For the process to stay finite, the spectral radius of `α` has to be below 1.
  The constructor checks this and throws if it is not.
- The long-run mean rates are closed-form: `λ_∞ = (I − α)⁻¹ μ`.

Events come from Ogata's thinning algorithm, which produces an exact sample path
rather than a time-discretised approximation:

1. Bound the total intensity by its current value `Λ = Σ_i λ_i(t)`. This is valid
   because between events the intensity only decays.
2. Draw a candidate time `t + Exponential(1/Λ)`.
3. Accept the candidate with probability `Λ(t_next)/Λ`. If accepted, pick the
   event type in proportion to its intensity, otherwise reject and try again.

Because the kernel is exponential, the intensity has a simple recursive update,
so each step is O(4) and the simulator spits out hundreds of thousands of events
in milliseconds.

**Validation.** Run it for a long horizon and the empirical event rates converge
to the closed-form stationary intensities:

```
[Hawkes validation] empirical vs closed-form rates (T=50000, 248873 events):
  market_buy    empirical=0.9868  closed-form=1.0000  rel.err=1.32%
  market_sell   empirical=0.9990  closed-form=1.0000  rel.err=0.10%
  limit_buy     empirical=1.4945  closed-form=1.5000  rel.err=0.37%
  limit_sell    empirical=1.4973  closed-form=1.5000  rel.err=0.18%
```

---

## Adverse selection: proof from the simulator

A latent fundamental price drifts with permanent impact: market buys push it up,
market sells push it down. Liquidity providers quote around that fundamental, so
when it walks away from a stale quote, the quote gets picked off. That is adverse
selection, the basic risk of making markets.

If the simulation is any good, signed order flow should predict price moves in the
same direction. The check is a regression of the per-bucket mid-price change on
the contemporaneous net market order flow. That slope is Kyle's λ:

```
[Adverse selection] OLS  mid_change ~ net_order_flow
  lambda_hat (price impact) = 0.03253 ticks/share
  intercept                 = -0.00036
  t-statistic               = 9.43
  R^2                       = 0.0018
  n (buckets)               = 49998
```

The slope is positive with a t-stat around 9.4, so net buying reliably pushes the
price up against resting liquidity, which is exactly what the theory predicts.
The R² is tiny, and that is expected: at this frequency, microstructure noise
swamps the variance of returns even though the impact is dead-on in expectation.

---

## Market-impact models

### Kyle (1985): market depth λ

Kyle's model makes price linear in signed order flow, `Δp = λ · q + noise`, so the
depth coefficient is

```
λ = Cov(Δp, q) / Var(q)
```

λ is how far the price moves per unit of net flow. A high λ is a thin market
where a little flow moves price a lot; a low λ is a deep one. `kyle_lambda(price_changes, order_flow)`
computes it, and on the simulated series it returns the same value as the
adverse-selection regression above. It has to, since `Cov/Var` is the regression
slope.

### Almgren-Chriss (2000): optimal liquidation

Selling `X` shares over a horizon `T` is a tug-of-war between two costs: temporary
impact (you pay a concession to trade fast) and timing risk (you carry variance
if you trade slow). Minimising `cost + λ·variance` has a closed-form solution in
hyperbolic functions:

```
κ̂² = λ σ² / η̃ ,   η̃ = η − ½ γ τ ,   κ = acosh(½ κ̂² τ² + 1) / τ
x_j = X · sinh(κ (T − t_j)) / sinh(κ T)
```

- Risk-neutral (`λ → 0`): `κ → 0` and the schedule flattens to a straight line,
  which is TWAP (equal slices).
- Risk-averse (`λ > 0`): the schedule front-loads, selling harder early to shed
  timing risk at the cost of more impact.

```
risk-neutral (TWAP): kappa=0.0000
  trade schedule (shares/slice): [100000 ×10]            E[cost]=2.61e6  Var=2.57e10
risk-averse:         kappa=1.8993
  trade schedule (shares/slice): [181738, 152133, 128032, 108563, 93022,
                                   80846, 71596, 64936, 60626, 58509]
                                                        E[cost]=3.01e6  Var=1.64e10
```

More urgency means a higher κ, which means less variance and more expected cost.
The schedule recovers that trade-off exactly.

---

## FIX protocol gateway

Exchanges and brokers talk FIX (Financial Information eXchange), a flat
`tag=value` format framed by `BeginString` (8), `BodyLength` (9), and `CheckSum`
(10). The engine ships a FIX 4.2-style `FixGateway` that parses inbound orders,
routes them to the book, and sends back the matching `ExecutionReport` (35=8) or
`OrderCancelReject` (35=9), the same exchange a real trading system would have
with a venue.

Supported inbound messages:

| MsgType | Name              | Action                                   |
|---------|-------------------|------------------------------------------|
| `D`     | NewOrderSingle    | `add_limit_order` / `add_market_order`   |
| `F`     | OrderCancelRequest| `cancel_order` by `OrigClOrdID`          |

```python
from lob_engine import OrderBook, FixGateway

book = OrderBook(tick_size=0.01)
gw = FixGateway(book=book)

# NewOrderSingle: buy 100 @ 100.50, then a market sell of 40.
gw.process("35=D|11=ORD1|54=1|38=100|44=100.50|40=2|", delim="|")
gw.process("35=D|11=ORD2|54=2|38=40|40=1|", delim="|")
# -> 8=FIX.4.2|9=100|35=8|49=LOB-ENGINE|56=CLIENT|17=E2|37=0|11=ORD2|54=2|
#    38=40|150=2|39=2|32=40|31=100.5000|14=40|151=0|10=053|
```

Every outgoing message gets a correct `BodyLength` and `CheckSum`
(`FixMessage::checksum_valid()` checks inbound frames), prices convert between
decimal and integer ticks at the boundary, and the gateway keeps the
`ClOrdID` to `engine order_id` mapping it needs to cancel a resting order.

---

## Benchmarks

Throughput comes from Google Benchmark; the latency percentiles come from a TSC
cycle-counting harness. Release build, MSVC 19.44, on a 24-core machine at about
2.0 GHz. These numbers move with your hardware, so regenerate them with
`bench_book.exe`.

**Throughput (Google Benchmark)**

| Operation                       | Throughput        |
|---------------------------------|-------------------|
| `add_limit_order`               | **8.36 M ops/sec** |
| `cancel_order`                  | **12.0 M ops/sec** |
| `add_market_order` (sweep 100)  | 37.0 M orders matched/sec |
| `add_market_order` (sweep 1000) | 41.7 M orders matched/sec |

The target was 1 M+/sec. This clears it by roughly 8x.

**Per-operation latency (nanoseconds, TSC-measured)**

| Operation                    |   p50 |   p95 |   p99 |  p99.9 |   mean |
|------------------------------|------:|------:|------:|-------:|-------:|
| `add_limit_order`            |  90.2 | 310.6 | 450.9 | 6762.8 |  215.3 |
| `cancel_order`               | 180.3 | 430.8 | 571.1 |  971.8 |  213.8 |
| `add_market_order` depth=1   |  90.2 | 130.2 | 150.3 |  230.4 |   96.5 |
| `add_market_order` depth=10  | 250.5 | 310.6 | 400.8 |  521.0 |  263.9 |
| `add_market_order` depth=100 | 2655  | 3426  | 3717  | 11341  |  2738  |
| `add_market_order` depth=1000| 23535 | 35698 | 43402 | 75123  | 25933  |

Market-order latency grows linearly with the number of resting orders it sweeps,
which is the O(k) you would expect for matching k orders. Everything else stays
O(1).

---

## Getting started

You need a C++17 compiler, CMake 3.16 or newer, and (for the Python module)
Python 3.8 or newer. The third-party dependencies (GoogleTest, Google Benchmark,
pybind11) are pulled in automatically by CMake `FetchContent`.

### Build and test the C++ engine

```bash
cmake -B build && cmake --build build && ctest --test-dir build --output-on-failure
```

### Build and run the benchmarks

```bash
cmake -B build -DLOB_BUILD_BENCHMARKS=ON
cmake --build build --target bench_book
./build/bench_book          # Windows: .\build\bench_book.exe
```

### Install the Python module and run the demo

```bash
pip install -e .            # scikit-build-core drives CMake under the hood
pytest tests/               # Python binding tests
python examples/demo.py     # end-to-end demo
```

On Windows, build from a Developer Command Prompt (or any shell with the MSVC
environment) so CMake can find the compiler, and bind against a 64-bit Python to
match a 64-bit build.

---

## Python API

```python
from lob_engine import OrderBook, HawkesSimulator, MarketSimulator, kyle_lambda, almgren_chriss

# Order book (float prices, integer-tick engine underneath)
book = OrderBook(tick_size=0.01)
oid  = book.add_limit_order(side='bid', price=100.50, qty=100)
book.add_limit_order(side='ask', price=100.55, qty=150)
book.cancel_order(oid)
fills = book.add_market_order(side='ask', qty=50)   # list of {maker_order_id, price, quantity}
book.best_bid(); book.best_ask(); book.mid_price(); book.spread()
book.depth(side='bid', levels=5)                    # [(price, volume), ...]
book.order_book_imbalance(levels=3)                 # (bid_vol-ask_vol)/(bid_vol+ask_vol)

# Hawkes order flow
mu    = [0.5, 0.5, 1.0, 1.0]            # market buy/sell, limit buy/sell
alpha = [[0.1]*4 for _ in range(4)]     # excitation matrix (spectral radius 0.4)
sim   = HawkesSimulator(mu=mu, alpha=alpha, beta=1.0, seed=42)
sim.spectral_radius(); sim.stationary_intensities()
events = sim.simulate(T=2000.0)         # [(time, 'market_buy'), ...]

# Coupled simulation + adverse-selection regression
res = MarketSimulator(bucket_dt=1.0).run(hawkes=sim, T=20000.0)
res['lambda_hat'], res['t_stat'], res['r_squared']

# Market impact
kyle_lambda(price_changes=[...], order_flow=[...])
schedule = almgren_chriss(total_shares=1_000_000, horizon=1.0, n_intervals=10,
                          sigma=0.3, eta=2.5e-6, gamma=2.5e-7, risk_aversion=1e-4)
schedule['trades'], schedule['kappa'], schedule['expected_cost'], schedule['variance']
```

---

## Project layout

```
lob-engine/
├── CMakeLists.txt          # FetchContent: GoogleTest, Google Benchmark, pybind11
├── pyproject.toml          # pip install -e .  (scikit-build-core -> CMake)
├── include/
│   ├── order.hpp           # intrusive Order struct + Fill
│   ├── order_book.hpp      # OrderPool, PriceLevel, OrderBook
│   ├── hawkes.hpp          # Hawkes + MarketSimulator
│   ├── impact.hpp          # Kyle + Almgren-Chriss
│   └── fix.hpp             # FIX message + gateway
├── src/
│   ├── order_book.cpp      # matching engine
│   ├── hawkes.cpp          # Ogata thinning, stationarity, adverse selection
│   ├── impact.cpp          # impact models
│   ├── fix.cpp             # FIX parser/encoder + gateway
│   └── bindings.cpp        # pybind11 module
├── tests/
│   ├── test_book.cpp       # GoogleTest: matching engine (18 cases)
│   ├── test_hawkes.cpp     # GoogleTest: Hawkes + adverse selection
│   ├── test_impact.cpp     # GoogleTest: Kyle + Almgren-Chriss
│   ├── test_fix.cpp        # GoogleTest: FIX gateway
│   └── test_python.py      # pytest: bindings
├── benchmarks/
│   └── bench_book.cpp      # Google Benchmark + TSC latency harness
├── examples/
│   └── demo.py             # Python usage demo
└── docs/
    └── theory.md           # queuing theory + market microstructure deep dive
```

---

## Further reading

For the deeper theory, see [`docs/theory.md`](docs/theory.md): the queuing-theory
view of a price level, the mathematics of Hawkes stationarity and Ogata thinning,
the derivation of Kyle's λ, and the Almgren-Chriss efficient frontier.

- Kyle, A. (1985). *Continuous Auctions and Insider Trading.* Econometrica.
- Almgren, R. & Chriss, N. (2000). *Optimal Execution of Portfolio Transactions.* J. Risk.
- Hawkes, A. (1971). *Spectra of Some Self-Exciting and Mutually Exciting Point Processes.* Biometrika.
- Ogata, Y. (1981). *On Lewis' Simulation Method for Point Processes.* IEEE Trans. Inf. Theory.
- Bacry, Mastromatteo, Muzy (2015). *Hawkes Processes in Finance.*
