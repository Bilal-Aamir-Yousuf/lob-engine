# lob-engine — High-Frequency Limit Order Book Engine

A production-grade **C++17** limit order book with **O(1) order insertion and
cancellation**, realistic order-flow simulation via **Hawkes processes**,
classic **market-impact models** (Kyle 1985, Almgren-Chriss 2000), a **FIX 4.2
protocol gateway** for external connectivity, **Python bindings** through
pybind11, and **benchmark-verified throughput of 8M+ orders per second**.

This is the kind of system that sits at the heart of every exchange and quant
trading firm — built from scratch, fast, and correct.

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
- [Why price-time priority + an intrusive linked list](#why-price-time-priority--an-intrusive-linked-list)
- [Hawkes process order flow (in plain English)](#hawkes-process-order-flow-in-plain-english)
- [Adverse selection — proof from the simulator](#adverse-selection--proof-from-the-simulator)
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

## Why price-time priority + an intrusive linked list

**Price-time priority** is the matching rule used by virtually every lit
exchange: orders are filled best-price-first, and within a price, oldest-first
(FIFO). It is *fair* (no jumping the queue without improving the price) and it
gives traders a well-defined incentive — to get filled sooner at a given price,
arrive earlier. Reproducing it correctly is the whole game; everything else is
performance.

The performance question is: **what data structure holds the resting orders so
that the three hot operations are each O(1)?**

| Operation        | What it needs                                  | Our cost |
|------------------|------------------------------------------------|----------|
| add (resting)    | append to the back of a price level (FIFO)     | O(1)\*   |
| cancel           | find the order, splice it out of its level     | O(1)     |
| match (best px)  | read the best level, pop from the front        | O(1)     |

\* O(log P) to *locate* the price level in the `std::map` where P = number of
distinct price levels (tiny and bounded in practice), then O(1) to append.

The trick that makes **cancellation O(1)** is the pairing of two structures:

1. **`std::unordered_map<order_id, Order*>`** — a global lookup that maps an
   order id straight to the node's address in constant time.
2. **An intrusive doubly-linked list** at each price level — because the node
   *is* the `Order` (the `prev`/`next` pointers live inside the struct), once we
   have the `Order*` we splice it out in O(1) with no search.

### What "intrusive" buys you over the naive version

A naive book stores `std::list<Order>` per level. That means:

- **Every order is a separate heap allocation** (the list node wraps the order).
  At millions of orders/second, `malloc`/`free` becomes the bottleneck and
  fragments memory.
- **Cancellation needs an iterator**, so you store `list::iterator` in the
  lookup map — workable, but you still pay the per-node allocation.
- **Cache misses** everywhere: list nodes are scattered across the heap.

The intrusive design fixes all three:

- The `next`/`prev` pointers are **fields of `Order`**, so there is no wrapper
  node and no extra allocation per order.
- Orders are handed out from a **free-list `OrderPool`** that allocates in big
  blocks and recycles freed slots (reusing the `next` pointer as the free-list
  link). In steady state there are **zero calls to the system allocator** on the
  hot path.
- Splicing is four pointer writes. Finding the node is one hash lookup.

The result is the latency profile in the [benchmarks](#benchmarks): ~90 ns
median to add a limit order, ~180 ns to cancel, and >8M orders/sec throughput on
a laptop-class CPU.

> **Why integer ticks?** Prices are stored as `int64_t` ticks (e.g. dollars ×
> 100), never floats. Floating-point prices break the most fundamental
> operation in a matching engine — *equality and ordering of price levels* —
> because `1.10` is not exactly representable. Integers make comparison exact
> and make `std::map` keys behave.

---

## Hawkes process order flow (in plain English)

Real order flow is **not** a steady drizzle of independent events. It arrives in
**bursts**: a large sell prints, which spooks other participants into selling,
which moves the price, which triggers stops and more selling. Trading begets
trading. Statisticians call a process with this "an event now makes more events
soon" property **self-exciting**, and the canonical model is the **Hawkes
process**.

We implement a **4-dimensional mutually-exciting** Hawkes process — the four
dimensions being `market_buy`, `market_sell`, `limit_buy`, `limit_sell`. Each
type has a **baseline rate** `μ_i` (the background drizzle) and every past event
*excites* all four future rates through an **excitation matrix** `α`:

```
λ_i(t) = μ_i + Σ_j Σ_{t_k < t, type=j}  α_ij · β · exp(−β · (t − t_k))
```

In words: the current intensity of type *i* is its baseline plus a sum of
exponentially-decaying "bumps", one for every past event, where a type-*j* event
bumps type-*i*'s rate by `α_ij`. The bump fades with timescale `1/β`.

- `α_ij` is the **branching ratio**: the expected number of type-*i* offspring
  triggered by one type-*j* event (the kernel `β·exp(−βt)` integrates to 1).
- **Stationarity** (the process doesn't explode) requires the **spectral radius
  of `α` to be < 1** — we *enforce* this in the constructor and throw otherwise.
- The long-run mean rates have a closed form: `λ_∞ = (I − α)⁻¹ μ`.

Events are generated by **Ogata's thinning algorithm**, which produces an
*exact* (not discretised) sample path:

1. Bound the total intensity by its current value `Λ = Σ_i λ_i(t)` (valid
   because between events the intensity only decays).
2. Draw a candidate time `t + Exponential(1/Λ)`.
3. Accept the candidate with probability `Λ(t_next)/Λ`; if accepted, pick the
   event type in proportion to its intensity. Otherwise reject and repeat.

Using the exponential kernel's recursive structure, each step is O(4), so we
simulate hundreds of thousands of events in milliseconds.

**Validation.** Run the simulator for a long horizon and the empirical event
rates converge to the closed-form stationary intensities:

```
[Hawkes validation] empirical vs closed-form rates (T=50000, 248873 events):
  market_buy    empirical=0.9868  closed-form=1.0000  rel.err=1.32%
  market_sell   empirical=0.9990  closed-form=1.0000  rel.err=0.10%
  limit_buy     empirical=1.4945  closed-form=1.5000  rel.err=0.37%
  limit_sell    empirical=1.4973  closed-form=1.5000  rel.err=0.18%
```

---

## Adverse selection — proof from the simulator

A **latent fundamental price** drifts with **permanent market impact**: each
market buy nudges it up, each market sell nudges it down. Liquidity providers
quote limit orders around the (moving) fundamental, so when the fundamental runs
away from a stale quote, that quote gets **picked off** — this is **adverse
selection**, the core risk of market making.

If the simulator is realistic, signed order flow should *predict* price moves in
the same direction. We test this directly by regressing the per-bucket mid-price
change on the contemporaneous **net market order flow** (this slope is exactly
**Kyle's λ**):

```
[Adverse selection] OLS  mid_change ~ net_order_flow
  lambda_hat (price impact) = 0.03253 ticks/share
  intercept                 = -0.00036
  t-statistic               = 9.43
  R^2                       = 0.0018
  n (buckets)               = 49998
```

The slope is **positive and overwhelmingly significant (t ≈ 9.4)** — net buying
pressure moves the price up against resting liquidity, exactly as theory
predicts. The small R² is itself realistic: at high frequency, microstructure
noise dominates the *variance* of returns while the systematic impact is small
but rock-solid in *expectation*.

---

## Market-impact models

### Kyle (1985) — market depth λ

In Kyle's model the price is linear in signed order flow, `Δp = λ · q + noise`,
and the depth coefficient is estimated as

```
λ = Cov(Δp, q) / Var(q)
```

λ is the price move per unit of net order flow: **high λ = thin/illiquid market**
(small flow moves the price a lot), **low λ = deep/liquid**. The estimator is
exposed as `kyle_lambda(price_changes, order_flow)` and, run on the simulated
series, returns the same number as the adverse-selection regression slope — as
it must, since `Cov/Var` *is* the OLS slope.

### Almgren-Chriss (2000) — optimal liquidation

Liquidating `X` shares over horizon `T` trades off two costs: **temporary
impact** (the concession paid for trading fast) versus **timing risk** (the
variance of holding inventory longer). Minimising `cost + λ·variance` yields a
closed-form schedule built on hyperbolic functions:

```
κ̂² = λ σ² / η̃ ,   η̃ = η − ½ γ τ ,   κ = acosh(½ κ̂² τ² + 1) / τ
x_j = X · sinh(κ (T − t_j)) / sinh(κ T)
```

- **Risk-neutral** (`λ → 0`): `κ → 0` and the schedule is a straight line — TWAP
  (equal slices).
- **Risk-averse** (`λ > 0`): the schedule is **front-loaded** — sell faster
  early to cut timing risk, accepting higher impact cost.

```
risk-neutral (TWAP): kappa=0.0000
  trade schedule (shares/slice): [100000 ×10]            E[cost]=2.61e6  Var=2.57e10
risk-averse:         kappa=1.8993
  trade schedule (shares/slice): [181738, 152133, 128032, 108563, 93022,
                                   80846, 71596, 64936, 60626, 58509]
                                                        E[cost]=3.01e6  Var=1.64e10
```

More urgency ⇒ higher κ ⇒ lower variance but higher expected cost — the
fundamental execution trade-off, recovered exactly.

---

## FIX protocol gateway

Real exchanges and brokers speak **FIX** (Financial Information eXchange) — a
flat `tag=value` wire format framed by a `BeginString` (8), `BodyLength` (9) and
a `CheckSum` (10). The engine ships a FIX 4.2-style `FixGateway` that parses
inbound order messages, routes them to the book, and emits the corresponding
**ExecutionReport** (35=8) / **OrderCancelReject** (35=9) responses — the same
shape an external trading system would exchange with a venue.

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

Every emitted message carries a correct `BodyLength` and `CheckSum`
(`FixMessage::checksum_valid()` verifies inbound frames), prices are converted
between decimal and integer ticks at the boundary, and the gateway maintains the
`ClOrdID ↔ engine order_id` mapping required to cancel a resting order.

---

## Benchmarks

Measured with Google Benchmark (throughput) and a TSC cycle-counting harness
(per-operation latency percentiles), Release build, MSVC 19.44, on a 24-core
~2.0 GHz machine. **Your numbers will vary with hardware** — regenerate with
`bench_book.exe`.

**Throughput (Google Benchmark)**

| Operation                       | Throughput        |
|---------------------------------|-------------------|
| `add_limit_order`               | **8.36 M ops/sec** |
| `cancel_order`                  | **12.0 M ops/sec** |
| `add_market_order` (sweep 100)  | 37.0 M orders matched/sec |
| `add_market_order` (sweep 1000) | 41.7 M orders matched/sec |

Target was 1 M+/sec; the engine clears it by ~8×.

**Per-operation latency (nanoseconds, TSC-measured)**

| Operation                    |   p50 |   p95 |   p99 |  p99.9 |   mean |
|------------------------------|------:|------:|------:|-------:|-------:|
| `add_limit_order`            |  90.2 | 310.6 | 450.9 | 6762.8 |  215.3 |
| `cancel_order`               | 180.3 | 430.8 | 571.1 |  971.8 |  213.8 |
| `add_market_order` depth=1   |  90.2 | 130.2 | 150.3 |  230.4 |   96.5 |
| `add_market_order` depth=10  | 250.5 | 310.6 | 400.8 |  521.0 |  263.9 |
| `add_market_order` depth=100 | 2655  | 3426  | 3717  | 11341  |  2738  |
| `add_market_order` depth=1000| 23535 | 35698 | 43402 | 75123  | 25933  |

Market-order latency scales linearly with the number of resting orders swept —
exactly the O(k) you'd expect for matching k orders, with everything else O(1).

---

## Getting started

**Prerequisites:** a C++17 compiler, CMake ≥ 3.16, and (for the Python module)
Python ≥ 3.8. All third-party dependencies (GoogleTest, Google Benchmark,
pybind11) are pulled automatically via CMake `FetchContent`.

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

> On Windows, build from a *Developer Command Prompt* (or any shell with the
> MSVC environment) so CMake can find the compiler, and bind against a **64-bit**
> Python to match a 64-bit build.

---

## Python API

```python
from lob_engine import OrderBook, HawkesSimulator, MarketSimulator, kyle_lambda, almgren_chriss

# --- Order book (float prices, integer-tick engine underneath) ---
book = OrderBook(tick_size=0.01)
oid  = book.add_limit_order(side='bid', price=100.50, qty=100)
book.add_limit_order(side='ask', price=100.55, qty=150)
book.cancel_order(oid)
fills = book.add_market_order(side='ask', qty=50)   # list of {maker_order_id, price, quantity}
book.best_bid(); book.best_ask(); book.mid_price(); book.spread()
book.depth(side='bid', levels=5)                    # [(price, volume), ...]
book.order_book_imbalance(levels=3)                 # (bid_vol-ask_vol)/(bid_vol+ask_vol)

# --- Hawkes order flow ---
mu    = [0.5, 0.5, 1.0, 1.0]            # market buy/sell, limit buy/sell
alpha = [[0.1]*4 for _ in range(4)]     # excitation matrix (spectral radius 0.4)
sim   = HawkesSimulator(mu=mu, alpha=alpha, beta=1.0, seed=42)
sim.spectral_radius(); sim.stationary_intensities()
events = sim.simulate(T=2000.0)         # [(time, 'market_buy'), ...]

# --- Coupled simulation + adverse-selection regression ---
res = MarketSimulator(bucket_dt=1.0).run(hawkes=sim, T=20000.0)
res['lambda_hat'], res['t_stat'], res['r_squared']

# --- Market impact ---
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

See [`docs/theory.md`](docs/theory.md) for the deeper theory: queuing-theory
view of a price level, the mathematics of Hawkes stationarity and Ogata
thinning, the derivation of Kyle's λ, and the Almgren-Chriss efficient frontier.

- Kyle, A. (1985). *Continuous Auctions and Insider Trading.* Econometrica.
- Almgren, R. & Chriss, N. (2000). *Optimal Execution of Portfolio Transactions.* J. Risk.
- Hawkes, A. (1971). *Spectra of Some Self-Exciting and Mutually Exciting Point Processes.* Biometrika.
- Ogata, Y. (1981). *On Lewis' Simulation Method for Point Processes.* IEEE Trans. Inf. Theory.
- Bacry, Mastromatteo, Muzy (2015). *Hawkes Processes in Finance.*
