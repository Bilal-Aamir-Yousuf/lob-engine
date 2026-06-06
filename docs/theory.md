# Theory: Queuing & Market Microstructure

This document explains the theory behind each design decision in the engine —
why the data structures are what they are, and what the simulation and impact
models actually mean. The [README](../README.md) has the practical overview;
this is the deeper dive.

---

## 1. A price level is a queue

A limit order book is, structurally, a collection of **FIFO queues** — one per
price level — with a matching rule that decides which queue gets served.

Under **price-time priority**:

- **Across** price levels, service is by price: the best price is served first
  (highest bid for buyers, lowest ask for sellers).
- **Within** a price level, service is by time: first-in-first-out.

So each price level is a textbook queue. Resting a limit order is an *enqueue*
at the tail; matching against that level is a *dequeue* from the head; a
cancellation is a *removal from the middle*. The engineering problem is to make
all three cheap.

### Complexity of the operations

Let `P` be the number of distinct, currently-occupied price levels and `n` the
number of resting orders.

| Operation                  | Structure used                        | Cost          |
|----------------------------|---------------------------------------|---------------|
| locate a price level       | `std::map<int64_t, PriceLevel>`       | `O(log P)`    |
| enqueue at a level (rest)  | intrusive list `append` at the tail   | `O(1)`        |
| dequeue at a level (match) | intrusive list pop from the head      | `O(1)` / fill |
| cancel by id               | `unordered_map` lookup + list splice  | `O(1)`        |
| best bid / best ask        | `map::rbegin` / `map::begin`          | `O(1)`        |
| depth / imbalance (top N)  | iterate N levels, read cached volume  | `O(N)`        |

`P` is small and bounded in practice (a liquid name has tens to low-hundreds of
active levels), so the `O(log P)` to find a level is effectively constant and is
the *only* non-constant factor on the add path. Everything that scales with the
number of orders — cancel, match-per-fill, top-of-book — is `O(1)`.

### Why `std::map` and not a heap

The book needs two things a binary heap can't give you simultaneously:

1. **Ordered iteration** for `depth()` and imbalance (walk levels best→worst).
2. **`O(log P)` find/erase of an arbitrary key**, because a level empties and
   must be removed, and a cancellation may empty a level in the middle.

A `std::map` (a balanced BST) gives ordered keys with `O(log P)` insert / find /
erase and `O(1)` access to both extremes via `begin()`/`rbegin()`. A heap gives
`O(1)` access to *one* extreme but `O(P)` to find an arbitrary level. The bid
side uses the map's maximum (`rbegin`) as the best; the ask side uses the
minimum (`begin`).

> A production matching engine for a single liquid instrument often replaces the
> map with a flat **array indexed by tick** (prices live in a known band), making
> level lookup truly `O(1)`. The map version here is general (handles arbitrary
> price ranges) and already far past the throughput target; the array is the
> natural next optimisation.

---

## 2. The intrusive list + free-list pool

The cancellation path is the one that separates a toy book from a real one.
Cancellations vastly outnumber trades in real markets (well over 90% of orders
are cancelled), so cancel must be `O(1)` *and* allocation-free.

**Intrusive** means the linked-list pointers (`prev`, `next`) are members of the
`Order` struct, not of a separate wrapper node. Consequences:

- Resting an order is **zero extra allocations** — the order *is* the node.
- The `unordered_map<id, Order*>` stores the node address directly, so a cancel
  is: one hash lookup → four pointer writes to splice → return the node to the
  pool. No traversal, no `std::list::iterator` bookkeeping.

The **`OrderPool`** is a free-list allocator: it grabs memory in large blocks and
hands out `Order` slots, recycling freed slots through an intrusive free list
(the freed order's `next` pointer links the free list). In steady state — a
book that adds and cancels at similar rates — the pool never touches the system
allocator, which is what keeps the p99 latency bounded and avoids heap
fragmentation.

Volume accounting (`total_volume` per level) is maintained incrementally on
every append, partial fill, and cancel, so `depth()` and
`order_book_imbalance()` never have to sum a list.

---

## 3. Hawkes processes: self-exciting order flow

### The model

A point process is **self-exciting** if the occurrence of an event raises the
probability of further events. The conditional intensity of a multivariate
Hawkes process with an exponential kernel is

```
λ_i(t) = μ_i + Σ_j  α_ij · Σ_{t_k < t, type=j}  β · exp(−β (t − t_k))
```

- `μ_i ≥ 0` — exogenous baseline intensity of type *i* (news, scheduled flow).
- `α_ij ≥ 0` — how strongly a type-*j* event excites type-*i*. Because the
  kernel `g(t) = β·exp(−βt)` integrates to 1 over `[0,∞)`, the *total* expected
  excitation from one type-*j* event onto type *i* is exactly `α_ij`. This makes
  `α` the **branching matrix** of an equivalent Galton-Watson cluster process.
- `β > 0` — decay rate. `1/β` is the memory: how long an event keeps exciting
  the system.

### Stationarity

Interpreting `α` as a branching matrix, the process is sub-critical (does not
explode) **iff the spectral radius `ρ(α) < 1`**. Each "immigrant" event from the
baseline spawns, in expectation, a finite cluster of `(I − α)⁻¹` descendants.
The engine computes `ρ(α)` by power iteration (valid since `α ≥ 0`, so
Perron-Frobenius guarantees a real non-negative dominant eigenvalue) and the
constructor **throws** if `ρ(α) ≥ 1`.

The closed-form long-run mean intensities solve the fixed point
`λ_∞ = μ + α λ_∞`, i.e.

```
λ_∞ = (I − α)⁻¹ μ
```

solved here by Gaussian elimination. (The scalar approximation
`λ_∞,i ≈ μ_i / (1 − ρ(α))` is exact only when `α` acts like a scalar on `μ`;
the engine uses the full matrix inverse.) The unit test confirms the empirical
rates match `λ_∞` to within ~1% over a long run.

### Exact simulation: Ogata's thinning

Thinning generates an exact sample path without time discretisation:

1. At the current time `t`, the total intensity `Λ(t) = Σ_i λ_i(t)` is an upper
   bound for the intensity over the next interval, because with no new events
   the intensity only **decays**.
2. Propose the next event at `t + w`, where `w ~ Exponential(Λ(t))`.
3. Compute the true total intensity `Λ(t+w)` and **accept** the proposal with
   probability `Λ(t+w)/Λ(t)`. If accepted, assign it type *i* with probability
   `λ_i(t+w)/Λ(t+w)`; otherwise reject (no event) and continue from `t+w`.

The exponential kernel makes this fast: the intensity state
`S_i(t) = Σ_{past} α_ij β e^{−β(t−t_k)}` satisfies a simple recursion — it
**decays multiplicatively** by `e^{−β·Δt}` between events and **jumps** by
`α_ij β` when a type-*j* event fires. So each accepted/rejected step is `O(4)`
arithmetic with no sum over history.

### Why this matters economically

Self-excitation is the mathematical encoding of a real phenomenon: **news and
liquidity shocks cause clustered trading**. A burst of selling raises the
short-term probability of more selling (momentum, stop cascades, herding);
mutual excitation across the four types captures effects like "aggressive buying
pulls in liquidity providers on the ask." A Poisson model — independent,
constant-rate arrivals — cannot produce the volatility clustering and the
fat-tailed inter-trade times that real tape shows. Hawkes can, with very few
parameters, which is why it is the standard for realistic order-flow simulation.

---

## 4. The latent fundamental and adverse selection

Layered on the flow is a **latent fundamental price** `F(t)` that moves with
**permanent market impact**: each market buy of `q` shares shifts `F` up by
`η·q`, each market sell down by `η·q`. Liquidity providers quote limit orders
around `F`, so when `F` drifts, stale quotes on the wrong side get executed —
**adverse selection**, the structural reason market making is risky.

We *prove* the simulator reproduces adverse selection by regressing the
mid-price change in a time bucket on the contemporaneous signed market order
flow:

```
Δmid_k = α + λ · netflow_k + ε_k
```

A positive, significant `λ` means order flow moves the price against resting
liquidity. The validation run gives `λ ≈ 0.033 ticks/share` with `t ≈ 9.4` —
decisive. The slope is identical to **Kyle's λ** below, because OLS slope is
`Cov/Var` by construction.

---

## 5. Kyle (1985): the price of information

Kyle's model has an informed trader, noise traders, and a competitive
market maker who can only see *aggregate* order flow. The market maker, unable
to tell informed from noise flow, sets a price that is **linear in net order
flow**:

```
p = p_0 + λ · (total order flow)
```

Equilibrium gives `λ = Cov(Δp, q) / Var(q)`. The interpretation is **market
depth**: `1/λ` is how much volume it takes to move the price one unit. A **high
λ is a thin, illiquid, or informationally-toxic market** (small flow moves price
a lot); a **low λ is deep and liquid**. Estimating λ from a price/flow series is
a one-line covariance ratio — exposed as `kyle_lambda()` and cross-checked
against the simulator's regression slope.

---

## 6. Almgren-Chriss (2000): optimal execution

You must liquidate `X` shares within horizon `T`. Two forces oppose each other:

- **Temporary impact** `η`: trading fast pays a price concession that scales with
  the trade *rate*. Cost of slice `n_j` over `τ = T/N` is `~ η · n_j² / τ`.
  Trading slower is cheaper.
- **Timing/volatility risk** `σ`: holding inventory longer exposes it to price
  variance, `~ σ² · Σ x_j² · τ`. Trading faster is safer.

Minimising **expected cost + λ · variance** (where `λ` is the trader's risk
aversion) is a quadratic program whose solution, in the continuous limit, is

```
x(t) = X · sinh(κ (T − t)) / sinh(κ T),     κ = sqrt(λ σ² / η̃)
```

with `η̃ = η − ½ γ τ` the risk-adjusted temporary impact and `γ` the permanent
impact. The discrete version uses `κ = acosh(½ κ̂²τ² + 1)/τ`.

- `λ → 0` (risk-neutral): `κ → 0`, and `sinh` linearises to a straight line —
  the schedule is **TWAP**, equal slices. There's no reason to rush.
- `λ` large (risk-averse): `κ` large, the schedule is steeply **front-loaded** —
  liquidate early to cut variance, eating more impact cost.

Sweeping `λ` traces the **efficient frontier of execution**: every point is a
schedule that is variance-minimal for its cost (or cost-minimal for its
variance). The unit tests verify both endpoints and the monotone trade-off:
more urgency ⇒ higher κ ⇒ lower variance, higher expected cost.

---

## References

- Kyle, A. S. (1985). *Continuous Auctions and Insider Trading.* Econometrica 53(6).
- Almgren, R. & Chriss, N. (2000). *Optimal Execution of Portfolio Transactions.* Journal of Risk 3(2).
- Hawkes, A. G. (1971). *Spectra of Some Self-Exciting and Mutually Exciting Point Processes.* Biometrika 58(1).
- Ogata, Y. (1981). *On Lewis' Simulation Method for Point Processes.* IEEE Transactions on Information Theory 27(1).
- Bacry, E., Mastromatteo, I., Muzy, J.-F. (2015). *Hawkes Processes in Finance.* Market Microstructure and Liquidity 1(1).
- Bouchaud, Bonart, Donier, Gould (2018). *Trades, Quotes and Prices.* Cambridge University Press.
