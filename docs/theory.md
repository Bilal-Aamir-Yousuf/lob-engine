# Theory: Queuing and Market Microstructure

This document covers the theory behind the design choices in the engine: why the
data structures are what they are, and what the simulation and impact models
actually mean. The [README](../README.md) has the practical overview; this is the
longer version.

---

## 1. A price level is a queue

Structurally, a limit order book is a set of FIFO queues, one per price level,
with a matching rule that decides which queue gets served.

Under price-time priority:

- Across price levels, service goes by price: the best price is served first
  (highest bid for buyers, lowest ask for sellers).
- Within a price level, service goes by time: first in, first out.

So each price level is a textbook queue. Resting a limit order is an enqueue at
the tail, matching against that level is a dequeue from the head, and a
cancellation is a removal from the middle. The engineering problem is making all
three cheap.

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

`P` is small and bounded in practice (a liquid name has tens to low hundreds of
active levels), so the `O(log P)` to find a level is effectively constant, and it
is the only non-constant factor on the add path. Everything that scales with the
number of orders, namely cancel, match-per-fill, and top-of-book, is `O(1)`.

### Why `std::map` and not a heap

The book needs two things a binary heap cannot give you at the same time:

1. Ordered iteration for `depth()` and imbalance (walking levels from best to
   worst).
2. `O(log P)` find and erase of an arbitrary key, because a level empties and has
   to be removed, and a cancellation can empty a level in the middle.

A `std::map` (a balanced BST) gives ordered keys with `O(log P)` insert, find, and
erase, plus `O(1)` access to both ends through `begin()` and `rbegin()`. A heap
gives `O(1)` access to one end but `O(P)` to find an arbitrary level. The bid side
uses the map's maximum (`rbegin`) as the best, and the ask side uses the minimum
(`begin`).

> A production matching engine for a single liquid instrument often swaps the map
> for a flat array indexed by tick (prices sit in a known band), which makes level
> lookup genuinely `O(1)`. The map version here is more general (it handles
> arbitrary price ranges) and already runs well past the throughput target. The
> array is the obvious next optimisation.

---

## 2. The intrusive list and free-list pool

The cancellation path is what separates a toy book from a real one.
Cancellations vastly outnumber trades in live markets (well over 90% of orders
are cancelled), so cancel has to be both `O(1)` and allocation-free.

Intrusive means the linked-list pointers (`prev`, `next`) are members of the
`Order` struct rather than a separate wrapper node. That has two consequences:

- Resting an order needs no extra allocation, because the order is the node.
- The `unordered_map<id, Order*>` stores the node address directly, so a cancel
  is a single hash lookup, four pointer writes to splice, and a return of the node
  to the pool. No traversal, no `std::list::iterator` bookkeeping.

The `OrderPool` is a free-list allocator. It grabs memory in large blocks, hands
out `Order` slots, and recycles freed slots through an intrusive free list (a
freed order's `next` pointer links the free list). In steady state, a book that
adds and cancels at similar rates never touches the system allocator, which is
what keeps the p99 latency bounded and avoids heap fragmentation.

Per-level volume (`total_volume`) is kept up to date on every append, partial
fill, and cancel, so `depth()` and `order_book_imbalance()` never have to sum a
list.

---

## 3. Hawkes processes: self-exciting order flow

### The model

A point process is self-exciting if an event raises the probability of more
events. The conditional intensity of a multivariate Hawkes process with an
exponential kernel is

```
λ_i(t) = μ_i + Σ_j  α_ij · Σ_{t_k < t, type=j}  β · exp(−β (t − t_k))
```

- `μ_i ≥ 0` is the exogenous baseline intensity of type *i* (news, scheduled
  flow).
- `α_ij ≥ 0` is how strongly a type-*j* event excites type *i*. Because the kernel
  `g(t) = β·exp(−βt)` integrates to 1 over `[0,∞)`, the total expected excitation
  from one type-*j* event onto type *i* is exactly `α_ij`. That makes `α` the
  branching matrix of an equivalent Galton-Watson cluster process.
- `β > 0` is the decay rate. `1/β` is the memory: how long an event keeps exciting
  the system.

### Stationarity

Treating `α` as a branching matrix, the process is sub-critical (it does not
explode) if and only if the spectral radius `ρ(α) < 1`. Each immigrant event from
the baseline spawns, in expectation, a finite cluster of `(I − α)⁻¹` descendants.
The engine computes `ρ(α)` by power iteration (valid since `α ≥ 0`, so
Perron-Frobenius guarantees a real, non-negative dominant eigenvalue), and the
constructor throws if `ρ(α) ≥ 1`.

The long-run mean intensities solve the fixed point `λ_∞ = μ + α λ_∞`, i.e.

```
λ_∞ = (I − α)⁻¹ μ
```

solved here by Gaussian elimination. (The scalar approximation
`λ_∞,i ≈ μ_i / (1 − ρ(α))` is exact only when `α` acts like a scalar on `μ`, so
the engine uses the full matrix inverse instead.) The unit test confirms the
empirical rates match `λ_∞` to within about 1% over a long run.

### Exact simulation: Ogata's thinning

Thinning generates an exact sample path with no time discretisation:

1. At the current time `t`, the total intensity `Λ(t) = Σ_i λ_i(t)` is an upper
   bound on the intensity over the next interval, because with no new events the
   intensity only decays.
2. Propose the next event at `t + w`, where `w ~ Exponential(Λ(t))`.
3. Compute the true total intensity `Λ(t+w)` and accept the proposal with
   probability `Λ(t+w)/Λ(t)`. If accepted, assign it type *i* with probability
   `λ_i(t+w)/Λ(t+w)`; otherwise reject (no event) and continue from `t+w`.

The exponential kernel makes this fast. The intensity state
`S_i(t) = Σ_{past} α_ij β e^{−β(t−t_k)}` follows a simple recursion: it decays
multiplicatively by `e^{−β·Δt}` between events and jumps by `α_ij β` when a
type-*j* event fires. So each step, accepted or rejected, is `O(4)` arithmetic
with no sum over history.

### Why this matters economically

Self-excitation is the math behind a real phenomenon: news and liquidity shocks
cause clustered trading. A burst of selling raises the short-term probability of
more selling (momentum, stop cascades, herding), and mutual excitation across the
four types captures effects like aggressive buying pulling in liquidity providers
on the ask. A Poisson model, with independent constant-rate arrivals, cannot
produce the volatility clustering or the fat-tailed inter-trade times that real
tape shows. Hawkes can, with very few parameters, which is why it is the standard
for realistic order-flow simulation.

---

## 4. The latent fundamental and adverse selection

On top of the flow sits a latent fundamental price `F(t)` that moves with
permanent market impact: each market buy of `q` shares shifts `F` up by `η·q`, and
each market sell shifts it down by `η·q`. Liquidity providers quote limit orders
around `F`, so when `F` drifts, stale quotes on the wrong side get executed. That
is adverse selection, the structural reason market making is risky.

The simulator's realism is checked by regressing the mid-price change in a time
bucket on the contemporaneous signed market order flow:

```
Δmid_k = α + λ · netflow_k + ε_k
```

A positive, significant `λ` means order flow moves the price against resting
liquidity. The validation run gives `λ ≈ 0.033 ticks/share` with `t ≈ 9.4`, which
is decisive. The slope is the same as Kyle's `λ` below, because the OLS slope is
`Cov/Var` by construction.

---

## 5. Kyle (1985): the price of information

Kyle's model has an informed trader, noise traders, and a competitive market
maker who only sees aggregate order flow. Unable to tell informed flow from noise,
the market maker sets a price that is linear in net order flow:

```
p = p_0 + λ · (total order flow)
```

Equilibrium gives `λ = Cov(Δp, q) / Var(q)`. The reading is market depth: `1/λ`
is how much volume it takes to move the price by one unit. A high `λ` is a thin,
illiquid, or informationally-toxic market (small flow moves price a lot); a low
`λ` is deep and liquid. Estimating `λ` from a price and flow series is a one-line
covariance ratio, exposed as `kyle_lambda()` and cross-checked against the
simulator's regression slope.

---

## 6. Almgren-Chriss (2000): optimal execution

Say you have to liquidate `X` shares within a horizon `T`. Two forces pull against
each other:

- Temporary impact `η`: trading fast pays a concession that scales with the trade
  rate. The cost of slice `n_j` over `τ = T/N` is about `η · n_j² / τ`. Trading
  slower is cheaper.
- Timing and volatility risk `σ`: holding inventory longer exposes it to price
  variance, about `σ² · Σ x_j² · τ`. Trading faster is safer.

Minimising expected cost plus `λ · variance` (where `λ` is the trader's risk
aversion) is a quadratic program whose solution, in the continuous limit, is

```
x(t) = X · sinh(κ (T − t)) / sinh(κ T),     κ = sqrt(λ σ² / η̃)
```

with `η̃ = η − ½ γ τ` the risk-adjusted temporary impact and `γ` the permanent
impact. The discrete version uses `κ = acosh(½ κ̂²τ² + 1)/τ`.

- `λ → 0` (risk-neutral): `κ → 0`, and `sinh` linearises to a straight line, so
  the schedule is TWAP, equal slices. There is no reason to rush.
- `λ` large (risk-averse): `κ` is large and the schedule front-loads steeply, so
  you liquidate early to cut variance while eating more impact cost.

Sweeping `λ` traces the efficient frontier of execution: every point is a schedule
that is variance-minimal for its cost (or cost-minimal for its variance). The unit
tests verify both endpoints and the monotone trade-off, where more urgency means
higher κ, lower variance, and higher expected cost.

---

## References

- Kyle, A. S. (1985). *Continuous Auctions and Insider Trading.* Econometrica 53(6).
- Almgren, R. & Chriss, N. (2000). *Optimal Execution of Portfolio Transactions.* Journal of Risk 3(2).
- Hawkes, A. G. (1971). *Spectra of Some Self-Exciting and Mutually Exciting Point Processes.* Biometrika 58(1).
- Ogata, Y. (1981). *On Lewis' Simulation Method for Point Processes.* IEEE Transactions on Information Theory 27(1).
- Bacry, E., Mastromatteo, I., Muzy, J.-F. (2015). *Hawkes Processes in Finance.* Market Microstructure and Liquidity 1(1).
- Bouchaud, Bonart, Donier, Gould (2018). *Trades, Quotes and Prices.* Cambridge University Press.
