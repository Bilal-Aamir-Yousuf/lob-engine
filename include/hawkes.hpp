#pragma once

#include <array>
#include <cstdint>
#include <random>
#include <vector>

#include "order_book.hpp"

namespace lob {

// The four event types of the order-flow process. The ordering is fixed and
// used to index the baseline vector mu and the excitation matrix alpha.
enum class EventType : uint8_t {
    MARKET_BUY  = 0,
    MARKET_SELL = 1,
    LIMIT_BUY   = 2,
    LIMIT_SELL  = 3,
};
inline constexpr int kNumEventTypes = 4;

struct Event {
    double    time = 0.0;
    EventType type = EventType::MARKET_BUY;
};

// Parameters of a 4-D mutually-exciting Hawkes process with an exponential
// kernel:
//
//   lambda_i(t) = mu_i + sum_j sum_{t_k < t, type=j} alpha_{ij} * beta * exp(-beta (t - t_k))
//
//   mu    : baseline (background) intensity per type        (events / unit time)
//   alpha : excitation matrix; alpha_{ij} is the expected number of type-i
//           offspring triggered by one type-j event (the kernel beta*exp(-beta t)
//           integrates to 1, so alpha_{ij} IS the branching ratio)
//   beta  : exponential decay rate of excitation (1/beta = memory timescale)
//
// Stationarity requires the spectral radius of alpha to be < 1.
struct HawkesParams {
    std::array<double, kNumEventTypes> mu{};
    std::array<std::array<double, kNumEventTypes>, kNumEventTypes> alpha{};
    double beta = 1.0;
};

// Exact simulation of the Hawkes process via Ogata's thinning algorithm,
// using the recursive exponential-kernel state so each step is O(N).
class HawkesSimulator {
public:
    // Throws std::invalid_argument if the process is non-stationary
    // (spectral_radius(alpha) >= 1) or beta <= 0.
    explicit HawkesSimulator(const HawkesParams& params, uint64_t seed = 42);

    // Simulate on [0, T]; returns events in time order.
    std::vector<Event> simulate(double T);

    // Spectral radius (dominant eigenvalue) of alpha, via power iteration.
    // alpha is entry-wise non-negative, so Perron-Frobenius guarantees a real,
    // non-negative dominant eigenvalue and convergence.
    double spectral_radius() const;

    // Closed-form stationary intensities: solves (I - alpha) * lambda_inf = mu.
    // This is the exact long-run mean rate per type. (The often-quoted scalar
    // form lambda_inf_i = mu_i / (1 - spectral_radius) is the special case where
    // alpha acts like a scalar on mu; we use the exact matrix inverse.)
    std::array<double, kNumEventTypes> stationary_intensities() const;

    const HawkesParams& params() const noexcept { return p_; }

private:
    HawkesParams p_;
    std::mt19937_64 rng_;
};

// ---------------------------------------------------------------------------
// MarketSimulator
// ---------------------------------------------------------------------------
// Drives an OrderBook with Hawkes-generated order flow and a latent fundamental
// price that moves with permanent market impact. Market buys push the
// fundamental up, market sells push it down; limit orders are quoted around the
// (drifting) fundamental, so stale quotes get run over -> adverse selection.
struct MarketParams {
    int64_t  init_price        = 10000;  // initial fundamental, in ticks
    double   impact_per_share  = 0.5;    // permanent impact eta (ticks / share)
    uint64_t market_order_size = 5;      // shares per market order
    uint64_t limit_order_size  = 10;     // shares per limit order
    int64_t  half_spread       = 5;      // base quote offset from fundamental (ticks)
    int64_t  spread_jitter     = 4;      // uniform extra offset [0, jitter]
    double   bucket_dt         = 1.0;    // time bucket for the flow/price series
    int      seed_levels       = 10;     // initial resting levels per side
};

// Per-bucket time series plus the adverse-selection regression result.
struct SimResult {
    std::vector<double> times;        // bucket end time
    std::vector<double> mid;          // mid price at bucket end (ticks)
    std::vector<double> fundamental;  // fundamental at bucket end (ticks)
    std::vector<double> net_flow;     // signed market volume in bucket (buy - sell)

    // OLS of future mid change (mid[k+1]-mid[k]) on net_flow[k]:
    double      lambda_hat = 0.0;  // slope == price impact per unit net flow
    double      intercept  = 0.0;
    double      t_stat     = 0.0;  // t-statistic of the slope
    double      r_squared  = 0.0;
    std::size_t n_obs      = 0;
};

class MarketSimulator {
public:
    explicit MarketSimulator(const MarketParams& mp = {}, uint64_t seed = 7)
        : mp_(mp), rng_(seed) {}

    // Run the coupled Hawkes + order-book simulation to horizon T.
    SimResult run(HawkesSimulator& hawkes, double T);

    OrderBook&       book() noexcept { return book_; }
    double           fundamental() const noexcept { return fundamental_; }

private:
    void seed_book();
    double current_mid() const;

    MarketParams    mp_;
    OrderBook       book_;
    double          fundamental_ = 0.0;
    std::mt19937_64 rng_;
};

}  // namespace lob
