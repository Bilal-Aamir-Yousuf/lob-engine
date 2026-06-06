#include "hawkes.hpp"

#include <cmath>
#include <stdexcept>

namespace lob {
namespace {

constexpr int N = kNumEventTypes;

// Dominant eigenvalue magnitude of a non-negative NxN matrix via power
// iteration (Perron-Frobenius makes this well-behaved).
double power_iteration_spectral_radius(
    const std::array<std::array<double, N>, N>& A) {
    std::array<double, N> v;
    v.fill(1.0 / std::sqrt(static_cast<double>(N)));
    double lambda = 0.0;
    for (int iter = 0; iter < 1000; ++iter) {
        std::array<double, N> w{};
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j) w[i] += A[i][j] * v[j];
        double norm = 0.0;
        for (int i = 0; i < N; ++i) norm += w[i] * w[i];
        norm = std::sqrt(norm);
        if (norm < 1e-300) return 0.0;  // zero matrix
        double new_lambda = norm;
        for (int i = 0; i < N; ++i) v[i] = w[i] / norm;
        if (std::abs(new_lambda - lambda) < 1e-12) return new_lambda;
        lambda = new_lambda;
    }
    return lambda;
}

// Solve (I - A) x = b for x via Gaussian elimination with partial pivoting.
std::array<double, N> solve_I_minus_A(
    const std::array<std::array<double, N>, N>& A,
    const std::array<double, N>& b) {
    std::array<std::array<double, N>, N> M{};
    std::array<double, N> rhs = b;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) M[i][j] = (i == j ? 1.0 : 0.0) - A[i][j];

    for (int col = 0; col < N; ++col) {
        // Partial pivot.
        int pivot = col;
        for (int r = col + 1; r < N; ++r)
            if (std::abs(M[r][col]) > std::abs(M[pivot][col])) pivot = r;
        std::swap(M[col], M[pivot]);
        std::swap(rhs[col], rhs[pivot]);

        const double diag = M[col][col];
        for (int r = col + 1; r < N; ++r) {
            const double f = M[r][col] / diag;
            for (int c = col; c < N; ++c) M[r][c] -= f * M[col][c];
            rhs[r] -= f * rhs[col];
        }
    }
    std::array<double, N> x{};
    for (int row = N - 1; row >= 0; --row) {
        double s = rhs[row];
        for (int c = row + 1; c < N; ++c) s -= M[row][c] * x[c];
        x[row] = s / M[row][row];
    }
    return x;
}

}  // namespace

// ---------------------------------------------------------------------------
// HawkesSimulator
// ---------------------------------------------------------------------------
HawkesSimulator::HawkesSimulator(const HawkesParams& params, uint64_t seed)
    : p_(params), rng_(seed) {
    if (p_.beta <= 0.0) {
        throw std::invalid_argument("HawkesSimulator: beta must be positive");
    }
    if (spectral_radius() >= 1.0) {
        throw std::invalid_argument(
            "HawkesSimulator: non-stationary process (spectral_radius(alpha) >= 1)");
    }
}

double HawkesSimulator::spectral_radius() const {
    return power_iteration_spectral_radius(p_.alpha);
}

std::array<double, kNumEventTypes> HawkesSimulator::stationary_intensities() const {
    return solve_I_minus_A(p_.alpha, p_.mu);
}

std::vector<Event> HawkesSimulator::simulate(double T) {
    std::vector<Event> events;
    std::uniform_real_distribution<double> unif(0.0, 1.0);

    // Recursive exponential-kernel state: S[i] = sum over past events j of
    // alpha[i][type_j] * beta * exp(-beta (t - t_j)). Then lambda_i = mu_i + S_i.
    std::array<double, N> S{};
    double t = 0.0;

    while (t < T) {
        // Upper bound M = total intensity at the current time. Between events
        // the intensity only decays, so M bounds it over the next interval.
        double M = 0.0;
        for (int i = 0; i < N; ++i) M += p_.mu[i] + S[i];
        if (M <= 0.0) break;

        // Draw candidate inter-arrival ~ Exponential(rate = M).
        const double w = -std::log(unif(rng_)) / M;
        t += w;
        if (t > T) break;

        // Decay the kernel state to the candidate time.
        const double decay = std::exp(-p_.beta * w);
        for (int i = 0; i < N; ++i) S[i] *= decay;

        // Actual total intensity at candidate time.
        std::array<double, N> lambda{};
        double lambda_total = 0.0;
        for (int i = 0; i < N; ++i) {
            lambda[i] = p_.mu[i] + S[i];
            lambda_total += lambda[i];
        }

        // Accept with probability lambda_total / M (thinning).
        if (unif(rng_) * M <= lambda_total) {
            // Select the event type proportionally to its intensity.
            double u = unif(rng_) * lambda_total;
            int k = 0;
            for (; k < N - 1; ++k) {
                if (u < lambda[k]) break;
                u -= lambda[k];
            }
            // The accepted event excites all types: jump the kernel state.
            for (int i = 0; i < N; ++i) S[i] += p_.alpha[i][k] * p_.beta;
            events.push_back({t, static_cast<EventType>(k)});
        }
        // On rejection, t and the decayed state simply carry forward.
    }
    return events;
}

// ---------------------------------------------------------------------------
// MarketSimulator
// ---------------------------------------------------------------------------
void MarketSimulator::seed_book() {
    // Lay down initial resting liquidity symmetrically around the fundamental.
    for (int i = 1; i <= mp_.seed_levels; ++i) {
        book_.add_limit_order(Side::BID, mp_.init_price - mp_.half_spread - i,
                              mp_.limit_order_size);
        book_.add_limit_order(Side::ASK, mp_.init_price + mp_.half_spread + i,
                              mp_.limit_order_size);
    }
}

double MarketSimulator::current_mid() const {
    auto m = book_.mid_price();
    return m ? *m : fundamental_;  // fall back to fundamental if a side is empty
}

SimResult MarketSimulator::run(HawkesSimulator& hawkes, double T) {
    fundamental_ = static_cast<double>(mp_.init_price);
    seed_book();

    const std::vector<Event> events = hawkes.simulate(T);
    std::uniform_int_distribution<int64_t> jitter(0, mp_.spread_jitter);

    SimResult res;
    std::size_t cur_bucket = 0;
    double flow_accum = 0.0;  // signed market volume in the current bucket

    auto close_bucket = [&](std::size_t b) {
        res.times.push_back(static_cast<double>(b + 1) * mp_.bucket_dt);
        res.net_flow.push_back(flow_accum);
        res.mid.push_back(current_mid());
        res.fundamental.push_back(fundamental_);
        flow_accum = 0.0;
    };

    for (const Event& e : events) {
        const std::size_t b = static_cast<std::size_t>(e.time / mp_.bucket_dt);
        while (cur_bucket < b) {
            close_bucket(cur_bucket);
            ++cur_bucket;
        }

        switch (e.type) {
            case EventType::MARKET_BUY: {
                fundamental_ += mp_.impact_per_share *
                                static_cast<double>(mp_.market_order_size);
                book_.add_market_order(Side::BID, mp_.market_order_size);
                flow_accum += static_cast<double>(mp_.market_order_size);
                break;
            }
            case EventType::MARKET_SELL: {
                fundamental_ -= mp_.impact_per_share *
                                static_cast<double>(mp_.market_order_size);
                book_.add_market_order(Side::ASK, mp_.market_order_size);
                flow_accum -= static_cast<double>(mp_.market_order_size);
                break;
            }
            case EventType::LIMIT_BUY: {
                const int64_t px = static_cast<int64_t>(std::llround(fundamental_)) -
                                   mp_.half_spread - jitter(rng_);
                book_.add_limit_order(Side::BID, px, mp_.limit_order_size);
                break;
            }
            case EventType::LIMIT_SELL: {
                const int64_t px = static_cast<int64_t>(std::llround(fundamental_)) +
                                   mp_.half_spread + jitter(rng_);
                book_.add_limit_order(Side::ASK, px, mp_.limit_order_size);
                break;
            }
        }
    }

    // ---- Adverse-selection regression (Kyle's lambda) ----------------------
    // Regress the mid-price change realised over a bucket on the contemporaneous
    // signed market order flow in that bucket:
    //   y_k = mid[k] - mid[k-1]   (price move across bucket k)
    //   x_k = net_flow[k]         (signed market volume during bucket k)
    // A positive, significant slope is exactly the adverse-selection signature:
    // net buying pressure moves the price up against the resting liquidity that
    // got run over. The slope is Kyle's price-impact coefficient lambda.
    const std::size_t buckets = res.net_flow.size();
    const std::size_t n = buckets >= 1 ? buckets - 1 : 0;  // pairs k=1..buckets-1
    if (n >= 2) {
        double sx = 0, sy = 0, sxx = 0, sxy = 0, syy = 0;
        for (std::size_t k = 1; k < buckets; ++k) {
            const double x = res.net_flow[k];
            const double y = res.mid[k] - res.mid[k - 1];
            sx += x; sy += y; sxx += x * x; sxy += x * y; syy += y * y;
        }
        const double dn = static_cast<double>(n);
        const double sxx_c = sxx - sx * sx / dn;   // centered Sxx
        const double sxy_c = sxy - sx * sy / dn;   // centered Sxy
        const double syy_c = syy - sy * sy / dn;   // centered Syy
        if (sxx_c > 0.0) {
            const double beta = sxy_c / sxx_c;
            const double alpha = (sy - beta * sx) / dn;
            const double sse = syy_c - beta * sxy_c;
            const double s2 = sse / (dn - 2.0);          // residual variance
            const double se = std::sqrt(s2 / sxx_c);     // std error of slope
            res.lambda_hat = beta;
            res.intercept  = alpha;
            res.t_stat     = se > 0.0 ? beta / se : 0.0;
            res.r_squared  = syy_c > 0.0 ? 1.0 - sse / syy_c : 0.0;
            res.n_obs      = n;
        }
    }
    return res;
}

}  // namespace lob
