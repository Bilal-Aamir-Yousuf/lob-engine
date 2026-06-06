#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>

#include "hawkes.hpp"

using lob::EventType;
using lob::HawkesParams;
using lob::HawkesSimulator;
using lob::MarketParams;
using lob::MarketSimulator;
using lob::kNumEventTypes;

namespace {

// A stationary 4-D process. alpha is 0.1 in every entry, so its spectral radius
// is 4 * 0.1 = 0.4 (rank-1 all-ones structure). With these mu the closed-form
// stationary intensities are {1.0, 1.0, 1.5, 1.5} (verified analytically via
// the Sherman-Morrison inverse of (I - 0.1*J)).
HawkesParams make_params() {
    HawkesParams p;
    p.mu = {0.5, 0.5, 1.0, 1.0};
    for (auto& row : p.alpha) row.fill(0.1);
    p.beta = 1.0;
    return p;
}

}  // namespace

// ---------------------------------------------------------------------------
// Stationarity enforcement
// ---------------------------------------------------------------------------
TEST(Hawkes, RejectsNonStationaryProcess) {
    HawkesParams p = make_params();
    for (auto& row : p.alpha) row.fill(0.3);  // spectral radius 1.2 -> explosive
    EXPECT_THROW({ HawkesSimulator s(p); }, std::invalid_argument);
}

TEST(Hawkes, RejectsNonPositiveBeta) {
    HawkesParams p = make_params();
    p.beta = 0.0;
    EXPECT_THROW({ HawkesSimulator s(p); }, std::invalid_argument);
}

TEST(Hawkes, SpectralRadiusMatchesAnalytic) {
    HawkesSimulator sim(make_params());
    EXPECT_NEAR(sim.spectral_radius(), 0.4, 1e-9);
}

TEST(Hawkes, StationaryIntensitiesMatchAnalytic) {
    HawkesSimulator sim(make_params());
    auto s = sim.stationary_intensities();
    EXPECT_NEAR(s[0], 1.0, 1e-9);
    EXPECT_NEAR(s[1], 1.0, 1e-9);
    EXPECT_NEAR(s[2], 1.5, 1e-9);
    EXPECT_NEAR(s[3], 1.5, 1e-9);
}

// ---------------------------------------------------------------------------
// Empirical rates match the closed-form stationary intensities
// ---------------------------------------------------------------------------
TEST(Hawkes, EmpiricalRatesMatchStationary) {
    HawkesSimulator sim(make_params(), /*seed=*/12345);
    const double T = 50000.0;
    auto events = sim.simulate(T);

    std::array<long, kNumEventTypes> counts{};
    for (const auto& e : events) ++counts[static_cast<int>(e.type)];

    auto stationary = sim.stationary_intensities();
    std::cout << "\n[Hawkes validation] empirical vs closed-form rates (T=" << T
              << ", " << events.size() << " events):\n";
    const char* names[] = {"market_buy ", "market_sell", "limit_buy  ", "limit_sell "};
    for (int i = 0; i < kNumEventTypes; ++i) {
        const double emp = static_cast<double>(counts[i]) / T;
        std::cout << "  " << names[i] << "  empirical=" << std::fixed
                  << std::setprecision(4) << emp
                  << "  closed-form=" << stationary[i]
                  << "  rel.err=" << std::setprecision(2)
                  << 100.0 * std::abs(emp - stationary[i]) / stationary[i] << "%\n";
        EXPECT_NEAR(emp, stationary[i], 0.04 * stationary[i]);  // within 4%
    }
}

// ---------------------------------------------------------------------------
// Adverse selection: net order flow moves the mid against resting liquidity
// ---------------------------------------------------------------------------
TEST(Hawkes, AdverseSelectionRegressionIsPositiveAndSignificant) {
    HawkesSimulator sim(make_params(), /*seed=*/2024);
    MarketParams mp;
    mp.bucket_dt = 1.0;
    MarketSimulator market(mp, /*seed=*/99);

    auto res = market.run(sim, /*T=*/50000.0);

    std::cout << "\n[Adverse selection] OLS  mid_change ~ net_order_flow\n"
              << "  lambda_hat (price impact) = " << std::fixed
              << std::setprecision(5) << res.lambda_hat << " ticks/share\n"
              << "  intercept                 = " << res.intercept << "\n"
              << "  t-statistic               = " << std::setprecision(2)
              << res.t_stat << "\n"
              << "  R^2                       = " << std::setprecision(4)
              << res.r_squared << "\n"
              << "  n (buckets)               = " << res.n_obs << "\n";

    EXPECT_GT(res.n_obs, 100u);
    EXPECT_GT(res.lambda_hat, 0.0);  // positive price impact
    EXPECT_GT(res.t_stat, 5.0);      // overwhelmingly significant
}
