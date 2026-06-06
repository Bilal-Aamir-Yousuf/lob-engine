#include <gtest/gtest.h>

#include <iomanip>
#include <iostream>
#include <vector>

#include "hawkes.hpp"
#include "impact.hpp"

using lob::ACParams;
using lob::almgren_chriss;
using lob::HawkesParams;
using lob::HawkesSimulator;
using lob::kyle_lambda;
using lob::MarketParams;
using lob::MarketSimulator;

// ---------------------------------------------------------------------------
// Kyle (1985) lambda
// ---------------------------------------------------------------------------
TEST(Kyle, RecoversKnownSlope) {
    std::vector<double> flow = {1, -2, 3, -1, 2, -3, 4, 0};
    std::vector<double> dp;
    for (double q : flow) dp.push_back(0.5 * q);  // exact dp = 0.5 * flow
    EXPECT_NEAR(kyle_lambda(dp, flow), 0.5, 1e-12);
}

TEST(Kyle, ZeroVarianceFlowReturnsZero) {
    std::vector<double> flow = {2, 2, 2, 2};
    std::vector<double> dp   = {1, -1, 3, 0};
    EXPECT_DOUBLE_EQ(kyle_lambda(dp, flow), 0.0);
}

TEST(Kyle, HigherLambdaMeansThinnerMarket) {
    // Same flow, twice the price response -> twice the depth coefficient.
    std::vector<double> flow = {1, 2, 3, 4, 5};
    std::vector<double> dp_liquid, dp_thin;
    for (double q : flow) { dp_liquid.push_back(0.1 * q); dp_thin.push_back(0.4 * q); }
    EXPECT_LT(kyle_lambda(dp_liquid, flow), kyle_lambda(dp_thin, flow));
}

// Cross-check: Kyle's lambda computed on the simulated series must equal the
// OLS slope produced by the Phase-2 adverse-selection regression (Cov/Var IS
// the regression slope), and it must be positive.
TEST(Kyle, MatchesSimulationRegressionSlope) {
    HawkesParams hp;
    hp.mu = {0.5, 0.5, 1.0, 1.0};
    for (auto& row : hp.alpha) row.fill(0.1);
    hp.beta = 1.0;
    HawkesSimulator sim(hp, /*seed=*/2024);

    MarketParams mp;
    mp.bucket_dt = 1.0;
    MarketSimulator market(mp, /*seed=*/99);
    auto res = market.run(sim, /*T=*/50000.0);

    std::vector<double> flow, dp;
    for (std::size_t k = 1; k < res.net_flow.size(); ++k) {
        flow.push_back(res.net_flow[k]);
        dp.push_back(res.mid[k] - res.mid[k - 1]);
    }
    const double lam = kyle_lambda(dp, flow);
    std::cout << "\n[Kyle] lambda from simulation = " << std::fixed
              << std::setprecision(5) << lam << " ticks/share\n";

    EXPECT_GT(lam, 0.0);
    EXPECT_NEAR(lam, res.lambda_hat, 1e-9);  // identical to the OLS slope
}

// ---------------------------------------------------------------------------
// Almgren-Chriss (2000)
// ---------------------------------------------------------------------------
TEST(AlmgrenChriss, RiskNeutralIsTWAP) {
    ACParams p;
    p.total_shares  = 1.0e6;
    p.horizon       = 1.0;
    p.n_intervals   = 10;
    p.risk_aversion = 0.0;  // risk-neutral -> straight-line liquidation

    auto traj = almgren_chriss(p);
    ASSERT_EQ(traj.trades.size(), 10u);
    const double expected_slice = p.total_shares / p.n_intervals;
    for (double n : traj.trades) {
        EXPECT_NEAR(n, expected_slice, 1e-3);  // all slices equal
    }
    EXPECT_NEAR(traj.holdings.front(), p.total_shares, 1e-6);
    EXPECT_NEAR(traj.holdings.back(), 0.0, 1e-6);
    EXPECT_NEAR(traj.kappa, 0.0, 1e-12);
}

TEST(AlmgrenChriss, RiskAverseIsFrontLoaded) {
    ACParams p;
    p.total_shares  = 1.0e6;
    p.horizon       = 1.0;
    p.n_intervals   = 10;
    p.risk_aversion = 1.0e-4;  // strongly risk-averse

    auto traj = almgren_chriss(p);
    EXPECT_GT(traj.kappa, 0.0);

    // Trades strictly decrease (sell aggressively early to cut timing risk).
    for (std::size_t j = 1; j < traj.trades.size(); ++j) {
        EXPECT_LT(traj.trades[j], traj.trades[j - 1]);
    }
    // Holdings are convex and fully liquidate.
    double total = 0.0;
    for (double n : traj.trades) total += n;
    EXPECT_NEAR(total, p.total_shares, 1e-3);
    EXPECT_NEAR(traj.holdings.back(), 0.0, 1e-6);
}

TEST(AlmgrenChriss, RiskCostTradeoffIsMonotone) {
    ACParams base;
    base.total_shares = 1.0e6;
    base.horizon      = 1.0;
    base.n_intervals  = 20;

    ACParams patient = base;  patient.risk_aversion = 1.0e-7;
    ACParams urgent  = base;  urgent.risk_aversion  = 1.0e-3;

    auto tp = almgren_chriss(patient);
    auto tu = almgren_chriss(urgent);

    std::cout << "\n[Almgren-Chriss] patient (lambda=1e-7): kappa=" << std::fixed
              << std::setprecision(4) << tp.kappa
              << "  E[cost]=" << std::scientific << std::setprecision(3) << tp.expected_cost
              << "  Var=" << tp.variance << "\n"
              << "                 urgent  (lambda=1e-3): kappa=" << std::fixed
              << std::setprecision(4) << tu.kappa
              << "  E[cost]=" << std::scientific << std::setprecision(3) << tu.expected_cost
              << "  Var=" << tu.variance << "\n";

    // More risk aversion -> trade faster -> less timing risk but higher cost.
    EXPECT_GT(tu.kappa, tp.kappa);
    EXPECT_LT(tu.variance, tp.variance);
    EXPECT_GT(tu.expected_cost, tp.expected_cost);
}
