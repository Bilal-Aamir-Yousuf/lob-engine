#pragma once

#include <cstddef>
#include <vector>

namespace lob {

// ---------------------------------------------------------------------------
// Kyle (1985) -- market depth / price impact coefficient lambda
// ---------------------------------------------------------------------------
// In Kyle's model the equilibrium price is linear in the (signed) order flow:
//   delta_p = lambda * order_flow + noise
// and the depth coefficient is
//   lambda = Cov(delta_p, order_flow) / Var(order_flow).
// lambda is the price move per unit of net order flow: HIGH lambda == a thin,
// illiquid market (small flow moves the price a lot); LOW lambda == deep/liquid.
//
// `price_changes[k]` and `order_flow[k]` must be aligned and the same length.
// Returns 0 if there are fewer than 2 points or the flow has zero variance.
double kyle_lambda(const std::vector<double>& price_changes,
                   const std::vector<double>& order_flow);

// ---------------------------------------------------------------------------
// Almgren-Chriss (2000) -- optimal liquidation trajectory
// ---------------------------------------------------------------------------
// Liquidating X shares over horizon T in N equal steps trades off two costs:
//   * temporary impact (eta): the price concession paid for trading fast,
//   * timing risk (sigma):    the variance of holding inventory longer.
// Permanent impact (gamma) shifts the price for everyone but, for a single
// schedule, contributes a trajectory-independent constant 0.5*gamma*X^2.
//
// The cost-plus-(risk_aversion)*variance optimum has a closed form built on
// hyperbolic functions:
//   eta_tilde = eta - 0.5 * gamma * tau          (tau = T/N)
//   kappa_hat^2 = lambda * sigma^2 / eta_tilde
//   kappa = acosh(0.5 * kappa_hat^2 * tau^2 + 1) / tau
//   x_j = X * sinh(kappa * (T - t_j)) / sinh(kappa * T)
// With risk_aversion -> 0, kappa -> 0 and the schedule degenerates to TWAP
// (a straight line); higher risk_aversion front-loads the trading.
struct ACParams {
    double total_shares  = 1.0e6;  // X, shares to liquidate
    double horizon       = 1.0;    // T, total time
    int    n_intervals   = 10;     // N, number of trading slices
    double sigma         = 0.3;    // per-unit-time volatility of the price
    double eta           = 2.5e-6; // temporary impact coefficient
    double gamma         = 2.5e-7; // permanent impact coefficient
    double risk_aversion = 1.0e-6; // lambda; 0 == risk-neutral (TWAP)
};

struct ACTrajectory {
    std::vector<double> times;     // t_j, j = 0..N
    std::vector<double> holdings;  // x_j shares remaining, j = 0..N (x_0=X, x_N=0)
    std::vector<double> trades;    // n_j = x_{j-1}-x_j shares sold in slice j, j=1..N
    double kappa         = 0.0;    // urgency parameter (1/kappa = trade timescale)
    double expected_cost = 0.0;    // E[implementation shortfall]
    double variance      = 0.0;    // Var[implementation shortfall]
};

ACTrajectory almgren_chriss(const ACParams& p);

}  // namespace lob
