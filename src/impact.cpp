#include "impact.hpp"

#include <cmath>

namespace lob {

double kyle_lambda(const std::vector<double>& price_changes,
                   const std::vector<double>& order_flow) {
    const std::size_t n = std::min(price_changes.size(), order_flow.size());
    if (n < 2) return 0.0;

    double mean_dp = 0.0, mean_q = 0.0;
    for (std::size_t k = 0; k < n; ++k) {
        mean_dp += price_changes[k];
        mean_q  += order_flow[k];
    }
    mean_dp /= static_cast<double>(n);
    mean_q  /= static_cast<double>(n);

    double cov = 0.0, var = 0.0;
    for (std::size_t k = 0; k < n; ++k) {
        const double dq = order_flow[k] - mean_q;
        cov += (price_changes[k] - mean_dp) * dq;
        var += dq * dq;
    }
    if (var <= 0.0) return 0.0;
    return cov / var;  // Cov(dp, q) / Var(q)
}

ACTrajectory almgren_chriss(const ACParams& p) {
    ACTrajectory traj;
    const int N = p.n_intervals;
    const double X = p.total_shares;
    const double T = p.horizon;
    const double tau = T / static_cast<double>(N);

    // Risk-adjusted temporary impact and the urgency parameter kappa.
    const double eta_tilde = p.eta - 0.5 * p.gamma * tau;
    double kappa = 0.0;
    if (p.risk_aversion > 0.0 && eta_tilde > 0.0) {
        const double kappa_hat_sq = p.risk_aversion * p.sigma * p.sigma / eta_tilde;
        kappa = std::acosh(0.5 * kappa_hat_sq * tau * tau + 1.0) / tau;
    }
    traj.kappa = kappa;

    traj.times.resize(N + 1);
    traj.holdings.resize(N + 1);
    traj.trades.resize(N);

    const bool linear = (kappa * T < 1e-8);  // risk-neutral -> TWAP straight line
    const double sinh_kT = std::sinh(kappa * T);
    for (int j = 0; j <= N; ++j) {
        const double t = static_cast<double>(j) * tau;
        traj.times[j] = t;
        if (linear) {
            traj.holdings[j] = X * (1.0 - t / T);
        } else {
            traj.holdings[j] = X * std::sinh(kappa * (T - t)) / sinh_kT;
        }
    }
    traj.holdings[N] = 0.0;  // pin the endpoint exactly

    for (int j = 1; j <= N; ++j) {
        traj.trades[j - 1] = traj.holdings[j - 1] - traj.holdings[j];
    }

    // Closed-form expected cost and variance of the implementation shortfall.
    //   permanent (trajectory-independent) : 0.5 * gamma * X^2
    //   temporary                          : (eta_tilde / tau) * sum_j n_j^2
    //   variance (timing risk)             : sigma^2 * tau * sum_j x_j^2
    double temp_cost = 0.0;
    for (int j = 0; j < N; ++j) {
        temp_cost += traj.trades[j] * traj.trades[j];
    }
    temp_cost *= (eta_tilde > 0.0 ? eta_tilde : p.eta) / tau;
    traj.expected_cost = 0.5 * p.gamma * X * X + temp_cost;

    double var = 0.0;
    for (int j = 1; j <= N; ++j) {  // risk borne on inventory held through each slice
        var += traj.holdings[j] * traj.holdings[j];
    }
    traj.variance = p.sigma * p.sigma * tau * var;

    return traj;
}

}  // namespace lob
