// pybind11 module exposing the C++ limit-order-book engine to Python.
//
// The engine works internally in integer price ticks; the Python OrderBook
// wrapper accepts human-friendly float prices and a tick_size, converting at
// the boundary. Sides are given as strings ('bid'/'ask') for an idiomatic API.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "fix.hpp"
#include "hawkes.hpp"
#include "impact.hpp"
#include "order_book.hpp"

namespace py = pybind11;
using namespace lob;

namespace {

Side parse_side(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    if (s == "bid" || s == "buy" || s == "b") return Side::BID;
    if (s == "ask" || s == "sell" || s == "s" || s == "offer" || s == "a")
        return Side::ASK;
    throw std::invalid_argument("side must be 'bid'/'buy' or 'ask'/'sell', got: " + s);
}

const char* event_type_name(EventType t) {
    switch (t) {
        case EventType::MARKET_BUY:  return "market_buy";
        case EventType::MARKET_SELL: return "market_sell";
        case EventType::LIMIT_BUY:   return "limit_buy";
        case EventType::LIMIT_SELL:  return "limit_sell";
    }
    return "unknown";
}

// ---------------------------------------------------------------------------
// Python-facing OrderBook: float prices <-> integer ticks.
// ---------------------------------------------------------------------------
class PyOrderBook {
public:
    explicit PyOrderBook(double tick_size = 0.01) : tick_size_(tick_size) {
        if (tick_size_ <= 0.0)
            throw std::invalid_argument("tick_size must be positive");
    }

    uint64_t add_limit_order(const std::string& side, double price, uint64_t qty) {
        return book_.add_limit_order(parse_side(side), to_ticks(price), qty);
    }

    bool cancel_order(uint64_t order_id) { return book_.cancel_order(order_id); }

    py::list add_market_order(const std::string& side, uint64_t qty) {
        auto fills = book_.add_market_order(parse_side(side), qty);
        return fills_to_list(fills);
    }

    py::object best_bid() const { return opt_price(book_.best_bid()); }
    py::object best_ask() const { return opt_price(book_.best_ask()); }

    py::object mid_price() const {
        auto m = book_.mid_price();
        if (!m) return py::none();
        return py::float_(*m * tick_size_);
    }

    py::object spread() const {
        auto s = book_.spread();
        if (!s) return py::none();
        return py::float_(static_cast<double>(*s) * tick_size_);
    }

    py::list depth(const std::string& side, std::size_t levels) const {
        py::list out;
        for (auto& [px, vol] : book_.depth(parse_side(side), levels)) {
            out.append(py::make_tuple(static_cast<double>(px) * tick_size_, vol));
        }
        return out;
    }

    double order_book_imbalance(std::size_t levels) const {
        return book_.order_book_imbalance(levels);
    }

    std::size_t order_count() const { return book_.order_count(); }
    double tick_size() const { return tick_size_; }

    OrderBook& raw_book() { return book_; }  // for the FIX gateway

private:
    int64_t to_ticks(double price) const {
        return static_cast<int64_t>(std::llround(price / tick_size_));
    }
    py::object opt_price(std::optional<int64_t> ticks) const {
        if (!ticks) return py::none();
        return py::float_(static_cast<double>(*ticks) * tick_size_);
    }
    py::list fills_to_list(const std::vector<Fill>& fills) const {
        py::list out;
        for (const auto& f : fills) {
            py::dict d;
            d["maker_order_id"] = f.maker_order_id;
            d["price"]          = static_cast<double>(f.price) * tick_size_;
            d["quantity"]       = f.quantity;
            out.append(d);
        }
        return out;
    }

    double tick_size_;
    OrderBook book_;
};

// ---------------------------------------------------------------------------
// Python-facing Hawkes simulator built from plain Python lists.
// ---------------------------------------------------------------------------
// FIX gateway bound to a Python OrderBook (shares its book and tick size).
class PyFixGateway {
public:
    PyFixGateway(PyOrderBook& book, const std::string& comp_id)
        : gw_(book.raw_book(), book.tick_size(), comp_id) {}

    py::list process(const std::string& raw, const std::string& delim) {
        const char d = delim.empty() ? FIX_SOH : delim[0];
        py::list out;
        for (const auto& r : gw_.process_raw(raw, d)) out.append(r);
        return out;
    }

private:
    FixGateway gw_;
};

HawkesParams make_hawkes_params(const std::vector<double>& mu,
                                const std::vector<std::vector<double>>& alpha,
                                double beta) {
    if (mu.size() != static_cast<std::size_t>(kNumEventTypes))
        throw std::invalid_argument("mu must have length 4");
    if (alpha.size() != static_cast<std::size_t>(kNumEventTypes))
        throw std::invalid_argument("alpha must be 4x4");
    HawkesParams p;
    for (int i = 0; i < kNumEventTypes; ++i) {
        p.mu[i] = mu[i];
        if (alpha[i].size() != static_cast<std::size_t>(kNumEventTypes))
            throw std::invalid_argument("alpha must be 4x4");
        for (int j = 0; j < kNumEventTypes; ++j) p.alpha[i][j] = alpha[i][j];
    }
    p.beta = beta;
    return p;
}

}  // namespace

PYBIND11_MODULE(lob_engine, m) {
    m.doc() =
        "High-frequency limit order book engine: price-time priority matching, "
        "Hawkes order-flow simulation, and market-impact models (C++ core).";

    // ---- OrderBook --------------------------------------------------------
    py::class_<PyOrderBook>(m, "OrderBook")
        .def(py::init<double>(), py::arg("tick_size") = 0.01,
             "Create an order book. Prices are floats; tick_size sets the grid.")
        .def("add_limit_order", &PyOrderBook::add_limit_order, py::arg("side"),
             py::arg("price"), py::arg("qty"),
             "Add a limit order; crosses the spread if marketable. Returns order_id.")
        .def("cancel_order", &PyOrderBook::cancel_order, py::arg("order_id"),
             "Cancel a resting order. Returns False if the id is unknown.")
        .def("add_market_order", &PyOrderBook::add_market_order, py::arg("side"),
             py::arg("qty"), "Execute a market order; returns a list of fills.")
        .def("best_bid", &PyOrderBook::best_bid)
        .def("best_ask", &PyOrderBook::best_ask)
        .def("mid_price", &PyOrderBook::mid_price)
        .def("spread", &PyOrderBook::spread)
        .def("depth", &PyOrderBook::depth, py::arg("side"), py::arg("levels"),
             "List of (price, volume) for the top N levels on a side.")
        .def("order_book_imbalance", &PyOrderBook::order_book_imbalance,
             py::arg("levels"),
             "(bid_vol - ask_vol)/(bid_vol + ask_vol) over the top N levels.")
        .def("order_count", &PyOrderBook::order_count)
        .def_property_readonly("tick_size", &PyOrderBook::tick_size);

    // ---- FIX gateway ------------------------------------------------------
    py::class_<PyFixGateway>(m, "FixGateway")
        .def(py::init<PyOrderBook&, const std::string&>(), py::arg("book"),
             py::arg("comp_id") = "LOB-ENGINE", py::keep_alive<1, 2>(),
             "FIX 4.2-like gateway routing order messages to an OrderBook.")
        .def("process", &PyFixGateway::process, py::arg("message"),
             py::arg("delim") = "|",
             "Process one raw FIX message; returns a list of response messages "
             "(ExecutionReports / cancel rejects).");

    // ---- HawkesSimulator --------------------------------------------------
    py::class_<HawkesSimulator>(m, "HawkesSimulator")
        .def(py::init([](const std::vector<double>& mu,
                         const std::vector<std::vector<double>>& alpha, double beta,
                         uint64_t seed) {
                 return new HawkesSimulator(make_hawkes_params(mu, alpha, beta), seed);
             }),
             py::arg("mu"), py::arg("alpha"), py::arg("beta"), py::arg("seed") = 42,
             "4-D mutually-exciting Hawkes process. Raises ValueError if "
             "non-stationary (spectral_radius(alpha) >= 1).")
        .def(
            "simulate",
            [](HawkesSimulator& self, double T) {
                auto events = self.simulate(T);
                py::list out;
                for (const auto& e : events)
                    out.append(py::make_tuple(e.time, event_type_name(e.type)));
                return out;
            },
            py::arg("T"), "Simulate on [0, T]; returns a list of (time, type).")
        .def("spectral_radius", &HawkesSimulator::spectral_radius)
        .def("stationary_intensities", [](const HawkesSimulator& self) {
            auto s = self.stationary_intensities();
            return std::vector<double>(s.begin(), s.end());
        });

    // ---- MarketSimulator (Hawkes flow + book + latent fundamental) --------
    py::class_<MarketSimulator>(m, "MarketSimulator")
        .def(py::init([](int64_t init_price, double impact_per_share,
                         uint64_t market_order_size, uint64_t limit_order_size,
                         int64_t half_spread, int64_t spread_jitter, double bucket_dt,
                         int seed_levels, uint64_t seed) {
                 MarketParams mp;
                 mp.init_price        = init_price;
                 mp.impact_per_share  = impact_per_share;
                 mp.market_order_size = market_order_size;
                 mp.limit_order_size  = limit_order_size;
                 mp.half_spread       = half_spread;
                 mp.spread_jitter     = spread_jitter;
                 mp.bucket_dt         = bucket_dt;
                 mp.seed_levels       = seed_levels;
                 return new MarketSimulator(mp, seed);
             }),
             py::arg("init_price") = 10000, py::arg("impact_per_share") = 0.5,
             py::arg("market_order_size") = 5, py::arg("limit_order_size") = 10,
             py::arg("half_spread") = 5, py::arg("spread_jitter") = 4,
             py::arg("bucket_dt") = 1.0, py::arg("seed_levels") = 10,
             py::arg("seed") = 7)
        .def(
            "run",
            [](MarketSimulator& self, HawkesSimulator& hawkes, double T) {
                SimResult r = self.run(hawkes, T);
                py::dict d;
                d["times"]       = r.times;
                d["mid"]         = r.mid;
                d["fundamental"] = r.fundamental;
                d["net_flow"]    = r.net_flow;
                d["lambda_hat"]  = r.lambda_hat;
                d["intercept"]   = r.intercept;
                d["t_stat"]      = r.t_stat;
                d["r_squared"]   = r.r_squared;
                d["n_obs"]       = r.n_obs;
                return d;
            },
            py::arg("hawkes"), py::arg("T"),
            "Run the coupled Hawkes + order-book simulation; returns a dict with "
            "the per-bucket series and the adverse-selection regression result.");

    // ---- Market impact models --------------------------------------------
    m.def("kyle_lambda", &kyle_lambda, py::arg("price_changes"),
          py::arg("order_flow"),
          "Kyle (1985) depth coefficient: Cov(dp, flow) / Var(flow).");

    m.def(
        "almgren_chriss",
        [](double total_shares, double horizon, int n_intervals, double sigma,
           double eta, double gamma, double risk_aversion) {
            ACParams p{total_shares, horizon, n_intervals, sigma, eta, gamma,
                       risk_aversion};
            ACTrajectory t = almgren_chriss(p);
            py::dict d;
            d["times"]         = t.times;
            d["holdings"]      = t.holdings;
            d["trades"]        = t.trades;
            d["kappa"]         = t.kappa;
            d["expected_cost"] = t.expected_cost;
            d["variance"]      = t.variance;
            return d;
        },
        py::arg("total_shares"), py::arg("horizon"), py::arg("n_intervals") = 10,
        py::arg("sigma") = 0.3, py::arg("eta") = 2.5e-6, py::arg("gamma") = 2.5e-7,
        py::arg("risk_aversion") = 1.0e-6,
        "Almgren-Chriss (2000) optimal liquidation schedule (closed form).");
}
