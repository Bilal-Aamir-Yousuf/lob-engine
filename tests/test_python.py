"""Python-side tests for the pybind11 bindings (run with `pytest tests/`)."""

import math

import pytest

import lob_engine as lob


# ---------------------------------------------------------------------------
# OrderBook: the exact API from the project spec
# ---------------------------------------------------------------------------
def test_basic_api_matches_spec():
    book = lob.OrderBook(tick_size=0.01)
    oid = book.add_limit_order(side="bid", price=100.50, qty=100)
    assert isinstance(oid, int)
    assert book.best_bid() == pytest.approx(100.50)
    assert book.cancel_order(oid) is True
    assert book.best_bid() is None


def test_empty_book_returns_none():
    book = lob.OrderBook()
    assert book.best_bid() is None
    assert book.best_ask() is None
    assert book.mid_price() is None
    assert book.spread() is None
    assert book.depth(side="bid", levels=5) == []
    assert book.order_book_imbalance(5) == 0.0


def test_marketable_limit_crosses_spread():
    book = lob.OrderBook(tick_size=0.01)
    book.add_limit_order(side="ask", price=101.00, qty=5)
    # Buy limit at/above the ask lifts the offer.
    book.add_limit_order(side="bid", price=101.00, qty=5)
    assert book.best_ask() is None
    assert book.best_bid() is None
    assert book.order_count() == 0


def test_market_order_returns_fills_and_sweeps_levels():
    book = lob.OrderBook(tick_size=0.01)
    book.add_limit_order(side="ask", price=101.00, qty=2)
    book.add_limit_order(side="ask", price=101.01, qty=2)
    book.add_limit_order(side="ask", price=101.02, qty=2)

    fills = book.add_market_order(side="bid", qty=5)
    assert sum(f["quantity"] for f in fills) == 5
    assert fills[0]["price"] == pytest.approx(101.00)
    assert fills[1]["price"] == pytest.approx(101.01)
    assert book.best_ask() == pytest.approx(101.02)


def test_depth_and_imbalance():
    book = lob.OrderBook(tick_size=0.01)
    book.add_limit_order(side="bid", price=100.00, qty=6)
    book.add_limit_order(side="ask", price=100.05, qty=2)
    d = book.depth(side="bid", levels=5)
    assert d[0][0] == pytest.approx(100.00)
    assert d[0][1] == 6
    # (6 - 2)/(6 + 2) = 0.5
    assert book.order_book_imbalance(1) == pytest.approx(0.5)


def test_mid_and_spread():
    book = lob.OrderBook(tick_size=0.01)
    book.add_limit_order(side="bid", price=100.00, qty=1)
    book.add_limit_order(side="ask", price=100.10, qty=1)
    assert book.mid_price() == pytest.approx(100.05)
    assert book.spread() == pytest.approx(0.10)


def test_invalid_side_raises():
    book = lob.OrderBook()
    with pytest.raises(Exception):
        book.add_limit_order(side="upwards", price=1.0, qty=1)


# ---------------------------------------------------------------------------
# Hawkes process
# ---------------------------------------------------------------------------
def _stationary_params():
    mu = [0.5, 0.5, 1.0, 1.0]
    alpha = [[0.1] * 4 for _ in range(4)]
    return mu, alpha, 1.0


def test_hawkes_rejects_nonstationary():
    mu, _, beta = _stationary_params()
    alpha = [[0.3] * 4 for _ in range(4)]  # spectral radius 1.2
    with pytest.raises(Exception):
        lob.HawkesSimulator(mu=mu, alpha=alpha, beta=beta)


def test_hawkes_spectral_radius_and_stationary():
    mu, alpha, beta = _stationary_params()
    sim = lob.HawkesSimulator(mu=mu, alpha=alpha, beta=beta, seed=1)
    assert sim.spectral_radius() == pytest.approx(0.4, abs=1e-9)
    s = sim.stationary_intensities()
    assert s == pytest.approx([1.0, 1.0, 1.5, 1.5], abs=1e-9)


def test_hawkes_empirical_rates_match():
    mu, alpha, beta = _stationary_params()
    sim = lob.HawkesSimulator(mu=mu, alpha=alpha, beta=beta, seed=7)
    T = 20000.0
    events = sim.simulate(T)
    assert len(events) > 1000
    counts = {"market_buy": 0, "market_sell": 0, "limit_buy": 0, "limit_sell": 0}
    for _, etype in events:
        counts[etype] += 1
    assert counts["market_buy"] / T == pytest.approx(1.0, rel=0.06)
    assert counts["limit_buy"] / T == pytest.approx(1.5, rel=0.06)


# ---------------------------------------------------------------------------
# Market simulation + adverse selection
# ---------------------------------------------------------------------------
def test_market_simulation_shows_adverse_selection():
    mu, alpha, beta = _stationary_params()
    sim = lob.HawkesSimulator(mu=mu, alpha=alpha, beta=beta, seed=2024)
    market = lob.MarketSimulator(bucket_dt=1.0, seed=99)
    res = market.run(hawkes=sim, T=20000.0)
    assert res["lambda_hat"] > 0.0
    assert res["t_stat"] > 5.0
    assert res["n_obs"] > 100


# ---------------------------------------------------------------------------
# Market impact models
# ---------------------------------------------------------------------------
def test_kyle_lambda_recovers_slope():
    flow = [1.0, -2.0, 3.0, -1.0, 2.0, -3.0]
    dp = [0.5 * q for q in flow]
    assert lob.kyle_lambda(price_changes=dp, order_flow=flow) == pytest.approx(0.5)


def test_almgren_chriss_risk_neutral_is_twap():
    t = lob.almgren_chriss(total_shares=1_000_000, horizon=1.0,
                           n_intervals=10, risk_aversion=0.0)
    expected = 1_000_000 / 10
    assert all(abs(n - expected) < 1.0 for n in t["trades"])
    assert t["holdings"][-1] == pytest.approx(0.0, abs=1e-6)
    assert t["kappa"] == pytest.approx(0.0, abs=1e-12)


def test_almgren_chriss_risk_averse_is_front_loaded():
    t = lob.almgren_chriss(total_shares=1_000_000, horizon=1.0,
                           n_intervals=10, risk_aversion=1e-4)
    assert t["kappa"] > 0.0
    trades = t["trades"]
    assert all(trades[i] < trades[i - 1] for i in range(1, len(trades)))
    assert math.isclose(sum(trades), 1_000_000, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# FIX gateway
# ---------------------------------------------------------------------------
def _fix(*pairs):
    """Build a raw FIX message (| delimited) from (tag, value) pairs."""
    return "".join(f"{t}={v}|" for t, v in pairs)


def test_fix_limit_order_rests_and_acks():
    book = lob.OrderBook(tick_size=0.01)
    gw = lob.FixGateway(book=book)
    msg = _fix((35, "D"), (11, "ORD1"), (54, 1), (38, 100), (44, "100.50"), (40, 2))
    reports = gw.process(message=msg, delim="|")
    assert len(reports) == 1
    assert "35=8|" in reports[0]      # ExecutionReport
    assert "150=0|" in reports[0]     # New
    assert book.best_bid() == pytest.approx(100.50)


def test_fix_market_order_fills_and_cancel_roundtrip():
    book = lob.OrderBook(tick_size=0.01)
    gw = lob.FixGateway(book=book)
    gw.process(message=_fix((35, "D"), (11, "M1"), (54, 1), (38, 100),
                            (44, "100.50"), (40, 2)), delim="|")

    fills = gw.process(message=_fix((35, "D"), (11, "T1"), (54, 2), (38, 40),
                                    (40, 1)), delim="|")
    assert len(fills) == 1
    assert "32=40|" in fills[0]       # LastQty 40
    assert book.depth(side="bid", levels=1)[0][1] == 60

    # Cancel the remaining 60 of the resting bid M1.
    rej = gw.process(message=_fix((35, "F"), (41, "M1"), (11, "C1")), delim="|")
    assert "150=4|" in rej[0]         # Canceled
    assert book.best_bid() is None


def test_fix_cancel_unknown_is_rejected():
    book = lob.OrderBook()
    gw = lob.FixGateway(book=book)
    rep = gw.process(message=_fix((35, "F"), (41, "NOPE"), (11, "C9")), delim="|")
    assert "35=9|" in rep[0]          # OrderCancelReject
