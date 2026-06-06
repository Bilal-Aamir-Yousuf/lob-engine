"""End-to-end demo of the lob_engine Python API.

Run:  python examples/demo.py
"""

from lob_engine import (OrderBook, HawkesSimulator, MarketSimulator,
                        FixGateway, almgren_chriss)


def order_book_demo() -> None:
    print("=" * 64)
    print("Order book")
    print("=" * 64)

    book = OrderBook(tick_size=0.01)
    order_id = book.add_limit_order(side="bid", price=100.50, qty=100)
    book.add_limit_order(side="bid", price=100.49, qty=200)
    book.add_limit_order(side="ask", price=100.55, qty=150)
    book.add_limit_order(side="ask", price=100.56, qty=300)

    print(f"best_bid={book.best_bid()}  best_ask={book.best_ask()}  "
          f"mid={book.mid_price()}  spread={round(book.spread(), 4)}")
    print(f"bid depth (top 5): {book.depth(side='bid', levels=5)}")
    print(f"imbalance (3 lvls): {book.order_book_imbalance(levels=3):.4f}")

    book.cancel_order(order_id)
    print(f"after cancel, best_bid={book.best_bid()}")

    fills = book.add_market_order(side="ask", qty=50)  # sell into the bids
    print(f"market sell 50 -> fills: {fills}")
    print(f"best_bid now {book.best_bid()}, order_count={book.order_count()}")


def hawkes_demo() -> None:
    print("\n" + "=" * 64)
    print("Hawkes order-flow simulation")
    print("=" * 64)

    mu = [0.5, 0.5, 1.0, 1.0]          # market buy/sell, limit buy/sell
    alpha = [[0.1] * 4 for _ in range(4)]
    sim = HawkesSimulator(mu=mu, alpha=alpha, beta=1.0, seed=42)

    print(f"spectral_radius(alpha) = {sim.spectral_radius():.3f}  (< 1 => stationary)")
    print(f"closed-form stationary intensities = {sim.stationary_intensities()}")

    events = sim.simulate(T=2000.0)
    counts = {"market_buy": 0, "market_sell": 0, "limit_buy": 0, "limit_sell": 0}
    for _, etype in events:
        counts[etype] += 1
    print(f"{len(events)} events; empirical rates = "
          f"{ {k: round(v / 2000.0, 3) for k, v in counts.items()} }")


def adverse_selection_demo() -> None:
    print("\n" + "=" * 64)
    print("Coupled market simulation: adverse selection")
    print("=" * 64)

    mu = [0.5, 0.5, 1.0, 1.0]
    alpha = [[0.1] * 4 for _ in range(4)]
    sim = HawkesSimulator(mu=mu, alpha=alpha, beta=1.0, seed=2024)
    market = MarketSimulator(bucket_dt=1.0, seed=99)
    res = market.run(hawkes=sim, T=20000.0)

    print("OLS  mid_change ~ net_order_flow")
    print(f"  lambda_hat (price impact) = {res['lambda_hat']:.5f} ticks/share")
    print(f"  t-statistic               = {res['t_stat']:.2f}")
    print(f"  R^2                       = {res['r_squared']:.4f}")
    print(f"  n (buckets)               = {res['n_obs']}")


def almgren_chriss_demo() -> None:
    print("\n" + "=" * 64)
    print("Almgren-Chriss optimal liquidation")
    print("=" * 64)

    for label, risk in [("risk-neutral (TWAP)", 0.0), ("risk-averse", 1e-4)]:
        t = almgren_chriss(total_shares=1_000_000, horizon=1.0, n_intervals=10,
                           sigma=0.3, eta=2.5e-6, gamma=2.5e-7, risk_aversion=risk)
        trades = [round(n) for n in t["trades"]]
        print(f"\n{label}: kappa={t['kappa']:.4f}")
        print(f"  trade schedule (shares/slice): {trades}")
        print(f"  E[cost]={t['expected_cost']:.4e}  Var={t['variance']:.4e}")


def fix_demo() -> None:
    print("\n" + "=" * 64)
    print("FIX protocol gateway")
    print("=" * 64)

    book = OrderBook(tick_size=0.01)
    gw = FixGateway(book=book)

    def fix(*pairs):
        return "".join(f"{t}={v}|" for t, v in pairs)

    # NewOrderSingle: buy 100 @ 100.50 (limit)
    new = fix((35, "D"), (49, "CLIENT"), (11, "ORD1"), (54, 1), (38, 100),
              (44, "100.50"), (40, 2))
    print(f"--> {new}")
    for r in gw.process(message=new, delim="|"):
        print(f"<-- {r}")

    # Market sell 40 -> trades against the resting bid
    mkt = fix((35, "D"), (49, "CLIENT"), (11, "ORD2"), (54, 2), (38, 40), (40, 1))
    print(f"--> {mkt}")
    for r in gw.process(message=mkt, delim="|"):
        print(f"<-- {r}")

    print(f"book best_bid={book.best_bid()} depth={book.depth(side='bid', levels=1)}")


if __name__ == "__main__":
    order_book_demo()
    hawkes_demo()
    adverse_selection_demo()
    almgren_chriss_demo()
    fix_demo()
