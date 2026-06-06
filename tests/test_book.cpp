#include <gtest/gtest.h>

#include "order_book.hpp"

using lob::OrderBook;
using lob::Side;
using lob::Fill;

// ---------------------------------------------------------------------------
// Empty-book state queries
// ---------------------------------------------------------------------------
TEST(EmptyBook, QueriesReturnNullopt) {
    OrderBook book;
    EXPECT_FALSE(book.best_bid().has_value());
    EXPECT_FALSE(book.best_ask().has_value());
    EXPECT_FALSE(book.mid_price().has_value());
    EXPECT_FALSE(book.spread().has_value());
    EXPECT_EQ(book.order_count(), 0u);
    EXPECT_TRUE(book.depth(Side::BID, 5).empty());
    EXPECT_TRUE(book.depth(Side::ASK, 5).empty());
    EXPECT_DOUBLE_EQ(book.order_book_imbalance(5), 0.0);
}

TEST(EmptyBook, MarketOrderOnEmptyBookFillsNothing) {
    OrderBook book;
    auto fills = book.add_market_order(Side::BID, 100);
    EXPECT_TRUE(fills.empty());
    EXPECT_EQ(book.order_count(), 0u);
}

// ---------------------------------------------------------------------------
// Basic add and cancel
// ---------------------------------------------------------------------------
TEST(AddCancel, RestingOrdersSetTopOfBook) {
    OrderBook book;
    book.add_limit_order(Side::BID, 10000, 5);
    book.add_limit_order(Side::ASK, 10100, 7);

    ASSERT_TRUE(book.best_bid().has_value());
    ASSERT_TRUE(book.best_ask().has_value());
    EXPECT_EQ(*book.best_bid(), 10000);
    EXPECT_EQ(*book.best_ask(), 10100);
    EXPECT_EQ(*book.spread(), 100);
    EXPECT_DOUBLE_EQ(*book.mid_price(), 10050.0);
    EXPECT_EQ(book.order_count(), 2u);
}

TEST(AddCancel, BestBidIsHighestBestAskIsLowest) {
    OrderBook book;
    book.add_limit_order(Side::BID, 9900, 1);
    book.add_limit_order(Side::BID, 10000, 1);  // higher -> better bid
    book.add_limit_order(Side::BID, 9800, 1);
    book.add_limit_order(Side::ASK, 10300, 1);
    book.add_limit_order(Side::ASK, 10100, 1);  // lower -> better ask
    book.add_limit_order(Side::ASK, 10200, 1);

    EXPECT_EQ(*book.best_bid(), 10000);
    EXPECT_EQ(*book.best_ask(), 10100);
}

TEST(AddCancel, CancelRemovesOrderAndEmptyLevel) {
    OrderBook book;
    uint64_t id = book.add_limit_order(Side::BID, 10000, 5);
    EXPECT_EQ(book.order_count(), 1u);

    EXPECT_TRUE(book.cancel_order(id));
    EXPECT_EQ(book.order_count(), 0u);
    EXPECT_FALSE(book.best_bid().has_value());  // level erased
}

TEST(AddCancel, CancelOneOfManyAtSameLevel) {
    OrderBook book;
    uint64_t a = book.add_limit_order(Side::BID, 10000, 5);
    uint64_t b = book.add_limit_order(Side::BID, 10000, 3);
    (void)b;
    EXPECT_EQ(book.depth(Side::BID, 1)[0].second, 8u);

    EXPECT_TRUE(book.cancel_order(a));
    EXPECT_EQ(book.order_count(), 1u);
    ASSERT_EQ(book.depth(Side::BID, 1).size(), 1u);
    EXPECT_EQ(book.depth(Side::BID, 1)[0].second, 3u);  // only b's qty remains
}

// ---------------------------------------------------------------------------
// Cancellation of a non-existent order (graceful)
// ---------------------------------------------------------------------------
TEST(Cancel, NonExistentOrderReturnsFalse) {
    OrderBook book;
    EXPECT_FALSE(book.cancel_order(42));  // never existed

    uint64_t id = book.add_limit_order(Side::ASK, 10100, 1);
    EXPECT_TRUE(book.cancel_order(id));
    EXPECT_FALSE(book.cancel_order(id));  // double-cancel is a no-op, not a crash
}

// ---------------------------------------------------------------------------
// Marketable limit order crossing the spread
// ---------------------------------------------------------------------------
TEST(MarketableLimit, FullyCrossesAndRestsNothing) {
    OrderBook book;
    book.add_limit_order(Side::ASK, 10100, 5);  // resting ask

    // Buy limit at 10100 crosses the spread and lifts the offer.
    book.add_limit_order(Side::BID, 10100, 5);

    ASSERT_EQ(book.last_fills().size(), 1u);
    EXPECT_EQ(book.last_fills()[0].price, 10100);
    EXPECT_EQ(book.last_fills()[0].quantity, 5u);
    EXPECT_EQ(book.order_count(), 0u);  // both gone
    EXPECT_FALSE(book.best_ask().has_value());
    EXPECT_FALSE(book.best_bid().has_value());
}

TEST(MarketableLimit, CrossesPartlyThenRestsRemainder) {
    OrderBook book;
    book.add_limit_order(Side::ASK, 10100, 3);  // only 3 available

    // Buy 8 at 10100: 3 execute, 5 rest as a new bid at 10100.
    book.add_limit_order(Side::BID, 10100, 8);

    ASSERT_EQ(book.last_fills().size(), 1u);
    EXPECT_EQ(book.last_fills()[0].quantity, 3u);
    EXPECT_FALSE(book.best_ask().has_value());      // ask consumed
    ASSERT_TRUE(book.best_bid().has_value());
    EXPECT_EQ(*book.best_bid(), 10100);             // remainder rests
    EXPECT_EQ(book.depth(Side::BID, 1)[0].second, 5u);
}

TEST(MarketableLimit, NonCrossingLimitJustRests) {
    OrderBook book;
    book.add_limit_order(Side::ASK, 10100, 5);
    book.add_limit_order(Side::BID, 10000, 5);  // below the ask -> no cross

    EXPECT_TRUE(book.last_fills().empty());
    EXPECT_EQ(book.order_count(), 2u);
    EXPECT_EQ(*book.spread(), 100);
}

TEST(MarketableLimit, ExecutesAtRestingPriceNotLimitPrice) {
    OrderBook book;
    book.add_limit_order(Side::ASK, 10100, 5);   // best ask cheap
    // Aggressive buy limit at 10200 should still fill at the resting 10100.
    book.add_limit_order(Side::BID, 10200, 5);

    ASSERT_EQ(book.last_fills().size(), 1u);
    EXPECT_EQ(book.last_fills()[0].price, 10100);
}

// ---------------------------------------------------------------------------
// Partial fills (reduce in place)
// ---------------------------------------------------------------------------
TEST(PartialFill, RestingOrderReducedInPlace) {
    OrderBook book;
    uint64_t maker = book.add_limit_order(Side::BID, 10000, 10);

    // Sell 4 against the bid: maker reduced from 10 to 6, stays resting.
    auto fills = book.add_market_order(Side::ASK, 4);
    ASSERT_EQ(fills.size(), 1u);
    EXPECT_EQ(fills[0].maker_order_id, maker);
    EXPECT_EQ(fills[0].quantity, 4u);
    EXPECT_EQ(book.order_count(), 1u);
    EXPECT_EQ(book.depth(Side::BID, 1)[0].second, 6u);
}

// ---------------------------------------------------------------------------
// Multi-level market order execution
// ---------------------------------------------------------------------------
TEST(MarketOrder, SweepsMultipleLevels) {
    OrderBook book;
    book.add_limit_order(Side::ASK, 10100, 2);
    book.add_limit_order(Side::ASK, 10200, 2);
    book.add_limit_order(Side::ASK, 10300, 2);

    // Buy 5: takes 2@10100, 2@10200, 1@10300; 1 left resting at 10300.
    auto fills = book.add_market_order(Side::BID, 5);
    ASSERT_EQ(fills.size(), 3u);
    EXPECT_EQ(fills[0].price, 10100);
    EXPECT_EQ(fills[0].quantity, 2u);
    EXPECT_EQ(fills[1].price, 10200);
    EXPECT_EQ(fills[1].quantity, 2u);
    EXPECT_EQ(fills[2].price, 10300);
    EXPECT_EQ(fills[2].quantity, 1u);

    ASSERT_TRUE(book.best_ask().has_value());
    EXPECT_EQ(*book.best_ask(), 10300);
    EXPECT_EQ(book.depth(Side::ASK, 1)[0].second, 1u);
}

TEST(MarketOrder, FifoTimePriorityWithinLevel) {
    OrderBook book;
    uint64_t first  = book.add_limit_order(Side::ASK, 10100, 3);
    uint64_t second = book.add_limit_order(Side::ASK, 10100, 3);

    // Buy 4: fully fills the first (3), partially the second (1).
    auto fills = book.add_market_order(Side::BID, 4);
    ASSERT_EQ(fills.size(), 2u);
    EXPECT_EQ(fills[0].maker_order_id, first);   // oldest first
    EXPECT_EQ(fills[0].quantity, 3u);
    EXPECT_EQ(fills[1].maker_order_id, second);
    EXPECT_EQ(fills[1].quantity, 1u);
    EXPECT_EQ(book.depth(Side::ASK, 1)[0].second, 2u);  // 2 left on second
}

TEST(MarketOrder, ConsumesAllLiquidityThenStops) {
    OrderBook book;
    book.add_limit_order(Side::BID, 10000, 2);
    book.add_limit_order(Side::BID, 9900, 2);

    // Sell 10 but only 4 available: 4 fill, rest is dropped (market doesn't rest).
    auto fills = book.add_market_order(Side::ASK, 10);
    uint64_t total = 0;
    for (const auto& f : fills) total += f.quantity;
    EXPECT_EQ(total, 4u);
    EXPECT_EQ(book.order_count(), 0u);
    EXPECT_FALSE(book.best_bid().has_value());
}

// ---------------------------------------------------------------------------
// Depth and imbalance
// ---------------------------------------------------------------------------
TEST(Depth, ReturnsTopNLevelsBestFirst) {
    OrderBook book;
    book.add_limit_order(Side::BID, 10000, 1);
    book.add_limit_order(Side::BID, 9900, 2);
    book.add_limit_order(Side::BID, 9800, 3);

    auto d = book.depth(Side::BID, 2);
    ASSERT_EQ(d.size(), 2u);
    EXPECT_EQ(d[0].first, 10000);
    EXPECT_EQ(d[0].second, 1u);
    EXPECT_EQ(d[1].first, 9900);
    EXPECT_EQ(d[1].second, 2u);
}

TEST(Imbalance, ComputedOverTopNLevels) {
    OrderBook book;
    book.add_limit_order(Side::BID, 10000, 6);  // bid vol 6
    book.add_limit_order(Side::ASK, 10100, 2);  // ask vol 2
    // (6 - 2) / (6 + 2) = 0.5
    EXPECT_DOUBLE_EQ(book.order_book_imbalance(1), 0.5);
}

TEST(Imbalance, BalancedBookIsZero) {
    OrderBook book;
    book.add_limit_order(Side::BID, 10000, 5);
    book.add_limit_order(Side::ASK, 10100, 5);
    EXPECT_DOUBLE_EQ(book.order_book_imbalance(1), 0.0);
}
