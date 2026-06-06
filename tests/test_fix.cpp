#include <gtest/gtest.h>

#include "fix.hpp"
#include "order_book.hpp"

using lob::FixGateway;
using lob::FixMessage;
using lob::OrderBook;

namespace {

// Build a NewOrderSingle (35=D). Use '|' as the delimiter for readability.
FixMessage new_order(const std::string& clord, const std::string& side,
                     long long qty, const std::string& price,
                     const std::string& ordtype = "2") {
    FixMessage m;
    m.set(35, std::string("D"));
    m.set(49, std::string("CLIENT"));
    m.set(56, std::string("LOB-ENGINE"));
    m.set(11, clord);
    m.set(54, side);          // 1=buy, 2=sell
    m.set(38, qty);
    if (ordtype == "2") m.set(44, price);
    m.set(40, ordtype);       // 1=market, 2=limit
    return m;
}

}  // namespace

// ---------------------------------------------------------------------------
// Wire format: framing, checksum, round-trip
// ---------------------------------------------------------------------------
TEST(FixMessage, EncodeFramesAndChecksums) {
    FixMessage m = new_order("ORD1", "1", 100, "100.50");
    const std::string wire = m.encode('|');

    // Must start with BeginString then BodyLength, end with CheckSum.
    EXPECT_EQ(wire.rfind("8=FIX.4.2|9=", 0), 0u);
    EXPECT_NE(wire.find("|10="), std::string::npos);
    EXPECT_TRUE(FixMessage::checksum_valid(wire, '|'));
}

TEST(FixMessage, RoundTripPreservesFields) {
    FixMessage m = new_order("ORD1", "1", 100, "100.50");
    FixMessage p = FixMessage::parse(m.encode('|'), '|');
    EXPECT_EQ(p.msg_type(), "D");
    EXPECT_EQ(p.get_or(11, ""), "ORD1");
    EXPECT_EQ(p.get_int(38), 100);
    EXPECT_DOUBLE_EQ(p.get_double(44), 100.50);
}

TEST(FixMessage, DetectsCorruptedChecksum) {
    std::string wire = new_order("ORD1", "1", 100, "100.50").encode('|');
    // Flip a byte in the body; the trailing checksum no longer matches.
    const std::size_t pos = wire.find("54=1");
    ASSERT_NE(pos, std::string::npos);
    wire[pos + 3] = '2';
    EXPECT_FALSE(FixMessage::checksum_valid(wire, '|'));
}

// ---------------------------------------------------------------------------
// Gateway routing
// ---------------------------------------------------------------------------
TEST(FixGateway, LimitOrderRestsAndAcks) {
    OrderBook book;
    FixGateway gw(book, /*tick_size=*/0.01);

    auto reps = gw.process(new_order("ORD1", "1", 100, "100.50"));
    ASSERT_EQ(reps.size(), 1u);
    EXPECT_EQ(reps[0].msg_type(), "8");     // ExecutionReport
    EXPECT_EQ(reps[0].get_or(150, ""), "0"); // New
    EXPECT_EQ(reps[0].get_int(151), 100);    // LeavesQty
    ASSERT_TRUE(book.best_bid().has_value());
    EXPECT_EQ(*book.best_bid(), 10050);      // 100.50 / 0.01
}

TEST(FixGateway, MarketOrderFillsAndReportsTrades) {
    OrderBook book;
    FixGateway gw(book, 0.01);
    gw.process(new_order("MAKER", "1", 100, "100.50"));  // resting bid

    auto reps = gw.process(new_order("TAKER", "2", 40, "", "1"));  // market sell 40
    ASSERT_EQ(reps.size(), 1u);
    EXPECT_EQ(reps[0].get_or(150, ""), "2");        // fully filled (taker)
    EXPECT_EQ(reps[0].get_int(32), 40);             // LastQty
    EXPECT_DOUBLE_EQ(reps[0].get_double(31), 100.50);  // LastPx
    EXPECT_EQ(book.depth(lob::Side::BID, 1)[0].second, 60u);  // 60 left resting
}

TEST(FixGateway, MarketableLimitProducesFillThenRests) {
    OrderBook book;
    FixGateway gw(book, 0.01);
    gw.process(new_order("ASK1", "2", 30, "100.50"));  // resting ask 30

    // Buy 50 limit at 100.50: 30 fill, 20 rest.
    auto reps = gw.process(new_order("BUY1", "1", 50, "100.50"));
    ASSERT_EQ(reps.size(), 1u);
    EXPECT_EQ(reps[0].get_or(150, ""), "1");   // partial fill
    EXPECT_EQ(reps[0].get_int(32), 30);        // LastQty
    EXPECT_EQ(reps[0].get_int(151), 20);       // LeavesQty rests
    EXPECT_EQ(*book.best_bid(), 10050);
}

TEST(FixGateway, CancelKnownOrder) {
    OrderBook book;
    FixGateway gw(book, 0.01);
    gw.process(new_order("ORD1", "1", 100, "100.50"));

    FixMessage cxl;
    cxl.set(35, std::string("F"));
    cxl.set(41, std::string("ORD1"));  // OrigClOrdID
    cxl.set(11, std::string("CXL1"));
    auto reps = gw.process(cxl);

    ASSERT_EQ(reps.size(), 1u);
    EXPECT_EQ(reps[0].msg_type(), "8");
    EXPECT_EQ(reps[0].get_or(150, ""), "4");   // Canceled
    EXPECT_FALSE(book.best_bid().has_value());
    EXPECT_EQ(book.order_count(), 0u);
}

TEST(FixGateway, CancelUnknownOrderIsRejected) {
    OrderBook book;
    FixGateway gw(book, 0.01);

    FixMessage cxl;
    cxl.set(35, std::string("F"));
    cxl.set(41, std::string("DOES-NOT-EXIST"));
    cxl.set(11, std::string("CXL9"));
    auto reps = gw.process(cxl);

    ASSERT_EQ(reps.size(), 1u);
    EXPECT_EQ(reps[0].msg_type(), "9");        // OrderCancelReject
    EXPECT_EQ(reps[0].get_or(434, ""), "1");   // in response to a cancel request
}

TEST(FixGateway, RawStringInOutAndChecksum) {
    OrderBook book;
    FixGateway gw(book, 0.01);
    const std::string in = new_order("ORD1", "1", 100, "100.50").encode('|');

    auto out = gw.process_raw(in, '|');
    ASSERT_EQ(out.size(), 1u);
    EXPECT_TRUE(FixMessage::checksum_valid(out[0], '|'));
    FixMessage parsed = FixMessage::parse(out[0], '|');
    EXPECT_EQ(parsed.msg_type(), "8");
}
