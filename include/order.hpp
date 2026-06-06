#pragma once

#include <cstdint>

namespace lob {

// Which side of the book an order sits on.
//   BID = buy order  (rests on the bid side, max-price priority)
//   ASK = sell order (rests on the ask side, min-price priority)
enum class Side : uint8_t { BID = 0, ASK = 1 };

inline Side opposite(Side s) noexcept {
    return s == Side::BID ? Side::ASK : Side::BID;
}

// A single resting order.
//
// The order is *intrusive*: the `prev`/`next` pointers that thread it into a
// price level's doubly-linked list live inside the struct itself, so there is
// no separate wrapper node to allocate. This eliminates one heap allocation
// per order and keeps the list nodes (the orders) contiguous-ish in the pool,
// which is friendly to the cache during matching.
//
// `next` is reused as the free-list link while the order sits in the pool.
struct Order {
    uint64_t order_id   = 0;          // unique, monotonically assigned
    Side     side       = Side::BID;  // BID or ASK
    int64_t  price      = 0;          // price in integer ticks (e.g. dollars * 100)
    uint64_t quantity   = 0;          // remaining (resting) quantity
    uint64_t timestamp  = 0;          // arrival sequence, ns-scale; gives time priority

    // Intrusive doubly-linked-list pointers within a PriceLevel.
    Order*   prev       = nullptr;
    Order*   next       = nullptr;
};

// The result of one resting order being (partially) executed by an incoming
// marketable order. One incoming order can produce many fills.
struct Fill {
    uint64_t maker_order_id = 0;  // the resting order that was hit
    int64_t  price          = 0;  // execution price in ticks (the maker's price)
    uint64_t quantity       = 0;  // quantity traded in this fill
};

}  // namespace lob
