#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

#include "order.hpp"

namespace lob {

// ---------------------------------------------------------------------------
// OrderPool
// ---------------------------------------------------------------------------
// A simple free-list object pool for Order structs. Real exchanges never call
// malloc on the hot path; we allocate orders in large blocks and recycle freed
// slots via an intrusive free list (reusing Order::next). Allocation and
// deallocation are O(1) with no per-order heap traffic in steady state.
class OrderPool {
public:
    explicit OrderPool(std::size_t block_size = 4096) : block_size_(block_size) {}

    Order* allocate() {
        if (free_head_ == nullptr) {
            grow();
        }
        Order* o = free_head_;
        free_head_ = free_head_->next;
        *o = Order{};  // reset to a clean state
        return o;
    }

    void deallocate(Order* o) noexcept {
        o->next = free_head_;
        free_head_ = o;
    }

private:
    void grow() {
        auto block = std::make_unique<Order[]>(block_size_);
        // Thread the new slots onto the free list.
        for (std::size_t i = 0; i < block_size_; ++i) {
            block[i].next = free_head_;
            free_head_ = &block[i];
        }
        blocks_.push_back(std::move(block));
    }

    std::size_t block_size_;
    std::vector<std::unique_ptr<Order[]>> blocks_;
    Order* free_head_ = nullptr;
};

// ---------------------------------------------------------------------------
// PriceLevel
// ---------------------------------------------------------------------------
// All resting orders at one price, kept as an intrusive FIFO doubly-linked
// list (head = oldest = first to execute, giving time priority). total_volume
// is the sum of resting quantities so depth/imbalance queries are O(1) per
// level.
struct PriceLevel {
    Order*   head         = nullptr;
    Order*   tail         = nullptr;
    uint64_t total_volume = 0;
    uint32_t order_count  = 0;

    // Append to the back: newest order, lowest time priority.
    void append(Order* o) noexcept {
        o->prev = tail;
        o->next = nullptr;
        if (tail) {
            tail->next = o;
        } else {
            head = o;
        }
        tail = o;
        total_volume += o->quantity;
        ++order_count;
    }

    // Splice an order out of the list. Does NOT touch total_volume — callers
    // adjust volume explicitly (matching decrements as it trades; cancel
    // subtracts the remaining quantity first). This keeps the volume
    // accounting in one place per operation.
    void unlink(Order* o) noexcept {
        if (o->prev) {
            o->prev->next = o->next;
        } else {
            head = o->next;
        }
        if (o->next) {
            o->next->prev = o->prev;
        } else {
            tail = o->prev;
        }
        o->prev = o->next = nullptr;
        --order_count;
    }

    bool empty() const noexcept { return head == nullptr; }
};

// ---------------------------------------------------------------------------
// OrderBook
// ---------------------------------------------------------------------------
// Price-time priority limit order book.
//
//   bids_ : std::map<int64_t, PriceLevel>  -> best bid = highest price = rbegin()
//   asks_ : std::map<int64_t, PriceLevel>  -> best ask = lowest price  = begin()
//   lookup_ : order_id -> Order*           -> O(1) cancellation
//
// Prices are integer ticks throughout (no floating point in the engine).
class OrderBook {
public:
    OrderBook() = default;

    // Add a limit order. If it crosses the spread it executes immediately as a
    // marketable limit order against the opposing side; any unfilled remainder
    // rests in the book. Returns the assigned order_id.
    uint64_t add_limit_order(Side side, int64_t price, uint64_t quantity);

    // Cancel a resting order by id. Returns false if the id is unknown
    // (already filled, already cancelled, or never existed) -- handled
    // gracefully, no throw.
    bool cancel_order(uint64_t order_id);

    // Add a market order: sweep the opposing side from best to worst until the
    // quantity is filled or liquidity runs out. Returns the list of fills.
    std::vector<Fill> add_market_order(Side side, uint64_t quantity);

    // The fills produced by the most recent add_limit_order / add_market_order
    // call (useful for limit orders, which return an id rather than fills).
    const std::vector<Fill>& last_fills() const noexcept { return last_fills_; }

    // ---- Book state queries (all in ticks) -------------------------------
    std::optional<int64_t> best_bid() const;
    std::optional<int64_t> best_ask() const;
    std::optional<double>  mid_price() const;
    std::optional<int64_t> spread() const;

    // Volume at each of the top n price levels, best to worst.
    std::vector<std::pair<int64_t, uint64_t>> depth(Side side, std::size_t n_levels) const;

    // (bid_vol - ask_vol) / (bid_vol + ask_vol) over the top n levels.
    // Returns 0.0 when both sides are empty.
    double order_book_imbalance(std::size_t n_levels) const;

    // Number of resting orders currently in the book.
    std::size_t order_count() const noexcept { return lookup_.size(); }

private:
    // Match `quantity` of an incoming `taker_side` order against the book.
    // For a market order pass is_market = true (price ignored). For a
    // marketable limit, pass the limit price; matching stops once the best
    // opposing level is no longer crossable. `quantity` is updated in place to
    // the unfilled remainder.
    void match(Side taker_side, int64_t limit_price, uint64_t& quantity, bool is_market);

    std::map<int64_t, PriceLevel> bids_;
    std::map<int64_t, PriceLevel> asks_;
    std::unordered_map<uint64_t, Order*> lookup_;
    OrderPool pool_;

    uint64_t next_id_ = 1;
    uint64_t clock_   = 0;  // monotonic arrival sequence -> time priority
    std::vector<Fill> last_fills_;
};

}  // namespace lob
