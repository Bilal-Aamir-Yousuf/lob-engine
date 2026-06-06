#include "order_book.hpp"

#include <algorithm>

namespace lob {

void OrderBook::match(Side taker_side, int64_t limit_price, uint64_t& quantity,
                      bool is_market) {
    if (taker_side == Side::BID) {
        // Incoming buy: take liquidity from the ask side, cheapest first.
        auto it = asks_.begin();
        while (quantity > 0 && it != asks_.end()) {
            const int64_t level_price = it->first;
            if (!is_market && level_price > limit_price) {
                break;  // best ask is above our limit -> no more crossing
            }
            PriceLevel& level = it->second;
            while (quantity > 0 && level.head != nullptr) {
                Order* maker = level.head;  // oldest order at this level (FIFO)
                const uint64_t traded = std::min(quantity, maker->quantity);
                maker->quantity     -= traded;
                level.total_volume  -= traded;
                quantity            -= traded;
                last_fills_.push_back({maker->order_id, level_price, traded});
                if (maker->quantity == 0) {
                    level.unlink(maker);
                    lookup_.erase(maker->order_id);
                    pool_.deallocate(maker);
                }
            }
            if (level.empty()) {
                it = asks_.erase(it);
            } else {
                ++it;  // level not exhausted means quantity hit 0; loop ends
            }
        }
    } else {
        // Incoming sell: take liquidity from the bid side, highest first.
        auto it = bids_.rbegin();
        while (quantity > 0 && it != bids_.rend()) {
            const int64_t level_price = it->first;
            if (!is_market && level_price < limit_price) {
                break;  // best bid is below our limit -> no more crossing
            }
            PriceLevel& level = it->second;
            while (quantity > 0 && level.head != nullptr) {
                Order* maker = level.head;
                const uint64_t traded = std::min(quantity, maker->quantity);
                maker->quantity     -= traded;
                level.total_volume  -= traded;
                quantity            -= traded;
                last_fills_.push_back({maker->order_id, level_price, traded});
                if (maker->quantity == 0) {
                    level.unlink(maker);
                    lookup_.erase(maker->order_id);
                    pool_.deallocate(maker);
                }
            }
            if (level.empty()) {
                // Erase via the underlying forward iterator. For a reverse
                // iterator `it`, std::next(it).base() is the forward iterator to
                // the *same* element. erase() invalidates that iterator and
                // returns the next forward element; wrapping the RETURN value in
                // make_reverse_iterator lands us on the next-best bid. (Using the
                // pre-erase iterator here would be a dangling-iterator bug.)
                auto fwd = std::next(it).base();        // forward iterator to current level
                auto after = bids_.erase(fwd);          // -> next forward element
                it = std::make_reverse_iterator(after); // -> next-best (lower) bid
            } else {
                ++it;
            }
        }
    }
}

uint64_t OrderBook::add_limit_order(Side side, int64_t price, uint64_t quantity) {
    last_fills_.clear();
    const uint64_t id = next_id_++;

    // Execute any marketable portion first.
    if (quantity > 0) {
        match(side, price, quantity, /*is_market=*/false);
    }

    // Rest the remainder (if any).
    if (quantity > 0) {
        Order* o = pool_.allocate();
        o->order_id  = id;
        o->side      = side;
        o->price     = price;
        o->quantity  = quantity;
        o->timestamp = clock_++;

        auto& book = (side == Side::BID) ? bids_ : asks_;
        book[price].append(o);
        lookup_[id] = o;
    }
    return id;
}

bool OrderBook::cancel_order(uint64_t order_id) {
    auto it = lookup_.find(order_id);
    if (it == lookup_.end()) {
        return false;  // unknown id -> graceful no-op
    }
    Order* o = it->second;
    auto& book = (o->side == Side::BID) ? bids_ : asks_;
    auto level_it = book.find(o->price);

    PriceLevel& level = level_it->second;
    level.total_volume -= o->quantity;  // subtract remaining qty before unlinking
    level.unlink(o);
    if (level.empty()) {
        book.erase(level_it);
    }
    lookup_.erase(it);
    pool_.deallocate(o);
    return true;
}

std::vector<Fill> OrderBook::add_market_order(Side side, uint64_t quantity) {
    last_fills_.clear();
    if (quantity > 0) {
        match(side, /*limit_price=*/0, quantity, /*is_market=*/true);
    }
    return last_fills_;  // unfilled remainder (if liquidity ran out) is dropped
}

std::optional<int64_t> OrderBook::best_bid() const {
    if (bids_.empty()) return std::nullopt;
    return bids_.rbegin()->first;
}

std::optional<int64_t> OrderBook::best_ask() const {
    if (asks_.empty()) return std::nullopt;
    return asks_.begin()->first;
}

std::optional<double> OrderBook::mid_price() const {
    auto b = best_bid();
    auto a = best_ask();
    if (!b || !a) return std::nullopt;
    return (static_cast<double>(*b) + static_cast<double>(*a)) / 2.0;
}

std::optional<int64_t> OrderBook::spread() const {
    auto b = best_bid();
    auto a = best_ask();
    if (!b || !a) return std::nullopt;
    return *a - *b;
}

std::vector<std::pair<int64_t, uint64_t>> OrderBook::depth(Side side,
                                                           std::size_t n_levels) const {
    std::vector<std::pair<int64_t, uint64_t>> out;
    out.reserve(n_levels);
    if (side == Side::BID) {
        // Highest price first.
        for (auto it = bids_.rbegin(); it != bids_.rend() && out.size() < n_levels; ++it) {
            out.emplace_back(it->first, it->second.total_volume);
        }
    } else {
        // Lowest price first.
        for (auto it = asks_.begin(); it != asks_.end() && out.size() < n_levels; ++it) {
            out.emplace_back(it->first, it->second.total_volume);
        }
    }
    return out;
}

double OrderBook::order_book_imbalance(std::size_t n_levels) const {
    uint64_t bid_vol = 0;
    uint64_t ask_vol = 0;
    {
        std::size_t k = 0;
        for (auto it = bids_.rbegin(); it != bids_.rend() && k < n_levels; ++it, ++k) {
            bid_vol += it->second.total_volume;
        }
    }
    {
        std::size_t k = 0;
        for (auto it = asks_.begin(); it != asks_.end() && k < n_levels; ++it, ++k) {
            ask_vol += it->second.total_volume;
        }
    }
    const uint64_t total = bid_vol + ask_vol;
    if (total == 0) return 0.0;
    return (static_cast<double>(bid_vol) - static_cast<double>(ask_vol)) /
           static_cast<double>(total);
}

}  // namespace lob
