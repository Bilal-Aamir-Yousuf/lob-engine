#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "order_book.hpp"

namespace lob {

// Field separator used by the FIX protocol (Start Of Header, 0x01). For
// human-readable logging/tests a caller may substitute '|'.
constexpr char FIX_SOH = '\x01';

// ---------------------------------------------------------------------------
// FixMessage -- an ordered list of tag=value fields.
// ---------------------------------------------------------------------------
// FIX is a flat, self-describing, tag=value wire format. Every message is a
// sequence of `tag=value<SOH>` fields, framed by:
//   8  = BeginString  (e.g. "FIX.4.2")            -- always first
//   9  = BodyLength    (#bytes between 9 and 10)   -- always second
//   35 = MsgType       (D=NewOrderSingle, F=Cancel, 8=ExecReport, 9=CxlReject)
//   ...
//   10 = CheckSum      (sum of all prior bytes mod 256, 3 digits) -- always last
class FixMessage {
public:
    void set(int tag, const std::string& value);
    void set(int tag, long long value);

    bool has(int tag) const { return index_.count(tag) > 0; }
    std::optional<std::string> get(int tag) const;
    std::string get_or(int tag, const std::string& def) const;
    long long get_int(int tag, long long def = 0) const;
    double get_double(int tag, double def = 0.0) const;
    std::string msg_type() const { return get_or(35, ""); }

    const std::vector<std::pair<int, std::string>>& fields() const { return fields_; }

    // Serialize with correct 8/9/10 framing. begin_string defaults to FIX.4.2;
    // any 8/9/10 already present is ignored and recomputed.
    std::string encode(char delim = FIX_SOH, const std::string& begin_string = "FIX.4.2") const;

    // Parse a raw FIX string. Tolerant of '\x01' or '|' (or any delim). Fields
    // are stored in order; CheckSum is NOT required to be valid to parse.
    static FixMessage parse(const std::string& raw, char delim = FIX_SOH);

    // Validate that tag 10 matches the recomputed checksum (true if absent).
    static bool checksum_valid(const std::string& raw, char delim = FIX_SOH);

private:
    std::vector<std::pair<int, std::string>> fields_;
    std::unordered_map<int, std::string> index_;
};

// ---------------------------------------------------------------------------
// FixGateway -- routes FIX order messages to an OrderBook.
// ---------------------------------------------------------------------------
// Bridges the external decimal-price FIX world to the integer-tick engine,
// tracks the ClOrdID <-> engine order_id mapping needed for cancels, and emits
// ExecutionReports (35=8) / OrderCancelReject (35=9) responses.
class FixGateway {
public:
    explicit FixGateway(OrderBook& book, double tick_size = 0.01,
                        std::string comp_id = "LOB-ENGINE")
        : book_(book), tick_size_(tick_size), comp_id_(std::move(comp_id)) {}

    // Process one inbound message; returns the response messages (structured).
    std::vector<FixMessage> process(const FixMessage& in);

    // Convenience: parse a raw inbound string and return encoded responses.
    std::vector<std::string> process_raw(const std::string& raw, char delim = FIX_SOH);

private:
    FixMessage handle_new_order(const FixMessage& in);          // 35=D
    std::vector<FixMessage> handle_new_order_multi(const FixMessage& in);
    FixMessage handle_cancel(const FixMessage& in);             // 35=F

    FixMessage base_report(const FixMessage& in, char msg_type);
    int64_t to_ticks(double price) const;
    double  to_price(int64_t ticks) const;

    OrderBook& book_;
    double tick_size_;
    std::string comp_id_;
    std::unordered_map<std::string, uint64_t> clord_to_id_;
    std::unordered_map<uint64_t, std::string> id_to_clord_;
    uint64_t exec_seq_ = 1;
};

}  // namespace lob
