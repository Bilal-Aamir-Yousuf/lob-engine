#include "fix.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace lob {

// ---------------------------------------------------------------------------
// FixMessage
// ---------------------------------------------------------------------------
void FixMessage::set(int tag, const std::string& value) {
    if (index_.count(tag)) {
        for (auto& f : fields_)
            if (f.first == tag) { f.second = value; break; }
    } else {
        fields_.emplace_back(tag, value);
    }
    index_[tag] = value;
}

void FixMessage::set(int tag, long long value) { set(tag, std::to_string(value)); }

std::optional<std::string> FixMessage::get(int tag) const {
    auto it = index_.find(tag);
    if (it == index_.end()) return std::nullopt;
    return it->second;
}

std::string FixMessage::get_or(int tag, const std::string& def) const {
    auto it = index_.find(tag);
    return it == index_.end() ? def : it->second;
}

long long FixMessage::get_int(int tag, long long def) const {
    auto it = index_.find(tag);
    return it == index_.end() ? def : std::strtoll(it->second.c_str(), nullptr, 10);
}

double FixMessage::get_double(int tag, double def) const {
    auto it = index_.find(tag);
    return it == index_.end() ? def : std::strtod(it->second.c_str(), nullptr);
}

std::string FixMessage::encode(char delim, const std::string& begin_string) const {
    // Body = every field except 8/9/10, in insertion order.
    std::string body;
    for (const auto& [tag, value] : fields_) {
        if (tag == 8 || tag == 9 || tag == 10) continue;
        body += std::to_string(tag) + "=" + value + delim;
    }
    std::string head = "8=" + begin_string + delim + "9=" +
                       std::to_string(body.size()) + delim;
    std::string msg = head + body;

    unsigned int sum = 0;
    for (unsigned char c : msg) sum += c;
    char cs[8];
    std::snprintf(cs, sizeof(cs), "%03u", sum % 256);
    msg += "10=" + std::string(cs) + delim;
    return msg;
}

FixMessage FixMessage::parse(const std::string& raw, char delim) {
    FixMessage m;
    std::size_t i = 0;
    while (i < raw.size()) {
        std::size_t eq = raw.find('=', i);
        if (eq == std::string::npos) break;
        std::size_t end = raw.find(delim, eq + 1);
        if (end == std::string::npos) end = raw.size();
        const int tag = std::atoi(raw.substr(i, eq - i).c_str());
        m.set(tag, raw.substr(eq + 1, end - eq - 1));
        i = end + 1;
    }
    return m;
}

bool FixMessage::checksum_valid(const std::string& raw, char delim) {
    const std::string needle = std::string("10=");
    std::size_t pos = raw.rfind(needle);
    if (pos == std::string::npos) return true;  // no checksum -> nothing to check
    unsigned int sum = 0;
    for (std::size_t k = 0; k < pos; ++k) sum += static_cast<unsigned char>(raw[k]);
    std::size_t end = raw.find(delim, pos);
    if (end == std::string::npos) end = raw.size();
    const int claimed = std::atoi(raw.substr(pos + needle.size(),
                                              end - pos - needle.size()).c_str());
    return static_cast<int>(sum % 256) == claimed;
}

// ---------------------------------------------------------------------------
// FixGateway
// ---------------------------------------------------------------------------
namespace {
// FIX tag numbers used here.
enum Tag {
    BeginString = 8, BodyLength = 9, MsgType = 35, CheckSum = 10,
    SenderCompID = 49, TargetCompID = 56,
    ClOrdID = 11, OrigClOrdID = 41, OrderID = 37, ExecID = 17,
    SideTag = 54, OrderQty = 38, Price = 44, OrdType = 40,
    ExecType = 150, OrdStatus = 39, LastQty = 32, LastPx = 31,
    CumQty = 14, LeavesQty = 151, Text = 58,
    CxlRejResponseTo = 434, CxlRejReason = 102,
};
}  // namespace

int64_t FixGateway::to_ticks(double price) const {
    return static_cast<int64_t>(std::llround(price / tick_size_));
}
double FixGateway::to_price(int64_t ticks) const {
    return static_cast<double>(ticks) * tick_size_;
}

FixMessage FixGateway::base_report(const FixMessage& in, char msg_type) {
    FixMessage r;
    r.set(MsgType, std::string(1, msg_type));
    r.set(SenderCompID, comp_id_);                       // we are the sender now
    r.set(TargetCompID, in.get_or(SenderCompID, "CLIENT"));  // reply to them
    r.set(ExecID, "E" + std::to_string(exec_seq_++));
    return r;
}

std::vector<FixMessage> FixGateway::process(const FixMessage& in) {
    const std::string type = in.msg_type();
    if (type == "D") return handle_new_order_multi(in);
    if (type == "F") return {handle_cancel(in)};
    // Unsupported business message -> a reject-style ExecutionReport.
    FixMessage r = base_report(in, '8');
    r.set(ClOrdID, in.get_or(ClOrdID, ""));
    r.set(ExecType, "8");   // Rejected
    r.set(OrdStatus, "8");
    r.set(Text, "unsupported MsgType=" + type);
    return {r};
}

std::vector<FixMessage> FixGateway::handle_new_order_multi(const FixMessage& in) {
    const std::string clord = in.get_or(ClOrdID, "");
    const std::string side_s = in.get_or(SideTag, "1");     // 1=Buy, 2=Sell
    const lob::Side side = (side_s == "1") ? lob::Side::BID : lob::Side::ASK;
    const uint64_t qty = static_cast<uint64_t>(in.get_int(OrderQty, 0));
    const std::string ordtype = in.get_or(OrdType, "2");    // 1=Market, 2=Limit
    const bool is_market = (ordtype == "1");

    std::vector<FixMessage> out;
    if (qty == 0) {
        FixMessage r = base_report(in, '8');
        r.set(ClOrdID, clord);
        r.set(SideTag, side_s);
        r.set(ExecType, "8");
        r.set(OrdStatus, "8");
        r.set(Text, "OrderQty must be > 0");
        return {r};
    }

    std::vector<Fill> fills;
    uint64_t engine_id = 0;
    uint64_t leaves = qty;

    if (is_market) {
        fills = book_.add_market_order(side, qty);
    } else {
        const int64_t px = to_ticks(in.get_double(Price, 0.0));
        engine_id = book_.add_limit_order(side, px, qty);
        fills = book_.last_fills();
    }

    uint64_t cum = 0;
    for (std::size_t k = 0; k < fills.size(); ++k) {
        cum += fills[k].quantity;
        leaves -= fills[k].quantity;
        const bool done = (leaves == 0);
        FixMessage r = base_report(in, '8');
        r.set(OrderID, std::to_string(engine_id));
        r.set(ClOrdID, clord);
        r.set(SideTag, side_s);
        r.set(OrderQty, static_cast<long long>(qty));
        r.set(ExecType, done ? "2" : "1");      // Fill / Partial fill
        r.set(OrdStatus, done ? "2" : "1");
        r.set(LastQty, static_cast<long long>(fills[k].quantity));
        char px[32];
        std::snprintf(px, sizeof(px), "%.4f", to_price(fills[k].price));
        r.set(LastPx, std::string(px));
        r.set(CumQty, static_cast<long long>(cum));
        r.set(LeavesQty, static_cast<long long>(leaves));
        out.push_back(r);
    }

    if (!is_market && leaves > 0) {
        // Remainder rests in the book -> trackable for cancellation.
        clord_to_id_[clord] = engine_id;
        id_to_clord_[engine_id] = clord;
        if (fills.empty()) {  // pure New acknowledgement
            FixMessage r = base_report(in, '8');
            r.set(OrderID, std::to_string(engine_id));
            r.set(ClOrdID, clord);
            r.set(SideTag, side_s);
            r.set(OrderQty, static_cast<long long>(qty));
            r.set(ExecType, "0");   // New
            r.set(OrdStatus, "0");
            r.set(CumQty, "0");
            r.set(LeavesQty, static_cast<long long>(leaves));
            out.push_back(r);
        }
    } else if (is_market && fills.empty()) {
        // Market order with no liquidity -> cannot rest, reject.
        FixMessage r = base_report(in, '8');
        r.set(ClOrdID, clord);
        r.set(SideTag, side_s);
        r.set(ExecType, "8");
        r.set(OrdStatus, "8");
        r.set(Text, "no liquidity for market order");
        out.push_back(r);
    }
    return out;
}

FixMessage FixGateway::handle_cancel(const FixMessage& in) {
    const std::string orig = in.get_or(OrigClOrdID, "");
    const std::string clord = in.get_or(ClOrdID, "");

    auto it = clord_to_id_.find(orig);
    if (it == clord_to_id_.end() || !book_.cancel_order(it->second)) {
        // Unknown / already-gone order -> OrderCancelReject (35=9).
        FixMessage r = base_report(in, '9');
        r.set(ClOrdID, clord);
        r.set(OrigClOrdID, orig);
        r.set(OrdStatus, "8");           // Rejected
        r.set(CxlRejResponseTo, "1");    // response to OrderCancelRequest
        r.set(CxlRejReason, "1");        // unknown order
        r.set(Text, "unknown or already-inactive order");
        return r;
    }

    const uint64_t engine_id = it->second;
    clord_to_id_.erase(it);
    id_to_clord_.erase(engine_id);

    FixMessage r = base_report(in, '8');
    r.set(OrderID, std::to_string(engine_id));
    r.set(ClOrdID, clord);
    r.set(OrigClOrdID, orig);
    r.set(ExecType, "4");    // Canceled
    r.set(OrdStatus, "4");
    r.set(LeavesQty, "0");
    return r;
}

std::vector<std::string> FixGateway::process_raw(const std::string& raw, char delim) {
    FixMessage in = FixMessage::parse(raw, delim);
    std::vector<std::string> out;
    for (const auto& r : process(in)) out.push_back(r.encode(delim));
    return out;
}

}  // namespace lob
