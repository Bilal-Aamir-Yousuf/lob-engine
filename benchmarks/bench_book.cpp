// Benchmarks for the limit order book.
//
//  * Throughput  -> Google Benchmark (items/second).
//  * Latency tail-> a direct TSC harness that records EVERY operation's cycle
//                   count, converts to nanoseconds, and reports p50/p95/p99/
//                   p99.9. (Google Benchmark aggregates per-repetition, so it
//                   cannot report true per-operation percentiles.)

#include <benchmark/benchmark.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

#include "order_book.hpp"

using lob::OrderBook;
using lob::Side;

// ===========================================================================
// Google Benchmark: throughput
// ===========================================================================
namespace {
constexpr int64_t kBase = 100000;       // base price in ticks
constexpr std::size_t kResetEvery = 100000;  // bound memory in add-only loops
}  // namespace

static void BM_AddLimitOrder(benchmark::State& state) {
    OrderBook book;
    uint64_t i = 0;
    std::size_t since_reset = 0;
    for (auto _ : state) {
        book.add_limit_order(Side::BID, kBase + static_cast<int64_t>(i++ % 1024), 1);
        if (++since_reset == kResetEvery) {
            state.PauseTiming();
            book = OrderBook();
            since_reset = 0;
            state.ResumeTiming();
        }
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_AddLimitOrder);

static void BM_CancelOrder(benchmark::State& state) {
    OrderBook book;
    std::vector<uint64_t> ids;
    std::size_t idx = 0;
    for (auto _ : state) {
        if (idx >= ids.size()) {  // refill (untimed)
            state.PauseTiming();
            book = OrderBook();
            ids.clear();
            ids.reserve(kResetEvery);
            for (std::size_t k = 0; k < kResetEvery; ++k)
                ids.push_back(book.add_limit_order(
                    Side::BID, kBase + static_cast<int64_t>(k % 1024), 1));
            idx = 0;
            state.ResumeTiming();
        }
        book.cancel_order(ids[idx++]);
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_CancelOrder);

static void BM_AddMarketOrder(benchmark::State& state) {
    const int64_t depth = state.range(0);  // resting orders the market order sweeps
    OrderBook book;
    for (auto _ : state) {
        state.PauseTiming();
        for (int64_t k = 0; k < depth; ++k)
            book.add_limit_order(Side::ASK, kBase, 1);
        state.ResumeTiming();
        book.add_market_order(Side::BID, static_cast<uint64_t>(depth));
    }
    state.SetItemsProcessed(state.iterations() * depth);  // orders matched
}
BENCHMARK(BM_AddMarketOrder)->Arg(1)->Arg(10)->Arg(100)->Arg(1000);

// ===========================================================================
// TSC latency harness: per-operation percentiles
// ===========================================================================
namespace {

inline uint64_t rdtscp_now() {
#if defined(_MSC_VER)
    unsigned int aux;
    return __rdtscp(&aux);
#else
    return __builtin_ia32_rdtsc();
#endif
}

// Calibrate TSC cycles-per-nanosecond against a steady wall clock.
double calibrate_cycles_per_ns() {
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();
    const uint64_t c0 = rdtscp_now();
    while (std::chrono::duration_cast<std::chrono::milliseconds>(clock::now() - t0)
               .count() < 200) {
        benchmark::DoNotOptimize(rdtscp_now());
    }
    const uint64_t c1 = rdtscp_now();
    const auto t1 = clock::now();
    const double ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    return static_cast<double>(c1 - c0) / ns;
}

// Median cycle overhead of a back-to-back rdtscp pair (subtracted off).
uint64_t measure_rdtscp_overhead() {
    constexpr int kN = 4096;
    std::vector<uint64_t> v(kN);
    for (int i = 0; i < kN; ++i) {
        const uint64_t a = rdtscp_now();
        const uint64_t b = rdtscp_now();
        v[i] = b - a;
    }
    std::sort(v.begin(), v.end());
    return v[kN / 2];
}

struct Stats {
    double p50, p95, p99, p999, mean, min;
    std::size_t n;
};

double pct(const std::vector<double>& sorted, double q) {
    if (sorted.empty()) return 0.0;
    const std::size_t i = static_cast<std::size_t>(q * (sorted.size() - 1));
    return sorted[i];
}

Stats summarize(std::vector<double>& ns) {
    std::sort(ns.begin(), ns.end());
    double sum = 0.0;
    for (double x : ns) sum += x;
    Stats s;
    s.n = ns.size();
    s.mean = ns.empty() ? 0.0 : sum / ns.size();
    s.min = ns.empty() ? 0.0 : ns.front();
    s.p50 = pct(ns, 0.50);
    s.p95 = pct(ns, 0.95);
    s.p99 = pct(ns, 0.99);
    s.p999 = pct(ns, 0.999);
    return s;
}

double cpn;          // cycles per ns (global, set in main)
double overhead_ns;  // rdtscp pair overhead in ns

double to_ns(uint64_t cycles) {
    double v = static_cast<double>(cycles) / cpn - overhead_ns;
    return v > 0.0 ? v : 0.0;
}

Stats latency_add_limit(std::size_t n) {
    std::vector<double> lat;
    lat.reserve(n);
    OrderBook book;
    std::size_t since_reset = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const int64_t px = kBase + static_cast<int64_t>(i % 1024);
        const uint64_t a = rdtscp_now();
        book.add_limit_order(Side::BID, px, 1);
        const uint64_t b = rdtscp_now();
        lat.push_back(to_ns(b - a));
        if (++since_reset == kResetEvery) { book = OrderBook(); since_reset = 0; }
    }
    return summarize(lat);
}

Stats latency_cancel(std::size_t n) {
    std::vector<double> lat;
    lat.reserve(n);
    std::size_t produced = 0;
    while (produced < n) {
        OrderBook book;
        std::vector<uint64_t> ids;
        const std::size_t batch = std::min(kResetEvery, n - produced);
        ids.reserve(batch);
        for (std::size_t k = 0; k < batch; ++k)
            ids.push_back(book.add_limit_order(
                Side::BID, kBase + static_cast<int64_t>(k % 1024), 1));
        for (uint64_t id : ids) {
            const uint64_t a = rdtscp_now();
            book.cancel_order(id);
            const uint64_t b = rdtscp_now();
            lat.push_back(to_ns(b - a));
        }
        produced += batch;
    }
    return summarize(lat);
}

Stats latency_market(int64_t depth, std::size_t samples) {
    std::vector<double> lat;
    lat.reserve(samples);
    OrderBook book;
    for (std::size_t s = 0; s < samples; ++s) {
        for (int64_t k = 0; k < depth; ++k)  // rebuild the level (untimed)
            book.add_limit_order(Side::ASK, kBase, 1);
        const uint64_t a = rdtscp_now();
        book.add_market_order(Side::BID, static_cast<uint64_t>(depth));
        const uint64_t b = rdtscp_now();
        lat.push_back(to_ns(b - a));
    }
    return summarize(lat);
}

void print_row(const char* name, const Stats& s) {
    std::printf("%-28s %8.1f %8.1f %8.1f %9.1f %8.1f\n", name, s.p50, s.p95,
                s.p99, s.p999, s.mean);
}

void run_latency_table() {
    std::printf("\n");
    std::printf("============================================================================\n");
    std::printf(" Per-operation latency (nanoseconds), TSC-measured\n");
    std::printf("============================================================================\n");
    std::printf("%-28s %8s %8s %8s %9s %8s\n", "operation", "p50", "p95", "p99",
                "p99.9", "mean");
    std::printf("----------------------------------------------------------------------------\n");

    print_row("add_limit_order", latency_add_limit(1000000));
    print_row("cancel_order", latency_cancel(1000000));
    char label[64];
    for (int64_t depth : {1, 10, 100, 1000}) {
        std::snprintf(label, sizeof(label), "add_market_order depth=%lld",
                      static_cast<long long>(depth));
        print_row(label, latency_market(depth, 20000));
    }
    std::printf("----------------------------------------------------------------------------\n");

    const double add_p50 = latency_add_limit(200000).p50;
    if (add_p50 > 0.0) {
        std::printf(" add_limit_order throughput (1 / p50): %.2f million orders/sec\n",
                    1000.0 / add_p50);
    }
    std::printf("============================================================================\n");
}

}  // namespace

int main(int argc, char** argv) {
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;

    cpn = calibrate_cycles_per_ns();
    overhead_ns = static_cast<double>(measure_rdtscp_overhead()) / cpn;
    std::printf("TSC calibration: %.4f cycles/ns, rdtscp overhead ~%.1f ns\n", cpn,
                overhead_ns);

    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();

    run_latency_table();
    return 0;
}
