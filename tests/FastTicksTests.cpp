// nps5 cpcommit: Common::FastTicks (common/fastTicks.h), the tick source of the command
// processor's per-draw statistics (CpGaps, CommitStats, BufferLookupStats). Checks what those
// users rely on: ticks never go backwards on a thread, threads agree about the clock, the rate against the
// steady clock is stable (they calibrate per report), intervals measured in ticks agree with the
// steady clock, and a read is cheap. Standalone: clang++ -std=c++20 -O2 -Isrc tests/FastTicksTests.cpp
#include "common/fastTicks.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <thread>
#include <vector>

namespace {

int g_failures = 0;
uint64_t g_checks = 0;

void Expect(bool condition, const char* what) {
	g_checks++;
	if (!condition) {
		std::printf("FastTicksTests: failed: %s\n", what);
		g_failures++;
	}
}

uint64_t SteadyNs() {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}

// Ticks per nanosecond over `millis` (the calibration CpGaps::Attach and every report make).
double Rate(int millis) {
	const auto ns0 = SteadyNs();
	const auto t0  = Common::FastTicks();
	while (SteadyNs() - ns0 < static_cast<uint64_t>(millis) * 1'000'000u) {
	}
	const auto ns1 = SteadyNs();
	const auto t1  = Common::FastTicks();
	return static_cast<double>(t1 - t0) / static_cast<double>(ns1 - ns0);
}

void TestMonotonicThread() {
	uint64_t previous = Common::FastTicks();
	uint64_t backwards = 0;
	uint64_t advanced  = 0;
	for (int i = 0; i < 20'000'000; i++) {
		const auto now = Common::FastTicks();
		backwards += now < previous ? 1u : 0u;
		advanced += now > previous ? 1u : 0u;
		previous = now;
	}
	Expect(backwards == 0, "ticks went backwards on one thread");
	Expect(advanced != 0, "ticks never advanced");
}

// One system-wide clock, not per core: a value published by one thread is not above a value another
// thread reads afterwards, up to the read's own reordering. Like RDTSC, the counter read does not
// serialise: it may execute before an earlier load of the same thread has completed, so "afterwards"
// holds only to within the out-of-order window (tens of nanoseconds; measured here, bound 2 us).
// The statistics only ever subtract two reads of one thread.
void TestSkewAcrossThreads(double ticks_per_ns) {
	std::atomic<uint64_t> latest {0};
	std::atomic<uint64_t> worst {0};
	std::atomic<uint64_t> behind {0};
	std::atomic<uint64_t> reads {0};
	std::atomic<bool>     stop {false};
	std::vector<std::thread> threads;
	for (int t = 0; t < 6; t++) {
		threads.emplace_back([&] {
			uint64_t bad = 0, most = 0, n = 0;
			while (!stop.load(std::memory_order_relaxed)) {
				const auto seen = latest.load(std::memory_order_seq_cst);
				const auto now  = Common::FastTicks();
				n++;
				if (now < seen) {
					bad++;
					most = seen - now > most ? seen - now : most;
				}
				auto expected = seen;
				while (expected < now &&
				       !latest.compare_exchange_weak(expected, now, std::memory_order_seq_cst)) {
				}
			}
			behind.fetch_add(bad);
			reads.fetch_add(n);
			auto current = worst.load();
			while (current < most && !worst.compare_exchange_weak(current, most)) {
			}
		});
	}
	std::this_thread::sleep_for(std::chrono::milliseconds(300));
	stop = true;
	for (auto& thread: threads) {
		thread.join();
	}
	const double worst_ns = static_cast<double>(worst.load()) / ticks_per_ns;
	std::printf("FastTicksTests: 6 threads, %llu reads, %llu behind another thread's earlier value, "
	            "worst %.0f ns\n",
	            static_cast<unsigned long long>(reads.load()),
	            static_cast<unsigned long long>(behind.load()), worst_ns);
	Expect(worst_ns < 2000.0, "threads disagree about the clock by 2 us or more");
}

double TestRate() {
	const double first = Rate(20);
	// 1 MHz .. 10 GHz: a generic timer (24 MHz on Apple silicon, 1 GHz on ARMv8.6) or a TSC.
	Expect(first > 1e-3 && first < 10.0, "tick rate outside 1 MHz .. 10 GHz");
	double low = first, high = first;
	for (int i = 0; i < 8; i++) {
		const double rate = Rate(20);
		low  = rate < low ? rate : low;
		high = rate > high ? rate : high;
	}
	// Each sample is quantised by one tick and one clock read at each end: far below 0.5 %.
	Expect((high - low) / first < 0.005, "tick rate not stable within 0.5 %");
	std::printf("FastTicksTests: rate %.6f ticks/ns (%.3f MHz), spread %.4f %%\n", first, first * 1e3,
	            (high - low) / first * 100.0);
	return first;
}

// A sleep measured in ticks (scaled by a rate calibrated before it) equals the steady clock's.
void TestIntervals() {
	const double rate = Rate(50);
	for (const int millis: {1, 5, 20, 100}) {
		const auto ns0 = SteadyNs();
		const auto t0  = Common::FastTicks();
		std::this_thread::sleep_for(std::chrono::milliseconds(millis));
		const auto t1  = Common::FastTicks();
		const auto ns1 = SteadyNs();
		const double ticks_ns  = static_cast<double>(t1 - t0) / rate;
		const double steady_ns = static_cast<double>(ns1 - ns0);
		// The two reads at each end are not simultaneous: allow 20 us plus 0.5 %.
		Expect(std::fabs(ticks_ns - steady_ns) < 20'000.0 + steady_ns * 0.005,
		       "an interval in ticks differs from the steady clock");
	}
}

// Sums of short intervals (what the statistics add up: scopes of 50 ns .. 5 us, each quantised by
// the tick) are unbiased: the sum of N consecutive scopes equals the enclosing interval exactly.
void TestConsecutiveScopesAddUp() {
	uint64_t sum   = 0;
	const auto begin = Common::FastTicks();
	uint64_t last  = begin;
	volatile uint64_t sink = 0;
	for (int i = 0; i < 2'000'000; i++) {
		for (int k = 0; k < (i & 31); k++) {
			sink = sink + static_cast<uint64_t>(k);
		}
		const auto now = Common::FastTicks();
		sum += now - last;
		last = now;
	}
	Expect(sum == last - begin, "consecutive scopes do not add up to the enclosing interval");
}

void Bench() {
	constexpr int N = 20'000'000;
	uint64_t acc = 0;
	const auto a = SteadyNs();
	for (int i = 0; i < N; i++) {
		acc += Common::FastTicks();
	}
	const auto b = SteadyNs();
	for (int i = 0; i < N; i++) {
		acc += SteadyNs();
	}
	const auto c = SteadyNs();
	const double fast   = static_cast<double>(b - a) / N;
	const double steady = static_cast<double>(c - b) / N;
	std::printf("FastTicksTests: bench FastTicks %.2f ns/read, steady_clock %.2f ns/read (%.0fx), "
	            "14 reads/draw x 5000 draws = %.3f -> %.3f ms/frame [%llu]\n",
	            fast, steady, steady / fast, steady * 14 * 5000 * 1e-6, fast * 14 * 5000 * 1e-6,
	            static_cast<unsigned long long>(acc & 1u));
#if defined(__aarch64__) || defined(__x86_64__) || defined(_M_X64)
	Expect(fast < steady, "FastTicks is not cheaper than the steady clock");
	Expect(fast < 10.0, "FastTicks costs 10 ns or more per read");
#endif
}

} // namespace

int main() {
	TestMonotonicThread();
	const double rate = TestRate();
	TestSkewAcrossThreads(rate);
	TestIntervals();
	TestConsecutiveScopesAddUp();
	Bench();
	std::printf("FastTicksTests: %llu checks, %d failures\n",
	            static_cast<unsigned long long>(g_checks), g_failures);
	return g_failures == 0 ? 0 : 1;
}
