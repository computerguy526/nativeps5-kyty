// nps5 cpusurvey: unit test of the KYTY_CP_LEAD_MAX_MS bookkeeping (cpLeadBound.h). Standalone:
//   clang++ -std=c++20 -O2 -Isrc tests/CpLeadBoundTests.cpp -o /tmp/cp_lead_bound_tests && /tmp/cp_lead_bound_tests
// CMake target: cp_lead_bound_tests.
#include "graphics/guest_gpu/command_processor/cpLeadBound.h"

#include <cstdio>

using namespace Libs::Graphics::CpLeadBound;

static int g_checks = 0, g_failed = 0;
#define CHECK(c)                                                                                   \
	do {                                                                                           \
		g_checks++;                                                                                \
		if (!(c)) {                                                                                \
			g_failed++;                                                                            \
			std::printf("FAILED %s:%d %s\n", __FILE__, __LINE__, #c);                              \
		}                                                                                          \
	} while (0)

static constexpr uint64_t kMs = 1'000'000u;

int main() {
	// Parsing: only plain decimal numbers switch it on.
	CHECK(ParseMs(nullptr) == 0);
	CHECK(ParseMs("") == 0);
	CHECK(ParseMs("0") == 0);
	CHECK(ParseMs("12") == 12);
	CHECK(ParseMs("200") == 200);
	CHECK(ParseMs("5000") == 200);
	CHECK(ParseMs("-3") == 0);
	CHECK(ParseMs("+3") == 0);
	CHECK(ParseMs("12ms") == 0);
	CHECK(ParseMs("on") == 0);
	// cplreview: trailing whitespace, overflow, other junk.
	CHECK(ParseMs("16\r") == 16);
	CHECK(ParseMs("16 ") == 16);
	CHECK(ParseMs("16\n") == 16);
	CHECK(ParseMs(" 16") == 0);
	CHECK(ParseMs("16 x") == 0);
	CHECK(ParseMs("0x10") == 0);
	CHECK(ParseMs("1.5") == 0);
	CHECK(ParseMs("007") == 7);
	CHECK(ParseMs("99999999999999999999999999") == 200);
	CHECK(ParseMs("18446744073709551615") == 200);
	CHECK(ParseMs("-0") == 0);
	CHECK(ParseMs("\r") == 0);
	// The wait timeout: the bound plus 50 ms, the bound clamped as ParseMs does.
	CHECK(WaitTimeoutNs(16) == 66 * kMs);
	CHECK(WaitTimeoutNs(200) == 250 * kMs);
	CHECK(WaitTimeoutNs(5000) == 250 * kMs);
	CHECK(WaitTimeoutNs(-5) == 50 * kMs);
	static_assert(WaitTimeoutNs(1) == 51'000'000u);

	// Nothing tracked, or one recording: never a wait (the GPU would be left without work).
	{
		Tracker t;
		CHECK(t.WaitTick(100 * kMs, 10 * kMs) == 0);
		t.Submitted(5, 0);
		CHECK(t.WaitTick(100 * kMs, 10 * kMs) == 0);
	}
	// The newest recording older than the bound is the one waited for; younger ones stay queued.
	{
		Tracker t;
		t.Submitted(10, 0 * kMs);
		t.Submitted(11, 4 * kMs);
		t.Submitted(12, 9 * kMs);
		t.Submitted(13, 15 * kMs);
		CHECK(t.WaitTick(15 * kMs, 20 * kMs) == 0);  // oldest is 15 ms old
		CHECK(t.WaitTick(15 * kMs, 15 * kMs) == 0);  // exactly at the bound: no wait
		CHECK(t.WaitTick(15 * kMs, 12 * kMs) == 10); // only tick 10 is older than 12 ms
		CHECK(t.WaitTick(15 * kMs, 10 * kMs) == 11);
		CHECK(t.WaitTick(15 * kMs, 5 * kMs) == 12);
		CHECK(t.WaitTick(15 * kMs, 0) == 12); // never the last one submitted
		// Completion removes entries; a completed tick is not waited for again.
		t.Completed(11);
		CHECK(t.Size() == 2);
		CHECK(t.WaitTick(15 * kMs, 10 * kMs) == 0);
		CHECK(t.WaitTick(15 * kMs, 5 * kMs) == 12);
		t.Completed(100);
		CHECK(t.Size() == 0);
		CHECK(t.WaitTick(99 * kMs, 0) == 0);
	}
	// After the wait for the returned tick, no incomplete recording but the last is older than the bound.
	{
		Tracker  t;
		uint64_t now = 0;
		for (uint64_t tick = 1; tick <= 2000; tick++) {
			now += (tick % 7 + 1) * kMs / 2; // 0.5 .. 3.5 ms between submits
			t.Submitted(tick, now);
			const auto limit = 12 * kMs;
			if (const auto wait = t.WaitTick(now, limit); wait != 0) {
				CHECK(wait < tick);
				t.Completed(wait);
				CHECK(t.WaitTick(now, limit) == 0);
			}
			CHECK(t.Size() >= 1);
			CHECK(t.Size() <= 25); // 12 ms / 0.5 ms + the last
		}
	}
	// A full ring forgets its oldest entry instead of growing or failing.
	{
		Tracker t;
		for (uint64_t tick = 1; tick <= Tracker::kCapacity + 50; tick++) {
			t.Submitted(tick, tick);
		}
		CHECK(t.Size() == Tracker::kCapacity);
		CHECK(t.WaitTick(10'000 * kMs, 1) == Tracker::kCapacity + 49);
		t.Clear();
		CHECK(t.Size() == 0);
	}
	// A repeated tick (a flush that submitted nothing new) is harmless.
	{
		Tracker t;
		t.Submitted(7, 0);
		t.Submitted(7, 5 * kMs);
		t.Submitted(8, 6 * kMs);
		CHECK(t.WaitTick(30 * kMs, 10 * kMs) == 7);
		t.Completed(7);
		CHECK(t.Size() == 1);
	}
	// cplreview: a timed-out or failed wait is abandoned: no wait until its tick completes, so a
	// long or stuck GPU job costs one timeout, not one per submit.
	{
		Tracker t;
		for (uint64_t tick = 1; tick <= 10; tick++) {
			t.Submitted(tick, tick * kMs);
		}
		const auto stuck = t.WaitTick(100 * kMs, 16 * kMs);
		CHECK(stuck == 9);
		t.Abandon(stuck);
		CHECK(t.Held());
		CHECK(t.Size() == 1); // tick 10 stays tracked
		for (uint64_t tick = 11; tick <= 300; tick++) {
			t.Submitted(tick, tick * kMs);
			t.Completed(5); // the GPU has not reached the stuck tick
			CHECK(t.WaitTick(tick * kMs, 16 * kMs) == 0);
		}
		CHECK(t.Size() <= Tracker::kCapacity);
		t.Completed(8);
		CHECK(t.Held()); // still short of the stuck tick
		t.Completed(9);
		CHECK(!t.Held());
		CHECK(t.WaitTick(300 * kMs, 16 * kMs) == 283); // waits resume on what is still pending
		t.Abandon(0); // a defensive tick 0 still holds (until tick 1 completes)
		CHECK(t.Held());
		t.Completed(1);
		CHECK(!t.Held());
		t.Abandon(500);
		t.Clear(); // switching off releases the hold
		CHECK(!t.Held());
		CHECK(t.Size() == 0);
	}
	// A submit stamped after `now` (clock skew between reads) counts as age 0, not as ~2^64 ns.
	{
		Tracker t;
		t.Submitted(1, 50 * kMs);
		t.Submitted(2, 60 * kMs);
		CHECK(t.WaitTick(40 * kMs, 1 * kMs) == 0);
	}
	// Stress: random submit/complete/abandon interleavings keep the invariants.
	{
		Tracker  t;
		uint64_t rng = 0x9e3779b97f4a7c15ull, now = 0, gpu = 0, tick = 0;
		for (int i = 0; i < 200000; i++) {
			rng ^= rng << 13;
			rng ^= rng >> 7;
			rng ^= rng << 17;
			now += rng % (3 * kMs);
			t.Submitted(++tick, now);
			if (rng % 3 == 0 && gpu + 1 < tick) {
				gpu += 1 + (rng >> 8) % (tick - gpu - 1);
				t.Completed(gpu);
			}
			const auto wait = t.WaitTick(now, 16 * kMs);
			CHECK(wait == 0 || (wait > gpu && wait < tick));
			if (wait != 0) {
				if ((rng >> 20) % 50 == 0) {
					t.Abandon(wait);
					CHECK(t.WaitTick(now, 16 * kMs) == 0);
				} else {
					gpu = wait;
					t.Completed(wait);
				}
			}
			CHECK(t.Size() <= Tracker::kCapacity);
		}
	}
	std::printf("CpLeadBoundTests: %d checks, %d failed\n", g_checks, g_failed);
	return g_failed == 0 ? 0 : 1;
}
