// nps5 cpusurvey: unit test of the KYTY_CP_LEAD_MAX_MS bookkeeping (cpLeadBound.h). Standalone:
//   clang++ -std=c++20 -O2 -Isrc tests/CpLeadBoundTests.cpp -o /tmp/cp_lead_bound_tests && /tmp/cp_lead_bound_tests
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
	std::printf("CpLeadBoundTests: %d checks, %d failed\n", g_checks, g_failed);
	return g_failed == 0 ? 0 : 1;
}
