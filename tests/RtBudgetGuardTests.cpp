#include "graphics/host_gpu/renderer/pipeline/rtSession.h"

#include <cstdio>
#include <cstdlib>

static int g_checks = 0;
#define CHECK(c)                                                                                   \
	do {                                                                                           \
		++g_checks;                                                                                \
		if (!(c)) {                                                                                \
			std::fprintf(stderr, "RtBudgetGuardTests: FAIL line %d: %s\n", __LINE__, #c);          \
			std::abort();                                                                          \
		}                                                                                          \
	} while (0)

using Libs::Graphics::RtSession::BudgetGuard;
using Libs::Graphics::RtSession::LoadHold;

int main() {
	{ // a valid BVH: the counter never moves, nothing is ever skipped
		BudgetGuard g;
		for (uint64_t t = 0; t < 100000; t += 16) CHECK(g.Allow(0, t));
		CHECK(g.Trips() == 0);
	}
	{ // lanes exhausted before the first look do not trip
		BudgetGuard g;
		CHECK(g.Allow(500000, 0));
		CHECK(g.Allow(500000, 16));
		CHECK(g.Trips() == 0);
	}
	{ // a rise below the threshold (a few stray lanes) does not trip
		BudgetGuard g;
		CHECK(g.Allow(0, 0));
		CHECK(g.Allow(1023, 16));
		CHECK(g.Allow(2046, 32));
		CHECK(g.Trips() == 0);
	}
	{ // the level-entry case: first dispatch runs, counter rises while it runs, the rest is held
		BudgetGuard g;
		CHECK(g.Allow(0, 1000));        // dispatch 1 of the pair (bad BVH) runs
		CHECK(!g.Allow(40000, 1001));   // dispatch 2: lanes of 1 already exhausted -> skipped
		CHECK(g.Trips() == 1 && g.CurrentHoldMs() == 500);
		CHECK(!g.Allow(131072, 1200));  // still counting: no new trip, hold not extended
		CHECK(!g.Allow(262144, 1499));
		CHECK(g.Trips() == 1);
		CHECK(!g.Allow(262144, 1500));  // 1 ms before the hold ends
		CHECK(g.Allow(262144, 1501));   // hold over, counter quiet: probe runs
		CHECK(g.Allow(262144, 1516));   // BVH valid now: normal again
		CHECK(g.Trips() == 1);
	}
	{ // BVH still invalid at the probe: the hold doubles, up to the cap; calm resets it
		BudgetGuard g;
		CHECK(g.Allow(0, 0));
		CHECK(!g.Allow(5000, 10));
		uint64_t t = 10 + 500;
		uint32_t c = 5000;
		uint64_t expect = 1000;
		for (int i = 0; i < 6; i++) {
			CHECK(g.Allow(c, t));          // probe runs
			c += 131072;
			CHECK(!g.Allow(c, t + 5));     // and exhausts again
			CHECK(g.CurrentHoldMs() == (expect > 8000 ? 8000 : expect));
			t += 5 + g.CurrentHoldMs();
			expect *= 2;
		}
		CHECK(g.CurrentHoldMs() == 8000);
		CHECK(g.Allow(c, t));
		t += 8000 + 30000 + 1;             // a long calm stretch
		CHECK(g.Allow(c, t));
		CHECK(!g.Allow(c + 2000, t + 16)); // a new episode starts at the base hold
		CHECK(g.CurrentHoldMs() == 500);
	}
	{ // counter wrap
		BudgetGuard g;
		CHECK(g.Allow(0xFFFFFF00u, 0));
		CHECK(!g.Allow(0x00000400u, 16)); // rose by 0x500 across the wrap
		CHECK(g.Trips() == 1);
	}
	const uint64_t MB = 1ull << 20;
	{ // gameplay: slow trickle of reads (8 MB/s) never holds
		LoadHold h;
		uint64_t b = 0;
		for (uint64_t t = 0; t < 60000; t += 16) {
			b += 8 * MB * 16 / 1000;
			CHECK(!h.Hold(b, 0, t));
		}
		CHECK(h.Holds() == 0);
	}
	{ // level load: 200 MB/s for 3 s, then quiet: held from the first sample to quiet_ms after the last
		LoadHold h;
		uint64_t b = 0, first_hold = 0, last_hold = 0;
		for (uint64_t t = 0; t < 10000; t += 16) {
			if (t >= 1000 && t < 4000) b += 200 * MB * 16 / 1000;
			if (h.Hold(b, 0, t)) {
				if (first_hold == 0) first_hold = t;
				last_hold = t;
			}
		}
		CHECK(first_hold >= 1000 && first_hold <= 1000 + 2 * (250 + 16)); // first full sample window
		CHECK(last_hold >= 4000 + 1500 - 250 - 32 && last_hold <= 4000 + 1500 + 250 + 16);
		CHECK(h.Holds() == 1);
	}
	{ // an endless stream of loads: one hold ends at max_hold_ms and no new one starts without a pause
		LoadHold h;
		uint64_t b = 0, held_ms = 0;
		for (uint64_t t = 0; t < 120000; t += 16) {
			b += 200 * MB * 16 / 1000;
			if (h.Hold(b, 0, t)) held_ms += 16;
		}
		CHECK(held_ms >= 29000 && held_ms <= 30100);
		CHECK(h.Holds() == 1);
	}
	{ // phase changes count only when asked for
		LoadHold off;
		CHECK(!off.Hold(0, 1, 0));
		CHECK(!off.Hold(0, 2, 16));
		LoadHold on;
		on.use_phase = true;
		CHECK(!on.Hold(0, 1, 0));
		CHECK(on.Hold(0, 2, 16));
		CHECK(on.Hold(0, 2, 1500));
		CHECK(!on.Hold(0, 2, 1516));
	}
	{ // a counter that goes backwards (runtime restart of the counter) is not a sign
		LoadHold h;
		CHECK(!h.Hold(500 * MB, 0, 0));
		CHECK(!h.Hold(0, 0, 300));
		CHECK(h.Holds() == 0);
	}
	std::printf("RtBudgetGuardTests: %d checks passed\n", g_checks);
	return 0;
}
