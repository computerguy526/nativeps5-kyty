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
	std::printf("RtBudgetGuardTests: %d checks passed\n", g_checks);
	return 0;
}
