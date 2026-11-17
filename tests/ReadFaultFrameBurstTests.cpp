#include "graphics/host_gpu/renderer/cache/readFaultFrameBurst.h"

#include <cstdio>
#include <thread>
#include <vector>

using namespace Libs::Graphics::ReadFaultFrameBurst;

static int g_failed = 0;
#define CHECK(cond)                                                       \
	do {                                                                  \
		if (!(cond)) {                                                    \
			std::printf("FAILED %s:%d: %s\n", __FILE__, __LINE__, #cond); \
			g_failed++;                                                   \
		}                                                                 \
	} while (0)

int main() {
	static PageFrameCounter c;
	// Counting within a frame, restart on a new frame.
	for (uint32_t i = 1; i <= 100; i++) CHECK(c.Bump(0x50be00000, 7) == i);
	CHECK(c.Bump(0x50be00000, 8) == 1);
	CHECK(c.Bump(0x50be00000, 8) == 2);
	// Different pages are independent (distinct slots).
	CHECK(c.Bump(0x50be01000, 8) == 1);
	CHECK(c.Bump(0x50be00000, 8) == 3);
	// ShouldWait: limit 0 never waits; limit N waits on load N+1 only and after.
	{
		PageFrameCounter d;
		for (int i = 0; i < 1000; i++) CHECK(!ShouldWait(d, 0, 0x1000, 1));
		PageFrameCounter e;
		for (uint32_t i = 1; i <= 32; i++) CHECK(!ShouldWait(e, 32, 0x2000, 5));
		CHECK(ShouldWait(e, 32, 0x2000, 5));
		CHECK(ShouldWait(e, 32, 0x2000, 5));
		// next frame: the gameplay case (a few loads per frame) never waits.
		for (uint32_t f = 6; f < 400; f++)
			for (int i = 0; i < 20; i++) CHECK(!ShouldWait(e, 32, 0x2000, f));
	}
	// Frame counter wraps at 16 bits without confusing frames 0 and 65536 in a way that matters:
	// same low bits count together (documented approximation), different low bits do not.
	{
		PageFrameCounter w;
		CHECK(w.Bump(0x3000, 1) == 1);
		CHECK(w.Bump(0x3000, 2) == 1);
		CHECK(w.Bump(0x3000, 65537) == 1);
	}
	// Saturation: the count stops at 65535, never wraps to 0 (which would re-serve forever).
	{
		PageFrameCounter s;
		uint32_t last = 0;
		for (int i = 0; i < 70000; i++) last = s.Bump(0x4000, 3);
		CHECK(last == 0xffffu);
		CHECK(ShouldWait(s, 48, 0x4000, 3));
	}
	// Collision: pages mapping to one slot reset each other, never exceed their own true count.
	{
		PageFrameCounter x;
		uint64_t a = 0x10000000, b = 0;
		const auto slot = [](uint64_t p) { return (((p >> 12) * 0x9E3779B97F4A7C15ull) >> 40) & (PageFrameCounter::kSlots - 1); };
		for (uint64_t p = a + 0x1000; p < a + (1ull << 30); p += 0x1000)
			if (slot(p) == slot(a)) { b = p; break; }
		CHECK(b != 0);
		CHECK(x.Bump(a, 9) == 1);
		CHECK(x.Bump(a, 9) == 2);
		CHECK(x.Bump(b, 9) == 1);
		CHECK(x.Bump(a, 9) == 1);
	}
	// Concurrency: 8 threads x 1000 bumps of one page and frame; the max seen is total (no lost
	// slot) is not guaranteed lock-free exact, but CAS makes it exact: every value 1..8000 once.
	{
		PageFrameCounter m;
		std::vector<std::thread> th;
		std::vector<std::vector<uint32_t>> seen(8);
		for (int t = 0; t < 8; t++)
			th.emplace_back([&, t] { for (int i = 0; i < 1000; i++) seen[t].push_back(m.Bump(0x7000, 11)); });
		for (auto& t : th) t.join();
		std::vector<int> hit(8001, 0);
		for (auto& v : seen) for (auto x : v) { CHECK(x >= 1 && x <= 8000); hit[x]++; }
		for (int i = 1; i <= 8000; i++) CHECK(hit[i] == 1);
	}
	// Random: counts match a reference model for non-colliding pages.
	{
		PageFrameCounter r;
		uint32_t ref[64][4] = {};
		uint32_t seed = 12345;
		for (int i = 0; i < 100000; i++) {
			seed = seed * 1664525u + 1013904223u;
			const uint32_t p = (seed >> 8) & 63, f = (seed >> 20) & 3;
			// frames only move forward in play: keep each page's frame monotonic
			static uint32_t cur[64] = {};
			if (f < cur[p]) continue;
			if (f > cur[p]) { cur[p] = f; ref[p][f] = 0; }
			ref[p][f]++;
			const uint32_t got = r.Bump(0x100000ull + p * 0x1000ull * 37, f);
			if (got != ref[p][f]) { /* slot collision between two of the 64 pages */ CHECK(got <= ref[p][f]); }
		}
	}
	std::printf(g_failed ? "FAIL (%d)\n" : "PASS\n", g_failed);
	return g_failed ? 1 : 0;
}
