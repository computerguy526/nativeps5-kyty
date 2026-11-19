#include "graphics/host_gpu/renderer/cache/faultClass.h"

#include <cstdio>
#include <thread>
#include <unistd.h>
#include <vector>

using namespace Libs::Graphics::FaultClass;

static int g_failed = 0;
static int g_checks = 0;
#define CHECK(cond)                                                       \
	do {                                                                  \
		g_checks++;                                                       \
		if (!(cond)) {                                                    \
			std::printf("FAILED %s:%d: %s\n", __FILE__, __LINE__, #cond); \
			g_failed++;                                                   \
		}                                                                 \
	} while (0)

int main() {
	const uint64_t P = 0x50be00000ull;
	// 1. Promotion needs distinct frames: many faults in one frame are one frame.
	{
		Table t;
		for (int i = 0; i < 100; i++) CHECK(!t.NoteRead(P, 10));
		CHECK(t.Get(P, 10) == Class::Unknown);
		CHECK(!t.NoteRead(P, 11));
		CHECK(t.NoteRead(P, 12)); // third distinct frame
		CHECK(t.Get(P, 12) == Class::Late);
		CHECK(t.NoteRead(P, 500)); // stays Late
		CHECK(t.TakeCounters().promoted == 1);
	}
	// 2. A one-shot reader (one frame), or reads spread wider than the window, never promote.
	{
		Table t;
		CHECK(!t.NoteRead(P, 1));
		CHECK(!t.NoteRead(P, 200));
		CHECK(!t.NoteRead(P, 400));
		CHECK(!t.NoteRead(P, 600));
		CHECK(t.Get(P, 600) == Class::Unknown);
		// ...but three frames inside one window do.
		CHECK(!t.NoteRead(P, 601));
		CHECK(t.NoteRead(P, 602));
	}
	// 3. Pages are independent.
	{
		Table t;
		for (uint32_t f = 1; f <= 3; f++) (void)t.NoteRead(P, f);
		CHECK(t.Get(P, 3) == Class::Late);
		CHECK(t.Get(P + 0x1000, 3) == Class::Unknown);
		CHECK(!t.NoteRead(P + 0x1000, 3));
	}
	// 4. Store conflicts: only Late pages count; demotion after 3 within the window; Exact refuses reads.
	{
		Table t;
		CHECK(!t.NoteStoreConflict(P, 1)); // unknown page
		for (uint32_t f = 1; f <= 3; f++) (void)t.NoteRead(P, f);
		CHECK(!t.NoteStoreConflict(P, 4));
		CHECK(!t.NoteStoreConflict(P, 5));
		CHECK(t.NoteStoreConflict(P, 6));
		CHECK(t.Get(P, 6) == Class::Exact);
		for (uint32_t f = 7; f < 600; f++) CHECK(!t.NoteRead(P, f));
		CHECK(!t.NoteStoreConflict(P, 100)); // not Late any more
		// Exact period over (demoted at 6, 600 frames): Unknown, earns Late again.
		CHECK(t.Get(P, 606) == Class::Unknown);
		CHECK(!t.NoteRead(P, 606));
		CHECK(!t.NoteRead(P, 607));
		CHECK(t.NoteRead(P, 608));
		// Second demotion lasts twice as long.
		for (uint32_t f = 609; f < 611; f++) CHECK(!t.NoteStoreConflict(P, f));
		CHECK(t.NoteStoreConflict(P, 611));
		CHECK(t.Get(P, 611 + 1199) == Class::Exact);
		CHECK(t.Get(P, 611 + 1200) == Class::Unknown);
		const auto c = t.TakeCounters();
		CHECK(c.promoted == 2 && c.demoted == 2 && c.conflicts == 6);
	}
	// 5. Conflicts spread wider than the window do not demote.
	{
		Table t;
		for (uint32_t f = 1; f <= 3; f++) (void)t.NoteRead(P, f);
		for (uint32_t f = 10; f < 5000; f += 200) CHECK(!t.NoteStoreConflict(P, f));
		CHECK(t.Get(P, 5000) == Class::Late);
	}
	// 6. Closed by the caller: back to Unknown, has to be read again in 3 frames.
	{
		Table t;
		for (uint32_t f = 1; f <= 3; f++) (void)t.NoteRead(P, f);
		t.NoteClosed(P);
		CHECK(t.Get(P, 4) == Class::Unknown);
		CHECK(!t.NoteRead(P, 4));
		CHECK(!t.NoteRead(P, 5));
		CHECK(t.NoteRead(P, 6));
	}
	// 7. Seeds: Late at once, listed until opened, not in the run's own profile until read.
	{
		Table t;
		t.Seed(P);
		t.Seed(P + 0x1000);
		t.Seed(P); // twice: once
		std::vector<uint64_t> pending;
		t.PendingSeeds(pending, 16);
		CHECK(pending.size() == 2);
		CHECK(t.LatePages().empty());
		t.NoteSeedOpened(P);
		t.PendingSeeds(pending, 16);
		CHECK(pending.size() == 1 && pending[0] == P + 0x1000);
		CHECK(t.LatePages().size() == 1);
		CHECK(t.NoteRead(P + 0x1000, 1)); // a seed is Late on its first fault
		// a demoted seed is no seed any more
		for (uint32_t f = 2; f <= 4; f++) (void)t.NoteStoreConflict(P + 0x1000, f);
		t.PendingSeeds(pending, 16);
		CHECK(pending.empty());
		// round robin with a small max
		Table u;
		for (uint64_t i = 0; i < 10; i++) u.Seed(P + i * 0x1000);
		std::vector<uint64_t> seen;
		for (int round = 0; round < 5; round++) {
			u.PendingSeeds(pending, 2);
			CHECK(pending.size() == 2);
			seen.insert(seen.end(), pending.begin(), pending.end());
		}
		std::sort(seen.begin(), seen.end());
		CHECK(std::unique(seen.begin(), seen.end()) == seen.end() && seen.size() == 10);
		const auto c = t.TakeCounters();
		CHECK(c.seeded == 2 && c.seed_hits == 1);
	}
	// 8. Profile round trip (runs are merged, garbage lines skipped, cap respected).
	{
		char path[] = "/tmp/faultclass_test_XXXXXX";
		const int fd = mkstemp(path);
		CHECK(fd >= 0);
		close(fd);
		std::vector<uint64_t> pages {P, P + 0x1000, P + 0x2000, P + 0x10000, P + 0x40000, P + 0x41000};
		CHECK(SaveProfile(path, pages, 0x1000));
		Table t;
		CHECK(LoadProfile(path, 0x1000, 1000, t) == 6);
		std::vector<uint64_t> pending;
		t.PendingSeeds(pending, 100);
		std::sort(pending.begin(), pending.end());
		CHECK(pending == pages);
		Table capped;
		CHECK(LoadProfile(path, 0x1000, 4, capped) == 4);
		std::FILE* f = std::fopen(path, "a");
		std::fprintf(f, "garbage\n123 5\n0 4\n%llx 99999999\n", static_cast<unsigned long long>(P));
		std::fclose(f);
		Table g;
		CHECK(LoadProfile(path, 0x1000, 1000, g) == 6); // unaligned, zero and oversized lines skipped
		Table none;
		CHECK(LoadProfile("/nonexistent/faultclass", 0x1000, 10, none) == 0);
		unlink(path);
	}
	// 9. Capacity: idle Unknown pages are dropped, Late ones kept.
	{
		Limits limits;
		limits.capacity = 64;
		Table t(limits);
		for (uint32_t f = 1; f <= 3; f++) (void)t.NoteRead(P, f);
		for (uint64_t i = 1; i <= 64; i++) (void)t.NoteRead(P + i * 0x1000, 4);
		for (uint64_t i = 100; i < 300; i++) (void)t.NoteRead(P + i * 0x1000, 1000);
		CHECK(t.Size() <= 64 + 200);
		CHECK(t.Get(P, 1000) == Class::Late);
		CHECK(t.Get(P + 0x1000, 1000) == Class::Unknown);
	}
	// 10. Threads: readers and conflicts on shared pages, no crash, counters consistent.
	{
		Table                    t;
		std::vector<std::thread> threads;
		for (int n = 0; n < 8; n++) {
			threads.emplace_back([&t, n, P] {
				for (uint32_t f = 1; f < 3000; f++) {
					const auto page = P + ((f + n) % 37) * 0x1000ull;
					if (t.NoteRead(page, f) && (f % 5) == 0) {
						(void)t.NoteStoreConflict(page, f);
					}
				}
			});
		}
		for (auto& th: threads) th.join();
		const auto c = t.TakeCounters();
		CHECK(c.promoted >= 37);
		CHECK(c.demoted <= c.promoted);
	}
	std::printf("FaultClassTests: %d checks, %d failed\n", g_checks, g_failed);
	return g_failed == 0 ? 0 : 1;
}
