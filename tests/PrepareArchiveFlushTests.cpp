#include "graphics/host_gpu/renderer/pipeline/prepareArchiveFlush.h"

#include <cstdio>
#include <cstdlib>

using namespace Libs::Graphics;

static void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "FAIL: %s\n", message);
		std::exit(1);
	}
}

static int     g_flush_calls = 0;
static uint32_t g_flush_wait  = 0;
static int64_t g_flush_result = 0;

static int64_t FakeFlush(uint32_t wait_ms) {
	++g_flush_calls;
	g_flush_wait = wait_ms;
	return g_flush_result;
}

int main() {
	int    sleeps = 0;
	double slept  = 0.0;
	const auto sleep = [&](double s) { ++sleeps; slept += s; };

	// no archive dir: nothing to wait for, whatever the gate says
	auto o = PrepareArchiveWait(false, true, FakeFlush, 1000, 10.0, sleep);
	Check(!o.archive && !o.synced && sleeps == 0 && g_flush_calls == 0, "no archive dir: no flush, no sleep");

	// gate off: the old fixed wait (period + 3 s), the export is never called
	o = PrepareArchiveWait(true, false, FakeFlush, 1000, 10.0, sleep);
	Check(o.archive && !o.synced && sleeps == 1 && slept == 13.0 && g_flush_calls == 0 && o.note.empty(), "gate off: sleeps 13 s");

	// gate on + export: called once with the wait, no sleep
	sleeps = 0; slept = 0.0; g_flush_result = 41;
	o = PrepareArchiveWait(true, true, FakeFlush, 1234, 10.0, sleep);
	Check(o.synced && o.written == 41 && sleeps == 0 && g_flush_calls == 1 && g_flush_wait == 1234, "gate on: one flush, no sleep");

	// nothing pending (0 written) is still a success
	g_flush_result = 0;
	o = PrepareArchiveWait(true, true, FakeFlush, 1000, 10.0, sleep);
	Check(o.synced && o.written == 0 && sleeps == 0, "0 written is synced");

	// gate on, symbol missing: falls back to the sleep and says why
	o = PrepareArchiveWait(true, true, nullptr, 1000, 10.0, sleep);
	Check(!o.synced && sleeps == 1 && slept == 13.0 && !o.note.empty(), "no symbol: sleeps");

	// the export fails (-1 lock timeout, -2 no segment archive): sleep as before, never lose the segment
	sleeps = 0; slept = 0.0; g_flush_result = -1;
	o = PrepareArchiveWait(true, true, FakeFlush, 1000, 5.0, sleep);
	Check(!o.synced && sleeps == 1 && slept == 8.0 && o.note.find("lock") != std::string::npos, "timeout: falls back to period + 3");
	g_flush_result = -2;
	o = PrepareArchiveWait(true, true, FakeFlush, 1000, 5.0, sleep);
	Check(!o.synced && sleeps == 2 && o.note.find("segment archive") != std::string::npos, "no segment archive: falls back");

	// M3: the ICD is opened RTLD_LOCAL, so the global lookup misses and the loaded image's handle finds the export
	static int fake_export;
	const auto miss = [](const char*) -> void* { return nullptr; };
	const auto hit  = [](const char*) -> void* { return &fake_export; };
	const char* via = nullptr;
	Check(ResolveDriverSymbol("nps5_kk_archive_flush", miss, hit, &via) == &fake_export && std::string(via) == "driver image handle",
	      "image-handle lookup finds the export the global lookup misses");
	Check(ResolveDriverSymbol("x", hit, miss, &via) == &fake_export && std::string(via) == "global namespace", "global hit wins");
	Check(ResolveDriverSymbol("x", miss, miss, &via) == nullptr && std::string(via) == "not found", "both miss: null");
	Check(ResolveDriverSymbol("x", nullptr, hit) == &fake_export && ResolveDriverSymbol("x", nullptr, nullptr) == nullptr, "null lookups are safe");
	Check(ClampFlushWaitMs(0) == 1 && ClampFlushWaitMs(60000) == 60000 && ClampFlushWaitMs(0x100000000ull) == 0xFFFFFFFFu &&
	          ClampFlushWaitMs(0xFFFFFFFFull) == 0xFFFFFFFFu, "wait ms clamps to [1, UINT32_MAX]");

	std::puts("prepare_archive_flush_tests: all passed");
	return 0;
}
