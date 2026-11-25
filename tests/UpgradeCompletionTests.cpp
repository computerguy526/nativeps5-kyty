// nps5 archfix T4: end-of-replay order (flush, save, release) with fakes, no GPU.
#include "graphics/host_gpu/renderer/pipeline/upgradeCompletion.h"

#include <cstdio>
#include <cstdlib>
#include <string>

using namespace Libs::Graphics;

static void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "FAIL: %s\n", message);
		std::exit(1);
	}
}

static std::string Run(bool held, bool saves) {
	std::string log;
	CompleteUpgradeReplay(held, saves, [&] { log += "F"; }, [&] { log += "S"; }, [&] { log += "R"; });
	return log;
}

int main() {
	// gate on, upgrade / resume / first start: flush, driver cache + stamp, then the hold is released
	Check(Run(true, true) == "FSR", "held: flush, save, release in this order");
	// gate on, Warm start / stopped replay / list prepare: nothing is written, the hold is still released
	Check(Run(true, false) == "R", "held, nothing to save: release only");
	// gate off: the caller released the hold before (old order); this step saves and never releases
	Check(Run(false, true) == "FS", "not held: flush, save, no release here");
	// gate off, Warm start: no call at all (the Warm start calls none of flush / save)
	Check(Run(false, false).empty(), "not held, nothing to save: no call");
	// a failing save (fake that reports nothing) never skips the release: the release is after the save call unconditionally
	int released = 0;
	CompleteUpgradeReplay(true, true, [] {}, [] { /* save failed: the cache stays 'preparing' */ }, [&] { ++released; });
	Check(released == 1, "a failed save still releases the hold");
	std::printf("upgrade_completion_tests: all passed\n");
	return 0;
}
