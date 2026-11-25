// v1.4 review M4/M5: the tier-1 hold state machine of JournalReplay (tierGate.h) with a fake Replay, no GPU.
#include "graphics/host_gpu/renderer/pipeline/tierGate.h"

#include <atomic>
#include <cstdio>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

using namespace Libs::Graphics;

static int g_failed = 0;
#define CHECK(cond)                                                       \
	do {                                                                  \
		if (!(cond)) {                                                    \
			std::printf("FAILED %s:%d: %s\n", __FILE__, __LINE__, #cond); \
			g_failed++;                                                   \
		}                                                                 \
	} while (0)

// A model of JournalReplay::Worker: `order` = tier 1 (first_level records) then tier 2; replay(item) false = set aside (deferred).
struct Sim {
	TierGate              gate;
	std::atomic<size_t>   next {0};
	std::atomic<bool>     stop {false};
	std::atomic<int>      releases {0};
	std::atomic<int>      bg_started {0};
	std::atomic<int>      bg_started_before_tier1 {0};
	std::atomic<int>      built {0};
	std::mutex            m;
	std::vector<size_t>   deferred;
	size_t                total = 0, first_level = 0;
	std::function<bool(size_t)> replay;

	void Worker() {
		while (!stop.load()) {
			const auto index = next.fetch_add(1);
			if (index >= total) break;
			if (index >= first_level) { // BgLoop
				if (!gate.WaitTier1(stop, [] { std::this_thread::sleep_for(std::chrono::microseconds(50)); })) break;
				bg_started++;
				if (!gate.Tier1Settled()) bg_started_before_tier1++;
				built++;
				continue;
			}
			const bool ok = replay(index);
			if (!ok) {
				std::scoped_lock lock(m);
				deferred.push_back(index);
			} else {
				built++;
			}
			if (gate.Settle(!ok)) releases++;
		}
	}
	void Run(size_t tier1, size_t tier2, unsigned workers) {
		first_level = tier1;
		total = tier1 + tier2;
		gate.Reset(tier1);
		std::vector<std::thread> t;
		for (unsigned i = 0; i < workers; i++) t.emplace_back([this] { Worker(); });
		for (auto& th: t) th.join();
	}
};

int main() {
	{ // every record builds: one release, the background part starts after tier 1
		Sim s; s.replay = [](size_t) { return true; };
		s.Run(20, 30, 4);
		CHECK(s.releases == 1 && s.bg_started == 30 && s.bg_started_before_tier1 == 0 && s.built == 50 && s.deferred.empty());
		CHECK(s.gate.Tier1Settled() && s.gate.deferred == 0);
	}
	{ // M4: some tier-1 records are set aside (modules not published): the hold still releases and the background part still starts
		Sim s; s.replay = [](size_t i) { return i % 3 != 0; };
		s.Run(21, 10, 3);
		CHECK(s.releases == 1);
		CHECK(s.bg_started == 10 && s.bg_started_before_tier1 == 0);
		CHECK(s.deferred.size() == 7 && s.gate.deferred == 7 && s.built == 14 + 10);
	}
	{ // M4 worst case: every tier-1 record deferred (EARLY=2 with the shader replay still running): no spin, release once
		Sim s; s.replay = [](size_t) { return false; };
		s.Run(8, 5, 2);
		CHECK(s.releases == 1 && s.bg_started == 5 && s.deferred.size() == 8);
	}
	{ // tier 1 empty (first_level == 0): nothing to settle, the background part starts at once, no release from the workers
		Sim s; s.replay = [](size_t) { return true; };
		s.Run(0, 6, 3);
		CHECK(s.releases == 0 && s.bg_started == 6 && s.gate.Tier1Settled());
	}
	{ // stop ordering: a stopped replay leaves the background wait without starting it
		Sim s; s.replay = [](size_t) { return true; };
		s.gate.Reset(5); s.stop = true;
		CHECK(!s.gate.WaitTier1(s.stop, [] {}));
		TierGate g2; g2.Reset(2);
		std::atomic<bool> go {false};
		int n = 0;
		CHECK(g2.WaitTier1(go, [&] { if (++n == 2) { g2.Settle(false); g2.Settle(true); } }) && n == 2);
		CHECK(g2.deferred == 1);
	}
	{ // Settle reports the last one exactly once, even past the count (a late duplicate never re-releases)
		TierGate g; g.Reset(3);
		CHECK(!g.Settle(false) && !g.Settle(true) && g.Settle(false) && !g.Settle(false));
	}
	{ // BgPoll scene entry: a new scene in the database promotes once; unknown scenes are remembered without promoting; same scene is idle
		CHECK(!SceneEntered("", "", -1).remember);
		auto r = SceneEntered("", "hub", 3); CHECK(r.remember && r.promote);
		r = SceneEntered("hub", "hub", 3); CHECK(!r.remember && !r.promote);
		r = SceneEntered("hub", "cutscene_x", -1); CHECK(r.remember && !r.promote);
		r = SceneEntered("hub", "", 3); CHECK(!r.remember && !r.promote);
	}
	std::printf("tier_gate_tests: %s\n", g_failed == 0 ? "all passed" : "FAILED");
	return g_failed == 0 ? 0 : 1;
}
