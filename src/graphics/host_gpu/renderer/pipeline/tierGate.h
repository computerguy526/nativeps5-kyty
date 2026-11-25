#ifndef GRAPHICS_HOST_GPU_RENDERER_PIPELINE_TIERGATE_H_
#define GRAPHICS_HOST_GPU_RENDERER_PIPELINE_TIERGATE_H_

#include <atomic>
#include <chrono>
#include <cstdint>
#include <string>
#include <thread>

// KYTY_SHADER_PREPARE=tier / scene: the state machine of the launch hold, pulled out of JournalReplay (pipelineCache.cpp) so it can be
// tested with a fake Replay (v1.4 review M4/M5, tests/TierGateTests.cpp). Tier 1 is the first `tier1_items` records of the replay
// order; the hold ends with the LAST of them being SETTLED, and only then do the background workers start on tier 2.
//   Settled = built, or set aside (modules not published yet, KYTY_PIPELINE_JOURNAL_EARLY). v1.4 counted only the built ones, so one
//   deferred tier-1 record kept the hold and the background part waiting until the 900 s launch-hold maximum. A set-aside record is
//   still retried by the deferred passes afterwards, behind the game.

namespace Libs::Graphics {

struct TierGate {
	std::atomic<uint64_t> settled {0};
	std::atomic<uint64_t> deferred {0}; // how many of the settled ones were set aside (for the log line)
	uint64_t              tier1_items = 0;

	void Reset(uint64_t tier1) {
		settled.store(0, std::memory_order_relaxed);
		deferred.store(0, std::memory_order_relaxed);
		tier1_items = tier1;
	}
	// One tier-1 record is settled. True for exactly the call that settles the last one: that caller releases the launch hold.
	bool Settle(bool was_deferred) {
		if (was_deferred) deferred.fetch_add(1, std::memory_order_relaxed);
		return settled.fetch_add(1, std::memory_order_acq_rel) + 1 == tier1_items;
	}
	bool Tier1Settled() const { return settled.load(std::memory_order_acquire) >= tier1_items; }

	// BgLoop's first step: wait for tier 1. Returns false when the replay was stopped first.
	template <class Sleep>
	bool WaitTier1(const std::atomic<bool>& stop, Sleep&& sleep) const {
		while (!stop.load(std::memory_order_acquire) && !Tier1Settled()) sleep();
		return !stop.load(std::memory_order_acquire);
	}
};

// BgPoll: the scene the player just entered promotes its background pipelines once. `index` is the scene's row in the level database
// (-1 = not in it: remembered, nothing promoted).
struct ScenePromotion {
	bool remember = false; // update the remembered scene
	bool promote  = false; // promote its pipelines now
};
inline ScenePromotion SceneEntered(const std::string& remembered, const std::string& current, int index) {
	ScenePromotion r;
	if (current.empty() || current == remembered) return r;
	r.remember = true;
	r.promote  = index >= 0;
	return r;
}

} // namespace Libs::Graphics

#endif
