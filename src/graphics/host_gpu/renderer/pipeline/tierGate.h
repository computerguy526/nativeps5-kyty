#ifndef GRAPHICS_HOST_GPU_RENDERER_PIPELINE_TIERGATE_H_
#define GRAPHICS_HOST_GPU_RENDERER_PIPELINE_TIERGATE_H_

#include <atomic>
#include <chrono>
#include <cstdint>
#include <string>
#include <thread>

// KYTY_SHADER_PREPARE=tier / scene: the state machine of the launch hold, pulled out of
// JournalReplay (pipelineCache.cpp) so it can be tested with a fake Replay (v1.4 review M4/M5,
// tests/TierGateTests.cpp). Tier 1 is the first `tier1_items` records of the replay order; the hold
// ends with the LAST of them being SETTLED, and only then do the background workers start on
// tier 2.
//   Settled = built, or (set aside, modules not published yet, KYTY_PIPELINE_JOURNAL_EARLY) retried
//   successfully, or tried in the final pass after the shader replay finished. A set-aside record
//   is NOT settled (round-2 R2-M2): under EARLY=2 every tier-1 record is set aside within
//   milliseconds, so settling at the set-aside made the hold a no-op and the game started cold.
//   v1.4 had counted only built ones, which is right, but the final pass of the deferred records
//   after the shader replay is also a settle point so the hold cannot wait for a record that never
//   builds (the 900 s launch-hold maximum is the last bound).

namespace Libs::Graphics {

struct TierGate {
	std::atomic<uint64_t> settled {0};
	std::atomic<uint64_t> deferred {
	    0}; // how many tier-1 records were set aside at the first try (for the log line)
	uint64_t tier1_items = 0;

	void Reset(uint64_t tier1) {
		settled.store(0, std::memory_order_relaxed);
		deferred.store(0, std::memory_order_relaxed);
		tier1_items = tier1;
	}
	// A tier-1 record was set aside: counted for the log, NOT settled.
	void Defer() { deferred.fetch_add(1, std::memory_order_relaxed); }
	// A tier-1 record is settled (built, or a set-aside one built later / tried in the final pass).
	// True for exactly the call that settles the last one: that caller releases the launch hold.
	bool Settle() { return settled.fetch_add(1, std::memory_order_acq_rel) + 1 == tier1_items; }
	bool Tier1Settled() const { return settled.load(std::memory_order_acquire) >= tier1_items; }

	// BgLoop's first step: wait for tier 1. Returns false when the replay was stopped first.
	template <class Sleep>
	bool WaitTier1(const std::atomic<bool>& stop, Sleep&& sleep) const {
		while (!stop.load(std::memory_order_acquire) && !Tier1Settled())
			sleep();
		return !stop.load(std::memory_order_acquire);
	}
};

// BgPoll: the scene the player just entered promotes its background pipelines once. `index` is the
// scene's row in the level database
// (-1 = not in it: remembered, nothing promoted).
struct ScenePromotion {
	bool remember = false; // update the remembered scene
	bool promote  = false; // promote its pipelines now
};
inline ScenePromotion SceneEntered(const std::string& remembered, const std::string& current,
                                   int index) {
	ScenePromotion r;
	if (current.empty() || current == remembered) return r;
	r.remember = true;
	r.promote  = index >= 0;
	return r;
}

} // namespace Libs::Graphics

#endif
