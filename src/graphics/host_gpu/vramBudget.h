#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_VRAMBUDGET_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_VRAMBUDGET_H_

#include <algorithm>
#include <array>
#include <cstdint>

// KYTY_VRAM_GC_BUDGET=1 (default 0: off; live switch): the caches collect against the device memory
// budget as it is now, frame-aged, instead of thresholds fixed at startup and submission ages.
//
// The stock collectors compute their marks once, from the budget at startup, as fractions of
// min(budget, 8 GiB): on a 10 GiB budget (a 12 GB card) the image collector's trigger is 1 GiB and
// its critical mark 8.4 GiB, the buffer collector's critical mark 8.4 GiB. The image collector frees
// images unused for 16-160 completed submissions (Astro Bot: ~180 per frame) and the buffer collector
// buffers unused for 80-160 (it cannot see uses through BDA pointers) with a GPU wait for the dirty
// ones, so on cards below ~16 GB both run all the time, re-create and re-upload what every frame
// uses, and the startup budget misses the budget other programs leave later.
//
// With the switch on, once per frame the planning budget B (GraphicContext::GetTotalMemoryBudget: the
// driver's VK_EXT_memory_budget budget minus a reserve) is read again (the collectors run on the
// command-processor thread after every completed submission, so not per collection) and:
//   - images: above ImageTrigger(B) the collector frees, once per frame and oldest first, clean
//     images not used for AgeFrames() frames (a quarter of that, at least 4, above Critical(B)),
//     until usage is back at the trigger; never a GPU-modified, bound or target image, never a
//     download (TextureCache::RetireUnusedImages); the native image pool keeps nothing above it;
//   - buffers: the stock buffer collector runs only above B (it is the last resort: its LRU cannot
//     see BDA reads), with its downloads only above Critical(B) + B / 16.
// KYTY_VRAM_PRESSURE_FRAMES (if set) is the age; otherwise 30 frames.
namespace Libs::Graphics::VramBudget {

[[nodiscard]] bool GcEnabled() noexcept;

constexpr uint64_t DefaultAgeFrames = 30;

[[nodiscard]] constexpr uint64_t ImageTrigger(uint64_t budget) noexcept {
	return budget / 4 * 3;
}

[[nodiscard]] constexpr uint64_t Critical(uint64_t budget) noexcept {
	return budget - budget / 16;
}

[[nodiscard]] constexpr uint64_t BufferTrigger(uint64_t budget) noexcept {
	return budget;
}

[[nodiscard]] constexpr uint64_t BufferCritical(uint64_t budget) noexcept {
	return budget + budget / 16;
}

// KYTY_GC_SIDE_SETTLE_LAZY (nps5 slowlvl; BufferCache::RunGarbageCollector): a buffer collection
// pass must first settle the pending side readbacks, which waits for the GPU when one of them is
// not complete yet (the eager copies end every recording, so that is nearly every pass: the
// command processor then runs in step with the GPU). Below the critical mark such a pass is put
// off: it runs when settling does not wait, or after BufferSettleMaxDeferred passes in a row were
// put off (then with the wait, as before). A pass that ran late may retire as many buffers as the
// passes it replaces would have (capped), so the retirement rate does not fall. At or above the
// critical mark, or with the switch off: every pass runs, as before.
inline constexpr uint64_t BufferSettleMaxDeferred = 64;
inline constexpr uint64_t BufferPassLimit         = 32; // buffers per pass below the critical mark
inline constexpr uint64_t BufferPassLimitCritical = 64;
inline constexpr uint64_t BufferPassLimitMaxScale = 16;

struct BufferPass {
	bool     run      = true;
	uint64_t limit    = BufferPassLimit; // buffers this pass may retire
	uint64_t deferred = 0;               // passes put off in a row, after this one
};

[[nodiscard]] constexpr BufferPass PlanBufferPass(bool lazy, bool critical, bool settle_waits,
                                                  uint64_t deferred) noexcept {
	if (critical) {
		return {true, BufferPassLimitCritical, 0};
	}
	if (!lazy) {
		return {true, BufferPassLimit, 0};
	}
	if (settle_waits && deferred < BufferSettleMaxDeferred) {
		return {false, 0, deferred + 1};
	}
	return {true, BufferPassLimit * std::min<uint64_t>(deferred + 1, BufferPassLimitMaxScale), 0};
}

struct ImagePlan {
	bool     retire     = false;
	uint64_t age_frames = 0; // free clean images unused for more than this many frames
	uint64_t bytes      = 0; // until this many bytes are freed
};

// What the image collector does this frame at `usage` of planning budget `budget`.
[[nodiscard]] constexpr ImagePlan PlanImages(uint64_t budget, uint64_t usage, uint64_t age_frames) noexcept {
	const auto trigger = ImageTrigger(budget);
	if (budget == 0 || usage <= trigger) {
		return {};
	}
	const auto age = age_frames != 0 ? age_frames : DefaultAgeFrames;
	return {true, usage >= Critical(budget) ? std::max<uint64_t>(age / 4, 4) : age, usage - trigger};
}

// KYTY_TEXTURE_GC_REUSE=1 (default 0: off; live switch): reuse-aware budget collection. Above the
// critical mark PlanImages frees images unused for age/4 (7) frames, but a game that streams more
// than the budget re-uses images every 2-10 frames: each is freed, created again and re-uploaded
// (shuriken_ninja_castle at 6144 MiB: 63 GiB freed in 140 s, ~10 fps). ChurnTracker remembers the
// guest address and frame of every image the budget collector freed; an image created again at
// that address within ChurnWindow frames is a churn hit and raises a floor under the collector's
// age (x2, from MinFloor up to MaxFloor); the floor decays by half every DecayFrames frames
// without a hit. The floor is ignored above HardLimit(B) (the budget stays a budget), and one
// frame frees at most BurstBytes (no multi-GiB eviction stalls). Never-reused images are the ones
// the floor does not protect: they stay oldest-first LRU candidates after the floor's age.
constexpr uint64_t ChurnWindow = 120;
constexpr uint64_t MinFloor    = 16;
constexpr uint64_t MaxFloor    = 240;
constexpr uint64_t DecayFrames = 300;
constexpr uint64_t BurstBytes  = uint64_t {512} << 20u;

// Above this usage the floor no longer protects images.
[[nodiscard]] constexpr uint64_t HardLimit(uint64_t budget) noexcept {
	return budget + budget / 4;
}

[[nodiscard]] constexpr uint64_t EffectiveAge(uint64_t plan_age, uint64_t floor, uint64_t budget,
                                              uint64_t usage) noexcept {
	return usage >= HardLimit(budget) ? plan_age : std::max(plan_age, floor);
}

class ChurnTracker {
public:
	static constexpr size_t Entries = 4096;

	void NoteFreed(uint64_t address, uint64_t frame) noexcept {
		m_table[Slot(address)] = {address, frame + 1};
	}
	// An image was created at `address`; true when the collector freed one there within ChurnWindow.
	bool NoteCreated(uint64_t address, uint64_t frame) noexcept {
		auto& e = m_table[Slot(address)];
		if (e.frame_plus_one == 0 || e.address != address) {
			return false;
		}
		const auto freed = e.frame_plus_one - 1;
		e                = {};
		if (frame < freed || frame - freed > ChurnWindow) {
			return false;
		}
		m_floor     = std::min(MaxFloor, std::max(MinFloor, m_floor * 2));
		m_changed   = frame;
		++m_hits;
		return true;
	}
	// Once per frame: decay the floor after DecayFrames frames without a hit.
	void Tick(uint64_t frame) noexcept {
		if (m_floor != 0 && frame >= m_changed && frame - m_changed >= DecayFrames) {
			m_floor   = m_floor / 2 < MinFloor ? 0 : m_floor / 2;
			m_changed = frame;
		}
	}
	[[nodiscard]] uint64_t Floor() const noexcept { return m_floor; }
	[[nodiscard]] uint64_t Hits() const noexcept { return m_hits; }

private:
	struct Entry {
		uint64_t address        = 0;
		uint64_t frame_plus_one = 0;
	};
	[[nodiscard]] static size_t Slot(uint64_t address) noexcept {
		return static_cast<size_t>((address >> 8u) * 0x9E3779B97F4A7C15ull >> 52u) % Entries;
	}
	std::array<Entry, Entries> m_table {};
	uint64_t                   m_floor   = 0;
	uint64_t                   m_changed = 0;
	uint64_t                   m_hits    = 0;
};

[[nodiscard]] bool ReuseEnabled() noexcept;

} // namespace Libs::Graphics::VramBudget

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_VRAMBUDGET_H_ */
