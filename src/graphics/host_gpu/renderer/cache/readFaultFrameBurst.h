// nps5: adaptive per-page, per-guest-frame limit for served read faults.
//
// A load of a GPU-owned page can be completed inside the fault handler (the bytes are published
// already) without making the page readable, so the next load faults again. A game that samples a
// GPU simulation from many worker threads (hoover_beach fluid) issues thousands of such loads per
// frame: the signal round trips dominate. This counter bounds the loads served per page and frame:
// past the limit the caller waits once for the page's publication (the page is then readable and
// stops faulting). Gameplay pages (a handful of served loads per frame) never reach the limit.
//
// Lock-free, fixed size, approximate on hash collisions (a collision only resets a count, which
// at worst serves a few more loads or waits slightly early).
#pragma once

#include <atomic>
#include <cstdint>

namespace Libs::Graphics::ReadFaultFrameBurst {

class PageFrameCounter {
public:
	static constexpr uint32_t kSlots = 4096; // power of two

	// Counts one load of `page` (any granularity-aligned address) in guest frame `frame`;
	// returns how many loads of that page this frame have been counted, including this one.
	uint32_t Bump(uint64_t page, uint32_t frame) {
		const uint64_t tag  = (page >> 12) & 0xffffffffull;
		const uint64_t h    = (page >> 12) * 0x9E3779B97F4A7C15ull;
		auto&          slot = m_slots[(h >> 40) & (kSlots - 1)];
		const uint64_t key  = (tag << 32) | (static_cast<uint64_t>(frame & 0xffffu) << 16);
		auto           cur  = slot.load(std::memory_order_relaxed);
		for (;;) {
			uint64_t next;
			uint32_t count;
			if ((cur & ~0xffffull) == key) {
				count = static_cast<uint32_t>(cur & 0xffffu);
				if (count == 0xffffu) {
					return count;
				}
				count++;
			} else {
				count = 1;
			}
			next = key | count;
			if (slot.compare_exchange_weak(cur, next, std::memory_order_relaxed)) {
				return count;
			}
		}
	}

private:
	std::atomic<uint64_t> m_slots[kSlots] {};
};

// True when this load must wait for the publication instead of being served.
// limit == 0: the frame limit is off.
inline bool ShouldWait(PageFrameCounter& counter, uint32_t limit, uint64_t page, uint32_t frame) {
	return limit != 0 && counter.Bump(page, frame) > limit;
}

} // namespace Libs::Graphics::ReadFaultFrameBurst
