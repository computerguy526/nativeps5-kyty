#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_BDASCANWALK_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_BDASCANWALK_H_

// nps5 findbuf (KYTY_FINDBUF_FAST / KYTY_FINDBUF_VERIFY): the buffer walks of a BDA pass
// (BufferCache::SynchronizeBuffersInRange), header-only so tests/BdaScanWalkTests.cpp checks them
// without the renderer.
//
// The reference walk visits every buffer of the address-sorted map that overlaps [vaddr, end)
// (starting from the predecessor of vaddr, as the original loop does), clipped to the range. For a
// read-only BDA synchronization a visited buffer whose clipped range has no CPU-dirty page (and no
// missing tracker region) does nothing: BufferCache::RelaxedNothingToUpload returns at once.
// The dirty walk visits exactly the reference walk's buffers whose clipped range intersects a
// "dirty or undecided" run (MemoryTracker::ForEachCpuDirtyRunRelaxed), in the same order with the
// same arguments, and finds them from the runs: O(runs * log buffers) instead of O(buffers).
// Exact for maps of disjoint buffers, which BufferCache keeps (CreateBuffer joins every overlap):
// each run looks back one buffer only. With overlapping buffers it visits a subset of the
// reference's buffers (tests/BdaScanWalkTests.cpp).

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <utility>
#include <vector>

namespace Libs::Graphics::BdaScanWalk {

struct Visit {
	uint64_t key    = 0; // the map key of the buffer
	uint64_t start  = 0; // the clipped range [start, finish)
	uint64_t finish = 0;
	bool     operator==(const Visit& other) const {
		return key == other.key && start == other.start && finish == other.finish;
	}
};

// bounds(mapped value) -> std::pair<uint64_t, uint64_t> {buffer begin, buffer end}.
// func(iterator, start, finish) for every overlapping buffer, in map order.
template <typename Map, typename Bounds, typename Func>
void ForEachOverlapping(const Map& buffers, uint64_t vaddr, uint64_t end, Bounds&& bounds,
                        Func&& func) {
	auto it = buffers.upper_bound(vaddr);
	if (it != buffers.begin()) {
		--it;
	}
	for (; it != buffers.end() && it->first < end; ++it) {
		const auto [begin, finish_all] = bounds(it->second);
		const auto start               = std::max(begin, vaddr);
		const auto finish              = std::min(finish_all, end);
		if (start < finish) {
			func(it, start, finish);
		}
	}
}

// for_each_run(callback(run_begin, run_end)) enumerates ascending, disjoint runs inside
// [vaddr, end). Collects the visits into `out` (cleared first), in map order, then the caller
// performs them; `scratch` keeps its capacity between calls.
template <typename Map, typename Bounds, typename Runs>
void CollectDirtyVisits(const Map& buffers, uint64_t vaddr, uint64_t end, Bounds&& bounds,
                        Runs&& for_each_run, std::vector<Visit>& out) {
	out.clear();
	if (buffers.empty() || vaddr >= end) {
		return;
	}
	// The reference walk's first buffer: nothing before it is ever visited.
	auto first = buffers.upper_bound(vaddr);
	if (first != buffers.begin()) {
		--first;
	}
	if (first == buffers.end()) {
		return;
	}
	const uint64_t first_key = first->first;
	bool           sorted    = true;
	for_each_run([&](uint64_t run_begin, uint64_t run_end) {
		auto it = buffers.upper_bound(run_begin);
		if (it != buffers.begin()) {
			--it;
		}
		if (it->first < first_key) {
			it = first;
		}
		for (; it != buffers.end() && it->first < run_end; ++it) {
			const auto [begin, finish_all] = bounds(it->second);
			const auto start               = std::max(begin, vaddr);
			const auto finish              = std::min(finish_all, end);
			if (start >= finish || finish <= run_begin || start >= run_end) {
				continue;
			}
			if (!out.empty() && it->first <= out.back().key) {
				if (it->first == out.back().key) {
					continue;
				}
				sorted = false; // overlapping buffers: order fixed below
			}
			out.push_back({it->first, start, finish});
		}
	});
	if (!sorted) {
		std::sort(out.begin(), out.end(),
		          [](const Visit& a, const Visit& b) { return a.key < b.key; });
		out.erase(std::unique(out.begin(), out.end(),
		                      [](const Visit& a, const Visit& b) { return a.key == b.key; }),
		          out.end());
	}
}

} // namespace Libs::Graphics::BdaScanWalk

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_BDASCANWALK_H_
