#ifndef EMULATOR_SRC_KERNEL_HOSTEDRANGES_H_
#define EMULATOR_SRC_KERNEL_HOSTEDRANGES_H_

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <map>
#include <utility>
#include <vector>

// nps5 hosted memory: the map of the guest ranges the host registered (kernel/memory.cpp Hosted)
// and its two writers, header only so tests/HostedRangesTests.cpp can drive them without the
// kernel. Readers hold the shared guard of kernel/hostedGuard.h; the writers below take the
// exclusive guard themselves (`Exclusive` is default constructible and holds it for its lifetime).
namespace Common::HostedRanges {

struct Range {
	uint64_t       size  = 0;
	uint32_t       flags = 0;
	const uint8_t* alias = nullptr; // host view of the range's first byte, or null
};

using Map    = std::map<uint64_t, Range>;                // by guest base, never overlapping
using Pieces = std::vector<std::pair<uint64_t, uint64_t>>; // (guest address, bytes)

// Takes [base, base + size) out of `ranges`; a range that sticks out keeps its outer parts.
// `removed` gets the parts of registered ranges that were inside. Caller holds the exclusive guard.
// The outer parts of a cut range lose their alias: an alias is one host mapping of the whole range
// and cannot be split, so the host frees it with the cut and registers the parts again with new
// aliases (hybrid/kytyGpuApi.h unregister_memory). Until then their bytes are read through the
// guest view. Keeping `alias + offset` here left every reader of the outer parts on freed memory
// between the host's free and its next register_memory.
inline void Cut(Map& ranges, uint64_t base, uint64_t size, Pieces& removed) {
	const auto end = base + size;
	auto       it  = ranges.upper_bound(base);
	if (it != ranges.begin()) {
		--it;
	}
	while (it != ranges.end() && it->first < end) {
		const auto start = it->first;
		const auto range = it->second;
		const auto stop  = start + range.size;
		if (stop <= base) {
			++it;
			continue;
		}
		it = ranges.erase(it);
		if (start < base) {
			ranges[start] = {base - start, range.flags, nullptr};
		}
		if (stop > end) {
			ranges[end] = {stop - end, range.flags, nullptr};
		}
		removed.emplace_back(std::max(start, base), std::min(stop, end) - std::max(start, base));
		it = ranges.lower_bound(std::min(stop, end));
	}
}

// unregister_memory: false when nothing in [base, base + size) was registered.
template <typename Exclusive>
inline bool Unregister(Map& ranges, std::atomic<uint64_t>& generation, uint64_t base, uint64_t size, Pieces& removed) {
	const Exclusive lock;
	Cut(ranges, base, size, removed);
	if (!removed.empty()) {
		generation.fetch_add(1, std::memory_order_relaxed);
	}
	return !removed.empty();
}

// register_memory: [base, base + size) becomes one range; what was registered there before is in
// `removed`. One step under one guard: a reader sees the old ranges or the new one, never the range
// missing (the host registers the outer parts of a cut range again while the GPU threads read them).
template <typename Exclusive>
inline void Register(Map& ranges, std::atomic<uint64_t>& generation, uint64_t base, uint64_t size, uint32_t flags,
                     const uint8_t* alias, Pieces& removed) {
	const Exclusive lock;
	Cut(ranges, base, size, removed);
	ranges[base] = {size, flags, alias};
	generation.fetch_add(1, std::memory_order_relaxed);
}

} // namespace Common::HostedRanges

#endif // EMULATOR_SRC_KERNEL_HOSTEDRANGES_H_
