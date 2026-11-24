#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_BITRUNS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_BITRUNS_H_

#include <cstddef>
#include <cstdint>

namespace Libs::Graphics {

// nps5 findbuf: the runs of set bits among bits [first, last) of a bit array read one 64-bit word
// at a time (word(index) -> uint64_t, bit i of word w is bit w * 64 + i), as func(begin, end) in
// ascending order. A run crossing a word boundary is reported per word (callers merge).
// Tested by tests/BdaScanWalkTests.cpp.
template <typename Word, typename Func>
void ForEachSetBitRun(Word&& word, size_t first, size_t last, Func&& func) {
	if (first >= last) {
		return;
	}
	for (size_t index = first / 64u; index <= (last - 1) / 64u; index++) {
		uint64_t bits = word(index);
		if (bits == 0) {
			continue;
		}
		const size_t base = index * 64u;
		if (first > base) {
			bits &= ~uint64_t {0} << (first - base);
		}
		if (last < base + 64u) {
			bits &= (uint64_t {1} << (last - base)) - 1u;
		}
		while (bits != 0) {
			const auto     lo      = static_cast<size_t>(__builtin_ctzll(bits));
			const uint64_t shifted = bits >> lo;
			const size_t   n = ~shifted == 0 ? 64u - lo : static_cast<size_t>(__builtin_ctzll(~shifted));
			func(base + lo, base + lo + n);
			bits = lo + n >= 64u ? 0 : bits & (~uint64_t {0} << (lo + n));
		}
	}
}

// nps5 findbuf: MemoryTracker::ForEachCpuDirtyRunRelaxed's walk over regions of `region_size`
// bytes split into `page_size` pages: region(index) returns a pointer (nullptr: the region does not
// exist) and word(pointer, w) its w-th 64-page word of dirty bits. Calls func(begin, end) for the
// ascending, merged runs of [vaddr, vaddr + size) that are dirty pages or missing regions, clipped
// to the range.
template <typename Region, typename Word, typename Func>
void ForEachDirtyOrMissingRun(uint64_t vaddr, uint64_t size, uint64_t region_size,
                              uint64_t page_size, Region&& region, Word&& word, Func&& func) {
	const uint64_t end       = vaddr + size;
	uint64_t       run_begin = 0;
	uint64_t       run_end   = 0;
	bool           open      = false;
	const auto     add       = [&](uint64_t begin, uint64_t finish) {
		begin  = begin < vaddr ? vaddr : begin;
		finish = finish > end ? end : finish;
		if (begin >= finish) {
			return;
		}
		if (open && begin <= run_end) {
			run_end = finish > run_end ? finish : run_end;
			return;
		}
		if (open) {
			func(run_begin, run_end);
		}
		run_begin = begin;
		run_end   = finish;
		open      = true;
	};
	uint64_t index  = vaddr / region_size;
	uint64_t offset = vaddr % region_size;
	uint64_t left   = size;
	while (left != 0) {
		const auto bytes   = region_size - offset < left ? region_size - offset : left;
		const auto base    = index * region_size;
		const auto* manager = region(index);
		if (manager == nullptr) {
			add(base + offset, base + offset + bytes);
		} else {
			const auto first_page = static_cast<size_t>(offset / page_size);
			const auto last_page  = static_cast<size_t>((offset + bytes + page_size - 1) / page_size);
			ForEachSetBitRun([&](size_t w) { return word(manager, w); }, first_page, last_page,
			                 [&](size_t page, size_t page_end) {
				                 add(base + page * page_size, base + page_end * page_size);
			                 });
		}
		left -= bytes;
		offset = 0;
		index++;
	}
	if (open) {
		func(run_begin, run_end);
	}
}

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_BITRUNS_H_
