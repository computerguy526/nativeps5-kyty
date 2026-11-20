#pragma once
// KYTY_ALIAS_REBUILD_PARTIAL (TextureCache::TryOwnedPartialRebuild): the bytes of an image's
// guest range whose native contents are known to equal the canonical bytes (the cache buffer, or
// guest memory where no buffer holds them). A rebuild reads the canonical bytes, so it may skip a
// block in which every byte is owned by the image (native is newer) or coherent (native is the
// same): the old whole-image rebuild would have written identical texels there.
//
// The set only ever claims equality, so every event that can break it must remove bytes; a
// missed addition costs only bandwidth. TextureCache keeps it (only while the switch is on):
//   added    after a whole refresh from the canonical bytes (all but the owned bytes), after the
//            partial rebuild (its segments), and when MaterializeOwnedBytes moves owned bytes
//            to the buffer (the canonical bytes become the native ones);
//   removed  by any GPU buffer write over the range (InvalidateMemoryFromGPU), any write of
//            canonical bytes from an image or binding (BufferCache::NoteBufferContentWrite,
//            CPU readback), a write of this image (the claimed bytes; everything for writes that
//            claim the whole image, copies into it and unbounded writers), and cleared by every
//            CPU-dirty transition, residency change and loss of GPU ownership (Image).
//   kept     by a write of ANOTHER image over the range: it changes neither this image's native
//            bytes nor the canonical ones (until materialized, which is a canonical write).
// Invariant: no owned byte is coherent (claims remove them), so an owned byte that leaves
// without materialization (a copy taking it over, another image's write) is never coherent.
// Header-only; tests/OwnedRebuildPlanTests.cpp covers it.
#include "graphics/host_gpu/rangeSet.h"

#include <cstdint>
#include <memory>

namespace Libs::Graphics {

class ImageCoherence final {
public:
	void               Clear() noexcept { m_set.reset(); }
	[[nodiscard]] bool Empty() const noexcept { return !m_set || m_set->Empty(); }

	// Every byte of [base, base + size) except the owned ones (owns_all: none; owned == nullptr
	// and !owns_all: no owned byte).
	void SetAllExcept(uint64_t base, uint64_t size, const RangeSet* owned, bool owns_all) {
		m_set = std::make_unique<RangeSet>();
		if (owns_all || size == 0) {
			return;
		}
		m_set->Add(base, size);
		if (owned != nullptr) {
			owned->ForEachInRange(base, size, [&](uint64_t begin, uint64_t end) {
				m_set->Subtract(begin, end - begin);
			});
		}
	}
	void Add(uint64_t address, uint64_t size) {
		if (size == 0) {
			return;
		}
		if (!m_set) {
			m_set = std::make_unique<RangeSet>();
		}
		m_set->Add(address, size);
	}
	void Subtract(uint64_t address, uint64_t size) {
		if (m_set && size != 0) {
			m_set->Subtract(address, size);
		}
	}
	[[nodiscard]] bool Contains(uint64_t address, uint64_t size) const {
		return size == 0 || (m_set && m_set->Contains(address, size));
	}
	// Whether every byte of [address, address + size) is owned (owns_all, or in owned) or
	// coherent.
	[[nodiscard]] bool CoveredWith(const RangeSet* owned, bool owns_all, uint64_t address,
	                               uint64_t size) const {
		if (owns_all || size == 0) {
			return true;
		}
		uint64_t cursor = address;
		bool     ok     = true;
		if (owned != nullptr) {
			owned->ForEachInRange(address, size, [&](uint64_t begin, uint64_t end) {
				ok     = ok && Contains(cursor, begin - cursor);
				cursor = end;
			});
		}
		return ok && Contains(cursor, address + size - cursor);
	}

private:
	std::unique_ptr<RangeSet> m_set;
};

} // namespace Libs::Graphics
