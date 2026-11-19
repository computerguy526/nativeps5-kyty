#pragma once
// memtrim: pure helpers (no Vulkan) so tests/memtrim can unit-test them
// (work/memtrim/tests/run.sh).
#include <algorithm>
#include <cstdint>
#include <vector>

namespace memtrim {

// Pages of [first, first + size_pages) that fit in a BDA page table of `entries` entries.
inline uint64_t BdaCovered(uint64_t first, uint64_t size_pages, uint64_t entries) {
	return first < entries ? std::min<uint64_t>(size_pages, entries - first) : 0;
}

// Pages of [first, last_exclusive) that lie above the table (0 if it all fits).
inline uint64_t BdaUncovered(uint64_t first, uint64_t last_exclusive, uint64_t entries) {
	return last_exclusive <= entries ? 0 : last_exclusive - std::max(first, entries);
}

// Guest ranges whose GPU-written image was retired without a download (contents lost).
class LostRanges {
public:
	struct Entry {
		uint64_t address;
		uint64_t end;
		uint64_t frame;
	};
	void Add(uint64_t address, uint64_t size, uint64_t frame) {
		m_ranges.push_back({address, address + size, frame});
		if (m_ranges.size() > 4096) {
			m_ranges.erase(m_ranges.begin());
		}
	}
	// An image is registered over [begin, end): the first overlapping lost range is removed and
	// returned.
	bool Take(uint64_t begin, uint64_t end, Entry& out) {
		for (auto it = m_ranges.begin(); it != m_ranges.end(); ++it) {
			if (begin < it->end && it->address < end) {
				out = *it;
				m_ranges.erase(it);
				return true;
			}
		}
		return false;
	}
	[[nodiscard]] bool   Empty() const noexcept { return m_ranges.empty(); }
	[[nodiscard]] size_t Size() const noexcept { return m_ranges.size(); }

private:
	std::vector<Entry> m_ranges;
};

} // namespace memtrim
