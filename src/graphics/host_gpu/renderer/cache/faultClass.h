// nps5 faultclass: learned read class per guest tracker page (KYTY_FAULT_CLASS, default off).
//
// A guest load of a page whose bytes the GPU owns faults. The handler can serve the load from the
// page's published bytes (KYTY_READ_FAULT_PUBLISHED), or wait for the writer. A page the game reads
// in frame after frame pays that fault (and, past the burst limits, the wait) every frame. This
// table learns which pages are read like that and tells the buffer cache to keep them READ-OPEN
// (MemoryTracker::OpenRead: readable across new GPU writers, their eager copy publishes the new
// bytes when the writer completed), i.e. the load sees what a served load sees, without a fault.
//
// Classes of a page:
//   Unknown  the default: every rule of the fault path as without this table.
//   Late     read faults in `promote_frames` distinct guest frames within `window_frames`: the
//            caller opens the page. A page read once (a one-shot readback behind a label) never
//            gets here, so its read keeps waiting for the writer.
//   Exact    a Late page whose guest stores collided `demote_conflicts` times within
//            `window_frames` with GPU bytes not yet published (each such store drains the command
//            processor; with the read watch on, the reader's wait had cleaned the page before the
//            store): the caller closes the page, and it stays closed for `exact_frames` (doubling
//            per demotion up to 16x), then counts as Unknown again.
// Seeds (a list of pages that were Late in an earlier run, KYTY_FAULT_CLASS_PROFILE) start as Late
// candidates: the caller makes them read-hot and opens them once their bytes are published, without
// a first fault. A seed that is never GPU-owned costs one map entry.
//
// Only bookkeeping lives here; the caller decides whether a page may be opened (buffer bytes only,
// published recently) and closes it by its own rules as well (not published recently, image,
// expiry). Thread-safe (one mutex; faults of all guest threads come through here).
#pragma once

#include <algorithm>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics::FaultClass {

enum class Class : uint8_t { Unknown, Late, Exact };

struct Limits {
	uint32_t promote_frames   = 3;   // distinct frames with a read fault before a page is Late
	uint32_t window_frames    = 120; // the frames those faults / conflicts must fall into
	uint32_t demote_conflicts = 3;   // store conflicts within the window before a Late page is Exact
	uint32_t exact_frames     = 600; // first Exact period
	uint32_t capacity         = 16384; // pages tracked; past it idle Unknown pages are dropped
};

struct Counters {
	uint64_t promoted  = 0;
	uint64_t demoted   = 0;
	uint64_t conflicts = 0;
	uint64_t seeded    = 0;
	uint64_t seed_hits = 0; // seeds the caller reported opened
};

class Table final {
public:
	Table() = default;
	explicit Table(Limits limits): m_limits(limits) {}

	// A guest read fault on `page` (tracker-page aligned) in guest frame `frame`, served or
	// waited. True: the page is (now) Late and the caller may open it.
	bool NoteRead(uint64_t page, uint32_t frame) {
		std::lock_guard lock(m_mutex);
		auto&           entry = Find(page, frame);
		Expire(entry, frame);
		if (entry.cls == Class::Exact) {
			return false;
		}
		if (entry.cls == Class::Late) {
			entry.last_read = frame;
			return true;
		}
		if (entry.read_frames == 0 || frame - entry.first_read > m_limits.window_frames) {
			entry.first_read  = frame;
			entry.read_frames = 1;
		} else if (entry.last_read != frame) {
			entry.read_frames++;
		}
		entry.last_read = frame;
		if (entry.read_frames >= m_limits.promote_frames) {
			entry.cls       = Class::Late;
			entry.conflicts = 0;
			m_counters.promoted++;
			return true;
		}
		return false;
	}

	// A guest store on a Late (open) page had to wait for GPU bytes. True: the page is now Exact
	// and the caller must close it.
	bool NoteStoreConflict(uint64_t page, uint32_t frame) {
		std::lock_guard lock(m_mutex);
		const auto      it = m_pages.find(page);
		if (it == m_pages.end() || it->second.cls != Class::Late) {
			return false;
		}
		auto& entry = it->second;
		m_counters.conflicts++;
		if (entry.conflicts == 0 || frame - entry.first_conflict > m_limits.window_frames) {
			entry.first_conflict = frame;
			entry.conflicts      = 0;
		}
		if (++entry.conflicts < m_limits.demote_conflicts) {
			return false;
		}
		entry.cls         = Class::Exact;
		entry.demotions   = static_cast<uint8_t>(std::min<uint32_t>(entry.demotions + 1u, 5u));
		entry.exact_until = frame + (m_limits.exact_frames << (entry.demotions - 1u));
		entry.read_frames = 0;
		entry.conflicts   = 0;
		entry.seed        = false;
		m_counters.demoted++;
		return true;
	}

	// The caller closed an open page for its own reasons (not published recently, image, expiry):
	// the page has to earn Late again, unless it was demoted (Exact keeps its period).
	void NoteClosed(uint64_t page) {
		std::lock_guard lock(m_mutex);
		const auto      it = m_pages.find(page);
		if (it != m_pages.end() && it->second.cls == Class::Late) {
			it->second.cls         = Class::Unknown;
			it->second.read_frames = 0;
		}
	}

	[[nodiscard]] Class Get(uint64_t page, uint32_t frame) {
		std::lock_guard lock(m_mutex);
		const auto      it = m_pages.find(page);
		if (it == m_pages.end()) {
			return Class::Unknown;
		}
		Expire(it->second, frame);
		return it->second.cls;
	}

	// The caller opened `page` without a fault (a neighbour in its host page): Late unless Exact.
	void SetLate(uint64_t page) {
		std::lock_guard lock(m_mutex);
		auto&           entry = m_pages[page];
		if (entry.cls == Class::Unknown) {
			entry.cls       = Class::Late;
			entry.conflicts = 0;
		}
	}

	// Seeds: Late from the start, flagged so the caller can open them without a fault.
	void Seed(uint64_t page) {
		std::lock_guard lock(m_mutex);
		auto&           entry = m_pages[page];
		if (entry.cls == Class::Unknown) {
			entry.cls  = Class::Late;
			entry.seed = true;
			m_seeds.push_back(page);
			m_counters.seeded++;
		}
	}
	// Up to `max` seeds not yet reported opened, round robin (the caller's per-frame check).
	void PendingSeeds(std::vector<uint64_t>& out, size_t max) {
		std::lock_guard lock(m_mutex);
		out.clear();
		for (size_t n = 0; n < m_seeds.size() && out.size() < max; n++) {
			m_seed_cursor       = m_seed_cursor >= m_seeds.size() ? 0 : m_seed_cursor;
			const auto page     = m_seeds[m_seed_cursor++];
			const auto it       = m_pages.find(page);
			if (it != m_pages.end() && it->second.seed && it->second.cls == Class::Late) {
				out.push_back(page);
			}
		}
	}
	void NoteSeedOpened(uint64_t page) {
		std::lock_guard lock(m_mutex);
		const auto      it = m_pages.find(page);
		if (it != m_pages.end() && it->second.seed) {
			it->second.seed = false;
			m_counters.seed_hits++;
		}
	}

	// Every page that is Late now (sorted): the profile of this run.
	[[nodiscard]] std::vector<uint64_t> LatePages() {
		std::lock_guard       lock(m_mutex);
		std::vector<uint64_t> pages;
		for (const auto& [page, entry]: m_pages) {
			if (entry.cls == Class::Late && !entry.seed) {
				pages.push_back(page);
			}
		}
		std::sort(pages.begin(), pages.end());
		return pages;
	}

	[[nodiscard]] Counters TakeCounters() {
		std::lock_guard lock(m_mutex);
		const auto      taken = m_counters;
		m_counters            = {};
		return taken;
	}
	[[nodiscard]] size_t Size() {
		std::lock_guard lock(m_mutex);
		return m_pages.size();
	}

private:
	struct Entry {
		uint32_t first_read     = 0;
		uint32_t last_read      = 0;
		uint32_t first_conflict = 0;
		uint32_t exact_until    = 0;
		uint16_t read_frames    = 0;
		uint16_t conflicts      = 0;
		uint8_t  demotions      = 0;
		Class    cls            = Class::Unknown;
		bool     seed           = false;
	};

	Entry& Find(uint64_t page, uint32_t frame) {
		const auto it = m_pages.find(page);
		if (it != m_pages.end()) {
			return it->second;
		}
		if (m_pages.size() >= m_limits.capacity) {
			std::erase_if(m_pages, [&](const auto& item) {
				return item.second.cls == Class::Unknown && item.second.demotions == 0 &&
				       frame - item.second.last_read > m_limits.window_frames;
			});
		}
		return m_pages[page];
	}
	void Expire(Entry& entry, uint32_t frame) const {
		if (entry.cls == Class::Exact && static_cast<int32_t>(frame - entry.exact_until) >= 0) {
			entry.cls         = Class::Unknown;
			entry.read_frames = 0;
		}
	}

	Limits                              m_limits;
	std::mutex                          m_mutex;
	std::unordered_map<uint64_t, Entry> m_pages;
	std::vector<uint64_t>               m_seeds;
	size_t                              m_seed_cursor = 0;
	Counters                            m_counters;
};

// Profile file: one line per run of pages, "<first page hex> <page count>". Unknown lines are
// skipped; page addresses must be `page_size` aligned.
inline bool SaveProfile(const std::string& path, const std::vector<uint64_t>& sorted_pages,
                        uint64_t page_size) {
	const std::string tmp  = path + ".tmp";
	std::FILE*        file = std::fopen(tmp.c_str(), "w");
	if (file == nullptr) {
		return false;
	}
	std::fprintf(file, "# kyty fault class profile v1 page=%" PRIu64 "\n", page_size);
	for (size_t i = 0; i < sorted_pages.size();) {
		size_t n = 1;
		while (i + n < sorted_pages.size() && sorted_pages[i + n] == sorted_pages[i] + n * page_size) {
			n++;
		}
		std::fprintf(file, "%" PRIx64 " %zu\n", sorted_pages[i], n);
		i += n;
	}
	const bool ok = std::fclose(file) == 0;
	return ok && std::rename(tmp.c_str(), path.c_str()) == 0;
}

inline size_t LoadProfile(const std::string& path, uint64_t page_size, size_t max_pages, Table& table) {
	std::FILE* file = std::fopen(path.c_str(), "r");
	if (file == nullptr) {
		return 0;
	}
	char   line[128];
	size_t loaded = 0;
	while (std::fgets(line, sizeof(line), file) != nullptr && loaded < max_pages) {
		uint64_t           first = 0;
		unsigned long long count = 0;
		if (line[0] == '#' || std::sscanf(line, "%" SCNx64 " %llu", &first, &count) != 2 ||
		    first == 0 || (first % page_size) != 0 || count == 0 || count > 65536) {
			continue;
		}
		for (unsigned long long i = 0; i < count && loaded < max_pages; i++, loaded++) {
			table.Seed(first + i * page_size);
		}
	}
	std::fclose(file);
	return loaded;
}

} // namespace Libs::Graphics::FaultClass
