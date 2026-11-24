#ifndef KYTY_HYBRID_PENDINGMEMORY_H_
#define KYTY_HYBRID_PENDINGMEMORY_H_

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <vector>

// nps5 hosted memory: the notifications a 1.6 host sends between init and start_graphics (it
// registers everything mapped before the renderer exists). They are kept in order and applied once
// the library's memory subsystem is up (kytyGpuBridge.cpp RendererBegin). Header only:
// tests/HybridSeamTests.cpp.
namespace Hybrid {

class PendingMemory {
public:
	enum class Kind : uint8_t { Register, Unregister, Protect };
	struct Note {
		Kind        kind  = Kind::Register;
		uint64_t    base  = 0;
		uint64_t    size  = 0;
		uint32_t    flags = 0;
		const void* alias = nullptr;
	};

	// Queues the note and returns true while the memory subsystem is not ready; false once it is
	// (the caller then applies the note itself).
	bool Queue(const Note& note) {
		std::lock_guard lock(m_mutex);
		if (m_ready) {
			return false;
		}
		m_notes.push_back(note);
		return true;
	}

	// Applies every queued note in order through apply(note) and marks the subsystem ready; notes
	// that arrive meanwhile wait on the lock and are then applied by their callers. Returns the
	// count.
	template <typename Apply>
	size_t Drain(Apply&& apply) {
		std::lock_guard lock(m_mutex);
		for (const auto& note: m_notes) {
			apply(note);
		}
		const auto count = m_notes.size();
		m_notes.clear();
		m_ready = true;
		return count;
	}

	// True when the queued notes leave all of [base, base + size) registered: the last register or
	// unregister note that touches the range decides.
	[[nodiscard]] bool Covers(uint64_t base, uint64_t size) {
		std::lock_guard lock(m_mutex);
		bool            covered = false;
		for (const auto& note: m_notes) {
			if (note.kind != Kind::Protect && note.base < base + size &&
			    base < note.base + note.size) {
				covered = note.kind == Kind::Register && note.base <= base &&
				          base + size <= note.base + note.size;
			}
		}
		return covered;
	}

private:
	std::mutex        m_mutex;
	std::vector<Note> m_notes;
	bool              m_ready = false;
};

// NPS5_SELF_MEM_LIMIT_MB: the self memory limit in MiB, 0 = off. The limit protects the machine, so
// only a clean "0" turns it off: text that is not a non-negative number in range keeps `fallback`
// (strtol alone reads "abc" as 0 = off).
[[nodiscard]] inline long SelfMemLimitMb(const char* text, long fallback = 18944) {
	if (text == nullptr || text[0] == '\0') {
		return fallback;
	}
	char* end        = nullptr;
	errno            = 0;
	const long value = std::strtol(text, &end, 10);
	if (end == text || *end != '\0' || errno != 0 || value < 0 || value > (1L << 30)) {
		return fallback;
	}
	return value;
}

} // namespace Hybrid

#endif // KYTY_HYBRID_PENDINGMEMORY_H_
