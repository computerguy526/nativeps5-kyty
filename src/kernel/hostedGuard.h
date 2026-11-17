#ifndef EMULATOR_SRC_KERNEL_HOSTEDGUARD_H_
#define EMULATOR_SRC_KERNEL_HOSTEDGUARD_H_

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <shared_mutex>
#include <thread>

// nps5 slowlvl: a reader-writer guard for a read-mostly structure whose readers are on hot paths
// (the hosted guest-memory range map, kernel/memory.cpp: every recorded guest read of a draw
// preparation looked a range up under a std::shared_mutex, a pthread rwlock on macOS; with six
// draw-prep workers and the command processor that was about 40 lock round trips per draw and
// most of a preparation: KYTY_DRAW_PREP_PHASES reads 73 of 87 us per slot at Sky Garden).
//
// A reader announces itself in a slot of its own (one cache line per thread; no shared write, no
// system call) and takes the shared lock only while a writer is in, or when the slots have run
// out. A writer takes the mutex exclusively (other writers, readers on the lock), raises the
// writer count and waits until every announced reader has left.
// Dekker pairs, all seq_cst: reader = slot store, then writer-count load; writer = writer-count
// store, then slot loads. Either the reader sees the writer (and backs out to the lock), or the
// writer sees the reader (and waits). So, exactly as with the lock alone, nothing a writer does is
// visible to a reader inside its guard. Guards nest on one thread while the outer one is on its
// slot (a count). A reader on the lock that nests can deadlock against a waiting writer, and a
// thread that writes while it reads does, both as on the lock alone.
namespace Common {

template <uint32_t SlotCount>
class ReaderSlots {
public:
	class Shared {
	public:
		Shared(ReaderSlots& slots, std::shared_mutex& mutex, bool use_slots = true) {
			if (auto* slot = use_slots ? slots.ThreadSlot() : nullptr; slot != nullptr) {
				// Only this thread writes its slot.
				const auto depth = slot->load(std::memory_order_relaxed);
				if (depth != 0) {
					// Nested: the outer guard already keeps writers out (one that arrived since
					// is waiting for this very slot, so backing out to the lock would deadlock).
					slot->store(depth + 1, std::memory_order_relaxed);
					m_slot = slot;
					return;
				}
				slot->store(1, std::memory_order_seq_cst);
				if (slots.m_writers.load(std::memory_order_seq_cst) == 0) {
					m_slot = slot;
					return;
				}
				slot->store(0, std::memory_order_seq_cst);
			}
			m_lock = std::shared_lock(mutex);
		}
		~Shared() {
			if (m_slot != nullptr) {
				m_slot->store(m_slot->load(std::memory_order_relaxed) - 1, std::memory_order_release);
			}
		}
		Shared(const Shared&)            = delete;
		Shared& operator=(const Shared&) = delete;
		[[nodiscard]] bool OnSlot() const noexcept { return m_slot != nullptr; }

	private:
		std::atomic<uint32_t>*              m_slot = nullptr;
		std::shared_lock<std::shared_mutex> m_lock;
	};

	class Exclusive {
	public:
		Exclusive(ReaderSlots& slots, std::shared_mutex& mutex): m_slots(slots), m_lock(mutex) {
			slots.m_writers.fetch_add(1, std::memory_order_seq_cst);
			// Slots handed out after this load belong to readers that see the writer count.
			const auto used = std::min(slots.m_used.load(std::memory_order_seq_cst), SlotCount);
			for (uint32_t index = 0; index < used; index++) {
				while (slots.m_slots[index].depth.load(std::memory_order_seq_cst) != 0) {
					std::this_thread::yield();
				}
			}
		}
		~Exclusive() { m_slots.m_writers.fetch_sub(1, std::memory_order_seq_cst); }
		Exclusive(const Exclusive&)            = delete;
		Exclusive& operator=(const Exclusive&) = delete;

	private:
		ReaderSlots&                        m_slots;
		std::unique_lock<std::shared_mutex> m_lock;
	};

private:
	struct Slot {
		alignas(128) std::atomic<uint32_t> depth {0};
	};

	// This thread's slot in this object, taken on first use; null once the slots have run out.
	std::atomic<uint32_t>* ThreadSlot() {
		struct Entry {
			const ReaderSlots*     owner = nullptr;
			std::atomic<uint32_t>* slot  = nullptr;
		};
		static thread_local Entry entry;
		if (entry.owner != this) {
			const auto index = m_used.fetch_add(1, std::memory_order_seq_cst);
			entry            = {this, index < SlotCount ? &m_slots[index].depth : nullptr};
		}
		return entry.slot;
	}

	Slot                  m_slots[SlotCount];
	std::atomic<uint32_t> m_used {0};
	std::atomic<uint32_t> m_writers {0};
};

} // namespace Common

#endif // EMULATOR_SRC_KERNEL_HOSTEDGUARD_H_
