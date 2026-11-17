#ifndef EMULATOR_SRC_KERNEL_HOSTEDGUARD_H_
#define EMULATOR_SRC_KERNEL_HOSTEDGUARD_H_

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <vector>

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
// A thread hands its slot back when it ends, so the slots follow the live threads (the game starts
// and ends job threads all the time; without that the 128 slots were used up for good after the
// 128th thread and every later reader was back on the lock).
namespace Common {

namespace Detail {
inline uint64_t NextReaderSlotsId() noexcept {
	static std::atomic<uint64_t> next {1};
	return next.fetch_add(1, std::memory_order_relaxed);
}
} // namespace Detail

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
			// Slots handed out for the first time after this load belong to readers that see the
			// writer count; a slot handed out again is below `used` already.
			auto&      table = *slots.m_table;
			const auto used  = std::min(table.used.load(std::memory_order_seq_cst), SlotCount);
			for (uint32_t index = 0; index < used; index++) {
				while (table.slots[index].depth.load(std::memory_order_seq_cst) != 0) {
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

	ReaderSlots(): m_table(std::make_shared<Table>()), m_id(Detail::NextReaderSlotsId()) {}
	ReaderSlots(const ReaderSlots&)            = delete;
	ReaderSlots& operator=(const ReaderSlots&) = delete;

private:
	struct Slot {
		alignas(128) std::atomic<uint32_t> depth {0};
	};

	// The slots live in a block of their own that every thread holding one keeps alive, so a thread
	// can hand its slot back when it exits, whatever became of the ReaderSlots object.
	struct Table {
		Table() { free.reserve(SlotCount); }
		Slot                  slots[SlotCount];
		std::atomic<uint32_t> used {0}; // slots [0, used) were handed out at least once
		std::mutex            free_mutex;
		std::vector<uint32_t> free; // handed back by threads that ended (or moved to another object)

		// A slot index, or SlotCount when none is left.
		uint32_t Take() {
			{
				std::lock_guard lock(free_mutex);
				if (!free.empty()) {
					const auto index = free.back();
					free.pop_back();
					return index;
				}
			}
			if (used.load(std::memory_order_seq_cst) >= SlotCount) {
				return SlotCount;
			}
			const auto index = used.fetch_add(1, std::memory_order_seq_cst);
			return index < SlotCount ? index : SlotCount;
		}
		void Release(uint32_t index) {
			std::lock_guard lock(free_mutex);
			free.push_back(index);
		}
	};

	// What a thread holds of the ReaderSlots objects it read under (by id: an object that is gone
	// and another one at its address are different owners).
	struct Entry {
		uint64_t               owner = 0;
		std::atomic<uint32_t>* slot  = nullptr; // null: the slots had run out
		std::shared_ptr<Table> table;
		uint32_t               index = SlotCount;
		uint32_t               tries = 0;
		void Take() {
			index = table->Take();
			slot  = index < SlotCount ? &table->slots[index].depth : nullptr;
		}
		void Drop() {
			if (slot != nullptr) {
				table->Release(index);
			}
			*this = {};
		}
	};
	struct ThreadEntries {
		static constexpr uint32_t Count = 4;
		Entry entries[Count];
		// The thread ends: its slots go back (guards are scoped, so none is held).
		~ThreadEntries() {
			for (auto& entry: entries) {
				entry.Drop();
			}
		}
	};

	// This thread's slot in this object, taken on first use and handed back when the thread ends;
	// null while the slots have run out (asked again now and then: threads end).
	std::atomic<uint32_t>* ThreadSlot() {
		static thread_local ThreadEntries thread;
		for (auto& entry: thread.entries) {
			if (entry.owner == m_id) {
				if (entry.slot == nullptr && (++entry.tries & 255u) == 0) [[unlikely]] {
					entry.Take();
				}
				return entry.slot;
			}
		}
		// First use of this object on this thread: a free entry, else one whose slot is idle.
		Entry* target = nullptr;
		for (auto& entry: thread.entries) {
			if (entry.owner == 0) {
				target = &entry;
				break;
			}
		}
		for (uint32_t i = 0; target == nullptr && i < ThreadEntries::Count; i++) {
			const auto* slot = thread.entries[i].slot;
			if (slot == nullptr || slot->load(std::memory_order_relaxed) == 0) {
				target = &thread.entries[i];
			}
		}
		if (target == nullptr) {
			return nullptr; // guards of four other objects are held right now: the lock
		}
		target->Drop();
		target->owner = m_id;
		target->table = m_table;
		target->Take();
		return target->slot;
	}

	std::shared_ptr<Table> m_table;
	uint64_t               m_id;
	std::atomic<uint32_t>  m_writers {0};
};

} // namespace Common

#endif // EMULATOR_SRC_KERNEL_HOSTEDGUARD_H_
