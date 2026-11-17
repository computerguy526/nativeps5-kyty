// nps5 slowlvl: Common::ReaderSlots (kernel/hostedGuard.h), the reader-writer guard of the hosted
// guest-memory range map. Exclusion (a writer's intermediate states are never seen, writers never
// overlap), nesting, slot exhaustion (readers fall back to the lock), the lock-only mode, and a
// micro-benchmark against std::shared_mutex.
#include "kernel/hostedGuard.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <shared_mutex>
#include <thread>
#include <vector>

namespace {

int               g_failures = 0;
std::atomic<int>  g_thread_failures {0};
std::atomic<uint64_t> g_checks {0};

void Expect(bool condition, const char* what) {
	if (!condition) {
		std::printf("HostedGuardTests: failed: %s\n", what);
		g_failures++;
	}
}

template <uint32_t N>
void Stress(const char* name, uint32_t readers, uint32_t writers, bool use_slots, int millis) {
	Common::ReaderSlots<N> slots;
	std::shared_mutex      mutex;
	// The invariant a writer breaks inside its guard: a == b, and in_writer == 0 outside.
	volatile uint64_t  a = 0, b = 0;
	std::atomic<int>   in_writer {0};
	std::atomic<bool>  stop {false};
	std::atomic<uint64_t> reads {0}, writes {0}, on_slot {0};
	std::vector<std::thread> threads;
	for (uint32_t r = 0; r < readers; r++) {
		threads.emplace_back([&, r] {
			uint64_t n = 0, s = 0;
			while (!stop.load(std::memory_order_relaxed)) {
				{
					const typename Common::ReaderSlots<N>::Shared guard(slots, mutex, use_slots);
					s += guard.OnSlot() ? 1 : 0;
					const auto x = a;
					if (in_writer.load(std::memory_order_seq_cst) != 0) g_thread_failures++;
					if (guard.OnSlot() && (n & 7) == r % 8) { // nested guard now and then (slot readers only)
						const typename Common::ReaderSlots<N>::Shared inner(slots, mutex, use_slots);
						if (a != x) g_thread_failures++;
					}
					const auto y = b;
					if (x != y) g_thread_failures++;
				}
				n++;
			}
			reads += n;
			on_slot += s;
		});
	}
	for (uint32_t w = 0; w < writers; w++) {
		threads.emplace_back([&] {
			uint64_t n = 0;
			while (!stop.load(std::memory_order_relaxed)) {
				{
					const typename Common::ReaderSlots<N>::Exclusive guard(slots, mutex);
					if (in_writer.fetch_add(1, std::memory_order_seq_cst) != 0) g_thread_failures++;
					a = a + 1;
					for (int spin = 0; spin < 50; spin++) { std::atomic_thread_fence(std::memory_order_seq_cst); }
					b = b + 1;
					in_writer.fetch_sub(1, std::memory_order_seq_cst);
				}
				n++;
				if ((n & 15) == 0) std::this_thread::sleep_for(std::chrono::microseconds(50));
			}
			writes += n;
		});
	}
	std::this_thread::sleep_for(std::chrono::milliseconds(millis));
	stop = true;
	for (auto& t: threads) t.join();
	Expect(a == b, "writers left the invariant");
	Expect(a == writes.load(), "every write counted once (writers excluded each other)");
	Expect(reads.load() > 0 && (writers == 0 || writes.load() > 0), "both sides made progress");
	if (!use_slots) Expect(on_slot.load() == 0, "lock-only mode never uses a slot");
	g_checks += 3 * reads.load();
	std::printf("  %-34s readers %u writers %u: %llu reads (%llu on a slot), %llu writes\n", name, readers,
	            writers, static_cast<unsigned long long>(reads.load()),
	            static_cast<unsigned long long>(on_slot.load()), static_cast<unsigned long long>(writes.load()));
}

void TestSingleThread() {
	Common::ReaderSlots<4> slots;
	std::shared_mutex      mutex;
	{
		const Common::ReaderSlots<4>::Shared outer(slots, mutex);
		Expect(outer.OnSlot(), "first reader gets a slot");
		const Common::ReaderSlots<4>::Shared inner(slots, mutex);
		Expect(inner.OnSlot(), "nested reader stays on the slot");
	}
	{ const Common::ReaderSlots<4>::Exclusive writer(slots, mutex); } // no reader left behind: returns
	{
		const Common::ReaderSlots<4>::Shared locked(slots, mutex, false);
		Expect(!locked.OnSlot(), "lock-only reader");
	}
	// Slot exhaustion: threads beyond the capacity read on the lock.
	std::atomic<int> on_slot {0}, on_lock {0};
	std::vector<std::thread> threads;
	for (int i = 0; i < 12; i++) {
		threads.emplace_back([&] {
			const Common::ReaderSlots<4>::Shared guard(slots, mutex);
			(guard.OnSlot() ? on_slot : on_lock)++;
		});
	}
	for (auto& t: threads) t.join();
	Expect(on_slot == 3 && on_lock == 9, "3 more slots (4 with this thread's), the rest on the lock");
	{ const Common::ReaderSlots<4>::Exclusive writer(slots, mutex); }
}

template <typename Body>
double NsPerOp(uint32_t threads_n, Body body) {
	std::atomic<bool>     go {false}, stop {false};
	std::atomic<uint64_t> total {0};
	std::vector<std::thread> threads;
	for (uint32_t t = 0; t < threads_n; t++) {
		threads.emplace_back([&] {
			while (!go.load()) {}
			uint64_t n = 0;
			while (!stop.load(std::memory_order_relaxed)) { body(); n++; }
			total += n;
		});
	}
	const auto start = std::chrono::steady_clock::now();
	go = true;
	std::this_thread::sleep_for(std::chrono::milliseconds(300));
	stop = true;
	for (auto& t: threads) t.join();
	const double ns = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - start).count();
	return ns * threads_n / static_cast<double>(total.load());
}

void Bench() {
	static Common::ReaderSlots<128> slots;
	static std::shared_mutex        mutex;
	for (uint32_t n: {1u, 7u}) {
		const double lock = NsPerOp(n, [] { std::shared_lock guard(mutex); });
		const double slot = NsPerOp(n, [] { const Common::ReaderSlots<128>::Shared guard(slots, mutex); });
		std::printf("  bench %u reader thread(s): shared_mutex %.0f ns per guard, reader slot %.1f ns\n", n, lock, slot);
	}
}

} // namespace

int main() {
	TestSingleThread();
	Stress<128>("slots, readers only", 7, 0, true, 150);
	Stress<128>("slots, 7 readers 1 writer", 7, 1, true, 400);
	Stress<128>("slots, 7 readers 3 writers", 7, 3, true, 400);
	Stress<4>("4 slots, 10 readers 2 writers", 10, 2, true, 400);
	Stress<128>("lock only, 7 readers 2 writers", 7, 2, false, 300);
	Stress<1>("1 slot, 6 readers 2 writers", 6, 2, true, 300);
	Bench();
	g_failures += g_thread_failures.load();
	if (g_failures != 0) {
		std::printf("%d failure(s)\n", g_failures);
		return 1;
	}
	std::printf("hosted_guard: all tests passed (%llu checks)\n", static_cast<unsigned long long>(g_checks.load()));
	return 0;
}
