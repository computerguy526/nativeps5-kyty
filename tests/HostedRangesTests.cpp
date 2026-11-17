// nps5: the hosted guest-memory range map (kernel/hostedRanges.h) under its guard
// (kernel/hostedGuard.h), as kernel/memory.cpp uses them.
// - A range that is cut keeps its outer parts WITHOUT the alias: the host frees the alias of the whole
//   range with the cut and re-registers the parts with new ones (an alias cannot be split).
// - register_memory over registered memory is one step for readers: the range is never seen missing.
// - Stress: readers on per-thread slots against a host thread that cuts, frees (poisons) the alias
//   and re-registers; a reader that ever follows a freed alias reads the poison.
#include "kernel/hostedGuard.h"
#include "kernel/hostedRanges.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <shared_mutex>
#include <thread>
#include <vector>

namespace {

using namespace Common::HostedRanges;

int g_failures = 0;

void Expect(bool condition, const char* what) {
	if (!condition) {
		std::printf("HostedRangesTests: failed: %s\n", what);
		g_failures++;
	}
}

Common::ReaderSlots<128> g_slots;
std::shared_mutex        g_mutex;
struct Shared: Common::ReaderSlots<128>::Shared {
	Shared(): Common::ReaderSlots<128>::Shared(g_slots, g_mutex) {}
};
struct Exclusive: Common::ReaderSlots<128>::Exclusive {
	Exclusive(): Common::ReaderSlots<128>::Exclusive(g_slots, g_mutex) {}
};

const Range* Find(const Map& ranges, uint64_t address) {
	auto it = ranges.upper_bound(address);
	if (it == ranges.begin()) {
		return nullptr;
	}
	--it;
	return address - it->first < it->second.size ? &it->second : nullptr;
}

void TestCut() {
	Map                   ranges;
	std::atomic<uint64_t> generation {1};
	Pieces                removed;
	const auto*           alias = reinterpret_cast<const uint8_t*>(uintptr_t {0x7000000000});
	Register<Exclusive>(ranges, generation, 0x10000, 0x30000, 0x33, alias, removed);
	Expect(removed.empty() && ranges.size() == 1 && generation.load() > 1, "first registration");
	Expect(ranges[0x10000].alias == alias && ranges[0x10000].size == 0x30000, "registered as given");

	removed.clear();
	Expect(Unregister<Exclusive>(ranges, generation, 0x20000, 0x10000, removed), "cut inside a range");
	Expect(removed.size() == 1 && removed[0].first == 0x20000 && removed[0].second == 0x10000, "removed part");
	Expect(ranges.size() == 2 && ranges.count(0x10000) == 1 && ranges.count(0x30000) == 1, "outer parts stay");
	Expect(ranges[0x10000].size == 0x10000 && ranges[0x30000].size == 0x10000, "outer parts' sizes");
	Expect(ranges[0x10000].flags == 0x33 && ranges[0x30000].flags == 0x33, "outer parts keep the flags");
	// The fix: the host frees the whole alias with the cut.
	Expect(ranges[0x10000].alias == nullptr, "the part before a cut has no alias until it is registered again");
	Expect(ranges[0x30000].alias == nullptr, "the part behind a cut has no alias until it is registered again");

	// A cut that removes a whole range leaves its neighbours' aliases alone.
	const auto* alias2 = reinterpret_cast<const uint8_t*>(uintptr_t {0x7100000000});
	removed.clear();
	Register<Exclusive>(ranges, generation, 0x10000, 0x10000, 0x33, alias, removed);
	Register<Exclusive>(ranges, generation, 0x30000, 0x10000, 0x33, alias2, removed);
	removed.clear();
	Expect(Unregister<Exclusive>(ranges, generation, 0x30000, 0x10000, removed), "cut of a whole range");
	Expect(ranges.size() == 1 && ranges[0x10000].alias == alias, "a neighbour keeps its alias");
	removed.clear();
	Expect(!Unregister<Exclusive>(ranges, generation, 0x80000, 0x1000, removed) && removed.empty(), "nothing there");

	// Replacement across several ranges.
	removed.clear();
	Register<Exclusive>(ranges, generation, 0x30000, 0x10000, 0x33, alias2, removed);
	removed.clear();
	Register<Exclusive>(ranges, generation, 0x18000, 0x20000, 0x13, nullptr, removed);
	Expect(removed.size() == 2, "a registration over two ranges removes a part of each");
	Expect(ranges.size() == 3 && ranges[0x10000].size == 0x8000 && ranges[0x18000].size == 0x20000 &&
	           ranges[0x38000].size == 0x8000,
	       "replacement layout");
	Expect(ranges[0x10000].alias == nullptr && ranges[0x38000].alias == nullptr, "cut neighbours lose the alias");
}

// Readers must never see a registered range missing while the host registers it again.
void TestReplaceIsOneStep(int millis) {
	Map                   ranges;
	std::atomic<uint64_t> generation {1};
	{
		Pieces removed;
		Register<Exclusive>(ranges, generation, 0x100000, 0x40000, 1, nullptr, removed);
	}
	std::atomic<bool>     stop {false};
	std::atomic<uint64_t> missing {0}, reads {0}, writes {0};
	std::vector<std::thread> threads;
	for (int r = 0; r < 4; r++) {
		threads.emplace_back([&] {
			while (!stop.load(std::memory_order_relaxed)) {
				const Shared guard;
				if (Find(ranges, 0x110000) == nullptr) {
					missing.fetch_add(1, std::memory_order_relaxed);
				}
				reads.fetch_add(1, std::memory_order_relaxed);
			}
		});
	}
	threads.emplace_back([&] {
		Pieces removed;
		while (!stop.load(std::memory_order_relaxed)) {
			removed.clear();
			Register<Exclusive>(ranges, generation, 0x100000, 0x40000, 1, nullptr, removed);
			writes.fetch_add(1, std::memory_order_relaxed);
		}
	});
	std::this_thread::sleep_for(std::chrono::milliseconds(millis));
	stop.store(true);
	for (auto& thread: threads) {
		thread.join();
	}
	std::printf("HostedRangesTests: replace: %llu reads, %llu registrations, %llu reads saw the range missing\n",
	            static_cast<unsigned long long>(reads.load()), static_cast<unsigned long long>(writes.load()),
	            static_cast<unsigned long long>(missing.load()));
	Expect(missing.load() == 0, "a range that is registered again is never missing for a reader");
	Expect(reads.load() > 1000 && writes.load() > 100, "the replace stress ran");
}

// The host's drop_span (nativeps5 runtime): unregister a part, free the alias of the WHOLE range, register the
// outer parts again with new aliases. A freed alias is poisoned here instead of unmapped.
void TestCutFreesAlias(int millis) {
	constexpr uint64_t Base = 0x200000, Size = 0x300, Page = 0x100; // small 'pages': many rounds, little memory
	constexpr uint8_t  Live = 0xaa, Poison = 0xdd;
	Map                   ranges;
	std::atomic<uint64_t> generation {1};
	std::vector<uint8_t>  guest(Size, Live); // the guest view: always valid
	// Aliases are never really freed while the test runs (a late reader must read poison, not crash).
	std::vector<std::unique_ptr<uint8_t[]>> aliases;
	const auto make_alias = [&](uint64_t bytes) {
		aliases.push_back(std::make_unique<uint8_t[]>(bytes));
		std::memset(aliases.back().get(), Live, bytes);
		return aliases.back().get();
	};
	uint8_t* whole = make_alias(Size);
	{
		Pieces removed;
		Register<Exclusive>(ranges, generation, Base, Size, 1, whole, removed);
	}
	std::atomic<bool>     stop {false};
	std::atomic<uint64_t> poisoned {0}, reads {0}, rounds {0};
	std::vector<std::thread> threads;
	for (int r = 0; r < 4; r++) {
		threads.emplace_back([&, r] {
			// The outer parts only: the middle page is the one the guest unmaps.
			const uint64_t address = r % 2 == 0 ? Base + 0x10 : Base + 2 * Page + 0x10;
			while (!stop.load(std::memory_order_relaxed)) {
				const Shared guard;
				auto         it = ranges.upper_bound(address);
				if (it == ranges.begin()) {
					continue;
				}
				--it;
				if (address - it->first >= it->second.size) {
					continue;
				}
				const uint8_t* host = it->second.alias != nullptr ? it->second.alias + (address - it->first)
				                                                  : guest.data() + (address - Base);
				if (*static_cast<const volatile uint8_t*>(host) != Live) {
					poisoned.fetch_add(1, std::memory_order_relaxed);
				}
				reads.fetch_add(1, std::memory_order_relaxed);
			}
		});
	}
	threads.emplace_back([&] {
		Pieces   removed;
		uint8_t* current = whole;
		uint64_t current_bytes = Size;
		while (!stop.load(std::memory_order_relaxed) && aliases.size() < 150000) {
			// unregister_memory(middle page), then alias_free of the whole range's alias
			removed.clear();
			(void)Unregister<Exclusive>(ranges, generation, Base + Page, Page, removed);
			std::memset(current, Poison, current_bytes);
			std::this_thread::yield();
			// the outer parts again, each with a fresh alias; then the middle page comes back (a new mapping)
			uint8_t* low  = make_alias(Page);
			uint8_t* high = make_alias(Page);
			removed.clear();
			Register<Exclusive>(ranges, generation, Base, Page, 1, low, removed);
			removed.clear();
			Register<Exclusive>(ranges, generation, Base + 2 * Page, Page, 1, high, removed);
			// drop_span of everything and one range again, as at the start
			removed.clear();
			(void)Unregister<Exclusive>(ranges, generation, Base, Size, removed);
			std::memset(low, Poison, Page);
			std::memset(high, Poison, Page);
			current       = make_alias(Size);
			current_bytes = Size;
			removed.clear();
			Register<Exclusive>(ranges, generation, Base, Size, 1, current, removed);
			rounds.fetch_add(1, std::memory_order_relaxed);
		}
	});
	std::this_thread::sleep_for(std::chrono::milliseconds(millis));
	stop.store(true);
	for (auto& thread: threads) {
		thread.join();
	}
	std::printf("HostedRangesTests: cut: %llu reads, %llu cut/free/register rounds, %llu reads through a freed alias\n",
	            static_cast<unsigned long long>(reads.load()), static_cast<unsigned long long>(rounds.load()),
	            static_cast<unsigned long long>(poisoned.load()));
	Expect(poisoned.load() == 0, "no reader follows the alias of a range that was cut");
	Expect(reads.load() > 1000 && rounds.load() > 100, "the cut stress ran");
}

} // namespace

int main() {
	TestCut();
	TestReplaceIsOneStep(400);
	TestCutFreesAlias(600);
	if (g_failures != 0) {
		std::printf("HostedRangesTests: %d FAILED\n", g_failures);
		return 1;
	}
	std::printf("HostedRangesTests: all passed\n");
	return 0;
}
