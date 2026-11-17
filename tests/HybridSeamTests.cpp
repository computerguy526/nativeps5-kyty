// nps5: the header-only parts of the hybrid bridge (hybrid/pendingMemory.h, hybrid/mainLoopGate.h).
#include "hybrid/mainLoopGate.h"
#include "hybrid/pendingMemory.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

namespace {

int g_failures = 0;

void Expect(bool condition, const char* what) {
	if (!condition) {
		std::printf("HybridSeamTests: failed: %s\n", what);
		g_failures++;
	}
}

using Hybrid::PendingMemory;
using Kind = PendingMemory::Kind;

void TestPendingMemory() {
	PendingMemory pending;
	Expect(pending.Queue({Kind::Register, 0x1000, 0x4000, 0x33, nullptr}), "register is queued before the start");
	Expect(pending.Covers(0x1000, 0x4000) && pending.Covers(0x2000, 0x1000), "queued registration covers");
	Expect(!pending.Covers(0x0, 0x2000) && !pending.Covers(0x4000, 0x2000), "partly outside is not covered");
	// The guest changed the protection before the renderer started: it must reach the tracker, in order.
	Expect(pending.Queue({Kind::Protect, 0x2000, 0x1000, 0x11, nullptr}), "protect is queued before the start");
	Expect(pending.Covers(0x1000, 0x4000), "a protection change does not end the cover");
	Expect(pending.Queue({Kind::Unregister, 0x3000, 0x1000, 0, nullptr}), "unregister is queued");
	Expect(!pending.Covers(0x1000, 0x4000), "the last note that touches the range decides");
	std::string order;
	uint32_t    protect_flags = 0;
	const auto  count = pending.Drain([&](const PendingMemory::Note& note) {
        order += note.kind == Kind::Register ? 'R' : note.kind == Kind::Unregister ? 'U' : 'P';
        if (note.kind == Kind::Protect) {
            protect_flags = note.flags;
        }
    });
	Expect(order == "RPU", "notes are applied in arrival order, the protection change included");
	Expect(count == 3 && protect_flags == 0x11, "the protection change keeps its flags");
	Expect(!pending.Queue({Kind::Register, 0x9000, 0x1000, 1, nullptr}), "after the start the caller applies the note");
	Expect(!pending.Queue({Kind::Protect, 0x9000, 0x1000, 1, nullptr}), "after the start the caller applies a protect");
	Expect(!pending.Covers(0x1000, 0x1000), "nothing is pending after the start");
	// Overflowing ranges never cover.
	PendingMemory wide;
	(void)wide.Queue({Kind::Register, UINT64_MAX - 0x1000, 0x1000, 1, nullptr});
	Expect(!wide.Covers(UINT64_MAX - 0x800, 0x1000), "a range that wraps is not covered");
}

void TestSelfMemLimit() {
	using Hybrid::SelfMemLimitMb;
	Expect(SelfMemLimitMb(nullptr) == 18944 && SelfMemLimitMb("") == 18944, "default limit");
	Expect(SelfMemLimitMb("0") == 0, "0 turns the limit off");
	Expect(SelfMemLimitMb("12000") == 12000, "a number is taken");
	// The limit protects the machine: text that is not a number must not turn it off.
	Expect(SelfMemLimitMb("abc") == 18944, "unparsable text keeps the default limit");
	Expect(SelfMemLimitMb("12000MB") == 18944, "trailing text keeps the default limit");
	Expect(SelfMemLimitMb("-5") == 18944, "a negative value keeps the default limit");
	Expect(SelfMemLimitMb(" 16000") == 16000, "leading space is fine");
	Expect(SelfMemLimitMb("99999999999999999999") == 18944, "an out-of-range number keeps the default limit");
}

using Hybrid::MainLoopGate;
using State = MainLoopGate::State;

void TestGate() {
	using namespace std::chrono_literals;
	{
		MainLoopGate gate;
		Expect(gate.WaitEnded(1ms) == State::Idle, "no loop started: nothing to wait for");
		Expect(!gate.TryBeginEnd(), "a loop that never started cannot end");
	}
	{
		// The ordinary shutdown: the loop thread ends after the quit event; shutdown returns only then.
		MainLoopGate      gate;
		std::atomic<bool> quit {false}, torn_down {false};
		gate.Start();
		std::thread loop([&] {
			while (gate.Running() && !quit.load()) {
				std::this_thread::sleep_for(1ms);
			}
			if (gate.TryBeginEnd()) {
				std::this_thread::sleep_for(30ms); // the teardown
				torn_down.store(true);
				gate.FinishEnd();
			}
		});
		quit.store(true);
		const auto state = gate.WaitEnded(5000ms);
		Expect(state == State::Ended && torn_down.load(), "shutdown returns after the loop's teardown");
		Expect(!gate.Running() && !gate.TryBeginEnd(), "an ended loop does not end twice");
		loop.join();
	}
	{
		// A teardown that takes longer than the grace period is waited for (it has started).
		MainLoopGate      gate;
		std::atomic<bool> torn_down {false};
		gate.Start();
		std::thread loop([&] {
			if (gate.TryBeginEnd()) {
				std::this_thread::sleep_for(150ms);
				torn_down.store(true);
				gate.FinishEnd();
			}
		});
		const auto state = gate.WaitEnded(10ms);
		Expect(state == State::Ended && torn_down.load(), "a running teardown is waited for beyond the grace period");
		loop.join();
	}
	{
		// The main thread does not serve its queue any more: shutdown gives up and the loop may not tear down later.
		MainLoopGate gate;
		gate.Start();
		const auto started = std::chrono::steady_clock::now();
		Expect(gate.WaitEnded(20ms) == State::Abandoned, "a loop that does not end is abandoned");
		Expect(std::chrono::steady_clock::now() - started < 2000ms, "the wait is bounded");
		Expect(!gate.Running(), "an abandoned loop does not take another step");
		Expect(!gate.TryBeginEnd(), "an abandoned loop is not torn down behind the exiting process");
	}
	// Race: the loop ends while shutdown gives up. Exactly one of them wins; never a teardown after 'abandoned'.
	uint64_t ended = 0, abandoned = 0;
	for (int i = 0; i < 2000; i++) {
		MainLoopGate      gate;
		std::atomic<bool> torn_down {false};
		gate.Start();
		std::thread loop([&] {
			if (i % 3 != 0) {
				std::this_thread::yield();
			}
			if (gate.TryBeginEnd()) {
				torn_down.store(true);
				gate.FinishEnd();
			}
		});
		const auto state = gate.WaitEnded(0ms);
		loop.join();
		if (state == State::Ended) {
			ended++;
			Expect(torn_down.load(), "race: ended means torn down");
		} else {
			abandoned++;
			Expect(state == State::Abandoned && !torn_down.load(), "race: abandoned means no teardown, then or later");
		}
	}
	std::printf("HybridSeamTests: gate race: %llu ended, %llu abandoned\n", static_cast<unsigned long long>(ended),
	            static_cast<unsigned long long>(abandoned));
}

} // namespace

int main() {
	TestPendingMemory();
	TestSelfMemLimit();
	TestGate();
	if (g_failures != 0) {
		std::printf("HybridSeamTests: %d FAILED\n", g_failures);
		return 1;
	}
	std::printf("HybridSeamTests: all passed\n");
	return 0;
}
