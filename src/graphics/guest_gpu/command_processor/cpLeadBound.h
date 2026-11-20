#ifndef EMULATOR_SRC_GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_CPLEADBOUND_H_
#define EMULATOR_SRC_GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_CPLEADBOUND_H_

// nps5 cpusurvey: KYTY_CP_LEAD_MAX_MS (live switch, default 0 = off, 1..200).
//
// The command processor (CP) records faster than the GPU executes, so in a GPU-limited scene it
// runs ahead: the newest submitted recording completes 20-25 ms after it was recorded (Sky
// Garden, [cbtime] shift). Every guest thread that faults on a GPU-written page waits for the
// GPU to reach the page's writer, i.e. for about that lead, and a write fault drains the CP for
// it. The lead itself buys nothing once the GPU has work queued: the CP spends the time saved
// idle (CpGaps idle 6-21 ms per frame).
//
// With the switch on, a submit made by the CP (CommandProcessor::BufferFlush) waits until no
// submitted, incomplete recording is older than the bound. The wait is the one a readback drain
// makes (MasterSemaphore::Wait on an earlier tick): it changes when the CP continues, never what
// it records. Recordings younger than the bound stay queued, so the GPU keeps work while the CP
// waits.
//
// This header is the bookkeeping only (no Kyty types), so that it can be unit-tested alone
// (tests/CpLeadBoundTests.cpp). GPU thread only: no locking.

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

namespace Libs::Graphics::CpLeadBound {

// "0", unset, junk, signs = off; 1..200 = the bound in ms (larger values clamp to 200).
[[nodiscard]] inline int64_t ParseMs(const char* value) noexcept {
	if (value == nullptr || value[0] < '0' || value[0] > '9') {
		return 0;
	}
	char*      end    = nullptr;
	const auto parsed = std::strtoull(value, &end, 10);
	if (end == nullptr || *end != '\0') {
		return 0;
	}
	return static_cast<int64_t>(parsed > 200u ? 200u : parsed);
}

class Tracker {
public:
	static constexpr size_t kCapacity = 256;

	// The CP submitted the recording `tick` at `now_ns`. Ticks arrive in increasing order. When the
	// ring is full the oldest entry is forgotten (its recording is then not waited for).
	void Submitted(uint64_t tick, uint64_t now_ns) noexcept {
		if (m_count == kCapacity) {
			Pop();
		}
		m_entries[(m_head + m_count) % kCapacity] = {tick, now_ns};
		m_count++;
	}

	// The GPU has completed every tick <= gpu_tick.
	void Completed(uint64_t gpu_tick) noexcept {
		while (m_count != 0 && m_entries[m_head].tick <= gpu_tick) {
			Pop();
		}
	}

	// The tick to wait for so that no tracked recording submitted more than limit_ns ago is still
	// incomplete: the newest such recording (its completion implies the older ones'). 0 = none.
	// The recording submitted last is never returned: with one recording in flight there is
	// nothing queued behind it, and waiting for it would leave the GPU without work.
	[[nodiscard]] uint64_t WaitTick(uint64_t now_ns, uint64_t limit_ns) const noexcept {
		uint64_t tick = 0;
		for (size_t i = 0; i + 1 < m_count; i++) {
			const auto& entry = m_entries[(m_head + i) % kCapacity];
			if (now_ns - entry.ns <= limit_ns) {
				break;
			}
			tick = entry.tick;
		}
		return tick;
	}

	[[nodiscard]] size_t Size() const noexcept { return m_count; }
	void                 Clear() noexcept {
		m_head  = 0;
		m_count = 0;
	}

private:
	struct Entry {
		uint64_t tick = 0;
		uint64_t ns   = 0;
	};
	void Pop() noexcept {
		m_head = (m_head + 1) % kCapacity;
		m_count--;
	}
	std::array<Entry, kCapacity> m_entries {};
	size_t                       m_head  = 0;
	size_t                       m_count = 0;
};

} // namespace Libs::Graphics::CpLeadBound

#endif // EMULATOR_SRC_GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_CPLEADBOUND_H_
