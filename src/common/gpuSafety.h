#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>

// nps5 gpusafe: host-side half of the GPU watchdog. KosmicKrisp (the Vulkan driver) owns the watchdog
// itself (a committed Metal command buffer incomplete after KYTY_GPU_WATCHDOG_S seconds, default 4,
// 0 = off -> stops submissions, prints its flight summary, sends SIGUSR2, _exit(70)). This module
// keeps a lock-free ring of the last native submissions (tick, guest submit id, debug op/args) and
// prints it from the SIGUSR2 handler, so the hang can be tied to a guest submission.
// Cost per submission: one relaxed fetch_add and five relaxed stores. No allocation, no formatting.
namespace GpuSafety {

constexpr size_t RingSize = 64; // power of two

struct Entry {
	std::atomic<uint64_t> tick {0}, submit {0}, args {0}, ns {0}, seq {0};
};

extern std::atomic<int>      g_state; // 0 = not initialised, 1 = on, 2 = off
extern std::atomic<uint64_t> g_head;
extern Entry                 g_ring[RingSize];

void InitSlow() noexcept; // reads KYTY_GPU_WATCHDOG_S, installs the SIGUSR2 handler when on
uint64_t NowNs() noexcept;

// seconds < 0 or unparsable -> default 4; "0" -> off.
[[nodiscard]] bool WatchdogEnabledFromEnv(const char* value) noexcept;

inline void NoteSubmit(uint64_t tick, uint64_t submit, uint32_t op, uint32_t arg0) noexcept {
	int s = g_state.load(std::memory_order_relaxed);
	if (s == 0) {
		InitSlow();
		s = g_state.load(std::memory_order_relaxed);
	}
	if (s != 1) {
		return;
	}
	const uint64_t n = g_head.fetch_add(1, std::memory_order_relaxed) + 1;
	Entry&         e = g_ring[n & (RingSize - 1)];
	e.tick.store(tick, std::memory_order_relaxed);
	e.submit.store(submit, std::memory_order_relaxed);
	e.args.store((static_cast<uint64_t>(op) << 32) | arg0, std::memory_order_relaxed);
	e.ns.store(NowNs(), std::memory_order_relaxed);
	e.seq.store(n, std::memory_order_release);
}

// Newest first, at most `max` entries, into buf (NUL terminated); returns the length. Async-signal
// tolerant: no allocation (snprintf of integers only).
size_t FormatRing(char* buf, size_t cap, size_t max) noexcept;

} // namespace GpuSafety
