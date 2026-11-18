#pragma once

#include "common/flightRec.h"
#include <atomic>
#include <cstddef>
#include <cstdint>

// nps5 gpusafe: host-side half of the GPU watchdog. KosmicKrisp (the Vulkan driver) owns the watchdog
// itself (a committed Metal command buffer incomplete after KYTY_GPU_WATCHDOG_S seconds, default 4,
// 0 = off -> stops submissions, prints its flight summary, sends SIGUSR2, _exit(70)). This module
// keeps a lock-free ring of the last native submissions (tick, guest submit id, debug op/args) and
// prints it from the SIGUSR2 handler, so the hang can be tied to a guest submission.
// Cost per submission: one relaxed fetch_add and six release stores. No allocation, no formatting.
namespace GpuSafety {

constexpr size_t RingSize = 64; // power of two

struct Entry {
	std::atomic<uint64_t> tick {0}, submit {0}, args {0}, ns {0}, seq {0};
};

extern std::atomic<int>      g_state; // 0 = not initialised, 1 = on, 2 = off, 3 = being initialised
extern std::atomic<uint64_t> g_head;
extern Entry                 g_ring[RingSize];

// Reads KYTY_GPU_WATCHDOG_S and installs the SIGUSR2 handler when on, once (the first caller; submissions
// that race with it are not noted). The handler prints the ring and then calls the handler that was
// installed before it (the host process's own), with SA_RESTART so that a SIGUSR2 from anywhere does not
// fail the host's system calls with EINTR.
void InitSlow() noexcept;
uint64_t NowNs() noexcept;

// seconds < 0 or unparsable -> default 4; "0" -> off.
[[nodiscard]] bool WatchdogEnabledFromEnv(const char* value) noexcept;

inline void NoteSubmit(uint64_t tick, uint64_t submit, uint32_t op, uint32_t arg0) noexcept {
	FlightRec::Rec(FlightRec::EvNativeSubmit, tick, submit, op, arg0);
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
	// A seqlock for the SIGUSR2 reader: not a complete entry while it is written (FormatRing reads seq
	// before and after the fields). Release stores and acquire loads throughout: plain ones may pass the
	// seq accesses on arm64 and a line would mix two submissions.
	e.seq.store(0, std::memory_order_release);
	e.tick.store(tick, std::memory_order_release);
	e.submit.store(submit, std::memory_order_release);
	e.args.store((static_cast<uint64_t>(op) << 32) | arg0, std::memory_order_release);
	e.ns.store(NowNs(), std::memory_order_release);
	e.seq.store(n, std::memory_order_release);
}

// Newest first, at most `max` entries, into buf (NUL terminated); returns the length. Async-signal
// tolerant: no allocation (snprintf of integers only).
size_t FormatRing(char* buf, size_t cap, size_t max) noexcept;

} // namespace GpuSafety
