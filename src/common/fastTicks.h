#ifndef EMULATOR_SRC_COMMON_FASTTICKS_H_
#define EMULATOR_SRC_COMMON_FASTTICKS_H_

#include <cstdint>

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__x86_64__)
#include <x86intrin.h>
#elif !defined(__aarch64__)
#include <chrono>
#endif

// nps5 cpcommit: the tick source of the command processor's per-draw statistics (CpGaps,
// CommitStats, BufferLookupStats). Every user calibrates ticks against the steady clock, so the
// unit is free. x86: the TSC, as before. arm64: the virtual counter (CNTVCT_EL0, user readable
// on macOS and Linux; 24 MHz on Apple silicon, the resolution of mach_absolute_time and so of
// std::chrono::steady_clock there). The previous arm64 fallback was steady_clock::now(), about
// 14 ns per read against 0.3 ns: with KYTY_CP_GAP_STATS=1 KYTY_CP_PASS_STATS=1 (the canonical
// runner) a committed draw reads the clock about 14 times and every other CP op twice.
namespace Common {

[[nodiscard]] inline uint64_t FastTicks() noexcept {
#if defined(_M_X64) || defined(__x86_64__)
	return __rdtsc();
#elif defined(__aarch64__) && (defined(__clang__) || defined(__GNUC__))
	uint64_t ticks;
	__asm__ volatile("mrs %0, cntvct_el0" : "=r"(ticks));
	return ticks;
#else
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
#endif
}

} // namespace Common

#endif // EMULATOR_SRC_COMMON_FASTTICKS_H_
