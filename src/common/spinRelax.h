#ifndef EMULATOR_SRC_COMMON_SPINRELAX_H_
#define EMULATOR_SRC_COMMON_SPINRELAX_H_

#include <atomic>
#include <thread>

#if defined(_M_X64) || defined(__x86_64__)
#include <immintrin.h>
#endif

// nps5 (kcpu): the one "relax" step of every bounded spin-wait (draw-prep workers and the
// producer's head wait, the CP sequencer and resolver, the stage-prep helper, the scheduler's
// priority wait, YieldAndPauseMicro).
//
// On x86 that step is _mm_pause: a few nanoseconds on the CPU, so a "10 us spin, look at the clock
// every 256 steps" costs what it says. The arm64 build had std::this_thread::yield() there instead
// (sched_yield: a system call, swtch_pri on macOS, about a microsecond when nothing else is
// runnable and a whole scheduling quantum when something is). A 10 us spin was then at least 256
// system calls long, the spinning thread kept giving its core away on a busy machine, and the
// spins showed as the largest CPU consumer of a level (swtch_pri in the host profile).
//
// KYTY_SPIN_RELAX (live switch, common/liveSwitch.h; defined in threads.cpp), arm64 only:
//   yield (or 0, default until measured): std::this_thread::yield(), the previous behaviour;
//   pause (or 1): the CPU's own spin hint (isb, the arm64 counterpart of pause: `yield` is a no-op
//                 on Apple cores), no system call.
// x86 builds always use _mm_pause, other targets always yield.
namespace Common {

// 0 = sched_yield, 1 = CPU pause. Written by the KYTY_SPIN_RELAX switch's parser (threads.cpp); an
// inline variable so that header-only users (the draw-prep unit tests) need no library.
inline std::atomic<int> g_spin_relax_pause {0};

inline void SpinRelax() noexcept {
#if defined(_M_X64) || defined(__x86_64__)
	_mm_pause();
#elif defined(__aarch64__)
	if (g_spin_relax_pause.load(std::memory_order_relaxed) != 0) {
		__asm__ volatile("isb");
	} else {
		std::this_thread::yield();
	}
#else
	std::this_thread::yield();
#endif
}

} // namespace Common

#endif // EMULATOR_SRC_COMMON_SPINRELAX_H_
