#ifndef EMULATOR_SRC_COMMON_LATTRACE_H_
#define EMULATOR_SRC_COMMON_LATTRACE_H_

#include <chrono>
#include <cstdio>
#include <cstdlib>

// nps5: KYTY_LAT_TRACE=1 prints "[klat] <stage> t=<ms> n=<id>" on the host's steady clock (the
// clock of AnyPS5's APS5_TRACE_LAT "[lat]" lines) for the stages Kyty owns: flip (the guest's flip
// is reserved = the guest finished that frame's commands), present (vkQueuePresentKHR returned) and
// pad (the input relay published a state). Off by default, one cached getenv.
namespace Common::LatTrace {

inline bool Enabled() {
	static const bool on = std::getenv("KYTY_LAT_TRACE") != nullptr;
	return on;
}

inline void Log(const char* stage, long long n = -1) {
	if (!Enabled()) {
		return;
	}
	const double t = std::chrono::duration<double, std::milli>(
	                     std::chrono::steady_clock::now().time_since_epoch())
	                     .count();
	std::fprintf(stderr, "[klat] %s t=%.2f n=%lld\n", stage, t, n);
}

} // namespace Common::LatTrace

#endif
