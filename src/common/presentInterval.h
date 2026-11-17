#ifndef EMULATOR_SRC_COMMON_PRESENTINTERVAL_H_
#define EMULATOR_SRC_COMMON_PRESENTINTERVAL_H_

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <vector>

// nps5 meas: KYTY_PRESENT_INTERVAL_LOG=<file> (default off; one cached getenv when off): one line per ACTUAL present
// (vkQueuePresentKHR returned) "t_us,interval_us" on the steady clock, plus a stats line to stderr every
// KYTY_PRESENT_INTERVAL_STATS_S (default 10) seconds:
//   [present-interval] n=<presents> p50/p95/p99=<ms> mean=<ms> stdev=<ms> judder=<ms> max=<ms>
// judder = mean |interval_i - interval_{i-1}| (frame-to-frame change). KYTY_FRAME_TIMES_LOG is the guest FLIP interval
// (time between guest flips, not on-screen presents).
namespace Common::PresentInterval {

inline const char* Path() {
	static const char* p = [] { const auto* v = std::getenv("KYTY_PRESENT_INTERVAL_LOG"); return v != nullptr && v[0] != 0 ? v : nullptr; }();
	return p;
}

inline void Note() {
	if (Path() == nullptr) {
		return;
	}
	struct State {
		std::mutex mutex;
		FILE*      file = nullptr;
		int64_t    previous_us = 0, window_start_us = 0, period_us = 10000000;
		std::vector<double> iv;
		State() {
			file = std::fopen(Path(), "w");
			if (const auto* e = std::getenv("KYTY_PRESENT_INTERVAL_STATS_S")) { const auto v = std::strtol(e, nullptr, 10); if (v > 0) period_us = v * 1000000; }
		}
	};
	static State s;
	const int64_t now = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
	std::lock_guard lock(s.mutex);
	if (s.window_start_us == 0) s.window_start_us = now;
	if (s.previous_us != 0) {
		const int64_t d = now - s.previous_us;
		if (s.file != nullptr) { std::fprintf(s.file, "%lld,%lld\n", static_cast<long long>(now), static_cast<long long>(d)); }
		s.iv.push_back(static_cast<double>(d) / 1000.0);
	}
	s.previous_us = now;
	if (now - s.window_start_us >= s.period_us) {
		if (s.file != nullptr) std::fflush(s.file);
		auto& v = s.iv;
		if (v.size() >= 2) {
			double sum = 0, sq = 0, jd = 0;
			for (size_t i = 0; i < v.size(); i++) { sum += v[i]; if (i) jd += std::fabs(v[i] - v[i - 1]); }
			const double mean = sum / static_cast<double>(v.size());
			for (const auto x : v) sq += (x - mean) * (x - mean);
			std::vector<double> o = v; std::sort(o.begin(), o.end());
			const auto q = [&](double p) { return o[std::min(o.size() - 1, static_cast<size_t>(p * static_cast<double>(o.size())))]; };
			std::fprintf(stderr, "[present-interval] n=%zu p50=%.2f p95=%.2f p99=%.2f mean=%.2f stdev=%.2f judder=%.2f max=%.2f\n", v.size(), q(0.5), q(0.95), q(0.99),
			             mean, std::sqrt(sq / static_cast<double>(v.size())), jd / static_cast<double>(v.size() - 1), o.back());
		}
		v.clear(); s.window_start_us = now;
	}
}

} // namespace Common::PresentInterval

#endif
