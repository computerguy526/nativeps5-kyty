#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_AUTOCAP_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_AUTOCAP_H_

// nps5 autocap: KYTY_FPS_LIMIT_MILLI=auto picks the present-side fps cap from the display the
// window is on.
//  - adaptive-refresh display (NSScreen minimumRefreshInterval < maximumRefreshInterval): no cap;
//  - fixed refresh R Hz: cap = R/n, smallest n >= 2 with R/n <= the sustained-fps estimate (n = 1,
//  cap R, only when the
//    game sustains >= R), but never below the floor (KYTY_FPS_AUTOCAP_MIN, default 30 fps): a game
//    that cannot sustain the lowest allowed R/n runs UNCAPPED (an even-pacing cap at low fps costs
//    more than it gains) and is re-checked continuously. Moves down (possibly to uncapped) after ~2
//    s below the cap, probes one step up after ~5 s (30 s for R/2 -> R) with a doubling back-off
//    (max 16x) when a probe fails within 12 s.
//  - headroom of a capped game comes from the frame work time (GPU service time of a frame,
//  Estimator::OnWork), not from
//    the presented interval, which saturates at the cap: a probe up needs work-fps >= the next cap.
// This header is pure (no SDL / no objc) so tests/AutoCapTests.cpp can compile it standalone.

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>

namespace AutoCap {

struct Screen {
	std::string name;
	double      refresh_hz = 0.0;   // 0 = unknown
	bool        adaptive   = false; // ProMotion / Adaptive-Sync (min < max refresh interval)
};

constexpr double kDefaultFloorFps = 30.0;

// Cap for a fixed R Hz display and a sustained estimate (fps): R when est >= R, else R/n for the
// smallest n >= 2 with R/n <= est. 0 = no cap (also when that cap would be below the floor).
inline double PickCap(double refresh_hz, double est_fps, double floor_fps = kDefaultFloorFps) {
	if (refresh_hz < floor_fps || refresh_hz <= 0.0) {
		return 0.0;
	}
	if (est_fps >= refresh_hz) {
		return refresh_hz;
	}
	int n = 2;
	while (refresh_hz / n > est_fps) {
		n++;
	}
	const double cap = refresh_hz / n;
	return cap >= floor_fps ? cap : 0.0;
}

// EMA of present intervals (seconds). Gaps above max_gap_s (loading stalls, pauses) are not
// samples.
class Estimator {
public:
	void Reset() {
		m_ema  = 0.0;
		m_last = -1.0;
	}
	void OnPresent(double t) {
		if (m_last >= 0.0) {
			const double dt = t - m_last;
			if (dt > 0.0 && dt <= kMaxGap) {
				m_ema = m_ema == 0.0 ? dt : m_ema + kAlpha * (dt - m_ema);
			}
		}
		m_last = t;
	}
	// 0 = no estimate yet
	double Fps() const { return m_ema > 0.0 ? 1.0 / m_ema : 0.0; }

private:
	static constexpr double kAlpha  = 0.1;
	static constexpr double kMaxGap = 0.5;
	double                  m_ema   = 0.0;
	double                  m_last  = -1.0;
};

// EMA of per-frame work times (seconds): what a frame needs from start to ready, independent of the
// cap.
class WorkEstimator {
public:
	void Reset() { m_ema = 0.0; }
	void OnWork(double s) {
		if (s > 0.0 && s <= 0.5) {
			m_ema = m_ema == 0.0 ? s : m_ema + 0.1 * (s - m_ema);
		}
	}
	double Fps() const { return m_ema > 0.0 ? 1.0 / m_ema : 0.0; }

private:
	double m_ema = 0.0;
};

class Controller {
public:
	static constexpr double kDownS      = 2.0;
	static constexpr double kUpS        = 5.0;
	static constexpr double kFullUpS    = 30.0; // R/2 -> R
	static constexpr double kSettleS    = 1.0;
	static constexpr double kProbeFailS = 12.0;

	explicit Controller(double floor_fps = kDefaultFloorFps): m_floor(floor_fps) {}

	// New screen (or display mode): restart from n = 2. Returns the new cap (0 = none).
	double SetScreen(const Screen& s, double t) {
		m_screen  = s;
		m_backoff = 1;
		m_below = m_above = -1.0;
		m_last_up         = -1e9;
		m_settle          = t + kSettleS;
		m_cap             = Initial();
		return m_cap;
	}
	double        Cap() const { return m_cap; }
	const Screen& Current() const { return m_screen; }

	// present_fps = sustained presented fps (0 = none); work_fps = 1 / frame work time (0 = none).
	// Returns the cap (0 = none); Changed() tells whether it moved.
	double Tick(double t, double present_fps, double work_fps = 0.0) {
		m_changed = false;
		if (m_screen.adaptive || m_screen.refresh_hz < m_floor || present_fps <= 0.0 ||
		    t < m_settle) {
			return m_cap;
		}
		const double R = m_screen.refresh_hz;
		if (m_cap == 0.0) {
			// uncapped: the presented rate is the game's real rate; cap once it sustains a capped
			// rate >= floor
			const double nc = PickCap(R, present_fps, m_floor);
			if (nc > 0.0) {
				if (m_above < 0.0) m_above = t;
				const double need = (nc >= R - 1e-6 ? kFullUpS : kUpS) * m_backoff;
				if (t - m_above >= need) {
					m_last_up = t;
					Apply(nc, t);
				}
			} else {
				m_above = -1.0;
			}
			return m_cap;
		}
		// capped: a miss shows in the presented rate or in the work rate
		const double cap_fps = work_fps > 0.0 ? std::fmin(present_fps, work_fps) : present_fps;
		const double n       = std::round(R / m_cap);
		const double up_cap  = n >= 2.0 ? R / (n - 1.0) : 0.0;
		if (cap_fps < 0.93 * m_cap) {
			m_above = -1.0;
			if (m_below < 0.0) m_below = t;
			if (t - m_below >= kDownS) {
				const double nc = PickCap(R, cap_fps, m_floor); // 0 = uncapped
				if (nc < m_cap - 1e-6) {
					if (t - m_last_up < kProbeFailS && m_backoff < 16) m_backoff *= 2;
					Apply(nc, t);
				} else {
					m_below = -1.0;
				}
			}
		} else if (up_cap > 0.0 && present_fps >= 0.97 * m_cap &&
		           (work_fps <= 0.0 || work_fps >= 0.97 * up_cap)) {
			m_below = -1.0;
			if (m_above < 0.0) m_above = t;
			const bool   full = std::round(R / m_cap) <= 2.0; // R/2 -> R is the long probe
			const double need = (full ? kFullUpS : kUpS) * m_backoff;
			if (t - m_above >= need) {
				m_last_up = t;
				Apply(up_cap, t);
			}
		} else {
			m_below = m_above = -1.0;
		}
		return m_cap;
	}
	bool Changed() const { return m_changed; }

private:
	double Initial() const {
		if (m_screen.adaptive || m_screen.refresh_hz <= 0.0) return 0.0;
		const double c = m_screen.refresh_hz / 2.0;
		return c >= m_floor ? c : 0.0;
	}
	void Apply(double cap, double t) {
		m_cap     = cap;
		m_changed = true;
		m_below = m_above = -1.0;
		m_settle          = t + kSettleS;
	}

	Screen m_screen;
	double m_floor = kDefaultFloorFps;
	double m_cap   = 0.0;
	double m_below = -1.0, m_above = -1.0;
	double m_last_up = -1e9;
	double m_settle  = 0.0;
	int    m_backoff = 1;
	bool   m_changed = false;
};

// ---- shared state: the window thread publishes the screen, the present thread consumes it
// -------------------------
struct Shared {
	std::mutex            mutex;
	Screen                screen;
	std::atomic<uint32_t> seq {0}; // bumped on each published change
	std::atomic<uint32_t> cap_milli {
	    0}; // effective cap (milli-fps), 0 = none; written by the present thread
};
inline Shared& State() {
	static Shared s;
	return s;
}
inline void Publish(const Screen& s) {
	auto& st = State();
	{
		std::lock_guard<std::mutex> g(st.mutex);
		if (st.seq.load() != 0 && st.screen.name == s.name && st.screen.adaptive == s.adaptive &&
		    std::fabs(st.screen.refresh_hz - s.refresh_hz) < 0.01) {
			return;
		}
		st.screen = s;
	}
	st.seq.fetch_add(1);
}
inline Screen Snapshot() {
	auto&                       st = State();
	std::lock_guard<std::mutex> g(st.mutex);
	return st.screen;
}

} // namespace AutoCap

#endif
