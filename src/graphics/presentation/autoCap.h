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

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <vector>

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

// ---- ProMotion mode (opt-in, KYTY_FPS_AUTOCAP=promotion) --------------------------------------
// An adaptive-refresh display (ProMotion, max R = 120 Hz) normally gets no cap. Here the cap is the
// highest R/n (n = 2..5: 60 / 40 / 30 / 24 at 120 Hz) the scene SUSTAINS, so flips land on whole
// display refreshes (even pacing) while the game cannot hold 60.
//  - down (fast): p90 of the presented intervals over the last kDownWindowS (3 s) > kMissRatio
//  (1.15)
//    x the cap's period; the new cap is the highest R/n whose period x 1.05 covers that p90 (floor
//    n = 5). A burst of slow frames (loading) has to fill more than 10 % of the window to count,
//    and the cap is re-probed upwards afterwards, so a spike never pins it.
//  - up (slow): p90 of the frame WORK time over the last kUpWindowS (10 s, no change inside it) <
//    kHeadroom (0.9) x the period of the higher cap, and the intervals held the current cap (p90 <=
//    1.05 x period). A probe that fails (a step down within kProbeFailS = 30 s of the step up)
//    doubles the wait for the next probe (max 8x), which bounds flip-flop at a boundary.
//  - intervals above kMaxGap (0.25 s: load stalls, pauses) are not samples.
// Pure: OnPresent / OnWork / Tick take a clock (seconds), so tests drive synthetic traces.
class PromoCap {
public:
	static constexpr int    kMinN        = 2;
	static constexpr int    kMaxN        = 5;
	static constexpr int    kStartN      = 4; // 30 at 120 Hz
	static constexpr double kDownWindowS = 3.0;
	static constexpr double kUpWindowS   = 10.0;
	static constexpr double kMissRatio   = 1.15;
	static constexpr double kHeadroom    = 0.9;
	static constexpr double kHoldRatio   = 1.05;
	static constexpr double kProbeFailS  = 30.0;
	static constexpr double kMaxGap      = 0.25;
	static constexpr size_t kMinSamples  = 20;
	static constexpr int    kMaxBackoff  = 8;

	explicit PromoCap(double refresh_hz = 120.0) { SetRefresh(refresh_hz, 0.0); }

	void SetRefresh(double refresh_hz, double t) {
		m_r            = refresh_hz;
		m_n            = kStartN;
		m_backoff      = 1;
		m_last_up      = -1e9;
		m_last_change  = t;
		m_last_present = -1.0;
		Clear(t);
		m_changed = false;
	}
	double Cap() const { return m_r / m_n; }
	int    N() const { return m_n; }
	bool   Changed() const { return m_changed; }
	double LastP90Interval() const { return m_p90_i; }
	double LastP90Work() const { return m_p90_w; }

	void OnPresent(double t) {
		if (m_last_present >= 0.0) {
			const double dt = t - m_last_present;
			if (dt > 0.0 && dt <= kMaxGap) Push(m_iv, t, dt);
		}
		m_last_present = t;
	}
	void OnWork(double t, double work_s) {
		if (work_s > 0.0 && work_s <= kMaxGap) Push(m_wk, t, work_s);
	}

	// Returns the cap (fps). Changed() tells whether it moved on this call.
	double Tick(double t) {
		m_changed = false;
		if (t < m_settle || t < m_next_eval) return Cap();
		m_next_eval = t + 0.1;
		if (t - m_last_change > 60.0 && m_backoff > 1) m_backoff = 1;
		const double P  = static_cast<double>(m_n) / m_r;
		const double pd = Pct90(m_iv, t, kDownWindowS, m_cnt_d);
		if (m_cnt_d >= kMinSamples && pd > kMissRatio * P) {
			m_p90_i = pd;
			int n   = m_n;
			while (n < kMaxN && (n / m_r) * kHoldRatio < pd)
				n++;
			if (n != m_n) {
				if (t - m_last_up < kProbeFailS && m_backoff < kMaxBackoff) m_backoff *= 2;
				Apply(n, t, false);
			}
			return Cap();
		}
		if (m_n > kMinN && t - m_last_change >= kUpWindowS * m_backoff) {
			const double pi = Pct90(m_iv, t, kUpWindowS, m_cnt_u);
			const double pw = Pct90(m_wk, t, kUpWindowS, m_cnt_w);
			if (m_cnt_u >= kMinSamples * 3 && m_cnt_w >= 10 && pi <= kHoldRatio * P) {
				int n = m_n;
				while (n > kMinN && pw < kHeadroom * ((n - 1) / m_r))
					n--;
				if (n != m_n) {
					m_p90_i = pi;
					m_p90_w = pw;
					Apply(n, t, true);
				}
			}
		}
		return Cap();
	}

private:
	struct S {
		double t, v;
	};
	void Push(std::deque<S>& q, double t, double v) {
		q.push_back({t, v});
		while (!q.empty() && t - q.front().t > kUpWindowS)
			q.pop_front();
	}
	static double Pct90(const std::deque<S>& q, double t, double win, size_t& count) {
		std::vector<double> v;
		for (auto it = q.rbegin(); it != q.rend() && t - it->t <= win; ++it)
			v.push_back(it->v);
		count = v.size();
		if (v.empty()) return 0.0;
		const size_t k = static_cast<size_t>(0.9 * static_cast<double>(v.size() - 1) + 0.5);
		std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(k), v.end());
		return v[k];
	}
	void Clear(double t) {
		m_iv.clear();
		m_wk.clear();
		m_settle = t + 1.0;
	}
	void Apply(int n, double t, bool up) {
		m_n           = n;
		m_changed     = true;
		m_last_change = t;
		if (up) m_last_up = t;
		Clear(t);
	}

	double        m_r       = 120.0;
	int           m_n       = kStartN;
	int           m_backoff = 1;
	double        m_last_up = -1e9, m_last_change = -1e9, m_last_present = -1.0, m_settle = 0.0,
	              m_next_eval = 0.0;
	double        m_p90_i = 0.0, m_p90_w = 0.0;
	size_t        m_cnt_d = 0, m_cnt_u = 0, m_cnt_w = 0;
	bool          m_changed = false;
	std::deque<S> m_iv, m_wk;
};

// KYTY_FPS_AUTOCAP=promotion (or NPS5_AUTOCAP_PROMOTION=1)
inline bool PromotionEnabled() {
	const char* a = std::getenv("KYTY_FPS_AUTOCAP");
	const char* b = std::getenv("NPS5_AUTOCAP_PROMOTION");
	return (a != nullptr && std::strcmp(a, "promotion") == 0) ||
	       (b != nullptr && std::strcmp(b, "1") == 0);
}

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
