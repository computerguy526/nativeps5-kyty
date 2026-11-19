#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_AUTOCAP_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_AUTOCAP_H_

// nps5 autocap: KYTY_FPS_LIMIT_MILLI=auto picks the present-side fps cap from the display the window is on.
//  - adaptive-refresh display (NSScreen minimumRefreshInterval < maximumRefreshInterval): no cap;
//  - fixed refresh R Hz: cap = R/n, smallest n >= 2 with R/n <= the sustained-fps estimate (n = 1, cap R, only when the
//    game sustains >= R). Moves down after ~2 s below the cap, probes one step up after ~5 s on it (30 s for R/2 -> R),
//    with a doubling back-off (max 16x) when a probe fails within 12 s.
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

constexpr double kMinCapFps = 10.0;

// Cap for a fixed R Hz display and a sustained estimate (fps). 0 = no cap.
inline double PickCap(double refresh_hz, double est_fps) {
	if (refresh_hz < 2.0 * kMinCapFps) {
		return 0.0;
	}
	if (est_fps >= refresh_hz) {
		return refresh_hz;
	}
	int n = 2;
	while (refresh_hz / (n + 1) >= kMinCapFps && refresh_hz / n > est_fps) {
		n++;
	}
	return refresh_hz / n;
}

// EMA of present intervals (seconds). Gaps above max_gap_s (loading stalls, pauses) are not samples.
class Estimator {
public:
	void Reset() { m_ema = 0.0; m_last = -1.0; }
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

class Controller {
public:
	static constexpr double kDownS    = 2.0;
	static constexpr double kUpS      = 5.0;
	static constexpr double kFullUpS  = 30.0; // R/2 -> R
	static constexpr double kSettleS  = 1.0;
	static constexpr double kProbeFailS = 12.0;

	// New screen (or display mode): restart from n = 2. Returns the new cap (0 = none).
	double SetScreen(const Screen& s, double t) {
		m_screen   = s;
		m_backoff  = 1;
		m_below    = m_above = -1.0;
		m_last_up  = -1e9;
		m_settle   = t + kSettleS;
		m_cap      = Initial();
		return m_cap;
	}
	double Cap() const { return m_cap; }
	const Screen& Current() const { return m_screen; }

	// Feed the sustained-fps estimate (0 = none). Returns the cap (0 = none); Changed() tells whether it moved.
	double Tick(double t, double est_fps) {
		m_changed = false;
		if (m_cap == 0.0 || m_screen.adaptive || m_screen.refresh_hz <= 0.0 || est_fps <= 0.0 || t < m_settle) {
			return m_cap;
		}
		const double R = m_screen.refresh_hz;
		if (est_fps < 0.93 * m_cap) {
			m_above = -1.0;
			if (m_below < 0.0) m_below = t;
			if (t - m_below >= kDownS) {
				const double nc = PickCap(R, est_fps);
				if (nc > 0.0 && nc < m_cap - 1e-6) {
					if (t - m_last_up < kProbeFailS && m_backoff < 16) m_backoff *= 2;
					Apply(nc, t);
				} else {
					m_below = -1.0;
				}
			}
		} else if (est_fps >= 0.97 * m_cap) {
			m_below = -1.0;
			if (m_above < 0.0) m_above = t;
			const bool   full = std::round(R / m_cap) <= 2.0; // R/2 -> R is the long probe
			const double need = (full ? kFullUpS : kUpS) * m_backoff;
			if (t - m_above >= need) {
				const double n  = std::round(R / m_cap);
				const double nc = R / (n - 1.0);
				if (n >= 2.0 && nc > m_cap) {
					m_last_up = t;
					Apply(nc, t);
				} else {
					m_above = -1.0;
				}
			}
		} else {
			m_below = m_above = -1.0;
		}
		return m_cap;
	}
	bool Changed() const { return m_changed; }

private:
	double Initial() const {
		if (m_screen.adaptive || m_screen.refresh_hz < 2.0 * kMinCapFps) return 0.0;
		return m_screen.refresh_hz / 2.0;
	}
	void Apply(double cap, double t) {
		m_cap     = cap;
		m_changed = true;
		m_below   = m_above = -1.0;
		m_settle  = t + kSettleS;
	}

	Screen m_screen;
	double m_cap      = 0.0;
	double m_below    = -1.0, m_above = -1.0;
	double m_last_up  = -1e9;
	double m_settle   = 0.0;
	int    m_backoff  = 1;
	bool   m_changed  = false;
};

// ---- shared state: the window thread publishes the screen, the present thread consumes it -------------------------
struct Shared {
	std::mutex           mutex;
	Screen               screen;
	std::atomic<uint32_t> seq {0};   // bumped on each published change
	std::atomic<uint32_t> cap_milli {0}; // effective cap (milli-fps), 0 = none; written by the present thread
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
	auto& st = State();
	std::lock_guard<std::mutex> g(st.mutex);
	return st.screen;
}

} // namespace AutoCap

#endif
