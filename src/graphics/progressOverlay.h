#pragma once

// On-screen shader-compile progress (nps5 hosted Kyty): state and animation timing, header only so
// the unit test (tests/ProgressOverlayTests.cpp) needs no emulator or GPU.
//
// Two looks, both drawn by SystemOverlay (graphics/presentation/systemOverlay.cpp) from this state:
//  - Card: full screen, from the moment the window presents until the first real game frame (the
//    replay shows the bar; afterwards "LOADING" with a sweeping bar). Dark panel, spinner, "COMPILING SHADERS", amber rounded bar, "N / M",
//    percent and ETA (the look of the AnyPS5 prebuild screen, ProgressScreen.cpp).
//  - Badge: small corner pill with a spinner during gameplay while live compiles are in flight or a
//    draw waits. It appears only after kBadgeDelayMs of continuous need (no flicker for short
//    compiles), stays at least kBadgeHoldMs and fades out when idle.
// The machine is stepped only from the present thread (Step) and allocates nothing.
// KYTY_PROGRESS_OVERLAY=0 turns the overlay off; KYTY_PROGRESS_OVERLAY_LOG=1 logs state changes.

#include "graphics/compileProgress.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace Libs::Graphics::ProgressOverlay {

inline constexpr uint64_t kBadgeDelayMs  = 300;  // continuous need before the badge shows
inline constexpr uint64_t kBadgeHoldMs   = 1000; // minimum time on screen once shown (and after the last compile ends)
inline constexpr uint64_t kCardHoldMs    = 250;  // card lingers after the work ends
inline constexpr uint64_t kFadeInMs      = 150;
inline constexpr uint64_t kBadgeFadeMs   = 400;
inline constexpr uint64_t kCardFadeMs    = 300;
inline constexpr uint64_t kSpinnerMs     = 900;  // one revolution
inline constexpr double   kBarTauMs      = 160.0;  // bar easing time constant
inline constexpr uint64_t kEtaMinMs      = 1500; // no ETA before this much replay time
inline constexpr uint64_t kMaxStepMs     = 250;  // a longer gap counts as this much (fade timing)

enum class Mode : uint8_t { Hidden, Card, Badge };

inline const char* ModeName(Mode m) { return m == Mode::Card ? "card" : m == Mode::Badge ? "badge" : "hidden"; }

inline bool Enabled() {
	static const bool enabled = [] {
		const char* v = std::getenv("KYTY_PROGRESS_OVERLAY");
		return v == nullptr || v[0] != '0';
	}();
	return enabled;
}

inline bool LogEnabled() {
	static const bool enabled = [] {
		const char* v = std::getenv("KYTY_PROGRESS_OVERLAY_LOG");
		return v != nullptr && v[0] == '1';
	}();
	return enabled;
}

// KYTY_PROGRESS_OVERLAY_DEMO=1 (diagnostic): for the first kDemoMs after the overlay is first asked, a synthetic
// replay (0..kDemoTotal) and "no game frame yet" are fed to the overlay, so the card can be seen and logged on a warm run.
inline bool DemoEnabled() {
	static const bool enabled = [] {
		const char* v = std::getenv("KYTY_PROGRESS_OVERLAY_DEMO");
		return v != nullptr && v[0] == '1';
	}();
	return enabled;
}
// KYTY_PROGRESS_OVERLAY_IN_DUMPS=1 (diagnostic): do not keep the overlay off KYTY_DUMP_FRAMES frames.
inline bool InDumps() {
	static const bool enabled = [] {
		const char* v = std::getenv("KYTY_PROGRESS_OVERLAY_IN_DUMPS");
		return v != nullptr && v[0] == '1';
	}();
	return enabled;
}
inline constexpr uint64_t kDemoMs = 12000, kDemoTotal = 400;
// Applies the demo to a snapshot; returns true while the demo overrides (then the caller treats the game frame as unseen).
inline bool ApplyDemo(uint64_t now_ms, CompileProgress::Snapshot& s) {
	static uint64_t start = 0;
	if (!DemoEnabled()) return false;
	if (start == 0) start = now_ms;
	const uint64_t t = now_ms - start;
	if (t >= kDemoMs) return false;
	s.replay_total = kDemoTotal;
	s.replay_done  = t * kDemoTotal / kDemoMs;
	return true;
}

// Set by the presenter when a real (non blank) main-bus guest frame is presented.
inline std::atomic<bool>& GameFrameSeen() {
	static std::atomic<bool> seen {false};
	return seen;
}

inline bool ReplayActive(const CompileProgress::Snapshot& s) { return s.replay_total != 0 && s.replay_done < s.replay_total; }

inline bool LiveWork(const CompileProgress::Snapshot& s) { return s.live > 0 || s.waiting > 0; }

// What the snapshot asks for right now, before any timing: Card, Badge or Hidden.
inline Mode Wanted(const CompileProgress::Snapshot& s, bool game_frame) {
	if (!game_frame) return Mode::Card; // from the first overlay paint until the first real game frame
	return LiveWork(s) ? Mode::Badge : Mode::Hidden;
}

struct View {
	Mode     mode        = Mode::Hidden;
	float    alpha       = 0.0f;  // 0..1 of the shown mode (fades in/out)
	float    bar         = 0.0f;  // eased fraction 0..1 (card)
	float    spinner     = 0.0f;  // turns 0..1
	uint64_t done        = 0, total = 0;  // replay counters (card)
	int32_t  live        = 0;
	bool     waiting     = false;
	double   eta_seconds = -1.0;  // < 0 unknown
};

inline std::size_t FormatEta(char* out, std::size_t size, double seconds) {
	if (!(seconds >= 0.0) || seconds > 99.0 * 3600.0) return static_cast<std::size_t>(std::snprintf(out, size, "ETA --"));
	const auto total = static_cast<long>(std::lround(seconds));
	if (total >= 3600) return static_cast<std::size_t>(std::snprintf(out, size, "ETA %ldH %02ldM", total / 3600, (total / 60) % 60));
	if (total >= 60) return static_cast<std::size_t>(std::snprintf(out, size, "ETA %ldM %02ldS", total / 60, total % 60));
	return static_cast<std::size_t>(std::snprintf(out, size, "ETA %ldS", total));
}

class Machine {
public:
	// Advances the state to `now_ms` (monotonic ms) and returns the view to draw.
	View Step(uint64_t now_ms, const CompileProgress::Snapshot& s, bool game_frame) {
		uint64_t dt = m_have ? (now_ms >= m_last_ms ? now_ms - m_last_ms : 0) : 0;
		dt          = std::min(dt, kMaxStepMs);
		m_have      = true;
		m_last_ms   = now_ms;

		const Mode want = Wanted(s, game_frame);
		if (m_game_frame != game_frame && game_frame) {
			// First game frame: the card goes away at once (no hold) and the card ETA state resets.
			m_card_hold_until = 0;
			m_replay_started  = false;
		}
		m_game_frame = game_frame;

		if (ReplayActive(s)) {
			if (!m_replay_started) {
				m_replay_started  = true;
				m_replay_start_ms = now_ms;
			}
		}

		// Card.
		const bool card_want = want == Mode::Card;
		if (card_want) m_card_hold_until = now_ms + kCardHoldMs;
		const bool card_on = card_want || (!game_frame && now_ms < m_card_hold_until);
		m_card_alpha = Ramp(m_card_alpha, card_on, dt, kFadeInMs, kCardFadeMs);

		// Badge: the delay counts continuous need; once shown it holds kBadgeHoldMs.
		const bool badge_want = want == Mode::Badge;
		if (badge_want) {
			if (!m_badge_need) {
				m_badge_need       = true;
				m_badge_need_since = now_ms;
			}
			if (!m_badge_shown && now_ms - m_badge_need_since >= kBadgeDelayMs) {
				m_badge_shown       = true;
				m_badge_hold_until  = now_ms + kBadgeHoldMs;
			}
		} else {
			m_badge_need = false;
			if (m_badge_shown && now_ms >= m_badge_hold_until) m_badge_shown = false;
		}
		// While needed the hold keeps sliding, so it only matters after the need ends.
		if (badge_want && m_badge_shown) m_badge_hold_until = now_ms + kBadgeHoldMs;
		const bool badge_on = m_badge_shown;
		m_badge_alpha = Ramp(m_badge_alpha, badge_on, dt, kFadeInMs, kBadgeFadeMs);

		// Bar easing toward the replay fraction; never visibly backwards unless the total changed.
		const double target = s.replay_total != 0 ? std::clamp(static_cast<double>(s.replay_done) / static_cast<double>(s.replay_total), 0.0, 1.0)
		                                          : (game_frame ? 1.0 : m_bar);
		if (s.replay_total != m_last_total) {
			m_last_total = s.replay_total;
			if (target < m_bar) m_bar = target;
		}
		if (s.replay_total != 0 || game_frame) {
			const double k = 1.0 - std::exp(-static_cast<double>(dt) / kBarTauMs);
			if (target >= m_bar) m_bar += (target - m_bar) * k;
		}
		m_bar = std::clamp(m_bar, 0.0, 1.0);

		View v;
		v.spinner = static_cast<float>(now_ms % kSpinnerMs) / static_cast<float>(kSpinnerMs);
		v.done    = s.replay_done;
		v.total   = s.replay_total;
		v.live    = std::max(s.live, s.waiting);
		v.waiting = s.waiting > 0;
		v.bar     = static_cast<float>(m_bar);
		if (m_card_alpha > 0.0f) {
			v.mode  = Mode::Card;
			v.alpha = m_card_alpha;
		} else if (m_badge_alpha > 0.0f) {
			v.mode  = Mode::Badge;
			v.alpha = m_badge_alpha;
		}
		if (ReplayActive(s) && m_replay_started && s.replay_done != 0 && now_ms - m_replay_start_ms >= kEtaMinMs) {
			const double per = static_cast<double>(now_ms - m_replay_start_ms) / 1000.0 / static_cast<double>(s.replay_done);
			v.eta_seconds    = per * static_cast<double>(s.replay_total - s.replay_done);
		}
		return v;
	}

	// load card: the load card stops at once (no fade) when the game flips again.
	void Cut() {
		m_card_alpha      = 0.0f;
		m_card_hold_until = 0;
	}

	// Still on screen (shown or fading): the presenter must keep repainting.
	bool Visible() const { return m_card_alpha > 0.0f || m_badge_alpha > 0.0f; }

private:
	static float Ramp(float alpha, bool on, uint64_t dt, uint64_t in_ms, uint64_t out_ms) {
		if (on) return std::min(1.0f, alpha + static_cast<float>(dt) / static_cast<float>(in_ms));
		return std::max(0.0f, alpha - static_cast<float>(dt) / static_cast<float>(out_ms));
	}

	bool     m_have             = false;
	bool     m_game_frame       = false;
	uint64_t m_last_ms          = 0;
	uint64_t m_card_hold_until  = 0;
	uint64_t m_badge_need_since = 0;
	uint64_t m_badge_hold_until = 0;
	uint64_t m_replay_start_ms  = 0;
	bool     m_replay_started   = false;
	uint64_t m_last_total       = 0;
	bool     m_badge_shown      = false;
	bool     m_badge_need       = false;
	float    m_card_alpha       = 0.0f;
	float    m_badge_alpha      = 0.0f;
	double   m_bar              = 0.0;
};

} // namespace Libs::Graphics::ProgressOverlay
