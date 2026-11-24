#pragma once

// nps5 load card: self-present watchdog for level-load compiles (hosted Kyty).
//
// When the game stops flipping (its command processor waits for pipeline/shader compiles) the
// window stays on the last presented image or black, with no sign of life. A watchdog thread owned
// by the presenter (graphics/presentation/window/swapchain.cpp) watches the flip sequence number
// and the compile counters (graphics/compileProgress.h); when no game flip has happened for
// kStartMs and compiles are pending or in flight it presents the last game frame plus the
// "COMPILING SHADERS N / M" card from its own thread at ~kPresentHz, until the game flips again.
// The decision logic lives here (header only, no GPU) so tests/LoadCardTests.cpp can drive it
// offline.
//
//   KYTY_LOADCARD=0           off
//   KYTY_LOADCARD_ANY=0       card only while compiles are pending (default 1: any reason: boot,
//   data loading, compiles;
//                             'LOADING... Ns' without compiles, 'COMPILING SHADERS N / M' with
//                             them; also before the first flip)
//   KYTY_LOADCARD_MS=<ms>     idle time without a game flip before the card (default 500)
//   KYTY_LOADCARD_HZ=<hz>     card present rate (default 30, 1..60)
//   KYTY_LOADCARD_LOG=1       log start/stop to stderr
//   KYTY_LOADCARD_MIDGAME=1   re-enable the mid-game card (default 0: the full card belongs to the
//   opening
//                             load ONLY; later loads show just the small bottom-left "compiling"
//                             badge). With 1, after the first real content frame the card can
//                             return for a genuine stall, see below.
//   The opening card stops for good at the first frame the game presents that is not fully black
//   (any pixel above kBlackMax on the probe), and never shows while a video plays
//   (nps5_avplayer_playing_count() > 0). KYTY_LOADCARD_GAP_S=<s>   policy: the full-screen card
//   belongs to the session start (before the first
//                             real content frame). Afterwards the game's own black fades/cuts show
//                             nothing; the card returns only when no content frame was presented
//                             for this long (default 4.0 s) AND the game is still loading (shader
//                             compiles pending/in flight, or VFS bytes still moving), and it fades
//                             in (SystemOverlay) and is cut at once by the next content frame.

#include "graphics/compileProgress.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>

namespace Libs::Graphics::LoadCard {

inline constexpr uint64_t kDefaultStartMs =
    500; // no game flip for this long (and work pending) starts the card
inline constexpr uint64_t kPendingGraceMs =
    200; // pending counters may blink off between two compiles
inline constexpr uint64_t kIdleStopMs =
    2000; // card stops (one clean repaint) after this long without work and still no flip
inline constexpr uint32_t kDefaultHz = 30;

inline uint64_t EnvU64(const char* name, uint64_t def, uint64_t lo, uint64_t hi) {
	const char* v = std::getenv(name);
	if (v == nullptr || v[0] == '\0') return def;
	char*      end = nullptr;
	const auto n   = std::strtoull(v, &end, 10);
	if (end == v) return def;
	return std::clamp<uint64_t>(n, lo, hi);
}
inline bool Enabled() {
	static const bool enabled = [] {
		const char* v = std::getenv("KYTY_LOADCARD");
		return v == nullptr || v[0] != '0';
	}();
	return enabled;
}
inline bool AnyReason() {
	static const bool any = [] {
		const char* v = std::getenv("KYTY_LOADCARD_ANY");
		return v == nullptr || v[0] != '0';
	}();
	return any;
}
// KYTY_LOADCARD_BLACK=0: black game frames count as flips (no luma probe).
inline bool BlackEnabled() {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_LOADCARD_BLACK");
		return Enabled() && AnyReason() && (v == nullptr || v[0] != '0');
	}();
	return on;
}
// KYTY_LOADCARD_MIDGAME=0: no full-screen card after the first content frame.
inline bool MidgameEnabled() {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_LOADCARD_MIDGAME");
		return v != nullptr && v[0] != '\0' && v[0] != '0';
	}();
	return on;
}
// card3: a video (AvPlayer, e.g. the PlayStation Studios logo) is playing: that is content, never
// card. Runtime hook nps5_avplayer_playing_count() (ps5recomp AvPlayer); absent -> 0.
inline int VideoPlayingCount() {
	static const auto fn =
	    reinterpret_cast<int (*)()>(dlsym(RTLD_DEFAULT, "nps5_avplayer_playing_count"));
	return fn != nullptr ? fn() : 0;
}
inline bool VideoPlaying() {
	return VideoPlayingCount() > 0;
}
// card3: the launch compile phase (library replay, pipeline list replay, compute prewarm:
// CompileProgress unit) is still running. The opening card stays up until it is done AND the game
// presents a non-black frame. KYTY_LOADCARD_PHASE=0 turns the wait off.
inline bool PhaseWaitEnabled() {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_LOADCARD_PHASE");
		return v == nullptr || v[0] != '0';
	}();
	return on;
}
inline bool PhasePending(uint64_t unit_total, uint64_t unit_done) {
	return unit_total != 0 && unit_done < unit_total;
}
inline bool LaunchPhasePending() {
	if (!PhaseWaitEnabled()) return false;
	// zerocomp: a hold that is over ends the phase for the card even when the unit is not done
	// (scene mode).
	if (CompileProgress::Get().launch_hold.load(std::memory_order_acquire) == 2) return false;
	const auto& c = CompileProgress::Get();
	return PhasePending(c.unit_total.load(std::memory_order_relaxed),
	                    c.unit_done.load(std::memory_order_relaxed));
}
// Does a presented frame end the opening card? A playing video always does (never card over a
// video); else the launch compile phase must be done, and the frame must not be fully black.
inline bool EndsOpening(bool video, bool phase_pending, bool frame_black) {
	if (video) return true;
	if (phase_pending) return false;
	return !frame_black;
}
// card3 luma test: a frame is "black" only when no sampled pixel channel exceeds kBlackMax (max,
// not mean: a dark frame with a small logo is content).
inline constexpr unsigned kBlackMax = 2;
inline bool               FrameIsBlack(unsigned max_channel) {
	return max_channel <= kBlackMax;
}
inline uint64_t GapMs() {
	static const uint64_t ms = [] {
		const char* v = std::getenv("KYTY_LOADCARD_GAP_S");
		double      s = 4.0;
		if (v != nullptr && v[0] != '\0') {
			char*        end = nullptr;
			const double d   = std::strtod(v, &end);
			if (end != v && d >= 0.0) s = d;
		}
		return static_cast<uint64_t>(std::clamp(s, 0.05, 3600.0) * 1000.0);
	}();
	return ms;
}
inline uint64_t StartMs() {
	static const uint64_t ms = EnvU64("KYTY_LOADCARD_MS", kDefaultStartMs, 50, 600000);
	return ms;
}
inline uint64_t PresentIntervalMs() {
	static const uint64_t ms = 1000 / EnvU64("KYTY_LOADCARD_HZ", kDefaultHz, 1, 60);
	return ms;
}
inline bool LogEnabled() {
	static const bool enabled = [] {
		const char* v = std::getenv("KYTY_LOADCARD_LOG");
		return v != nullptr && v[0] == '1';
	}();
	return enabled;
}

// ---- Shared state (relaxed atomics; written by the presenter under its present mutex or by the
// watchdog) ----
struct Shared {
	std::atomic<uint64_t> flip_seq {0}; // game (non blank, main bus) flips presented
	std::atomic<bool> active {false};   // the card is wanted on the next present of the last frame
	std::atomic<uint64_t> last_flip_ms {
	    0}; // watchdog clock (steady ms) of the last game flip it saw (LOADING... Ns)
	std::atomic<uint8_t> probe_state {
	    0}; // luma probe of the game frame: 0 unknown, 1 black, 2 content
	std::atomic<uint32_t> unknown_flips {0}; // game flips seen while the probe state was unknown
	std::atomic<bool>     content_seen {
	    false}; // a real (non black) content frame was presented this session
	std::atomic<uint32_t> cut_gen {
	    0}; // bumped when the card stops: the overlay drops its fade at once
	// Compile totals at the last time the watchdog saw the game flipping (baseline of the card's N
	// / M), and the live compile totals of the current card session.
	std::atomic<uint64_t> base_started {0}, base_finished {0};
};
inline Shared& State() {
	static Shared s;
	return s;
}
inline bool Active() {
	return State().active.load(std::memory_order_acquire);
}

inline constexpr uint32_t kUnknownFlips =
    30; // probe never answered: after this many flips they count as content

inline void NoteProbe(bool black) {
	State().probe_state.store(black ? 1 : 2, std::memory_order_relaxed);
}

// The current game frame is (still) black: the last probe said black, or no answer yet for the
// first flips.
inline bool BlackNow() {
	if (!BlackEnabled()) return false;
	if (VideoPlaying()) return false; // card3: a playing video is content
	auto& s = State();
	if (!s.content_seen.load(std::memory_order_relaxed) && LaunchPhasePending())
		return true; // opening: phase not done
	const auto p = s.probe_state.load(std::memory_order_relaxed);
	if (p == 1) return true;
	if (p == 2) return false;
	// Counted up to the limit only (a counter that kept running would wrap to "black" again).
	auto seen = s.unknown_flips.load(std::memory_order_relaxed);
	while (seen < kUnknownFlips) {
		if (s.unknown_flips.compare_exchange_weak(seen, seen + 1, std::memory_order_relaxed)) {
			return true;
		}
	}
	return false;
}

// load card: while the card is up it owns every present. A black game frame arriving then is held
// (recorded as the last frame, not presented): the watchdog repaints card + last frame at a steady
// rate, so game black frames never interleave with card presents on screen (flicker). Content
// frames end the card and present normally.
inline bool HoldGameFrame(bool black, bool card_active) {
	return black && card_active;
}

// Counts a main-bus game flip (called under the presenter mutex before the frame is presented): the
// card is off for this very frame. A black game frame (black = true) is not a flip for the card: it
// stays up over the game's own black loading frames until the first frame with content.
inline void NoteGameFlip(bool black = false) {
	if (black) return;
	auto& s = State();
	if (!s.content_seen.exchange(true, std::memory_order_acq_rel)) {
		// hotfiles interface: the first picture, for NPS5_VFS_LOADPROFILE (no-op without the
		// symbol).
		if (auto* mark =
		        reinterpret_cast<void (*)()>(dlsym(RTLD_DEFAULT, "nps5_vfs_mark_first_picture")))
			mark();
	}
	s.flip_seq.fetch_add(1, std::memory_order_acq_rel);
	if (s.active.exchange(false, std::memory_order_acq_rel))
		s.cut_gen.fetch_add(1, std::memory_order_relaxed);
}

inline bool Pending(const CompileProgress::Snapshot& s) {
	return (s.replay_total != 0 && s.replay_done < s.replay_total) || s.live > 0 || s.waiting > 0;
}

// policy: what the watchdog is allowed to do. Before the first content frame: the startup card (any
// reason, start_ms). After it: only a long stall (start_ms = gap) while work is pending (compiles
// or VFS bytes moving); never with midgame disabled.
struct Policy {
	bool     armed    = false;
	bool     any      = false;
	uint64_t start_ms = 0;
};
inline Policy MakePolicy(bool content_seen, bool content_now, uint64_t seq, bool paused,
                         bool any_reason, bool midgame, uint64_t start_ms, uint64_t gap_ms,
                         bool video = false) {
	Policy p;
	if (video) return p; // card3: never over a video
	if (!content_seen) {
		p.armed    = (any_reason || seq != 0) && !content_now && !paused;
		p.any      = any_reason;
		p.start_ms = start_ms;
	} else {
		p.armed    = midgame && !paused;
		p.any      = false;
		p.start_ms = std::max(gap_ms, start_ms);
	}
	return p;
}

// "The game is still loading data": VFS read counter changed within the last window.
class VfsActivity {
public:
	static constexpr uint64_t kWindowMs = 1500;
	bool                      Update(uint64_t now_ms, uint64_t bytes) {
		if (!m_have || bytes != m_bytes) {
			if (m_have) m_moved_ms = now_ms;
			m_bytes = bytes;
			m_have  = true;
		}
		return m_moved_ms != 0 && now_ms - m_moved_ms <= kWindowMs;
	}

private:
	bool     m_have  = false;
	uint64_t m_bytes = 0, m_moved_ms = 0;
};

// The state machine. One thread (the watchdog) calls Step; times are monotonic ms.
class Watchdog {
public:
	struct Decision {
		bool active      = false; // the card must be on screen
		bool present_now = false; // present the card frame now
		bool started     = false; // transition off -> on
		bool stopped     = false; // transition on -> off
		bool repaint_clean =
		    false; // stopped without a flip: repaint the last frame without the card once
	};

	explicit Watchdog(uint64_t start_ms = kDefaultStartMs, uint64_t interval_ms = 1000 / kDefaultHz)
	    : m_start_ms(start_ms), m_interval_ms(interval_ms) {}

	// flip_seq: game flip counter; pending: compile work is waiting or in flight; armed: the window
	// has presented a game frame at least once and the watchdog may act (the startup card has its
	// own path).
	void SetStartMs(uint64_t ms) { m_start_ms = ms; }

	Decision Step(uint64_t now_ms, uint64_t flip_seq, bool pending, bool armed,
	              bool any_reason = false) {
		Decision d;
		if (!m_have || flip_seq != m_seq) {
			// A game flip (or the first look): the card stops at once, the idle clock restarts.
			if (m_active) d.stopped = true;
			m_active       = false;
			m_have         = true;
			m_seq          = flip_seq;
			m_last_flip_ms = now_ms;
			m_pending_ms   = pending ? now_ms : 0;
			m_have_pending = pending;
			m_present_ms   = 0;
			d.active       = false;
			return d;
		}
		if (pending) {
			m_pending_ms   = now_ms;
			m_have_pending = true;
		}
		const bool pending_recent = m_have_pending && now_ms - m_pending_ms <= kPendingGraceMs;
		if (!m_active) {
			if (armed && now_ms - m_last_flip_ms >= m_start_ms && (pending_recent || any_reason)) {
				m_active      = true;
				d.started     = true;
				d.present_now = true;
				m_present_ms  = now_ms;
			}
		} else if (!any_reason && !pending_recent && now_ms - m_pending_ms >= kIdleStopMs) {
			// The work ended and the game still does not flip (waiting on something else): do not
			// leave a stale card up; repaint once without it and wait for new work.
			m_active        = false;
			d.stopped       = true;
			d.repaint_clean = true;
			m_have_pending  = false;
		} else if (now_ms - m_present_ms >= m_interval_ms) {
			d.present_now = true;
			m_present_ms  = now_ms;
		}
		d.active = m_active;
		return d;
	}

	bool     IsActive() const { return m_active; }
	uint64_t LastFlipMs() const { return m_last_flip_ms; }

private:
	uint64_t m_start_ms, m_interval_ms;
	bool     m_have         = false;
	bool     m_active       = false;
	bool     m_have_pending = false;
	uint64_t m_seq          = 0;
	uint64_t m_last_flip_ms = 0;
	uint64_t m_pending_ms   = 0;
	uint64_t m_present_ms   = 0;
};

// The "N / M" shown on the card. Replay work wins; otherwise the live compiles since the last flip
// the watchdog saw (finished / started), at least `waiting` pending. total == 0 means
// indeterminate.
inline CompileProgress::Snapshot CardSnapshot(const CompileProgress::Snapshot& s, uint64_t started,
                                              uint64_t finished, uint64_t base_started,
                                              uint64_t base_finished, bool any_reason = false) {
	if (s.replay_total != 0 && s.replay_done < s.replay_total)
		return s; // the list unit (Preparing shaders X / T)
	CompileProgress::Snapshot c = s;
	if (any_reason) {
		// Not the list: "LOADING... Ns" and, when compiles outside the list run, the live count as
		// a detail line.
		c.replay_total = 0;
		c.replay_done  = 0;
		return c;
	}
	const uint64_t done     = finished >= base_finished ? finished - base_finished : 0;
	const uint64_t started_ = started >= base_started ? started - base_started : 0;
	const uint64_t flight   = static_cast<uint64_t>(std::max(0, std::max(s.live, s.waiting)));
	const uint64_t total    = std::max(started_, done + flight);
	c.replay_total          = total;
	c.replay_done           = done;
	return c;
}

} // namespace Libs::Graphics::LoadCard
