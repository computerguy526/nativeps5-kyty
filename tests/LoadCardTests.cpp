#include "graphics/loadCard.h"
#include "graphics/progressOverlay.h"

#include <cstdio>

using namespace Libs::Graphics::LoadCard;
using Libs::Graphics::CompileProgress::Snapshot;

static int g_failed = 0;
#define CHECK(cond)                                                          \
	do {                                                                     \
		if (!(cond)) {                                                       \
			std::printf("FAILED %s:%d: %s\n", __FILE__, __LINE__, #cond);    \
			g_failed++;                                                      \
		}                                                                    \
	} while (0)

int main() {
	// Pending().
	CHECK(!Pending({}));
	CHECK(Pending({.live = 1}));
	CHECK(Pending({.waiting = 1}));
	CHECK(Pending({.replay_done = 1, .replay_total = 5}));
	CHECK(!Pending({.replay_done = 5, .replay_total = 5}));

	// Normal play: flips every 16 ms with compiles pending never starts the card.
	{
		Watchdog w(500, 66);
		uint64_t seq = 1;
		for (uint64_t t = 0; t < 5000; t += 16) {
			const auto d = w.Step(t, seq++, true, true);
			CHECK(!d.active && !d.started && !d.present_now);
		}
	}
	// No flip but nothing pending: never starts.
	{
		Watchdog w(500, 66);
		for (uint64_t t = 0; t < 10000; t += 20) CHECK(!w.Step(t, 7, false, true).active);
	}
	// Not armed (before the first game frame): never starts.
	{
		Watchdog w(500, 66);
		for (uint64_t t = 0; t < 3000; t += 20) CHECK(!w.Step(t, 0, true, false).active);
	}
	// Start exactly at the threshold, present rate, immediate stop on a flip.
	{
		Watchdog w(500, 66);
		CHECK(!w.Step(1000, 5, true, true).active); // first look: last flip = 1000
		uint64_t start = 0;
		int      presents = 0;
		for (uint64_t t = 1020; t <= 2000; t += 20) {
			const auto d = w.Step(t, 5, true, true);
			if (d.started) {
				CHECK(start == 0);
				start = t;
				CHECK(d.present_now);
			}
			if (d.present_now) presents++;
			CHECK(d.active == (t >= 1500));
		}
		CHECK(start == 1500);
		CHECK(presents >= 7 && presents <= 9); // 500 ms at 66 ms interval from 1500 -> 8 incl. the first
		// Flip: stops in the same step, no present, and the idle clock restarts.
		auto d = w.Step(2010, 6, true, true);
		CHECK(!d.active && d.stopped && !d.present_now && !d.repaint_clean);
		CHECK(!w.Step(2400, 6, true, true).active); // 390 ms since the flip: below the threshold
		CHECK(w.Step(2510, 6, true, true).started); // 500 ms
	}
	// Hysteresis: pending blinking off for < grace keeps the card; the gap before the start is also bridged.
	{
		Watchdog w(500, 66);
		(void)w.Step(0, 1, true, true);
		bool on = false;
		for (uint64_t t = 20; t <= 700; t += 20) {
			const bool pend = (t % 100) != 0 || t < 400; // one 20 ms hole every 100 ms after the start
			on = w.Step(t, 1, pend, true).active;
		}
		CHECK(on);
	}
	// Work ends while the game still does not flip: after kIdleStopMs a clean repaint and stop; new work restarts.
	{
		Watchdog w(500, 66);
		(void)w.Step(0, 1, true, true);
		CHECK(w.Step(600, 1, true, true).started);
		bool repaint = false, stopped = false;
		uint64_t stop_t = 0;
		for (uint64_t t = 620; t <= 4000; t += 20) {
			const auto d = w.Step(t, 1, false, true);
			if (d.repaint_clean) {
				repaint = true;
				stop_t  = t;
			}
			stopped |= d.stopped;
			if (stop_t != 0) CHECK(!d.active);
		}
		CHECK(repaint && stopped);
		CHECK(stop_t >= 600 + kIdleStopMs && stop_t <= 600 + kIdleStopMs + 40);
		CHECK(w.Step(4020, 1, true, true).started); // new compiles, still no flip
	}
	// A flip and new pending in the same step: a clean restart of the idle clock, never an immediate card.
	{
		Watchdog w(500, 66);
		(void)w.Step(0, 1, true, true);
		CHECK(w.Step(600, 1, true, true).active);
		const auto d = w.Step(620, 2, true, true);
		CHECK(!d.active && d.stopped);
		CHECK(!w.Step(640, 2, true, true).active);
	}
	// Guest paused (not armed) while the card is up: no more presents, state kept consistent.
	{
		Watchdog w(500, 66);
		(void)w.Step(0, 1, true, true);
		CHECK(w.Step(600, 1, true, true).active);
		CHECK(w.Step(700, 1, true, true).active); // armed=true here; pause handling is the caller's (loop skips presents)
	}
	// Shared state: NoteGameFlip cuts the card at once and bumps the generation once.
	{
		auto& s = State();
		const auto seq0 = s.flip_seq.load();
		const auto gen0 = s.cut_gen.load();
		s.active.store(true);
		NoteGameFlip();
		CHECK(!Active() && s.flip_seq.load() == seq0 + 1 && s.cut_gen.load() == gen0 + 1);
		NoteGameFlip(); // not active: no cut
		CHECK(s.cut_gen.load() == gen0 + 1);
	}
	// CardSnapshot: replay wins; otherwise finished/total of the live compiles since the baseline.
	{
		Snapshot replay {.replay_done = 3, .replay_total = 10, .live = 2};
		auto c = CardSnapshot(replay, 50, 40, 10, 10);
		CHECK(c.replay_total == 10 && c.replay_done == 3);
		Snapshot live {.live = 2};
		c = CardSnapshot(live, 15, 12, 10, 10); // 2 done, 5 started, 2 in flight
		CHECK(c.replay_done == 2 && c.replay_total == 5);
		c = CardSnapshot(live, 10, 14, 10, 10); // compile begun before the baseline finished
		CHECK(c.replay_done == 4 && c.replay_total == 6);
		Snapshot wait {.waiting = 1};
		c = CardSnapshot(wait, 10, 10, 10, 10);
		CHECK(c.replay_done == 0 && c.replay_total == 1);
		c = CardSnapshot({}, 10, 10, 10, 10);
		CHECK(c.replay_total == 0);
	}
	// Overlay: Cut() drops the card at once (no fade), a following game frame stays clean.
	{
		namespace PO = Libs::Graphics::ProgressOverlay;
		PO::Machine m;
		Snapshot    s {.replay_done = 1, .replay_total = 10};
		(void)m.Step(1000, s, false);
		auto v = m.Step(1200, s, false);
		CHECK(v.mode == PO::Mode::Card && v.alpha == 1.0f);
		m.Cut();
		v = m.Step(1216, {}, true);
		CHECK(v.mode == PO::Mode::Hidden && !m.Visible());
	}
	// any-reason mode: starts with nothing pending (boot / data loading), also before the first flip (seq 0),
	// never idle-stops, stops at once on a flip, not when paused (armed false).
	{
		Watchdog w(500, 66);
		CHECK(!w.Step(0, 0, false, true, true).active);
		int presents = 0;
		for (uint64_t t = 20; t <= 20000; t += 20) {
			const auto d = w.Step(t, 0, false, true, true);
			CHECK(d.active == (t >= 500));
			CHECK(!d.repaint_clean);
			if (d.present_now) presents++;
		}
		CHECK(presents > 200);
		const auto f = w.Step(20020, 1, false, true, true);
		CHECK(f.stopped && !f.active);
		for (uint64_t t = 20040; t < 20500; t += 20) CHECK(!w.Step(t, 1, false, true, true).active);
		CHECK(w.Step(20540, 1, false, true, true).started);
		Watchdog p(500, 66);
		for (uint64_t t = 0; t < 5000; t += 20) CHECK(!p.Step(t, 0, false, false, true).active);
		Watchdog legacy(500, 66);
		for (uint64_t t = 0; t < 5000; t += 20) CHECK(!legacy.Step(t, 3, false, true).active);
	}
	// black game frames: not a flip for the card; the first content frame is.
	{
		auto& st = State();
		const auto seq0 = st.flip_seq.load();
		st.active.store(true);
		NoteGameFlip(true);
		CHECK(st.flip_seq.load() == seq0 && Active());
		NoteProbe(true);
		CHECK(BlackNow() == BlackEnabled());
		NoteProbe(false);
		CHECK(!BlackNow());
		NoteGameFlip(false);
		CHECK(st.flip_seq.load() == seq0 + 1 && !Active());
		st.probe_state.store(0);
		st.unknown_flips.store(0);
		int black = 0;
		for (int i = 0; i < 100; i++) black += BlackNow() ? 1 : 0;
		CHECK(black == (BlackEnabled() ? int(kUnknownFlips) : 0));
		// the count stops at the limit (a running counter wrapped to "black" again).
		st.unknown_flips.store(0xffffffffu);
		CHECK(!BlackNow());
		CHECK(!BlackNow());
		st.unknown_flips.store(0);
	}
	{
		Snapshot s;
		s.live = 3;
		const auto c = CardSnapshot(s, 10, 5, 0, 0, true);
		CHECK(c.replay_total == 0 && c.live == 3);
		Snapshot l;
		l.replay_total = 1396;
		l.replay_done  = 700;
		const auto d = CardSnapshot(l, 0, 0, 0, 0, true);
		CHECK(d.replay_total == 1396 && d.replay_done == 700);
	}
	// load card: the card owns presents: black game frames are held while it is up, content frames never.
	CHECK(HoldGameFrame(true, true) && !HoldGameFrame(false, true) && !HoldGameFrame(true, false) && !HoldGameFrame(false, false));
	// load card: the session policy driven like the presenter loop (20 ms ticks).
	{
		struct Sim {
			Watchdog    w {500, 66};
			VfsActivity vfs;
			uint64_t    seq = 0, bytes = 0, t = 0;
			bool        content_seen = false, content_now = false, midgame = true, black_frames = false;
			bool        card = false, shown_ever = false, card_at_content = false;
			uint64_t    card_ms = 0;
			// phase: black = game presents black frames (no flip for the card), else content flips.
			void Tick(bool presenting_content, bool presenting_black, bool compiling, bool reading) {
				t += 20;
				if (presenting_content) { seq++; content_seen = true; content_now = true; card = false; }
				if (presenting_black) content_now = false;
				if (reading) bytes += 4096;
				const auto pol = MakePolicy(content_seen, content_now, seq, false, true, midgame, 500, 4000);
				w.SetStartMs(pol.start_ms);
				bool work = compiling;
				if (content_seen) work = vfs.Update(t, bytes) || work;
				const auto d = w.Step(t, seq, work, pol.armed, pol.any);
				if (d.started) card = true;
				if (d.stopped) card = false;
				if (card) { shown_ever = true; card_ms += 20; }
			}
		};
		// startup shows the card (no flip yet, any reason), first content hides it.
		{
			Sim m;
			for (int i = 0; i < 100; i++) m.Tick(false, false, false, false); // 2 s of nothing
			CHECK(m.card);
			m.Tick(true, false, false, false);
			CHECK(!m.card && m.content_seen);
		}
		// 0.5 s and 2 s black fades mid-game show nothing (even while compiling / reading).
		for (int ms : {500, 2000}) {
			Sim m;
			for (int i = 0; i < 100; i++) m.Tick(true, false, false, false); // content
			m.shown_ever = false;
			for (int i = 0; i < ms / 20; i++) m.Tick(false, true, true, true); // black
			for (int i = 0; i < 100; i++) m.Tick(true, false, true, true);
			CHECK(!m.shown_ever);
		}
		// 5 s stall with compiling shows the card (after the 4 s gap), next content frame hides it at once.
		{
			Sim m;
			for (int i = 0; i < 100; i++) m.Tick(true, false, false, false);
			uint64_t t0 = m.t; bool seen_early = false;
			for (int i = 0; i < 250; i++) {
				m.Tick(false, true, true, false);
				if (m.card && m.t - t0 < 3900) seen_early = true;
			}
			CHECK(!seen_early);
			CHECK(m.card);
			m.Tick(true, false, true, false);
			CHECK(!m.card);
		}
		// 5 s stall with nothing loading (no compiles, no VFS bytes) shows nothing.
		{
			Sim m;
			for (int i = 0; i < 100; i++) m.Tick(true, false, false, false);
			m.shown_ever = false;
			for (int i = 0; i < 250; i++) m.Tick(false, true, false, false);
			CHECK(!m.shown_ever);
		}
		// 5 s stall with only VFS bytes moving shows the card.
		{
			Sim m;
			for (int i = 0; i < 100; i++) m.Tick(true, false, false, false);
			for (int i = 0; i < 250; i++) m.Tick(false, true, false, true);
			CHECK(m.card);
		}
		// KYTY_LOADCARD_MIDGAME=0: a long stall shows nothing.
		{
			Sim m;
			m.midgame = false;
			for (int i = 0; i < 100; i++) m.Tick(true, false, false, false);
			for (int i = 0; i < 400; i++) m.Tick(false, true, true, true);
			CHECK(!m.shown_ever);
		}
		// card3: the mid-game card is OFF by default: a long stall with compiles + VFS bytes shows nothing.
		{
			CHECK(!MidgameEnabled()); // env unset in the test process
			Sim m;
			m.midgame = MidgameEnabled();
			for (int i = 0; i < 100; i++) m.Tick(true, false, false, false);
			for (int i = 0; i < 600; i++) m.Tick(false, true, true, true);
			CHECK(!m.shown_ever);
		}
		// card3: dark logo frame (small bright logo on a near-black frame) is content; only fully black is black.
		CHECK(FrameIsBlack(0) && FrameIsBlack(kBlackMax) && !FrameIsBlack(kBlackMax + 1));
		CHECK(!FrameIsBlack(13)); // flat dark game frame / dark logo frame
		CHECK(!FrameIsBlack(40)); // dim logo on black
		// card3: a playing video is content: policy never arms, before or after the first content frame.
		CHECK(!MakePolicy(false, false, 0, false, true, true, 500, 4000, true).armed);
		CHECK(!MakePolicy(true, false, 9, false, true, true, 500, 4000, true).armed);
		CHECK(VideoPlayingCount() >= 0 && !VideoPlaying()); // no AvPlayer hook in the test process
		// card3: the opening card stays through the launch compile phase; hides only when it is done AND a non-black frame.
		CHECK(PhasePending(200, 199) && !PhasePending(200, 200) && !PhasePending(0, 0));
		CHECK(!EndsOpening(false, true, false));  // content frame while ~200 of T compiled: card stays
		CHECK(!EndsOpening(false, true, true));
		CHECK(!EndsOpening(false, false, true));  // phase done but frame fully black: stays
		CHECK(EndsOpening(false, false, false));  // phase done + non-black: hides
		CHECK(EndsOpening(true, true, true));     // a playing video always ends it (never card over a video)
		{
			auto& cu = Libs::Graphics::CompileProgress::Get();
			cu.unit_total.store(1500); cu.unit_done.store(200);
			CHECK(LaunchPhasePending());
			cu.unit_done.store(1500);
			CHECK(!LaunchPhasePending());
			cu.unit_total.store(0); cu.unit_done.store(0);
		}
		// Policy: startup is any-reason with the short start, afterwards pending-only with the gap.
		{
			auto a = MakePolicy(false, false, 0, false, true, true, 500, 4000);
			CHECK(a.armed && a.any && a.start_ms == 500);
			auto b = MakePolicy(true, false, 9, false, true, true, 500, 4000);
			CHECK(b.armed && !b.any && b.start_ms == 4000);
			CHECK(!MakePolicy(true, false, 9, false, true, false, 500, 4000).armed);
		}
	}
	// load card: badge hysteresis: shows after 300 ms of continuous compiling, a 100 ms compile never shows; it stays at
	// least 1 s and does not flicker over short need gaps.
	{
		namespace PO = Libs::Graphics::ProgressOverlay;
		PO::Machine m;
		Snapshot    work {.live = 1}, idle {};
		uint64_t    t = 1000;
		bool        shown = false;
		(void)m.Step(t, idle, true);
		for (int i = 0; i < 6; i++) { t += 16; shown |= m.Step(t, work, true).mode == PO::Mode::Badge; } // ~100 ms
		for (int i = 0; i < 60; i++) { t += 16; shown |= m.Step(t, idle, true).mode == PO::Mode::Badge; }
		CHECK(!shown);
		uint64_t on_since = 0, off_at = 0;
		for (int i = 0; i < 40; i++) { // 640 ms of compiling
			t += 16;
			if (m.Step(t, work, true).mode == PO::Mode::Badge && on_since == 0) on_since = t;
		}
		CHECK(on_since != 0);
		// need gap of 500 ms: the badge stays (min on-time / hold 1 s)
		bool gone = false;
		for (int i = 0; i < 31; i++) { t += 16; if (m.Step(t, idle, true).mode != PO::Mode::Badge) gone = true; }
		CHECK(!gone);
		for (int i = 0; i < 200 && off_at == 0; i++) { t += 16; if (m.Step(t, idle, true).mode == PO::Mode::Hidden) off_at = t; }
		CHECK(off_at != 0 && off_at - on_since >= 1000);
	}
	std::printf(g_failed == 0 ? "load_card_tests: all passed\n" : "load_card_tests: %d FAILED\n", g_failed);
	return g_failed == 0 ? 0 : 1;
}
