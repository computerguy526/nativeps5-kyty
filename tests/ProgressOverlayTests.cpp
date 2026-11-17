#include "graphics/progressOverlay.h"

#include <cstdio>
#include <cstring>

using namespace Libs::Graphics::ProgressOverlay;
using Libs::Graphics::CompileProgress::Snapshot;

static int g_failed = 0;
#define CHECK(cond)                                                                      \
	do {                                                                                 \
		if (!(cond)) {                                                                   \
			std::printf("FAILED %s:%d: %s\n", __FILE__, __LINE__, #cond);                \
			g_failed++;                                                                  \
		}                                                                                \
	} while (0)

int main() {
	// Wanted().
	CHECK(Wanted({}, false) == Mode::Card); // loading card until the first game frame
	CHECK(Wanted({}, true) == Mode::Hidden);
	CHECK(Wanted({.replay_done = 1, .replay_total = 5}, false) == Mode::Card);
	CHECK(Wanted({.live = 1}, false) == Mode::Card);
	CHECK(Wanted({.replay_done = 5, .replay_total = 5}, false) == Mode::Card);
	CHECK(Wanted({.replay_done = 1, .replay_total = 5}, true) == Mode::Hidden); // replay in play: no badge
	CHECK(Wanted({.live = 1}, true) == Mode::Badge);
	CHECK(Wanted({.waiting = 1}, true) == Mode::Badge);

	// Card: shows at once, fades in, bar eases to the fraction, spinner turns.
	{
		Machine m;
		Snapshot s {.replay_done = 0, .replay_total = 100};
		auto v = m.Step(1000, s, false);
		CHECK(v.mode == Mode::Hidden && v.alpha == 0.0f); // first step has dt 0: nothing visible yet
		v = m.Step(1075, s, false);
		CHECK(v.mode == Mode::Card && v.alpha > 0.4f && v.alpha < 0.6f); // 75 of 150 ms
		v = m.Step(1200, s, false);
		CHECK(v.alpha == 1.0f);
		s.replay_done = 50;
		float prev = v.bar;
		bool monotone = true;
		for (uint64_t t = 1250; t <= 3250; t += 50) {
			v = m.Step(t, s, false);
			monotone = monotone && v.bar >= prev;
			prev = v.bar;
		}
		CHECK(monotone && v.bar > 0.49f && v.bar <= 0.5f);
		CHECK(v.eta_seconds > 1.5 && v.eta_seconds < 2.5); // 50 of 100 done after ~2.25 s: about 2.25 s left
		// Spinner: 0..1, one turn per kSpinnerMs.
		CHECK(m.Step(900, s, false).spinner == 0.0f);
		CHECK(m.Step(1350, s, false).spinner == 0.5f);
		// Replay done but no game frame yet: the card stays (LOADING).
		Machine n;
		Snapshot on {.replay_done = 1, .replay_total = 2}, off {};
		for (uint64_t t = 0; t <= 400; t += 50) (void)n.Step(t, on, false);
		for (uint64_t t = 450; t <= 5000; t += 50) CHECK(n.Step(t, off, false).alpha == 1.0f);
		// The first real game frame fades it out over kCardFadeMs.
		auto fading = n.Step(5100, off, true);
		CHECK(fading.alpha > 0.4f && fading.alpha < 0.8f);
		for (uint64_t t = 5150; t <= 5500; t += 50) (void)n.Step(t, off, true);
		CHECK(!n.Visible());
		// First game frame ends the card without the hold.
		Machine g;
		for (uint64_t t = 0; t <= 400; t += 50) (void)g.Step(t, on, false);
		auto gv = g.Step(450, on, true);
		CHECK(gv.mode == Mode::Card && gv.alpha < 1.0f); // fading already
		for (uint64_t t = 500; t <= 900; t += 50) gv = g.Step(t, on, true);
		CHECK(gv.mode == Mode::Hidden);
	}

	// Badge: delay, hold, fade-out; short compiles never show it.
	{
		Machine m;
		Snapshot live {.live = 2};
		for (uint64_t t = 0; t < kBadgeDelayMs; t += 20) CHECK(m.Step(t, live, true).mode == Mode::Hidden);
		(void)m.Step(kBadgeDelayMs + 20, live, true); // delay elapsed: fading in
		auto v = m.Step(kBadgeDelayMs + 120, live, true);
		CHECK(v.mode == Mode::Badge && v.alpha > 0.0f && v.live == 2);
		v = m.Step(kBadgeDelayMs + 300, live, true);
		CHECK(v.alpha == 1.0f);
		// Need ends: stays kBadgeHoldMs after the last need, then fades out over kBadgeFadeMs.
		const uint64_t end = kBadgeDelayMs + 300;
		CHECK(m.Step(end + 100, {}, true).alpha == 1.0f);
		CHECK(m.Step(end + kBadgeHoldMs - 10, {}, true).alpha == 1.0f);
		auto f = m.Step(end + kBadgeHoldMs + 200, {}, true);
		CHECK(f.alpha > 0.4f && f.alpha < 0.6f);
		for (uint64_t t = end + kBadgeHoldMs + 250; t <= end + kBadgeHoldMs + 800; t += 50) (void)m.Step(t, {}, true);
		CHECK(!m.Visible());

		// A 100 ms blip never appears.
		Machine b;
		for (uint64_t t = 0; t < 100; t += 20) (void)b.Step(t, live, true);
		for (uint64_t t = 100; t < 1500; t += 20) CHECK(b.Step(t, {}, true).mode == Mode::Hidden);
		// An interrupted need restarts the delay.
		Machine c;
		for (uint64_t t = 0; t < 200; t += 20) (void)c.Step(t, live, true);
		(void)c.Step(220, {}, true);
		for (uint64_t t = 240; t < 240 + kBadgeDelayMs - 20; t += 20) CHECK(c.Step(t, live, true).mode == Mode::Hidden);
	}

	// Large time gaps are clamped for fade timing (a stalled present thread must not skip the fade).
	{
		Machine m;
		Snapshot on {.replay_done = 1, .replay_total = 2};
		(void)m.Step(0, on, false);
		auto v = m.Step(60000, on, false);
		CHECK(v.alpha > 0.9f && v.alpha <= 1.0f); // 250 ms ramp at most
	}

	// Bar resets down only when the total changes.
	{
		Machine m;
		Snapshot a {.replay_done = 90, .replay_total = 100};
		for (uint64_t t = 0; t < 3000; t += 50) (void)m.Step(t, a, false);
		Snapshot b {.replay_done = 5, .replay_total = 400};
		auto v = m.Step(3050, b, false);
		CHECK(v.bar < 0.1f);
	}

	char eta[32];
	FormatEta(eta, sizeof(eta), -1.0);
	CHECK(!std::strcmp(eta, "ETA --"));
	FormatEta(eta, sizeof(eta), 42.0);
	CHECK(!std::strcmp(eta, "ETA 42S"));
	FormatEta(eta, sizeof(eta), 125.0);
	CHECK(!std::strcmp(eta, "ETA 2M 05S"));
	FormatEta(eta, sizeof(eta), 3700.0);
	CHECK(!std::strcmp(eta, "ETA 1H 01M"));

	if (g_failed == 0) std::printf("progress_overlay_tests: all passed\n");
	return g_failed == 0 ? 0 : 1;
}
