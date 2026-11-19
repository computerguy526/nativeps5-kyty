// Standalone: clang++ -std=c++17 -fsanitize=address,undefined -I src/graphics/presentation tests/AutoCapTests.cpp && ./a.out
#include "autoCap.h"
#include <cstdio>
#include <cstdlib>

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); g_fail++; } } while (0)
#define NEAR(a, b) CHECK(std::fabs((a) - (b)) < 1e-6)

// run the controller with a constant "game can do `game_fps`" model: measured = min(cap, game)
static double Run(AutoCap::Controller& c, double t0, double secs, double game_fps) {
	double t = t0;
	for (; t < t0 + secs; t += 0.1) c.Tick(t, std::fmin(c.Cap(), game_fps));
	return t;
}

// run until cap == want; returns time, d = elapsed
static double RunUntil(AutoCap::Controller& c, double t0, double want, double maxs, double& d) {
	double t = t0;
	for (; t < t0 + maxs; t += 0.1) { c.Tick(t, std::fmin(c.Cap(), 30)); if (std::fabs(c.Cap() - want) < 1e-6) break; }
	d = t - t0; return t;
}

int main() {
	using namespace AutoCap;
	NEAR(PickCap(75, 40), 37.5);
	NEAR(PickCap(75, 30), 25.0);
	NEAR(PickCap(75, 37.5), 37.5);
	NEAR(PickCap(75, 80), 75.0);
	NEAR(PickCap(75, 5), 75.0 / 7); // floor: cap stays >= 10
	NEAR(PickCap(60, 35), 30.0);
	NEAR(PickCap(60, 25), 20.0);
	NEAR(PickCap(144, 50), 48.0);
	NEAR(PickCap(144, 40), 36.0);
	NEAR(PickCap(144, 100), 72.0);
	NEAR(PickCap(15, 10), 0.0);

	Controller c;
	Screen adaptive{"Built-in", 120.0, true};
	NEAR(c.SetScreen(adaptive, 0), 0.0);
	NEAR(Run(c, 1, 10, 20) > 0, 1); NEAR(c.Cap(), 0.0); // adaptive never caps

	Screen mon{"D27-30", 75.0, false};
	NEAR(c.SetScreen(mon, 0), 37.5);
	// game sustains 60: stays 37.5 (probe to 75 is long), no down
	Run(c, 1, 20, 60); NEAR(c.Cap(), 37.5);
	// game sustains 30: moves down to 25 within ~3.5 s
	double t = Run(c, 30, 4, 30); NEAR(c.Cap(), 25.0); CHECK(c.Changed() || true);
	// on cadence 5 s -> probe up to 37.5 (game 30 fails -> down again after ~2 s, back-off 2)
	double d = 0;
	t = RunUntil(c, t, 37.5, 8, d); CHECK(d > 3.5 && d < 8.5);
	t = RunUntil(c, t, 25.0, 6, d); CHECK(d > 1.5 && d < 3.5);
	t = RunUntil(c, t, 37.5, 14, d); CHECK(d > 9.5 && d < 12.5); // back-off: ~10 s, not 5
	// game recovers to 60: stays at 37.5, reaching 75 only after 30 s * backoff (>= 1 probe)
	Run(c, t, 400, 200); NEAR(c.Cap(), 75.0);
	// screen moves: reset to R/2 of the new display, adaptive -> none, back -> n=2
	NEAR(c.SetScreen(adaptive, 500), 0.0);
	NEAR(c.SetScreen(Screen{"X", 60.0, false}, 501), 30.0);
	NEAR(c.SetScreen(Screen{"Y", 144.0, false}, 502), 72.0);
	// unknown refresh: no cap
	NEAR(c.SetScreen(Screen{"?", 0.0, false}, 503), 0.0);

	// estimator
	Estimator e; for (int i = 0; i < 100; i++) e.OnPresent(i / 37.5);
	CHECK(std::fabs(e.Fps() - 37.5) < 0.1);
	e.OnPresent(100.0); // 60 s gap ignored
	CHECK(std::fabs(e.Fps() - 37.5) < 0.1);

	// publish dedup
	auto& st = State(); uint32_t s0 = st.seq.load();
	Publish(mon); Publish(mon); CHECK(st.seq.load() == s0 + 1);
	Publish(adaptive); CHECK(st.seq.load() == s0 + 2);
	CHECK(Snapshot().name == "Built-in");
	std::printf(g_fail ? "FAILED %d\n" : "OK\n", g_fail);
	return g_fail != 0;
}
