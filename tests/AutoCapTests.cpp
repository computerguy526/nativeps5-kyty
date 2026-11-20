// Standalone: clang++ -std=c++17 -fsanitize=address,undefined -I src/graphics/presentation tests/AutoCapTests.cpp && ./a.out
#include "autoCap.h"
#include <cstdio>
#include <cstdlib>

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); g_fail++; } } while (0)
#define NEAR(a, b) CHECK(std::fabs((a) - (b)) < 1e-6)

// run the controller with a constant "game can do `game_fps`" model: measured = min(cap, game)
// (presented = min(cap, game) when capped, game when uncapped; work fps = game)
static double Run(AutoCap::Controller& c, double t0, double secs, double game_fps) {
	double t = t0;
	for (; t < t0 + secs; t += 0.1) c.Tick(t, c.Cap() > 0 ? std::fmin(c.Cap(), game_fps) : game_fps, game_fps);
	return t;
}

// run until cap == want; returns time, d = elapsed
static double RunUntil(AutoCap::Controller& c, double t0, double want, double maxs, double& d) {
	double t = t0;
	for (; t < t0 + maxs; t += 0.1) { c.Tick(t, std::fmin(c.Cap(), 30), 30); if (std::fabs(c.Cap() - want) < 1e-6) break; }
	d = t - t0; return t;
}

int main() {
	using namespace AutoCap;
	NEAR(PickCap(75, 40), 37.5);
	NEAR(PickCap(75, 30), 0.0);
	NEAR(PickCap(75, 37.5), 37.5);
	NEAR(PickCap(75, 80), 75.0);
	NEAR(PickCap(75, 5), 0.0);   // floor: never below 30 fps -> uncapped
	NEAR(PickCap(75, 20), 0.0);  // 18.75 < floor
	NEAR(PickCap(75, 28), 0.0);  // 25 < floor
	NEAR(PickCap(60, 35), 30.0);
	NEAR(PickCap(60, 25), 0.0);
	NEAR(PickCap(60, 25, 10), 20.0); // KYTY_FPS_AUTOCAP_MIN=10
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
	// game sustains 30 (< floor-limited 37.5): 25 would be below the floor -> uncapped within ~3.5 s, and stays so
	double t = Run(c, 30, 4, 30); NEAR(c.Cap(), 0.0);
	t = Run(c, t, 60, 30); NEAR(c.Cap(), 0.0);
	// game recovers to 40: capped at 37.5 after ~5 s
	double d = 0;
	{ double t0 = t; for (; t < t0 + 20; t += 0.1) { c.Tick(t, c.Cap() > 0 ? std::fmin(c.Cap(), 40) : 40, 40); if (c.Cap() > 0) break; }
	  NEAR(c.Cap(), 37.5); CHECK(t - t0 > 4.5 && t - t0 < 7.5); }
	// game sustains 60: 37.5 holds until work fps shows headroom for 75 (needs 72.75): 60 is not enough
	t = Run(c, t, 100, 60); NEAR(c.Cap(), 37.5);
	// work time shows headroom (game could do 200 fps; presented saturates at the cap): reaches 75 after ~30 s
	t = Run(c, t, 100, 200); NEAR(c.Cap(), 75.0);
	// game drops to 20 at cap 75: straight to uncapped (no 15/18.75 steps)
	t = Run(c, t, 5, 20); NEAR(c.Cap(), 0.0);
	// failing probes back off: uncapped at 36 (cap 37.5 ok? no, 36 < 37.5 -> PickCap 0 at 75 Hz) stays uncapped
	t = Run(c, t, 60, 36); NEAR(c.Cap(), 0.0);
	(void)RunUntil;
	// scenarios from the field (floor 30)
	{ Controller x; x.SetScreen(Screen{"M", 75.0, false}, 0); Run(x, 1, 60, 20); NEAR(x.Cap(), 0.0); }   // 75 Hz, 20 fps
	{ Controller x; x.SetScreen(Screen{"M", 75.0, false}, 0); Run(x, 1, 60, 40); NEAR(x.Cap(), 37.5); }  // 75 Hz, 40 fps
	{ Controller x; x.SetScreen(Screen{"M", 60.0, false}, 0); Run(x, 1, 60, 25); NEAR(x.Cap(), 0.0); }   // 60 Hz, 25 fps
	{ Controller x; x.SetScreen(Screen{"M", 144.0, false}, 0); Run(x, 1, 60, 40); NEAR(x.Cap(), 36.0); } // 144 Hz, 40 fps
	{ Controller x(10.0); x.SetScreen(Screen{"M", 60.0, false}, 0); Run(x, 1, 60, 25); NEAR(x.Cap(), 20.0); } // custom floor
	// screen moves: reset to R/2 of the new display, adaptive -> none, back -> n=2
	NEAR(c.SetScreen(adaptive, 500), 0.0);
	NEAR(c.SetScreen(Screen{"X", 60.0, false}, 501), 30.0);
	NEAR(c.SetScreen(Screen{"Y", 144.0, false}, 502), 72.0);
	// unknown refresh: no cap
	NEAR(c.SetScreen(Screen{"?", 0.0, false}, 503), 0.0);

	// work estimator
	{ WorkEstimator w; for (int i = 0; i < 100; i++) w.OnWork(0.01); CHECK(std::fabs(w.Fps() - 100.0) < 0.5); w.OnWork(2.0); CHECK(std::fabs(w.Fps() - 100.0) < 0.5); }
	// estimator
	Estimator e; for (int i = 0; i < 100; i++) e.OnPresent(i / 37.5);
	CHECK(std::fabs(e.Fps() - 37.5) < 0.1);
	e.OnPresent(100.0); // 60 s gap ignored
	CHECK(std::fabs(e.Fps() - 37.5) < 0.1);

	// ---- PromoCap (ProMotion mode): synthetic frame-time traces ----
	{
		struct Sim {
			PromoCap pc {120.0};
			double   t = 0.0;
			uint32_t rng = 12345;
			int      changes = 0;
			double   Rand() { rng = rng * 1664525u + 1013904223u; return (rng >> 8) / 16777216.0; }
			// run `secs` of a game whose frame work time is base*(1 +- jit), plus `spike_every`-th
			// frame spike_s (0 = none). Present interval = max(cap period, work).
			void Run(double secs, double base, double jit, int spike_every = 0, double spike_s = 0.0) {
				const double end = t + secs; int i = 0;
				while (t < end) {
					double w = base * (1.0 + jit * (2.0 * Rand() - 1.0));
					if (spike_every && ++i % spike_every == 0) w = spike_s;
					const double iv = std::fmax(1.0 / pc.Cap(), w);
					t += iv;
					pc.OnPresent(t);
					pc.OnWork(t, w <= 0.25 ? w : 0.0);
					const double c0 = pc.Cap(); pc.Tick(t);
					if (pc.Cap() != c0) changes++;
				}
			}
		};
		{ Sim s; s.Run(30, 0.042, 0.03); NEAR(s.pc.Cap(), 24.0); }          // Sky Garden 42 ms -> 24
		{ Sim s; s.Run(40, 0.026, 0.03); NEAR(s.pc.Cap(), 30.0); }          // 26 ms -> 30
		{ Sim s; s.Run(60, 0.022, 0.02); NEAR(s.pc.Cap(), 40.0); }          // hub 22 ms -> 40
		{ Sim s; s.Run(60, 0.012, 0.05); NEAR(s.pc.Cap(), 60.0); }          // easy scene -> 60
		{ Sim s; s.Run(60, 0.060, 0.02); NEAR(s.pc.Cap(), 24.0); }          // below the lowest: floor 24
		// quick step down: first change within ~5 s of a slow scene
		{ Sim s; s.Run(5, 0.042, 0.02); NEAR(s.pc.Cap(), 24.0); }
		// no flip-flop near a boundary: 25 ms of work vs the 40 fps period (25 ms) +-6 %
		{ Sim s; s.Run(300, 0.025, 0.06); CHECK(s.changes <= 4); CHECK(s.pc.Cap() <= 40.0 + 1e-6); }
		// 100 ms spikes on every 25th frame (4 %) do not drop the cap
		{ Sim s; s.Run(20, 0.022, 0.02); NEAR(s.pc.Cap(), 40.0); const int c = s.changes; s.Run(30, 0.022, 0.02, 25, 0.1); CHECK(s.changes == c); NEAR(s.pc.Cap(), 40.0); }
		// a 3 s loading burst (every frame 150 ms) may drop the cap, but it recovers once stable
		{ Sim s; s.Run(30, 0.022, 0.02); s.Run(3, 0.15, 0.02); s.Run(80, 0.022, 0.02); NEAR(s.pc.Cap(), 40.0); }
		// a scene that gets heavier then lighter follows both ways
		{ Sim s; s.Run(40, 0.022, 0.02); NEAR(s.pc.Cap(), 40.0); s.Run(20, 0.042, 0.02); NEAR(s.pc.Cap(), 24.0); s.Run(120, 0.022, 0.02); NEAR(s.pc.Cap(), 40.0); }
		// stalls longer than kMaxGap are not samples
		{ PromoCap p(120.0); p.OnPresent(0.0); p.OnPresent(5.0); p.OnPresent(5.03); CHECK(true); }
		NEAR(PromoCap(120.0).Cap(), 30.0);
		NEAR(PromoCap(100.0).Cap(), 25.0);
	}

	// publish dedup
	auto& st = State(); uint32_t s0 = st.seq.load();
	Publish(mon); Publish(mon); CHECK(st.seq.load() == s0 + 1);
	Publish(adaptive); CHECK(st.seq.load() == s0 + 2);
	CHECK(Snapshot().name == "Built-in");
	std::printf(g_fail ? "FAILED %d\n" : "OK\n", g_fail);
	return g_fail != 0;
}
