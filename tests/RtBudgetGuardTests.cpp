#include "graphics/host_gpu/renderer/pipeline/rtSession.h"

#include <cstdio>
#include <cstdlib>

static int g_checks = 0;
#define CHECK(c)                                                                                   \
	do {                                                                                           \
		++g_checks;                                                                                \
		if (!(c)) {                                                                                \
			std::fprintf(stderr, "RtBudgetGuardTests: FAIL line %d: %s\n", __LINE__, #c);          \
			std::abort();                                                                          \
		}                                                                                          \
	} while (0)

using Libs::Graphics::RtSession::BudgetGuard;
using Libs::Graphics::RtSession::LoadHold;
using Libs::Graphics::RtSession::ProbeGate;

int main() {
	{ // a valid BVH: the counter never moves, nothing is ever skipped
		BudgetGuard g;
		for (uint64_t t = 0; t < 100000; t += 16) CHECK(g.Allow(0, t));
		CHECK(g.Trips() == 0);
	}
	{ // lanes exhausted before the first look do not trip
		BudgetGuard g;
		CHECK(g.Allow(500000, 0));
		CHECK(g.Allow(500000, 16));
		CHECK(g.Trips() == 0);
	}
	{ // a rise below the threshold (a few stray lanes) does not trip
		BudgetGuard g;
		CHECK(g.Allow(0, 0));
		CHECK(g.Allow(1023, 16));
		CHECK(g.Allow(2046, 32));
		CHECK(g.Trips() == 0);
	}
	{ // the level-entry case: first dispatch runs, counter rises while it runs, the rest is held
		BudgetGuard g;
		CHECK(g.Allow(0, 1000));        // dispatch 1 of the pair (bad BVH) runs
		CHECK(!g.Allow(40000, 1001));   // dispatch 2: lanes of 1 already exhausted -> skipped
		CHECK(g.Trips() == 1 && g.CurrentHoldMs() == 500);
		CHECK(!g.Allow(131072, 1200));  // still counting: no new trip, hold not extended
		CHECK(!g.Allow(262144, 1499));
		CHECK(g.Trips() == 1);
		CHECK(!g.Allow(262144, 1500));  // 1 ms before the hold ends
		CHECK(g.Allow(262144, 1501));   // hold over, counter quiet: probe runs
		CHECK(g.Allow(262144, 1516));   // BVH valid now: normal again
		CHECK(g.Trips() == 1);
	}
	{ // BVH still invalid at the probe: the hold doubles, up to the cap; calm resets it
		BudgetGuard g;
		CHECK(g.Allow(0, 0));
		CHECK(!g.Allow(5000, 10));
		uint64_t t = 10 + 500;
		uint32_t c = 5000;
		uint64_t expect = 1000;
		for (int i = 0; i < 6; i++) {
			CHECK(g.Allow(c, t));          // probe runs
			c += 131072;
			CHECK(!g.Allow(c, t + 5));     // and exhausts again
			CHECK(g.CurrentHoldMs() == (expect > 8000 ? 8000 : expect));
			t += 5 + g.CurrentHoldMs();
			expect *= 2;
		}
		CHECK(g.CurrentHoldMs() == 8000);
		CHECK(g.Allow(c, t));
		t += 8000 + 30000 + 1;             // a long calm stretch
		CHECK(g.Allow(c, t));
		CHECK(!g.Allow(c + 2000, t + 16)); // a new episode starts at the base hold
		CHECK(g.CurrentHoldMs() == 500);
	}
	{ // counter wrap
		BudgetGuard g;
		CHECK(g.Allow(0xFFFFFF00u, 0));
		CHECK(!g.Allow(0x00000400u, 16)); // rose by 0x500 across the wrap
		CHECK(g.Trips() == 1);
	}
	const uint64_t MB = 1ull << 20;
	{ // gameplay: slow trickle of reads (8 MB/s) never holds
		LoadHold h;
		uint64_t b = 0;
		for (uint64_t t = 0; t < 60000; t += 16) {
			b += 8 * MB * 16 / 1000;
			CHECK(!h.Hold(b, 0, t));
		}
		CHECK(h.Holds() == 0);
	}
	{ // level load: 200 MB/s for 3 s, then quiet: held from the first sample to quiet_ms after the last
		LoadHold h;
		uint64_t b = 0, first_hold = 0, last_hold = 0;
		for (uint64_t t = 0; t < 10000; t += 16) {
			if (t >= 1000 && t < 4000) b += 200 * MB * 16 / 1000;
			if (h.Hold(b, 0, t)) {
				if (first_hold == 0) first_hold = t;
				last_hold = t;
			}
		}
		CHECK(first_hold >= 1000 && first_hold <= 1000 + 2 * (250 + 16)); // first full sample window
		CHECK(last_hold >= 4000 + 1500 - 250 - 32 && last_hold <= 4000 + 1500 + 250 + 16);
		CHECK(h.Holds() == 1);
	}
	{ // an endless stream of loads: one hold ends at max_hold_ms and no new one starts without a pause
		LoadHold h;
		uint64_t b = 0, held_ms = 0;
		for (uint64_t t = 0; t < 120000; t += 16) {
			b += 200 * MB * 16 / 1000;
			if (h.Hold(b, 0, t)) held_ms += 16;
		}
		CHECK(held_ms >= 59000 && held_ms <= 60100);
		CHECK(h.Holds() == 1);
	}
	{ // phase: off when asked, on by default; a change holds for phase_quiet_ms
		LoadHold off;
		off.use_phase = false;
		CHECK(!off.Hold(0, 1, 0));
		CHECK(!off.Hold(0, 2, 16));
		LoadHold on;
		CHECK(!on.Hold(0, 1, 0));
		CHECK(on.Hold(0, 2, 16));
		CHECK(on.Hold(0, 2, 5000));
		CHECK(!on.Hold(0, 2, 5016));
		CHECK(on.Holds() == 1);
	}
	{ // direct level boot (a test run): no fast reads at all, level opens, UI scenes follow;
	  // the RT burst 1.4 s after the level opened is held, RT is back 5 s after the last new scene
		LoadHold h;
		uint64_t b = 0;
		int      phase = 3;
		bool     burst_held = false, late_free = false;
		for (uint64_t t = 0; t < 30000; t += 16) {
			b += 5 * MB * 16 / 1000;            // slow trickle, far below the VFS trigger
			if (t == 13104) phase++;            // level opened
			if (t == 13504) phase++;            // ui_level_name_banner (new scene)
			const bool held = h.Hold(b, phase, t);
			if (t == 14496) burst_held = held;  // the RT burst
			if (t == 18528) late_free = !held;  // first tick 5 s after the last new scene
		}
		CHECK(burst_held);
		CHECK(late_free);
		CHECK(h.Holds() == 1);
	}
	{ // whichever is later: reads still fast after the phase window keep the hold; bounded by max
		LoadHold h;
		uint64_t b = 0, last_hold = 0, held_ms = 0;
		for (uint64_t t = 0; t < 60000; t += 16) {
			if (t >= 1000 && t < 9000) b += 200 * MB * 16 / 1000;
			if (h.Hold(b, t >= 1000 ? 1 : 0, t)) {
				last_hold = t;
				held_ms += 16;
			}
		}
		CHECK(last_hold >= 9000 + 1500 - 300 && last_hold <= 9000 + 1500 + 300);
		CHECK(held_ms < 30000);
		LoadHold m; // phase changes every 2 s: each restarts the budget; bounded once they stop
		uint64_t mh = 0;
		int      ph = 0;
		for (uint64_t t = 0; t < 200000; t += 16) {
			if (t % 2000 == 0 && t <= 100000) ph++;
			if (m.Hold(0, ph, t)) mh += 16;
		}
		CHECK(mh >= 103000 - 100 && mh <= 103000 + 100); // first change seen at 2 s, last at 100 s, + 5 s
		CHECK(m.Holds() == 1);
	}
	{ // hub -> level (from a play session): hub scenes at 383-389 s, level at 404.5 s, reads
	  // stay fast until 420 s; RT must still be held at 416 s and be free again by 423 s
		LoadHold h;
		uint64_t b = 0;
		int      ph = 0;
		bool     held_416 = false, free_423 = false;
		for (uint64_t t = 370000; t < 430000; t += 16) {
			if (t >= 383200 && t < 420000) b += 100 * MB * 16 / 1000;
			if (t == 383200 || t == 385008 || t == 389008 || t == 404496) ph++;
			const bool held = h.Hold(b, ph, t);
			if (t == 416000) held_416 = held;
			if (t == 423008) free_423 = !held;
		}
		CHECK(held_416);
		CHECK(free_423);
		CHECK(h.Holds() == 1);
	}
	{ // a counter that goes backwards (runtime restart of the counter) is not a sign
		LoadHold h;
		CHECK(!h.Hold(500 * MB, 0, 0));
		CHECK(!h.Hold(0, 0, 300));
		CHECK(h.Holds() == 0);
	}
	{ // probe gate, valid BVH: the first RT dispatch is the probe, the rest of that frame waits, RT
	  // opens when the GPU completed the probe without exhausted lanes
		ProbeGate g;
		bool      sent = false;
		CHECK(g.Allow(0, false, 0, sent) && sent);
		for (uint64_t t = 1; t < 40; t++) CHECK(!g.Allow(0, false, t, sent) && !sent);
		CHECK(g.Allow(0, true, 40, sent) && !sent);
		CHECK(g.Current() == ProbeGate::State::Open);
		for (uint64_t t = 41; t < 10000; t += 16) CHECK(g.Allow(0, true, t, sent));
		CHECK(g.Probes() == 1 && g.Failed() == 0 && g.Opened() == 1);
	}
	{ // a few stray lanes below the threshold do not fail a probe
		ProbeGate g;
		bool      sent = false;
		CHECK(g.Allow(100, false, 0, sent));
		CHECK(g.Allow(100 + 1023, true, 50, sent));
		CHECK(g.Failed() == 0);
	}
	// A level-entry timeline: the load hold ends at `hold_end`, the game attempts `per_frame` RT
	// dispatches every 27 ms, the BVH stays invalid until `valid_at`; a dispatch on an invalid BVH
	// exhausts 30000 lanes, the GPU completes a dispatch 60 ms after it was recorded. Returns how
	// many dispatches ran on the invalid BVH.
	const auto level_entry = [&](uint64_t hold_end, uint64_t valid_at, uint64_t end, uint32_t per_frame,
	                             uint64_t& ran_valid, uint64_t& open_at) {
		ProbeGate g;
		bool      sent = false;
		CHECK(g.Allow(0, false, 0, sent) && sent);   // session start, long before the level
		CHECK(g.Allow(0, true, 100, sent));
		uint32_t counter = 0, pending = 0;
		uint64_t ran_invalid = 0, probe_done = 0;
		bool     waiting = false;
		ran_valid = 0;
		open_at   = 0;
		g.Arm();   // the load hold ended
		for (uint64_t t = hold_end; t < end; t += 27) {
			for (uint32_t i = 0; i < per_frame; i++) {
				const bool done = waiting && t >= probe_done;
				if (done && pending != 0) {
					counter += pending;
					pending = 0;
				}
				const auto before = g.Current();
				if (g.Allow(counter, done, t, sent)) {
					if (t < valid_at) {
						ran_invalid++;
						pending = 30000;
					} else {
						ran_valid++;
					}
					if (sent) {
						waiting    = true;
						probe_done = t + 60;
					} else if (before == ProbeGate::State::Waiting && open_at == 0) {
						open_at = t;
					}
				}
			}
		}
		CHECK(g.Current() == ProbeGate::State::Open);
		return ran_invalid;
	};
	{ // route run (hold over at level-open + 13 s, BVH invalid ~2 s longer, 45 attempts per frame):
	  // without the gate about 135 dispatches ran on the invalid BVH before the hold came back
		uint64_t valid = 0, open_at = 0;
		const auto bad = level_entry(85000, 87000, 95000, 45, valid, open_at);
		CHECK(bad >= 1 && bad <= 3);              // one per probe: 0 s, +1 s (and +2 s at the edge)
		CHECK(open_at >= 87000 && open_at <= 87000 + 2000 + 200);
		CHECK(valid > 1000);
	}
	{ // play session (hold over at 413 s, BVH invalid until ~421 s): a handful of probes, never a burst
		uint64_t valid = 0, open_at = 0;
		const auto bad = level_entry(413000, 421000, 440000, 45, valid, open_at);
		CHECK(bad >= 3 && bad <= 5);              // probes at 0, +1, +3, +7 s (retry doubles)
		CHECK(open_at >= 421000 && open_at <= 421000 + 8000 + 200);
	}
	{ // clean level entry (BVH valid when the hold ends): one probe, open after ~60 ms
		uint64_t valid = 0, open_at = 0;
		const auto bad = level_entry(30000, 0, 35000, 45, valid, open_at);
		CHECK(bad == 0);
		CHECK(open_at >= 30000 + 54 && open_at <= 30000 + 120);
	}
	{ // a probe the GPU never completes counts as invalid after the timeout; a guard trip closes
		ProbeGate g;
		bool      sent = false;
		CHECK(g.Allow(0, false, 0, sent) && sent);
		CHECK(!g.Allow(0, false, 3999, sent));
		CHECK(!g.Allow(0, false, 4000, sent));
		CHECK(g.Failed() == 1 && g.Current() == ProbeGate::State::Closed);
		CHECK(!g.Allow(0, false, 4999, sent));
		CHECK(g.Allow(0, false, 5000, sent) && sent);    // retry 1 s later: a new probe
		CHECK(g.Allow(0, true, 5100, sent) && !sent);
		g.Close(6000);                                   // budget guard trip while open
		CHECK(!g.Allow(5000, true, 6500, sent));
		CHECK(g.Allow(5000, false, 7000, sent) && sent);
	}
	std::printf("RtBudgetGuardTests: %d checks passed\n", g_checks);
	return 0;
}
