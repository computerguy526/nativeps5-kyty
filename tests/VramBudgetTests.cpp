// KYTY_VRAM_GC_BUDGET marks and plans (graphics/host_gpu/vramBudget.h) for a 24 GB, a 12 GB and an
// 8 GB card's planning budgets.
#include "graphics/host_gpu/vramBudget.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <map>

namespace {

using namespace Libs::Graphics::VramBudget;

constexpr uint64_t MiB = 1024ull * 1024;
constexpr uint64_t GiB = 1024 * MiB;

int g_failures = 0;

void Expect(bool condition, const char* what) {
	if (!condition) {
		std::printf("VramBudgetTests: failed: %s\n", what);
		g_failures++;
	}
}

void TestMarks() {
	const uint64_t budget = 9 * GiB + 512 * MiB; // a 12 GB card's planning budget
	Expect(ImageTrigger(budget) == budget / 4 * 3, "image trigger is three quarters of the budget");
	Expect(Critical(budget) == budget - budget / 16, "critical mark 1/16 below the budget");
	Expect(BufferTrigger(budget) == budget, "buffer collector only at the budget");
	Expect(BufferCritical(budget) > budget, "buffer downloads only above the budget");
	Expect(ImageTrigger(budget) < Critical(budget), "trigger below the critical mark");
}

void TestPlans() {
	// 24 GB card, Sky Garden (9.6 GB): nothing to do (the stock trigger would be 7.3 GiB).
	const uint64_t large = 22 * GiB + 512 * MiB;
	Expect(!PlanImages(large, 9600 * MiB, 0).retire, "24 GB: no retirement at 9.6 GB");
	// 12 GB card: 5.8 GB (Sky Garden with KYTY_FUNCTION_ARRAY_SHRINK) is below the trigger.
	const uint64_t medium = 9 * GiB + 512 * MiB;
	Expect(!PlanImages(medium, 5800 * MiB, 0).retire, "12 GB: no retirement at 5.8 GB");
	const auto above = PlanImages(medium, 8 * GiB, 0);
	Expect(above.retire && above.age_frames == DefaultAgeFrames, "12 GB: 8 GiB retires 30-frame-old images");
	Expect(above.bytes == 8 * GiB - ImageTrigger(medium), "12 GB: down to the trigger");
	const auto critical = PlanImages(medium, 9 * GiB + 256 * MiB, 0);
	Expect(critical.retire && critical.age_frames == DefaultAgeFrames / 4,
	       "12 GB: above the critical mark a quarter of the age");
	// 8 GB card: 5.8 GB is above the trigger.
	const uint64_t small = 6 * GiB + 300 * MiB;
	const auto     tight = PlanImages(small, 5800 * MiB, 0);
	Expect(tight.retire && tight.age_frames == DefaultAgeFrames, "8 GB: 5.8 GB retires 30-frame-old images");
	// KYTY_VRAM_PRESSURE_FRAMES as the age, and its floor of 4 frames.
	Expect(PlanImages(small, 5800 * MiB, 60).age_frames == 60, "age from KYTY_VRAM_PRESSURE_FRAMES");
	Expect(PlanImages(small, 6200 * MiB, 8).age_frames == 4, "a quarter, at least 4 frames");
	// No budget (the driver reports none): nothing.
	Expect(!PlanImages(0, 5 * GiB, 0).retire, "no budget: no retirement");
}

// KYTY_GC_SIDE_SETTLE_LAZY: PlanBufferPass.
void TestBufferPass() {
	// Off, or at the critical mark: every pass runs with the old limits, whatever the state.
	for (uint64_t deferred = 0; deferred <= 2 * BufferSettleMaxDeferred; deferred++) {
		for (int waits = 0; waits < 2; waits++) {
			const auto off = PlanBufferPass(false, false, waits != 0, deferred);
			Expect(off.run && off.limit == 32 && off.deferred == 0, "switch off: the pass as before");
			for (int lazy = 0; lazy < 2; lazy++) {
				const auto crit = PlanBufferPass(lazy != 0, true, waits != 0, deferred);
				Expect(crit.run && crit.limit == 64 && crit.deferred == 0,
				       "critical mark: the pass as before");
			}
		}
	}
	// Lazy, nothing to wait for: runs at once with the base limit.
	const auto free_pass = PlanBufferPass(true, false, false, 0);
	Expect(free_pass.run && free_pass.limit == 32 && free_pass.deferred == 0, "no wait: runs");
	// Lazy, would wait: put off, at most BufferSettleMaxDeferred times in a row, then it runs.
	uint64_t deferred = 0;
	uint64_t put_off  = 0;
	for (;;) {
		const auto pass = PlanBufferPass(true, false, true, deferred);
		if (pass.run) {
			Expect(pass.limit == 32 * BufferPassLimitMaxScale, "forced pass: the capped limit");
			Expect(pass.deferred == 0, "forced pass resets the count");
			break;
		}
		Expect(pass.limit == 0 && pass.deferred == deferred + 1, "put off: counted");
		deferred = pass.deferred;
		put_off++;
		if (put_off > 10 * BufferSettleMaxDeferred) {
			Expect(false, "a waiting pass is never put off forever");
			break;
		}
	}
	Expect(put_off == BufferSettleMaxDeferred, "exactly BufferSettleMaxDeferred passes put off");
	// A pass that runs late makes up for the passes it replaced (capped).
	Expect(PlanBufferPass(true, false, false, 3).limit == 4 * 32, "3 put off: 4 passes' worth");
	Expect(PlanBufferPass(true, false, false, 15).limit == 16 * 32, "15 put off: 16 passes' worth");
	Expect(PlanBufferPass(true, false, false, 40).limit == 16 * 32, "the scale is capped");
	// Random walks: never more than BufferSettleMaxDeferred passes without a run; the retired
	// capacity over the walk is at least min(passes, cap) x 32 per run group; state stays bounded.
	uint64_t seed = 0x9e3779b97f4a7c15ull;
	const auto next = [&seed] {
		seed ^= seed << 13;
		seed ^= seed >> 7;
		seed ^= seed << 17;
		return seed;
	};
	for (int walk = 0; walk < 2000; walk++) {
		uint64_t state = 0, streak = 0;
		const uint64_t bias = next() % 100; // % of passes that would wait
		for (int step = 0; step < 500; step++) {
			const bool crit  = next() % 97 == 0;
			const bool waits = next() % 100 < bias;
			const auto pass  = PlanBufferPass(true, crit, waits, state);
			if (pass.run) {
				Expect(pass.limit >= 32, "a running pass retires at least the base limit");
				Expect(crit || !waits || state >= BufferSettleMaxDeferred,
				       "a waiting pass below the critical mark runs only when forced");
				Expect(crit || pass.limit == 32 * std::min<uint64_t>(state + 1, BufferPassLimitMaxScale),
				       "late pass limit");
				streak = 0;
			} else {
				Expect(!crit && waits, "only a waiting pass below the critical mark is put off");
				streak++;
			}
			Expect(streak <= BufferSettleMaxDeferred, "bounded run of put-off passes");
			Expect(pass.deferred <= BufferSettleMaxDeferred, "bounded state");
			state = pass.deferred;
		}
	}
}

void TestChurn() {
	const uint64_t budget = 6 * GiB;
	ChurnTracker   t;
	Expect(t.Floor() == 0, "floor starts at 0");
	Expect(!t.NoteCreated(0x1000, 10), "never freed: no hit");
	t.NoteFreed(0x1000, 100);
	Expect(!t.NoteCreated(0x2000, 105), "other address: no hit");
	Expect(t.NoteCreated(0x1000, 105), "re-created 5 frames after the free: hit");
	Expect(t.Floor() == MinFloor && t.Hits() == 1, "first hit sets the minimum floor");
	Expect(!t.NoteCreated(0x1000, 106), "entry consumed by the hit");
	t.NoteFreed(0x1000, 200);
	Expect(!t.NoteCreated(0x1000, 200 + ChurnWindow + 1), "created after the window: no hit (never reused soon)");
	Expect(t.Floor() == MinFloor, "a miss leaves the floor");
	for (uint64_t i = 0; i < 20; ++i) {
		t.NoteFreed(0x10000 + i * 0x1000, 1000 + i);
		(void)t.NoteCreated(0x10000 + i * 0x1000, 1001 + i);
	}
	Expect(t.Floor() == MaxFloor, "floor doubles up to the maximum");
	// Decay: halves every DecayFrames frames without a hit, then off.
	const uint64_t last = 1001 + 19;
	t.Tick(last + DecayFrames - 1);
	Expect(t.Floor() == MaxFloor, "no decay before DecayFrames");
	t.Tick(last + DecayFrames);
	Expect(t.Floor() == MaxFloor / 2, "decays by half");
	uint64_t f = last + DecayFrames;
	for (int i = 0; i < 10; ++i) {
		f += DecayFrames;
		t.Tick(f);
	}
	Expect(t.Floor() == 0, "decays to zero");
	// Effective age: the floor protects, the hard limit overrides it, never lowers the plan's age.
	Expect(EffectiveAge(7, 60, budget, budget) == 60, "floor above the plan age wins");
	Expect(EffectiveAge(7, 4, budget, budget) == 7, "floor below the plan age: plan age");
	Expect(EffectiveAge(7, 60, budget, HardLimit(budget)) == 7, "above the hard limit: plan age");
	Expect(EffectiveAge(7, 0, budget, budget) == 7, "floor 0: plan age");
	// Table overwrite: a colliding address replaces the older entry, never a false hit.
	ChurnTracker u;
	for (uint64_t i = 0; i < 3 * ChurnTracker::Entries; ++i) {
		u.NoteFreed(i << 12u, 50);
	}
	uint64_t hits = 0;
	for (uint64_t i = 0; i < 3 * ChurnTracker::Entries; ++i) {
		hits += u.NoteCreated(i << 12u, 60) ? 1 : 0;
	}
	Expect(hits <= ChurnTracker::Entries && hits > 0, "bounded table: at most Entries hits");
	// Stress: random free/create against a reference map, hits must be a subset of true hits.
	ChurnTracker                              r;
	std::map<uint64_t, uint64_t>              ref;
	uint64_t                                  seed = 12345, bad = 0;
	for (uint64_t frame = 0; frame < 200000; ++frame) {
		seed         = seed * 6364136223846793005ull + 1442695040888963407ull;
		const auto a = ((seed >> 33) % 3000) << 12u;
		if ((seed >> 20) & 1) {
			r.NoteFreed(a, frame);
			ref[a] = frame;
		} else if (r.NoteCreated(a, frame)) {
			auto it = ref.find(a);
			if (it == ref.end() || frame - it->second > ChurnWindow) {
				++bad;
			}
		}
		r.Tick(frame);
		if (r.Floor() > MaxFloor) {
			++bad;
		}
	}
	Expect(bad == 0, "stress: no false hit, floor bounded");
	Expect(BurstBytes == 512ull * MiB, "burst cap 512 MiB");
}

} // namespace

// KYTY_GC_SIDE_SETTLE_LAZY_CRITICAL: PlanBufferPassLazyCritical (nps5 cpcommit2).
void TestBufferPassLazyCritical() {
	// Nothing to wait for: runs at once; the limit makes up for the passes put off (capped).
	for (uint64_t max = 0; max <= 2 * BufferSettleMaxDeferredCriticalLimit; max++) {
		for (uint64_t deferred = 0; deferred <= 80; deferred++) {
			const auto pass = PlanBufferPassLazyCritical(false, deferred, max);
			Expect(pass.run && pass.deferred == 0, "critical lazy, no wait: runs");
			Expect(pass.limit == BufferPassLimitCritical *
			                         std::min<uint64_t>(deferred + 1, BufferPassLimitMaxScale),
			       "critical lazy, no wait: late pass limit");
		}
	}
	// max 0: every pass runs with the critical limit, as PlanBufferPass does at the critical mark.
	for (int waits = 0; waits < 2; waits++) {
		const auto pass = PlanBufferPassLazyCritical(waits != 0, 0, 0);
		const auto old  = PlanBufferPass(false, true, waits != 0, 0);
		Expect(pass.run == old.run && pass.limit == old.limit && pass.deferred == old.deferred,
		       "critical lazy with max 0 equals the old critical pass");
	}
	// Would wait: put off exactly max times in a row (clamped), then forced.
	for (uint64_t max = 1; max <= 2 * BufferSettleMaxDeferredCriticalLimit; max++) {
		const auto clamped  = std::min(max, BufferSettleMaxDeferredCriticalLimit);
		uint64_t   deferred = 0, put_off = 0;
		for (;;) {
			const auto pass = PlanBufferPassLazyCritical(true, deferred, max);
			if (pass.run) {
				Expect(pass.deferred == 0, "critical lazy: a forced pass resets the count");
				Expect(pass.limit == BufferPassLimitCritical *
				                         std::min<uint64_t>(clamped + 1, BufferPassLimitMaxScale),
				       "critical lazy: forced pass limit");
				break;
			}
			Expect(pass.limit == 0 && pass.deferred == deferred + 1, "critical lazy: put off, counted");
			deferred = pass.deferred;
			if (++put_off > 1000) {
				Expect(false, "critical lazy: a waiting pass is never put off forever");
				break;
			}
		}
		Expect(put_off == clamped, "critical lazy: exactly max passes put off");
	}
	// Retirement capacity: over any run of passes, with max <= BufferPassLimitMaxScale - 1 the
	// passes that run may retire as many buffers as every pass running would have (64 each).
	uint64_t seed = 0x2545f4914f6cdd1dull;
	const auto next = [&seed] {
		seed ^= seed << 13;
		seed ^= seed >> 7;
		seed ^= seed << 17;
		return seed;
	};
	for (int walk = 0; walk < 2000; walk++) {
		const uint64_t max  = 1 + next() % (BufferPassLimitMaxScale - 1);
		const uint64_t bias = next() % 101;
		uint64_t state = 0, capacity = 0, streak = 0, runs = 0;
		const int steps = 400;
		for (int step = 0; step < steps; step++) {
			const bool waits = next() % 100 < bias;
			const auto pass  = PlanBufferPassLazyCritical(waits, state, max);
			if (pass.run) {
				capacity += pass.limit;
				runs++;
				Expect(!waits || streak == max, "critical lazy: a waiting pass runs only when forced");
				streak = 0;
			} else {
				Expect(waits, "critical lazy: only a waiting pass is put off");
				streak++;
				Expect(streak <= max, "critical lazy: at most max passes in a row put off");
			}
			state = pass.deferred;
		}
		Expect(capacity + streak * BufferPassLimitCritical == uint64_t {steps} * BufferPassLimitCritical,
		       "critical lazy: retirement capacity equals every pass running");
		Expect(runs * (max + 1) + max >= uint64_t {steps}, "critical lazy: at least one run per max+1 passes");
	}
}

int main() {
	TestMarks();
	TestPlans();
	TestBufferPass();
	TestBufferPassLazyCritical();
	TestChurn();
	if (g_failures != 0) {
		std::printf("%d failure(s)\n", g_failures);
		return 1;
	}
	std::printf("vram_budget: all tests passed\n");
	return 0;
}
