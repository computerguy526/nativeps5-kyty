// Footprint governor (hybrid/footprintGovernor.h): synthetic footprint series, no GPU.
#include "hybrid/footprintGovernor.h"

#include <cstdint>
#include <cstdio>

namespace {

using namespace Hybrid;

constexpr uint64_t ByteMiB = 1024ull * 1024;
constexpr uint64_t ByteGiB = 1024 * ByteMiB;
constexpr uint64_t Limit = 18944;

int g_failures = 0;

void Expect(bool condition, const char* what) {
	if (!condition) {
		std::printf("FootprintGovernorTests: failed: %s\n", what);
		g_failures++;
	}
}

FootprintGovernorConfig On(uint64_t limit = Limit) {
	FootprintGovernorConfig c;
	c.enabled   = true;
	c.limit_mib = limit;
	return c;
}

// Feeds a sample every 250 ms; returns the number of activations / deactivations.
struct Counts {
	int on = 0, off = 0;
};
Counts Feed(FootprintGovernor& g, uint64_t& now, uint64_t mib, uint64_t ms) {
	Counts c;
	for (uint64_t t = 0; t < ms; t += 250) {
		now += 250;
		const auto tr = g.Step(mib, now);
		c.on += tr == FootprintGovernor::Transition::Activated;
		c.off += tr == FootprintGovernor::Transition::Deactivated;
	}
	return c;
}

void TestBelowSoftNeverActivates() {
	FootprintGovernor g(On());
	uint64_t          now = 0;
	// the rc86 plateau 15.8-16.2 GiB is below 86 % of 18944 (16292 MiB)
	Expect(g.SoftMib() == 16291, "soft mark is 86 % of the limit");
	for (const uint64_t mib: {15911, 15821, 16078, 16171, 16088, 16197}) {
		const auto c = Feed(g, now, mib, 2000);
		Expect(c.on == 0 && !g.Active() && g.FactorPermille() == 1000, "plateau below the soft mark: never active");
	}
}

void TestCrossingActivatesOnceAndHysteresis() {
	FootprintGovernor g(On());
	uint64_t          now = 0;
	Feed(g, now, 16000, 1000);
	auto c = Feed(g, now, 16678, 250);
	Expect(c.on == 1 && g.Active() && g.FactorPermille() == 600, "crossing activates once with the factor");
	// noise of +-100 MiB around the soft mark does not flap (leave mark is 4 % = 757 MiB lower)
	for (int i = 0; i < 40; i++) {
		c = Feed(g, now, (i & 1) ? 16391 : 16191, 250);
		Expect(c.on == 0 && c.off == 0 && g.Active(), "no flap on +-100 MiB noise");
	}
	// a long time above the leave mark never deactivates
	c = Feed(g, now, 15600, 60000);
	Expect(c.off == 0 && g.Active(), "above the leave mark stays active");
}

void TestHoldTime() {
	FootprintGovernor g(On());
	uint64_t          now = 0;
	Feed(g, now, 17000, 500);
	Expect(g.Active(), "active");
	const uint64_t low = g.LeaveMib() - 1;
	auto           c   = Feed(g, now, low, 4500); // 4.5 s below the leave mark: held
	Expect(c.off == 0 && g.Active(), "hold: still active before 5 s");
	// a spike back above the leave mark restarts the hold
	Feed(g, now, g.LeaveMib() + 1, 250);
	c = Feed(g, now, low, 4500);
	Expect(c.off == 0 && g.Active(), "hold restarts after a spike");
	c = Feed(g, now, low, 1000);
	Expect(c.off == 1 && !g.Active() && g.FactorPermille() == 1000, "released after the hold");
	// and it can activate again
	c = Feed(g, now, 17500, 250);
	Expect(c.on == 1 && g.Active(), "re-activates");
}

void TestOffAndGate() {
	FootprintGovernor off_limit(On(0));
	FootprintGovernorConfig gated = On();
	gated.enabled = false;
	FootprintGovernor off_gate(gated);
	uint64_t          now = 0;
	Expect(Feed(off_limit, now, 40000, 5000).on == 0 && off_limit.FactorPermille() == 1000, "limit 0: never active");
	Expect(Feed(off_gate, now, 40000, 5000).on == 0 && off_gate.FactorPermille() == 1000 && !off_gate.Active(),
	       "gate off: never active, factor 1.0");
	const auto env_off = FootprintConfigFromEnv(nullptr, nullptr, nullptr, nullptr, Limit);
	const auto env_0   = FootprintConfigFromEnv("0", nullptr, nullptr, nullptr, Limit);
	const auto env_on  = FootprintConfigFromEnv("1", "90", "2000", "50", Limit);
	const auto env_bad = FootprintConfigFromEnv("1", "x", "-1", "5", Limit);
	Expect(!env_off.enabled && !env_0.enabled, "default off, 0 off");
	Expect(env_on.enabled && env_on.soft_pct == 90 && env_on.hold_ms == 2000 && env_on.factor_permille == 500,
	       "env values");
	Expect(env_bad.soft_pct == 86 && env_bad.hold_ms == 5000 && env_bad.factor_permille == 600,
	       "bad env keeps the defaults");
}

void TestScaledBudget() {
	Expect(ScaledBudget(10 * ByteGiB, 600) == 6 * ByteGiB, "0.6 of 10 ByteGiB");
	Expect(ScaledBudget(10 * ByteGiB, 1000) == 10 * ByteGiB, "factor 1.0 unchanged");
	Expect(ScaledBudget(10 * ByteGiB, 5000) == 10 * ByteGiB, "never raises");
	Expect(ScaledBudget(1500 * ByteMiB, 200) == 1 * ByteGiB, "clamps to >= 1 ByteGiB");
	Expect(ScaledBudget(512 * ByteMiB, 200) == 512 * ByteMiB, "a budget under 1 ByteGiB is not raised");
	Expect(ScaledBudget(0, 600) == 0, "zero stays zero");
	g_footprint_budget_permille.store(600);
	Expect(FootprintScaledBudget(10 * ByteGiB) == 6 * ByteGiB, "published factor applies");
	g_footprint_budget_permille.store(1000);
	Expect(FootprintScaledBudget(10 * ByteGiB) == 10 * ByteGiB, "published 1.0");
}

void TestSoftClamp() {
	FootprintGovernorConfig c = On(4096);
	c.soft_pct                = 98;
	FootprintGovernor g(c);
	Expect(g.SoftMib() == 4096 - 512, "soft above limit-512 MiB is clamped to limit-512 MiB");
	FootprintGovernor g2(On(1000));
	Expect(g2.SoftMib() < 1000, "soft below the limit");
}

void TestLogGate() {
	LogGate gate(1000);
	Expect(gate.Allow(0) && !gate.Allow(500) && !gate.Allow(999) && gate.Allow(1000) && !gate.Allow(1500),
	       "log rate limit");
}

} // namespace

int main() {
	TestBelowSoftNeverActivates();
	TestCrossingActivatesOnceAndHysteresis();
	TestHoldTime();
	TestOffAndGate();
	TestScaledBudget();
	TestSoftClamp();
	TestLogGate();
	if (g_failures != 0) {
		return 1;
	}
	std::printf("footprint_governor: all tests passed\n");
	return 0;
}
