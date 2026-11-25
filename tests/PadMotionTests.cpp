// Motion relay helpers (hybrid/padMotion.h): unit conversion, per-controller mocks (DualSense, DualShock 4, Switch Pro
// Controller, Joy-Con), sign conventions, rate estimation, orientation-derived rates, trace record/replay, hotplug.
// No SDL calls and no controller.
#include "graphics/presentation/window/padFocus.h"
#include "hybrid/padMotion.h"

#include <SDL3/SDL_gamepad.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#ifndef PAD_MOTION_TRACE
#define PAD_MOTION_TRACE "tests/data/pad_motion_trace.txt"
#endif

namespace PM = Hybrid::PadMotion;

namespace {
int g_checks   = 0;
int g_failures = 0;

void Expect(bool condition, const char* what) {
	++g_checks;
	if (!condition) {
		std::printf("PadMotionTests: failed: %s\n", what);
		++g_failures;
	}
}
bool Near(float a, float b, float eps) {
	return std::fabs(a - b) <= eps;
}

// What a controller of that type reports lying flat and still on a table, in SDL's normalised axes: gravity on +y, a
// small per-model gyro bias and accelerometer noise (the values differ, the axes do not).
struct Mock {
	const char* name;
	int         sdl_type;
	float       rate_hz;
	float       accel[3];
	float       gyro[3];
};
const Mock kMocks[] = {
    {"DualSense", SDL_GAMEPAD_TYPE_PS5, 250.0f, {0.03f, 9.78f, 0.05f}, {0.002f, -0.001f, 0.0005f}},
    {"DualShock4", SDL_GAMEPAD_TYPE_PS4, 250.0f, {-0.10f, 9.84f, 0.12f}, {-0.004f, 0.003f, 0.002f}},
    {"SwitchPro", SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_PRO, 200.0f, {0.20f, 9.70f, -0.15f}, {0.010f, 0.007f, -0.012f}},
    {"JoyConLeft", SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_JOYCON_LEFT, 200.0f, {-0.18f, 9.90f, 0.20f}, {-0.008f, 0.009f, 0.004f}},
};

void TestTypeNames() {
	Expect(std::string(PM::TypeName(SDL_GAMEPAD_TYPE_PS5)) == "ps5", "type name ps5");
	Expect(std::string(PM::TypeName(SDL_GAMEPAD_TYPE_PS4)) == "ps4", "type name ps4");
	Expect(std::string(PM::TypeName(SDL_GAMEPAD_TYPE_PS3)) == "ps3", "type name ps3");
	Expect(std::string(PM::TypeName(SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_PRO)) == "switch-pro", "type name switch pro");
	Expect(std::string(PM::TypeName(SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_JOYCON_LEFT)) == "switch-joycon-left", "type name joycon left");
	Expect(std::string(PM::TypeName(SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_JOYCON_RIGHT)) == "switch-joycon-right", "type name joycon right");
	Expect(std::string(PM::TypeName(SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_JOYCON_PAIR)) == "switch-joycon-pair", "type name joycon pair");
	Expect(std::string(PM::TypeName(SDL_GAMEPAD_TYPE_XBOX360)) == "xbox360", "type name xbox360");
	Expect(std::string(PM::TypeName(SDL_GAMEPAD_TYPE_XBOXONE)) == "xboxone", "type name xboxone");
	Expect(std::string(PM::TypeName(SDL_GAMEPAD_TYPE_STANDARD)) == "standard", "type name standard");
	Expect(std::string(PM::TypeName(SDL_GAMEPAD_TYPE_UNKNOWN)) == "other" && std::string(PM::TypeName(999)) == "other", "unknown type");
}

// Kyty's pad data carries g; the relay hands the host m/s^2. Flat on the table = +1 g on +y, for every controller type.
void TestConversionPerController() {
	for (const Mock& m : kMocks) {
		const float accel_g[3] = {m.accel[0] / PM::kGravity, m.accel[1] / PM::kGravity, m.accel[2] / PM::kGravity};
		PM::Sample  s;
		Expect(PM::FromPadData(accel_g, m.gyro, 123456, &s), m.name);
		Expect(s.valid && s.time_us == 123456, "sample valid with its time");
		Expect(Near(s.accel[1], m.accel[1], 1e-4f) && Near(s.accel[0], m.accel[0], 1e-4f) && Near(s.accel[2], m.accel[2], 1e-4f), "accel round trip in m/s^2");
		const float mag = std::sqrt(s.accel[0] * s.accel[0] + s.accel[1] * s.accel[1] + s.accel[2] * s.accel[2]);
		Expect(Near(mag, PM::kGravity, 0.4f), "flat: |a| is about 1 g");
		Expect(s.accel[1] > 0.95f * PM::kGravity * 0.98f && std::fabs(s.accel[0]) < 0.3f && std::fabs(s.accel[2]) < 0.3f, "flat: gravity on +y");
		Expect(Near(s.gyro[0], m.gyro[0], 1e-6f) && Near(s.gyro[1], m.gyro[1], 1e-6f) && Near(s.gyro[2], m.gyro[2], 1e-6f), "gyro rad/s untouched");
	}
	// Sign conventions: no axis is flipped or swapped (SDL already normalised them).
	const float tilt_g[3] = {0.5f, 0.0f, -0.8f};
	const float spin[3]   = {1.0f, -2.0f, 3.0f};
	PM::Sample  s;
	Expect(PM::FromPadData(tilt_g, spin, 1, &s), "tilt sample");
	Expect(s.accel[0] > 0 && s.accel[2] < 0 && s.gyro[0] == 1.0f && s.gyro[1] == -2.0f && s.gyro[2] == 3.0f, "signs and axes preserved");
	// Right edge down (+x gravity) reads +x; nose-down (+z gravity: screen facing the player's feet) reads +z.
	const float on_edge[3] = {1.0f, 0.0f, 0.0f};
	const float zero[3]    = {};
	PM::FromPadData(on_edge, zero, 1, &s);
	Expect(Near(s.accel[0], PM::kGravity, 1e-4f) && s.accel[1] == 0.0f, "1 g on x");
	// Glitches: out-of-range values clamp, non-finite readings are refused and leave the sample alone.
	const float huge_a[3] = {500.0f, 0.0f, 0.0f};
	const float huge_g[3] = {1000.0f, -1000.0f, 0.0f};
	Expect(PM::FromPadData(huge_a, huge_g, 2, &s) && s.accel[0] == PM::kMaxAccel && s.gyro[0] == PM::kMaxGyro && s.gyro[1] == -PM::kMaxGyro, "clamped");
	const float nan_v[3] = {NAN, 0.0f, 0.0f};
	const float inf_v[3] = {0.0f, INFINITY, 0.0f};
	PM::Sample  before = s;
	Expect(!PM::FromPadData(nan_v, zero, 3, &s) && !PM::FromPadData(zero, inf_v, 3, &s), "non-finite refused");
	Expect(s.time_us == before.time_us && s.accel[0] == before.accel[0], "refused sample leaves the output alone");
}

void TestRateEstimator() {
	for (const Mock& m : kMocks) {
		PM::RateEstimator r;
		Expect(r.Hz() == 0.0 && r.Count() == 0, "no rate before samples");
		const uint64_t step = static_cast<uint64_t>(1.0e6 / m.rate_hz);
		uint64_t       t    = 5000000;
		for (int i = 0; i < 500; ++i) {
			r.Add(t);
			t += step;
		}
		Expect(Near(static_cast<float>(r.Hz()), m.rate_hz, m.rate_hz * 0.01f), m.name);
	}
	// A jittery 250 Hz stream (+-1 ms) still reads about 250 Hz; duplicates and time going backwards add nothing.
	PM::RateEstimator j;
	uint64_t          t = 1000;
	for (int i = 0; i < 2000; ++i) {
		j.Add(t + ((i * 7919) % 2001) - 1000);
		t += 4000;
	}
	Expect(Near(static_cast<float>(j.Hz()), 250.0f, 25.0f), "jittered rate");
	const auto n = j.Count();
	j.Add(1);
	j.Add(1);
	Expect(j.Count() == n, "samples going backwards ignored");
	// A gap longer than a second (pad asleep) does not drag the rate to zero.
	PM::RateEstimator g;
	for (int i = 0; i < 100; ++i) g.Add(1000 + i * 5000ull);
	const double hz = g.Hz();
	g.Add(1000 + 100 * 5000ull + 5000000ull);
	Expect(Near(static_cast<float>(g.Hz()), static_cast<float>(hz), 1.0f), "long gap ignored");
}

void Quat(float angle, const float axis[3], float q[4]) {
	const float s = std::sin(angle / 2);
	q[0] = axis[0] * s;
	q[1] = axis[1] * s;
	q[2] = axis[2] * s;
	q[3] = std::cos(angle / 2);
}

void TestRateFromOrientation() {
	const float axes[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
	for (int a = 0; a < 3; ++a) {
		for (float omega : {0.5f, 2.0f, -3.0f}) {
			const float dt = 0.008f;
			float       p[4], q[4], rate[3] = {};
			Quat(0.3f, axes[a], p);
			Quat(0.3f + omega * dt, axes[a], q);
			Expect(PM::RateFromOrientation(p, q, dt, rate), "rate computed");
			for (int i = 0; i < 3; ++i) {
				Expect(Near(rate[i], i == a ? omega : 0.0f, 0.01f), "rate about the rotation axis only");
			}
			// -q is the same rotation: the same rate.
			const float nq[4] = {-q[0], -q[1], -q[2], -q[3]};
			float       rate2[3];
			Expect(PM::RateFromOrientation(p, nq, dt, rate2) && Near(rate2[a], omega, 0.01f), "double cover handled");
		}
	}
	float p[4] = {0, 0, 0, 1}, q[4] = {0, 0, 0, 1}, r[3];
	Expect(!PM::RateFromOrientation(p, q, 0.0f, r) && !PM::RateFromOrientation(p, q, 0.5f, r) && !PM::RateFromOrientation(p, q, -1.0f, r), "bad dt refused");
	Expect(PM::RateFromOrientation(p, q, 0.01f, r) && r[0] == 0.0f && r[1] == 0.0f && r[2] == 0.0f, "no change, no rate");
}

void TestTraceFormat() {
	PM::Sample s;
	s.time_us  = 987654321;
	s.accel[0] = 1.25f;
	s.accel[1] = 9.80665f;
	s.accel[2] = -0.5f;
	s.gyro[0]  = 0.1f;
	s.gyro[1]  = -2.5f;
	s.gyro[2]  = 3.0f;
	PM::Sample back;
	Expect(PM::ParseTraceLine(PM::FormatTraceLine(s).c_str(), &back), "trace line parses");
	Expect(back.time_us == s.time_us && Near(back.accel[0], 1.25f, 1e-5f) && Near(back.accel[1], 9.80665f, 1e-4f) && Near(back.gyro[1], -2.5f, 1e-5f) && back.valid, "trace line round trip");
	PM::Sample junk;
	for (const char* bad : {"", "\n", "# comment", "1 2 3", "x y z", "1 nan 0 0 0 0 0", "1 0 0 0 0 0 inf", "5 1 2 3 4 5"}) {
		Expect(!PM::ParseTraceLine(bad, &junk), bad);
	}
	Expect(!PM::ParseTraceLine(nullptr, &junk), "null line");
	const std::string line = PM::FormatLogLine("ps5", 249.6, s, 1234);
	Expect(line.find("[pad] motion") == 0 && line.find("type=ps5") != std::string::npos && line.find("rate=250Hz") != std::string::npos &&
	           line.find("n=1234") != std::string::npos,
	       "'[pad] motion' line carries type, rate and count");
}

void TestPolicy() {
	Expect(PM::Wanted(nullptr, true) && !PM::Wanted(nullptr, false), "auto: on when the pad has sensors");
	Expect(PM::Wanted("auto", true) && !PM::Wanted("auto", false) && PM::Wanted("", true), "auto spelled out");
	Expect(!PM::Wanted("0", true) && !PM::Wanted("0", false), "0 forces off");
	Expect(PM::Wanted("1", false) && PM::Wanted("1", true), "1 forces on");
	// Hotplug: connect (no sensors yet), a DualSense appears, goes away, a Switch Pro appears, a pad without sensors.
	const bool has[] = {false, true, true, false, true, false};
	const bool want[] = {false, true, true, false, true, false};
	for (int i = 0; i < 6; ++i) Expect(PM::Wanted(nullptr, has[i]) == want[i], "hotplug sequence follows the active pad");
}

// Replays a recorded (or synthetic) trace through the helpers the way the relay does.
void TestReplay(const char* path, bool synthetic) {
	std::FILE* f = std::fopen(path, "r");
	Expect(f != nullptr, "trace file opens");
	if (f == nullptr) {
		return;
	}
	PM::RateEstimator r;
	std::vector<PM::Sample> samples;
	char line[256];
	int  bad = 0;
	while (std::fgets(line, sizeof(line), f) != nullptr) {
		PM::Sample s;
		if (PM::ParseTraceLine(line, &s)) {
			samples.push_back(s);
			r.Add(s.time_us);
		} else if (line[0] != '#' && line[0] != '\n') {
			++bad;
		}
	}
	std::fclose(f);
	Expect(bad == 0 && samples.size() > 100, "trace parses completely");
	bool mono = true;
	for (size_t i = 1; i < samples.size(); ++i) mono = mono && samples[i].time_us > samples[i - 1].time_us;
	Expect(mono, "trace timestamps strictly increase");
	Expect(r.Hz() >= 100.0 && r.Hz() <= 1100.0, "trace rate plausible (>= 100 Hz, handled up to 1 kHz)");
	// Gravity at rest: the mean |a| over the whole trace stays near 1 g; the peak shows the shake.
	double mean = 0, peak = 0, yaw = 0;
	for (size_t i = 0; i < samples.size(); ++i) {
		const PM::Sample& s = samples[i];
		const double      m = std::sqrt(s.accel[0] * s.accel[0] + s.accel[1] * s.accel[1] + s.accel[2] * s.accel[2]);
		mean += m / samples.size();
		peak = std::max(peak, std::fabs(static_cast<double>(s.accel[0])));
		if (i > 0) yaw += s.gyro[1] * (samples[i].time_us - samples[i - 1].time_us) * 1e-6;
	}
	Expect(mean > PM::kGravity * 0.9 && mean < PM::kGravity * 1.6, "trace mean |a| near 1 g");
	if (synthetic) {
		Expect(Near(static_cast<float>(r.Hz()), 250.0f, 2.0f), "synthetic trace: 250 Hz");
		Expect(peak > 1.8 * PM::kGravity && peak < 2.05 * PM::kGravity, "synthetic shake peak about 2 g");
		Expect(Near(static_cast<float>(yaw), 1.0f, 0.02f), "synthetic yaw integrates to 1 rad");
		// Resampled to a 60 Hz game poll: the last sample at or before each poll time; the peak survives.
		double poll_peak = 0;
		size_t k = 0;
		for (uint64_t t = samples.front().time_us; t <= samples.back().time_us; t += 16667) {
			while (k + 1 < samples.size() && samples[k + 1].time_us <= t) ++k;
			poll_peak = std::max(poll_peak, std::fabs(static_cast<double>(samples[k].accel[0])));
		}
		Expect(poll_peak > 1.5 * PM::kGravity, "a 60 Hz poll still sees the shake");
	}
}
} // namespace

// The neutral default pose is never published as live data while the pad claims sensors.
void TestNeutralNotPublished() {
	PM::Sample neutral; // Sample's default: accel (0, g, 0), gyro 0
	Expect(PM::IsNeutralDefault(neutral), "the default sample is the neutral pose");
	PM::Sample real = neutral;
	real.gyro[2]    = 0.017f;
	Expect(!PM::IsNeutralDefault(real), "a sample with gyro noise is not neutral");
	real      = neutral;
	real.accel[0] = -0.32f;
	Expect(!PM::IsNeutralDefault(real), "a tilted sample is not neutral");
	Expect(PM::PlanRelay(true, false).motion_flagged, "claimed sensors without live data: flagged");
	Expect(!PM::PlanRelay(true, true).motion_flagged, "claimed sensors with live data: not flagged");
	Expect(!PM::PlanRelay(false, false).motion_flagged, "emulated tilt (no sensors) is not flagged");
	for (bool has: {false, true}) {
		for (bool live: {false, true}) {
			const auto plan = PM::PlanRelay(has, live);
			Expect(plan.publish_state && plan.publish_motion, "pad state is never held back");
		}
	}
	const std::string line = PM::FormatStatusLine(true, false, 0, 0, 0, 0, 0, true);
	Expect(line.find("live=NO") != std::string::npos && line.find("NEUTRAL-DEFAULT") != std::string::npos,
	       "the status line flags dead sensors");
	const std::string ok = PM::FormatStatusLine(true, true, 10, 10, 20, 0, 1, false);
	Expect(ok.find("live=yes") != std::string::npos && ok.find("rearms=1") != std::string::npos,
	       "the status line shows live sensors");
	// gyrofable: the line says whether SDL delivers pad events at all (keyboard focus / background input).
	Expect(ok.find("focus=? bg_input=?") != std::string::npos, "focus is '?' when not passed");
	const std::string dropped = PM::FormatStatusLine(true, false, 0, 0, 0, 0, 0, true, 0, 100, 0, 0);
	Expect(dropped.find("focus=NO bg_input=0") != std::string::npos, "the status line names the focus drop");
	const std::string bg = PM::FormatStatusLine(true, true, 5, 5, 10, 0, 0, false, 0, 100, 0, 1);
	Expect(bg.find("focus=NO bg_input=1") != std::string::npos, "the status line shows background input");
}

// padFocus.h: the pad is taken without keyboard focus unless KYTY_PAD_BACKGROUND_INPUT=0, and the
// model of SDL's rule (what the relay's focus line reports).
void TestPadFocusPolicy() {
	using namespace Libs::Graphics;
	Expect(PadBackgroundInputWanted(nullptr), "background pad input is on by default");
	Expect(PadBackgroundInputWanted("1") && PadBackgroundInputWanted("") && PadBackgroundInputWanted("00"),
	       "only the exact value 0 turns background pad input off");
	Expect(!PadBackgroundInputWanted("0"), "KYTY_PAD_BACKGROUND_INPUT=0 turns it off");
	Expect(!PadEventsDelivered(false, true, false), "SDL default, window without focus: pad events dropped");
	Expect(PadEventsDelivered(false, true, true), "SDL default, focused window: delivered");
	Expect(PadEventsDelivered(false, false, false), "no window (tools, tests): delivered");
	Expect(PadEventsDelivered(true, true, false), "background input: delivered without focus");
	Expect(ApplyPadBackgroundInput(nullptr), "the hint is set on by default");
	const char* hint = SDL_GetHint(SDL_HINT_JOYSTICK_ALLOW_BACKGROUND_EVENTS);
	Expect(hint != nullptr && hint[0] == '1', "SDL's hint reads 1 after the default");
	Expect(!ApplyPadBackgroundInput("0"), "the hint is set off with 0");
	hint = SDL_GetHint(SDL_HINT_JOYSTICK_ALLOW_BACKGROUND_EVENTS);
	Expect(hint != nullptr && hint[0] == '0', "SDL's hint reads 0 after KYTY_PAD_BACKGROUND_INPUT=0");
}

int main(int argc, char** argv) {
	TestNeutralNotPublished();
	TestPadFocusPolicy();
	TestTypeNames();
	TestConversionPerController();
	TestRateEstimator();
	TestRateFromOrientation();
	TestTraceFormat();
	TestPolicy();
	TestReplay(PAD_MOTION_TRACE, true);
	// A real recording (ASTROHYBRID_PAD_MOTION_REC) replays through the same checks: PadMotionTests <trace>.
	if (argc > 1) {
		TestReplay(argv[1], false);
	}
	std::printf("PadMotionTests: %d checks, %d failures\n", g_checks, g_failures);
	return g_failures == 0 ? 0 : 1;
}
