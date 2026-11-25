// Motion sensors of the host's pad, relayed to the game (hybrid bridge). Pure helpers: no SDL
// calls, so they are unit-tested without a controller (tests/PadMotionTests.cpp).
//
// SDL3 normalises every gamepad's sensors to one coordinate system (right-handed, x right, y up, z
// towards the player when the pad is held flat in front of you; gravity at rest on a table is +y;
// gyro in rad/s, accelerometer in m/s^2), so DualSense, DualShock 4, Switch Pro Controller and
// Joy-Cons need no per-model axis code; the host pad library (libScePad) fuses the orientation from
// these values itself.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace Hybrid::PadMotion {

constexpr float kGravity = 9.80665f;

struct Sample {
	float    accel[3] {0.0f, kGravity, 0.0f}; // m/s^2 including gravity (SDL axes)
	float    gyro[3] {};                      // rad/s
	uint64_t time_us = 0;                     // monotonic, microseconds
	bool     valid   = false;                 // the pad has sensors and delivered a reading
};

// Plausibility window for a sample: a DualSense reports +-4 g / +-2000 deg/s; anything beyond is a
// glitch.
constexpr float kMaxAccel = 16.0f * kGravity;
constexpr float kMaxGyro  = 40.0f; // rad/s (2290 deg/s)

inline bool Finite(const float* v, int n) {
	for (int i = 0; i < n; ++i) {
		if (!std::isfinite(v[i])) {
			return false;
		}
	}
	return true;
}

// Kyty's pad data (accel in g, gyro rad/s) -> a relay sample. Returns false for non-finite
// readings.
inline bool FromPadData(const float accel_g[3], const float gyro[3], uint64_t time_us,
                        Sample* out) {
	if (!Finite(accel_g, 3) || !Finite(gyro, 3)) {
		return false;
	}
	for (int i = 0; i < 3; ++i) {
		out->accel[i] = std::clamp(accel_g[i] * kGravity, -kMaxAccel, kMaxAccel);
		out->gyro[i]  = std::clamp(gyro[i], -kMaxGyro, kMaxGyro);
	}
	out->time_us = time_us;
	out->valid   = true;
	return true;
}

// Name of an SDL_GamepadType (numeric value; the enum is stable API) for the log line.
inline const char* TypeName(int sdl_type) {
	switch (sdl_type) {
		case 4: return "ps3";
		case 5: return "ps4";
		case 6: return "ps5";
		case 7: return "switch-pro";
		case 8: return "switch-joycon-left";
		case 9: return "switch-joycon-right";
		case 10: return "switch-joycon-pair";
		case 2: return "xbox360";
		case 3: return "xboxone";
		case 1: return "standard";
		default: return "other";
	}
}

// Measured sensor rate (Hz): exponential average of the interval between distinct samples.
class RateEstimator {
public:
	void Add(uint64_t time_us) {
		if (m_last != 0 && time_us > m_last) {
			const double dt = static_cast<double>(time_us - m_last);
			if (dt < 1.0e6) {
				m_interval = m_count == 0 ? dt : m_interval * 0.95 + dt * 0.05;
				++m_count;
			}
		}
		m_last = std::max(m_last, time_us);
	}
	double   Hz() const { return m_count > 0 && m_interval > 0.0 ? 1.0e6 / m_interval : 0.0; }
	uint64_t Count() const { return m_count; }

private:
	uint64_t m_last     = 0;
	uint64_t m_count    = 0;
	double   m_interval = 0.0;
};

// Angular velocity from the change of an orientation quaternion (x y z w), for pads that report
// orientation but no gyro (Kyty's emulated tilt). Body-frame rate from conj(prev) * q. dt in
// seconds.
inline bool RateFromOrientation(const float prev[4], const float q[4], float dt, float rate[3]) {
	if (!(dt > 0.0005f && dt < 0.1f)) {
		return false;
	}
	const float w    = prev[3] * q[3] + prev[0] * q[0] + prev[1] * q[1] + prev[2] * q[2];
	const float d[3] = {prev[3] * q[0] - prev[0] * q[3] - prev[1] * q[2] + prev[2] * q[1],
	                    prev[3] * q[1] + prev[0] * q[2] - prev[1] * q[3] - prev[2] * q[0],
	                    prev[3] * q[2] - prev[0] * q[1] + prev[1] * q[0] - prev[2] * q[3]};
	const float sign = w < 0.0f ? -1.0f : 1.0f;
	for (int i = 0; i < 3; ++i) {
		rate[i] = sign * 2.0f * d[i] / dt;
	}
	return true;
}

// ---- trace (ASTROHYBRID_PAD_MOTION_REC=<file>): one sample per line, "<time_us> ax ay az gx gy
// gz", SDL units.
inline std::string FormatTraceLine(const Sample& s) {
	char line[160];
	std::snprintf(line, sizeof(line), "%llu %.6f %.6f %.6f %.6f %.6f %.6f\n",
	              static_cast<unsigned long long>(s.time_us), s.accel[0], s.accel[1], s.accel[2],
	              s.gyro[0], s.gyro[1], s.gyro[2]);
	return line;
}
inline bool ParseTraceLine(const char* line, Sample* s) {
	if (line == nullptr || line[0] == '#' || line[0] == '\n' || line[0] == '\0') {
		return false;
	}
	unsigned long long t = 0;
	float              v[6];
	if (std::sscanf(line, "%llu %f %f %f %f %f %f", &t, &v[0], &v[1], &v[2], &v[3], &v[4], &v[5]) !=
	        7 ||
	    !Finite(v, 6)) {
		return false;
	}
	std::memcpy(s->accel, v, sizeof(float) * 3);
	std::memcpy(s->gyro, v + 3, sizeof(float) * 3);
	s->time_us = t;
	s->valid   = true;
	return true;
}

// The 1 Hz '[pad] motion' line (ASTROHYBRID_PAD_MOTION_LOG=1).
inline std::string FormatLogLine(const char* type, double rate_hz, const Sample& s,
                                 uint64_t samples) {
	char line[256];
	std::snprintf(
	    line, sizeof(line),
	    "[pad] motion type=%s rate=%.0fHz n=%llu accel=%.2f,%.2f,%.2f m/s^2 (|a|=%.2f) "
	    "gyro=%.3f,%.3f,%.3f rad/s",
	    type, rate_hz, static_cast<unsigned long long>(samples), s.accel[0], s.accel[1], s.accel[2],
	    std::sqrt(s.accel[0] * s.accel[0] + s.accel[1] * s.accel[1] + s.accel[2] * s.accel[2]),
	    s.gyro[0], s.gyro[1], s.gyro[2]);
	return line;
}

// The default pose a Kyty pad reports before any sensor event arrived: accel exactly (0, 1 g, 0),
// gyro exactly 0. A real sensor never reads exactly this (it always has noise).
inline bool IsNeutralDefault(const Sample& s) {
	return s.accel[0] == 0.0f && s.accel[1] == kGravity && s.accel[2] == 0.0f && s.gyro[0] == 0.0f &&
	       s.gyro[1] == 0.0f && s.gyro[2] == 0.0f;
}

// What the relay sends while a pad claims sensors but SDL delivers none: everything, as always.
// Buttons, sticks, triggers, touch and the connection are NEVER held back for motion (a held-back
// pad state starved the game of input, rc11). The motion fields then carry Kyty's neutral default
// pose; that is flagged (motion_flagged: status log line), not suppressed.
struct RelayPlan {
	bool publish_state  = true; // buttons, sticks, triggers, touch, connected: always
	bool publish_motion = true;
	bool motion_flagged = false; // sensors claimed but not live: the motion values are the default pose
};
inline RelayPlan PlanRelay(bool pad_has_sensors, bool live) {
	RelayPlan plan;
	plan.motion_flagged = pad_has_sensors && !live;
	return plan;
}

// The status part of the 1 Hz log line: how many SDL sensor events arrived, whether they are live
// now, and whether the relay is holding the samples back.
inline std::string FormatStatusLine(bool pad_has_sensors, bool live, uint64_t accel_events,
                                    uint64_t gyro_events, uint64_t events_total,
                                    uint64_t events_unrouted, uint64_t rearms, bool flagged,
                                    uint64_t queued = 0, uint64_t window_steps = 0,
                                    int keyboard_focus = -1, int background_input = -1) {
	// keyboard_focus / background_input: 1, 0 or -1 (not known). focus=NO with bg_input=0 is SDL
	// dropping every pad event (padFocus.h): the reason for live=NO, not the pad.
	char line[320];
	std::snprintf(line, sizeof(line),
	              "[pad] motion status: sensors=%s live=%s motion=%s sdl_events accel=%llu "
	              "gyro=%llu total=%llu unrouted=%llu rearms=%llu sdl_queued=%llu window_steps=%llu "
	              "focus=%s bg_input=%s",
	              pad_has_sensors ? "yes" : "no", live ? "yes" : "NO",
	              flagged ? "NEUTRAL-DEFAULT(no sensor data)" : "ok", static_cast<unsigned long long>(accel_events),
	              static_cast<unsigned long long>(gyro_events),
	              static_cast<unsigned long long>(events_total),
	              static_cast<unsigned long long>(events_unrouted),
	              static_cast<unsigned long long>(rearms), static_cast<unsigned long long>(queued),
	              static_cast<unsigned long long>(window_steps),
	              keyboard_focus < 0 ? "?" : (keyboard_focus != 0 ? "yes" : "NO"),
	              background_input < 0 ? "?" : (background_input != 0 ? "1" : "0"));
	return line;
}

// ASTROHYBRID_PAD_MOTION: unset/"auto" = on when the pad has sensors; "0" = off; "1" = on.
inline bool Wanted(const char* env, bool pad_has_sensors) {
	if (env != nullptr && env[0] == '0') {
		return false;
	}
	if (env != nullptr && env[0] == '1') {
		return true;
	}
	return pad_has_sensors;
}

// v1.4: a host library without the pad-motion export (an older libnps5_gpu picked up through the binary's LC_RPATH when
// DYLD_LIBRARY_PATH named a directory that did not exist) silently dropped every gyro sample. Returns the one-time ERROR text when the
// motion relay is wanted (the pad has sensors, not switched off) but the host cannot receive it; empty when all is well.
inline std::string HostMotionProblem(const char* env, bool pad_has_sensors, bool host_has_publish, const char* dyld_path) {
	if (!Wanted(env, pad_has_sensors) || host_has_publish) {
		return {};
	}
	return std::string("ERROR: the pad has motion sensors but the host cannot receive motion (no Nps5KytyHostPublishPadMotion): "
	                   "the host library in use is not this release's; DYLD_LIBRARY_PATH=") +
	       (dyld_path != nullptr && dyld_path[0] != '\0' ? dyld_path : "(unset)");
}

} // namespace Hybrid::PadMotion
