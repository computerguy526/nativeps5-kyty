// A virtual DualSense / Switch Pro made with SDL3's virtual joystick (SDL_AttachVirtualJoystick)
// that sends accel + gyro at 1000 Hz (a tilt ramp, then a shake spike, repeating) through SDL's
// real event path. Used by tests/PadMotionE2ETests.cpp and, behind
// ASTROHYBRID_PAD_VIRTUAL=ps5|switch (default off), by a real title run (kytyGpuBridge.cpp): the
// same motion status line must then say live=yes (work/release/v140/rc/check-motion-live.sh,
// rehearse R2).
#pragma once

#include <SDL3/SDL.h>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <thread>

namespace Hybrid::VirtualPad {

enum class Kind { PS5, SwitchPro };

constexpr float kG = 9.80665f;

// The motion every virtual pad plays, t in ms since it started: roll 0 -> 45 degrees over 1 s (gyro
// z = the ramp rate), rest, a 60 ms shake spike (3 g on z, 10 rad/s on x) at 1.2 s, rest to 3 s,
// repeat.
struct Pose {
	float accel[3];
	float gyro[3];
	float roll_rad; // the angle the ramp has turned through (the orientation to expect)
	bool  spike;
};
inline Pose PoseAt(uint64_t t_ms) {
	const float    pi = 3.14159265358979f;
	const uint64_t t  = t_ms % 3000;
	Pose           p {};
	const float    ramp_end = pi / 4.0f;
	if (t < 1000) {
		p.roll_rad = ramp_end * static_cast<float>(t) / 1000.0f;
		p.gyro[2]  = ramp_end; // rad/s, 1 s ramp
	} else {
		p.roll_rad = ramp_end;
	}
	// gravity in the pad's frame after rolling about z: (sin, cos, 0)
	p.accel[0] = kG * std::sin(p.roll_rad);
	p.accel[1] = kG * std::cos(p.roll_rad);
	if (t >= 1200 && t < 1260) {
		p.spike    = true;
		p.accel[2] = 3.0f * kG;
		p.gyro[0]  = 10.0f;
	}
	return p;
}

struct Pad {
	SDL_Joystick*  joystick = nullptr;
	SDL_JoystickID id       = 0;
	Kind           kind     = Kind::PS5;
};

inline bool Attach(Kind kind, Pad* pad) {
	static const SDL_VirtualJoystickSensorDesc sensors[2] = {{SDL_SENSOR_ACCEL, 1000.0f},
	                                                         {SDL_SENSOR_GYRO, 1000.0f}};
	SDL_VirtualJoystickDesc                    desc;
	SDL_INIT_INTERFACE(&desc);
	desc.type        = SDL_JOYSTICK_TYPE_GAMEPAD;
	desc.vendor_id   = kind == Kind::PS5 ? 0x054c : 0x057e;
	desc.product_id  = kind == Kind::PS5 ? 0x0ce6 : 0x2009;
	desc.naxes       = 6;
	desc.nbuttons    = 15;
	desc.button_mask = (1u << 15) - 1;
	desc.axis_mask   = 0x3f;
	desc.nsensors    = 2;
	desc.sensors     = sensors;
	desc.name        = kind == Kind::PS5 ? "Virtual DualSense" : "Virtual Switch Pro";
	pad->kind        = kind;
	pad->id          = SDL_AttachVirtualJoystick(&desc);
	if (pad->id == 0) {
		return false;
	}
	pad->joystick = SDL_OpenJoystick(pad->id);
	return pad->joystick != nullptr;
}

// One 1 kHz sample of the pose at t_ms (sensor timestamps in ns on a clock starting at base_ns).
inline void SendSample(const Pad& pad, uint64_t t_ms, uint64_t base_ns) {
	const Pose p  = PoseAt(t_ms);
	const auto ts = base_ns + t_ms * 1000000ull;
	(void)SDL_SendJoystickVirtualSensorData(pad.joystick, SDL_SENSOR_ACCEL, ts, p.accel, 3);
	(void)SDL_SendJoystickVirtualSensorData(pad.joystick, SDL_SENSOR_GYRO, ts, p.gyro, 3);
}

// Title-run hook: a thread that plays the pose in real time (1 ms per sample).
class Player {
public:
	bool Start(Kind kind) {
		if (!Attach(kind, &m_pad)) {
			return false;
		}
		m_stop   = false;
		m_thread = std::thread([this] {
			const auto start = std::chrono::steady_clock::now();
			uint64_t   sent  = 0;
			while (!m_stop) {
				const auto elapsed =
				    static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
				                              std::chrono::steady_clock::now() - start)
				                              .count());
				for (; sent <= elapsed; ++sent) {
					SendSample(m_pad, sent, 1000000000ull);
				}
				std::this_thread::sleep_for(std::chrono::milliseconds(1));
			}
		});
		return true;
	}
	void Stop() {
		m_stop = true;
		if (m_thread.joinable()) {
			m_thread.join();
		}
	}
	~Player() { Stop(); }

private:
	Pad               m_pad;
	std::atomic<bool> m_stop {true};
	std::thread       m_thread;
};

// ASTROHYBRID_PAD_VIRTUAL: "ps5" or "switch" (anything else: off).
inline bool ParseEnv(const char* value, Kind* kind) {
	if (value == nullptr) {
		return false;
	}
	if (std::strcmp(value, "ps5") == 0) {
		*kind = Kind::PS5;
		return true;
	}
	if (std::strcmp(value, "switch") == 0) {
		*kind = Kind::SwitchPro;
		return true;
	}
	return false;
}

} // namespace Hybrid::VirtualPad
