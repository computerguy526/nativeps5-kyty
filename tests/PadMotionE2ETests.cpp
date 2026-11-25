// End-to-end gyro regression (USER 2026-11-25: "add a gyro regression test so this never breaks
// again"). A real SDL3 virtual joystick presents as a DualSense and as a Switch Pro and sends 1000 Hz
// accel + gyro (a tilt ramp, a shake spike) plus a button press, through SDL's real event queue,
// the window loop's real per-step event drain (DrainEvents), the real sensor dispatch
// (DispatchGamepadSensor), the production controller.cpp (Connect -> SetDevices -> Motion ->
// pad_fill_data) and PadReadState / PadGetMotionStatus. The loop is throttled to ~60 steps/s like the
// stepped hybrid window loop. Not covered: WindowContext itself (needs a Vulkan window) and
// hostInput's button mapping (the test maps South to Cross).
// Includes a negative control: the old one-event-per-step loop MUST fail the same checks.
#include <SDL3/SDL.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <thread>

#include "graphics/presentation/window/eventDrain.h"
#include "graphics/presentation/window/gamepadSensorDispatch.h"
#include "hybrid/virtualPad.h"

#include "libs/controller.cpp"

namespace Libs::Controller {
int PadVrPortHand(int) { return -1; }
int PadVrPort(int, int, int, bool) { return PAD_ERROR_INVALID_ARG; }
int PadVrCheckHandle(int) { return PAD_ERROR_INVALID_HANDLE; }
int PadVrClose(int) { return PAD_ERROR_INVALID_HANDLE; }
} // namespace Libs::Controller
namespace Libs::Controller::DualSenseHaptics {
bool SetVibration(int, uint8_t, uint8_t, uint32_t) { return false; }
void Shutdown() {}
} // namespace Libs::Controller::DualSenseHaptics
namespace Libs::LibKernel {
uint64_t KYTY_SYSV_ABI KernelGetProcessTime() {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}
} // namespace Libs::LibKernel
namespace Loader::Timer {
double GetTimeMs() {
	return static_cast<double>(SDL_GetTicks());
}
} // namespace Loader::Timer

namespace {
using namespace Libs::Controller;
namespace VP = Hybrid::VirtualPad;

int g_failures = 0;
void Check(bool ok, const char* what, int detail = 0) {
	if (!ok) {
		std::fprintf(stderr, "PadMotionE2ETests: FAILED: %s (%d)\n", what, detail);
		g_failures++;
	}
}

constexpr int kStepMs    = 16; // ~62 steps/s
constexpr int kSteps     = 190; // ~3 s: the whole ramp, rest, the spike at 1.2 s
constexpr int kButtonOn  = 40;
constexpr int kButtonOff = 55;

struct Result {
	int      live_after_steps = -1; // steps after connect until live
	int      max_backlog      = 0;  // queued sensor events at a step end
	int      worst_lag_samples = 0; // injected - applied accel samples at a step end
	float    max_roll_error_deg = 0.0f;
	float    max_accel_error_g  = 0.0f;
	float    peak_az_g = 0.0f, peak_gx = 0.0f, peak_orientation_x = 0.0f;
	bool     button_in_step = false, button_released = false;
	uint64_t unrouted = 0;
};

// One run of the harness; per_step_budget 1 = the old window loop (negative control).
Result Run(VP::Kind kind, int per_step_budget, int steps) {
	Result r;
	Initialize();
	Check(PadInit() == 0 && PadOpen(Config::GetUserId(), 0, 0, nullptr) == 1, "pad port did not open");

	VP::Pad pad;
	Check(VP::Attach(kind, &pad), "virtual joystick did not attach");

	bool     connected = false;
	int      connect_step = -1;
	uint64_t injected = 0, sim_ms = 0;
	const uint64_t base_ns = 1000000000ull;
	bool     button_sent = false;

	for (int step = 0; step < steps; step++) {
		const auto step_start = std::chrono::steady_clock::now();
		// The pad sends one step's worth of samples (1 kHz) once the game's pad is connected.
		if (connected) {
			for (int i = 0; i < kStepMs; i++) {
				VP::SendSample(pad, sim_ms++, base_ns);
				injected++;
			}
		}
		if (connected && step == connect_step + kButtonOn) {
			SDL_SetJoystickVirtualButton(pad.joystick, SDL_GAMEPAD_BUTTON_SOUTH, true);
			button_sent = true;
		}
		if (connected && step == connect_step + kButtonOff) {
			SDL_SetJoystickVirtualButton(pad.joystick, SDL_GAMEPAD_BUTTON_SOUTH, false);
		}
		SDL_Event ev;
		Libs::Graphics::DrainEvents(
		    [&](int timeout) { return timeout > 0 ? SDL_WaitEventTimeout(&ev, timeout) : SDL_PollEvent(&ev); },
		    [&] {
			    switch (ev.type) {
				    case SDL_EVENT_GAMEPAD_ADDED: {
					    auto* gp = SDL_OpenGamepad(ev.gdevice.which);
					    if (gp != nullptr) {
						    Connect(static_cast<int>(SDL_GetJoystickID(SDL_GetGamepadJoystick(gp))));
						    connected    = true;
						    connect_step = step + 1; // samples start with the next step
					    }
					    break;
				    }
				    case SDL_EVENT_GAMEPAD_SENSOR_UPDATE: Libs::Graphics::DispatchGamepadSensor(ev.gsensor); break;
				    case SDL_EVENT_GAMEPAD_BUTTON_DOWN:
				    case SDL_EVENT_GAMEPAD_BUTTON_UP:
					    if (ev.gbutton.button == SDL_GAMEPAD_BUTTON_SOUTH) {
						    SetButton(static_cast<int>(ev.gbutton.which), PAD_BUTTON_CROSS, ev.gbutton.down);
					    }
					    break;
				    default: break;
			    }
		    },
		    [] {
			    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
			                                     std::chrono::steady_clock::now().time_since_epoch())
			                                     .count());
		    },
		    kStepMs / 2, per_step_budget);

		if (connected && step >= connect_step) {
			MotionStatus status {};
			PadData      data {};
			Check(PadGetMotionStatus(1, &status) && PadReadState(1, &data) == 0, "pad read failed");
			if (r.live_after_steps < 0 && status.live) {
				r.live_after_steps = step - connect_step + 1;
			}
			r.max_backlog = std::max(r.max_backlog, static_cast<int>(std::max(
			                    0, SDL_PeepEvents(nullptr, 0, SDL_PEEKEVENT, SDL_EVENT_GAMEPAD_SENSOR_UPDATE,
			                                      SDL_EVENT_GAMEPAD_SENSOR_UPDATE))));
			r.worst_lag_samples = std::max(r.worst_lag_samples, static_cast<int>(injected - status.accel_events));
			r.unrouted = status.events_unrouted;
			// What the pad sent last (sim_ms - 1) vs what the game reads now.
			const VP::Pose want = VP::PoseAt(sim_ms - 1);
			if (status.accel_events == injected) {
				// The ramp (and the rest before the spike): the shake rotates about x on purpose, so the
				// orientation is only compared with the ramp's roll up to 1.15 s.
				if ((sim_ms - 1) % 3000 < 1150) {
					const float roll = 2.0f * std::atan2(data.orientation_z, data.orientation_w);
					r.max_roll_error_deg = std::max(
					    r.max_roll_error_deg, std::fabs(roll - want.roll_rad) * 57.29578f);
				}
				for (int i = 0; i < 3; i++) {
					const float g = (i == 0 ? data.acceleration_x : i == 1 ? data.acceleration_y : data.acceleration_z);
					r.max_accel_error_g = std::max(r.max_accel_error_g, std::fabs(g - want.accel[i] / VP::kG));
				}
			}
			r.peak_az_g = std::max(r.peak_az_g, data.acceleration_z);
			r.peak_gx   = std::max(r.peak_gx, data.angular_velocity_x);
			r.peak_orientation_x = std::max(r.peak_orientation_x, std::fabs(data.orientation_x));
			if (button_sent && step == connect_step + kButtonOn && (data.buttons & PAD_BUTTON_CROSS) != 0) {
				r.button_in_step = true;
			}
			if (step == connect_step + kButtonOff && (data.buttons & PAD_BUTTON_CROSS) == 0) {
				r.button_released = true;
			}
		}
		// Throttle to the step rate.
		std::this_thread::sleep_until(step_start + std::chrono::milliseconds(kStepMs));
	}
	SDL_CloseJoystick(pad.joystick);
	SDL_DetachVirtualJoystick(pad.id);
	SDL_PumpEvents();
	SDL_FlushEvents(SDL_EVENT_FIRST, SDL_EVENT_LAST);
	Shutdown();
	return r;
}

void TestKind(VP::Kind kind, const char* name) {
	std::fprintf(stderr, "PadMotionE2ETests: %s\n", name);
	const Result r = Run(kind, Libs::Graphics::kEventDrainBudget, kSteps);
	std::fprintf(stderr,
	             "  live after %d steps, max backlog %d, worst lag %d samples, roll err %.2f deg, accel err %.3f g, "
	             "peak az %.2f g, peak gx %.1f rad/s, unrouted %llu\n",
	             r.live_after_steps, r.max_backlog, r.worst_lag_samples, static_cast<double>(r.max_roll_error_deg),
	             static_cast<double>(r.max_accel_error_g), static_cast<double>(r.peak_az_g),
	             static_cast<double>(r.peak_gx), static_cast<unsigned long long>(r.unrouted));
	Check(r.live_after_steps >= 1 && r.live_after_steps <= 6, "motion not live within 100 ms of connect",
	      r.live_after_steps);
	Check(r.max_backlog < 64, "sensor event backlog grows", r.max_backlog);
	Check(r.worst_lag_samples == 0, "the game's pad lags the pad by samples", r.worst_lag_samples);
	Check(r.max_roll_error_deg < 4.0f, "orientation does not track the ramp", static_cast<int>(r.max_roll_error_deg * 10));
	Check(r.peak_orientation_x > 0.1f, "the shake did not move the orientation");
	Check(r.max_accel_error_g < 0.02f, "accel does not track the ramp");
	Check(r.peak_az_g > 2.5f && r.peak_gx > 9.0f, "the shake spike is not visible");
	Check(r.button_in_step, "a button was not handled in its own step despite the sensor flood");
	Check(r.button_released, "a button release was not handled in its own step");
	Check(r.unrouted == 0, "sensor events were unrouted");

	// Negative control: one event per step (the old loop) must fail the same checks.
	const Result old_loop = Run(kind, 1, 120);
	std::fprintf(stderr, "  old loop: max backlog %d, worst lag %d samples, button in step %d\n", old_loop.max_backlog,
	             old_loop.worst_lag_samples, old_loop.button_in_step ? 1 : 0);
	Check(old_loop.max_backlog > 500 && old_loop.worst_lag_samples > 500 && !old_loop.button_in_step,
	      "the old one-event-per-step loop was not detected by this test");
}
} // namespace

int main() {
	Config::Initialize();
	SDL_SetHint(SDL_HINT_JOYSTICK_ALLOW_BACKGROUND_EVENTS, "1");
	SDL_SetHint(SDL_HINT_JOYSTICK_HIDAPI, "0");
	SDL_SetHint(SDL_HINT_JOYSTICK_MFI, "0");
	if (!SDL_Init(SDL_INIT_GAMEPAD)) {
		std::fprintf(stderr, "PadMotionE2ETests: SDL_Init failed: %s\n", SDL_GetError());
		return 1;
	}
	TestKind(VP::Kind::PS5, "virtual DualSense");
	TestKind(VP::Kind::SwitchPro, "virtual Switch Pro");
	SDL_Quit();
	Config::Shutdown();
	std::fprintf(stderr, "PadMotionE2ETests: %d failures\n", g_failures);
	return g_failures == 0 ? 0 : 1;
}
