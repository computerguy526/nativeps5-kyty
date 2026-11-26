// Pad input without keyboard focus (gyrofable, 2026-11-25).
//
// SDL's default (SDL_HINT_JOYSTICK_ALLOW_BACKGROUND_EVENTS unset = "0"): while the process has a
// window but that window is not the key window, SDL drops button presses, stick moves away from the
// centre and EVERY motion sensor sample (SDL_SendJoystickSensor returns before it stores the
// sample, so SDL_GetGamepadSensorData stays frozen too, while SDL_GamepadSensorEnabled still says
// "on" and the pad keeps sending reports). A game started from a launcher, a terminal or a script
// opens its window WITHOUT keyboard focus on macOS until the player clicks it, and loses it
// whenever the player looks at another window: the game then saw a pad with sensors that never
// delivered a sample
// ("sensors=yes live=NO, sdl_events total=0"), and tilt / shake / steering did nothing. Measured
// with a standalone SDL 3.4.18 probe on the DualSense: 250 raw HID reports/s, 0 sensor events until
// the probe's window was clicked; all events with the hint on.
//
// A pad is not shared with other applications the way a keyboard is, so the library takes pad input
// (buttons, sticks, sensors) whether or not its window has the focus. KYTY_PAD_BACKGROUND_INPUT=0
// restores SDL's default. A wheel configuration that is loaded later sets the hint from its own
// background_input field (libs/wheelHost.cpp).
#pragma once

#include <SDL3/SDL.h>
#include <cstdlib>

namespace Libs::Graphics {

// env: the value of KYTY_PAD_BACKGROUND_INPUT (null = unset). Only "0" turns it off.
inline bool PadBackgroundInputWanted(const char* env) {
	return !(env != nullptr && env[0] == '0' && env[1] == '\0');
}

// Sets the SDL hint (SDL applies it at once, also after the joystick subsystem started).
// Returns what was set.
inline bool ApplyPadBackgroundInput(const char* env) {
	const bool on = PadBackgroundInputWanted(env);
	SDL_SetHint(SDL_HINT_JOYSTICK_ALLOW_BACKGROUND_EVENTS, on ? "1" : "0");
	return on;
}
inline bool ApplyPadBackgroundInput() {
	return ApplyPadBackgroundInput(std::getenv("KYTY_PAD_BACKGROUND_INPUT"));
}

// What SDL does with a pad event right now: true = delivered. (SDL_joystick.c
// SDL_PrivateJoystickShouldIgnoreEvent: ignored when the hint is off, the process has a window and
// no window has the keyboard focus.) has_window: the process created an SDL window.
inline bool PadEventsDelivered(bool background_input, bool has_window, bool keyboard_focus) {
	return background_input || !has_window || keyboard_focus;
}

} // namespace Libs::Graphics
