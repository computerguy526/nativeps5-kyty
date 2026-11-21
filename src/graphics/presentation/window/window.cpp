#include "graphics/presentation/window.h"

#include "common/assert.h"
#include "common/common.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/stringUtils.h"
#include "common/systemInfo.h"
#include "common/threads.h"
#include "common/timer.h"
#include "graphics/compileProgress.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/presentation/autoCap.h"
#include "graphics/presentation/performanceOverlay.h"
#include "graphics/presentation/renderDoc.h"
#include "graphics/presentation/systemOverlay.h"
#include "graphics/presentation/videoOut.h"
#include "graphics/presentation/window/hostInput.h"
#include "graphics/presentation/window/windowInternal.h"
#include "kytyGitVersion.h"
#include "libs/controller.h"
#include "loader/systemContent.h"

#include <SDL3/SDL.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <filesystem>
#include <fmt/format.h>
#include <memory>
#include <mutex>
#include <objc/message.h>
#include <objc/runtime.h>
#include <string>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>
#include <vulkan/vk_platform.h>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_SIMD
#include "stb_image.h"

// IWYU pragma: no_include <intrin.h>

#define KYTY_DBG_INPUT

namespace Libs::Graphics {

struct EventKeyboard {
	bool     down;
	bool     up;
	bool     pressed;
	bool     released;
	bool     repeat;
	int      scan_code;
	int      key_code;
	uint16_t mod;
	double   timestamp_seconds;
};


static Controller::Axis ControllerAxisFromSdl(int axis_id) {
	switch (axis_id) {
		case SDL_GAMEPAD_AXIS_LEFTX: return Controller::Axis::LeftX;
		case SDL_GAMEPAD_AXIS_LEFTY: return Controller::Axis::LeftY;
		case SDL_GAMEPAD_AXIS_RIGHTX: return Controller::Axis::RightX;
		case SDL_GAMEPAD_AXIS_RIGHTY: return Controller::Axis::RightY;
		case SDL_GAMEPAD_AXIS_LEFT_TRIGGER: return Controller::Axis::TriggerLeft;
		case SDL_GAMEPAD_AXIS_RIGHT_TRIGGER: return Controller::Axis::TriggerRight;
		default: return Controller::Axis::AxisMax;
	}
}

static bool ControllerAxisIsTrigger(int axis_id) {
	return axis_id == SDL_GAMEPAD_AXIS_LEFT_TRIGGER ||
	       axis_id == SDL_GAMEPAD_AXIS_RIGHT_TRIGGER;
}

static int ControllerAxisValueFromSdl(int axis_id, int axis_value) {
	return ControllerAxisIsTrigger(axis_id)
	           ? Controller::controller_get_axis(0, SDL_JOYSTICK_AXIS_MAX, axis_value)
	           : Controller::controller_get_axis(SDL_JOYSTICK_AXIS_MIN, SDL_JOYSTICK_AXIS_MAX,
	                                             axis_value);
}

struct EventMouse {
	bool   down;
	bool   up;
	bool   left;
	bool   middle;
	bool   right;
	bool   x1;
	bool   x2;
	bool   touch;
	bool   pressed;
	bool   released;
	int    num_of_clicks;
	bool   wheel;
	int    x;
	int    y;
	bool   motion;
	int    motion_x;
	int    motion_y;
	double timestamp_seconds;
};

struct EventFinger {
	bool   down;
	bool   up;
	bool   motion;
	int    touch_id;
	int    finger_id;
	float  x;
	float  y;
	float  dx;
	float  dy;
	float  pressure;
	double timestamp_seconds;
};

struct EventController {
	int    id;
	int    button;
	int    axis_id;
	int    axis_value;
	bool   down;
	bool   up;
	bool   added;
	bool   removed;
	bool   remapped;
	bool   axis;
	bool   pressed;
	bool   released;
	double timestamp_seconds;
};

namespace {

std::unique_ptr<WindowContext> g_window;

} // namespace

constexpr const char* KYTY_SDL_WINDOW_CAPTION = "Game";

static void SetPause(WindowLoopState& game, bool flag) {
	LOGF("Pause: %s\n", flag ? "true" : "false");

	game.paused.store(flag, std::memory_order_release);
}

static void GameEventQuit(WindowLoopState& game) {
	LOGF("Event: quit\n");

	game.need_exit = true;
}

static void GameEventTerminate(WindowLoopState& game) {
	LOGF("Event: terminate\n");

	game.need_exit = true;
}

static void ToggleDesktopFullscreen() {
	if (g_window == nullptr || g_window->window == nullptr) {
		return;
	}

	const auto flags = static_cast<uint32_t>(SDL_GetWindowFlags(g_window->window));
	const bool fullscreen = (flags & SDL_WINDOW_FULLSCREEN) != 0u;
	if (!SDL_SetWindowFullscreen(g_window->window, !fullscreen)) {
		LOGF("Toggle fullscreen failed: %s\n", SDL_GetError());
	}
}

static void GameEventKeyboard(const EventKeyboard& key) {
	static SDL_Keycode fullscreen_key = SDLK_UNKNOWN;

#ifdef KYTY_DBG_INPUT
	LOGF("Key: time = %.04f, %s%s, %s%s, %s, scan = %d, key = %d, mod = %04" PRIx16 "\n",
	     key.timestamp_seconds, (key.down ? "down" : ""), (key.up ? "up" : ""),
	     (key.pressed ? "pressed" : ""), (key.released ? "released" : ""),
	     (key.repeat ? "repeat" : ""), key.scan_code, key.key_code, key.mod);
#endif

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS || KYTY_PLATFORM == KYTY_PLATFORM_LINUX
	if (key.down) {
		switch (key.key_code) {
			case SDLK_F1:
				if (!key.repeat) {
					RenderDocRequestCapture();
				}
				break;
			case SDLK_F11:
				if (!key.repeat) {
					ToggleDesktopFullscreen();
				}
				break;
			case SDLK_RETURN:
			case SDLK_KP_ENTER:
				if (!key.repeat && (key.mod & SDL_KMOD_ALT) != 0) {
					ToggleDesktopFullscreen();
					fullscreen_key = key.key_code;
				}
				break;
			default: break;
		}
	}

#endif

	const bool fullscreen_key_event =
	    fullscreen_key != SDLK_UNKNOWN && key.key_code == fullscreen_key;
	if (fullscreen_key_event && key.up) {
		fullscreen_key = SDLK_UNKNOWN;
	}
	if ((key.down || key.up) && !key.repeat && !fullscreen_key_event) {
		HostInputKey(key.key_code, key.down);
	}
}

static void GameEventMouse([[maybe_unused]] const EventMouse& mb) {
#ifdef KYTY_DBG_INPUT
	if (mb.wheel) {
		LOGF("Mouse wheel: time = %.04f, %s[%d, %d]\n", mb.timestamp_seconds,
		     (mb.touch ? "touch, " : ""), mb.x, mb.y);
	} else if (mb.motion) {
		LOGF("Mouse motion: time = %.04f, %s%s%s%s%s%s, [%d, %d], (%d, %d)\n", mb.timestamp_seconds,
		     (mb.left ? "left" : ""), (mb.middle ? "middle" : ""), (mb.right ? "right" : ""),
		     (mb.x1 ? "x1" : ""), (mb.x2 ? "x2" : ""), (mb.touch ? "_touch" : ""), mb.x, mb.y,
		     mb.motion_x, mb.motion_y);
	} else {
		LOGF("Mouse click: time = %.04f, %d, %s%s%s%s%s%s, %s%s, %s%s, [%d, %d]\n",
		     mb.timestamp_seconds, mb.num_of_clicks, (mb.left ? "left" : ""),
		     (mb.middle ? "middle" : ""), (mb.right ? "right" : ""), (mb.x1 ? "x1" : ""),
		     (mb.x2 ? "x2" : ""), (mb.touch ? "_touch" : ""), (mb.down ? "down" : ""),
		     (mb.up ? "up" : ""), (mb.pressed ? "pressed" : ""), (mb.released ? "released" : ""),
		     mb.x, mb.y);
	}
#endif

	if (mb.down || mb.up) {
		uint8_t mouse_button = 0;
		if (mb.left) {
			mouse_button = SDL_BUTTON_LEFT;
		} else if (mb.middle) {
			mouse_button = SDL_BUTTON_MIDDLE;
		} else if (mb.right) {
			mouse_button = SDL_BUTTON_RIGHT;
		} else if (mb.x1) {
			mouse_button = SDL_BUTTON_X1;
		} else if (mb.x2) {
			mouse_button = SDL_BUTTON_X2;
		}

		HostInputMouseButton(mouse_button, mb.down);
		if (!mb.touch && mb.left) {
			HostInputMouseTouchPad(mb.down, static_cast<float>(mb.x), static_cast<float>(mb.y));
		}
	} else if (mb.motion && !mb.touch) {
		if (mb.left) {
			HostInputMouseTouchPad(true, static_cast<float>(mb.x), static_cast<float>(mb.y));
		} else {
			HostInputMouseMotion(static_cast<float>(mb.x), static_cast<float>(mb.y));
		}
	}
}

static void GameEventFinger([[maybe_unused]] const EventFinger& f) {
#ifdef KYTY_DBG_INPUT
	if (f.motion) {
		LOGF("Finger motion: time = %.04f, %d, %d, (x,y) = [%f, %f], (dx,dy) = [%f, %f], pressure "
		     "= %f\n",
		     f.timestamp_seconds, f.touch_id, f.finger_id, f.x, f.y, f.dx, f.dy, f.pressure);
	} else {
		LOGF("Finger press: time = %.04f, %d, %d, %s%s, (x,y) = [%f, %f], (dx,dy) = [%f, %f], "
		     "pressure = %f\n",
		     f.timestamp_seconds, f.touch_id, f.finger_id, (f.down ? "down" : ""),
		     (f.up ? "up" : ""), f.x, f.y, f.dx, f.dy, f.pressure);
	}
#endif
}

static void GameEventController([[maybe_unused]] const EventController& f) {
	EXIT_NOT_IMPLEMENTED(f.remapped);

#ifdef KYTY_DBG_INPUT
	if (f.added || f.removed) {
		LOGF("Controller %s: %d, time = %.04f\n", (f.added ? "added" : "removed"), f.id,
		     f.timestamp_seconds);
	} else if (f.axis) {
		LOGF("Controller axis: %d, axis = %d, value = %d, time = %.04f\n", f.id, f.axis_id,
		     f.axis_value, f.timestamp_seconds);
	} else {
		LOGF("Controller button: "
		     "%d, %s%s, %s%s, button = %d, time = %.04f\n",
		     f.id, (f.down ? "down" : ""), (f.up ? "up" : ""), (f.pressed ? "pressed" : ""),
		     (f.released ? "released" : ""), f.button, f.timestamp_seconds);
	}
#endif

	if (f.added) {
		auto* pad = SDL_OpenGamepad(f.id);
		if (pad == nullptr) {
			LOGF("Controller: ignoring gamepad %d that could not be opened: %s\n", f.id,
			     SDL_GetError());
			return;
		}
		int id = SDL_GetJoystickID(SDL_GetGamepadJoystick(pad));
		Controller::Connect(id);
	}

	if (f.removed) {
		HostInputDisconnectGamepad(f.id);
		if (auto* pad = SDL_GetGamepadFromID(f.id); pad != nullptr) {
			Controller::Disconnect(f.id);
			SDL_CloseGamepad(pad);
		}
	}

	if (f.down) {
		Controller::PadPressed(f.id);
	}
	if (f.down || f.up) {
		HostInputPadButton(f.id, f.button, f.down);
	}

	if (f.axis) {
		if (f.axis_id == SDL_GAMEPAD_AXIS_LEFT_TRIGGER ||
		    f.axis_id == SDL_GAMEPAD_AXIS_RIGHT_TRIGGER) {
			HostInputPadTrigger(f.id, f.axis_id == SDL_GAMEPAD_AXIS_LEFT_TRIGGER,
			                    ControllerAxisValueFromSdl(f.axis_id, f.axis_value));
		} else if (const auto axis = ControllerAxisFromSdl(f.axis_id);
		           axis != Controller::Axis::AxisMax) {
			const int value = ControllerAxisValueFromSdl(f.axis_id, f.axis_value);
			if ((f.axis_id == SDL_GAMEPAD_AXIS_LEFTX || f.axis_id == SDL_GAMEPAD_AXIS_LEFTY) &&
			    HostInputGamepadAxis(f.id, f.axis_id == SDL_GAMEPAD_AXIS_LEFTX, value)) {
				return;
			}
			Controller::SetAxis(f.id, axis, value);
		}
	}
}

static void GameEventLowMemory() {
	LOGF("Event: low_memory\n");
}

static void GameEventWillEnterBackground(WindowLoopState& game) {
	LOGF("Event: will_enter_background\n");

	SetPause(game, true);
}

static void GameEventDidEnterBackground() {
	LOGF("Event: did_enter_background\n");
}

static void GameEventWillEnterForeground() {
	LOGF("Event: will_enter_foreground\n");
}

static void GameEventDidEnterForeground(WindowLoopState& game) {
	LOGF("Event: did_enter_foreground\n");

	SetPause(game, false);
}

/// Resizes the drawable surface to the given pixel dimensions.
/// Sets `minimized = false` on a positive size; sets `minimized = true` and returns early on a
/// nonpositive one.
void WindowContext::Resize(int new_width, int new_height) {
	if (new_width <= 0 || new_height <= 0) {
		LOGF("WindowContext::Resize(): ignoring nonpositive resize request (%dx%d); window is "
		     "likely minimized/hidden\n",
		     new_width, new_height);
		minimized.store(true, std::memory_order_release);
		return;
	}
	Common::LockGuard lock(mutex);
	graphic_ctx.screen_width  = static_cast<uint32_t>(new_width);
	graphic_ctx.screen_height = static_cast<uint32_t>(new_height);
	minimized.store(false, std::memory_order_release);
}

/// Queries the current drawable pixel size from SDL and calls Resize() to clear `minimized`.
/// Skips the call when the window is still minimised or SDL reports a non-positive size.
void WindowContext::RefreshSizeFromWindow() {
	if (SDL_GetWindowFlags(window) & SDL_WINDOW_MINIMIZED) {
		return;
	}
	int width  = 0;
	int height = 0;
	if (SDL_GetWindowSizeInPixels(window, &width, &height) && width > 0 && height > 0) {
		Resize(width, height);
	}
}

/// Dispatches a single SDL window event and keeps `minimized` up to date.
/// Sets `minimized = true` on HIDDEN/MINIMIZED or non-positive sizes;
/// clears it on RESTORED/MAXIMIZED, positive PIXEL_SIZE_CHANGED, SHOWN, or EXPOSED.
void WindowContext::ProcessWindowEvent(const SDL_WindowEvent& event) {
	const auto& window_event = event;
	switch (window_event.type) {
		case SDL_EVENT_WINDOW_SHOWN: {
			LOGF("Window %" PRIu32 " shown\n", window_event.windowID);
			RefreshSizeFromWindow();
			break;
		}

		case SDL_EVENT_WINDOW_HIDDEN: {
			LOGF("Window %" PRIu32 " hidden\n", window_event.windowID);
			minimized.store(true, std::memory_order_release);
			break;
		}

		case SDL_EVENT_WINDOW_EXPOSED: {
			LOGF("Window %" PRIu32 " exposed\n", window_event.windowID);
			RefreshSizeFromWindow();
			break;
		}

		case SDL_EVENT_WINDOW_MOVED:
			LOGF("Window %" PRIu32 " moved to %" PRId32 ",%" PRId32 "\n", window_event.windowID,
			     window_event.data1, window_event.data2);
			break;

		case SDL_EVENT_WINDOW_RESIZED:
			LOGF("Window %" PRIu32 " resized to %" PRId32 "x%" PRId32 "\n", window_event.windowID,
			     window_event.data1, window_event.data2);
			if (window_event.data1 <= 0 || window_event.data2 <= 0) {
				LOGF("Window %" PRIu32 " ignoring non-positive resize %" PRId32 "x%" PRId32 "\n",
				     window_event.windowID, window_event.data1, window_event.data2);
				minimized.store(true, std::memory_order_release);
			}
			break;

		case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED:
			LOGF("Window %" PRIu32 " size changed to %" PRId32 "x%" PRId32 "\n",
			     window_event.windowID, window_event.data1, window_event.data2);

			if (window_event.data1 > 0 && window_event.data2 > 0) {
				Resize(window_event.data1, window_event.data2);
			} else {
				LOGF("Window %" PRIu32 " ignoring non-positive size change %" PRId32 "x%" PRId32
				     "\n",
				     window_event.windowID, window_event.data1, window_event.data2);
				minimized.store(true, std::memory_order_release);
			}

			break;

		case SDL_EVENT_WINDOW_MINIMIZED:
			LOGF("Window %" PRIu32 " minimized\n", window_event.windowID);
			minimized.store(true, std::memory_order_release);
			break;
		case SDL_EVENT_WINDOW_MAXIMIZED:
			LOGF("Window %" PRIu32 " maximized\n", window_event.windowID);
			minimized.store(false, std::memory_order_release);
			// Refresh the cached drawable size in case no PIXEL_SIZE_CHANGED event follows.
			RefreshSizeFromWindow();
			break;
		case SDL_EVENT_WINDOW_RESTORED:
			LOGF("Window %" PRIu32 " restored\n", window_event.windowID);
			minimized.store(false, std::memory_order_release);
			// Refresh the cached drawable size in case no PIXEL_SIZE_CHANGED event follows.
			RefreshSizeFromWindow();
			break;
		case SDL_EVENT_WINDOW_MOUSE_ENTER:
			LOGF("Mouse entered window %" PRIu32 "\n", window_event.windowID);
			break;
		case SDL_EVENT_WINDOW_MOUSE_LEAVE:
			LOGF("Mouse left window %" PRIu32 "\n", window_event.windowID);
			HostInputSetTouchPadMode(false);
			break;
		case SDL_EVENT_WINDOW_FOCUS_GAINED:
			LOGF("Window %" PRIu32 " gained keyboard focus\n", window_event.windowID);
			break;
		case SDL_EVENT_WINDOW_FOCUS_LOST:
			LOGF("Window %" PRIu32 " lost keyboard focus\n", window_event.windowID);
			break;
		case SDL_EVENT_WINDOW_CLOSE_REQUESTED:
			LOGF("Window %" PRIu32 " closed\n", window_event.windowID);
			break;
		default:
			LOGF("Window %" PRIu32 " got unknown event %" PRIu32 "\n", window_event.windowID,
			     static_cast<uint32_t>(window_event.type));
			break;
	}
}

void WindowContext::ProcessDisplayEvent(const SDL_DisplayEvent& display) {
	switch (display.type) {
		case SDL_EVENT_DISPLAY_ORIENTATION: {
			LOGF("Display %" PRIu32 " changed orientation to %d - ", display.displayID,
			     static_cast<int>(display.data1));

			switch (display.data1) {
				case SDL_ORIENTATION_UNKNOWN: LOGF("UNKNOWN\n"); break;
				case SDL_ORIENTATION_LANDSCAPE: LOGF("LANDSCAPE\n"); break;
				case SDL_ORIENTATION_LANDSCAPE_FLIPPED: LOGF("LANDSCAPE_FLIPPED\n"); break;
				case SDL_ORIENTATION_PORTRAIT: LOGF("PORTRAIT\n"); break;
				case SDL_ORIENTATION_PORTRAIT_FLIPPED: LOGF("PORTRAIT_FLIPPED\n"); break;
				default: LOGF("???\n");
			}

			break;
		}
		default:
			LOGF("Display %" PRIu32 " got unknown event 0x%" PRIx32 "\n", display.displayID,
			     static_cast<uint32_t>(display.type));
			break;
	}
}

void WindowContext::ProcessEvent(double time_s) {
	auto& game  = loop;
	auto* event = &game.event;
	if (event->type == SDL_EVENT_KEY_DOWN || event->type == SDL_EVENT_KEY_UP) {
		const bool pressed = event->type == SDL_EVENT_KEY_DOWN && event->key.repeat == 0;
		switch (event->key.key) {
#if defined(__APPLE__)
			// winresize: F11 and Cmd+Ctrl+F (the macOS shortcut) toggle desktop fullscreen; Alt+Enter too.
			case SDLK_F11:
				if (pressed) {
					ToggleDesktopFullscreen();
				}
				return;
			case SDLK_F:
				if ((event->key.mod & SDL_KMOD_GUI) != 0 && (event->key.mod & SDL_KMOD_CTRL) != 0) {
					if (pressed) {
						ToggleDesktopFullscreen();
					}
					return;
				}
				break;
			case SDLK_RETURN:
			case SDLK_KP_ENTER:
				if ((event->key.mod & SDL_KMOD_ALT) != 0) {
					if (pressed) {
						ToggleDesktopFullscreen();
					}
					return;
				}
				break;
#endif
			case SDLK_F7:
				if (pressed) {
					HostInputToggleMouseToJoystick();
				}
				return;
			case SDLK_F8:
				if (pressed) {
					Controller::CycleKeyboardPlayer();
				}
				return;
#if defined(__APPLE__)
			case SDLK_P:
				if ((event->key.mod & SDL_KMOD_GUI) == 0) {
					break;
				}
				[[fallthrough]];
#endif
			case SDLK_F2:
				if (pressed) {
					if ((event->key.mod & SDL_KMOD_CTRL) != 0 && VideoOut::CycleFrameLimit()) {
						if (!PerformanceOverlayEnabled()) {
							TogglePerformanceOverlay();
						}
					} else if ((event->key.mod & SDL_KMOD_SHIFT) != 0) {
						TogglePerformanceOverlayDetails();
					} else {
						TogglePerformanceOverlay();
					}
				}
				return;
			default: break;
		}
	}
	if (ProcessSystemOverlayInput(*event)) {
		return;
	}
	if (event->type >= SDL_EVENT_WINDOW_FIRST && event->type <= SDL_EVENT_WINDOW_LAST) {
		ProcessWindowEvent(event->window);
		return;
	}
	if (event->type >= SDL_EVENT_DISPLAY_FIRST && event->type <= SDL_EVENT_DISPLAY_LAST) {
		ProcessDisplayEvent(event->display);
		return;
	}

	switch (event->type) {
		case SDL_EVENT_QUIT: GameEventQuit(game); break;

		case SDL_EVENT_TERMINATING: GameEventTerminate(game); break;

		case SDL_EVENT_LOW_MEMORY: GameEventLowMemory(); break;

		case SDL_EVENT_WILL_ENTER_BACKGROUND: GameEventWillEnterBackground(game); break;

		case SDL_EVENT_DID_ENTER_BACKGROUND: GameEventDidEnterBackground(); break;

		case SDL_EVENT_WILL_ENTER_FOREGROUND: GameEventWillEnterForeground(); break;

		case SDL_EVENT_DID_ENTER_FOREGROUND: GameEventDidEnterForeground(game); break;

		case SDL_EVENT_KEY_DOWN:
		case SDL_EVENT_KEY_UP: {
			EventKeyboard key {};

			key.down              = (event->type == SDL_EVENT_KEY_DOWN);
			key.up                = (event->type == SDL_EVENT_KEY_UP);
			key.pressed           = event->key.down;
			key.released          = !event->key.down;
			key.repeat            = (event->key.repeat != 0u);
			key.scan_code         = event->key.scancode;
			key.key_code          = event->key.key;
			key.mod               = event->key.mod;
			key.timestamp_seconds = time_s;

			GameEventKeyboard(key);

			break;
		}

		case SDL_EVENT_MOUSE_BUTTON_DOWN:
		case SDL_EVENT_MOUSE_BUTTON_UP: {
			EventMouse mb {};

			mb.down              = (event->button.type == SDL_EVENT_MOUSE_BUTTON_DOWN);
			mb.up                = (event->button.type == SDL_EVENT_MOUSE_BUTTON_UP);
			mb.left              = (event->button.button == SDL_BUTTON_LEFT);
			mb.middle            = (event->button.button == SDL_BUTTON_MIDDLE);
			mb.right             = (event->button.button == SDL_BUTTON_RIGHT);
			mb.x1                = (event->button.button == SDL_BUTTON_X1);
			mb.x2                = (event->button.button == SDL_BUTTON_X2);
			mb.touch             = (event->button.which == SDL_TOUCH_MOUSEID);
			mb.pressed           = event->button.down;
			mb.released          = !event->button.down;
			mb.num_of_clicks     = event->button.clicks;
			mb.wheel             = false;
			mb.x                 = event->button.x;
			mb.y                 = event->button.y;
			mb.motion            = false;
			mb.motion_x          = 0;
			mb.motion_y          = 0;
			mb.timestamp_seconds = time_s;

			GameEventMouse(mb);

			break;
		}

		case SDL_EVENT_MOUSE_WHEEL: {
			EventMouse mb {};

			mb.down              = false;
			mb.up                = false;
			mb.left              = false;
			mb.middle            = false;
			mb.right             = false;
			mb.x1                = false;
			mb.x2                = false;
			mb.touch             = (event->wheel.which == SDL_TOUCH_MOUSEID);
			mb.pressed           = false;
			mb.released          = false;
			mb.num_of_clicks     = 0;
			mb.wheel             = true;
			mb.x                 = event->wheel.x;
			mb.y                 = event->wheel.y;
			mb.motion            = false;
			mb.motion_x          = 0;
			mb.motion_y          = 0;
			mb.timestamp_seconds = time_s;

			GameEventMouse(mb);

			break;
		}

		case SDL_EVENT_MOUSE_MOTION: {
			EventMouse mb {};

			mb.down              = false;
			mb.up                = false;
			mb.left              = ((event->motion.state & SDL_BUTTON_LMASK) != 0u);
			mb.middle            = ((event->motion.state & SDL_BUTTON_MMASK) != 0u);
			mb.right             = ((event->motion.state & SDL_BUTTON_RMASK) != 0u);
			mb.x1                = ((event->motion.state & SDL_BUTTON_X1MASK) != 0u);
			mb.x2                = ((event->motion.state & SDL_BUTTON_X2MASK) != 0u);
			mb.touch             = (event->motion.which == SDL_TOUCH_MOUSEID);
			mb.pressed           = false;
			mb.released          = false;
			mb.num_of_clicks     = 0;
			mb.wheel             = false;
			mb.x                 = event->motion.x;
			mb.y                 = event->motion.y;
			mb.motion            = true;
			mb.motion_x          = event->motion.xrel;
			mb.motion_y          = event->motion.yrel;
			mb.timestamp_seconds = time_s;

			GameEventMouse(mb);

			break;
		}

		case SDL_EVENT_FINGER_MOTION:
		case SDL_EVENT_FINGER_DOWN:
		case SDL_EVENT_FINGER_UP: {
			EventFinger f {};

			f.down              = (event->tfinger.type == SDL_EVENT_FINGER_DOWN);
			f.up                = (event->tfinger.type == SDL_EVENT_FINGER_UP);
			f.motion            = (event->tfinger.type == SDL_EVENT_FINGER_MOTION);
			f.finger_id         = static_cast<int>(event->tfinger.fingerID);
			f.touch_id          = static_cast<int>(event->tfinger.touchID);
			f.x                 = event->tfinger.x;
			f.y                 = event->tfinger.y;
			f.dx                = event->tfinger.dx;
			f.dy                = event->tfinger.dy;
			f.pressure          = event->tfinger.pressure;
			f.timestamp_seconds = time_s;

			GameEventFinger(f);

			break;
		}

		case SDL_EVENT_GAMEPAD_AXIS_MOTION: {
			EventController c {};

			c.id                = event->gaxis.which;
			c.button            = SDL_GAMEPAD_BUTTON_INVALID;
			c.axis_id           = event->gaxis.axis;
			c.axis_value        = event->gaxis.value;
			c.down              = false;
			c.up                = false;
			c.added             = false;
			c.removed           = false;
			c.remapped          = false;
			c.axis              = true;
			c.pressed           = false;
			c.released          = false;
			c.timestamp_seconds = time_s;

			GameEventController(c);

			break;
		}

		case SDL_EVENT_GAMEPAD_BUTTON_DOWN:
		case SDL_EVENT_GAMEPAD_BUTTON_UP: {
			EventController c {};

			c.id                = event->gbutton.which;
			c.button            = event->gbutton.button;
			c.axis_id           = SDL_GAMEPAD_AXIS_INVALID;
			c.axis_value        = 0;
			c.down              = event->gbutton.down;
			c.up                = !event->gbutton.down;
			c.added             = false;
			c.removed           = false;
			c.remapped          = false;
			c.axis              = false;
			c.pressed           = event->gbutton.down;
			c.released          = !event->gbutton.down;
			c.timestamp_seconds = time_s;

			GameEventController(c);

			break;
		}

		case SDL_EVENT_GAMEPAD_TOUCHPAD_DOWN:
		case SDL_EVENT_GAMEPAD_TOUCHPAD_MOTION:
		case SDL_EVENT_GAMEPAD_TOUCHPAD_UP:
			if (event->gtouchpad.touchpad == 0) {
				Controller::SetTouchPad(event->gtouchpad.which, event->gtouchpad.finger,
				                        event->gtouchpad.type != SDL_EVENT_GAMEPAD_TOUCHPAD_UP,
				                        event->gtouchpad.x, event->gtouchpad.y);
			}
			break;

		case SDL_EVENT_GAMEPAD_SENSOR_UPDATE: {
			const auto& sensor = event->gsensor;
			if (sensor.sensor == SDL_SENSOR_ACCEL || sensor.sensor == SDL_SENSOR_GYRO) {
				Controller::SetSensor(sensor.which,
				                      sensor.sensor == SDL_SENSOR_ACCEL ? Controller::Sensor::Accel
				                                                        : Controller::Sensor::Gyro,
				                      sensor.data,
				                      sensor.sensor_timestamp != 0
				                          ? sensor.sensor_timestamp / 1000
				                          : sensor.timestamp / 1000);
			}
			break;
		}

		case SDL_EVENT_GAMEPAD_ADDED:
		case SDL_EVENT_GAMEPAD_REMOVED:
		case SDL_EVENT_GAMEPAD_REMAPPED: {
			EventController c {};

			c.id                = event->gdevice.which;
			c.button            = SDL_GAMEPAD_BUTTON_INVALID;
			c.axis_id           = SDL_GAMEPAD_AXIS_INVALID;
			c.axis_value        = 0;
			c.down              = false;
			c.up                = false;
			c.added             = (event->gdevice.type == SDL_EVENT_GAMEPAD_ADDED);
			c.removed           = (event->gdevice.type == SDL_EVENT_GAMEPAD_REMOVED);
			c.remapped          = (event->gdevice.type == SDL_EVENT_GAMEPAD_REMAPPED);
			c.axis              = false;
			c.pressed           = false;
			c.released          = false;
			c.timestamp_seconds = time_s;

			GameEventController(c);

			break;
		}
	}
}

namespace {

// The window title with compile progress (window thread only). `base` is the diagnostic title
// UpdateTitle built (empty before the first frame); the status is appended on every poll.
struct ProgressTitle {
	std::string                       base;
	bool                              first_frame = false;
	CompileProgress::TitleThrottle    throttle;
};

ProgressTitle& GetProgressTitle() {
	static ProgressTitle state;
	return state;
}

std::string ProgressTitleName() {
	static std::string name;
	char               title[128];
	if (name.empty() && Loader::SystemContentParamSfoGetString("TITLE", title, sizeof(title))) {
		name = title;
	}
	return name.empty() ? std::string("Kyty") : name;
}

// nps5 autocap: the display the window is on, from its NSScreen (objc runtime: no AppKit link
// needed, the window exists so AppKit is loaded). Adaptive = NSScreen minimumRefreshInterval <
// maximumRefreshInterval (ProMotion / Adaptive-Sync). Runs on the window (main) thread. Returns
// false when there is no screen yet.
bool QueryWindowScreen(SDL_Window* window, AutoCap::Screen& out) {
	const auto props = SDL_GetWindowProperties(window);
	void* nswin = props != 0
	                  ? SDL_GetPointerProperty(props, SDL_PROP_WINDOW_COCOA_WINDOW_POINTER, nullptr)
	                  : nullptr;
	if (nswin == nullptr) {
		return false;
	}
	using IdFn          = void* (*)(void*, SEL);
	using DblFn         = double (*)(void*, SEL);
	using IntFn         = long (*)(void*, SEL);
	using StrFn         = const char* (*)(void*, SEL);
	const auto send_id  = reinterpret_cast<IdFn>(objc_msgSend);
	const auto send_dbl = reinterpret_cast<DblFn>(objc_msgSend);
	const auto send_int = reinterpret_cast<IntFn>(objc_msgSend);
	const auto send_str = reinterpret_cast<StrFn>(objc_msgSend);
	void*      screen   = send_id(nswin, sel_registerName("screen"));
	if (screen == nullptr) {
		return false;
	}
	const auto  fps    = send_int(screen, sel_registerName("maximumFramesPerSecond"));
	const auto  min_iv = send_dbl(screen, sel_registerName("minimumRefreshInterval"));
	const auto  max_iv = send_dbl(screen, sel_registerName("maximumRefreshInterval"));
	void*       name   = send_id(screen, sel_registerName("localizedName"));
	const char* utf8   = name != nullptr ? send_str(name, sel_registerName("UTF8String")) : nullptr;
	out.name           = utf8 != nullptr ? utf8 : "screen";
	out.refresh_hz     = static_cast<double>(fps);
	out.adaptive       = min_iv > 0.0 && max_iv > min_iv + 1e-6;
	return true;
}

void RefreshProgressTitle(SDL_Window* window, uint64_t now_ms) {
	auto&      state = GetProgressTitle();
	const auto text  = CompileProgress::ComposeTitle(state.first_frame ? state.base : ProgressTitleName(),
	                                                  CompileProgress::Read(), state.first_frame);
	if (!state.throttle.ShouldSet(now_ms, text)) {
		return;
	}
	SDL_SetWindowTitle(window, text.c_str());
	if (CompileProgress::LogEnabled()) {
		std::fprintf(stderr, "[progress-title] %llu ms: %s\n", static_cast<unsigned long long>(now_ms),
		             text.c_str());
	}
}

} // namespace

namespace {

// winresize: while the user drags a window edge or Cocoa animates a zoom, the main thread sits in
// AppKit's tracking loop inside SDL_PumpEvents/SDL_WaitEvent and Run() does not get to see the
// events until it ends. SDL calls event watches synchronously from where the event is pushed, so
// the drawable size is updated live and the presenter's swapchain follows during the drag.
bool SDLCALL LiveResizeWatch(void* user, SDL_Event* event) {
	auto* context = static_cast<WindowContext*>(user);
	if (event->type == SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED && event->window.data1 > 0 &&
	    event->window.data2 > 0 && context->window != nullptr &&
	    event->window.windowID == SDL_GetWindowID(context->window)) {
		context->Resize(event->window.data1, event->window.data2);
	}
	return true;
}

// KYTY_TEST_RESIZE='t1:WxH,t2:max,t3:fs,t4:restore' (t = seconds after the first presented frame):
// resizes the window programmatically on the window thread; logs each step and its outcome.
struct ResizeScript {
	std::vector<PresentGeometry::Step> steps;
	size_t                             next      = 0;
	uint64_t                           start_ms  = 0;
	bool                               started   = false;
	uint64_t                           last_log  = 0;
	std::string                        last_what;
};

void LogWindowState(SDL_Window* window, const char* what) {
	int w = 0, h = 0, pw = 0, ph = 0;
	SDL_GetWindowSize(window, &w, &h);
	SDL_GetWindowSizeInPixels(window, &pw, &ph);
	const auto flags = static_cast<uint64_t>(SDL_GetWindowFlags(window));
	std::fprintf(stderr, "[winresize] %s: window %dx%d points, %dx%d pixels%s%s%s\n", what, w, h, pw, ph,
	             (flags & SDL_WINDOW_FULLSCREEN) != 0u ? " fullscreen" : "",
	             (flags & SDL_WINDOW_MAXIMIZED) != 0u ? " maximized" : "",
	             (flags & SDL_WINDOW_MINIMIZED) != 0u ? " minimized" : "");
}

void RunResizeStep(SDL_Window* window, const PresentGeometry::Step& step) {
	using PresentGeometry::StepKind;
	switch (step.kind) {
		case StepKind::Size:
			SDL_SetWindowFullscreen(window, false);
			SDL_RestoreWindow(window);
			SDL_SetWindowSize(window, step.w, step.h);
			std::fprintf(stderr, "[winresize] step t=%.1f: size %dx%d\n", step.at_s, step.w, step.h);
			break;
		case StepKind::Maximize:
			SDL_MaximizeWindow(window);
			std::fprintf(stderr, "[winresize] step t=%.1f: maximize\n", step.at_s);
			break;
		case StepKind::Fullscreen:
			SDL_SetWindowFullscreen(window, true);
			std::fprintf(stderr, "[winresize] step t=%.1f: fullscreen\n", step.at_s);
			break;
		case StepKind::Restore:
			SDL_SetWindowFullscreen(window, false);
			SDL_RestoreWindow(window);
			std::fprintf(stderr, "[winresize] step t=%.1f: restore\n", step.at_s);
			break;
	}
}

} // namespace

namespace {
// winresize: the window loop as resumable steps. On macOS the loop runs inside a block of the
// process main queue (a serial queue): one never-returning block starves every other main-queue
// block (AppKit's fullscreen/zoom completion, Spaces transitions, CAMetalLayer layout), which is
// the maximise hang. The bridge therefore runs one step per main-queue block and re-queues itself.
struct RunState {
	Common::Timer          timer;
	ResizeScript           script;
	uint64_t               title_time   = 0;
	uint64_t               title_frames = 0;
	bool                   progress_title = false;
	uint64_t               close_after_ms = 0; // KYTY_TEST_CLOSE_S: test-only window close
	uint64_t               first_frame_ms = 0;
	bool                   close_sent     = false;
};
std::unique_ptr<RunState> g_run;
constexpr uint64_t        kTitleIntervalMs = 1000;
} // namespace

void WindowContext::RunBegin() {
	g_run = std::make_unique<RunState>();
	g_run->timer.Start();
	SDL_AddEventWatch(LiveResizeWatch, this);
	if (const char* text = std::getenv("KYTY_TEST_RESIZE"); text != nullptr && *text != '\0') {
		g_run->script.steps = PresentGeometry::ParseResizeScript(text);
		std::fprintf(stderr, "[winresize] KYTY_TEST_RESIZE: %zu steps\n", g_run->script.steps.size());
	}
	// KYTY_TEST_CLOSE_S=<s> (test only, unset = off): <s> seconds after the first presented frame the window gets the
	// events a click on its close button produces (CLOSE_REQUESTED, then QUIT for the last window), so a script can
	// check the normal end of a session (renderer: window loop ended -> stopped, exit code) without GUI scripting.
	if (const char* text = std::getenv("KYTY_TEST_CLOSE_S"); text != nullptr && *text != '\0') {
		const double s = std::strtod(text, nullptr);
		if (s > 0.0 && s < 86400.0) {
			g_run->close_after_ms = static_cast<uint64_t>(s * 1000.0);
			std::fprintf(stderr, "[winclose] KYTY_TEST_CLOSE_S: close %.1f s after the first frame\n", s);
		}
	}
	loop.event     = {};
	loop.need_exit = false;
	loop.paused.store(false, std::memory_order_release);
	g_run->title_time   = SDL_GetTicks();
	g_run->title_frames = loop.presented_frames.load(std::memory_order_relaxed);
	// Compile progress in the title (KYTY_PROGRESS_TITLE=0 off): polled here on the window thread
	// every CompileProgress::kTitleIntervalMs, so the loop wakes that often.
	g_run->progress_title = CompileProgress::Enabled();
	if (g_run->progress_title) {
		RefreshProgressTitle(window, SDL_GetTicks());
	}
}

bool WindowContext::RunStep(int max_wait_ms) {
	auto& r      = *g_run;
	auto& script = r.script;
	if (loop.need_exit) {
		return false;
	}
	if (loop.paused.load(std::memory_order_acquire)) {
		if (!r.timer.IsPaused()) {
			r.timer.Pause();
		}
	} else if (r.timer.IsPaused()) {
		r.timer.Resume();
	}

	// Refresh the title on the main thread without making presentation wait for it.
	const auto now     = SDL_GetTicks();
	const auto elapsed = now - r.title_time;
	if (elapsed >= kTitleIntervalMs) {
		const auto frames = loop.presented_frames.load(std::memory_order_relaxed);
		if (frames != 0) {
			UpdateTitle(frames, static_cast<double>(frames - r.title_frames) * 1000.0 /
			                        static_cast<double>(elapsed));
		}
		r.title_time   = now;
		r.title_frames = frames;
	}
	if (r.progress_title) {
		RefreshProgressTitle(window, now);
	}
	{
		// KYTY_FPS_LIMIT_MILLI=auto: re-read the window's screen (moved to another display, display
		// mode changed).
		static const bool autocap = [] {
			const char* v = std::getenv("KYTY_FPS_LIMIT_MILLI");
			return v != nullptr && std::strcmp(v, "auto") == 0;
		}();
		static uint64_t last_screen_ms = 0;
		if (autocap && (last_screen_ms == 0 || now - last_screen_ms >= 500)) {
			last_screen_ms = now;
			AutoCap::Screen screen;
			if (QueryWindowScreen(window, screen)) {
				AutoCap::Publish(screen);
			}
		}
	}
	if (r.close_after_ms != 0 && !r.close_sent) {
		if (r.first_frame_ms == 0 && loop.presented_frames.load(std::memory_order_relaxed) > 0) {
			r.first_frame_ms = now;
		}
		if (r.first_frame_ms != 0 && now - r.first_frame_ms >= r.close_after_ms) {
			r.close_sent = true;
			SDL_Event ev {};
			ev.type            = SDL_EVENT_WINDOW_CLOSE_REQUESTED;
			ev.window.windowID = SDL_GetWindowID(window);
			SDL_PushEvent(&ev);
			SDL_Event quit {};
			quit.type = SDL_EVENT_QUIT;
			SDL_PushEvent(&quit);
			std::fprintf(stderr, "[winclose] close requested\n");
		}
	}
	if (script.next < script.steps.size() || script.last_log != 0) {
		const auto frames_now = loop.presented_frames.load(std::memory_order_relaxed);
		if (!script.started && frames_now > 0) {
			script.started  = true;
			script.start_ms = now;
			LogWindowState(window, "initial");
		}
		while (script.started && script.next < script.steps.size() &&
		       static_cast<double>(now - script.start_ms) >= script.steps[script.next].at_s * 1000.0) {
			RunResizeStep(window, script.steps[script.next]);
			script.next++;
			script.last_log = now;
		}
		// report the state 1.5 s after each step
		if (script.last_log != 0 && now - script.last_log >= 1500) {
			LogWindowState(window, "after step");
			script.last_log = script.next >= script.steps.size() ? 0 : now;
		}
	}
	auto wait_ms = static_cast<int>(kTitleIntervalMs - (now - r.title_time));
	if (script.next < script.steps.size() || script.last_log != 0) {
		wait_ms = std::min(wait_ms, 50);
	}
	if (r.progress_title && wait_ms > static_cast<int>(CompileProgress::kTitleIntervalMs)) {
		wait_ms = static_cast<int>(CompileProgress::kTitleIntervalMs);
	}
	wait_ms = std::max(0, std::min(wait_ms, max_wait_ms));
	if (HostInputWaitEvent(&loop.event, wait_ms)) {
		ProcessEvent(r.timer.GetTimeS());
	}
	return !loop.need_exit;
}

void WindowContext::RunEnd() {
	SDL_RemoveEventWatch(LiveResizeWatch, this);
	g_run.reset();
}

void WindowContext::Run() {
	RunBegin();
	while (RunStep(1000)) {
	}
	RunEnd();
}

// nps5: window placement like AnyPS5's DisplayWindow (same env names). NPS5_WINDOW_SCREEN=<builtin|main|index>[:tl|:tr|:bl|:br|:center]
// (default builtin:tl = laptop panel, top-left), NPS5_WINDOW_PERCENT=1..100 of the usable display (default 33), NPS5_WINDOW_STACK=0 turns
// off the slot stacking (flock() slot files in NPS5_WINDOW_SLOT_DIR, shared with AnyPS5 windows), NPS5_WINDOW_TOP=1 keeps it on top.
namespace {
int g_window_slot    = -1;
int g_window_slot_fd = -1;

void ReleaseWindowSlot() noexcept {
	if (g_window_slot_fd >= 0) {
		if (ftruncate(g_window_slot_fd, 0) != 0) {}
		close(g_window_slot_fd);
		g_window_slot_fd = -1;
	}
	g_window_slot = -1;
}

int TakeWindowSlot() {
	const char* off = std::getenv("NPS5_WINDOW_STACK");
	if (off != nullptr && std::string(off) == "0") return -1;
	std::string dir;
	if (const char* custom = std::getenv("NPS5_WINDOW_SLOT_DIR"); custom != nullptr && *custom != '\0') {
		dir = custom;
	} else {
		char tmp_buffer[1024] = {};
		const char* tmp       = std::getenv("TMPDIR");
#if defined(__APPLE__)
		if (confstr(_CS_DARWIN_USER_TEMP_DIR, tmp_buffer, sizeof(tmp_buffer)) > 0) tmp = tmp_buffer;
#endif
		dir = std::string(tmp != nullptr && *tmp != '\0' ? tmp : "/tmp");
		while (dir.size() > 1 && dir.back() == '/') dir.pop_back();
		dir += "/nps5-windows";
	}
	mkdir(dir.c_str(), 0700);
	struct stat info {};
	if (stat(dir.c_str(), &info) != 0 || !S_ISDIR(info.st_mode)) return -1;
	for (int slot = 0; slot < 64; ++slot) {
		const std::string path = dir + "/slot-" + std::to_string(slot);
		const int fd           = open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0600);
		if (fd < 0) return -1;
		if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
			close(fd);
			continue;
		}
		const auto text = std::to_string(getpid());
		if (ftruncate(fd, 0) != 0) {}
		if (pwrite(fd, text.data(), text.size(), 0) < 0) {}
		static bool registered = false;
		if (!registered) {
			registered = true;
			std::atexit([] { ReleaseWindowSlot(); });
		}
		g_window_slot_fd = fd;
		return slot;
	}
	return -1;
}

// SDL display id of the built-in panel (0 if none). CoreGraphics and SDL share the global top-left point coordinates, so the display
// whose bounds start at the panel's origin is the panel. CoreGraphics is resolved with dlsym (the dylib does not link it).
SDL_DisplayID BuiltinDisplayId() {
#if defined(__APPLE__)
	struct Rect { double x, y, w, h; };
	using GetList   = int32_t (*)(uint32_t, uint32_t*, uint32_t*);
	using IsBuiltin = uint32_t (*)(uint32_t);
	using Bounds    = Rect (*)(uint32_t);
	auto get_list   = reinterpret_cast<GetList>(dlsym(RTLD_DEFAULT, "CGGetActiveDisplayList"));
	auto is_builtin = reinterpret_cast<IsBuiltin>(dlsym(RTLD_DEFAULT, "CGDisplayIsBuiltin"));
	auto bounds     = reinterpret_cast<Bounds>(dlsym(RTLD_DEFAULT, "CGDisplayBounds"));
	if (get_list == nullptr || is_builtin == nullptr || bounds == nullptr) return 0;
	uint32_t ids[16];
	uint32_t count = 0;
	if (get_list(16, ids, &count) != 0) return 0;
	int n                 = 0;
	SDL_DisplayID* displays = SDL_GetDisplays(&n);
	SDL_DisplayID found   = 0;
	for (uint32_t i = 0; i < count && found == 0; ++i) {
		if (is_builtin(ids[i]) == 0) continue;
		const Rect b = bounds(ids[i]);
		for (int d = 0; d < n && displays != nullptr; ++d) {
			SDL_Rect r {};
			if (SDL_GetDisplayBounds(displays[d], &r) && r.x == static_cast<int>(b.x) && r.y == static_cast<int>(b.y)) {
				found = displays[d];
				break;
			}
		}
	}
	SDL_free(displays);
	return found;
#else
	return 0;
#endif
}

// Applies the placement to a freshly created window of content size w x h. Logs '[window] display N (builtin) at x,y size WxH'.
void PlaceWindow(SDL_Window* window, int w, int h) {
	std::string spec = "builtin:tl";
	if (const char* v = std::getenv("NPS5_WINDOW_SCREEN"); v != nullptr && *v != '\0') spec = v;
	std::string screen = spec;
	char corner        = 't'; // t = top-left, r = top-right, l = bottom-left, b = bottom-right, c = center
	if (const auto colon = spec.find(':'); colon != std::string::npos) {
		screen          = spec.substr(0, colon);
		const auto where = spec.substr(colon + 1);
		corner           = where == "tl" ? 't' : where == "tr" ? 'r' : where == "bl" ? 'l' : (where == "br" || where == "corner") ? 'b' : 'c';
	} else if (std::getenv("NPS5_WINDOW_SCREEN") != nullptr) {
		corner = 'c';
	}
	int n                   = 0;
	SDL_DisplayID* displays = SDL_GetDisplays(&n);
	SDL_DisplayID id        = 0;
	const char* kind        = "main";
	if (screen == "builtin") {
		id   = BuiltinDisplayId();
		kind = id != 0 ? "builtin" : "main (no builtin)";
	} else if (screen != "main") {
		char* end        = nullptr;
		const auto index = std::strtol(screen.c_str(), &end, 10);
		if (end != screen.c_str() && *end == '\0' && index >= 0 && index < n && displays != nullptr) {
			id   = displays[index];
			kind = "index";
		} else {
			LOGF("[window] ignoring NPS5_WINDOW_SCREEN=%s\n", spec.c_str());
		}
	}
	if (id == 0) id = SDL_GetPrimaryDisplay();
	int display_no = 0;
	for (int d = 0; d < n && displays != nullptr; ++d) {
		if (displays[d] == id) display_no = d;
	}
	SDL_free(displays);
	SDL_Rect usable {};
	if (!SDL_GetDisplayUsableBounds(id, &usable) || usable.w <= 0 || usable.h <= 0) {
		return;
	}
	long percent = 33; // about a third of the panel
	if (const char* v = std::getenv("NPS5_WINDOW_PERCENT"); v != nullptr && *v != '\0') {
		char* end        = nullptr;
		const auto parsed = std::strtol(v, &end, 10);
		if (end != v && *end == '\0' && parsed >= 1 && parsed <= 100) percent = parsed;
	}
	const int bound_w = std::max(320, static_cast<int>(static_cast<int64_t>(usable.w) * percent / 100));
	const int bound_h = std::max(180, static_cast<int>(static_cast<int64_t>(usable.h) * percent / 100));
	// contain the source aspect inside the bounds
	int ww = bound_w;
	int wh = static_cast<int>(static_cast<int64_t>(bound_w) * h / w);
	if (wh > bound_h) {
		wh = bound_h;
		ww = static_cast<int>(static_cast<int64_t>(bound_h) * w / h);
	}
	constexpr int title_bar = 28;
	const bool left         = corner == 't' || corner == 'l';
	const bool top          = corner == 't' || corner == 'r';
	int x = usable.x + (usable.w - ww) / 2;
	int y = usable.y + (usable.h - wh) / 2;
	if (corner != 'c') {
		x = left ? usable.x : usable.x + usable.w - ww;
		y = top ? usable.y + title_bar : usable.y + usable.h - wh;
	}
	if (const int slot = TakeWindowSlot(); slot >= 0) {
		g_window_slot   = slot;
		const int pitch = wh + title_bar;
		const int rows  = std::max(1, usable.h / pitch);
		const int cols  = std::max(1, usable.w / ww);
		x               = usable.x + ((slot / rows) % cols) * ww;
		y               = usable.y + title_bar + (slot % rows) * pitch;
	}
	SDL_SetWindowSize(window, ww, wh);
	SDL_SetWindowPosition(window, x, y);
	if (const char* t = std::getenv("NPS5_WINDOW_TOP"); t != nullptr && std::atoi(t) != 0) SDL_SetWindowAlwaysOnTop(window, true);
	std::fprintf(stderr, "[window] display %d (%s) at %d,%d size %dx%d slot %d\n", display_no, kind, x, y, ww, wh, g_window_slot);
}

// Fullscreen: the same NPS5_WINDOW_SCREEN choice (builtin / main / index; unset = main, the primary
// display as before), centred on that display so SDL's fullscreen lands there. Logs '[window]
// fullscreen on display N (kind)'.
void PlaceFullscreenWindow(SDL_Window* window) {
	std::string screen = "main";
	if (const char* v = std::getenv("NPS5_WINDOW_SCREEN"); v != nullptr && *v != '\0') screen = v;
	if (const auto colon = screen.find(':'); colon != std::string::npos)
		screen = screen.substr(0, colon);
	int            n        = 0;
	SDL_DisplayID* displays = SDL_GetDisplays(&n);
	SDL_DisplayID  id       = 0;
	const char*    kind     = "main";
	if (screen == "builtin") {
		id   = BuiltinDisplayId();
		kind = id != 0 ? "builtin" : "main (no builtin)";
	} else if (screen != "main") {
		char*      end   = nullptr;
		const auto index = std::strtol(screen.c_str(), &end, 10);
		if (end != screen.c_str() && *end == '\0' && index >= 0 && index < n &&
		    displays != nullptr) {
			id   = displays[index];
			kind = "index";
		}
	}
	if (id == 0) id = SDL_GetPrimaryDisplay();
	int display_no = 0;
	for (int d = 0; d < n && displays != nullptr; ++d) {
		if (displays[d] == id) display_no = d;
	}
	SDL_free(displays);
	SDL_SetWindowPosition(window, static_cast<int>(SDL_WINDOWPOS_CENTERED_DISPLAY(id)),
	                      static_cast<int>(SDL_WINDOWPOS_CENTERED_DISPLAY(id)));
	std::fprintf(stderr, "[window] fullscreen on display %d (%s)\n", display_no, kind);
}
} // namespace

static void WindowCreate(WindowContext& context) {
	EXIT_IF(context.window != nullptr);
	EXIT_IF(context.graphic_ctx.screen_width == 0);
	EXIT_IF(context.graphic_ctx.screen_height == 0);

	int width  = static_cast<int>(context.graphic_ctx.screen_width);
	int height = static_cast<int>(context.graphic_ctx.screen_height);

#if KYTY_PLATFORM == KYTY_PLATFORM_LINUX
	// RenderDoc can hide Wayland Vulkan surface support; prefer X11 for captures.
	if (Config::RenderDocEnabled()) {
		SDL_SetHint(SDL_HINT_VIDEO_DRIVER, "x11,wayland");
	}
#endif
	if (!SDL_InitSubSystem(SDL_INIT_VIDEO | SDL_INIT_GAMEPAD)) {
		EXIT("%s\n", SDL_GetError());
	}

	LOGF("WindowCreate(): width = %d, height = %d\n", width, height);

	uint32_t window_flags = WindowContext::InitialWindowFlags(Config::FullscreenEnabled());
	// With NPS5_WINDOW_SCREEN set: create windowed, move to the chosen display, then go fullscreen
	// there (a window created fullscreen opens on the primary display). Unset: the window is
	// created fullscreen as before (opt-in).
	const char* screen_env = std::getenv("NPS5_WINDOW_SCREEN");
	const bool  fs_on_chosen =
	    Config::FullscreenEnabled() && screen_env != nullptr && *screen_env != '\0';
	if (fs_on_chosen) window_flags &= ~static_cast<uint32_t>(SDL_WINDOW_FULLSCREEN);
#if defined(__APPLE__)
	// SDL loads Vulkan while creating a Vulkan window, so select the bundled
	// MoltenVK loader before calling SDL_CreateWindow. Keep an explicit user
	// override, and fall back to SDL's normal loader search when no bundle is present.
	if (std::getenv("SDL_VULKAN_LIBRARY") == nullptr) {
		if (const char* base_path = SDL_GetBasePath(); base_path != nullptr) {
			const std::string base_path_str = base_path;
			std::string moltenvk_path = base_path_str + "libMoltenVK.dylib";
			if (!Common::File::IsFileExisting(moltenvk_path)) {
				moltenvk_path = base_path_str + "../Frameworks/libMoltenVK.dylib";
			}
			if (Common::File::IsFileExisting(moltenvk_path) &&
			    SDL_setenv_unsafe("SDL_VULKAN_LIBRARY", moltenvk_path.c_str(), 0) == 0) {
				LOGF("Vulkan loader: %s\n", moltenvk_path.c_str());
			}
		}
	}

	// macOS 26 window chrome (CoreUI asset decode, SwiftUI titlebar) has been observed
	// throwing NSExceptions under Rosetta during the first CATransaction commit. A
	// borderless window skips that machinery entirely.
	if (std::getenv("KYTY_BORDERLESS") != nullptr) {
		window_flags |= static_cast<uint32_t>(SDL_WINDOW_BORDERLESS);
	}
#endif
	context.window = SDL_CreateWindow(KYTY_SDL_WINDOW_CAPTION, width, height, window_flags);

	if (context.window == nullptr) {
		EXIT("%s\n", SDL_GetError());
	}

	if (Config::FullscreenEnabled() && !fs_on_chosen) {
		SDL_SetWindowPosition(context.window, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED);
	} else if (Config::FullscreenEnabled()) {
		PlaceFullscreenWindow(context.window);
		SDL_SyncWindow(context.window);
		if (!SDL_SetWindowFullscreen(context.window, true)) {
			std::fprintf(stderr, "[window] fullscreen failed: %s\n", SDL_GetError());
		}
		SDL_SyncWindow(context.window);
		const SDL_DisplayID now = SDL_GetDisplayForWindow(context.window);
		std::fprintf(stderr, "[window] fullscreen display id %u (builtin id %u)\n",
		             static_cast<unsigned>(now), static_cast<unsigned>(BuiltinDisplayId()));
	} else {
		PlaceWindow(context.window, width, height);
	}
	HostInputInit(context.window);
	InitializeSystemOverlayInput(context.window);
	context.UpdateIcon();
}

uint32_t WindowContext::InitialWindowFlags(bool fullscreen) noexcept {
	auto flags = static_cast<uint32_t>(SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);
	if (fullscreen) {
		flags |= static_cast<uint32_t>(SDL_WINDOW_FULLSCREEN);
	}
#if defined(__APPLE__)
	// crisptext: size the drawable in backing pixels (Retina 2x). Without this the swapchain is the window size in
	// points and the compositor magnifies it 2x (chunky text). KYTY_HIDPI=0 restores the point-sized drawable.
	if (const char* v = std::getenv("KYTY_HIDPI"); v == nullptr || std::string_view(v) != "0") {
		flags |= static_cast<uint32_t>(SDL_WINDOW_HIGH_PIXEL_DENSITY);
	}
#endif
	return flags;
}

Presenter& WindowInit(uint32_t width, uint32_t height) {
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
	EXIT_IF(g_window != nullptr);

	auto window = std::make_unique<WindowContext>();

	window->graphic_ctx.screen_width  = width;
	window->graphic_ctx.screen_height = height;

	WindowCreate(*window);
	window->CreateVulkan();
	auto& presenter = *window->presenter;
	g_window        = std::move(window);
	return presenter;
}

void WindowRun() {
	KYTY_PROFILER_THREAD("Thread_Window");
	EXIT_IF(g_window == nullptr);

	g_window->Run();
	g_window->render_context->GetPipelineCache().Save();
}

void WindowRunBegin() {
	KYTY_PROFILER_THREAD("Thread_Window");
	EXIT_IF(g_window == nullptr);
	g_window->RunBegin();
}

bool WindowRunStep(int max_wait_ms) {
	EXIT_IF(g_window == nullptr);
	return g_window->RunStep(max_wait_ms);
}

void WindowRunEnd() {
	EXIT_IF(g_window == nullptr);
	g_window->RunEnd();
	g_window->render_context->GetPipelineCache().Save();
}

void WindowShutdown() {
	if (g_window != nullptr) {
		HostInputShutdown();
		Controller::EmergencyShutdown();
		g_window.reset();
	}
}

static int WindowIconRead(void* user, char* data, int size) {
	auto*    src        = static_cast<Common::File*>(user);
	uint32_t bytes_read = 0;
	src->Read(data, static_cast<uint32_t>(size), &bytes_read);
	return static_cast<int>(bytes_read);
}

static void WindowIconSkip(void* user, int n) {
	auto*          src      = static_cast<Common::File*>(user);
	const uint64_t position = src->Tell();

	if (n >= 0) {
		src->Seek(position + static_cast<uint64_t>(n));
	} else {
		const uint64_t distance = static_cast<uint64_t>(-static_cast<int64_t>(n));
		EXIT_IF(distance > position);
		src->Seek(position - distance);
	}
}

static int WindowIconEof(void* user) {
	auto* src = static_cast<Common::File*>(user);
	return src->IsEOF() ? 1 : 0;
}

struct WindowIcon {
	SDL_Surface* surface = nullptr;
	void*        pixels  = nullptr;

	~WindowIcon() {
		SDL_DestroySurface(surface);
		stbi_image_free(pixels);
	}
};

static void WindowLoadPngIcon(const std::filesystem::path& path, WindowIcon* icon) {
	Common::File f;
	if (!f.Open(path, Common::File::Mode::Read)) {
		EXIT("Can't open icon file %s\n", Common::PathToString(path).c_str());
	}

	int width  = 0;
	int height = 0;

	stbi_io_callbacks cb {};
	cb.read = WindowIconRead;
	cb.skip = WindowIconSkip;
	cb.eof  = WindowIconEof;

	icon->pixels = stbi_load_from_callbacks(&cb, &f, &width, &height, nullptr, 4);
	f.Close();

	EXIT_IF(icon->pixels == nullptr);

	icon->surface = SDL_CreateSurfaceFrom(width, height, SDL_PIXELFORMAT_RGBA32, icon->pixels,
	                                      width * 4);
	EXIT_NOT_IMPLEMENTED(icon->surface == nullptr);
}

void WindowContext::UpdateIcon() {
	static WindowIcon icon;
	static bool       icon_loaded = false;

	if (!icon_loaded) {
		std::filesystem::path icon_path;
		if (Loader::SystemContentGetIconPath(&icon_path)) {
			WindowLoadPngIcon(icon_path, &icon);
		}
		icon_loaded = true;
	}

	if (icon.surface != nullptr) {
		SDL_SetWindowIcon(window, icon.surface);
	}
}

void WindowContext::UpdateTitle(uint64_t frame_num, double current_fps) {
	static char title[128];
	static char title_id[12];
	static char app_ver[12];
	static bool has_title = Loader::SystemContentParamSfoGetString("TITLE", title, sizeof(title));
	static bool has_title_id =
	    Loader::SystemContentParamSfoGetString("TITLE_ID", title_id, sizeof(title_id));
	static bool has_app_ver =
	    Loader::SystemContentParamSfoGetString("APP_VER", app_ver, sizeof(app_ver));
	static const std::string processor_name = Common::GetSystemInfo().ProcessorName;

#if KYTY_BUILD == KYTY_BUILD_DEBUG
	static constexpr auto build_type = "Debug";
#elif KYTY_BUILD == KYTY_BUILD_RELEASE
	static constexpr auto build_type = "Release";
#else
	static constexpr auto build_type = "Unknown";
#endif

	const auto* device_name = graphic_ctx.GetPhysicalDeviceProperties().deviceName.data();
	auto text = fmt::format(
	    "[{} | {}] {}{}{}{}{}{}[{}] [{}], frame: {}, fps: {:.0f}", KYTY_BUILD_LABEL, build_type,
	    (has_title ? title : ""), (has_title ? ", " : ""), (has_title_id ? title_id : ""),
	    (has_title_id ? ", " : ""), (has_app_ver ? app_ver : ""), (has_app_ver ? " " : ""),
	    device_name, processor_name, frame_num, current_fps);
	if (const auto* limit = VideoOut::FrameLimitLabel()) {
		text += fmt::format(" | FPS limit: {}", limit);
	}
	// ASTROHYBRID_WINDOW_TITLE (default unset): a fixed window title instead of the diagnostic one.
	static const char* fixed_title = std::getenv("ASTROHYBRID_WINDOW_TITLE");
	if (fixed_title != nullptr && fixed_title[0] != '\0') {
		text = fixed_title;
	}

	if (CompileProgress::Enabled()) {
		auto& state       = GetProgressTitle();
		state.base        = text;
		state.first_frame = true;
		RefreshProgressTitle(window, SDL_GetTicks());
		return;
	}
	SDL_SetWindowTitle(window, text.c_str());
}

} // namespace Libs::Graphics
