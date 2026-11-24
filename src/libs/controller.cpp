#include "libs/controller.h"

#include "common/assert.h"
#include "common/common.h"
#include "common/emulatorConfig.h"
#include "common/hostInputTrace.h"
#include "common/logging/log.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "kernel/pthread.h"
#include "libs/dualSenseHaptics.h"
#include "libs/errno.h"
#include "libs/libs.h"
#include "libs/padData.h"

#include <SDL3/SDL.h>
#include <algorithm>
#include <arpa/inet.h>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <string_view>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace Libs::Controller {

LIB_NAME("Pad", "Pad");

constexpr int PAD_HANDLE = 1;
static_assert(PAD_HANDLE + Config::MAX_PLAYERS <= PAD_HANDLE_VR_LEFT);

constexpr uint32_t RUMBLE_DURATION_MS = 0xffff;
constexpr uint32_t RELEASE_FLUSH_MS   = 50;

struct PadControllerInformation {
	float    touch_pixel_density;
	uint16_t touch_resolution_x;
	uint16_t touch_resolution_y;
	uint8_t  stick_dead_zone_left;
	uint8_t  stick_dead_zone_right;
	uint8_t  connection_type;
	uint8_t  connected_count;
	bool     connected;
	int      device_class;
	uint8_t  reserve[8];
};

struct PadLightBarParam {
	uint8_t r;
	uint8_t g;
	uint8_t b;
	uint8_t reserve;
};

struct PadVibrationParam {
	uint8_t large_motor;
	uint8_t small_motor;
};

struct PadTriggerEffectCommand {
	uint32_t mode;
	uint8_t  reserve[4];
	uint8_t  data[48];
};

struct PadTriggerEffectParam {
	uint8_t                 trigger_mask;
	uint8_t                 reserve[7];
	PadTriggerEffectCommand command[2];
};

struct PadTriggerEffectStateInformation {
	int32_t state[2];
};

static_assert(sizeof(PadTriggerEffectCommand) == 56);
static_assert(sizeof(PadTriggerEffectParam) == 120);
static_assert(sizeof(PadTriggerEffectStateInformation) == 8);

struct DualSenseEffects {
	uint8_t enable_bits;
	uint8_t reserve[9];
	uint8_t right_trigger[11];
	uint8_t left_trigger[11];
};

static_assert(sizeof(DualSenseEffects) == 32);

struct ControllerState {
	struct Touch {
		uint8_t  id   = 0;
		bool     down = false;
		uint16_t x    = 0;
		uint16_t y    = 0;
	};

	uint64_t time                                  = 0;
	uint32_t buttons                               = 0;
	int      axes[static_cast<int>(Axis::AxisMax)] = {128, 128, 128, 128, 0, 0};
	Touch    touch[2];
	std::array<float, 3> accel {0.0f, 1.0f, 0.0f};
	std::array<float, 3> gyro {};
	std::array<float, 4> orientation {0.0f, 0.0f, 0.0f, 1.0f};
};

class GameController {
public:
	explicit GameController(int player): m_player(player) {}

	KYTY_CLASS_NO_COPY(GameController);

	[[nodiscard]] int Handle() const { return PAD_HANDLE + m_player; }

	int InitPort();
	int OpenPort(int user_id, int type, int index);
	int GetPortHandle(int user_id, int type, int index);
	int ClosePort(int handle);
	int CheckPortHandle(int handle);

	void SetDevices(int active_id);
	void Button(int id, uint32_t button, bool down);
	void Axis(int id, Axis axis, int value);
	void RightStick(int id, int x, int y);
	void TouchPad(int id, int finger, bool down, float x, float y);
	void Motion(int id, Sensor sensor, const float* data, uint64_t time_us);
	void SetMotionSensorState(bool enable);
	void ResetOrientation();
	void ToggleGyroEmulation(int id);
	void ResetInputState();
	void GetConnectionInfo(bool* flag, int* count);
	void SetVibration(uint8_t large_motor, uint8_t small_motor);
	int  GetActiveControllerId();
	void SetLightBar(uint8_t r, uint8_t g, uint8_t b);
	bool SetTriggerEffect(const PadTriggerEffectParam& param);
	void GetTriggerEffectState(int32_t* state);
	void  CycleSetting(Setting setting);
	float GetSettingScale(Setting setting) const;
	void ReadState(ControllerState* state, bool* flag, int* count);
	int  ReadStates(ControllerState* states, int states_num, bool* flag, int* count);

private:
	static constexpr uint32_t STATES_MAX = 64;
	enum class PortState { Uninitialized, Closed, Open };

	int CheckPortHandleLocked(int handle) const;
	[[nodiscard]] bool PadOpenArgsAreValid(int user_id, int type, int index) const;

	const int m_player;

	void AddState();
	// Emulated tilt: the right stick drives pitch and roll instead of the stick axes.
	void UpdateEmulatedTilt();
	// Apply cached output without changing its game-requested lifetime; caller holds m_mutex.
	void ApplyVibration();
	bool SendTriggerEffect(const PadTriggerEffectParam& param);

	Common::Mutex    m_mutex;
	PortState        m_port_state      = PortState::Uninitialized;
	int              m_active_id       = -1;
	bool             m_connected       = false;
	int              m_connected_count = 0;
	bool             m_motion_enabled  = true;
	uint64_t         m_gyro_time       = 0;
	bool             m_gyro_emulation  = false;
	uint64_t         m_emulated_time   = 0;
	std::array<int, 2> m_emulated_stick {128, 128};
	// Accelerometer gravity direction in the reset frame.
	std::array<float, 3> m_up_reference {0.0f, 1.0f, 0.0f};
	bool                 m_up_reference_valid = false;
	ControllerState  m_state;
	ControllerState  m_states[STATES_MAX];
	bool             m_obtained[STATES_MAX] {};
	uint32_t         m_states_num    = 0;
	uint32_t         m_first_state   = 0;
	uint8_t          m_next_touch_id = 1;
	// The game's last requests, re-sent when their intensity setting changes.
	std::array<uint8_t, 2> m_vibration {};
	uint64_t               m_vibration_until = 0;
	PadTriggerEffectParam  m_trigger_effect {};
	// Every effect the game set, for scePadGetTriggerEffectState (m_trigger_effect only keeps sent ones).
	PadTriggerEffectParam  m_state_effect {};
	// Setting changes share the output lock; audio only needs an atomic scale snapshot.
	std::array<std::atomic<uint32_t>, 3> m_setting_steps {};
};

static std::array<std::unique_ptr<GameController>, Config::MAX_PLAYERS> g_players;

// Lock order (pad code): SDL's joystick lock (SDL_LockJoysticks, also taken inside every SDL
// gamepad call) is always the INNERMOST lock. Never acquire a Kyty pad mutex (g_router_mutex,
// GameController::m_mutex, the haptics locks) while holding the SDL joystick lock, and do not hold
// GameController::m_mutex across SDL gamepad calls in SetDevices / StopHostPad (they run on the
// main thread from pad-added events). The inverse order deadlocked the main thread (window pump)
// against the pad relay thread: spinning wheel, controller dead.

// Maps host gamepads to players; the keyboard follows the configured keyboard player.
static Common::Mutex g_router_mutex;
static std::vector<std::pair<int, int>> g_pad_players;

static uint32_t g_keyboard_player = 0;

static void RefreshPlayers();
static void StopHostPad(SDL_Gamepad* pad);

static int KeyboardPlayer() {
	return g_keyboard_player == 0 ? 0 : static_cast<int>(g_keyboard_player) - 1;
}

// Caller holds g_router_mutex.
static GameController* PlayerForDevice(int id) {
	if (id == HOST_INPUT_CONTROLLER_ID) {
		return g_players[KeyboardPlayer()].get();
	}
	for (const auto& [pad, player]: g_pad_players) {
		if (pad == id) {
			return player >= 0 ? g_players[player].get() : nullptr;
		}
	}
	return nullptr;
}

static GameController* PlayerForHandle(int handle) {
	if (handle >= PAD_HANDLE && handle < PAD_HANDLE + Config::MAX_PLAYERS) {
		return g_players[handle - PAD_HANDLE].get();
	}
	// The Sense controller ports belong to the initial user.
	return PadVrHand(handle) >= 0 ? g_players[0].get() : nullptr;
}

// A trigger's travel, 0..255. An L2/R2 that arrives as a button only (input remapping, pads or tools
// that report the trigger digitally) counts as full travel, as for the keyboard.
static int TriggerTravel(const ControllerState& state, Axis axis) {
	const auto button = axis == Axis::TriggerLeft ? PAD_BUTTON_L2 : PAD_BUTTON_R2;
	const int  value  = state.axes[static_cast<int>(axis)];
	// An analog press also sets the button, so only a button without any travel is digital.
	return (state.buttons & button) != 0 && value == 0 ? 255 : value;
}

static void pad_fill_data(PadData* data, const ControllerState& state, bool connected,
                          int connected_count) {
	EXIT_IF(data == nullptr);

	std::memset(data, 0, sizeof(*data));

	data->buttons           = state.buttons;
	data->left_stick_x      = state.axes[static_cast<int>(Axis::LeftX)];
	data->left_stick_y      = state.axes[static_cast<int>(Axis::LeftY)];
	data->right_stick_x     = state.axes[static_cast<int>(Axis::RightX)];
	data->right_stick_y     = state.axes[static_cast<int>(Axis::RightY)];
	data->analog_buttons_l2 = static_cast<uint8_t>(TriggerTravel(state, Axis::TriggerLeft));
	data->analog_buttons_r2 = static_cast<uint8_t>(TriggerTravel(state, Axis::TriggerRight));
	data->acceleration_x     = state.accel[0];
	data->acceleration_y     = state.accel[1];
	data->acceleration_z     = state.accel[2];
	data->angular_velocity_x = state.gyro[0];
	data->angular_velocity_y = state.gyro[1];
	data->angular_velocity_z = state.gyro[2];
	data->orientation_x      = state.orientation[0];
	data->orientation_y      = state.orientation[1];
	data->orientation_z      = state.orientation[2];
	data->orientation_w      = state.orientation[3];
	for (const auto& touch: state.touch) {
		if (touch.down) {
			auto& output = data->touch_data.touch[data->touch_data.touch_num++];
			output.x     = touch.x;
			output.y     = touch.y;
			output.id    = touch.id;
		}
	}
	data->connected              = connected;
	data->timestamp              = state.time;
	data->connected_count        = static_cast<uint8_t>(std::min(connected_count, 255));
	data->device_unique_data_len = 0;
}

// In cycle order, starting at the PS5 default.
constexpr std::array SPEAKER_VOLUME = {1.0f, 0.0f, 0.02f, 0.12f, 0.42f};
constexpr std::array INTENSITY      = {1.0f, 0.0f, 0.33f, 0.66f};

void CycleSetting(Setting setting) {
	for (const auto& player: g_players) {
		if (player != nullptr) {
			player->CycleSetting(setting);
		}
	}
}

float GetSettingScale(Setting setting) {
	return g_players[0] != nullptr ? g_players[0]->GetSettingScale(setting) : 1.0f;
}

void GameController::CycleSetting(Setting setting) {
	Common::LockGuard lock(m_mutex);
	auto&             step = m_setting_steps[static_cast<size_t>(setting)];
	const auto count = setting == Setting::SpeakerVolume ? SPEAKER_VOLUME.size() : INTENSITY.size();
	step.store((step.load(std::memory_order_relaxed) + 1) % count, std::memory_order_relaxed);
	if (setting == Setting::VibrationIntensity) {
		ApplyVibration();
	} else if (setting == Setting::TriggerEffectIntensity) {
		(void)SendTriggerEffect(m_trigger_effect);
	}
}

float GameController::GetSettingScale(Setting setting) const {
	const auto step = m_setting_steps[static_cast<size_t>(setting)].load(std::memory_order_relaxed);
	if (setting == Setting::SpeakerVolume) {
		// The old speaker level matches the PS5 slider's midpoint. Give the upper
		// half of the global slider up to 6 dB of extra software gain.
		return SPEAKER_VOLUME[step] * (Config::GetControllerSpeakerVolume() / 50.0f);
	}
	if (setting == Setting::VibrationIntensity) {
		return INTENSITY[step] * (Config::GetControllerVibrationIntensity() / 100.0f);
	}
	return INTENSITY[step];
}

static uint8_t Scale(uint8_t value, float scale) {
	if (value == 0 || scale == 0.0f) {
		return 0;
	}
	return static_cast<uint8_t>(std::max(1L, std::lround(value * scale)));
}

static bool trigger_effect_zones(uint8_t* effect, const uint8_t* strengths, float scale,
                                 uint8_t type, uint8_t frequency = 0) {
	uint16_t active = 0;
	uint32_t packed = 0;
	for (int i = 0; i < 10; i++) {
		if (strengths[i] > 8) {
			return false;
		}
		const auto strength = Scale(strengths[i], scale);
		if (strength != 0) {
			active |= static_cast<uint16_t>(1u << i);
			packed |= static_cast<uint32_t>(strength - 1u) << (3 * i);
		}
	}

	effect[0] = active != 0 && (type != 0x26 || frequency != 0) ? type : 0x05;
	effect[1] = static_cast<uint8_t>(active);
	effect[2] = static_cast<uint8_t>(active >> 8u);
	effect[3] = static_cast<uint8_t>(packed);
	effect[4] = static_cast<uint8_t>(packed >> 8u);
	effect[5] = static_cast<uint8_t>(packed >> 16u);
	effect[6] = static_cast<uint8_t>(packed >> 24u);
	effect[9] = frequency;
	return true;
}

static bool trigger_effect_to_dualsense(const PadTriggerEffectCommand& command, uint8_t* effect,
                                        float scale) {
	std::memset(effect, 0, 11);
	effect[0] = 0x05;

	uint8_t strengths[10] = {};
	switch (command.mode) {
		case 0: return true;
		case 1:
			if (command.data[0] > 9 || command.data[1] > 8) {
				return false;
			}
			std::fill(strengths + command.data[0], strengths + 10, command.data[1]);
			return trigger_effect_zones(effect, strengths, scale, 0x21);
		case 2: {
			const auto start    = command.data[0];
			const auto end      = command.data[1];
			const auto strength = command.data[2];
			if (start < 2 || start > 7 || end <= start || end > 8 || strength > 8) {
				return false;
			}
			const auto scaled = Scale(strength, scale);
			if (scaled == 0) {
				return true;
			}
			const uint16_t zones = static_cast<uint16_t>((1u << start) | (1u << end));
			effect[0]            = 0x25;
			effect[1]            = static_cast<uint8_t>(zones);
			effect[2]            = static_cast<uint8_t>(zones >> 8u);
			effect[3]            = scaled - 1u;
			return true;
		}
		case 3:
			if (command.data[0] > 9 || command.data[1] > 8) {
				return false;
			}
			std::fill(strengths + command.data[0], strengths + 10, command.data[1]);
			return trigger_effect_zones(effect, strengths, scale, 0x26, command.data[2]);
		case 4: return trigger_effect_zones(effect, command.data, scale, 0x21);
		case 5: {
			const int start          = command.data[0];
			const int end            = command.data[1];
			const int start_strength = command.data[2];
			const int end_strength   = command.data[3];
			if (start > 8 || end <= start || end > 9 || start_strength < 1 || start_strength > 8 ||
			    end_strength < 1 || end_strength > 8) {
				return false;
			}
			for (int i = start; i < 10; i++) {
				const int delta  = (end_strength - start_strength) * (i - start);
				const int length = end - start;
				strengths[i]     = static_cast<uint8_t>(
				    i >= end ? end_strength
				             : start_strength +
				                   (delta + (delta < 0 ? -length / 2 : length / 2)) / length);
			}
			return trigger_effect_zones(effect, strengths, scale, 0x21);
		}
		case 6: return trigger_effect_zones(effect, command.data + 1, scale, 0x26, command.data[0]);
		default: return false;
	}
}

// scePadGetTriggerEffectState values (libScePad; duaLib names them SCE_PAD_TRIGGER_STATE_*).
enum TriggerState : int32_t {
	TRIGGER_STATE_FEEDBACK_NO_FORCE     = 1,
	TRIGGER_STATE_FEEDBACK_IS_PUSHING   = 2,
	TRIGGER_STATE_WEAPON_NOT_PRESSED    = 3,
	TRIGGER_STATE_WEAPON_ALMOST_PRESSED = 4,
	TRIGGER_STATE_WEAPON_FULLY_PRESSED  = 5,
	TRIGGER_STATE_VIBRATION_NOT_FIRING  = 6,
	TRIGGER_STATE_VIBRATION_IS_FIRING   = 7,
};

// A real DualSense reports which effect runs and where the trigger is against it. Without that
// report (SDL does not expose it, and other pads have no trigger motors) the state follows from the
// game's effect and the trigger's travel: 0..255 maps to the effect's ten positions 0..9. Games
// such as Astro Bot wait for the weapon or feedback state before acting on L2/R2; the state
// stayed 0 before, so those actions never happened.
// Travel below which a trigger counts as released (analog rest noise).
constexpr int TRIGGER_PRESSED_MIN = 8;

int32_t TriggerEffectState(const PadTriggerEffectCommand& command, int value) {
	const int pos = std::clamp(value, 0, 255) * 10 / 256;
	// A feedback trigger pushes, and a vibration trigger fires, only while it is pressed: an effect from
	// position 0 on an untouched trigger is "no force" (Astro's Playroom waits for the change).
	const bool pressed = value >= TRIGGER_PRESSED_MIN;
	switch (command.mode) {
		case 1: // feedback: resistance from position data[0], strength data[1]
			return pressed && command.data[1] != 0 && pos >= command.data[0] ? TRIGGER_STATE_FEEDBACK_IS_PUSHING
			                                                      : TRIGGER_STATE_FEEDBACK_NO_FORCE;
		case 2: // weapon: resistance from data[0], released at data[1]
			if (pos >= command.data[1]) {
				return TRIGGER_STATE_WEAPON_FULLY_PRESSED;
			}
			return pos >= command.data[0] ? TRIGGER_STATE_WEAPON_ALMOST_PRESSED
			                              : TRIGGER_STATE_WEAPON_NOT_PRESSED;
		case 3: // vibration from position data[0], amplitude data[1], frequency data[2]
			return pressed && command.data[1] != 0 && command.data[2] != 0 && pos >= command.data[0]
			           ? TRIGGER_STATE_VIBRATION_IS_FIRING
			           : TRIGGER_STATE_VIBRATION_NOT_FIRING;
		case 4: // multiple-position feedback: strength per position
			return pressed && command.data[pos] != 0 ? TRIGGER_STATE_FEEDBACK_IS_PUSHING
			                              : TRIGGER_STATE_FEEDBACK_NO_FORCE;
		case 5: // slope feedback from position data[0]
			return pressed && pos >= command.data[0] ? TRIGGER_STATE_FEEDBACK_IS_PUSHING
			                              : TRIGGER_STATE_FEEDBACK_NO_FORCE;
		case 6: // multiple-position vibration: frequency data[0], amplitude per position
			return pressed && command.data[0] != 0 && command.data[1 + pos] != 0
			           ? TRIGGER_STATE_VIBRATION_IS_FIRING
			           : TRIGGER_STATE_VIBRATION_NOT_FIRING;
		default: return 0;
	}
}

void GetTriggerEffectState(int32_t* state) {
	if (g_players[0] == nullptr) {
		state[0] = 0;
		state[1] = 0;
		return;
	}
	g_players[0]->GetTriggerEffectState(state);
}

// NPS5_PAD_NET=<port>: a controller forwarded from another machine (28-byte packets over a loopback
// TCP stream, normally an SSH local forward) drives the host-input port like the keyboard does. Off
// unless the variable is set; listens on 127.0.0.1 only, one client; no packet for 500 ms or a
// disconnect releases everything. No rumble goes back.
//   u16 magic 0x504e | u8 ver 1 | u8 flags | u32 seq | u32 ms | u32 buttons (bit n = SDL gamepad
//   button n) | i16 axes[6]
namespace {
std::atomic<bool> g_netpad_stop {false};
std::thread       g_netpad_thread;

struct NetPadState {
	uint32_t pad = 0; // PS button bits currently pressed by the forwarded pad
	int      axes[6] {128, 128, 128, 128, 0, 0};
};

void NetPadApply(NetPadState& shown, const NetPadState& want) {
	for (uint32_t bit = 1; bit != 0; bit <<= 1) {
		if ((want.pad & bit) != (shown.pad & bit)) {
			SetButton(HOST_INPUT_CONTROLLER_ID, bit, (want.pad & bit) != 0);
		}
	}
	static constexpr Axis kAxes[6] = {Axis::LeftX,  Axis::LeftY,       Axis::RightX,
	                                  Axis::RightY, Axis::TriggerLeft, Axis::TriggerRight};
	for (int i = 0; i < 6; i++) {
		if (want.axes[i] != shown.axes[i]) {
			SetAxis(HOST_INPUT_CONTROLLER_ID, kAxes[i], want.axes[i]);
		}
	}
	shown = want;
}

NetPadState NetPadFromPacket(const uint8_t* p) {
	static constexpr struct {
		int      sdl;
		uint32_t ps;
	} kMap[] = {{0, PAD_BUTTON_CROSS},    {1, PAD_BUTTON_CIRCLE},    {2, PAD_BUTTON_SQUARE},
	            {3, PAD_BUTTON_TRIANGLE}, {4, PAD_BUTTON_TOUCH_PAD}, {6, PAD_BUTTON_OPTIONS},
	            {7, PAD_BUTTON_L3},       {8, PAD_BUTTON_R3},        {9, PAD_BUTTON_L1},
	            {10, PAD_BUTTON_R1},      {11, PAD_BUTTON_UP},       {12, PAD_BUTTON_DOWN},
	            {13, PAD_BUTTON_LEFT},    {14, PAD_BUTTON_RIGHT}};
	const uint32_t buttons =
	    p[12] | (p[13] << 8) | (p[14] << 16) | (static_cast<uint32_t>(p[15]) << 24);
	NetPadState state;
	for (const auto& m: kMap) {
		if ((buttons >> m.sdl) & 1u) {
			state.pad |= m.ps;
		}
	}
	for (int i = 0; i < 6; i++) {
		const int v =
		    static_cast<int16_t>(static_cast<uint16_t>(p[16 + 2 * i] | (p[17 + 2 * i] << 8)));
		state.axes[i] = i < 4 ? std::clamp((v + 32768) >> 8, 0, 255) : std::clamp(v >> 7, 0, 255);
	}
	return state;
}

void NetPadRun(int listen_fd) {
	using Clock              = std::chrono::steady_clock;
	int               client = -1;
	uint8_t           pending[28];
	size_t            pending_size = 0;
	NetPadState       shown;
	Clock::time_point last_packet {};
	bool              have = false;
	while (!g_netpad_stop) {
		pollfd fds[2] = {{listen_fd, POLLIN, 0}, {client, POLLIN, 0}};
		if (poll(fds, client >= 0 ? 2 : 1, 50) > 0) {
			if (fds[0].revents & POLLIN) {
				const int next = accept(listen_fd, nullptr, nullptr);
				if (next >= 0) {
					if (client >= 0) {
						close(client);
					}
					client = next;
					int on = 1;
					setsockopt(client, IPPROTO_TCP, TCP_NODELAY, &on, sizeof on);
					pending_size = 0;
					have         = false;
					NetPadApply(
					    shown,
					    NetPadState {}); // replaced sender: release whatever the old one held
					std::printf("[netpad] sender connected\n");
				}
			}
			if (client >= 0 && (fds[1].revents & (POLLIN | POLLHUP | POLLERR))) {
				uint8_t    buf[256];
				const auto got = recv(client, buf, sizeof buf, 0);
				bool       ok  = got > 0;
				for (ssize_t i = 0; ok && i < got;) {
					const size_t take = std::min<size_t>(sizeof pending - pending_size,
					                                     static_cast<size_t>(got - i));
					std::memcpy(pending + pending_size, buf + i, take);
					pending_size += take;
					i += static_cast<ssize_t>(take);
					if (pending_size == sizeof pending) {
						pending_size = 0;
						if (pending[0] != 0x4e || pending[1] != 0x50 || pending[2] != 1) {
							ok = false;
							break;
						}
						NetPadApply(shown, NetPadFromPacket(pending));
						last_packet = Clock::now();
						have        = true;
					}
				}
				if (!ok) {
					close(client);
					client = -1;
					have   = false;
					NetPadApply(shown, NetPadState {});
					std::printf("[netpad] sender disconnected\n");
				}
			}
		}
		if (have && Clock::now() - last_packet > std::chrono::milliseconds(500)) {
			have = false;
			NetPadApply(shown, NetPadState {}); // silent sender: neutral
		}
	}
	if (client >= 0) {
		close(client);
	}
	NetPadApply(shown, NetPadState {});
	close(listen_fd);
}

void NetPadStart() {
	const char* text = std::getenv("NPS5_PAD_NET");
	if (text == nullptr || *text == '\0' || g_netpad_thread.joinable()) {
		return;
	}
	const int port = std::atoi(text);
	if (port <= 0 || port > 65535) {
		std::printf("[netpad] NPS5_PAD_NET='%s' is not a port\n", text);
		return;
	}
	const int fd = socket(AF_INET, SOCK_STREAM, 0);
	int       on = 1;
	setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &on, sizeof on);
	sockaddr_in addr {};
	addr.sin_family      = AF_INET;
	addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK); // loopback only
	addr.sin_port        = htons(static_cast<uint16_t>(port));
	if (fd < 0 || bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof addr) != 0 ||
	    listen(fd, 1) != 0) {
		std::printf("[netpad] cannot listen on 127.0.0.1:%d\n", port);
		if (fd >= 0) {
			close(fd);
		}
		return;
	}
	g_netpad_stop   = false;
	g_netpad_thread = std::thread(NetPadRun, fd);
	std::printf("[netpad] listening on 127.0.0.1:%d (NPS5_PAD_NET)\n", port);
}

void NetPadStop() {
	g_netpad_stop = true;
	if (g_netpad_thread.joinable()) {
		g_netpad_thread.join();
	}
}
} // namespace

void Initialize() {
	EXIT_IF(g_players[0] != nullptr);

	for (int i = 0; i < Config::MAX_PLAYERS; i++) {
		g_players[i] = std::make_unique<GameController>(i);
	}
	g_keyboard_player = Config::GetKeyboardPlayer();
	RefreshPlayers();
	NetPadStart();
}

void Shutdown() {
	NetPadStop();
	EmergencyShutdown();
	for (auto& player: g_players) {
		player.reset();
	}
}

void EmergencyShutdown() {
	Common::LockGuard lock(g_router_mutex);
	DualSenseHaptics::Shutdown();
	std::vector<SDL_Gamepad*> pads;
	for (const auto& [id, player]: g_pad_players) {
		if (auto* pad = SDL_GetGamepadFromID(static_cast<SDL_JoystickID>(id))) {
			StopHostPad(pad);
			pads.push_back(pad);
		}
	}
	if (!pads.empty()) {
		SDL_Delay(RELEASE_FLUSH_MS);
		for (auto* pad: pads) {
			SDL_CloseGamepad(pad);
		}
	}
	g_pad_players.clear();
	for (const auto& player: g_players) {
		if (player != nullptr) {
			player->SetDevices(-1);
		}
	}
}

static bool HostInputOnly() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_HOST_INPUT_ONLY");
		return value != nullptr && std::strcmp(value, "1") == 0;
	}();
	return enabled;
}

void GameController::SetDevices(int active_id) {
	{
		Common::LockGuard lock(m_mutex);
		if (m_active_id == active_id) {
			return;
		}
	}
	// SDL gamepad calls take SDL's joystick lock: make them without m_mutex (lock order above). The
	// callers serialise SetDevices through g_router_mutex, so the state below is still set once per
	// change.
	if (active_id != HOST_INPUT_CONTROLLER_ID && active_id != -1) {
		if (auto* pad = SDL_GetGamepadFromID(static_cast<SDL_JoystickID>(active_id));
		    pad != nullptr) {
			if (const auto& color = Config::GetControllerColor()) {
				(void)SDL_SetGamepadLED(pad, (*color)[0], (*color)[1], (*color)[2]);
			}
			for (auto sensor: {SDL_SENSOR_ACCEL, SDL_SENSOR_GYRO}) {
				if (SDL_GamepadHasSensor(pad, sensor) &&
				    !SDL_SetGamepadSensorEnabled(pad, sensor, true)) {
					LOGF("\t enabling controller sensor failed: %s\n", SDL_GetError());
				}
			}
		}
	}

	Common::LockGuard lock(m_mutex);
	if (m_active_id == active_id) {
		return;
	}
	const bool new_connected = active_id != -1;
	if (!m_connected && new_connected) {
		m_connected_count++;
	}
	m_active_id     = active_id;
	m_connected     = new_connected;
	m_state         = {};
	m_gyro_time     = 0;
	m_gyro_emulation = false;
	m_emulated_time  = 0;
	m_emulated_stick = {128, 128};
	m_up_reference_valid = false;
	m_states_num    = 0;
	m_first_state   = 0;
	m_next_touch_id = 1;
	m_vibration          = {};
	m_vibration_until    = 0;
	m_trigger_effect     = {};
	m_state_effect       = {};
}

// KYTY_PAD_STATE_COALESCE=1 (default off): an input event that changes no button and no touch
// (a stick or trigger step, a finger moving) updates the newest state while no read has taken it,
// instead of adding a state. scePadRead returns the unread states oldest first; a moving stick
// adds one per report, so more than 63 events between two reads (a frame hitch) pushed an unread
// press, or a whole tap, out of the 64-state history. With it every press and release keeps a
// state of its own and the newest state has the latest sticks. Unread states dropped are counted
// and reported on stdout at most every 5 s, with the switch on or off.
// Ported from chenxiao07/KytyPS5 167ed5886 (Stefano Scalfari).
static bool PadStateCoalescing() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_PAD_STATE_COALESCE");
		return value != nullptr && std::string_view(value) == "1";
	}();
	return on;
}

void GameController::AddState() {
	if (m_states_num != 0 && PadStateCoalescing()) {
		const auto  newest = (m_first_state + m_states_num - 1) % STATES_MAX;
		const auto& last   = m_states[newest];
		if (!m_obtained[newest] && last.buttons == m_state.buttons &&
		    last.touch[0].down == m_state.touch[0].down && last.touch[0].id == m_state.touch[0].id &&
		    last.touch[1].down == m_state.touch[1].down && last.touch[1].id == m_state.touch[1].id) {
			m_states[newest] = m_state;
			return;
		}
	}
	if (m_states_num >= STATES_MAX) {
		if (!m_obtained[m_first_state]) {
			static uint64_t dropped = 0, unreported = 0;
			static auto     reported = std::chrono::steady_clock::time_point {};
			dropped++;
			unreported++;
			const auto now = std::chrono::steady_clock::now();
			if (now - reported >= std::chrono::seconds(5)) {
				std::printf("Pad: %llu input states dropped unread (the 64-state history was full; "
				            "%llu in all)\n",
				            static_cast<unsigned long long>(unreported),
				            static_cast<unsigned long long>(dropped));
				unreported = 0;
				reported   = now;
			}
		}
		m_states_num  = STATES_MAX - 1;
		m_first_state = (m_first_state + 1) % STATES_MAX;
	}

	const auto index  = (m_first_state + m_states_num) % STATES_MAX;
	m_states[index]   = m_state;
	m_obtained[index] = false;
	m_states_num++;
}

void GameController::Button(int id, uint32_t button, bool down) {
	Common::LockGuard lock(m_mutex);

	// Keyboard events are routed to the configured player's port.
	if (m_active_id == id || id == HOST_INPUT_CONTROLLER_ID) {
		m_state.time = LibKernel::KernelGetProcessTime();

		m_state.buttons = down ? m_state.buttons | button : m_state.buttons & ~button;

		AddState();
	}
}

void GameController::Axis(int id, Controller::Axis axis, int value) {
	Common::LockGuard lock(m_mutex);

	if (m_active_id == id || id == HOST_INPUT_CONTROLLER_ID) {
		m_state.time = LibKernel::KernelGetProcessTime();

		int axis_id = static_cast<int>(axis);

		EXIT_IF(axis_id < 0 || axis_id >= static_cast<int>(Controller::Axis::AxisMax));

		if (m_gyro_emulation && (axis == Controller::Axis::RightX || axis == Controller::Axis::RightY)) {
			m_emulated_stick[axis == Controller::Axis::RightY ? 1 : 0] = value;
			UpdateEmulatedTilt();
			return;
		}

		m_state.axes[axis_id] = value;
		Common::HostInputTrace("axis", id, axis_id, value);

		uint32_t trigger = 0;
		if (axis == Controller::Axis::TriggerLeft) {
			trigger = PAD_BUTTON_L2;
		} else if (axis == Controller::Axis::TriggerRight) {
			trigger = PAD_BUTTON_R2;
		}
		if (trigger != 0) {
			m_state.buttons = value > 0 ? m_state.buttons | trigger : m_state.buttons & ~trigger;
		}

		AddState();
	}
}

void GameController::RightStick(int id, int x, int y) {
	Common::LockGuard lock(m_mutex);

	if (m_active_id == id || id == HOST_INPUT_CONTROLLER_ID) {
		m_state.time = LibKernel::KernelGetProcessTime();
		if (m_gyro_emulation) {
			m_emulated_stick = {x, y};
			UpdateEmulatedTilt();
			return;
		}
		m_state.axes[static_cast<int>(Axis::RightX)] = x;
		m_state.axes[static_cast<int>(Axis::RightY)] = y;
		AddState();
	}
}

void GameController::TouchPad(int id, int finger, bool down, float x, float y) {
	if (finger < 0 || finger >= 2) {
		return;
	}

	Common::LockGuard lock(m_mutex);
	if (m_active_id == id || id == HOST_INPUT_CONTROLLER_ID) {
		auto& touch  = m_state.touch[finger];
		m_state.time = LibKernel::KernelGetProcessTime();
		if (down && !touch.down) {
			touch.id        = m_next_touch_id;
			m_next_touch_id = m_next_touch_id == 127 ? 1 : m_next_touch_id + 1;
		}
		touch.down = down;
		touch.x    = static_cast<uint16_t>(std::clamp(x, 0.0f, 1.0f) * 1920.0f);
		touch.y    = static_cast<uint16_t>(std::clamp(y, 0.0f, 1.0f) * 943.0f);
		if (id == HOST_INPUT_CONTROLLER_ID) {
			m_state.buttons = down ? m_state.buttons | PAD_BUTTON_TOUCH_PAD
			                       : m_state.buttons & ~PAD_BUTTON_TOUCH_PAD;
		}
		AddState();
	}
}

namespace {

using Vec3 = std::array<float, 3>;
using Quat = std::array<float, 4>; // x, y, z, w

Vec3 QuatRotate(const Quat& q, const Vec3& v) {
	const Vec3 t {2.0f * (q[1] * v[2] - q[2] * v[1]), 2.0f * (q[2] * v[0] - q[0] * v[2]),
	              2.0f * (q[0] * v[1] - q[1] * v[0])};
	return {v[0] + q[3] * t[0] + (q[1] * t[2] - q[2] * t[1]),
	        v[1] + q[3] * t[1] + (q[2] * t[0] - q[0] * t[2]),
	        v[2] + q[3] * t[2] + (q[0] * t[1] - q[1] * t[0])};
}

Quat QuatConjugate(const Quat& q) {
	return {-q[0], -q[1], -q[2], q[3]};
}

float Length(const Vec3& v) {
	return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

} // namespace

void GameController::Motion(int id, Sensor sensor, const float* data, uint64_t time_us) {
	Common::LockGuard lock(m_mutex);
	if (id != m_active_id || !m_motion_enabled) {
		return;
	}

	// Convert acceleration to G; angular velocity is already in rad/s.
	if (sensor == Sensor::Accel) {
		for (int i = 0; i < 3; i++) {
			m_state.accel[i] = data[i] / SDL_STANDARD_GRAVITY;
		}
		// Capture gravity near 1 G to limit interference from linear acceleration.
		const float magnitude = Length(m_state.accel);
		if (!m_up_reference_valid && magnitude > 0.9f && magnitude < 1.1f) {
			const Vec3 up {m_state.accel[0] / magnitude, m_state.accel[1] / magnitude,
			               m_state.accel[2] / magnitude};
			m_up_reference       = QuatRotate(m_state.orientation, up);
			m_up_reference_valid = true;
		}
	} else {
		std::copy_n(data, 3, m_state.gyro.begin());
		// Mahony tilt correction; heading is unobservable and reported gyro values stay raw.
		auto        rate      = m_state.gyro;
		const float magnitude = Length(m_state.accel);
		if (m_up_reference_valid && magnitude > 0.8f && magnitude < 1.2f) {
			const Vec3 measured {m_state.accel[0] / magnitude, m_state.accel[1] / magnitude,
			                     m_state.accel[2] / magnitude};
			const Vec3 predicted = QuatRotate(QuatConjugate(m_state.orientation), m_up_reference);
			rate[0] += measured[1] * predicted[2] - measured[2] * predicted[1];
			rate[1] += measured[2] * predicted[0] - measured[0] * predicted[2];
			rate[2] += measured[0] * predicted[1] - measured[1] * predicted[0];
		}
		// Do not extrapolate a single sample across lost reports (e.g. loss of window focus).
		constexpr uint64_t max_gyro_interval_us = 100000;
		if (m_gyro_time != 0 && time_us > m_gyro_time &&
		    time_us - m_gyro_time <= max_gyro_interval_us) {
			const float dt         = static_cast<float>(time_us - m_gyro_time) * 0.000001f;
			const float speed      = Length(rate);
			const float half_angle = speed * dt * 0.5f;
			const float scale      = speed > 0.0f ? std::sin(half_angle) / speed : 0.0f;
			const float x          = rate[0] * scale;
			const float y          = rate[1] * scale;
			const float z          = rate[2] * scale;
			const float w          = std::cos(half_angle);
			const auto  q          = m_state.orientation;
			// Accumulate body-local rotation relative to connection / orientation reset.
			m_state.orientation = {q[3] * x + q[0] * w + q[1] * z - q[2] * y,
			                       q[3] * y - q[0] * z + q[1] * w + q[2] * x,
			                       q[3] * z + q[0] * y - q[1] * x + q[2] * w,
			                       q[3] * w - q[0] * x - q[1] * y - q[2] * z};
			float length        = 0.0f;
			for (float value: m_state.orientation) {
				length += value * value;
			}
			length = std::sqrt(length);
			for (float& value: m_state.orientation) {
				value /= length;
			}
		}
		m_gyro_time = time_us;
	}
	m_state.time = LibKernel::KernelGetProcessTime();
	AddState();
}

void GameController::SetMotionSensorState(bool enable) {
	Common::LockGuard lock(m_mutex);
	if (m_motion_enabled != enable) {
		m_motion_enabled     = enable;
		m_gyro_time          = 0;
		m_state.accel        = {0.0f, 1.0f, 0.0f};
		m_state.gyro         = {};
		m_up_reference_valid = false;
		m_state.time         = LibKernel::KernelGetProcessTime();
		AddState();
	}
}

static Quat QuatMultiply(const Quat& a, const Quat& b) {
	return {a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
	        a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
	        a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
	        a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2]};
}

void GameController::UpdateEmulatedTilt() {
	constexpr float max_angle = 0.7853982f; // 45 degrees at full stick deflection
	const auto      angle     = [](int value) {
        return std::clamp((value - 128) / 127.0f, -1.0f, 1.0f) * max_angle;
	};
	const float roll  = angle(m_emulated_stick[0]);
	const float pitch = angle(m_emulated_stick[1]);

	const Quat q_pitch {std::sin(pitch * 0.5f), 0.0f, 0.0f, std::cos(pitch * 0.5f)};
	const Quat q_roll {0.0f, 0.0f, std::sin(-roll * 0.5f), std::cos(-roll * 0.5f)};
	Quat       q = QuatMultiply(q_roll, q_pitch);

	const uint64_t now = LibKernel::KernelGetProcessTime();
	Vec3           rate {};
	if (m_emulated_time != 0 && now > m_emulated_time) {
		const float dt = std::clamp(static_cast<float>(now - m_emulated_time) * 0.000001f,
		                            0.001f, 0.1f);
		auto delta = QuatMultiply(QuatConjugate(m_state.orientation), q);
		if (delta[3] < 0.0f) {
			for (auto& v: delta) {
				v = -v;
			}
		}
		rate = {2.0f * delta[0] / dt, 2.0f * delta[1] / dt, 2.0f * delta[2] / dt};
	}
	m_emulated_time = now;

	m_state.time        = now;
	m_state.orientation = q;
	m_state.accel       = QuatRotate(QuatConjugate(q), {0.0f, 1.0f, 0.0f});
	m_state.gyro        = rate;
	AddState();
	// Angular velocity is only nonzero while the tilt changes.
	m_state.gyro = {};
}

void GameController::ToggleGyroEmulation(int id) {
	Common::LockGuard lock(m_mutex);
	if (id != HOST_INPUT_CONTROLLER_ID && id != m_active_id) {
		return;
	}
	// Pads with a real gyroscope report it directly.
	if (m_active_id != HOST_INPUT_CONTROLLER_ID) {
		auto* pad = SDL_GetGamepadFromID(static_cast<SDL_JoystickID>(m_active_id));
		if (pad != nullptr && SDL_GamepadHasSensor(pad, SDL_SENSOR_GYRO)) {
			return;
		}
	}

	m_gyro_emulation = !m_gyro_emulation;
	m_emulated_time  = 0;
	m_state.time     = LibKernel::KernelGetProcessTime();
	if (m_gyro_emulation) {
		m_emulated_stick = {m_state.axes[static_cast<int>(Axis::RightX)],
		                    m_state.axes[static_cast<int>(Axis::RightY)]};
		m_state.axes[static_cast<int>(Axis::RightX)] = 128;
		m_state.axes[static_cast<int>(Axis::RightY)] = 128;
		UpdateEmulatedTilt();
	} else {
		m_state.orientation = {0.0f, 0.0f, 0.0f, 1.0f};
		m_state.accel       = {0.0f, 1.0f, 0.0f};
		m_state.gyro        = {};
		m_state.axes[static_cast<int>(Axis::RightX)] = m_emulated_stick[0];
		m_state.axes[static_cast<int>(Axis::RightY)] = m_emulated_stick[1];
		AddState();
	}
	LOGF("Emulated gyro tilt: %s\n", m_gyro_emulation ? "enabled" : "disabled");
}

void GameController::ResetOrientation() {
	Common::LockGuard lock(m_mutex);
	m_state.orientation  = {0.0f, 0.0f, 0.0f, 1.0f};
	m_gyro_time          = 0;
	m_up_reference_valid = false;
	m_state.time        = LibKernel::KernelGetProcessTime();
	AddState();
}

void GameController::ResetInputState() {
	Common::LockGuard lock(m_mutex);
	m_state.buttons = 0;
	std::fill_n(m_state.axes, 4, 128);
	std::fill_n(m_state.axes + 4, 2, 0);
	for (auto& touch: m_state.touch) {
		touch = {};
	}
	m_state.time    = LibKernel::KernelGetProcessTime();
	m_states_num    = 0;
	m_first_state   = 0;
	m_next_touch_id = 1;
	AddState();
}

// SDL calls only: callers hold g_router_mutex but no GameController::m_mutex (lock order above).
static void StopHostPad(SDL_Gamepad* pad) {
	if (SDL_GetGamepadType(pad) == SDL_GAMEPAD_TYPE_PS5) {
		DualSenseEffects effect {};
		effect.enable_bits     = 0x0c;
		effect.right_trigger[0] = 0x05;
		effect.left_trigger[0]  = 0x05;
		(void)SDL_SendGamepadEffect(pad, &effect, sizeof(effect));
	}
	(void)SDL_RumbleGamepad(pad, 0, 0, 0);
	(void)SDL_SetGamepadLED(pad, 0, 0, 0);
}

// ASTROHYBRID_RUMBLE_SCALE=percent (default 100, 0..200): rumble strength for pads other than a
// DualSense. ASTROHYBRID_RUMBLE_SWITCH_FLOOR=n (default 0, 0..255): on Switch-type pads a motor that
// is on runs at least at n, for pads (third-party ones above all) that ignore weak rumble.
static uint16_t HostRumble(int id, uint8_t motor) {
	static const int scale = [] {
		const char* value = std::getenv("ASTROHYBRID_RUMBLE_SCALE");
		return value != nullptr ? std::clamp(std::atoi(value), 0, 200) : 100;
	}();
	static const int floor = [] {
		const char* value = std::getenv("ASTROHYBRID_RUMBLE_SWITCH_FLOOR");
		return value != nullptr ? std::clamp(std::atoi(value), 0, 255) : 0;
	}();
	if (motor == 0) {
		return 0;
	}
	int value = motor * scale / 100;
	if (floor > 0 && SDL_GetGamepadTypeForID(static_cast<SDL_JoystickID>(id)) >= SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_PRO &&
	    SDL_GetGamepadTypeForID(static_cast<SDL_JoystickID>(id)) <= SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_JOYCON_PAIR) {
		value = std::max(value, floor);
	}
	return static_cast<uint16_t>(std::clamp(value, 0, 255) * 0x101);
}

void GameController::SetVibration(uint8_t large_motor, uint8_t small_motor) {
	Common::LockGuard lock(m_mutex);
	m_vibration       = {large_motor, small_motor};
	m_vibration_until = SDL_GetTicks() + RUMBLE_DURATION_MS;
	ApplyVibration();
}

void GameController::ApplyVibration() {
	const auto now = SDL_GetTicks();
	const auto duration =
	    m_vibration_until > now ? static_cast<uint32_t>(m_vibration_until - now) : 0;
	const auto scale       = duration != 0 ? GetSettingScale(Setting::VibrationIntensity) : 0.0f;
	const auto large_motor = Scale(m_vibration[0], scale);
	const auto small_motor = Scale(m_vibration[1], scale);
	// The DualSense haptics backend drives a single pad, so only player 1 uses it.
	if (m_player == 0 &&
	    DualSenseHaptics::SetVibration(m_active_id, large_motor, small_motor, duration)) {
		return;
	}

	if (m_active_id == HOST_INPUT_CONTROLLER_ID) {
		return;
	}

	auto* pad = SDL_GetGamepadFromID(static_cast<SDL_JoystickID>(m_active_id));
	if (pad == nullptr) {
		return;
	}

	const auto large = HostRumble(m_active_id, large_motor);
	const auto small = HostRumble(m_active_id, small_motor);
	if (!SDL_RumbleGamepad(pad, large, small, duration)) {
		LOGF("\t rumble failed: %s\n", SDL_GetError());
	}
}

int GameController::GetActiveControllerId() {
	Common::LockGuard lock(m_mutex);
	return m_active_id;
}

void GameController::SetLightBar(uint8_t r, uint8_t g, uint8_t b) {
	Common::LockGuard lock(m_mutex);
	if (const auto& color = Config::GetControllerColor()) {
		r = (*color)[0];
		g = (*color)[1];
		b = (*color)[2];
	}
	if (auto* pad = SDL_GetGamepadFromID(static_cast<SDL_JoystickID>(m_active_id));
	    pad != nullptr) {
		(void)SDL_SetGamepadLED(pad, r, g, b);
	}
}

bool GameController::SetTriggerEffect(const PadTriggerEffectParam& param) {
	Common::LockGuard lock(m_mutex);
	// The state the game reads back follows every effect it sets, also one the DualSense output
	// cannot reproduce (values outside the ranges trigger_effect_to_dualsense converts).
	if ((param.trigger_mask & ~0x03u) == 0) {
		for (int i = 0; i < 2; i++) {
			if ((param.trigger_mask & (1u << i)) != 0) {
				m_state_effect.trigger_mask |= static_cast<uint8_t>(1u << i);
				m_state_effect.command[i] = param.command[i];
			}
		}
	}
	if (!SendTriggerEffect(param)) {
		static std::atomic<int> logged {0};
		if (logged.fetch_add(1) < 16) {
			for (int i = 0; i < 2; i++) {
				if ((param.trigger_mask & (1u << i)) != 0) {
					const auto& c = param.command[i];
					std::fprintf(stderr,
					             "Pad: trigger effect not sent to the controller: mask=0x%x trigger=%d mode=%u "
					             "data=%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u\n",
					             param.trigger_mask, i, c.mode, c.data[0], c.data[1], c.data[2], c.data[3],
					             c.data[4], c.data[5], c.data[6], c.data[7], c.data[8], c.data[9], c.data[10]);
				}
			}
		}
		return false;
	}
	for (int i = 0; i < 2; i++) {
		if ((param.trigger_mask & (1u << i)) != 0) {
			m_trigger_effect.trigger_mask |= static_cast<uint8_t>(1u << i);
			m_trigger_effect.command[i] = param.command[i];
		}
	}
	return true;
}

void GameController::GetTriggerEffectState(int32_t* state) {
	Common::LockGuard lock(m_mutex);
	const int axes[2] = {static_cast<int>(Controller::Axis::TriggerLeft), static_cast<int>(Controller::Axis::TriggerRight)};
	for (int i = 0; i < 2; i++) {
		state[i] = (m_state_effect.trigger_mask & (1u << i)) != 0
		               ? TriggerEffectState(m_state_effect.command[i],
		                                    TriggerTravel(m_state, static_cast<Controller::Axis>(axes[i])))
		               : 0;
	}
	// The first changes, for reports from games whose trigger actions still do not react.
	static int                    reports = 0;
	static std::array<int32_t, 2> last {-1, -1};
	if (reports < 32 && (state[0] != last[0] || state[1] != last[1])) {
		std::fprintf(stderr, "Pad: trigger state %d,%d (modes %u,%u, travel %d,%d)\n", state[0], state[1],
		             m_state_effect.command[0].mode, m_state_effect.command[1].mode,
		             TriggerTravel(m_state, Controller::Axis::TriggerLeft), TriggerTravel(m_state, Controller::Axis::TriggerRight));
		last = {state[0], state[1]};
		reports++;
	}
}

bool GameController::SendTriggerEffect(const PadTriggerEffectParam& param) {
	if ((param.trigger_mask & ~0x03u) != 0) {
		return false;
	}

	DualSenseEffects effect {};
	const auto       scale = GetSettingScale(Setting::TriggerEffectIntensity);
	if ((param.trigger_mask & 0x01u) != 0) {
		effect.enable_bits |= 0x08;
		if (!trigger_effect_to_dualsense(param.command[0], effect.left_trigger, scale)) {
			return false;
		}
	}
	if ((param.trigger_mask & 0x02u) != 0) {
		effect.enable_bits |= 0x04;
		if (!trigger_effect_to_dualsense(param.command[1], effect.right_trigger, scale)) {
			return false;
		}
	}
	if (effect.enable_bits == 0) {
		return true;
	}

	auto* pad = SDL_GetGamepadFromID(static_cast<SDL_JoystickID>(m_active_id));
	if (pad != nullptr && SDL_GetGamepadType(pad) == SDL_GAMEPAD_TYPE_PS5) {
		(void)SDL_SendGamepadEffect(pad, &effect, sizeof(effect));
	}
	return true;
}

void GameController::GetConnectionInfo(bool* flag, int* count) {
	EXIT_IF(flag == nullptr);
	EXIT_IF(count == nullptr);

	Common::LockGuard lock(m_mutex);

	*flag  = m_connected;
	*count = m_connected_count;
}

void GameController::ReadState(ControllerState* state, bool* flag, int* count) {
	EXIT_IF(flag == nullptr);
	EXIT_IF(count == nullptr);
	EXIT_IF(state == nullptr);

	Common::LockGuard lock(m_mutex);

	*flag  = m_connected;
	*count = m_connected_count;
	*state = m_state;
}

int GameController::ReadStates(ControllerState* states, int states_num, bool* flag, int* count) {
	EXIT_IF(flag == nullptr);
	EXIT_IF(count == nullptr);
	EXIT_IF(states == nullptr);
	EXIT_IF(states_num < 1 || states_num > STATES_MAX);

	Common::LockGuard lock(m_mutex);

	*flag  = m_connected;
	*count = m_connected_count;

	int ret_num = 0;

	if (m_connected) {
		if (m_states_num != 0) {
			for (uint32_t i = 0; i < m_states_num; i++) {
				if (ret_num >= states_num) {
					break;
				}
				auto index = (m_first_state + i) % STATES_MAX;
				if (!m_obtained[index]) {
					m_obtained[index] = true;

					states[ret_num++] = m_states[index];
				}
			}
		}
	}

	return ret_num;
}

// Keep every opened pad, including those waiting for a free player slot.
// Caller holds g_router_mutex.
static void RefreshPlayers() {
	std::array<bool, Config::MAX_PLAYERS> taken {};
	const int keyboard = KeyboardPlayer();
	if (g_keyboard_player != 0) {
		taken[keyboard] = true;
	}
	for (auto& [id, player]: g_pad_players) {
		if (g_keyboard_player != 0 && player == keyboard) {
			player = -1;
		}
		if (player >= 0) {
			taken[player] = true;
		}
	}
	for (auto& [id, player]: g_pad_players) {
		if (player >= 0) {
			continue;
		}
		const auto free = std::find(taken.begin(), taken.end(), false);
		if (free != taken.end()) {
			player = static_cast<int>(free - taken.begin());
			*free  = true;
			LOGF("Controller %d assigned to player %d\n", id, player + 1);
		}
	}
	std::array<int, Config::MAX_PLAYERS> active;
	active.fill(-1);
	active[keyboard] = HOST_INPUT_CONTROLLER_ID;
	for (const auto& [id, player]: g_pad_players) {
		if (player >= 0) {
			active[player] = id;
		}
	}
	// Stop old owners before configuring new owners of the same physical pad.
	for (int player = 0; player < Config::MAX_PLAYERS; player++) {
		const int previous = g_players[player]->GetActiveControllerId();
		if (previous != active[player]) {
			if (player == 0) {
				(void)DualSenseHaptics::SetVibration(previous, 0, 0, 0);
			}
			if (auto* pad = SDL_GetGamepadFromID(static_cast<SDL_JoystickID>(previous))) {
				StopHostPad(pad);
			}
		}
	}
	for (int player = 0; player < Config::MAX_PLAYERS; player++) {
		g_players[player]->SetDevices(active[player]);
	}
}

// ASTROHYBRID_PAD_PREFER (default unset: the first pad connected stays player 1, the only player
// the relay sends to the game): which pad plays. "auto": the pad last connected or pressed takes
// player 1; "ps", "xbox", "switch" or "generic": a pad of that family takes player 1 whenever one is
// connected, any other pad only while none is. Pads connected are logged to stderr, and a pad that
// is not a DualSense gives a short buzz when it takes player 1.
enum class PadFamily { PlayStation, Xbox, Switch, Generic };

static PadFamily FamilyOf(int id) {
	switch (SDL_GetGamepadTypeForID(static_cast<SDL_JoystickID>(id))) {
		case SDL_GAMEPAD_TYPE_PS3:
		case SDL_GAMEPAD_TYPE_PS4:
		case SDL_GAMEPAD_TYPE_PS5: return PadFamily::PlayStation;
		case SDL_GAMEPAD_TYPE_XBOX360:
		case SDL_GAMEPAD_TYPE_XBOXONE: return PadFamily::Xbox;
		case SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_PRO:
		case SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_JOYCON_LEFT:
		case SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_JOYCON_RIGHT:
		case SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_JOYCON_PAIR: return PadFamily::Switch;
		default: return PadFamily::Generic;
	}
}

// -1: off; 0: auto; 1 + PadFamily: that family.
static int PadPreference() {
	static const int mode = [] {
		const char* value = std::getenv("ASTROHYBRID_PAD_PREFER");
		if (value == nullptr || value[0] == 0) {
			return -1;
		}
		const std::string_view v(value);
		if (v == "ps" || v == "ps5") {
			return 1 + static_cast<int>(PadFamily::PlayStation);
		}
		if (v == "xbox") {
			return 1 + static_cast<int>(PadFamily::Xbox);
		}
		if (v == "switch") {
			return 1 + static_cast<int>(PadFamily::Switch);
		}
		if (v == "generic") {
			return 1 + static_cast<int>(PadFamily::Generic);
		}
		return 0;
	}();
	return mode;
}

static const char* FamilyName(PadFamily family) {
	switch (family) {
		case PadFamily::PlayStation: return "PlayStation";
		case PadFamily::Xbox: return "Xbox";
		case PadFamily::Switch: return "Switch";
		default: return "generic";
	}
}

static void LogPad(int id) {
	auto* pad = SDL_GetGamepadFromID(static_cast<SDL_JoystickID>(id));
	if (pad == nullptr) {
		return;
	}
	const auto props = SDL_GetGamepadProperties(pad);
	std::fprintf(stderr,
	             "[astrohybrid 0] controller %d: '%s' (%s, type %s, %04x:%04x), rumble %s, motion %s\n", id,
	             SDL_GetGamepadName(pad), FamilyName(FamilyOf(id)),
	             SDL_GetGamepadStringForType(SDL_GetGamepadType(pad)), SDL_GetGamepadVendor(pad),
	             SDL_GetGamepadProduct(pad),
	             SDL_GetBooleanProperty(props, SDL_PROP_GAMEPAD_CAP_RUMBLE_BOOLEAN, false) ? "yes" : "no",
	             SDL_GamepadHasSensor(pad, SDL_SENSOR_GYRO) ? "yes" : "no (Back/View toggles right-stick tilt)");
	std::fflush(stderr);
}

// Caller holds g_router_mutex.
static bool MakePlayerOne(int id) {
	if (g_keyboard_player == 1) {
		return false;
	}
	auto self = std::find_if(g_pad_players.begin(), g_pad_players.end(),
	                         [id](const auto& pad) { return pad.first == id; });
	if (self == g_pad_players.end() || self->second == 0) {
		return false;
	}
	auto owner = std::find_if(g_pad_players.begin(), g_pad_players.end(),
	                          [](const auto& pad) { return pad.second == 0; });
	if (owner != g_pad_players.end()) {
		owner->second = self->second;
	}
	self->second = 0;
	RefreshPlayers();
	std::fprintf(stderr, "[astrohybrid 0] controller %d is player 1 now\n", id);
	std::fflush(stderr);
	if (FamilyOf(id) != PadFamily::PlayStation) {
		if (auto* pad = SDL_GetGamepadFromID(static_cast<SDL_JoystickID>(id))) {
			(void)SDL_RumbleGamepad(pad, 0x6000, 0x6000, 150);
		}
	}
	return true;
}

// Caller holds g_router_mutex.
static bool PreferredPadConnected(PadFamily family) {
	return std::any_of(g_pad_players.begin(), g_pad_players.end(),
	                   [family](const auto& pad) { return FamilyOf(pad.first) == family; });
}

// Caller holds g_router_mutex. A pad was connected or pressed.
static void FollowPad(int id, bool connected) {
	const int mode = PadPreference();
	if (mode < 0 || id == HOST_INPUT_CONTROLLER_ID) {
		return;
	}
	if (mode == 0) {
		(void)MakePlayerOne(id);
		return;
	}
	const auto preferred = static_cast<PadFamily>(mode - 1);
	if (FamilyOf(id) == preferred || (!connected && !PreferredPadConnected(preferred))) {
		(void)MakePlayerOne(id);
	}
}

void PadPressed(int id) {
	if (PadPreference() < 0) {
		return;
	}
	Common::LockGuard lock(g_router_mutex);
	FollowPad(id, false);
}

void Connect(int id) {
	// KYTY_HOST_INPUT_ONLY=1: real pads are ignored; the host keyboard/mouse input drives player 1.
	if (HostInputOnly() && id != HOST_INPUT_CONTROLLER_ID) {
		return;
	}
	Common::LockGuard lock(g_router_mutex);
	if (std::any_of(g_pad_players.begin(), g_pad_players.end(),
	                [id](const auto& pad) { return pad.first == id; })) {
		return;
	}
	g_pad_players.emplace_back(id, -1);
	RefreshPlayers();
	if (PadPreference() >= 0) {
		LogPad(id);
		FollowPad(id, true);
	}
}

void Disconnect(int id) {
	Common::LockGuard lock(g_router_mutex);
	std::erase_if(g_pad_players, [id](const auto& pad) { return pad.first == id; });
	RefreshPlayers();
}

void CycleKeyboardPlayer() {
	Common::LockGuard lock(g_router_mutex);
	g_keyboard_player = (g_keyboard_player + 1) % (Config::MAX_PLAYERS + 1);
	RefreshPlayers();
	if (g_keyboard_player == 0) {
		LOGF("Keyboard shared with player 1\n");
	} else {
		LOGF("Keyboard assigned to player %u\n", g_keyboard_player);
	}
}

void SetButton(int id, uint32_t button, bool down) {
	Common::LockGuard lock(g_router_mutex);
	if (auto* player = PlayerForDevice(id)) {
		player->Button(id, button, down);
	}
}

void SetAxis(int id, Axis axis, int value) {
	Common::LockGuard lock(g_router_mutex);
	if (auto* player = PlayerForDevice(id)) {
		player->Axis(id, axis, value);
	}
}

void SetRightStick(int id, int x, int y) {
	Common::LockGuard lock(g_router_mutex);
	if (auto* player = PlayerForDevice(id)) {
		player->RightStick(id, x, y);
	}
}

void SetTouchPad(int id, int finger, bool down, float x, float y) {
	Common::LockGuard lock(g_router_mutex);
	if (auto* player = PlayerForDevice(id)) {
		player->TouchPad(id, finger, down, x, y);
	}
}

void SetSensor(int id, Sensor sensor, const float* data, uint64_t time_us) {
	Common::LockGuard lock(g_router_mutex);
	if (auto* player = PlayerForDevice(id)) {
		player->Motion(id, sensor, data, time_us);
	}
}

void ToggleGyroEmulation(int id) {
	Common::LockGuard lock(g_router_mutex);
	if (auto* player = PlayerForDevice(id)) {
		player->ToggleGyroEmulation(id);
	}
}

void ResetInputState() {
	for (const auto& player: g_players) {
		if (player != nullptr) {
			player->ResetInputState();
		}
	}
}

int GetActiveControllerId() {
	return g_players[0] != nullptr ? g_players[0]->GetActiveControllerId() : -1;
}

bool IsPlayerConnected(int player) {
	if (player < 0 || player >= Config::MAX_PLAYERS || g_players[player] == nullptr) {
		return false;
	}
	return g_players[player]->GetActiveControllerId() != -1;
}

bool GameController::PadOpenArgsAreValid(int user_id, int type, int index) const {
	// The host keyboard and gamepads of a player feed one standard port for that player's user.
	return user_id == Config::GetPlayerUserId(m_player) && type == 0 && index == 0;
}

int GameController::InitPort() {
	Common::LockGuard lock(m_mutex);
	if (m_port_state == PortState::Uninitialized) {
		m_port_state = PortState::Closed;
	}
	return OK;
}

int GameController::OpenPort(int user_id, int type, int index) {
	Common::LockGuard lock(m_mutex);
	const auto result = [&] {
		if (m_port_state == PortState::Uninitialized) {
			return PAD_ERROR_NOT_INITIALIZED;
		}
		if (!PadOpenArgsAreValid(user_id, type, index)) {
			return PAD_ERROR_INVALID_ARG;
		}
		if (m_port_state == PortState::Open) {
			return PAD_ERROR_ALREADY_OPENED;
		}
		m_port_state = PortState::Open;
		return Handle();
	}();
	// Games open their ports a few times at most; the first opens show which ports a title uses
	// (upstream b65dc1220 accepts only the standard port and rejects a second open).
	static std::atomic<uint32_t> logged {0};
	if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
		// stdout, flushed: shown whatever the printf direction (players' console logs).
		std::printf("Pad: open user %d type %d index %d -> %s (0x%08x)\n", user_id, type, index,
		            result == Handle() ? "handle" : "rejected", static_cast<uint32_t>(result));
		std::fflush(stdout);
	}
	return result;
}

int GameController::GetPortHandle(int user_id, int type, int index) {
	Common::LockGuard lock(m_mutex);
	if (m_port_state == PortState::Uninitialized) {
		return PAD_ERROR_NOT_INITIALIZED;
	}
	return m_port_state == PortState::Open && PadOpenArgsAreValid(user_id, type, index)
	           ? Handle()
	           : PAD_ERROR_NO_HANDLE;
}

int GameController::CheckPortHandleLocked(int handle) const {
	if (m_port_state == PortState::Uninitialized) {
		return PAD_ERROR_NOT_INITIALIZED;
	}
	return m_port_state == PortState::Open && handle == Handle() ? OK : PAD_ERROR_INVALID_HANDLE;
}

int GameController::CheckPortHandle(int handle) {
	Common::LockGuard lock(m_mutex);
	return CheckPortHandleLocked(handle);
}

int GameController::ClosePort(int handle) {
	Common::LockGuard lock(m_mutex);
	if (const auto error = CheckPortHandleLocked(handle); error != OK) {
		return error;
	}
	m_port_state = PortState::Closed;
	return OK;
}

int PadCheckHandle(int handle) {
	if (g_players[0] == nullptr) {
		return PAD_ERROR_NOT_INITIALIZED;
	}
	if (PadVrHand(handle) >= 0) {
		return PadVrCheckHandle(handle);
	}
	auto* player = PlayerForHandle(handle);
	return player != nullptr ? player->CheckPortHandle(handle) : PAD_ERROR_INVALID_HANDLE;
}

// Handle 0 is never valid, so this only reports whether scePadInit has run.
bool PadInitialized() {
	return g_players[0] != nullptr && g_players[0]->CheckPortHandle(0) != PAD_ERROR_NOT_INITIALIZED;
}

int KYTY_SYSV_ABI PadInit() {
	PRINT_NAME();
	for (const auto& player: g_players) {
		(void)player->InitPort();
	}
	return OK;
}

int KYTY_SYSV_ABI PadOpen(int user_id, int type, int index, const void* param) {
	PRINT_NAME();

	LOGF("\t user_id = %d\n"
	     "\t type    = %d\n"
	     "\t index   = %d\n"
	     "\t param   = 0x%016" PRIx64 "\n",
	     user_id, type, index, reinterpret_cast<uint64_t>(param));

	if (const int hand = PadVrPortHand(type); hand >= 0) {
		return PadVrPort(user_id, hand, index, true);
	}
	const int player = Config::GetPlayerFromUserId(user_id);
	return player >= 0 ? g_players[player]->OpenPort(user_id, type, index) : PAD_ERROR_INVALID_ARG;
}

int KYTY_SYSV_ABI PadGetHandle(int user_id, int type, int index) {
	PRINT_NAME();

	LOGF("\t user_id = %d\n"
	     "\t type    = %d\n"
	     "\t index   = %d\n",
	     user_id, type, index);

	if (const int hand = PadVrPortHand(type); hand >= 0) {
		return PadVrPort(user_id, hand, index, false);
	}
	const int player = Config::GetPlayerFromUserId(user_id);
	return player >= 0 ? g_players[player]->GetPortHandle(user_id, type, index)
	                   : PAD_ERROR_NO_HANDLE;
}

int KYTY_SYSV_ABI PadClose(int handle) {
	PRINT_NAME();
	LOGF("\t handle = %d\n", handle);
	if (PadVrHand(handle) >= 0) {
		return PadVrClose(handle);
	}
	auto* player = PlayerForHandle(handle);
	if (player == nullptr) {
		return PAD_ERROR_INVALID_HANDLE;
	}
	return player->ClosePort(handle);
}

int KYTY_SYSV_ABI PadSetMotionSensorState(int handle, bool enable) {
	PRINT_NAME();

	if (const auto error = PadCheckHandle(handle); error != OK) {
		return error;
	}

	LOGF("\t enable = %s\n", (enable ? "true" : "false"));
	PlayerForHandle(handle)->SetMotionSensorState(enable);

	return OK;
}

int KYTY_SYSV_ABI PadSetAngularVelocityDeadbandState(int handle, bool enable) {
	PRINT_NAME();

	if (const auto error = PadCheckHandle(handle); error != OK) {
		return error;
	}

	LOGF("\t enable = %s\n", (enable ? "true" : "false"));

	return OK;
}

int KYTY_SYSV_ABI PadResetOrientation(int handle) {
	PRINT_NAME();

	if (const auto error = PadCheckHandle(handle); error != OK) {
		return error;
	}

	PlayerForHandle(handle)->ResetOrientation();
	return OK;
}

int KYTY_SYSV_ABI PadGetControllerInformation(int handle, PadControllerInformation* info) {
	PRINT_NAME();

	if (const auto error = PadCheckHandle(handle); error != OK) {
		return error;
	}
	if (info == nullptr) {
		return PAD_ERROR_INVALID_ARG;
	}

	int  connected_count = 0;
	bool connected       = false;
	PlayerForHandle(handle)->GetConnectionInfo(&connected, &connected_count);

	std::memset(info, 0, sizeof(*info));

	info->touch_pixel_density   = 44.86f;
	info->touch_resolution_x    = 1920;
	info->touch_resolution_y    = 943;
	info->stick_dead_zone_left  = controller_get_axis(-32768, 32767, 8000) - 128;
	info->stick_dead_zone_right = controller_get_axis(-32768, 32767, 8000) - 128;
	info->connection_type       = 0;
	info->connected_count       = static_cast<uint8_t>(std::min(connected_count, 255));
	info->connected             = connected;
	info->device_class          = 0;

	return OK;
}

int KYTY_SYSV_ABI PadIsRemoteController(int handle, bool* is_remote) {
	PRINT_NAME();

	if (const auto error = PadCheckHandle(handle); error != OK) {
		return error;
	}
	if (is_remote == nullptr) {
		return PAD_ERROR_INVALID_ARG;
	}

	*is_remote = false;
	return OK;
}

int KYTY_SYSV_ABI PadReadState(int handle, PadData* data) {
	PRINT_NAME();

	if (const auto error = PadCheckHandle(handle); error != OK) {
		return error;
	}
	if (data == nullptr) {
		return PAD_ERROR_INVALID_ARG;
	}

	int             connected_count = 0;
	bool            connected       = false;
	ControllerState state;

	PlayerForHandle(handle)->ReadState(&state, &connected, &connected_count);

	pad_fill_data(data, state, connected, connected_count);

	return OK;
}

int KYTY_SYSV_ABI PadRead(int handle, PadData* data, int num) {
	PRINT_NAME();

	EXIT_NOT_IMPLEMENTED(num < 1 || num > 64);
	if (const auto error = PadCheckHandle(handle); error != OK) {
		return error;
	}
	if (data == nullptr) {
		return PAD_ERROR_INVALID_ARG;
	}

	std::memset(data, 0, sizeof(PadData) * static_cast<size_t>(num));

	int             connected_count = 0;
	bool            connected       = false;
	ControllerState states[64]      = {};

	auto* player  = PlayerForHandle(handle);
	int   ret_num = player->ReadStates(states, num, &connected, &connected_count);

	if (!connected || ret_num == 0) {
		if (connected) {
			player->ReadState(&states[0], &connected, &connected_count);
		}
		ret_num = 1;
	}

	for (int i = 0; i < ret_num; i++) {
		pad_fill_data(&data[i], states[i], connected, connected_count);
	}

	return ret_num;
}

int KYTY_SYSV_ABI PadSetVibration(int handle, const PadVibrationParam* param) {
	PRINT_NAME();
	Common::LockGuard lock(g_router_mutex);

	if (const auto error = PadCheckHandle(handle); error != OK) {
		return error;
	}
	if (param == nullptr) {
		return PAD_ERROR_INVALID_ARG;
	}

	LOGF("\t large_motor = %d\n"
	     "\t small_motor = %d\n",
	     static_cast<int>(param->large_motor), static_cast<int>(param->small_motor));

	PlayerForHandle(handle)->SetVibration(param->large_motor, param->small_motor);

	return OK;
}

int KYTY_SYSV_ABI PadResetLightBar(int handle) {
	PRINT_NAME();

	if (const auto error = PadCheckHandle(handle); error != OK) {
		return error;
	}

	return OK;
}

int KYTY_SYSV_ABI PadSetLightBar(int handle, const PadLightBarParam* param) {
	PRINT_NAME();
	Common::LockGuard lock(g_router_mutex);

	if (const auto error = PadCheckHandle(handle); error != OK) {
		return error;
	}
	if (param == nullptr) {
		return PAD_ERROR_INVALID_ARG;
	}

	PlayerForHandle(handle)->SetLightBar(param->r, param->g, param->b);

	return OK;
}

int KYTY_SYSV_ABI PadSetTriggerEffect(int handle, const PadTriggerEffectParam* param) {
	PRINT_NAME();
	Common::LockGuard lock(g_router_mutex);

	if (const auto error = PadCheckHandle(handle); error != OK) {
		return error;
	}
	if (param == nullptr) {
		return PAD_ERROR_INVALID_ARG;
	}

	return PlayerForHandle(handle)->SetTriggerEffect(*param) ? OK : PAD_ERROR_INVALID_ARG;
}

int KYTY_SYSV_ABI PadGetTriggerEffectState(int handle, PadTriggerEffectStateInformation* info) {
	PRINT_NAME();

	LOGF("\t handle = %d\n", handle);

	if (const auto error = PadCheckHandle(handle); error != OK) {
		return error;
	}
	if (info == nullptr) {
		return PAD_ERROR_INVALID_ARG;
	}

	PlayerForHandle(handle)->GetTriggerEffectState(info->state);

	return OK;
}

} // namespace Libs::Controller
