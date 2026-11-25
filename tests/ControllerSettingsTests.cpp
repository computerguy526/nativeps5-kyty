// Exercise the production controller with deterministic host output and time.
#include <SDL3/SDL.h>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <chrono>
#include <limits>
#include <mutex>
#include <thread>
#include <vector>
#include <cmath>
#include <string>

#include "hybrid/padMotion.h"

namespace {
void Check(bool condition, const char* text) {
	if (!condition) {
		std::fprintf(stderr, "ControllerSettingsTests: %s\n", text);
		std::abort();
	}
}

struct Rumble {
	Uint16 large, small;
	Uint32 duration;
};
Uint64                             now = 1000;
std::vector<Rumble>                rumble;
std::vector<Rumble>                haptics;
std::vector<std::array<Uint8, 32>> effects;
bool                               haptics_handles_rumble = false;
std::array<Rumble, 6>               pad_rumble {};
std::vector<int>                    closed_pads;
std::array<std::array<Uint8, 3>, 6> pad_colors {};
std::array<bool, 6>                 mono_led {};
// A fake pad's sensor side: what SDL reports (type, has sensors, "enabled" flags) and what the HID
// driver does (the IMU stream runs only after an enable; an output write can stop it).
struct FakePad {
	SDL_GamepadType type                = SDL_GAMEPAD_TYPE_PS5;
	bool            has_sensor          = false;
	bool            enabled[2]          = {false, false}; // accel, gyro (SDL's flag)
	bool            streaming           = false;          // the driver really sends IMU reports
	bool            output_clears_flag  = false;          // an LED / rumble write clears SDL's flag
	bool            output_stops_stream = false;          // ... or the IMU stream only
	int             enable_calls        = 0;
};
std::array<FakePad, 6> fake_pad {};
// SDL's joystick lock: every SDL gamepad call takes it (recursive in SDL). On only in TestLockOrder.
std::recursive_mutex sdl_joystick_lock;
std::atomic<bool>    sdl_lock_model {false};
struct SdlCall {
	SdlCall() {
		if (sdl_lock_model) {
			sdl_joystick_lock.lock();
			std::this_thread::sleep_for(std::chrono::microseconds(50)); // a HID write inside the lock
		}
	}
	~SdlCall() {
		if (sdl_lock_model) {
			sdl_joystick_lock.unlock();
		}
	}
	SdlCall(const SdlCall&)            = delete;
	SdlCall& operator=(const SdlCall&) = delete;
};
} // namespace

namespace Fake {
Uint64 GetTicks() {
	return now;
}
SDL_Gamepad* GetGamepadFromID(SDL_JoystickID id) {
	SdlCall sdl;
	return id >= 1 && id <= 5 ? reinterpret_cast<SDL_Gamepad*>(static_cast<uintptr_t>(id))
	                          : nullptr;
}
void PadOutput(SDL_Gamepad* pad) {
	auto& f = fake_pad[reinterpret_cast<uintptr_t>(pad)];
	if (f.output_clears_flag) {
		f.enabled[0] = f.enabled[1] = false;
		f.streaming                 = false;
	}
	if (f.output_stops_stream) {
		f.streaming = false;
	}
}
SDL_GamepadType GetGamepadType(SDL_Gamepad* pad) {
	return fake_pad[reinterpret_cast<uintptr_t>(pad)].type;
}
bool RumbleGamepad(SDL_Gamepad* pad, Uint16 large, Uint16 small, Uint32 duration) {
	rumble.push_back({large, small, duration});
	pad_rumble[reinterpret_cast<uintptr_t>(pad)] = rumble.back();
	PadOutput(pad);
	return true;
}
bool SendGamepadEffect(SDL_Gamepad*, const void* data, int size) {
	Check(size == 32, "unexpected DualSense effect size");
	std::array<Uint8, 32> effect {};
	std::memcpy(effect.data(), data, effect.size());
	effects.push_back(effect);
	return true;
}
bool SetGamepadLED(SDL_Gamepad* pad, Uint8 r, Uint8 g, Uint8 b) {
	SdlCall sdl;
	pad_colors[reinterpret_cast<uintptr_t>(pad)] = {r, g, b};
	PadOutput(pad);
	return true;
}
bool GamepadHasSensor(SDL_Gamepad* pad, SDL_SensorType) {
	SdlCall sdl;
	return fake_pad[reinterpret_cast<uintptr_t>(pad)].has_sensor;
}
bool GamepadSensorEnabled(SDL_Gamepad* pad, SDL_SensorType type) {
	return fake_pad[reinterpret_cast<uintptr_t>(pad)].enabled[type == SDL_SENSOR_GYRO ? 1 : 0];
}
bool SetGamepadSensorEnabled(SDL_Gamepad* pad, SDL_SensorType type, bool enabled) {
	auto&     f     = fake_pad[reinterpret_cast<uintptr_t>(pad)];
	const int index = type == SDL_SENSOR_GYRO ? 1 : 0;
	if (f.enabled[index] == enabled) {
		return true; // SDL returns early: the driver is not asked again
	}
	f.enabled[index] = enabled;
	f.enable_calls++;
	f.streaming = f.enabled[0] && f.enabled[1];
	return true;
}
void CloseGamepad(SDL_Gamepad* pad) {
	closed_pads.push_back(static_cast<int>(reinterpret_cast<uintptr_t>(pad)));
}
void Delay(Uint32) {}
SDL_PropertiesID GetGamepadProperties(SDL_Gamepad* pad) {
	return static_cast<SDL_PropertiesID>(reinterpret_cast<uintptr_t>(pad));
}
bool GetBooleanProperty(SDL_PropertiesID pad, const char* name, bool) {
	return std::strcmp(name, SDL_PROP_GAMEPAD_CAP_MONO_LED_BOOLEAN) == 0 && mono_led[pad];
}
} // namespace Fake

#define SDL_GetTicks                Fake::GetTicks
#define SDL_GetGamepadFromID        Fake::GetGamepadFromID
#define SDL_GetGamepadType          Fake::GetGamepadType
#define SDL_RumbleGamepad           Fake::RumbleGamepad
#define SDL_SendGamepadEffect       Fake::SendGamepadEffect
#define SDL_SetGamepadLED           Fake::SetGamepadLED
#define SDL_GamepadHasSensor        Fake::GamepadHasSensor
#define SDL_SetGamepadSensorEnabled Fake::SetGamepadSensorEnabled
#define SDL_GamepadSensorEnabled    Fake::GamepadSensorEnabled
#define SDL_CloseGamepad            Fake::CloseGamepad
#define SDL_Delay                   Fake::Delay
#define SDL_GetGamepadProperties    Fake::GetGamepadProperties
#define SDL_GetBooleanProperty      Fake::GetBooleanProperty
#include "libs/controller.cpp"
#undef SDL_GetTicks
#undef SDL_GetGamepadFromID
#undef SDL_GetGamepadType
#undef SDL_RumbleGamepad
#undef SDL_SendGamepadEffect
#undef SDL_SetGamepadLED
#undef SDL_GamepadHasSensor
#undef SDL_SetGamepadSensorEnabled
#undef SDL_GamepadSensorEnabled
#undef SDL_CloseGamepad
#undef SDL_Delay
#undef SDL_GetGamepadProperties
#undef SDL_GetBooleanProperty

namespace Libs::Controller {
// No virtual headset in these tests (padVr.cpp is not linked here).
int PadVrPortHand(int) { return -1; }
int PadVrPort(int, int, int, bool) { return PAD_ERROR_INVALID_ARG; }
int PadVrCheckHandle(int) { return PAD_ERROR_INVALID_HANDLE; }
int PadVrClose(int) { return PAD_ERROR_INVALID_HANDLE; }
} // namespace Libs::Controller

namespace Libs::Controller::DualSenseHaptics {
bool SetVibration(int, uint8_t large_motor, uint8_t small_motor, uint32_t duration_ms) {
	haptics.push_back({large_motor, small_motor, duration_ms});
	return haptics_handles_rumble;
}
void Shutdown() {}
} // namespace Libs::Controller::DualSenseHaptics

namespace Libs::LibKernel {
uint64_t KYTY_SYSV_ABI KernelGetProcessTime() {
	return now * 1000;
}
} // namespace Libs::LibKernel

namespace Loader::Timer {
double GetTimeMs() {
	return static_cast<double>(now);
}
} // namespace Loader::Timer

namespace {
using namespace Libs::Controller;

struct Controller {
	explicit Controller(SDL_GamepadType type = SDL_GAMEPAD_TYPE_PS5, bool sensors = false) {
		now                    = 1000;
		fake_pad               = {};
		fake_pad[1].type       = type;
		fake_pad[1].has_sensor = sensors;
		haptics_handles_rumble = false;
		Initialize();
		Check(PadInit() == 0 && PadOpen(Config::GetUserId(), 0, 0, nullptr) == 1,
		      "the standard controller port did not open");
		Connect(1);
		Check(GetSettingScale(Setting::SpeakerVolume) ==
		              Config::GetControllerSpeakerVolume() / 50.0f &&
		          GetSettingScale(Setting::VibrationIntensity) ==
		              Config::GetControllerVibrationIntensity() / 100.0f &&
		          GetSettingScale(Setting::TriggerEffectIntensity) == 1.0f,
		      "controller initialization retained old settings");
		rumble.clear();
		haptics.clear();
		effects.clear();
	}
	~Controller() { Shutdown(); }
};

DualSenseEffects LastEffect() {
	Check(!effects.empty(), "missing trigger output");
	DualSenseEffects effect {};
	std::memcpy(&effect, effects.back().data(), sizeof(effect));
	return effect;
}

int ZoneStrength(const uint8_t* effect, int zone) {
	const unsigned active = effect[1] | (effect[2] << 8u);
	const uint32_t packed = effect[3] | (effect[4] << 8u) | (effect[5] << 16u) |
	                        (static_cast<uint32_t>(effect[6]) << 24u);
	return (active & (1u << zone)) != 0 ? 1 + ((packed >> (3 * zone)) & 7u) : 0;
}

void SetRumble(uint8_t large, uint8_t small) {
	const PadVibrationParam param {large, small};
	Check(PadSetVibration(1, &param) == 0, "vibration request failed");
}

void TestPadPortOwnership() {
	Initialize();
	const int user = Config::GetUserId();
	constexpr auto not_initialized = static_cast<int32_t>(0x80920005u);
	constexpr auto already_opened = static_cast<int32_t>(0x80920004u);
	constexpr auto invalid_handle = static_cast<int32_t>(0x80920003u);
	constexpr auto no_handle = static_cast<int32_t>(0x80920008u);
	PadData data {};
	Check(PadOpen(user, 0, 0, nullptr) == not_initialized &&
	          PadGetHandle(user, 0, 0) == not_initialized &&
	          PadReadState(1, &data) == not_initialized,
	      "an uninitialized library exposed a controller port");
	Check(PadInit() == 0 && PadGetHandle(user, 0, 0) == no_handle,
	      "initializing the library opened a controller port");
	const int handle = PadOpen(user, 0, 0, nullptr);
	Check(handle == 1 && PadInit() == 0 && PadGetHandle(user, 0, 0) == handle,
	      "reinitializing the library lost the open port");
	for (int duplicate = 0; duplicate < 4; duplicate++) {
		const int rejected = PadOpen(user, 0, 0, nullptr);
		Check(rejected == already_opened,
		      "opening one pad five times exposed duplicate connected controllers");
		Check(PadClose(rejected) == invalid_handle && PadGetHandle(user, 0, 0) == handle,
		      "closing a rejected duplicate released the original owner");
	}
	Check(PadGetHandle(user + 1, 0, 0) == no_handle &&
	          PadGetHandle(user, 2, 0) == no_handle && PadGetHandle(user, 0, 1) == no_handle,
	      "a different port tuple aliased the open standard pad");
	SetButton(HOST_INPUT_CONTROLLER_ID, PAD_BUTTON_CROSS, true);
	Check(PadReadState(handle, &data) == 0 && data.connected &&
	          data.buttons == PAD_BUTTON_CROSS,
	      "keyboard Cross did not reach the owned port");
	SetButton(HOST_INPUT_CONTROLLER_ID, PAD_BUTTON_CROSS, false);
	Connect(1);
	SetButton(1, PAD_BUTTON_CROSS, true);
	Check(PadReadState(handle, &data) == 0 && data.buttons == PAD_BUTTON_CROSS,
	      "the physical controller did not share the owned port");
	Disconnect(1);
	Check(PadGetHandle(user, 0, 0) == handle && PadOpen(user, 0, 0, nullptr) == already_opened,
	      "host disconnection released guest port ownership");
	Check(PadClose(handle) == 0 && PadGetHandle(user, 0, 0) == no_handle &&
	          PadReadState(handle, &data) == invalid_handle &&
	          PadRead(handle, &data, 1) == invalid_handle && PadClose(handle) == invalid_handle,
	      "a closed port retained a usable handle");
	Check(PadOpen(user, 0, 0, nullptr) == handle && PadReadState(handle, &data) == 0,
	      "a closed port could not reopen");
	Shutdown();
}

void TestSpecialPortWithoutDevice() {
	Controller controller;
	const int user = Config::GetUserId();
	Check(PadOpenExt(user, 2, 0, nullptr) == PAD_ERROR_DEVICE_NOT_CONNECTED,
	      "a third-party controller port opened without a device");
	const int special = PadOpen(user, 2, 0, nullptr);
	Check(special > 1 && PadGetHandle(user, 2, 0) == special,
	      "the special port did not open its own handle");
	SetButton(1, PAD_BUTTON_CROSS, true);
	PadData data {};
	Check(PadReadState(special, &data) == OK && !data.connected && data.buttons == 0,
	      "host input reached the special port");
	Check(PadReadState(1, &data) == OK && data.connected && data.buttons == PAD_BUTTON_CROSS,
	      "the special port took the standard pad");
	Check(PadClose(special) == OK && PadGetHandle(user, 2, 0) == PAD_ERROR_NO_HANDLE &&
	          PadGetHandle(user, 0, 0) == 1,
	      "closing the special port changed the standard one");
}

void TestSettingCycles() {
	Controller controller;
	for (auto setting:
	     {Setting::SpeakerVolume, Setting::VibrationIntensity, Setting::TriggerEffectIntensity}) {
		Check(GetSettingScale(setting) == 1.0f, "initial setting is not strong");
		const std::vector<float> levels = setting == Setting::SpeakerVolume
		                                      ? std::vector<float> {0.0f, 0.02f, 0.12f, 0.42f, 1.0f}
		                                      : std::vector<float> {0.0f, 0.33f, 0.66f, 1.0f};
		for (int cycle = 0; cycle < 2; ++cycle) {
			for (float level: levels) {
				CycleSetting(setting);
				Check(GetSettingScale(setting) == level, "setting did not cycle or wrap");
			}
		}
	}
	CycleSetting(Setting::SpeakerVolume);
	CycleSetting(Setting::VibrationIntensity);
	CycleSetting(Setting::TriggerEffectIntensity);
}

void TestGlobalControllerLevels() {
	Config::ConfigOptions options;
	options.controller_speaker_volume      = 25;
	options.controller_vibration_intensity = 25;
	Config::Load(options);
	{
		Controller controller;
		Check(GetSettingScale(Setting::SpeakerVolume) == 0.5f &&
		          GetSettingScale(Setting::VibrationIntensity) == 0.25f &&
		          GetSettingScale(Setting::TriggerEffectIntensity) == 1.0f,
		      "global controller levels did not scale their outputs independently");
		SetRumble(200, 100);
		Check(rumble.back().large == 50 * 257 && rumble.back().small == 25 * 257,
		      "global vibration level did not scale rumble");
		CycleSetting(Setting::SpeakerVolume);
		CycleSetting(Setting::VibrationIntensity);
		Check(GetSettingScale(Setting::SpeakerVolume) == 0.0f &&
		          GetSettingScale(Setting::VibrationIntensity) == 0.0f,
		      "cycle hotkeys did not mute globally scaled outputs");
		CycleSetting(Setting::SpeakerVolume);
		CycleSetting(Setting::VibrationIntensity);
		Check(GetSettingScale(Setting::SpeakerVolume) == 0.01f &&
		          GetSettingScale(Setting::VibrationIntensity) == 0.0825f,
		      "cycle hotkeys did not multiply the global controller levels");
		Check(rumble.back().large == 17 * 257 && rumble.back().small == 8 * 257,
		      "cycling vibration did not apply the combined intensity to cached rumble");
	}
	options.controller_speaker_volume = 50;
	Config::Load(options);
	{
		Controller controller;
		Check(GetSettingScale(Setting::SpeakerVolume) == 1.0f,
		      "speaker midpoint did not preserve the previous default level");
	}
	options.controller_speaker_volume = 100;
	Config::Load(options);
	{
		Controller controller;
		Check(GetSettingScale(Setting::SpeakerVolume) == 2.0f,
		      "speaker maximum did not add 6 dB of gain");
		CycleSetting(Setting::SpeakerVolume);
		Check(GetSettingScale(Setting::SpeakerVolume) == 0.0f,
		      "speaker maximum prevented the hotkey from muting");
	}
	options.controller_speaker_volume      = 0;
	options.controller_vibration_intensity = 0;
	Config::Load(options);
	{
		Controller controller;
		Check(GetSettingScale(Setting::SpeakerVolume) == 0.0f &&
		          GetSettingScale(Setting::VibrationIntensity) == 0.0f,
		      "zero global controller levels did not mute outputs");
		SetRumble(200, 100);
		Check(rumble.back().large == 0 && rumble.back().small == 0,
		      "zero global vibration level did not mute motors");
	}
	Config::Load(Config::ConfigOptions {});
}

void TestVibrationLifetime() {
	Controller controller;
	SetRumble(255, 1);
	Check(rumble.size() == 1 && rumble.back().large == 65535 && rumble.back().small == 257 &&
	          rumble.back().duration == 65535,
	      "full vibration changed");
	now += 100;
	CycleSetting(Setting::VibrationIntensity);
	Check(rumble.back().large == 0 && rumble.back().small == 0, "muting did not stop vibration");
	now += 100;
	CycleSetting(Setting::VibrationIntensity);
	Check(rumble.back().large == 84 * 257 && rumble.back().small == 257 &&
	          rumble.back().duration == 65335,
	      "changing intensity lost a small motor or extended the vibration deadline");
	Check(effects.empty(), "vibration intensity resent triggers");
	now              = 1000 + 65535;
	const auto calls = rumble.size();
	CycleSetting(Setting::VibrationIntensity);
	CycleSetting(Setting::VibrationIntensity);
	Check(rumble.size() == calls || (rumble.back().large == 0 && rumble.back().small == 0),
	      "expired vibration was resurrected");
	SetRumble(0, 0);
	const auto stopped = rumble.size();
	CycleSetting(Setting::VibrationIntensity);
	CycleSetting(Setting::VibrationIntensity);
	Check(rumble.size() == stopped || (rumble.back().large == 0 && rumble.back().small == 0),
	      "explicitly stopped vibration was resurrected");
}

void TestMaskedTriggersAndValidation() {
	Controller            controller;
	PadTriggerEffectParam param {};
	param.trigger_mask       = 3;
	param.command[0].mode    = 1;
	param.command[0].data[0] = 2;
	param.command[0].data[1] = 8;
	param.command[1].mode    = 2;
	param.command[1].data[0] = 2;
	param.command[1].data[1] = 7;
	param.command[1].data[2] = 6;
	Check(PadSetTriggerEffect(1, &param) == 0, "initial trigger request failed");
	CycleSetting(Setting::TriggerEffectIntensity);
	Check(LastEffect().left_trigger[0] == 5 && LastEffect().right_trigger[0] == 5,
	      "muting did not disable both triggers");
	for (uint32_t mode = 1; mode <= 7; ++mode) {
		auto invalid            = param;
		invalid.command[1]      = {};
		invalid.command[1].mode = mode;
		std::memset(invalid.command[1].data, 255, sizeof(invalid.command[1].data));
		const auto calls = effects.size();
		Check(PadSetTriggerEffect(1, &invalid) == PAD_ERROR_INVALID_ARG && effects.size() == calls,
		      "muting bypassed trigger validation or sent a partial invalid request");
	}
	param.trigger_mask = 4;
	Check(PadSetTriggerEffect(1, &param) == PAD_ERROR_INVALID_ARG, "invalid trigger mask accepted");
	param.trigger_mask       = 1;
	param.command[0].data[1] = 6;
	Check(PadSetTriggerEffect(1, &param) == 0 && LastEffect().enable_bits == 8,
	      "left-only update touched the right trigger");
	CycleSetting(Setting::TriggerEffectIntensity);
	auto effect = LastEffect();
	Check(effect.enable_bits == 12 && effect.left_trigger[0] == 0x21 &&
	          ZoneStrength(effect.left_trigger, 1) == 0 &&
	          ZoneStrength(effect.left_trigger, 2) == 2 && effect.right_trigger[0] == 0x25 &&
	          effect.right_trigger[1] == 0x84 && effect.right_trigger[3] == 1,
	      "masked updates lost a cached trigger or failed to scale its strength");
	CycleSetting(Setting::TriggerEffectIntensity);
	CycleSetting(Setting::TriggerEffectIntensity);
	effect = LastEffect();
	Check(ZoneStrength(effect.left_trigger, 2) == 6 && effect.right_trigger[3] == 5,
	      "restoring strong intensity did not recover original trigger strengths");
	Check(rumble.empty() && haptics.empty(), "trigger intensity resent vibration");
}

void TestIndependentOutputsAndPadSwitch() {
	Controller controller;
	haptics_handles_rumble = true;
	SetRumble(200, 100);
	now += 100;
	CycleSetting(Setting::VibrationIntensity);
	CycleSetting(Setting::VibrationIntensity);
	Check(rumble.empty() && haptics.back().large == 66 && haptics.back().small == 33 &&
	          haptics.back().duration == 65435,
	      "haptics rumble did not receive scaled motors and remaining duration");
	PadTriggerEffectParam param {};
	param.trigger_mask       = 1;
	param.command[0].mode    = 1;
	param.command[0].data[1] = 8;
	Check(PadSetTriggerEffect(1, &param) == 0, "trigger request failed");
	const auto vibration_calls = haptics.size();
	const auto trigger_calls   = effects.size();
	CycleSetting(Setting::SpeakerVolume);
	Check(haptics.size() == vibration_calls && effects.size() == trigger_calls,
	      "speaker volume resent controller effects");
	Connect(2);
	Disconnect(1);
	Check(GetActiveControllerId() == HOST_INPUT_CONTROLLER_ID,
	      "disconnecting player 1 stole player 2's pad");
	Disconnect(2);
	Connect(2);
	Check(GetActiveControllerId() == 2, "active controller did not change");
	haptics.clear();
	effects.clear();
	CycleSetting(Setting::VibrationIntensity);
	CycleSetting(Setting::TriggerEffectIntensity);
	CycleSetting(Setting::TriggerEffectIntensity);
	Check(effects.empty(), "previous controller's trigger effects leaked to new pad");
	for (const auto& request: haptics) {
		Check(request.large == 0 && request.small == 0,
		      "previous controller's vibration leaked to new pad");
	}
	SetRumble(200, 100);
	Check(PadSetTriggerEffect(1, &param) == 0, "replacement pad trigger request failed");
	EmergencyShutdown();
	Check(GetActiveControllerId() == -1, "released controller remained active");
	haptics.clear();
	effects.clear();
	CycleSetting(Setting::VibrationIntensity);
	CycleSetting(Setting::TriggerEffectIntensity);
	Check(effects.empty(), "released controller received a trigger request");
	for (const auto& request: haptics) {
		Check(request.large == 0 && request.small == 0,
		      "released controller retained a cached vibration");
	}
}

void TestMonoLedPadGetsNoLightBar() {
	Controller             controller;
	const PadLightBarParam first {10, 20, 30, 0};
	const PadLightBarParam second {40, 50, 60, 0};
	Check(PadSetLightBar(1, &first) == 0 && pad_colors[1] == std::array<Uint8, 3> {10, 20, 30},
	      "a pad without an LED capability did not receive the light bar");
	mono_led[1] = true;
	Check(PadSetLightBar(1, &second) == 0 && pad_colors[1] == std::array<Uint8, 3> {10, 20, 30},
	      "a mono-LED pad received the light bar");
	mono_led[1] = false;
}

void TestPlayerUserIds() {
	Check(Config::IsConfiguredUserIdValid(Config::MAX_USER_ID) &&
	          !Config::IsConfiguredUserIdValid(Config::MAX_USER_ID + 1) &&
	          !Config::IsConfiguredUserIdValid(254) && !Config::IsConfiguredUserIdValid(255),
	      "configured user IDs did not reserve a valid range for guests");
	Config::ConfigOptions options;
	options.user_id = Config::MAX_USER_ID;
	Config::Load(options);
	for (int player = 0; player < Config::MAX_PLAYERS; player++) {
		Check(Config::GetPlayerFromUserId(Config::GetPlayerUserId(player)) == player,
		      "player user ID did not round-trip at the upper limit");
	}
	Check(Config::GetPlayerUserId(3) == std::numeric_limits<int32_t>::max() &&
	          Config::GetPlayerFromUserId(-1) == -1,
	      "guest user ID overflowed or accepted an unknown user");
	Config::Load(Config::ConfigOptions {});
}

void TestMultiplayerRouting() {
	Controller controller;
	for (int player = 1; player < Config::MAX_PLAYERS; player++) {
		Check(PadOpen(Config::GetPlayerUserId(player), 0, 0, nullptr) == player + 1,
		      "a guest could not open its own pad port");
	}
	for (int id = 2; id <= 5; id++) {
		Connect(id);
	}
	Connect(2); // Duplicate device reports must not consume another port.
	const auto read = [](int handle) {
		PadData data {};
		Check(PadReadState(handle, &data) == OK, "player pad read failed");
		return data;
	};
	SetButton(1, PAD_BUTTON_CROSS, true);
	SetButton(2, PAD_BUTTON_CIRCLE, true);
	SetButton(5, PAD_BUTTON_TRIANGLE, true);
	Check(read(1).buttons == PAD_BUTTON_CROSS && read(2).buttons == PAD_BUTTON_CIRCLE &&
	          read(3).buttons == 0 && read(4).buttons == 0,
	      "physical pads did not route independently or overflow reached a player");

	SetRumble(100, 50);
	CycleKeyboardPlayer(); // Dedicated player 1; physical pad 1 waits.
	Check(GetActiveControllerId() == HOST_INPUT_CONTROLLER_ID &&
	          read(2).buttons == PAD_BUTTON_CIRCLE && pad_rumble[1].large == 0 &&
	          pad_rumble[1].small == 0,
	      "keyboard reassignment changed an unaffected player or left pad output "
	      "running");
	for (int mode = 2; mode <= Config::MAX_PLAYERS; mode++) {
		CycleKeyboardPlayer();
	}
	CycleKeyboardPlayer(); // Shared player 1; all original pads must return.
	ResetInputState();
	for (int id = 1; id <= 5; id++) {
		SetButton(id, 1u << id, true);
	}
	for (int handle = 1; handle <= Config::MAX_PLAYERS; handle++) {
		Check(read(handle).buttons == (1u << handle),
		      "cycling keyboard players lost or duplicated a connected pad");
	}

	// Moving the keyboard clears its held input from the old player.
	SetButton(HOST_INPUT_CONTROLLER_ID, PAD_BUTTON_TRIANGLE, true);
	CycleKeyboardPlayer();
	CycleKeyboardPlayer();
	SetButton(HOST_INPUT_CONTROLLER_ID, PAD_BUTTON_TRIANGLE, true);
	CycleKeyboardPlayer(); // Keyboard now moves from player 2 to player 3.
	Check((read(2).buttons & PAD_BUTTON_TRIANGLE) == 0,
	      "moving the keyboard retained a held button on its old player");
	CycleKeyboardPlayer();
	CycleKeyboardPlayer();
	Disconnect(2);
	SetButton(5, PAD_BUTTON_SQUARE, true);
	Check(read(2).buttons == PAD_BUTTON_SQUARE && GetActiveControllerId() == 1,
	      "waiting pad did not fill a free slot or an unaffected pad moved");
	Disconnect(99);
	Check(PadCheckHandle(std::numeric_limits<int>::min()) == PAD_ERROR_INVALID_HANDLE &&
	          PadClose(std::numeric_limits<int>::max()) == PAD_ERROR_INVALID_HANDLE,
	      "an extreme handle aliased a player port");
	closed_pads.clear();
	EmergencyShutdown();
	std::sort(closed_pads.begin(), closed_pads.end());
	Check(closed_pads == std::vector<int> {1, 3, 4, 5},
	      "shutdown did not close each registered host pad exactly once");
}

void TestDedicatedKeyboardOutputs() {
	Config::ConfigOptions options;
	options.keyboard_player  = 1;
	options.controller_color = Config::ControllerColor {30, 40, 50};
	Config::Load(options);
	Initialize();
	PadInit();
	for (int player = 0; player < Config::MAX_PLAYERS; player++) {
		Check(PadOpen(Config::GetPlayerUserId(player), 0, 0, nullptr) == player + 1,
		      "dedicated keyboard port did not open");
	}
	Connect(1);
	Connect(2);
	const PadVibrationParam request {100, 50};
	Check(PadSetVibration(3, &request) == OK, "player three vibration failed");
	SetButton(2, PAD_BUTTON_CROSS, true);
	CycleKeyboardPlayer(); // Pad 1 moves from player 2 to player 1.
	PadData data {};
	Check(GetActiveControllerId() == 1 && pad_colors[1] == *options.controller_color,
	      "moving a pad to an earlier slot cleared its configured lightbar");
	Check(PadReadState(3, &data) == OK && data.buttons == PAD_BUTTON_CROSS &&
	          pad_rumble[2].large == 100 * 257 && pad_rumble[2].small == 50 * 257,
	      "keyboard reassignment reset an unaffected player's input or output");
	haptics_handles_rumble = true;
	SetRumble(200, 100);
	for (int mode = 0; mode < 4; mode++) {
		CycleKeyboardPlayer();
	}
	Check(GetActiveControllerId() == HOST_INPUT_CONTROLLER_ID && haptics.back().large == 0 &&
	          haptics.back().small == 0 && haptics.back().duration == 0,
	      "moving player one's pad retained its DualSense vibration request");
	Shutdown();
	Config::Load(Config::ConfigOptions {});
}

// scePadGetTriggerEffectState: the state of each trigger's effect at the trigger's travel.
void TestTriggerEffectState() {
	Controller controller;
	int32_t    state[2] = {-1, -1};
	GetTriggerEffectState(state);
	Check(state[0] == 0 && state[1] == 0, "a trigger without an effect reported a state");
	// The ABI call checks its handle before its output, with libScePad's codes (upstream #1132).
	PadTriggerEffectStateInformation info {{-1, -1}};
	Check(PadGetTriggerEffectState(2, &info) == PAD_ERROR_INVALID_HANDLE && info.state[0] == -1 &&
	          info.state[1] == -1,
	      "invalid trigger handle changed the output");
	Check(PadGetTriggerEffectState(2, nullptr) == PAD_ERROR_INVALID_HANDLE,
	      "trigger state checked the output before the handle");
	Check(PadGetTriggerEffectState(1, nullptr) == PAD_ERROR_INVALID_ARG,
	      "trigger state accepted a null output");
	Check(PadGetTriggerEffectState(1, &info) == OK && info.state[0] == 0 && info.state[1] == 0,
	      "unset trigger effects did not report off");

	PadTriggerEffectParam param {};
	param.trigger_mask       = 3;
	param.command[0].mode    = 1; // feedback from position 4
	param.command[0].data[0] = 4;
	param.command[0].data[1] = 8;
	param.command[1].mode    = 2; // weapon: resistance from 2, released at 7
	param.command[1].data[0] = 2;
	param.command[1].data[1] = 7;
	param.command[1].data[2] = 6;
	Check(PadSetTriggerEffect(1, &param) == 0, "trigger request failed");

	const auto states_at = [&](int left, int right) {
		SetAxis(1, Axis::TriggerLeft, left);
		SetAxis(1, Axis::TriggerRight, right);
		GetTriggerEffectState(state);
	};
	states_at(0, 0);
	Check(state[0] == 1 && state[1] == 3, "released triggers: feedback no force, weapon not pressed");
	states_at(102, 52); // positions 3 and 2
	Check(state[0] == 1 && state[1] == 4, "before the feedback zone, inside the weapon's resistance");
	states_at(103, 178); // positions 4 and 6
	Check(state[0] == 2 && state[1] == 4, "feedback pushing, weapon not yet released");
	states_at(255, 180); // positions 9 and 7
	Check(state[0] == 2 && state[1] == 5, "full feedback travel, weapon released");

	// Vibration, and the keyboard's digital full press.
	param.trigger_mask       = 1;
	param.command[0]         = {};
	param.command[0].mode    = 3;
	param.command[0].data[0] = 5;
	param.command[0].data[1] = 4;
	param.command[0].data[2] = 20;
	Check(PadSetTriggerEffect(1, &param) == 0, "vibration request failed");
	states_at(0, 255);
	Check(state[0] == 6 && state[1] == 5, "vibration not firing; right trigger kept its weapon");
	states_at(255, 0);
	Check(state[0] == 7 && state[1] == 3, "vibration firing at full travel");

	// Feedback from position 0 pushes only while the trigger is pressed (Astro's Playroom waits for it).
	param.trigger_mask       = 2;
	param.command[1]         = {};
	param.command[1].mode    = 1;
	param.command[1].data[0] = 0;
	param.command[1].data[1] = 5;
	Check(PadSetTriggerEffect(1, &param) == 0, "feedback request failed");
	states_at(0, 0);
	Check(state[1] == 1, "an untouched trigger with feedback from position 0 reported pushing");
	states_at(0, 40);
	Check(state[1] == 2, "a pressed trigger with feedback from position 0 reported no force");
	// Vibration from position 0 fires only while the trigger is pressed (Astro's Playroom's gun shot by itself).
	param.command[1]         = {};
	param.command[1].mode    = 3;
	param.command[1].data[0] = 0;
	param.command[1].data[1] = 4;
	param.command[1].data[2] = 20;
	Check(PadSetTriggerEffect(1, &param) == 0, "vibration request failed");
	states_at(255, 0);
	Check(state[1] == 6, "an untouched trigger with vibration from position 0 reported firing");
	states_at(255, 40);
	Check(state[1] == 7, "a pressed trigger with vibration from position 0 reported not firing");
	states_at(255, 0);

	// The intensity setting changes what the pad feels, not what the game reads.
	CycleSetting(Setting::TriggerEffectIntensity);
	GetTriggerEffectState(state);
	Check(state[0] == 7, "muting trigger effects changed the reported state");
}

// KYTY_PAD_SCRIPT entries: buttons, l2/r2, held sticks and hold times.
void TestPadScriptEntries() {
	const auto parse = [](const char* text, PadScriptEvent* event) {
		*event = {};
		return PadScriptParseEntry(text, std::strlen(text), event);
	};
	PadScriptEvent e;
	Check(parse("20:cross", &e) && e.button == PAD_BUTTON_CROSS && e.at == 20.0 && !e.by_flip &&
	          e.hold_us == 300000 && e.stick == 0,
	      "a plain press did not parse with the default hold");
	Check(parse("f1200:r2:2000", &e) && e.button == PAD_BUTTON_R2 && e.by_flip && e.at == 1200.0 &&
	          e.hold_us == 2000000,
	      "a timed R2 hold did not parse");
	Check(parse("40.5:lstick=128/0:2500", &e) && e.stick == 1 && e.stick_x == 128 && e.stick_y == 0 &&
	          e.button == 0 && e.hold_us == 2500000 && e.at == 40.5,
	      "a held left stick did not parse");
	Check(parse("1:rstick=255/128", &e) && e.stick == 2 && e.stick_x == 255 && e.hold_us == 300000,
	      "a right stick push did not parse");
	Check(!parse("1:lstick=300/0", &e) && !parse("1:lstick=12", &e) && !parse("1:cross:0", &e) &&
	          !parse("1:cross:x", &e) && !parse("cross", &e) && !parse("1:l3", &e),
	      "a malformed entry parsed");
	PadScriptEvent action;
	Check(PadScriptParseAction("lstick=0/128:900", 16, &action) && action.stick == 1 && action.stick_x == 0 &&
	          action.hold_us == 900000,
	      "a trigger-file stick action did not parse");

	// Holds: the stick entry fired last wins while both hold; l2/r2 count as buttons.
	g_pad_script.clear();
	g_pad_start_us = 0;
	PadScriptEvent up;
	parse("1:lstick=128/0:3000", &up);
	PadScriptEvent left;
	parse("2:lstick=0/128:500", &left);
	PadScriptEvent r2;
	parse("1.5:r2:1000", &r2);
	g_pad_script = {up, left, r2};
	auto held = PadScriptHeldAt(500000, 0);
	Check(!held.stick[0] && held.buttons == 0, "an entry held before its time");
	held = PadScriptHeldAt(1600000, 0);
	Check(held.stick[0] && held.y[0] == 0 && held.buttons == PAD_BUTTON_R2, "the stick and R2 holds were missing");
	held = PadScriptHeldAt(2100000, 0);
	Check(held.stick[0] && held.x[0] == 0 && held.y[0] == 128, "the later stick entry did not win");
	held = PadScriptHeldAt(2700000, 0);
	Check(held.stick[0] && held.y[0] == 0 && held.buttons == 0, "expired holds stayed or the earlier stick was lost");
	// Holds count from the poll that fired them (1.6 s here).
	held = PadScriptHeldAt(4500000, 0);
	Check(held.stick[0], "a stick hold ended early");
	held = PadScriptHeldAt(4700000, 0);
	Check(!held.stick[0], "a stick hold outlived its time");
	g_pad_script.clear();
}

void TestUnreadPressSurvivesStickBurst() {
	// Our coalescing is behind KYTY_PAD_STATE_COALESCE=1 (default off; upstream 39faa62eb).
	const char* coalesce = std::getenv("KYTY_PAD_STATE_COALESCE");
	if (coalesce == nullptr || std::strcmp(coalesce, "1") != 0) {
		return;
	}

	Controller              controller;
	std::array<PadData, 64> data {};
	SetButton(1, PAD_BUTTON_CROSS, true);
	for (int value = 0; value < 100; value++) {
		SetAxis(1, Axis::LeftX, value);
	}
	SetButton(1, PAD_BUTTON_CROSS, false);
	for (int value = 100; value < 200; value++) {
		SetAxis(1, Axis::LeftX, value);
	}
	Check(PadRead(1, data.data(), 64) == 2 && data[0].buttons == PAD_BUTTON_CROSS &&
	          data[0].left_stick_x == 99 && data[1].buttons == 0 && data[1].left_stick_x == 199,
	      "a stick burst pushed an unread press out of the pad history");
	SetAxis(1, Axis::LeftX, 50);
	SetButton(1, PAD_BUTTON_CROSS, true);
	Check(PadRead(1, data.data(), 64) == 2 && data[0].buttons == 0 &&
	          data[0].left_stick_x == 50 && data[1].buttons == PAD_BUTTON_CROSS,
	      "a stick move after a read did not add a sample");
}

void TestTouchWithoutClick() {
	Controller controller;
	PadData    data {};
	SetTouchPad(HOST_INPUT_CONTROLLER_ID, 1, true, 0.2f, 0.5f, false);
	Check(PadReadState(1, &data) == 0 && data.touch_data.touch_num == 1 && data.buttons == 0,
	      "a touch without a click pressed the touch pad button");
	SetTouchPad(HOST_INPUT_CONTROLLER_ID, 0, true, 0.25f, 0.5f);
	Check(PadReadState(1, &data) == 0 && data.buttons == PAD_BUTTON_TOUCH_PAD,
	      "a host input touch did not press the touch pad button");
}


// ---- motion: synthetic SDL sensor events -> GameController::Motion -> pad_fill_data -> PadData ----
namespace PM = Hybrid::PadMotion;

// One SDL sensor update pair (accel m/s^2, gyro rad/s), as the window thread forwards it:
// Controller::SetSensor(id, ...) with the sensor timestamp in microseconds.
void SdlSensorPair(int id, const float accel[3], const float gyro[3], uint64_t time_us) {
	SetSensor(id, Sensor::Accel, accel, time_us);
	SetSensor(id, Sensor::Gyro, gyro, time_us);
}

// What the fake HID driver delivers: samples every 5 ms (200 Hz) while the IMU stream runs.
void RunPad(int id, int ms, const float accel[3], const float gyro[3], uint64_t* time_us) {
	for (int i = 0; i < ms; i += 5) {
		now += 5;
		*time_us += 5000;
		if (fake_pad[id].streaming) {
			SdlSensorPair(id, accel, gyro, *time_us);
		}
	}
}

struct MotionModel {
	const char*     name;
	SDL_GamepadType type;
	float           bias[3]; // gyro bias at rest
};

void TestMotionPath() {
	const MotionModel models[] = {
	    {"switch-pro", SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_PRO, {-0.002f, 0.0f, 0.017f}},
	    {"dualsense", SDL_GAMEPAD_TYPE_PS5, {0.001f, -0.003f, 0.002f}},
	    {"dualshock4", SDL_GAMEPAD_TYPE_PS4, {0.004f, 0.002f, -0.006f}},
	};
	for (const auto& model: models) {
		Controller controller(model.type, true);
		Check(fake_pad[1].enabled[0] && fake_pad[1].enabled[1] && fake_pad[1].streaming,
		      "connecting a pad with sensors did not enable accel and gyro");
		MotionStatus status {};
		Check(PadGetMotionStatus(1, &status) && !status.live && status.accel_events == 0,
		      "a pad that delivered nothing was live");
		// Before any event the pad data is Kyty's neutral default pose: it must never count as live.
		PadData data {};
		Check(PadReadState(1, &data) == 0 && data.acceleration_x == 0.0f &&
		          data.acceleration_y == 1.0f && data.angular_velocity_x == 0.0f,
		      "the default pose changed");
		Check(PM::PlanRelay(true, status.live).motion_flagged, "neutral default is not flagged");

		uint64_t    t       = 1000000;
		const float rest[3] = {0.05f, SDL_STANDARD_GRAVITY, -0.03f};
		RunPad(1, 100, rest, model.bias, &t);
		Check(PadGetMotionStatus(1, &status) && status.live && status.accel_events >= 15 &&
		          status.gyro_events >= 15,
		      model.name);
		Check(PadReadState(1, &data) == 0, "read failed");
		Check(std::fabs(data.acceleration_y - 1.0f) < 0.01f &&
		          std::fabs(data.acceleration_x - 0.05f / SDL_STANDARD_GRAVITY) < 0.01f &&
		          std::fabs(data.angular_velocity_z - model.bias[2]) < 1e-6f &&
		          data.angular_velocity_x == model.bias[0],
		      "resting accel / gyro did not reach PadData");
		PM::Sample  rest_sample;
		const float accel_g[3] = {data.acceleration_x, data.acceleration_y, data.acceleration_z};
		const float gyro_v[3]  = {data.angular_velocity_x, data.angular_velocity_y,
		                          data.angular_velocity_z};
		Check(PM::FromPadData(accel_g, gyro_v, t, &rest_sample) && !PM::IsNeutralDefault(rest_sample),
		      "a real resting reading looked like the neutral default");

		// Shake: a 3 g spike and a 10 rad/s swing; the orientation must move off identity.
		const float spike[3] = {0.0f, SDL_STANDARD_GRAVITY, 3.0f * SDL_STANDARD_GRAVITY};
		const float swing[3] = {10.0f, 0.0f, 0.0f};
		RunPad(1, 50, spike, swing, &t);
		Check(PadReadState(1, &data) == 0 && std::fabs(data.acceleration_z - 3.0f) < 0.01f &&
		          std::fabs(data.angular_velocity_x - 10.0f) < 1e-4f &&
		          std::fabs(data.orientation_x) > 0.05f,
		      "a shake did not reach accel / gyro / orientation");
		const float norm = std::sqrt(data.orientation_x * data.orientation_x +
		                             data.orientation_y * data.orientation_y +
		                             data.orientation_z * data.orientation_z +
		                             data.orientation_w * data.orientation_w);
		Check(std::fabs(norm - 1.0f) < 1e-3f, "orientation is not a unit quaternion");

		// Steady tilt: 30 degrees, no rotation: accel follows, gyro stays at the bias.
		const float tilt[3] = {0.5f * SDL_STANDARD_GRAVITY, 0.866f * SDL_STANDARD_GRAVITY, 0.0f};
		RunPad(1, 200, tilt, model.bias, &t);
		Check(PadReadState(1, &data) == 0 && std::fabs(data.acceleration_x - 0.5f) < 0.01f &&
		          std::fabs(data.acceleration_y - 0.866f) < 0.01f &&
		          data.angular_velocity_x == model.bias[0],
		      "a steady tilt did not reach accel");
		Check(PadGetMotionStatus(1, &status) && status.live, "tilting pad not live");

		// The pad goes silent: after 250 ms it is not live any more.
		fake_pad[1].streaming = false;
		RunPad(1, 400, tilt, model.bias, &t);
		Check(PadGetMotionStatus(1, &status) && !status.live, "a silent pad stayed live");
	}
}

void TestSensorsSurviveOutputsAndReopen() {
	// An LED / rumble write that clears SDL's flag: re-asserted right after the write.
	{
		Controller controller(SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_PRO, true);
		fake_pad[1].output_clears_flag = true;
		const PadLightBarParam color {10, 20, 30, 0};
		Check(PadSetLightBar(1, &color) == 0, "light bar failed");
		Check(fake_pad[1].enabled[0] && fake_pad[1].enabled[1] && fake_pad[1].streaming,
		      "an LED write left the sensors off");
		fake_pad[1].streaming = false;
		SetRumble(200, 200);
		Check(fake_pad[1].enabled[0] && fake_pad[1].enabled[1] && fake_pad[1].streaming,
		      "a rumble write left the sensors off");
	}
	// An output write that stops the IMU stream behind SDL's back (flag stays on): no events; the
	// stale check re-arms once, then events flow again; no re-arm storm.
	{
		Controller  controller(SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_PRO, true);
		uint64_t    t       = 1000000;
		const float rest[3] = {0.0f, SDL_STANDARD_GRAVITY, 0.0f};
		const float bias[3] = {0.001f, 0.0f, 0.002f};
		RunPad(1, 100, rest, bias, &t);
		fake_pad[1].output_stops_stream = true;
		SetRumble(200, 200);
		fake_pad[1].output_stops_stream = false;
		Check(!fake_pad[1].streaming && fake_pad[1].enabled[0], "model: the stream should be stopped");
		MotionStatus status {};
		RunPad(1, 1000, rest, bias, &t);
		KeepHostPadSensors();
		Check(!fake_pad[1].streaming, "re-armed before the stale time");
		RunPad(1, 1000, rest, bias, &t);
		KeepHostPadSensors();
		Check(fake_pad[1].streaming && fake_pad[1].enabled[0] && fake_pad[1].enabled[1],
		      "a stalled IMU stream was not re-armed");
		Check(PadGetMotionStatus(1, &status) && status.rearms == 1, "re-arm not counted once");
		const int calls = fake_pad[1].enable_calls;
		RunPad(1, 100, rest, bias, &t);
		KeepHostPadSensors();
		Check(fake_pad[1].enable_calls == calls, "a live pad was re-armed");
		Check(PadGetMotionStatus(1, &status) && status.live, "events did not resume after the re-arm");
		fake_pad[1].streaming = false; // stalls again at once: rate limit (3 s)
		RunPad(1, 2000, rest, bias, &t);
		KeepHostPadSensors();
		Check(PadGetMotionStatus(1, &status) && status.rearms == 1,
		      "re-arm storm: second re-arm within 3 s");
	}
	// A pad that never sends an IMU stream: re-arms back off and stop at the cap.
	{
		Controller  controller(SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_PRO, true);
		uint64_t    t       = 1000000;
		const float rest[3] = {0.0f, SDL_STANDARD_GRAVITY, 0.0f};
		fake_pad[1].output_stops_stream = true;
		SetRumble(200, 200); // the stream never runs, a re-arm toggle does not help either
		fake_pad[1].output_stops_stream = false;
		int calls_after_cap = 0;
		for (int second = 0; second < 600; second++) {
			fake_pad[1].streaming = false;
			RunPad(1, 1000, rest, rest, &t);
			KeepHostPadSensors();
			if (second == 300) {
				calls_after_cap = fake_pad[1].enable_calls;
			}
		}
		MotionStatus status {};
		Check(PadGetMotionStatus(1, &status) && status.rearms == 6, "re-arms did not stop at the cap");
		Check(fake_pad[1].enable_calls == calls_after_cap, "a pad was toggled after the re-arm cap");
	}
	// Re-open: the pad leaves and returns (a fresh SDL gamepad: flags off, no stream).
	{
		Controller controller(SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_PRO, true);
		Disconnect(1);
		fake_pad[1].enabled[0] = fake_pad[1].enabled[1] = fake_pad[1].streaming = false;
		Connect(1);
		Check(fake_pad[1].enabled[0] && fake_pad[1].enabled[1] && fake_pad[1].streaming,
		      "a re-opened pad did not get its sensors enabled");
		uint64_t    t       = 1000000;
		const float rest[3] = {0.0f, SDL_STANDARD_GRAVITY, 0.1f};
		const float bias[3] = {0.001f, 0.0f, 0.002f};
		RunPad(1, 50, rest, bias, &t);
		MotionStatus status {};
		PadData      data {};
		Check(PadGetMotionStatus(1, &status) && status.live && PadReadState(1, &data) == 0 &&
		          data.angular_velocity_z == 0.002f,
		      "a re-opened pad's readings did not arrive");
	}
}

void TestNoNeutralPublishWhileSensorsClaimed() {
	{
		Controller   controller(SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_PRO, true);
		MotionStatus status {};
		PadData      data {};
		// Sensors are claimed, SDL delivers nothing: the default pose is flagged, not live.
		Check(PadGetMotionStatus(1, &status) && !status.live, "not live expected");
		Check(PadReadState(1, &data) == 0 && data.acceleration_y == 1.0f &&
		          data.angular_velocity_x == 0.0f && data.angular_velocity_y == 0.0f &&
		          data.angular_velocity_z == 0.0f,
		      "default pose expected");
		Check(PM::PlanRelay(true, status.live).motion_flagged,
		      "neutral default not flagged while sensors are claimed");
		Check(!PM::PlanRelay(false, status.live).motion_flagged, "emulated tilt flagged");
		// Buttons, sticks, triggers and the connection are published whatever motion does.
		const auto plan = PM::PlanRelay(true, status.live);
		Check(plan.publish_state && plan.publish_motion, "the relay holds back state");
		SetButton(1, PAD_BUTTON_CROSS, true);
		SetAxis(1, Axis::LeftX, 200);
		Check(PadReadState(1, &data) == 0 && data.connected && data.buttons == PAD_BUTTON_CROSS &&
		          data.left_stick_x == 200,
		      "buttons / sticks not available while motion is not live");
		Check(PadGetMotionStatus(1, &status) && !status.live, "still not live");
		// Events for a pad no player owns are counted, not applied.
		const float accel[3] = {1.0f, 2.0f, 3.0f};
		const float gyro[3]  = {1.0f, 1.0f, 1.0f};
		SdlSensorPair(77, accel, gyro, 5000);
		Check(PadGetMotionStatus(1, &status) && status.events_unrouted == 2 && !status.live &&
		          status.accel_events == 0,
		      "an unrouted sensor event was applied or not counted");
	}
	// Emulated tilt (a pad without sensors): live by definition.
	{
		Controller   controller;
		MotionStatus status {};
		ToggleGyroEmulation(1);
		Check(PadGetMotionStatus(1, &status) && status.live, "emulated tilt not live");
	}
	const std::string line = PM::FormatStatusLine(true, false, 0, 0, 5, 2, 0, true);
	Check(line.find("live=NO") != std::string::npos && line.find("NEUTRAL-DEFAULT") != std::string::npos,
	      "the status line does not flag the dead sensors");
}

} // namespace

// The pad relay thread reads the active controller while SDL's joystick lock is held (kytyGpuBridge.cpp pad_sensors
// before the fix took them in that order) while the main thread connects pads (Connect -> SetDevices -> SDL gamepad
// calls). SetDevices must not hold the controller mutex across SDL calls: with the old SetDevices this deadlocks
// within milliseconds (main: m_mutex -> SDL lock, relay: SDL lock -> m_mutex) and the watchdog fails the test.
void TestLockOrder() {
	Controller          controller;
	std::atomic<bool>   stop {false};
	std::atomic<bool>   done {false};
	std::atomic<uint64_t> relay_reads {0};
	std::thread watchdog([&] {
		for (int i = 0; i < 1000 && !done; i++) {
			std::this_thread::sleep_for(std::chrono::milliseconds(10));
		}
		Check(done, "lock-order deadlock: main thread (controller mutex -> SDL joystick lock) vs relay thread "
		            "(SDL joystick lock -> controller mutex)");
	});
	sdl_lock_model = true;
	std::thread relay([&] {
		while (!stop) {
			std::lock_guard<std::recursive_mutex> sdl(sdl_joystick_lock); // SDL_LockJoysticks()
			(void)GetActiveControllerId();                                // the old pad_sensors order
			relay_reads++;
		}
	});
	for (int i = 0; i < 400; i++) {
		Connect(2);
		Disconnect(1);
		Connect(1);
		Disconnect(2);
	}
	stop = true;
	relay.join();
	sdl_lock_model = false;
	done = true;
	watchdog.join();
	Check(relay_reads > 0, "the relay thread never ran");
}

int main() {
	Config::Initialize();
	TestPadScriptEntries();
	TestPlayerUserIds();
	TestPadPortOwnership();
	TestSpecialPortWithoutDevice();
	TestSettingCycles();
	TestGlobalControllerLevels();
	// Each following test initializes another controller and requires strong defaults.
	TestVibrationLifetime();
	TestMaskedTriggersAndValidation();
	TestIndependentOutputsAndPadSwitch();
	TestMonoLedPadGetsNoLightBar();
	TestTriggerEffectState();
	TestUnreadPressSurvivesStickBurst();
	TestTouchWithoutClick();
	TestMultiplayerRouting();
	TestDedicatedKeyboardOutputs();
	TestMotionPath();
	TestSensorsSurviveOutputsAndReopen();
	TestNoNeutralPublishWhileSensorsClaimed();
	TestLockOrder();
	Config::Shutdown();
	return 0;
}
