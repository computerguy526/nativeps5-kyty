#include "graphics/host_gpu/renderer/frameGen.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#if KYTY_STREAMLINE
#include <sl.h>
#include <sl_consts.h>
#include <sl_dlss_g.h>
#include <sl_pcl.h>
#include <sl_reflex.h>
#endif

namespace Libs::Graphics::FrameGen {

bool Requested() {
	static const bool requested = [] {
		const char* value = std::getenv("ASTROHYBRID_DLSSG");
		return value != nullptr && std::strcmp(value, "1") == 0;
	}();
	return requested;
}

#if !KYTY_STREAMLINE

PFN_vkGetInstanceProcAddr LoaderBeforeInstance() {
	if (Requested()) {
		std::printf("DLSS-G: not built in (KYTY_STREAMLINE_DIR)\n");
	}
	return nullptr;
}
void OnDeviceReady() {}
bool Loaded() {
	return false;
}
bool Active() {
	return false;
}
void GuestFrameEnd(uint64_t) {}
void RenderSubmitted(uint64_t) {}
void PresentStart(uint64_t) {}
void PresentEnd(uint64_t) {}
uint64_t CurrentFrame() {
	return 0;
}
bool CameraWanted() {
	return false;
}
double ShownPerGameFrame() {
	return 1.0;
}
void SetCamera(const float*) {}
bool TagFrame(VkCommandBuffer, const ImageInput&, const ImageInput&, const float*, uint32_t, uint32_t) {
	return false;
}

#else

namespace {

using Mat = std::array<float, 16>; // row-major, row vectors (Streamline's convention)

Mat Multiply(const Mat& a, const Mat& b) {
	Mat r {};
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			float sum = 0;
			for (int k = 0; k < 4; k++) {
				sum += a[i * 4 + k] * b[k * 4 + j];
			}
			r[i * 4 + j] = sum;
		}
	}
	return r;
}

bool Invert(const Mat& m, Mat& out) {
	double inv[16];
	const double* a = nullptr;
	double        d[16];
	for (int i = 0; i < 16; i++) {
		d[i] = m[i];
	}
	a = d;
	inv[0]  = a[5] * a[10] * a[15] - a[5] * a[11] * a[14] - a[9] * a[6] * a[15] + a[9] * a[7] * a[14] + a[13] * a[6] * a[11] - a[13] * a[7] * a[10];
	inv[4]  = -a[4] * a[10] * a[15] + a[4] * a[11] * a[14] + a[8] * a[6] * a[15] - a[8] * a[7] * a[14] - a[12] * a[6] * a[11] + a[12] * a[7] * a[10];
	inv[8]  = a[4] * a[9] * a[15] - a[4] * a[11] * a[13] - a[8] * a[5] * a[15] + a[8] * a[7] * a[13] + a[12] * a[5] * a[11] - a[12] * a[7] * a[9];
	inv[12] = -a[4] * a[9] * a[14] + a[4] * a[10] * a[13] + a[8] * a[5] * a[14] - a[8] * a[6] * a[13] - a[12] * a[5] * a[10] + a[12] * a[6] * a[9];
	inv[1]  = -a[1] * a[10] * a[15] + a[1] * a[11] * a[14] + a[9] * a[2] * a[15] - a[9] * a[3] * a[14] - a[13] * a[2] * a[11] + a[13] * a[3] * a[10];
	inv[5]  = a[0] * a[10] * a[15] - a[0] * a[11] * a[14] - a[8] * a[2] * a[15] + a[8] * a[3] * a[14] + a[12] * a[2] * a[11] - a[12] * a[3] * a[10];
	inv[9]  = -a[0] * a[9] * a[15] + a[0] * a[11] * a[13] + a[8] * a[1] * a[15] - a[8] * a[3] * a[13] - a[12] * a[1] * a[11] + a[12] * a[3] * a[9];
	inv[13] = a[0] * a[9] * a[14] - a[0] * a[10] * a[13] - a[8] * a[1] * a[14] + a[8] * a[2] * a[13] + a[12] * a[1] * a[10] - a[12] * a[2] * a[9];
	inv[2]  = a[1] * a[6] * a[15] - a[1] * a[7] * a[14] - a[5] * a[2] * a[15] + a[5] * a[3] * a[14] + a[13] * a[2] * a[7] - a[13] * a[3] * a[6];
	inv[6]  = -a[0] * a[6] * a[15] + a[0] * a[7] * a[14] + a[4] * a[2] * a[15] - a[4] * a[3] * a[14] - a[12] * a[2] * a[7] + a[12] * a[3] * a[6];
	inv[10] = a[0] * a[5] * a[15] - a[0] * a[7] * a[13] - a[4] * a[1] * a[15] + a[4] * a[3] * a[13] + a[12] * a[1] * a[7] - a[12] * a[3] * a[5];
	inv[14] = -a[0] * a[5] * a[14] + a[0] * a[6] * a[13] + a[4] * a[1] * a[14] - a[4] * a[2] * a[13] - a[12] * a[1] * a[6] + a[12] * a[2] * a[5];
	inv[3]  = -a[1] * a[6] * a[11] + a[1] * a[7] * a[10] + a[5] * a[2] * a[11] - a[5] * a[3] * a[10] - a[9] * a[2] * a[7] + a[9] * a[3] * a[6];
	inv[7]  = a[0] * a[6] * a[11] - a[0] * a[7] * a[10] - a[4] * a[2] * a[11] + a[4] * a[3] * a[10] + a[8] * a[2] * a[7] - a[8] * a[3] * a[6];
	inv[11] = -a[0] * a[5] * a[11] + a[0] * a[7] * a[9] + a[4] * a[1] * a[11] - a[4] * a[3] * a[9] - a[8] * a[1] * a[7] + a[8] * a[3] * a[5];
	inv[15] = a[0] * a[5] * a[10] - a[0] * a[6] * a[9] - a[4] * a[1] * a[10] + a[4] * a[2] * a[9] + a[8] * a[1] * a[6] - a[8] * a[2] * a[5];
	const double det = a[0] * inv[0] + a[1] * inv[4] + a[2] * inv[8] + a[3] * inv[12];
	if (std::fabs(det) < 1e-20) {
		return false;
	}
	for (int i = 0; i < 16; i++) {
		out[i] = static_cast<float>(inv[i] / det);
	}
	return true;
}

void Copy(sl::float4x4& to, const Mat& from) {
	for (int r = 0; r < 4; r++) {
		to.row[r] = sl::float4(from[r * 4 + 0], from[r * 4 + 1], from[r * 4 + 2], from[r * 4 + 3]);
	}
}

struct Api {
	HMODULE                       module = nullptr;
	PFun_slInit*                  init = nullptr;
	PFun_slSetTagForFrame*        set_tag = nullptr;
	PFun_slSetConstants*          set_constants = nullptr;
	PFun_slGetNewFrameToken*      new_token = nullptr;
	PFun_slGetFeatureFunction*    feature_function = nullptr;
	PFun_slDLSSGSetOptions*       dlssg_set_options = nullptr;
	PFun_slDLSSGGetState*         dlssg_get_state = nullptr;
	PFun_slReflexSetOptions*      reflex_set_options = nullptr;
	PFun_slReflexSleep*           reflex_sleep = nullptr;
	PFun_slPCLSetMarker*          pcl_marker = nullptr;
	PFN_vkGetInstanceProcAddr     gipa = nullptr;
};

struct State {
	Api                   api;
	std::atomic<bool>     active {false};
	std::atomic<uint64_t> cp_frame {0};
	std::mutex            camera_mutex;
	bool                  camera_wanted = true;
	bool                  camera_valid  = false;
	bool                  previous_valid = false;
	Mat                   projection {};   // unjittered, row-major row vectors
	Mat                   view {};
	Mat                   previous_view_projection {};
	uint32_t              frames_generated = 1;
	uint32_t              mvec_width = 0, mvec_height = 0;
	uint64_t              tags = 0, tag_failures = 0, sleeps = 0;
	uint64_t              presents = 0, presented_frames = 0;
	std::atomic<double>   shown_ratio {1.0};
	int64_t               print_ns = 0;
	// ASTROHYBRID_DLSSG_TARGET: adaptive multiplier (present thread, options under options_mutex).
	std::mutex            options_mutex;
	uint32_t              target_fps = 0;
	uint32_t              max_frames = 3;
	int64_t               window_ns = 0;
	uint32_t              window_presents = 0;
	uint32_t              lower_windows = 0;
	double                game_fps = 0.0;
	uint32_t              switches = 0;
};

State& Get() {
	static State state;
	return state;
}

int64_t NowNs() {
	return std::chrono::duration_cast<std::chrono::nanoseconds>(
	           std::chrono::steady_clock::now().time_since_epoch())
	    .count();
}

sl::FrameToken* Token(uint64_t frame) {
	auto& s = Get();
	sl::FrameToken* token = nullptr;
	const uint32_t  index = static_cast<uint32_t>(frame);
	if (s.api.new_token(token, &index) != sl::Result::eOk) {
		return nullptr;
	}
	return token;
}

void Marker(sl::PCLMarker marker, uint64_t frame) {
	auto& s = Get();
	if (!s.active.load(std::memory_order_acquire) || frame == 0) {
		return;
	}
	if (auto* token = Token(frame); token != nullptr) {
		s.api.pcl_marker(marker, *token);
	}
}

void ApplyOptions() {
	auto&              s = Get();
	std::scoped_lock   lock(s.options_mutex);
	sl::ViewportHandle viewport {0};
	sl::DLSSGOptions   options {};
	options.mode                = sl::DLSSGMode::eOn;
	options.numFramesToGenerate = s.frames_generated;
	options.mvecDepthWidth      = s.mvec_width;
	options.mvecDepthHeight     = s.mvec_height;
	const auto result           = s.api.dlssg_set_options(viewport, options);
	std::printf("DLSS-G: options mode on, %u generated frame(s) per frame, mvec/depth %ux%u -> %d\n",
	            s.frames_generated, s.mvec_width, s.mvec_height, static_cast<int>(result));
	std::fflush(stdout);
}

void Summary() {
	auto&      s   = Get();
	const auto now = NowNs();
	if (s.print_ns == 0) {
		s.print_ns = now;
		return;
	}
	if (now - s.print_ns < 10'000'000'000LL) {
		return;
	}
	s.print_ns = now;
	sl::ViewportHandle viewport {0};
	sl::DLSSGState     state {};
	const auto         result = s.api.dlssg_get_state(viewport, state, nullptr);
	std::printf("DLSS-G 10s: %llu game presents, %llu frames shown (%.2f per game frame) | state %d status 0x%x, last %u, max generated %u, "
	            "VRAM %.0f MiB | tags %llu (failed %llu), reflex sleeps %llu, camera %s | multiplier %ux, "
	            "game %.1f FPS, %u switches\n",
	            static_cast<unsigned long long>(s.presents), static_cast<unsigned long long>(s.presented_frames),
	            s.presents == 0 ? 0.0 : static_cast<double>(s.presented_frames) / static_cast<double>(s.presents),
	            static_cast<int>(result), static_cast<unsigned>(state.status),
	            state.numFramesActuallyPresented, state.numFramesToGenerateMax,
	            static_cast<double>(state.estimatedVRAMUsageInBytes) / 1048576.0,
	            static_cast<unsigned long long>(s.tags), static_cast<unsigned long long>(s.tag_failures),
	            static_cast<unsigned long long>(s.sleeps), s.camera_valid ? "yes" : "no",
	            s.frames_generated + 1, s.game_fps, s.switches);
	std::fflush(stdout);
	s.tags = s.tag_failures = s.sleeps = 0;
	s.presents = s.presented_frames = 0;
}

} // namespace

PFN_vkGetInstanceProcAddr LoaderBeforeInstance() {
	if (!Requested()) {
		return nullptr;
	}
	auto&        s          = Get();
	const char*  path_value = std::getenv("ASTROHYBRID_DLSS_PATH");
	std::string  dir        = path_value != nullptr ? path_value
	                                                : "E:\\AstroHybrid-Work\\sdk\\streamline-v2.14.16\\bin\\x64";
	const auto   dll        = dir + "\\sl.interposer.dll";
	s.api.module            = LoadLibraryA(dll.c_str());
	if (s.api.module == nullptr) {
		std::printf("DLSS-G: cannot load %s (%lu)\n", dll.c_str(), GetLastError());
		return nullptr;
	}
	const auto symbol = [&](const char* name) { return reinterpret_cast<void*>(GetProcAddress(s.api.module, name)); };
	s.api.init             = reinterpret_cast<PFun_slInit*>(symbol("slInit"));
	s.api.set_tag          = reinterpret_cast<PFun_slSetTagForFrame*>(symbol("slSetTagForFrame"));
	s.api.set_constants    = reinterpret_cast<PFun_slSetConstants*>(symbol("slSetConstants"));
	s.api.new_token        = reinterpret_cast<PFun_slGetNewFrameToken*>(symbol("slGetNewFrameToken"));
	s.api.feature_function = reinterpret_cast<PFun_slGetFeatureFunction*>(symbol("slGetFeatureFunction"));
	s.api.gipa             = reinterpret_cast<PFN_vkGetInstanceProcAddr>(symbol("vkGetInstanceProcAddr"));
	if (s.api.init == nullptr || s.api.set_tag == nullptr || s.api.set_constants == nullptr ||
	    s.api.new_token == nullptr || s.api.feature_function == nullptr || s.api.gipa == nullptr) {
		std::printf("DLSS-G: sl.interposer.dll lacks an entry point\n");
		return nullptr;
	}
	static std::wstring          wide(dir.begin(), dir.end());
	static const wchar_t*        paths[] = {wide.c_str()};
	static const sl::Feature     features[] = {sl::kFeatureDLSS_G, sl::kFeatureReflex, sl::kFeaturePCL};
	sl::Preferences              prefs {};
	prefs.showConsole        = false;
	prefs.logLevel           = sl::LogLevel::eDefault;
	prefs.pathsToPlugins     = paths;
	prefs.numPathsToPlugins  = 1;
	prefs.pathToLogsAndData  = nullptr;
	// Streamline's warnings and errors on the console (ASTROHYBRID_DLSSG_LOG=1: everything).
	static const bool verbose = [] {
		const char* value = std::getenv("ASTROHYBRID_DLSSG_LOG");
		return value != nullptr && value[0] == '1';
	}();
	prefs.logLevel           = verbose ? sl::LogLevel::eVerbose : sl::LogLevel::eDefault;
	prefs.logMessageCallback = [](sl::LogType type, const char* message) {
		if (type != sl::LogType::eInfo || verbose) {
			std::printf("SL: %s", message);
			std::fflush(stdout);
		}
	};
	prefs.flags              = sl::PreferenceFlags::eDisableCLStateTracking |
	                           sl::PreferenceFlags::eUseFrameBasedResourceTagging;
	prefs.featuresToLoad     = features;
	prefs.numFeaturesToLoad  = 3;
	prefs.engine             = sl::EngineType::eCustom;
	prefs.engineVersion      = "1.0";
	prefs.projectId          = "8e5a33f4-7e0f-4c5b-9a5c-1c2e7d4b3a10";
	prefs.renderAPI          = sl::RenderAPI::eVulkan;
	const auto result        = s.api.init(prefs, sl::kSDKVersion);
	if (result != sl::Result::eOk) {
		std::printf("DLSS-G: slInit failed (%d)\n", static_cast<int>(result));
		return nullptr;
	}
	if (const char* value = std::getenv("ASTROHYBRID_DLSSG_FRAMES"); value != nullptr) {
		s.frames_generated = std::max(1, std::atoi(value));
	}
	if (const char* value = std::getenv("ASTROHYBRID_DLSSG_TARGET"); value != nullptr) {
		s.target_fps = static_cast<uint32_t>(std::clamp(std::atoi(value), 0, 400));
	}
	if (const char* value = std::getenv("ASTROHYBRID_DLSSG_MAX_FRAMES"); value != nullptr) {
		s.max_frames = static_cast<uint32_t>(std::clamp(std::atoi(value), 1, 5));
	}
	std::printf("DLSS-G: Streamline initialised; Vulkan through sl.interposer.dll\n");
	std::fflush(stdout);
	return s.api.gipa;
}

void OnDeviceReady() {
	auto& s = Get();
	if (s.api.gipa == nullptr) {
		return;
	}
	const auto fetch = [&](sl::Feature feature, const char* name) {
		void* function = nullptr;
		return s.api.feature_function(feature, name, function) == sl::Result::eOk ? function : nullptr;
	};
	s.api.dlssg_set_options  = reinterpret_cast<PFun_slDLSSGSetOptions*>(fetch(sl::kFeatureDLSS_G, "slDLSSGSetOptions"));
	s.api.dlssg_get_state    = reinterpret_cast<PFun_slDLSSGGetState*>(fetch(sl::kFeatureDLSS_G, "slDLSSGGetState"));
	s.api.reflex_set_options = reinterpret_cast<PFun_slReflexSetOptions*>(fetch(sl::kFeatureReflex, "slReflexSetOptions"));
	s.api.reflex_sleep       = reinterpret_cast<PFun_slReflexSleep*>(fetch(sl::kFeatureReflex, "slReflexSleep"));
	s.api.pcl_marker         = reinterpret_cast<PFun_slPCLSetMarker*>(fetch(sl::kFeaturePCL, "slPCLSetMarker"));
	if (s.api.dlssg_set_options == nullptr || s.api.dlssg_get_state == nullptr ||
	    s.api.reflex_set_options == nullptr || s.api.reflex_sleep == nullptr || s.api.pcl_marker == nullptr) {
		std::printf("DLSS-G: feature functions unavailable (DLSS-G %p/%p, Reflex %p/%p, PCL %p): off\n",
		            reinterpret_cast<void*>(s.api.dlssg_set_options), reinterpret_cast<void*>(s.api.dlssg_get_state),
		            reinterpret_cast<void*>(s.api.reflex_set_options), reinterpret_cast<void*>(s.api.reflex_sleep),
		            reinterpret_cast<void*>(s.api.pcl_marker));
		std::fflush(stdout);
		return;
	}
	sl::ReflexOptions reflex {};
	reflex.mode = sl::ReflexMode::eLowLatency;
	const auto reflex_result = s.api.reflex_set_options(reflex);
	sl::ViewportHandle viewport {0};
	sl::DLSSGState     state {};
	const auto         state_result = s.api.dlssg_get_state(viewport, state, nullptr);
	std::printf("DLSS-G: Reflex low latency -> %d; runtime state -> %d, status 0x%x, max generated frames "
	            "%u, dynamic MFG supported %d\n",
	            static_cast<int>(reflex_result), static_cast<int>(state_result),
	            static_cast<unsigned>(state.status), state.numFramesToGenerateMax,
	            static_cast<int>(state.bIsDynamicMFGSupported));
	if (state.numFramesToGenerateMax != 0 && s.frames_generated > state.numFramesToGenerateMax) {
		std::printf("DLSS-G: %u generated frames requested, the runtime allows %u\n", s.frames_generated,
		            state.numFramesToGenerateMax);
		s.frames_generated = state.numFramesToGenerateMax;
	}
	if (state.numFramesToGenerateMax != 0) {
		s.max_frames = std::min(s.max_frames, state.numFramesToGenerateMax);
	}
	if (s.target_fps != 0) {
		std::printf("DLSS-G: adaptive multiplier, at least %u FPS on screen, up to %ux\n", s.target_fps,
		            s.max_frames + 1);
	}
	s.active.store(true, std::memory_order_release);
	ApplyOptions();
}

bool Loaded() {
	return Get().api.gipa != nullptr;
}

bool Active() {
	return Get().active.load(std::memory_order_acquire);
}

void GuestFrameEnd(uint64_t frame) {
	auto& s = Get();
	if (!Active()) {
		return;
	}
	Marker(sl::PCLMarker::eSimulationEnd, frame);
	if (auto* token = Token(frame + 1); token != nullptr) {
		s.api.reflex_sleep(*token);
		s.sleeps++;
	}
	Marker(sl::PCLMarker::eSimulationStart, frame + 1);
}

void RenderSubmitted(uint64_t frame) {
	auto& s = Get();
	s.cp_frame.store(frame, std::memory_order_release);
	Marker(sl::PCLMarker::eRenderSubmitEnd, frame);
	Marker(sl::PCLMarker::eRenderSubmitStart, frame + 1);
}

namespace {
// ASTROHYBRID_DLSSG_TARGET: the smallest multiplier that keeps the screen at or above the target for
// the game's measured frame rate (half-second windows); up after one short window, down only after
// four windows (2 s) in which the lower multiplier would have been enough.
void Adapt() {
	auto&      s   = Get();
	const auto now = NowNs();
	s.window_presents++;
	if (s.window_ns == 0) {
		s.window_ns = now;
		return;
	}
	const auto elapsed = now - s.window_ns;
	if (elapsed < 500'000'000LL) {
		return;
	}
	s.game_fps        = static_cast<double>(s.window_presents) * 1e9 / static_cast<double>(elapsed);
	s.window_ns       = now;
	s.window_presents = 0;
	if (s.game_fps < 1.0) {
		return;
	}
	const auto wanted = static_cast<uint32_t>(std::clamp(
	    std::ceil(static_cast<double>(s.target_fps) / s.game_fps - 0.02) - 1.0, 1.0,
	    static_cast<double>(s.max_frames)));
	if (wanted > s.frames_generated) {
		s.frames_generated = wanted;
		s.lower_windows    = 0;
		s.switches++;
		ApplyOptions();
	} else if (wanted < s.frames_generated) {
		if (++s.lower_windows >= 4) {
			s.frames_generated = wanted;
			s.lower_windows    = 0;
			s.switches++;
			ApplyOptions();
		}
	} else {
		s.lower_windows = 0;
	}
}
} // namespace

void PresentStart(uint64_t frame) {
	Marker(sl::PCLMarker::ePresentStart, frame);
}

void PresentEnd(uint64_t frame) {
	Marker(sl::PCLMarker::ePresentEnd, frame);
	if (Active()) {
		auto&              s = Get();
		sl::ViewportHandle viewport {0};
		sl::DLSSGState     state {};
		if (s.api.dlssg_get_state(viewport, state, nullptr) == sl::Result::eOk) {
			s.presents++;
			s.presented_frames += state.numFramesActuallyPresented;
			const double shown = std::max(1u, state.numFramesActuallyPresented);
			const double old   = s.shown_ratio.load(std::memory_order_relaxed);
			s.shown_ratio.store(old + (shown - old) * 0.1, std::memory_order_relaxed);
		}
		if (s.target_fps != 0) {
			Adapt();
		}
		Summary();
	}
}

uint64_t CurrentFrame() {
	return Get().cp_frame.load(std::memory_order_acquire) + 1;
}

double ShownPerGameFrame() {
	return Active() ? Get().shown_ratio.load(std::memory_order_relaxed) : 1.0;
}

bool CameraWanted() {
	auto& s = Get();
	return Active() && s.camera_wanted;
}

void SetCamera(const float* c) {
	auto&            s = Get();
	std::scoped_lock lock(s.camera_mutex);
	Mat projection {};
	Mat view {};
	std::memcpy(projection.data(), c, sizeof(projection));
	std::memcpy(view.data(), c + 16, sizeof(view));
	projection[8] = 0.0f;
	projection[9] = 0.0f;
	s.projection    = projection;
	s.view          = view;
	s.camera_valid  = true;
	s.camera_wanted = false;
}

bool TagFrame(VkCommandBuffer cmd, const ImageInput& depth, const ImageInput& motion,
              const float jitter_uv[2], uint32_t width, uint32_t height) {
	auto& s = Get();
	if (!Active()) {
		return false;
	}
	const uint64_t frame = CurrentFrame();
	auto*          token = Token(frame);
	if (token == nullptr) {
		s.tag_failures++;
		return false;
	}
	if (s.mvec_width != width || s.mvec_height != height) {
		s.mvec_width  = width;
		s.mvec_height = height;
		ApplyOptions();
	}
	const auto resource = [](const ImageInput& in) {
		sl::Resource r(sl::ResourceType::eTex2d, reinterpret_cast<void*>(in.image), nullptr,
		               reinterpret_cast<void*>(in.view), static_cast<uint32_t>(in.layout));
		r.width        = in.width;
		r.height       = in.height;
		r.nativeFormat = static_cast<uint32_t>(in.format);
		r.mipLevels    = 1;
		r.arrayLayers  = 1;
		r.flags        = in.flags;
		r.usage        = in.usage;
		return r;
	};
	sl::Resource       depth_res  = resource(depth);
	sl::Resource       motion_res = resource(motion);
	sl::ResourceTag    tags[]     = {
        sl::ResourceTag(&depth_res, sl::kBufferTypeDepth, sl::ResourceLifecycle::eOnlyValidNow),
        sl::ResourceTag(&motion_res, sl::kBufferTypeMotionVectors, sl::ResourceLifecycle::eOnlyValidNow)};
	sl::ViewportHandle viewport {0};
	if (s.api.set_tag(*token, viewport, tags, 2, reinterpret_cast<sl::CommandBuffer*>(cmd)) !=
	    sl::Result::eOk) {
		s.tag_failures++;
		return false;
	}
	sl::Constants consts {};
	{
		std::scoped_lock lock(s.camera_mutex);
		const Mat identity {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
		Mat       inverse_projection = identity;
		Mat       inverse_view       = identity;
		Invert(s.projection, inverse_projection);
		Invert(s.view, inverse_view);
		const Mat view_projection = Multiply(s.view, s.projection);
		Mat       inverse_view_projection = identity;
		Invert(view_projection, inverse_view_projection);
		Mat clip_to_prev = identity;
		if (s.previous_valid) {
			clip_to_prev = Multiply(inverse_view_projection, s.previous_view_projection);
		}
		Mat prev_to_clip = identity;
		Invert(clip_to_prev, prev_to_clip);
		Copy(consts.cameraViewToClip, s.projection);
		Copy(consts.clipToCameraView, inverse_projection);
		Copy(consts.clipToLensClip, identity);
		Copy(consts.clipToPrevClip, clip_to_prev);
		Copy(consts.prevClipToClip, prev_to_clip);
		consts.cameraRight = sl::float3(inverse_view[0], inverse_view[1], inverse_view[2]);
		consts.cameraUp    = sl::float3(inverse_view[4], inverse_view[5], inverse_view[6]);
		consts.cameraFwd   = sl::float3(-inverse_view[8], -inverse_view[9], -inverse_view[10]);
		consts.cameraPos   = sl::float3(inverse_view[12], inverse_view[13], inverse_view[14]);
		const float a = s.projection[10];
		const float b = s.projection[14];
		consts.cameraNear        = (a != 0.0f) ? b / a : 0.1f;
		consts.cameraFar         = (a + 1.0f != 0.0f) ? b / (a + 1.0f) : 1000.0f;
		consts.cameraFOV         = 2.0f * std::atan(1.0f / s.projection[5]);
		consts.cameraAspectRatio = s.projection[5] != 0.0f ? s.projection[5] / s.projection[0] : 1.0f;
		consts.depthInverted     = sl::Boolean::eFalse;
		consts.reset             = s.previous_valid && s.camera_valid ? sl::Boolean::eFalse : sl::Boolean::eTrue;
		s.previous_view_projection = view_projection;
		s.previous_valid           = s.camera_valid;
		s.camera_wanted            = true;
	}
	const float jitter_sign = [] {
		const char* value = std::getenv("ASTROHYBRID_DLSS_JITTER_SIGN");
		return value != nullptr && std::strtof(value, nullptr) < 0 ? -1.0f : 1.0f;
	}();
	consts.jitterOffset              = sl::float2(jitter_sign * jitter_uv[0] * static_cast<float>(width),
	                                              jitter_sign * jitter_uv[1] * static_cast<float>(height));
	consts.mvecScale                 = sl::float2(-1.0f, -1.0f);
	consts.cameraPinholeOffset       = sl::float2(0.0f, 0.0f);
	consts.cameraMotionIncluded      = sl::Boolean::eTrue;
	consts.motionVectors3D           = sl::Boolean::eFalse;
	consts.orthographicProjection    = sl::Boolean::eFalse;
	consts.motionVectorsDilated      = sl::Boolean::eFalse;
	consts.motionVectorsJittered     = sl::Boolean::eTrue;
	if (s.api.set_constants(consts, *token, viewport) != sl::Result::eOk) {
		s.tag_failures++;
		return false;
	}
	s.tags++;
	return true;
}

#endif

} // namespace Libs::Graphics::FrameGen
