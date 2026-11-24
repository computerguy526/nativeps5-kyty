#include "graphics/host_gpu/renderer/dlss.h"

#include "common/logging/log.h"
#include "common/stringUtils.h"
#include "graphics/guest_gpu/pssr.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/frameGen.h"
#include "graphics/shader/shaderBindings.h"

#include <SDL3/SDL_loadso.h>
#include <algorithm>
#include <array>
#include <span>
#include <string>
#include <string_view>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#include <windows.h>
#endif

namespace Libs::Graphics::Dlss {
namespace {

using Result = int;

constexpr int         SDK_VERSION    = 0x15;
constexpr int         ENGINE_CUSTOM  = 0;
constexpr int         SUPER_SAMPLING = 1;
constexpr const char* PROJECT_ID     = "a10e2777-9818-4681-b625-25d71ae45d72";

constexpr int DLAA = 5; // the last quality mode; it renders at the output size

// Feature creation flags.
constexpr int IS_HDR         = 1 << 0;
constexpr int MV_LOW_RES     = 1 << 1;
constexpr int DEPTH_INVERTED = 1 << 3;
constexpr int AUTO_EXPOSURE  = 1 << 6;

// The request flags a feature fixes at creation.
constexpr uint32_t FIXED_FLAGS = Pssr::kReverseDepth | Pssr::kAutoExposure;

struct FeatureCommonInfo {
	const wchar_t* const* paths;
	uint32_t              path_count;
	void*                 internal;
	void*                 log_callback;
	int32_t               log_level;
	uint8_t               log_callback_only;
};
static_assert(sizeof(FeatureCommonInfo) == 40);
static_assert(sizeof(Resource) == 56);

struct Api {
	Result (*required_extensions)(uint32_t*, const char***, uint32_t*, const char***);
	Result (*init)(const char*, int, const char*, const wchar_t*, VkInstance, VkPhysicalDevice,
	               VkDevice, PFN_vkGetInstanceProcAddr, PFN_vkGetDeviceProcAddr, int,
	               const FeatureCommonInfo*);
	Result (*shutdown)(VkDevice, int*);
	Result (*capabilities)(Parameter**);
	Result (*allocate)(Parameter**);
	Result (*destroy)(Parameter*);
	Result (*create)(VkDevice, VkCommandBuffer, int, Parameter*, Feature**);
	Result (*release)(Feature*);
	Result (*evaluate)(VkCommandBuffer, const Feature*, Parameter*, void*);
};

SDL_SharedObject* g_module = nullptr;
Api               g_api {};
VkDevice          g_device       = nullptr;
Parameter*        g_capabilities = nullptr;
Parameter*        g_parameters   = nullptr;

bool Succeeded(Result result) {
	return (static_cast<uint32_t>(result) & 0xfff00000u) != 0xbad00000u;
}

void Warn(std::string_view text) {
	Log::WriteToConsoleAndLog(fmt::format("WARNING: DLSS: {}; PSSR scales linearly.\n", text));
}

void OpenCore() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	// The driver's service install records its folder.
	wchar_t folder[MAX_PATH];
	DWORD   size = sizeof(folder);
	if (RegGetValueW(HKEY_LOCAL_MACHINE,
	                 L"SYSTEM\\CurrentControlSet\\Services\\nvlddmkm\\Parameters\\NGXCore",
	                 L"NGXPath", RRF_RT_REG_SZ, nullptr, folder, &size) == ERROR_SUCCESS) {
		g_module = SDL_LoadObject(
		    Common::PathToString(std::filesystem::path(folder) / L"_nvngx.dll").c_str());
	}
#else
	g_module = SDL_LoadObject("libnvidia-ngx.so.1");
#endif
}

void CloseCore() {
	SDL_UnloadObject(g_module);
	g_module = nullptr;
}

template <typename Function>
bool Bind(Function& function, const char* name) {
	function = reinterpret_cast<Function>(SDL_LoadFunction(g_module, name));
	return function != nullptr;
}

} // namespace

bool Accepts(const Pssr::Request& request) {
	ShaderTextureResource depth {}, motion {};
	std::copy_n(request.depth, 8, depth.fields);
	std::copy_n(request.motion, 8, motion.fields);
	return (request.flags & Pssr::kSpatialOnly) == 0 && !depth.IsNull() && !motion.IsNull();
}

bool Takes(const Creation& creation, const Pssr::Request& request, const Inputs& inputs) {
	return creation.output_width == inputs.output.width &&
	       creation.output_height == inputs.output.height &&
	       creation.flags == (request.flags & FIXED_FLAGS) && request.width >= creation.min_width &&
	       request.width <= creation.max_width && request.height >= creation.min_height &&
	       request.height <= creation.max_height;
}

std::optional<Creation> Choose(Parameter& capabilities, const Pssr::Request& request,
                               const Inputs& inputs) {
	void* callback = nullptr;
	if (!Succeeded(capabilities.Get("DLSSOptimalSettingsCallback", &callback)) ||
	    callback == nullptr) {
		return std::nullopt;
	}
	const auto settings = reinterpret_cast<Result (*)(Parameter*)>(callback);
	const auto read     = [&](const char* name, uint32_t& value) {
		uint32_t found = 0;
		if (Succeeded(capabilities.Get(name, &found)) && found != 0) {
			value = found;
		}
	};
	std::optional<Creation> best;
	uint32_t                best_distance = 0;
	const uint32_t          flags         = request.flags & FIXED_FLAGS;
	for (int mode = 0; mode <= DLAA; ++mode) {
		capabilities.Set("Width", inputs.output.width);
		capabilities.Set("Height", inputs.output.height);
		capabilities.Set("PerfQualityValue", mode);
		capabilities.Set("RTXValue", 0);
		uint32_t optimal_width  = 0;
		uint32_t optimal_height = 0;
		if (!Succeeded(settings(&capabilities))) {
			continue;
		}
		read("OutWidth", optimal_width);
		read("OutHeight", optimal_height);
		if (optimal_width == 0 || optimal_height == 0) {
			continue; // the model lacks the mode
		}
		Creation creation {
		    inputs.output.width, inputs.output.height, flags,         mode,
		    optimal_width,       optimal_height,       optimal_width, optimal_height};
		if (mode == DLAA) {
			creation.min_width = creation.max_width = inputs.output.width;
			creation.min_height = creation.max_height = inputs.output.height;
		} else {
			// Without a dynamic range the mode takes its optimal size only.
			read("DLSS.Get.Dynamic.Min.Render.Width", creation.min_width);
			read("DLSS.Get.Dynamic.Min.Render.Height", creation.min_height);
			read("DLSS.Get.Dynamic.Max.Render.Width", creation.max_width);
			read("DLSS.Get.Dynamic.Max.Render.Height", creation.max_height);
		}
		const auto distance =
		    std::max(optimal_width, request.width) - std::min(optimal_width, request.width);
		if (Takes(creation, request, inputs) && (!best || distance < best_distance)) {
			best          = creation;
			best_distance = distance;
		}
	}
	return best;
}

void SetCreation(Parameter& parameters, const Creation& creation, uint32_t preset) {
	int flags = IS_HDR | MV_LOW_RES;
	if ((creation.flags & Pssr::kReverseDepth) != 0) {
		flags |= DEPTH_INVERTED;
	}
	if ((creation.flags & Pssr::kAutoExposure) != 0) {
		flags |= AUTO_EXPOSURE;
	}
	parameters.Set("CreationNodeMask", 1u);
	parameters.Set("VisibilityNodeMask", 1u);
	parameters.Set("Width", creation.max_width);
	parameters.Set("Height", creation.max_height);
	parameters.Set("OutWidth", creation.output_width);
	parameters.Set("OutHeight", creation.output_height);
	parameters.Set("PerfQualityValue", creation.mode);
	parameters.Set("DLSS.Feature.Create.Flags", flags);
	for (const char* hint:
	     {"DLSS.Hint.Render.Preset.DLAA", "DLSS.Hint.Render.Preset.Quality",
	      "DLSS.Hint.Render.Preset.Balanced", "DLSS.Hint.Render.Preset.Performance",
	      "DLSS.Hint.Render.Preset.UltraPerformance", "DLSS.Hint.Render.Preset.UltraQuality"}) {
		parameters.Set(hint, preset);
	}
}

void SetEvaluation(Parameter& parameters, const Pssr::Request& request, Inputs& inputs,
                   bool reset) {
	parameters.Set("Color", static_cast<void*>(&inputs.color));
	parameters.Set("Depth", static_cast<void*>(&inputs.depth));
	parameters.Set("MotionVectors", static_cast<void*>(&inputs.motion));
	parameters.Set("ExposureTexture", static_cast<void*>(inputs.exposure));
	parameters.Set("Output", static_cast<void*>(&inputs.output));
	parameters.Set("Jitter.Offset.X", request.jitter[0]);
	parameters.Set("Jitter.Offset.Y", request.jitter[1]);
	parameters.Set("MV.Scale.X", request.motion_scale[0]);
	parameters.Set("MV.Scale.Y", request.motion_scale[1]);
	parameters.Set("DLSS.Pre.Exposure", request.pre_exposure);
	parameters.Set("Reset", reset ? 1 : 0);
	parameters.Set("DLSS.Render.Subrect.Dimensions.Width", request.width);
	parameters.Set("DLSS.Render.Subrect.Dimensions.Height", request.height);
	parameters.Set("DLSS.Input.Color.Subrect.Base.X", request.x);
	parameters.Set("DLSS.Input.Color.Subrect.Base.Y", request.y);
	parameters.Set("DLSS.Input.Depth.Subrect.Base.X", request.x);
	parameters.Set("DLSS.Input.Depth.Subrect.Base.Y", request.y);
	parameters.Set("DLSS.Input.MV.Subrect.Base.X", request.x);
	parameters.Set("DLSS.Input.MV.Subrect.Base.Y", request.y);
}

void Load() {
	if (!Config::Ps5ProEnabled() || Config::GetDlssFolder().empty()) {
		return;
	}
	OpenCore();
	if (g_module == nullptr) {
		Warn("the NVIDIA driver has no NGX library");
		return;
	}
	if (!Bind(g_api.required_extensions, "NVSDK_NGX_VULKAN_RequiredExtensions") ||
	    !Bind(g_api.init, "NVSDK_NGX_VULKAN_Init_ProjectID_Ext") ||
	    !Bind(g_api.shutdown, "NVSDK_NGX_VULKAN_Shutdown1") ||
	    !Bind(g_api.capabilities, "NVSDK_NGX_VULKAN_GetCapabilityParameters") ||
	    !Bind(g_api.allocate, "NVSDK_NGX_VULKAN_AllocateParameters") ||
	    !Bind(g_api.destroy, "NVSDK_NGX_VULKAN_DestroyParameters") ||
	    !Bind(g_api.create, "NVSDK_NGX_VULKAN_CreateFeature1") ||
	    !Bind(g_api.release, "NVSDK_NGX_VULKAN_ReleaseFeature") ||
	    !Bind(g_api.evaluate, "NVSDK_NGX_VULKAN_EvaluateFeature")) {
		Warn("the NVIDIA driver's NGX library lacks an entry point");
		CloseCore();
	}
}

void AppendExtensions(std::vector<const char*>&                   out,
                      const std::vector<vk::ExtensionProperties>& available, bool device) {
	std::array<uint32_t, 2>     counts {};
	std::array<const char**, 2> names {};
	if (g_module == nullptr ||
	    !Succeeded(g_api.required_extensions(&counts[0], &names[0], &counts[1], &names[1]))) {
		return;
	}
	for (const char* name: std::span(names[device], counts[device])) {
		const auto same = [&](std::string_view other) { return other == name; };
		if (std::ranges::any_of(available, same,
		                        [](const auto& properties) {
			                        return std::string_view(properties.extensionName.data());
		                        }) &&
		    std::ranges::none_of(out, same)) {
			out.push_back(name);
		}
	}
}

void Init(const GraphicContext& graphics, PFN_vkGetInstanceProcAddr get_instance_proc_addr,
          PFN_vkGetDeviceProcAddr get_device_proc_addr) {
	if (g_module == nullptr) {
		return;
	}
	// Static, since NGX may keep these pointers.
	static const std::wstring      path = Config::GetDlssFolder().wstring();
	static const wchar_t* const    paths[] {path.c_str()};
	static const FeatureCommonInfo info {.paths = paths, .path_count = 1};
	const auto                     folder = Common::PathToString(Config::GetDlssFolder());
	if (const auto result = g_api.init(
	        PROJECT_ID, ENGINE_CUSTOM, "1.0", paths[0], static_cast<VkInstance>(graphics.instance),
	        static_cast<VkPhysicalDevice>(graphics.physical_device),
	        static_cast<VkDevice>(graphics.device), get_instance_proc_addr, get_device_proc_addr,
	        SDK_VERSION, &info);
	    !Succeeded(result)) {
		Warn(fmt::format("NGX did not start (0x{:08x})", static_cast<uint32_t>(result)));
		CloseCore();
		return;
	}
	g_device      = static_cast<VkDevice>(graphics.device);
	int available = 0;
	int status    = 0;
	if (!Succeeded(g_api.capabilities(&g_capabilities)) ||
	    !Succeeded(g_capabilities->Get("SuperSampling.Available", &available)) || available == 0) {
		if (g_capabilities != nullptr) {
			(void)g_capabilities->Get("SuperSampling.FeatureInitResult", &status);
		}
		Warn(fmt::format("super resolution is unavailable (0x{:08x}); check the model in {}",
		                 static_cast<uint32_t>(status), folder));
		Shutdown();
		return;
	}
	if (!Succeeded(g_api.allocate(&g_parameters))) {
		Warn("NGX parameters could not be allocated");
		Shutdown();
		return;
	}
	Log::WriteToConsoleAndLog(fmt::format("DLSS: upscaling PSSR with the model in {}\n", folder));
}

void Shutdown() {
	if (g_parameters != nullptr) {
		g_api.destroy(g_parameters);
		g_parameters = nullptr;
	}
	if (g_capabilities != nullptr) {
		g_api.destroy(g_capabilities);
		g_capabilities = nullptr;
	}
	if (g_device != nullptr) {
		int in_use = 0;
		g_api.shutdown(g_device, &in_use);
		g_device = nullptr;
	}
	CloseCore();
}

Parameter* Capabilities() {
	return g_capabilities;
}

Feature* CreateFeature(vk::CommandBuffer command, const Creation& creation) {
	SetCreation(*g_parameters, creation, Config::GetDlssPreset());
	Feature* feature = nullptr;
	if (const auto result = g_api.create(g_device, static_cast<VkCommandBuffer>(command),
	                                     SUPER_SAMPLING, g_parameters, &feature);
	    !Succeeded(result)) {
		Warn(fmt::format("creating a {}x{} to {}x{} feature failed (0x{:08x})", creation.max_width,
		                 creation.max_height, creation.output_width, creation.output_height,
		                 static_cast<uint32_t>(result)));
		return nullptr;
	}
	return feature;
}

bool Evaluate(vk::CommandBuffer command, const Feature* feature, const Pssr::Request& request,
              Inputs& inputs, bool reset) {
	SetEvaluation(*g_parameters, request, inputs, reset);
	if (const auto result =
	        g_api.evaluate(static_cast<VkCommandBuffer>(command), feature, g_parameters, nullptr);
	    !Succeeded(result)) {
		Warn(fmt::format("evaluation failed (0x{:08x})", static_cast<uint32_t>(result)));
		return false;
	}
	return true;
}

void ReleaseFeature(Feature* feature) {
	g_api.release(feature);
}

} // namespace Libs::Graphics::Dlss

// ---- AstroHybrid hosted-renderer DLSS (ASTROHYBRID_DLSS, dlss.h second block) ----

#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/image/image.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"

#if KYTY_NGX
#include <nvsdk_ngx_helpers_vk.h>
#include <nvsdk_ngx_vk.h>
#endif

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#if defined(_WIN32) // nps5: the non-NGX/Streamline paths below build elsewhere
#include <windows.h>
#endif

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace Libs::Graphics {

namespace Dlss {

bool Requested() {
	static const bool requested = [] {
		const char* value = std::getenv("ASTROHYBRID_DLSS");
		return value != nullptr && std::strcmp(value, "1") == 0;
	}();
	return requested;
}

void RequiredExtensions(std::vector<const char*>& instance, std::vector<const char*>& device) {
#if KYTY_NGX
	if (!Requested()) {
		return;
	}
	unsigned int instance_count = 0;
	unsigned int device_count   = 0;
	const char** instance_names = nullptr;
	const char** device_names   = nullptr;
	if (NVSDK_NGX_VULKAN_RequiredExtensions(&instance_count, &instance_names, &device_count,
	                                        &device_names) != NVSDK_NGX_Result_Success) {
		std::printf("DLSS: NGX did not report its Vulkan extensions\n");
		return;
	}
	// ASTROHYBRID_DLSS_SKIP_EXT=<name>,<name>: extensions not to enable (diagnostic).
	const char*       skip_value = std::getenv("ASTROHYBRID_DLSS_SKIP_EXT");
	const std::string skip =
	    skip_value != nullptr ? std::string(",") + skip_value + "," : std::string();
	const auto skipped = [&](const char* name) {
		return !skip.empty() && skip.find(std::string(",") + name + ",") != std::string::npos;
	};
	static bool printed = false;
	for (unsigned int i = 0; i < instance_count; i++) {
		if (!printed) {
			std::printf("DLSS: NGX instance extension %s%s\n", instance_names[i],
			            skipped(instance_names[i]) ? " (skipped)" : "");
		}
		if (!skipped(instance_names[i])) {
			instance.push_back(instance_names[i]);
		}
	}
	for (unsigned int i = 0; i < device_count; i++) {
		if (!printed) {
			std::printf("DLSS: NGX device extension %s%s\n", device_names[i],
			            skipped(device_names[i]) ? " (skipped)" : "");
		}
		if (!skipped(device_names[i])) {
			device.push_back(device_names[i]);
		}
	}
	printed = true;
	std::fflush(stdout);
#else
	(void)instance;
	(void)device;
#endif
}

// ASTROHYBRID_DLSS_PROBE=1 (diagnostic): a throwaway device with only VK_NVX_binary_import,
// created through vulkan-1.dll directly; prints whether the driver accepts it at this point.
void ProbeBinaryImport(const char* when) {
	const char* value = std::getenv("ASTROHYBRID_DLSS_PROBE");
	if (value == nullptr || std::strcmp(value, "1") != 0) {
		return;
	}
#if !defined(_WIN32)
	(void)when; // nps5: an NVIDIA/Windows probe
#else
	// The largest free address ranges below 1 TiB at this point.
	{
		std::array<std::pair<uint64_t, uint64_t>, 4> largest {};
		uint64_t                                     address = 0x10000;
		MEMORY_BASIC_INFORMATION                     info {};
		while (address < (1ull << 40) && VirtualQuery(reinterpret_cast<void*>(address), &info,
		                                              sizeof(info)) == sizeof(info)) {
			const auto base = reinterpret_cast<uint64_t>(info.BaseAddress);
			if (info.State == MEM_FREE) {
				const auto size     = std::min<uint64_t>(info.RegionSize, (1ull << 40) - base);
				auto&      smallest = *std::min_element(
				    largest.begin(), largest.end(),
				    [](const auto& a, const auto& b) { return a.second < b.second; });
				if (size > smallest.second) {
					smallest = {base, size};
				}
			}
			address = base + info.RegionSize;
		}
		std::printf("DLSS probe (%s): largest free blocks below 1 TiB:", when);
		for (const auto& entry: largest) {
			std::printf(" %llx+%.1fGiB", static_cast<unsigned long long>(entry.first),
			            static_cast<double>(entry.second) / 1073741824.0);
		}
		std::printf("\n");
	}
	auto* module = LoadLibraryA("vulkan-1.dll");
	if (module == nullptr) {
		std::printf("DLSS probe (%s): no vulkan-1.dll\n", when);
		return;
	}
	const auto gipa = reinterpret_cast<PFN_vkGetInstanceProcAddr>(
	    reinterpret_cast<void*>(GetProcAddress(module, "vkGetInstanceProcAddr")));
	const auto create_instance =
	    reinterpret_cast<PFN_vkCreateInstance>(gipa(nullptr, "vkCreateInstance"));
	VkApplicationInfo app {VK_STRUCTURE_TYPE_APPLICATION_INFO};
	app.apiVersion = VK_API_VERSION_1_3;
	VkInstanceCreateInfo instance_info {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
	instance_info.pApplicationInfo = &app;
	VkInstance instance            = VK_NULL_HANDLE;
	if (create_instance(&instance_info, nullptr, &instance) != VK_SUCCESS) {
		std::printf("DLSS probe (%s): no instance\n", when);
		return;
	}
	const auto enumerate = reinterpret_cast<PFN_vkEnumeratePhysicalDevices>(
	    gipa(instance, "vkEnumeratePhysicalDevices"));
	const auto create_device =
	    reinterpret_cast<PFN_vkCreateDevice>(gipa(instance, "vkCreateDevice"));
	const auto destroy_device =
	    reinterpret_cast<PFN_vkDestroyDevice>(gipa(instance, "vkDestroyDevice"));
	const auto destroy_instance =
	    reinterpret_cast<PFN_vkDestroyInstance>(gipa(instance, "vkDestroyInstance"));
	uint32_t         count    = 1;
	VkPhysicalDevice physical = VK_NULL_HANDLE;
	enumerate(instance, &count, &physical);
	const float             priority = 1.0f;
	VkDeviceQueueCreateInfo queue {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
	queue.queueCount        = 1;
	queue.pQueuePriorities  = &priority;
	const char*        name = "VK_NVX_binary_import";
	VkDeviceCreateInfo device_info {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
	device_info.queueCreateInfoCount    = 1;
	device_info.pQueueCreateInfos       = &queue;
	device_info.enabledExtensionCount   = 1;
	device_info.ppEnabledExtensionNames = &name;
	VkDevice   device                   = VK_NULL_HANDLE;
	const auto result = physical != VK_NULL_HANDLE
	                        ? create_device(physical, &device_info, nullptr, &device)
	                        : VK_ERROR_INITIALIZATION_FAILED;
	std::printf("DLSS probe (%s): device with VK_NVX_binary_import -> %d\n", when,
	            static_cast<int>(result));
	std::fflush(stdout);
	if (device != VK_NULL_HANDLE) {
		destroy_device(device, nullptr);
	}
	destroy_instance(instance, nullptr);
#endif
}

} // namespace Dlss

namespace {

enum class Fallback : uint32_t {
	NotBuiltIn,
	InitFailed,
	Unavailable,
	Bindings,
	Sizes,
	NoDepth,
	NoMotionVectors,
	CreateFailed,
	EvaluateFailed,
	Count
};
constexpr std::array<const char*, static_cast<size_t>(Fallback::Count)> FALLBACK_NAMES {
    "not-built-in", "init",  "unavailable", "bindings", "sizes",
    "no-depth",     "no-mv", "create",      "evaluate"};

struct DepthPair {
	ImageId color;
	ImageId depth;
};

struct State {
	uint64_t                  pass_hash      = 0x2936e5eca99d75e7ull;
	float                     jitter_sign    = 1.0f;
	bool                      depth_inverted = false;
	int                       quality        = 0; // 0 quality, 1 balanced, 2 performance, 3 dlaa
	std::array<DepthPair, 16> depth_pairs {};
	uint32_t                  depth_next = 0;
	bool                      init_tried = false;
	bool                      ready      = false;
#if KYTY_NGX
	NVSDK_NGX_Parameter* params  = nullptr;
	NVSDK_NGX_Handle*    feature = nullptr;
#endif
	uint32_t feature_in_w = 0, feature_in_h = 0, feature_out_w = 0, feature_out_h = 0;
	uint32_t last_frame  = 0;
	uint64_t evaluations = 0;
	std::array<uint64_t, static_cast<size_t>(Fallback::Count)> fallbacks {};
	int64_t                                                    print_ns = 0;
	float                                                      last_jitter[2] {};
};

State& Get() {
	static State state = [] {
		State s;
		if (const char* value = std::getenv("ASTROHYBRID_DLSS_PASS"); value != nullptr) {
			s.pass_hash = std::strtoull(value, nullptr, 16);
		}
		if (const char* value = std::getenv("ASTROHYBRID_DLSS_JITTER_SIGN"); value != nullptr) {
			s.jitter_sign = std::strtof(value, nullptr) < 0 ? -1.0f : 1.0f;
		}
		if (const char* value = std::getenv("ASTROHYBRID_DLSS_DEPTH_INVERTED"); value != nullptr) {
			s.depth_inverted = std::strcmp(value, "0") != 0;
		}
		if (const char* value = std::getenv("ASTROHYBRID_DLSS_QUALITY"); value != nullptr) {
			s.quality = std::strcmp(value, "balanced") == 0      ? 1
			            : std::strcmp(value, "performance") == 0 ? 2
			            : std::strcmp(value, "dlaa") == 0        ? 3
			                                                     : 0;
		}
		return s;
	}();
	return state;
}

int64_t NowNs() {
	return std::chrono::duration_cast<std::chrono::nanoseconds>(
	           std::chrono::steady_clock::now().time_since_epoch())
	    .count();
}

void Summary(State& s) {
	const auto now = NowNs();
	if (s.print_ns == 0) {
		s.print_ns = now;
		return;
	}
	if (now - s.print_ns < 10'000'000'000LL) {
		return;
	}
	s.print_ns = now;
	std::string text;
	for (size_t i = 0; i < s.fallbacks.size(); i++) {
		if (s.fallbacks[i] != 0) {
			text += " " + std::string(FALLBACK_NAMES[i]) + "=" + std::to_string(s.fallbacks[i]);
		}
	}
	std::printf(
	    "DLSS 10s: evaluations %llu (%ux%u -> %ux%u, jitter %.3f %.3f px) | game pass kept:%s\n",
	    static_cast<unsigned long long>(s.evaluations), s.feature_in_w, s.feature_in_h,
	    s.feature_out_w, s.feature_out_h, static_cast<double>(s.last_jitter[0]),
	    static_cast<double>(s.last_jitter[1]), text.empty() ? " none" : text.c_str());
	std::fflush(stdout);
	s.evaluations = 0;
	s.fallbacks   = {};
}

bool Fail(State& s, Fallback reason) {
	s.fallbacks[static_cast<size_t>(reason)]++;
	Summary(s);
	return false;
}

} // namespace

void RenderExecutor::NoteFrameGenCamera(const PreparedBindings& stage) {
	if (!FrameGen::CameraWanted()) {
		return;
	}
	// The game's per-view constants: a perspective projection (16:9) at offset 128 of a 576-byte
	// buffer, followed by the view matrix and the inverse projection (frameGen.h).
	for (const auto& source: stage.buffer_sources) {
		if (source.address == 0 || source.size != 576) {
			continue;
		}
		const auto* m    = reinterpret_cast<const float*>(source.address + 128);
		const auto  zero = [](float v) { return std::fabs(v) < 1e-6f; };
		if (!(m[0] > 0.2f && m[5] > 0.2f) || m[5] / m[0] < 1.6f || m[5] / m[0] > 1.9f ||
		    !zero(m[1]) || !zero(m[4]) || !zero(m[15]) ||
		    std::fabs(std::fabs(m[11]) - 1.0f) > 1e-5f) {
			continue;
		}
		FrameGen::SetCamera(m);
		return;
	}
}

void RenderExecutor::TagFrameGen(CommandBuffer& buffer, PreparedBindings& pixel) {
	if (!FrameGen::Active() || pixel.runtime == nullptr || !*pixel.runtime) {
		return;
	}
	auto& s = Get();
	if (pixel.runtime->program->shader_hash != s.pass_hash || pixel.images.size() < 2 ||
	    pixel.buffer_sources.empty() || pixel.buffer_sources[0].address == 0 ||
	    pixel.buffer_sources[0].size < 56) {
		return;
	}
	const auto* cb = reinterpret_cast<const uint32_t*>(pixel.buffer_sources[0].address);
	float       jitter_uv[2];
	std::memcpy(jitter_uv, cb, sizeof(jitter_uv));
	const uint32_t in_w = cb[6], in_h = cb[7];
	auto&          cache = m_context.GetTextureCache();
	const auto     mv_id = pixel.images[0].image_id;
	ImageId        depth_id {};
	for (const auto& pair: s.depth_pairs) {
		if (pair.color == mv_id) {
			depth_id = pair.depth;
		}
	}
	if (!depth_id || cache.m_slot_images.try_get(depth_id) == nullptr) {
		return;
	}
	auto& mv    = cache.GetImage(mv_id);
	auto& depth = cache.GetImage(depth_id);
	if (mv.backing.image == nullptr || depth.backing.image == nullptr ||
	    mv.info.extent.width != in_w || depth.info.extent.width != in_w) {
		return;
	}
	buffer.EndRendering();
	const auto cmd = buffer.Handle();
	mv.Transit(vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits2::eShaderSampledRead, {},
	           cmd);
	depth.Transit(vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits2::eShaderSampledRead,
	              {}, cmd);
	const auto device    = m_context.GetGraphics().device;
	const auto make_view = [&](const Image& image, vk::ImageAspectFlags aspect) {
		vk::ImageViewCreateInfo view_info {};
		view_info.image            = image.backing.image;
		view_info.viewType         = vk::ImageViewType::e2D;
		view_info.format           = image.backing.format;
		view_info.subresourceRange = vk::ImageSubresourceRange {aspect, 0, 1, 0, 1};
		vk::ImageView view;
		return device.createImageView(&view_info, nullptr, &view) == vk::Result::eSuccess
		           ? view
		           : vk::ImageView {};
	};
	const std::array<vk::ImageView, 2> views {make_view(depth, vk::ImageAspectFlagBits::eDepth),
	                                          make_view(mv, vk::ImageAspectFlagBits::eColor)};
	const auto                         input = [](const Image& image, vk::ImageView view) {
		FrameGen::ImageInput in;
		in.image  = static_cast<VkImage>(image.backing.image);
		in.view   = static_cast<VkImageView>(view);
		in.format = static_cast<VkFormat>(image.backing.format);
		in.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		in.width  = image.info.extent.width;
		in.height = image.info.extent.height;
		in.usage  = static_cast<uint32_t>(static_cast<VkImageUsageFlags>(image.backing.usage));
		in.flags  = static_cast<uint32_t>(static_cast<VkImageCreateFlags>(image.backing.flags));
		return in;
	};
	if (views[0] && views[1]) {
		(void)FrameGen::TagFrame(static_cast<VkCommandBuffer>(cmd), input(depth, views[0]),
		                         input(mv, views[1]), jitter_uv, in_w, in_h);
	}
	buffer.ForgetBoundState();
	m_context.GetCommandScheduler().DeferOperation([device, views]() {
		for (const auto view: views) {
			if (view) {
				device.destroyImageView(view);
			}
		}
	});
}

void RenderExecutor::NoteDlssTargets(std::span<const RenderColorInfo> colors, ImageId depth) {
	if ((!Dlss::Requested() && !FrameGen::Requested()) || !depth) {
		return;
	}
	auto& s = Get();
	for (const auto& color: colors) {
		if (!color.image_id) {
			continue;
		}
		bool found = false;
		for (auto& pair: s.depth_pairs) {
			if (pair.color == color.image_id) {
				pair.depth = depth;
				found      = true;
				break;
			}
		}
		if (!found) {
			s.depth_pairs[s.depth_next] = {color.image_id, depth};
			s.depth_next                = (s.depth_next + 1) % s.depth_pairs.size();
		}
	}
}

bool RenderExecutor::TryDlssUpscale(CommandBuffer& buffer, PreparedBindings& pixel,
                                    ImageId output) {
	if (!Dlss::Requested() || pixel.runtime == nullptr || !*pixel.runtime) {
		return false;
	}
	auto& s = Get();
	if (pixel.runtime->program->shader_hash != s.pass_hash) {
		return false;
	}
#if !KYTY_NGX
	(void)buffer;
	(void)output;
	return Fail(s, Fallback::NotBuiltIn);
#else
	// The pass's bindings: motion vectors, colour, history; its constants (dlss.h).
	if (pixel.images.size() < 2 || !output || pixel.buffer_sources.empty() ||
	    pixel.buffer_sources[0].address == 0 || pixel.buffer_sources[0].size < 56) {
		return Fail(s, Fallback::Bindings);
	}
	const auto* cb = reinterpret_cast<const uint32_t*>(pixel.buffer_sources[0].address);
	float       jitter_uv[2];
	std::memcpy(jitter_uv, cb, sizeof(jitter_uv));
	const uint32_t out_w = cb[4], out_h = cb[5], in_w = cb[6], in_h = cb[7], frame = cb[11];
	if (cb[13] == 0) {
		return Fail(s, Fallback::NoMotionVectors);
	}
	auto&      cache    = m_context.GetTextureCache();
	const auto mv_id    = pixel.images[0].image_id;
	const auto color_id = pixel.images[1].image_id;
	ImageId    depth_id {};
	for (const auto& pair: s.depth_pairs) {
		if (pair.color == mv_id) {
			depth_id = pair.depth;
		}
	}
	if (!depth_id || cache.m_slot_images.try_get(depth_id) == nullptr) {
		return Fail(s, Fallback::NoDepth);
	}
	auto&      mv    = cache.GetImage(mv_id);
	auto&      color = cache.GetImage(color_id);
	auto&      depth = cache.GetImage(depth_id);
	auto&      out   = cache.GetImage(output);
	const auto same  = [](const Image& image, uint32_t w, uint32_t h) {
		return image.info.extent.width == w && image.info.extent.height == h &&
		       image.backing.image != nullptr && image.info.samples == 1;
	};
	if (!same(mv, in_w, in_h) || !same(color, in_w, in_h) || !same(depth, in_w, in_h) ||
	    !same(out, out_w, out_h) || !depth.info.IsDepth() ||
	    !(out.backing.usage & vk::ImageUsageFlagBits::eStorage)) {
		return Fail(s, Fallback::Sizes);
	}
	auto& graphics = m_context.GetGraphics();
	if (!s.init_tried) {
		s.init_tried            = true;
		const char*  path_value = std::getenv("ASTROHYBRID_DLSS_PATH");
		std::wstring path =
		    path_value != nullptr
		        ? std::wstring(path_value, path_value + std::strlen(path_value))
		        : std::wstring(L"E:\\AstroHybrid-Work\\sdk\\streamline-v2.14.16\\bin\\x64");
		const wchar_t*              paths[] = {path.c_str()};
		NVSDK_NGX_FeatureCommonInfo info {};
		info.PathListInfo.Path   = paths;
		info.PathListInfo.Length = 1;
		const auto result        = NVSDK_NGX_VULKAN_Init_with_ProjectID(
		    "8e5a33f4-7e0f-4c5b-9a5c-1c2e7d4b3a10", NVSDK_NGX_ENGINE_TYPE_CUSTOM, "1.0",
		    L"E:\\AstroHybrid-Work\\ngx-data", static_cast<VkInstance>(graphics.instance),
		    static_cast<VkPhysicalDevice>(graphics.physical_device),
		    static_cast<VkDevice>(graphics.device),
		    VULKAN_HPP_DEFAULT_DISPATCHER.vkGetInstanceProcAddr,
		    VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr, &info);
		if (NVSDK_NGX_FAILED(result) ||
		    NVSDK_NGX_FAILED(NVSDK_NGX_VULKAN_GetCapabilityParameters(&s.params))) {
			std::printf("DLSS: NGX initialisation failed (0x%08x)\n",
			            static_cast<unsigned>(result));
			std::fflush(stdout);
		} else {
			int available    = 0;
			int needs_driver = 0;
			NVSDK_NGX_Parameter_GetI(s.params, NVSDK_NGX_Parameter_SuperSampling_Available,
			                         &available);
			NVSDK_NGX_Parameter_GetI(s.params, NVSDK_NGX_Parameter_SuperSampling_NeedsUpdatedDriver,
			                         &needs_driver);
			s.ready = available != 0;
			std::printf("DLSS: NGX initialised, Super Resolution %s%s\n",
			            available != 0 ? "available" : "NOT available",
			            needs_driver != 0 ? " (driver update needed)" : "");
			std::fflush(stdout);
		}
	}
	if (s.params == nullptr) {
		return Fail(s, Fallback::InitFailed);
	}
	if (!s.ready) {
		return Fail(s, Fallback::Unavailable);
	}

	// Native recording: outside rendering, the recorder drained (CommandBuffer::Handle).
	buffer.EndRendering();
	const auto cmd = buffer.Handle();
	if (s.feature == nullptr || s.feature_in_w != in_w || s.feature_in_h != in_h ||
	    s.feature_out_w != out_w || s.feature_out_h != out_h) {
		if (s.feature != nullptr) {
			NVSDK_NGX_VULKAN_ReleaseFeature(s.feature);
			s.feature = nullptr;
		}
		NVSDK_NGX_DLSS_Create_Params create {};
		create.Feature.InWidth            = in_w;
		create.Feature.InHeight           = in_h;
		create.Feature.InTargetWidth      = out_w;
		create.Feature.InTargetHeight     = out_h;
		create.Feature.InPerfQualityValue = s.quality == 1   ? NVSDK_NGX_PerfQuality_Value_Balanced
		                                    : s.quality == 2 ? NVSDK_NGX_PerfQuality_Value_MaxPerf
		                                    : s.quality == 3
		                                        ? NVSDK_NGX_PerfQuality_Value_DLAA
		                                        : NVSDK_NGX_PerfQuality_Value_MaxQuality;
		create.InFeatureCreateFlags =
		    NVSDK_NGX_DLSS_Feature_Flags_IsHDR | NVSDK_NGX_DLSS_Feature_Flags_MVLowRes |
		    NVSDK_NGX_DLSS_Feature_Flags_MVJittered | NVSDK_NGX_DLSS_Feature_Flags_AutoExposure |
		    (s.depth_inverted ? NVSDK_NGX_DLSS_Feature_Flags_DepthInverted : 0);
		const auto result = NGX_VULKAN_CREATE_DLSS_EXT(static_cast<VkCommandBuffer>(cmd), 1, 1,
		                                               &s.feature, s.params, &create);
		buffer.ForgetBoundState();
		if (NVSDK_NGX_FAILED(result) || s.feature == nullptr) {
			std::printf("DLSS: feature creation failed (0x%08x) for %ux%u -> %ux%u\n",
			            static_cast<unsigned>(result), in_w, in_h, out_w, out_h);
			std::fflush(stdout);
			s.ready   = false;
			s.feature = nullptr;
			return Fail(s, Fallback::CreateFailed);
		}
		s.feature_in_w  = in_w;
		s.feature_in_h  = in_h;
		s.feature_out_w = out_w;
		s.feature_out_h = out_h;
		std::printf("DLSS: Super Resolution feature created, %ux%u -> %ux%u, preset family %d, "
		            "depth inverted %d, jitter sign %+.0f\n",
		            in_w, in_h, out_w, out_h, s.quality, s.depth_inverted ? 1 : 0,
		            static_cast<double>(s.jitter_sign));
		std::fflush(stdout);
	}

	// Layouts as Streamline's DLSS plugin uses them: inputs sampled, output storage.
	mv.Transit(vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits2::eShaderSampledRead, {},
	           cmd);
	color.Transit(vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits2::eShaderSampledRead,
	              {}, cmd);
	depth.Transit(vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits2::eShaderSampledRead,
	              {}, cmd);
	out.Transit(vk::ImageLayout::eGeneral,
	            vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite,
	            {}, cmd);

	// Views for this evaluation, destroyed once the command buffer completed.
	const auto                   device = graphics.device;
	std::array<vk::ImageView, 4> views {};
	const auto                   make_view = [&](const Image& image, vk::ImageAspectFlags aspect) {
		vk::ImageViewCreateInfo view_info {};
		view_info.image            = image.backing.image;
		view_info.viewType         = vk::ImageViewType::e2D;
		view_info.format           = image.backing.format;
		view_info.subresourceRange = vk::ImageSubresourceRange {aspect, 0, 1, 0, 1};
		vk::ImageView view;
		return device.createImageView(&view_info, nullptr, &view) == vk::Result::eSuccess
		           ? view
		           : vk::ImageView {};
	};
	views[0]            = make_view(color, vk::ImageAspectFlagBits::eColor);
	views[1]            = make_view(mv, vk::ImageAspectFlagBits::eColor);
	views[2]            = make_view(depth, vk::ImageAspectFlagBits::eDepth);
	views[3]            = make_view(out, vk::ImageAspectFlagBits::eColor);
	const auto resource = [&](const Image& image, vk::ImageView view, vk::ImageAspectFlags aspect,
	                          bool write) {
		const VkImageSubresourceRange range {static_cast<VkImageAspectFlags>(aspect), 0, 1, 0, 1};
		return NVSDK_NGX_Create_ImageView_Resource_VK(
		    static_cast<VkImageView>(view), static_cast<VkImage>(image.backing.image), range,
		    static_cast<VkFormat>(image.backing.format), image.info.extent.width,
		    image.info.extent.height, write);
	};
	auto color_res = resource(color, views[0], vk::ImageAspectFlagBits::eColor, false);
	auto mv_res    = resource(mv, views[1], vk::ImageAspectFlagBits::eColor, false);
	auto depth_res = resource(depth, views[2], vk::ImageAspectFlagBits::eDepth, false);
	auto out_res   = resource(out, views[3], vk::ImageAspectFlagBits::eColor, true);

	NVSDK_NGX_VK_DLSS_Eval_Params eval {};
	eval.Feature.pInColor  = &color_res;
	eval.Feature.pInOutput = &out_res;
	eval.pInDepth          = &depth_res;
	eval.pInMotionVectors  = &mv_res;
	// Jitter in input pixels; motion vectors in UV from previous to current (dlss.h).
	eval.InJitterOffsetX           = s.jitter_sign * jitter_uv[0] * static_cast<float>(in_w);
	eval.InJitterOffsetY           = s.jitter_sign * jitter_uv[1] * static_cast<float>(in_h);
	eval.InMVScaleX                = -static_cast<float>(in_w);
	eval.InMVScaleY                = -static_cast<float>(in_h);
	eval.InRenderSubrectDimensions = {in_w, in_h};
	eval.InReset                   = (s.last_frame != 0 && frame != s.last_frame + 1) ? 1 : 0;
	s.last_frame                   = frame;
	s.last_jitter[0]               = eval.InJitterOffsetX;
	s.last_jitter[1]               = eval.InJitterOffsetY;
	const auto result =
	    NGX_VULKAN_EVALUATE_DLSS_EXT(static_cast<VkCommandBuffer>(cmd), s.feature, s.params, &eval);
	buffer.ForgetBoundState();
	m_context.GetCommandScheduler().DeferOperation([device, views]() {
		for (const auto view: views) {
			if (view) {
				device.destroyImageView(view);
			}
		}
	});
	if (NVSDK_NGX_FAILED(result)) {
		static std::atomic<uint32_t> logged {0};
		if (logged.fetch_add(1) < 8) {
			std::printf("DLSS: evaluation failed (0x%08x)\n", static_cast<unsigned>(result));
			std::fflush(stdout);
		}
		return Fail(s, Fallback::EvaluateFailed);
	}
	// The output now holds DLSS's image: a GPU write of the target, as the draw's would be.
	out.NoteContentWrite();
	cache.CommitGpuWrite(out);
	s.evaluations++;
	Summary(s);
	return true;
#endif
}

} // namespace Libs::Graphics
