#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DLSS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DLSS_H_

#include "graphics/host_gpu/vulkanCommon.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace Libs::Graphics {
struct GraphicContext;
namespace Pssr {
struct Request;
}
} // namespace Libs::Graphics

// NVIDIA DLSS super resolution through the driver's NGX library and the user's model.
namespace Libs::Graphics::Dlss {

struct D3D11Resource;
struct D3D12Resource;
struct Feature;

// The NGX parameter object, declared in its interface's order.
class Parameter {
public:
	virtual void Set(const char* name, unsigned long long value)        = 0;
	virtual void Set(const char* name, float value)                     = 0;
	virtual void Set(const char* name, double value)                    = 0;
	virtual void Set(const char* name, unsigned int value)              = 0;
	virtual void Set(const char* name, int value)                       = 0;
	virtual void Set(const char* name, D3D11Resource* value)            = 0;
	virtual void Set(const char* name, D3D12Resource* value)            = 0;
	virtual void Set(const char* name, void* value)                     = 0;
	virtual int  Get(const char* name, unsigned long long* value) const = 0;
	virtual int  Get(const char* name, float* value) const              = 0;
	virtual int  Get(const char* name, double* value) const             = 0;
	virtual int  Get(const char* name, unsigned int* value) const       = 0;
	virtual int  Get(const char* name, int* value) const                = 0;
	virtual int  Get(const char* name, D3D11Resource** value) const     = 0;
	virtual int  Get(const char* name, D3D12Resource** value) const     = 0;
	virtual int  Get(const char* name, void** value) const              = 0;
	virtual void Reset()                                                = 0;

protected:
	~Parameter() = default;
};

// An image view NGX reads, or writes when read_write is set.
struct Resource {
	VkImageView             view;
	VkImage                 image;
	VkImageSubresourceRange range;
	VkFormat                format;
	uint32_t                width;
	uint32_t                height;
	int32_t                 type; // 0 is an image view
	uint8_t                 read_write;
};

struct Inputs {
	Resource  color;
	Resource  depth;
	Resource  motion;
	Resource* exposure; // null when absent
	Resource  output;
};

// What a feature fixes at creation: its quality mode and the render sizes that mode takes.
struct Creation {
	uint32_t output_width;
	uint32_t output_height;
	uint32_t flags; // Pssr::Flags fixed at creation
	int32_t  mode;
	uint32_t min_width;
	uint32_t min_height;
	uint32_t max_width; // the render size the feature is created for
	uint32_t max_height;

	bool operator==(const Creation&) const = default;
};

[[nodiscard]] bool Accepts(const Pssr::Request& request);
// Whether a feature made for creation takes the request's output, flags and render size.
[[nodiscard]] bool Takes(const Creation& creation, const Pssr::Request& request,
                         const Inputs& inputs);
// The quality mode whose render range holds the request's render size, with the optimal size
// nearest to it; nullopt when no mode takes the size.
[[nodiscard]] std::optional<Creation> Choose(Parameter& capabilities, const Pssr::Request& request,
                                             const Inputs& inputs);
void SetCreation(Parameter& parameters, const Creation& creation, uint32_t preset);
void SetEvaluation(Parameter& parameters, const Pssr::Request& request, Inputs& inputs, bool reset);

// Loads the driver's NGX library in Pro mode with a DLSS folder. Call before the instance.
void Load();
// Adds the instance or device extensions DLSS needs that `available` has.
void AppendExtensions(std::vector<const char*>&                   out,
                      const std::vector<vk::ExtensionProperties>& available, bool device);
// Starts DLSS on the device. It stays off when the model or the GPU can't run it.
void Init(const GraphicContext& graphics, PFN_vkGetInstanceProcAddr get_instance_proc_addr,
          PFN_vkGetDeviceProcAddr get_device_proc_addr);
// Call after every feature is released and before the device is destroyed.
void                     Shutdown();
[[nodiscard]] Parameter* Capabilities(); // null while DLSS is off

// Both record into command. A null feature or false means DLSS can't take the request.
[[nodiscard]] Feature* CreateFeature(vk::CommandBuffer command, const Creation& creation);
[[nodiscard]] bool     Evaluate(vk::CommandBuffer command, const Feature* feature,
                                const Pssr::Request& request, Inputs& inputs, bool reset);
void                   ReleaseFeature(Feature* feature);

} // namespace Libs::Graphics::Dlss

// ---- AstroHybrid hosted-renderer DLSS (ASTROHYBRID_DLSS), next to upstream's PSSR DLSS above ----
#include <cstdint>
#include <vector>

// ASTROHYBRID_DLSS=1 (default 0; needs a build with KYTY_NGX_SDK_DIR): NVIDIA DLSS Super Resolution
// in place of Astro Bot's own temporal upscale pass (RenderExecutor::TryDlssUpscale, dlss.cpp).
//
// The pass (pixel shader 2936e5eca99d75e7, ASTROHYBRID_DLSS_PASS overrides the hash) is a
// full-screen draw that reads the SMAA-resolved colour, the G-buffer target whose .xy are the
// game's motion vectors and the previous history, and writes the upscaled colour and the next
// history. Its constant buffer gives every value DLSS needs: the current and previous jitter
// (UV, dwords 0-3), the output size (4, 5), the input size (6, 7), a frame counter (11) and
// whether motion vectors are used (13). The motion vectors are in UV, point from the previous
// position to the current one and include the jitter (the pass removes current minus previous
// jitter itself). Depth is not bound to the pass: it is the depth attachment the G-buffer draws
// used while rendering into that motion-vector target (recorded per draw, NoteDlssTargets).
//
// DLSS is created for the pass's own input and output sizes (it writes into the game's output
// target, which every later pass reads), with HDR colour, auto exposure, low-resolution jittered
// motion vectors scaled by (-input width, -input height), and the jitter in input pixels.
// Anything missing or inconsistent (no NGX, DLSS unavailable, another size, no depth, the game
// not using motion vectors this frame) leaves the game's pass to run as before.
//
// NGX on Vulkan needs VK_NVX_binary_import, whose CUDA backing reserves a 48 GiB window below
// 1 TiB when the device is created. The PS5 guest arena fills that range unless the host leaves
// it free: run with ASTROHYBRID_CUDA_HOLE=1 (AnyPS5 libc, 0xDC00000000-0xEC00000000), otherwise
// device creation fails with the extension (ASTROHYBRID_DLSS_PROBE=1 reports it).
// Options: ASTROHYBRID_DLSS_QUALITY=quality|balanced|performance|dlaa (the DLSS preset family;
// default quality); ASTROHYBRID_DLSS_JITTER_SIGN=1|-1 (default 1);
// ASTROHYBRID_DLSS_DEPTH_INVERTED=1|0 (default 0: the game's projection maps the near plane to
// depth 0); ASTROHYBRID_DLSS_PATH: the folder holding nvngx_dlss.dll. One "DLSS 10s:" console line
// every 10 s with evaluations and fallbacks by reason.
namespace Libs::Graphics::Dlss {

[[nodiscard]] bool Requested();
// Device creation: the extensions NGX needs (empty when not requested or not built in).
void RequiredExtensions(std::vector<const char*>& instance, std::vector<const char*>& device);
// ASTROHYBRID_DLSS_PROBE=1: whether a device with VK_NVX_binary_import can be created now.
void ProbeBinaryImport(const char* when);

} // namespace Libs::Graphics::Dlss

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DLSS_H_
