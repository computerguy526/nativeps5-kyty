#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_FRAMEGEN_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_FRAMEGEN_H_

#include <cstdint>

#include <vulkan/vulkan_core.h>

// ASTROHYBRID_DLSSG=1 (default 0; needs a build with KYTY_STREAMLINE_DIR and the Streamline runtime
// in ASTROHYBRID_DLSS_PATH, plus ASTROHYBRID_CUDA_HOLE=1 like DLSS): NVIDIA DLSS Frame Generation
// through Streamline (frameGen.cpp).
//
// Streamline is initialised before Kyty's Vulkan instance (Vulkan, features DLSS-G, Reflex and PC
// latency; no over-the-air updates or downloaded plugins) and Kyty's dispatcher then takes its
// entry points from sl.interposer.dll, so Streamline adds what DLSS-G needs to the instance and
// device (extensions, the optical-flow queue) and intercepts the swapchain and presents.
//
// Per guest frame (numbered by flips, as for ASTROHYBRID_LOW_LATENCY, whose markers it takes over:
// the Vulkan low-latency path stays off while DLSS-G is on):
//  - the guest render thread ends the frame with Streamline's Reflex sleep and simulation markers;
//  - the command processor marks render submission at each reserved flip;
//  - the camera: the game's per-view constants (the projection with and without jitter, the view
//    matrix and the inverse projection, 48 floats) captured from the first draw of the frame that
//    binds them; the previous frame's are kept here for the clip-to-previous-clip matrices;
//  - at the game's temporal upscale pass: depth and motion vectors tagged (valid only now, copied by
//    Streamline in the command buffer) and the constants set for the frame;
//  - the presenter marks present start and end with the presented flip's frame.
// ASTROHYBRID_DLSSG_FRAMES=n (default 1): frames generated per rendered frame, clamped to what the
// runtime reports (numFramesToGenerateMax). Dynamic multi frame generation is not used (D3D12 only).
// ASTROHYBRID_DLSSG_TARGET=n (default 0 = fixed): adaptive whole multiplier instead: every half second
// the smallest multiplier keeping the screen at or above n FPS for the game's measured rate (up at
// once, down after 2 s), at most ASTROHYBRID_DLSSG_MAX_FRAMES (default 3) generated frames.
// One "DLSS-G 10s:" console line every 10 s with the runtime's status and frames presented.
namespace Libs::Graphics::FrameGen {

[[nodiscard]] bool Requested();
// Before the Vulkan instance: Streamline's vkGetInstanceProcAddr, or null (not requested, not
// built in or initialisation failed: the normal loader is used and DLSS-G stays off).
[[nodiscard]] PFN_vkGetInstanceProcAddr LoaderBeforeInstance();
// Streamline's interposer is Kyty's Vulkan loader (surfaces must then be created through it, so
// Streamline can associate the swapchain with its window).
[[nodiscard]] bool Loaded();
// After the device: Reflex, DLSS-G options and the runtime state.
void OnDeviceReady();
[[nodiscard]] bool Active();
// Frames on screen per game frame, averaged over recent presents (1 when generation is off or
// paused, e.g. the window lost focus); for the performance overlay. Any thread.
[[nodiscard]] double ShownPerGameFrame();

// Frame events (frame numbers as in lowLatency.h).
void GuestFrameEnd(uint64_t frame);
void RenderSubmitted(uint64_t frame);
void PresentStart(uint64_t frame);
void PresentEnd(uint64_t frame);
// The frame the command processor is working on (the next flip).
[[nodiscard]] uint64_t CurrentFrame();

// Command processor.
[[nodiscard]] bool CameraWanted();
void SetCamera(const float* constants);
struct ImageInput {
	VkImage     image  = VK_NULL_HANDLE;
	VkImageView view   = VK_NULL_HANDLE;
	VkFormat    format = VK_FORMAT_UNDEFINED;
	VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
	uint32_t    width  = 0;
	uint32_t    height = 0;
	uint32_t    usage  = 0;
	uint32_t    flags  = 0;
};
// At the temporal upscale pass: tags and constants for CurrentFrame(); false when not done.
bool TagFrame(VkCommandBuffer cmd, const ImageInput& depth, const ImageInput& motion,
              const float jitter_uv[2], uint32_t width, uint32_t height);

} // namespace Libs::Graphics::FrameGen

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_FRAMEGEN_H_
