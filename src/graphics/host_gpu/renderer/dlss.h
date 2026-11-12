#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DLSS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DLSS_H_

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
// default quality); ASTROHYBRID_DLSS_JITTER_SIGN=1|-1 (default 1); ASTROHYBRID_DLSS_DEPTH_INVERTED=1|0
// (default 1); ASTROHYBRID_DLSS_PATH: the folder holding nvngx_dlss.dll.
// One "DLSS 10s:" console line every 10 s with evaluations and fallbacks by reason.
namespace Libs::Graphics::Dlss {

[[nodiscard]] bool Requested();
// Device creation: the extensions NGX needs (empty when not requested or not built in).
void RequiredExtensions(std::vector<const char*>& instance, std::vector<const char*>& device);
// ASTROHYBRID_DLSS_PROBE=1: whether a device with VK_NVX_binary_import can be created now.
void ProbeBinaryImport(const char* when);

} // namespace Libs::Graphics::Dlss

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DLSS_H_
