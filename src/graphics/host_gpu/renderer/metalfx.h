#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_METALFX_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_METALFX_H_

// nps5: KYTY_METALFX=1|temporal|spatial (default off; macOS on KosmicKrisp with the
// nps5_kk_cmd_metalfx driver command of the patched KosmicKrisp driver): Apple MetalFX in place of Astro Bot's own
// temporal upscale pass (RenderExecutor::TryMetalFxUpscale, metalfx.cpp). Same hook, pass and
// inputs as ASTROHYBRID_DLSS (dlss.h): the pass reads the SMAA-resolved colour, the motion-vector
// target and the history, writes the upscaled colour (and the next history, which only the pass
// itself reads), and its constant buffer gives the jitter (UV, dwords 0-3: current, previous), the
// output size (4, 5), the input size (6, 7), a frame counter (11) and whether motion vectors are
// used (13). The upscale therefore runs BEFORE the game's UI/composite passes: HUD stays native.
//
//   temporal (=1): MTL4FXTemporalScaler with colour, depth, motion vectors and jitter. The game's
//     motion vectors include the projection jitter; macOS 26's MetalFX cannot be told so, so the
//     driver subtracts (current - previous jitter) in a small compute pass first
//     (KYTY_METALFX_MV_JITTER=1|0|-1, default 1: sign of that correction, 0 = none).
//   spatial: MTL4FXSpatialScaler (HDR colour processing) of the SMAA-resolved colour alone.
//
// Anything missing or inconsistent (no driver command, MetalFX refusing the formats, other sizes,
// no depth, motion vectors unused this frame) leaves the game's pass to run as before.
// Options: KYTY_METALFX_PASS=<hex pixel shader hash>|auto (default: the hashes known for
// 01.007/01.019; auto = any pass with the constant layout above, its hash logged once),
// KYTY_METALFX_JITTER_SIGN=1|-1 (default 1), KYTY_METALFX_DEPTH_REVERSED=1|0 (default 0),
// KYTY_METALFX_FIND=1 (log candidate passes, replace nothing).
// One "MetalFX 10s:" console line every 10 s with evaluations and fallbacks by reason.
// The temporal evaluation follows NXNative's DLSS-on-MetalFX path (github.com/seedds/nxnative, MIT).
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"

namespace Libs::Graphics::MetalFx {

[[nodiscard]] bool Requested();

// KYTY_METALFX_FIND=3 [KYTY_METALFX_FIND_W=<width>, default 2432]: once per shader, every draw into
// a colour target of that width and every compute dispatch with a bound image of that width, with
// its images and the first words of its small constant buffers (finding the upscale pass of a build).
[[nodiscard]] bool FindWide();
void FindLog(const char* kind, const PreparedBindings& bindings, TextureCache& cache, ImageId output,
             uint32_t color_count);

} // namespace Libs::Graphics::MetalFx

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_METALFX_H_
