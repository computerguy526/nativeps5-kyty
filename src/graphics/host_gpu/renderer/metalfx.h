#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_METALFX_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_METALFX_H_

// nps5: KYTY_METALFX=1|temporal|spatial (default off; macOS on KosmicKrisp with the
// nps5_kk_cmd_metalfx driver command, work/s3/kk-mfx2): Apple MetalFX at Astro Bot's own temporal
// upscale pass (RenderExecutor::TryMetalFxUpscale / FinishMetalFx, metalfx.cpp), so the upscale runs
// BEFORE the game's post-processing, UI and composite: the HUD stays native.
//
// The pass is found by a profile (metalfx.cpp KNOWN_PASSES; KYTY_METALFX_PASS=<hex hash>[:1019|:1007]
// for another build): its pixel shader hash, its shape and its constant layout.
//   01.007 (5a34b303576ee324, layout 1007): a TAA resolve drawn at the output size into two targets
//     (picture, next history). Images: motion vectors (.xy in UV, previous = uv - mv), scene colour at
//     the DRS level size with LINEAR view depth in .w, history. Constants: output size, input size,
//     input size - 1, "copy, no history" flag (camera cut), frame counter, motion-valid flag. No
//     projection jitter. The history target is read by other passes, so the game's pass still draws
//     and MetalFX then overwrites its first target.
//   01.019 (2936e5eca99d75e7, layout 1019, as ASTROHYBRID_DLSS, dlss.h): jitter in the constants,
//     depth = the attachment drawn with the motion target; MetalFX replaces the draw.
//
//   temporal (=1): MTL4FXTemporalScaler with colour, depth, motion vectors (and jitter where there
//     is one). 01.007 depth: the driver writes near / colour.w (reversed depth, KYTY_METALFX_DEPTH_NEAR,
//     default 0.1) into its own R32F texture; KYTY_METALFX_DEPTH=attachment uses the scene's depth
//     attachment instead (KYTY_METALFX_DEPTH_REVERSED=1|0). KYTY_METALFX_MV_SIGN=1|-1.
//   spatial: MTL4FXSpatialScaler (HDR colour processing: the input is linear HDR) of the colour alone.
//
// Only a real upscale from a scene size: input smaller than the output and at most
// KYTY_METALFX_MAX_IN_W (1920) wide.
// Anything missing or inconsistent (no driver command, MetalFX refusing the formats, other sizes,
// motion vectors unused this frame) leaves the game's picture as it is.
// One "MetalFX 10s:" console line every 10 s: evaluations of pass draws, and fallbacks by reason.
// The temporal evaluation follows NXNative's DLSS-on-MetalFX path (github.com/seedds/nxnative, MIT).
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"

namespace Libs::Graphics::MetalFx {

[[nodiscard]] bool Requested();

// KYTY_METALFX_JITTER=1 (experimental, temporal only, passes without a jitter of their own: 01.007):
// the renderer shifts the viewport of the frame's scene draws (a target of the pass's input size
// with a depth attachment, more than 4 vertices) by a Halton(2,3) sub-pixel offset, which MetalFX is
// told (KYTY_METALFX_JITTER_SIGN / _SIGN_Y = 1|-1). The motion vectors come from the game's
// unjittered matrices and need no correction.
// KYTY_METALFX_CYCLE=<flips>[,<list>] [KYTY_METALFX_CYCLE_FROM=<flip>] (in-run A/B and calibration):
// every <flips> flips the next configuration of the list (default gst): g the game's own picture,
// s spatial, t temporal, a/b/c/d temporal with the viewport jitter and MetalFX jitter signs ++, --,
// +-, -+. Each change resets the history and logs "MetalFX cycle: flip <n> config <c>".
[[nodiscard]] bool JitterRequested();
[[nodiscard]] bool ViewportJitter(uint32_t width, uint32_t height, bool has_depth, uint32_t vertices,
                                  float& x, float& y);
void NoteFlip();

// KYTY_METALFX_FIND=3 [KYTY_METALFX_FIND_W=<width>, default 2432]: once per shader, every draw into
// a colour target of that width and every compute dispatch with a bound image of that width, with
// its images and the first words of its small constant buffers (finding the upscale pass of a build).
[[nodiscard]] bool FindWide();
void FindLog(const char* kind, const PreparedBindings& bindings, TextureCache& cache, ImageId output,
             uint32_t color_count);

} // namespace Libs::Graphics::MetalFx

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_METALFX_H_
