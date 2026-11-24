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

namespace Libs::Graphics {
class CommandBuffer;
}

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

// mfxlayer (UE titles: SAROS PPSA07631; everything default off, no effect on other titles):
// KYTY_METALFX_FIND_ANCHOR=<hash> [KYTY_METALFX_FIND_ANCHOR_FRAMES=<n>, default 2] (needs
//   KYTY_METALFX_FIND=3; diagnostic): the last 64 passes FindLog saw before a dispatch of that
//   shader (e.g. TSR UpdateHistory) are printed as "MetalFX anchor: -<k> ..." with every image and
//   buffer, level 1 = shares an image with the anchor pass, 2 = shares one with a level-1 pass; the
//   small buffers of level-1 passes and of the anchor as hex words ("MetalFX anchor:   words"); and,
//   until the next flip, every later pass that binds one of the anchor's images ("+<k>": consumers).
//   Starts after KYTY_METALFX_FIND_RESET_S. FindLog only sees passes when FIND=3 is set.
// KYTY_CS_KO=<hash>[,<hash>...] (diagnostic knock-out: per-pass GPU cost without GPU timers): compute
//   dispatches of these shaders are not recorded (their outputs go stale: timing runs only).
//   KYTY_CS_KO_FROM_S=<s> (default 90) after the first dispatch; KYTY_CS_KO_CYCLE=<flips> alternates
//   off/on windows of that many flips (in-run ABAB; "CsKo cycle: flip <n> t_us <t> on|off" lines,
//   same clock as "MetalFX cycle:"); 0 = on all the time.
// KYTY_METALFX_COMPUTE_PASS=<hash>[:<in>:<out>]: the image indices of the colour and of the output
//   of the compute pass (default 0 and 1; SAROS TSR UpdateHistory baa4068f48160685:6:7).
// KYTY_METALFX_COMPUTE_REPLACE=1: MetalFX runs INSTEAD of that dispatch (which is not recorded)
//   when the upscale was recorded; any fallback leaves the game's dispatch.
// KYTY_METALFX_PASS=<hash>:blit[:<image index>]: a full-screen draw that scales one bound image to
//   a larger single colour target (UE's secondary upscale into the back buffer) is replaced by the
//   SPATIAL scaler; the colour is the widest bound image smaller than the target with its aspect.
//   KYTY_METALFX_PASS=auto:blit (experiments): any such draw by shape (target >= 2560 wide, exactly one
//   bound image > 8 texels, >= 960 wide, the target's aspect); "MetalFX blit: auto shader" names it.
// KYTY_METALFX_UE=log|fsr3 (metalfxUe.h): UE's third-party temporal upscaler slot with the FSR3 plugin
//   enabled by cvar (r.FidelityFX.FSR3.Enabled=1, dynamic resolution off). log: one "MetalFX ue:" line
//   per FSR3 shader (role, sizes, jitter), no behaviour change. fsr3 (needs KYTY_METALFX=temporal):
//   MetalFX temporal runs INSTEAD of FSR3's Accumulate dispatch (the last one: a render-size colour and
//   two or more upscale-size colour images; output = the highest-index one) with the scene depth and
//   the dilated RG16F motion vectors of the frame's FSR3 dispatches and the jitter of the block; the
//   first FSR3 dispatch of a frame (PrepareInputs: it writes the motion vectors) always runs, the
//   others are not recorded once 8 evaluations in a row succeeded (KYTY_METALFX_UE_SKIP=0 keeps
//   them); a refusal leaves FSR3's own picture and stops the skipping for 300 flips.
//   KYTY_METALFX_CYCLE=<flips>,gt alternates FSR3 (g) and the adapter (t).
//   Signs: KYTY_METALFX_MV_SIGN, KYTY_METALFX_JITTER_SIGN, KYTY_METALFX_JITTER_SIGN_Y.
// Returns 0: record the dispatch; 1: do not record it; 2: MetalFX was recorded in its place and the
// caller commits the GPU write of UeOutput().
[[nodiscard]] ImageId UeOutput();
[[nodiscard]] bool UeEnabled();
[[nodiscard]] int  UeDispatch(CommandBuffer& buffer, const PreparedBindings& bindings, TextureCache& cache);
[[nodiscard]] bool CsKnockoutEnabled();
[[nodiscard]] bool CsKnockout(uint64_t shader_hash);
[[nodiscard]] bool ComputeReplace(uint64_t shader_hash);

// KYTY_RES_AUDIT=1|2 (diagnostic, default off; independent of KYTY_METALFX): resolution audit for
// a forced internal resolution (720p patch of a title that only ever renders >= 1080p). Once per
// (shader, image extents, grid) signature, every draw or compute dispatch where a bound image or
// colour target has a 1080p-derived extent (1920x1080 / 2^k, k = 0..6, height rounded either way),
// or a direct dispatch whose thread grid (groups x threads per group) covers such an extent, is
// logged as "ResAudit:" with the shader hash, its guest code address, the grid, the target and
// every bound image (extent, format, guest address), plus flags: img1080/k (1080p-derived image),
// grid1080/k (grid covers 1920x1080/2^k), mixed (a 720p-derived image in the same call), grid>img
// (grid more than one group beyond every bound image). =2 logs every signature (census).
// KYTY_RES_AUDIT_FROM_S=<s>: start that many seconds after the first call (skip loading screens);
// KYTY_RES_AUDIT_MAX=<n> (default 4000) caps the lines. Base size: KYTY_RES_AUDIT_BASE=WxH
// (default 1920x1080), the internal one: KYTY_RES_AUDIT_INTERNAL=WxH (default 1280x720).
[[nodiscard]] bool ResAuditEnabled();
void ResAudit(const char* kind, const PreparedBindings& bindings, TextureCache& cache,
              ImageId output, uint32_t color_count, const uint32_t* groups, const uint32_t* threads,
              uint64_t code_address);

} // namespace Libs::Graphics::MetalFx

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_METALFX_H_
