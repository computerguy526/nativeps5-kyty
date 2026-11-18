#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_FLIGHTRECDRAW_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_FLIGHTRECDRAW_H_

// KYTY_FLIGHTREC draw/dispatch events (common/flightRec.h). Every draw records EvDraw (shader
// hashes, index in the flip, scheduler tick). Draws/dispatches whose shader hash starts with a
// KYTY_FLIGHTREC_KEYS prefix are KEYED: while their bindings are resolved the buffer cache records
// the per-binding state before the lookup (EvBindState: CPU-dirty / GPU-modified / readback
// pending), the lookup result (EvBindResult), every upload or skip decision (EvUpload /
// EvSyncSkip); after the bindings, EvBind (guest range, BufferId, FNV hash of the guest bytes at
// bind, first KYTY_FLIGHTREC_HASH_KB KiB, default 1024), EvTexture and EvTarget (image ids). The
// guest ranges of keyed binds are hashed again on the next KYTY_FLIGHTREC_REHASH (default 3) flips:
// EvRehash = the guest bytes changed after Kyty consumed them.
//   KYTY_FLIGHTREC_KEYS=<hash prefix>[,..] ("helmet" = pixid-2's helmet pipelines)

#include "common/flightRec.h"

#include <cstdint>
#include <span>

namespace Libs::Graphics {
class TextureCache;
struct PreparedBindings;
struct RenderColorInfo;
} // namespace Libs::Graphics

namespace Libs::Graphics::FlightRecDraw {

namespace Detail {
inline thread_local bool t_keyed = false;
} // namespace Detail

// True on this thread while a keyed draw/dispatch resolves its bindings (buffer cache hooks).
[[nodiscard]] inline bool Keyed() noexcept {
	return Detail::t_keyed;
}

// Before the bindings are resolved. kind 'D' draw, 'X' dispatch.
void Begin(char kind, std::span<PreparedBindings* const> stages, uint64_t tick) noexcept;
// After them (records the keyed draw's inputs; clears the keyed state).
void End(TextureCache& cache, std::span<PreparedBindings* const> stages,
         std::span<const RenderColorInfo> colors) noexcept;

} // namespace Libs::Graphics::FlightRecDraw

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_FLIGHTRECDRAW_H_ */
