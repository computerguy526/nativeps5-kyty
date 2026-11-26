#ifndef GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PREPAREARCHIVEFLUSH_H_
#define GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PREPAREARCHIVEFLUSH_H_

#include <cstdint>
#include <functional>
#include <string>

// nps5 prepfast S1: how a list prepare makes sure KosmicKrisp's pipeline archive is on disk before
// the process leaves with _Exit. Before: a fixed sleep of one flush period + 3 s (the flusher is a
// timer thread). With KYTY_PREPARE_FLUSH_SYNC=1 and a driver that exports nps5_kk_archive_flush,
// one call writes the pending segment (or, in compact mode, the compacted one) and returns when it
// is on disk. Any failure (no symbol, lock not taken in time, no segment archive) falls back to the
// sleep, so the gate can never lose pipelines. Pure logic, no renderer dependencies (unit tested on
// its own).

namespace Libs::Graphics {

using ArchiveFlushFn = int64_t (*)(uint32_t wait_ms);

struct PrepareFlushOutcome {
	bool        archive = false; // KK_PIPELINE_ARCHIVE_DIR was set: there was something to wait for
	bool        synced  = false; // the export wrote it and returned
	int64_t     written = 0;     // pipelines the export wrote
	double      slept_s = 0.0;   // seconds slept (0 when synced)
	std::string note;            // why the sleep was used although the gate was on
};

// v1.4 review M3: the Vulkan loader opens the KosmicKrisp ICD RTLD_LOCAL, so dlsym(RTLD_DEFAULT,
// ...) never finds its exports. `global` is the RTLD_DEFAULT lookup, `from_images` walks the loaded
// images and resolves the symbol from the handle of the KosmicKrisp one (shaders.cpp KkTag,
// metalfx.cpp do the same with _dyld_image_*). `via` (optional) receives which path found it.
inline void* ResolveDriverSymbol(const char* name, const std::function<void*(const char*)>& global,
                                 const std::function<void*(const char*)>& from_images,
                                 const char**                             via = nullptr) {
	const char* path = "not found";
	void*       sym  = global ? global(name) : nullptr;
	if (sym != nullptr) {
		path = "global namespace";
	} else if (from_images) {
		sym = from_images(name);
		if (sym != nullptr) path = "driver image handle";
	}
	if (via != nullptr) *via = path;
	return sym;
}

// KYTY_PREPARE_FLUSH_WAIT_MS: the lock wait handed to the driver; a uint32 cast used to wrap large
// values and 0 meant "forever".
inline uint32_t ClampFlushWaitMs(uint64_t ms) {
	return ms < 1 ? 1u : (ms > 0xFFFFFFFFull ? 0xFFFFFFFFu : static_cast<uint32_t>(ms));
}

// sleep: called at most once with the seconds to wait. flush may be null (symbol missing).
inline PrepareFlushOutcome PrepareArchiveWait(bool archive_dir_set, bool sync_gate,
                                              ArchiveFlushFn flush, uint32_t wait_ms,
                                              double                             flush_period_s,
                                              const std::function<void(double)>& sleep) {
	PrepareFlushOutcome out;
	out.archive = archive_dir_set;
	if (!archive_dir_set) return out;
	if (sync_gate) {
		if (flush == nullptr) {
			out.note = "the driver has no nps5_kk_archive_flush";
		} else {
			const int64_t r = flush(wait_ms);
			if (r >= 0) {
				out.synced  = true;
				out.written = r;
				return out;
			}
			out.note = r == -1 ? "the flush could not take the compile lock in time"
			                   : "the driver has no segment archive";
		}
	}
	out.slept_s = flush_period_s + 3.0;
	sleep(out.slept_s);
	return out;
}

// Window-close exit (v1.4 rc12 finding: a window close ends the process without KosmicKrisp's exit
// flush, so every pipeline compiled in the session was lost: "segments written 0", the next start
// compiled them all again). SaveForClose calls the exported flush once. Returns the pipelines
// written, -2 when the driver has no export, otherwise the driver's negative code (-1 lock timeout,
// -3 no archive).
inline int64_t CloseArchiveFlush(ArchiveFlushFn flush, uint32_t wait_ms) {
	return flush == nullptr ? -2 : flush(wait_ms);
}

} // namespace Libs::Graphics

#endif
