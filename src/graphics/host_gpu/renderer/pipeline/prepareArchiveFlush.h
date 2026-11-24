#ifndef GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PREPAREARCHIVEFLUSH_H_
#define GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PREPAREARCHIVEFLUSH_H_

#include <cstdint>
#include <functional>
#include <string>

// nps5 prepfast S1: how a list prepare makes sure KosmicKrisp's pipeline archive is on disk before the process leaves with _Exit.
// Before: a fixed sleep of one flush period + 3 s (the flusher is a timer thread). With KYTY_PREPARE_FLUSH_SYNC=1 and a driver
// that exports nps5_kk_archive_flush, one call writes the pending segment (or, in compact mode, the compacted one) and returns when
// it is on disk. Any failure (no symbol, lock not taken in time, no segment archive) falls back to the sleep, so the gate can
// never lose pipelines. Pure logic, no renderer dependencies (unit tested on its own).

namespace Libs::Graphics {

using ArchiveFlushFn = int64_t (*)(uint32_t wait_ms);

struct PrepareFlushOutcome {
	bool        archive = false; // KK_PIPELINE_ARCHIVE_DIR was set: there was something to wait for
	bool        synced  = false; // the export wrote it and returned
	int64_t     written = 0;     // pipelines the export wrote
	double      slept_s = 0.0;   // seconds slept (0 when synced)
	std::string note;            // why the sleep was used although the gate was on
};

// sleep: called at most once with the seconds to wait. flush may be null (symbol missing).
inline PrepareFlushOutcome PrepareArchiveWait(bool archive_dir_set, bool sync_gate, ArchiveFlushFn flush, uint32_t wait_ms,
                                              double flush_period_s, const std::function<void(double)>& sleep) {
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
			out.note = r == -1 ? "the flush could not take the compile lock in time" : "the driver has no segment archive";
		}
	}
	out.slept_s = flush_period_s + 3.0;
	sleep(out.slept_s);
	return out;
}

} // namespace Libs::Graphics

#endif
