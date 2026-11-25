#ifndef GRAPHICS_HOST_GPU_RENDERER_PIPELINE_UPGRADECOMPLETION_H_
#define GRAPHICS_HOST_GPU_RENDERER_PIPELINE_UPGRADECOMPLETION_H_

// nps5 archfix (KYTY_UPGRADE_FLUSH_HELD, default 1): the order of the end of a cache replay
// (upgrade / resume / first start).
//
// Apple's Metal crashes (-[_MTLBinaryArchive newArchiverIdWithBinaryKey:...], NULL airScript) when
// a pipeline compile STARTS on the archive serializer's compiler while
// serializeAsArchiveAndFlushToURL runs (5-10 s). The completion save calls KK's
// nps5_kk_archive_flush; if the launch hold was released first, the game draws its first frame
// during that serialize and its first MetalFX scaler creation compiles inside the window (two
// crashes seen in the v1.4 RC playtests). Holding the game until flush + driver cache + stamp are
// written means no frame is drawn during the serialize. Costs a few seconds behind the loading card
// on a non-Warm start only; a Warm start saves nothing here and releases at once.
//
//   held (gate on):  [flush, save]  then release               (the caller has NOT released the
//   hold before) not held (off):  [flush, save]  and no release here        (the caller released
//   the hold before, the old order)
// `saves` is false for a Warm start, a stopped replay and a list prepare: nothing is written, the
// hold is still released when held. A failed save never keeps the hold: the next start resumes.
// Pure logic, no renderer dependencies (unit tested on its own: tests/UpgradeCompletionTests.cpp).

namespace Libs::Graphics {

template <class FlushFn, class SaveFn, class ReleaseFn>
inline void CompleteUpgradeReplay(bool held, bool saves, FlushFn&& flush, SaveFn&& save,
                                  ReleaseFn&& release) {
	if (saves) {
		flush();
		save();
	}
	if (held) release();
}

} // namespace Libs::Graphics

#endif
