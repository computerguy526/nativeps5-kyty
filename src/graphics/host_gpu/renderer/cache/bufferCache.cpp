#include "graphics/host_gpu/renderer/cache/bufferCache.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "common/debugCounters.h"
#include "common/guestAddress.h" // nps5: guest bytes are at guest + bias
#include "common/guestLoad.h"
#include "common/hangTrace.h"
#include "common/hangWatchdog.h"
#include "common/liveSwitch.h"
#include "common/logging/log.h"
#include "common/onceValue.h"
#include "common/profiler.h"
#include "common/rendererBatch.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/cleanVerdictCache.h"
#include "graphics/host_gpu/faultCost.h"
#include "graphics/host_gpu/gpuBreadcrumbs.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/memoryStats.h"
#include "graphics/host_gpu/readFaultDiag.h"
#include "graphics/host_gpu/renderer/cache/bufferLookupStats.h"
#include "graphics/host_gpu/renderer/cache/faultClass.h"
#include "graphics/host_gpu/renderer/cache/memtrimUtil.h"
#include "graphics/host_gpu/renderer/cache/readFaultFrameBurst.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/cache/uploadDma.h"
#include "graphics/host_gpu/renderer/commandRecorder.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/cpCommit.h"
#include "graphics/host_gpu/renderer/drawPrep/commitStats.h"
#include "graphics/host_gpu/renderer/drawPrep/cpGaps.h"
#include "graphics/host_gpu/renderer/flightRecDraw.h"
#include "graphics/host_gpu/renderer/gpuOpProfiler.h"
#include "graphics/host_gpu/renderer/image/tilerTrace.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/syncEpoch.h"
#include "graphics/host_gpu/vramBudget.h"
#include "graphics/host_gpu/vramStats.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/host_gpu/watchdogSubmit.h"
#include "kernel/memory.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fmt/format.h>
#include <intrin.h>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(_MSC_VER) && !defined(__clang__) && (defined(_M_X64) || defined(_M_IX86))
#include <xmmintrin.h>
#endif

namespace Libs::Graphics {

namespace {

Live::Switch g_cpu_copy_page_skip("KYTY_CPU_COPY_PAGE_SKIP", Live::ParseDefaultOff);
Live::Switch g_shader_write_retick("KYTY_SHADER_WRITE_RETICK", Live::ParseDefaultOff);

constexpr uint64_t MiB           = 1024 * 1024;
constexpr uint64_t GdsBufferSize = 64 * 1024;

bool SmallUploadRingEnabled() {
	// Startup only: keep large requests correct through temporary upload buffers.
	const auto* flag = std::getenv("KYTY_RAM_SMALL_UPLOAD_RING");
	return flag != nullptr && std::strcmp(flag, "1") == 0;
}

uint64_t UploadRingSize() {
	return SmallUploadRingEnabled() ? 128 * MiB : 512 * MiB;
}

Live::Switch g_cpu_only_query("KYTY_CP_CPU_ONLY_QUERY", Live::ParseDefaultOff);
Live::Switch g_binding_memo_prefetch("KYTY_CP_BINDING_MEMO_PREFETCH", Live::ParseDefaultOff);
// KYTY_CP_BINDING_BATCH_PREFETCH (BufferCache::PrefetchReadBinding): 0 off, find (2) from
// FindBuffers, anything else (default) from RebindBuffers.
int64_t ParseBatchPrefetch(const char* value) {
	if (value != nullptr && std::strcmp(value, "find") == 0) {
		return 2;
	}
	return Live::ParseDefaultOn(value);
}
Live::Switch g_binding_batch_prefetch("KYTY_CP_BINDING_BATCH_PREFETCH", ParseBatchPrefetch);
std::atomic<uint64_t> g_binding_hot_generation {1};
void BindingHotChanged(int64_t, int64_t) {
	// The live registry serializes callbacks. Saturation makes lookups clear the tier each time.
	const auto generation = g_binding_hot_generation.load(std::memory_order_relaxed);
	if (generation != UINT64_MAX) {
		g_binding_hot_generation.store(generation + 1, std::memory_order_relaxed);
	}
}
Live::Switch g_binding_hot_memo("KYTY_CP_BINDING_HOT_MEMO", Live::ParseDefaultOff,
                               BindingHotChanged);
// 0 when unset, empty or "0"; 2 for "verify", 3 for "exit"; 1 otherwise.
int64_t ParseOnVerifyExit(const char* value) {
	if (value == nullptr || *value == '\0' || std::strcmp(value, "0") == 0) {
		return 0;
	}
	if (std::strcmp(value, "verify") == 0) {
		return 2;
	}
	return std::strcmp(value, "exit") == 0 ? 3 : 1;
}
// KYTY_BINDING_MEMO_TOUCH (BufferCache::BindingMemo).
Live::Switch g_binding_memo_touch("KYTY_BINDING_MEMO_TOUCH", Live::ParseDefaultOff);
// KYTY_BINDING_LEAN_MISS (BufferCache::TryLeanReadMiss): 0 (default) off, 1 on, verify (2), exit (3).
Live::Switch g_binding_lean_miss("KYTY_BINDING_LEAN_MISS", ParseOnVerifyExit);

// KYTY_FAULT_AHEAD_ADAPT (live): the fault-ahead window of write faults
// (MemoryTracker::SetFaultAheadOverride, applied at every guest flip).
//   auto   (default) 256 KiB, 512 KiB or 1 MiB by FaultCost::SlowLevel() 0 / 1 / 2: how slow
//          protection changes are on this PC (measured, only ever rising; Linux with mprotect
//          tracking: 1 MiB from the start)
//   <KiB>  a power of two, 8..4096
//   0      KYTY_FAULT_AHEAD_KB alone (32 KiB), as before
// At the Sky Garden the guest's job threads fill ~16 MB of triple-buffered data per frame. With
// 32 KiB windows that took ~1,700 faults per frame, 62% of them duplicates (another thread wrote
// the same page while the first fault was in flight). 256 KiB windows: ~300 faults and ~100
// loosening calls per frame instead of ~1,700 and ~620, for ~14% more upload bytes (pages a window
// opened that the guest did not write are uploaded once more than needed: bytes, not faults).
// 1 MiB here: ~110 faults, but ~72 us each in the handler and twice the upload bytes of 32 KiB, so
// only PCs with slow protection changes get it (the slow-PC simulation: 20.7 -> 26.4 fps).
// The handler's own time grows with the window, so it must not choose the window: an earlier rule
// based on it climbed to 1 MiB on this PC too.
Live::Switch g_fault_ahead_adapt("KYTY_FAULT_AHEAD_ADAPT", [](const char* value) -> int64_t {
	if (value == nullptr || value[0] == '\0' || std::strcmp(value, "auto") == 0) {
		return -2;
	}
	if (std::strcmp(value, "0") == 0) {
		return 0;
	}
	char*      end = nullptr;
	const auto kib = std::strtoll(value, &end, 10);
	if (end == value || kib < 8 || kib > 4096 || (kib & (kib - 1)) != 0) {
		return 0;
	}
	return kib;
});

uint32_t FaultAheadOverridePages() {
	const auto value = g_fault_ahead_adapt.Get();
	if (value == 0) {
		return 0;
	}
	uint64_t kib = static_cast<uint64_t>(value);
	if (value == -2) {
		kib = uint64_t {256} << std::clamp(FaultCost::SlowLevel(), 0, 2);
	}
	return static_cast<uint32_t>(kib * 1024 / TRACKER_PAGE_SIZE);
}

// KYTY_BDA_SYNC_PER_SUBMISSION=1|verify (default off; live): see BufferCache::SynchronizeBdaBuffers.
// 1: at most one BDA pass per guest submission while the BDA structure holds. verify: every pass the
// gate would skip runs anyway and its uploads are counted, i.e. the bytes the CPU wrote within the
// submission that 1 leaves for the next submission's pass. Both values only decide whether a pass
// runs now, and every pass records the epochs both read, so it can switch at any flip.
int64_t ParseBdaSyncPerSubmission(const char* value) {
	if (value == nullptr || value[0] == '\0' || std::strcmp(value, "0") == 0) {
		return 0;
	}
	return std::strcmp(value, "verify") == 0 ? 2 : 1;
}
Live::Switch g_bda_sync_per_submission("KYTY_BDA_SYNC_PER_SUBMISSION", ParseBdaSyncPerSubmission);

bool IncrementalBdaSyncEnabled() {
	const auto* value = std::getenv("KYTY_BDA_INCREMENTAL_SYNC");
	return value != nullptr && value[0] == '1' && value[1] == '\0';
}

// KYTY_BDA_HOT_SYNC=0 restores the U42 behaviour: any hot page disables incremental BDA
// synchronization (every pass scans every mapped buffer) and hot pages are snapshotted before
// their shadow compare. Only meaningful with KYTY_BDA_INCREMENTAL_SYNC=1.
bool BdaHotSyncEnabled() {
	const auto* value = std::getenv("KYTY_BDA_HOT_SYNC");
	return value == nullptr || !(value[0] == '0' && value[1] == '\0');
}

// KYTY_BDA_HOT_SYNC_VERIFY=1 follows every hot pass with the full scan it replaced and counts
// the CPU-dirty non-hot pages that scan finds while the pass's epochs still hold
// (BdaSyncHotVerifyMismatches, expected 0); "exit" stops on the first one. Diagnostic: it costs
// the full scans again.
int BdaHotSyncVerifyMode() {
	static const int mode = [] {
		const auto* value = std::getenv("KYTY_BDA_HOT_SYNC_VERIFY");
		if (value == nullptr || *value == '\0' || std::strcmp(value, "0") == 0) {
			return 0;
		}
		return std::strcmp(value, "exit") == 0 ? 2 : 1;
	}();
	return mode;
}

bool BdaHotSyncVerifyEnabled() {
	return BdaHotSyncVerifyMode() != 0;
}

// KYTY_BDA_DIRTY_LOG=0 restores the full scan of every mapped buffer whenever the fault epoch
// moved (bufferCache.h). Only meaningful with KYTY_BDA_INCREMENTAL_SYNC=1 and KYTY_BDA_HOT_SYNC.
bool BdaDirtyLogEnabled() {
	const auto* value = std::getenv("KYTY_BDA_DIRTY_LOG");
	return value == nullptr || !(value[0] == '0' && value[1] == '\0');
}

// KYTY_BDA_DIRTY_LOG_VERIFY=1 follows every dirty-log pass with the full scan it replaced and
// counts the CPU-dirty non-hot pages that scan finds while the pass's epochs still hold
// (BdaSyncLogVerifyMismatches, expected 0); "exit" stops on the first one. Diagnostic: it costs
// the full scans again.
int BdaDirtyLogVerifyMode() {
	const auto* value = std::getenv("KYTY_BDA_DIRTY_LOG_VERIFY");
	if (value == nullptr || *value == '\0' || std::strcmp(value, "0") == 0) {
		return 0;
	}
	return std::strcmp(value, "exit") == 0 ? 2 : 1;
}

// KYTY_UPLOAD_BATCH_SCOPED_FLUSH=0 restores recording every pending barrier at the end of each
// outermost UploadBatch scope, even when the scope queued no upload.
bool UploadBatchScopedFlushEnabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_UPLOAD_BATCH_SCOPED_FLUSH");
		return value == nullptr || !(value[0] == '0' && value[1] == '\0');
	}();
	return enabled;
}

// Guest read faults copy GPU-owned bytes on a side command buffer unless this is "0".
bool SideReadbackEnabled() {
	const auto* value = std::getenv("KYTY_READBACK_SIDE_COPY");
	return value == nullptr || !(value[0] == '0' && value[1] == '\0');
}

// GPU-thread reads (CP reads of GPU-written data, GPU-thread faults) also use side copies and wait
// for them in place, unless KYTY_READBACK_SIDE_GPU_THREAD is "0" (then they always drain).
bool SideReadbackGpuThreadEnabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_READBACK_SIDE_GPU_THREAD");
		return value == nullptr || !(value[0] == '0' && value[1] == '\0');
	}();
	return enabled;
}

// Aligned side-copy window in bytes: a power of two between 4 KiB and 1 MiB (default 64 KiB).
uint64_t SideReadbackWindow() {
	constexpr uint64_t Default = 64 * 1024;
	const auto*        value   = std::getenv("KYTY_READBACK_SIDE_COPY_WINDOW_KB");
	if (value == nullptr) {
		return Default;
	}
	char*      end = nullptr;
	const auto kib = std::strtoull(value, &end, 10);
	if (end == value || *end != '\0' || kib < 4 || kib > 1024 || (kib & (kib - 1)) != 0) {
		return Default;
	}
	return kib * 1024;
}

// KYTY_READBACK_MERGE_GAP_KB=<KiB> (live, default 0 = one region per range; 64 was the fork's): GPU-written ranges
// of one readback less than this far apart share one vkCmdCopyBuffer region (chenxiao07 9239c5773).
// The bytes between them are copied too but never written back: write-backs and publications keep
// the exact ranges. Default off: Astro Bot measured 14-18 readback commands of 20-32 ranges per flip
// (Sky Garden, snow, clock tower), which a 64 KiB gap cut to 14-20 regions while doubling the bytes
// copied at snow/clock tower (85 -> 176 KiB per flip); ours was one command per readback already.
Live::Switch g_readback_merge_gap("KYTY_READBACK_MERGE_GAP_KB", [](const char* value) -> int64_t {
	if (value == nullptr) {
		return 0;
	}
	return static_cast<int64_t>(std::min<uint64_t>(std::strtoull(value, nullptr, 10), 64 * 1024)) *
	       1024;
});

// KYTY_READBACK_FLUSH_SIDE (live, default 0; =1 on): a guest-thread read of
// GPU-written bytes that the unsubmitted recording wrote (or that an address-writing shader in it
// may have written) is served by a side copy instead of a drain. The command processor submits the
// recording (the drain submits it as well), which makes those bytes ones an already submitted
// recording wrote, issues the side copy that waits for it on the GPU, and returns; the guest thread
// then waits for the copy's publication, as for any side readback, while the command processor
// continues. The bytes, their publication and the protection rules are the side path's; only the
// command processor no longer waits for the GPU to finish everything queued before it serves the
// next op.
Live::Switch g_readback_flush_side("KYTY_READBACK_FLUSH_SIDE", Live::ParseDefaultOff);
// KYTY_GC_SIDE_SETTLE_LAZY (live, default 0; nps5 slowlvl): see BufferCache::RunGarbageCollector.
Live::Switch g_gc_side_settle_lazy("KYTY_GC_SIDE_SETTLE_LAZY", Live::ParseDefaultOff);
std::atomic<uint64_t> g_gc_settle_deferred_passes {0};
// nps5 cpcommit2, KYTY_GC_SETTLE_SKIP_EMPTY (live, default off in cpuframe; =1 on): a collection
// pass with no buffer old enough to look at returns before it settles the pending side readbacks.
// Settling is only the preparation of the pass's ownership checks and downloads
// (RetireUnusedBuffers does the same: candidates first); without a candidate it was a wait for the
// GPU and nothing else.
Live::Switch g_gc_settle_skip_empty("KYTY_GC_SETTLE_SKIP_EMPTY", Live::ParseDefaultOff);
// KYTY_GC_SIDE_SETTLE_LAZY_CRITICAL=<passes> (live, default 0 = off in cpuframe; 16 = cpcommit2
// default: every pass waits as before; vramBudget.h PlanBufferPassLazyCritical): at or above the
// critical mark a pass that would wait for the GPU is put off at most that many times in a row (16
// = about two frames of completed submissions in a level).
int64_t ParseGcLazyCritical(const char* value) {
	if (value == nullptr || value[0] == '\0') {
		return 0;
	}
	const auto passes = std::strtoll(value, nullptr, 10);
	return std::clamp<int64_t>(
	    passes, 0, static_cast<int64_t>(VramBudget::BufferSettleMaxDeferredCriticalLimit));
}
Live::Switch g_gc_side_settle_lazy_critical("KYTY_GC_SIDE_SETTLE_LAZY_CRITICAL",
                                            ParseGcLazyCritical);
// nps5 cpuframe, KYTY_GC_SETTLE_SELECTIVE (live, default ON since v1.1; =0 restores the full
// settle): a collection pass never settles all pending side readbacks (a wait for the GPU on the
// command-processor thread after nearly every completed submission in a level: 26-27 % of that
// thread in Sky Garden). It skips the candidate buffers whose pages a pending side readback covers
// instead; they are looked at again by a later pass. The other candidates' tracker state does not
// depend on those readbacks. v1.0.3 kept it off after a helmet flicker under VRAM pressure; with
// the v1.1 snapshot fix (KYTY_SNAPSHOT_CLEAN_READS) the crash-site flicker checks and the
// player's eye check are clean. While it is on, KYTY_GC_SIDE_SETTLE_LAZY_CRITICAL has no effect.
Live::Switch g_gc_settle_selective("KYTY_GC_SETTLE_SELECTIVE", Live::ParseDefaultOn);
// "BufferGc 10s" line (with KYTY_CP_GAP_STATS): what the collection passes did. GPU thread only.
struct BufferGcStats {
	uint64_t passes = 0, below = 0, empty = 0, deferred = 0, ran = 0, waited = 0, wait_ns = 0,
	         retired = 0, downloads = 0, critical = 0, busy = 0;
	std::chrono::steady_clock::time_point last {};
};
BufferGcStats g_buffer_gc_stats;
void          PrintBufferGcStats() {
	auto& st = g_buffer_gc_stats;
	if (!CpGaps::Enabled() || (st.passes & 63u) != 0) {
		return;
	}
	const auto now = std::chrono::steady_clock::now();
	if (st.last.time_since_epoch().count() == 0) {
		st.last = now;
		return;
	}
	if (now - st.last < std::chrono::seconds(10)) {
		return;
	}
	std::printf(
	    "BufferGc 10s: passes %llu (critical %llu) | below trigger %llu, no candidate %llu, put "
	    "off %llu, ran %llu | settle waited %llu times, %.1f ms | retired %llu, downloads %llu, "
	    "skipped (pending readback) %llu\n",
	    static_cast<unsigned long long>(st.passes), static_cast<unsigned long long>(st.critical),
	    static_cast<unsigned long long>(st.below), static_cast<unsigned long long>(st.empty),
	    static_cast<unsigned long long>(st.deferred), static_cast<unsigned long long>(st.ran),
	    static_cast<unsigned long long>(st.waited), static_cast<double>(st.wait_ns) * 1e-6,
	    static_cast<unsigned long long>(st.retired), static_cast<unsigned long long>(st.downloads),
	    static_cast<unsigned long long>(st.busy));
	std::fflush(stdout);
	st      = {};
	st.last = now;
}
std::atomic<uint64_t> g_readback_flush_side_copies {0};

// KYTY_READBACK_SIDE_WRITES (live, default on; =0 drains as before): a write fault on a GPU-owned
// page, taken by the exception handler on a thread other than the command processor (so that the
// faulting instruction runs again), takes the side path as well, instead of a drain. The command
// processor issues the side copy of the page's GPU-written bytes (submitting the current recording
// first when it wrote them, as KYTY_READBACK_FLUSH_SIDE does for reads) and returns; the faulting
// thread waits for the copy's publication, which leaves the page clean and write-protected, and
// returns from the fault without the CPU-dirty transition. The write then faults once more, now on
// a clean page, and that fault makes the page CPU-dirty and writable on the faulting thread, as any
// write fault does (a GPU writer recorded meanwhile makes it GPU-owned again: the second fault
// starts over). The drain made the page CPU-dirty on the command processor right after waiting for
// the GPU; the bytes the write sees and the order against newer writers are the same. Other
// CPU-dirty transitions (emulator writes reported through InvalidateMemory, direct HandleFault
// calls) still drain: nothing faults again after them.
Live::Switch g_readback_side_writes("KYTY_READBACK_SIDE_WRITES", Live::ParseDefaultOn);

// KYTY_READBACK_EAGER_WRITES (live, default off; =1 on; upstream a3458ee7, eager-writes part only):
// a guest write fault on GPU-owned bytes makes its pages eager-hot as a read fault does
// (KYTY_READBACK_EAGER). The next recordings that write them then end with a copy of their bytes,
// published when the recording completes, so a later write fault finds the page clean (an
// ordinary write fault, no command-processor round trip) or waits only for that copy. With it,
// raise KYTY_READBACK_EAGER_PAGES (default 64; up to 2048).
Live::Switch          g_readback_eager_writes("KYTY_READBACK_EAGER_WRITES", Live::ParseDefaultOff);
std::atomic<uint64_t> g_readback_side_writes_copies {0};

// KYTY_READ_FAULT_PUBLISHED (live, default off; =1 on, =verify): a guest thread's read fault on a
// page whose bytes the GPU owns does not wait for the GPU. When the faulting instruction is a
// plain load (common/guestLoad.h) of bytes within one page, no image owns the page, and a side or
// eager readback published the page at most KYTY_READ_FAULT_PUBLISHED_FRAMES guest frames ago
// (default 2), the load completes in the handler with the page's backing bytes: the GPU-written
// bytes as last published, the other bytes as they are. The page stays protected (no access) for
// everyone else, so the command processor and the emulator still synchronize through their own
// faults and later writes still fault; the read marks the page read-hot, so its eager copy
// publishes the new bytes once their writer completes. Other reads take the waiting path. This is
// what a CPU sees that reads GPU-written memory without waiting for the writer: legitimate only for
// reads the game does not order after the writer (with KYTY_LABEL_MODE=record, Kyty cannot tell,
// since end-of-pipe labels are written when recorded). Astro Bot's render threads read a GPU-written
// value this way every frame (eboot 0x736fa9e).
// =verify takes the waiting path for every read, and compares the bytes the load would have been
// served with those it finds after the wait (equal / different counts, the first differences).
// One "Kyty read-fault published" line every 10 s while on.
int64_t ParseReadFaultPublished(const char* value) {
	if (value == nullptr || std::strcmp(value, "0") == 0) {
		return 0;
	}
	return std::strcmp(value, "verify") == 0 ? 2 : 1;
}
Live::Switch g_read_fault_published("KYTY_READ_FAULT_PUBLISHED", ParseReadFaultPublished);

// nps5: KYTY_READ_FAULT_PUBLISHED_BURST (default 8; 0 = never): the number of consecutive served
// loads of one page by one thread, each within 1 ms of the previous, after which the next one waits
// for the page's publication instead (a copy loop would otherwise fault on every load). A wait only
// pays when the page then stays readable: for a page the GPU writes again every frame (a GPU
// simulation the game samples from worker threads) each wait costs milliseconds and buys nothing.
// nps5: KYTY_READ_FAULT_PUBLISHED_FRAME_BURST (default 0 = off; 48 is the tested value): the number of loads
// served per tracker page per guest frame (all threads together) after which the next one waits for
// the page's publication (readFaultFrameBurst.h). Independent of the per-thread burst above, which
// stays as configured. Live-safe: read once.
uint32_t ReadFaultPublishedFrameBurst() {
	static const uint32_t limit = [] {
		const auto* value = std::getenv("KYTY_READ_FAULT_PUBLISHED_FRAME_BURST");
		return value != nullptr ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10)) : 0u;
	}();
	return limit;
}

ReadFaultFrameBurst::PageFrameCounter& ReadFaultFrameCounter() {
	static ReadFaultFrameBurst::PageFrameCounter counter;
	return counter;
}

// nps5 faultwin: KYTY_READ_RELAX=<n> (default 0 = off; read once). A tracker page whose n-th load
// in one guest frame was served from its published bytes (KYTY_READ_FAULT_PUBLISHED) drops its read
// watch: it stays GPU-dirty and write-watched, and guest loads then read the published bytes
// directly, without a fault, until the page's next GPU ownership change (a newer writer re-arms the
// watch; the eager copy publishing it makes it readable anyway). Same bytes a served load returns
// (what a CPU reading GPU memory without waiting sees), without the fault per load and without the
// burst wait (KYTY_READ_FAULT_PUBLISHED_FRAME_BURST) for the publication. Only plain loads of
// buffer pages (no image) published recently qualify, as for serving. Diagnostic counters in the
// "Kyty read-fault published" line.
// Live switch: turning it off leaves relaxed pages until their next GPU transition.
Live::Switch g_read_relax("KYTY_READ_RELAX", [](const char* value) -> int64_t {
	return value != nullptr ? static_cast<int64_t>(
	                              std::min<unsigned long>(std::strtoul(value, nullptr, 10), 65535))
	                        : 0;
});
uint32_t     ReadRelaxAfter() {
	return static_cast<uint32_t>(g_read_relax.Get());
}

uint64_t ReadRelaxSpan() {
	static const uint64_t span = [] {
		const auto* value = std::getenv("KYTY_READ_RELAX_SPAN_KB");
		const auto  kib   = value != nullptr ? std::strtoull(value, nullptr, 10) : 16ull;
		return kib >= 4 && kib <= 1024 && (kib & (kib - 1)) == 0 ? kib * 1024 : 16 * 1024ull;
	}();
	return span;
}

ReadFaultFrameBurst::PageFrameCounter& ReadRelaxCounter() {
	static ReadFaultFrameBurst::PageFrameCounter counter;
	return counter;
}

// nps5 zerocopy4: KYTY_ZEROCOPY_READ_OPEN (live, default off; needs KYTY_READ_FAULT_PUBLISHED). A
// buffer page whose load a guest thread was just served from its published bytes becomes READ-OPEN:
// its read watch stays off across new GPU writers (MemoryTracker::OpenRead), so later guest loads
// read the page's backing directly and take no fault at all, as long as readbacks keep publishing
// the page. What a load of an open page returns is what a served load returns: the GPU-written
// bytes as last published (now by the GPU copying them straight into the guest pages,
// ZeroCopy::DirectReadback, or by a staged publication), the other bytes as they are. The page
// stays GPU-dirty and write-watched: guest writes, the command processor and every other library
// reader synchronize as before (they ask the tracker; they never relied on the read watch). Once
// per guest frame (GPU thread, BufferCache::IssueEagerReadbacks) every open page is checked:
//  - not published within KYTY_READ_FAULT_PUBLISHED_FRAMES frames, an image took the page, the
//  switch
//    went off, or it has been open for KYTY_ZEROCOPY_READ_OPEN_FRAMES (default 600) frames: closed
//    (GPU-dirty pages are access-watched again; a page still read re-opens at its next served
//    load);
//  - else it is kept read-hot (its eager copy keeps publishing the GPU's new bytes), and the other
//    tracker pages of its host page (KYTY_READ_RELAX_SPAN_KB, default 16 KiB) are opened too once
//    they are published recently, or made read-hot when the GPU owns them unpublished: a host page
//    faults for all of its tracker pages while one of them is access-watched.
// So no load of an open page sees bytes older than a served load could (publication at most
// KYTY_READ_FAULT_PUBLISHED_FRAMES + 1 frames old: the check runs once per frame).
Live::Switch g_read_open("KYTY_ZEROCOPY_READ_OPEN", [](const char* value) -> int64_t {
	return value != nullptr && value[0] != '0' && value[0] != 0 ? 1 : 0;
});

uint32_t ReadOpenMaxFrames() {
	static const uint32_t frames = [] {
		const auto* value = std::getenv("KYTY_ZEROCOPY_READ_OPEN_FRAMES");
		return value != nullptr ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10)) : 600u;
	}();
	return frames;
}

struct ReadOpenState {
	std::mutex mutex;
	std::unordered_map<uint64_t, uint32_t>
	                      pages; // open tracker page -> guest frame it was opened in
	std::atomic<uint64_t> opened {0};
	std::atomic<uint64_t> neighbours {0};
	std::atomic<uint64_t> promoted {0};
	std::atomic<uint64_t> closed_stale {0};
	std::atomic<uint64_t> closed_image {0};
	std::atomic<uint64_t> closed_expired {0};
	std::atomic<uint64_t> closed_off {0};
	std::atomic<uint32_t> sweep_frame {0}; // written by the GPU thread
	std::atomic<uint64_t> closed_unchecked {0};
	uint64_t              report_ns = 0; // GPU thread
};
ReadOpenState& ReadOpen() {
	static ReadOpenState state;
	return state;
}

// nps5 faultclass: KYTY_FAULT_CLASS (live, default off; needs KYTY_READ_FAULT_PUBLISHED and eager
// readbacks, like KYTY_ZEROCOPY_READ_OPEN). The learned read class per tracker page (faultClass.h):
// a page read-faulted in KYTY_FAULT_CLASS_FRAMES (3) distinct guest frames becomes read-open at its
// next served load (KYTY_ZEROCOPY_READ_OPEN opens at the FIRST served load of any page); a page
// read once keeps the waiting path. An open page whose guest stores wait for unpublished GPU bytes
// KYTY_FAULT_CLASS_CONFLICTS (3) times is closed and stays exact for 600+ frames. Every other rule
// of read-open pages applies unchanged (per-frame check, closing, the bytes a load sees).
// KYTY_FAULT_CLASS_PROFILE=<file>: pages that were open in an earlier run start read-hot and are
// opened by the per-frame check once published, without a first fault; KYTY_FAULT_CLASS_PROFILE_OUT
// =<file> (default: the same file) is rewritten with this run's open pages every 10 s.
// KYTY_FAULT_CLASS=0 (live) closes every page at the next per-frame check, unless
// KYTY_ZEROCOPY_READ_OPEN is on. One "Kyty fault class 10s" line while on.
Live::Switch g_fault_class("KYTY_FAULT_CLASS", Live::ParseDefaultOff);

struct FaultClassState {
	explicit FaultClassState(FaultClass::Limits limits): table(limits) {}
	FaultClass::Table     table;
	std::atomic<uint64_t> store_conflict_ns {0};
	std::atomic<uint64_t> seed_hot {0};
	std::string           profile_out;
	uint64_t              report_ns = 0; // GPU thread
	size_t                saved     = 0; // GPU thread
};
uint32_t FaultClassEnvU32(const char* name, uint32_t fallback) {
	const auto* value = std::getenv(name);
	return value != nullptr && value[0] != 0
	           ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10))
	           : fallback;
}
FaultClassState& FaultClasses() {
	static FaultClassState* state = [] {
		FaultClass::Limits limits;
		limits.promote_frames =
		    std::max(1u, FaultClassEnvU32("KYTY_FAULT_CLASS_FRAMES", limits.promote_frames));
		limits.window_frames =
		    std::max(1u, FaultClassEnvU32("KYTY_FAULT_CLASS_WINDOW", limits.window_frames));
		limits.demote_conflicts =
		    std::max(1u, FaultClassEnvU32("KYTY_FAULT_CLASS_CONFLICTS", limits.demote_conflicts));
		limits.exact_frames =
		    std::max(1u, FaultClassEnvU32("KYTY_FAULT_CLASS_EXACT_FRAMES", limits.exact_frames));
		auto*       made   = new FaultClassState(limits);
		const auto* in     = std::getenv("KYTY_FAULT_CLASS_PROFILE");
		const auto* out    = std::getenv("KYTY_FAULT_CLASS_PROFILE_OUT");
		size_t      loaded = 0;
		if (in != nullptr && in[0] != 0) {
			loaded = FaultClass::LoadProfile(in, TRACKER_PAGE_SIZE,
			                                 FaultClassEnvU32("KYTY_FAULT_CLASS_SEEDS", 4096),
			                                 made->table);
		}
		made->profile_out = out != nullptr ? out : in != nullptr ? in : "";
		std::printf("Kyty fault class (KYTY_FAULT_CLASS): late after %u read frames in %u, exact "
		            "after %u store "
		            "conflicts for %u frames; profile: %zu seed pages from %s, written to %s\n",
		            limits.promote_frames, limits.window_frames, limits.demote_conflicts,
		            limits.exact_frames, loaded, in != nullptr && in[0] != 0 ? in : "(none)",
		            made->profile_out.empty() ? "(none)" : made->profile_out.c_str());
		return made;
	}();
	return *state;
}

// nps5 zerocopy4: KYTY_ZC_UPLOAD_CLASSIFY=<n> (diagnostic, default off; read once): classifies the
// CPU->GPU upload traffic by what an in-place (zero-copy) upload would need. Every n-th tracker
// page (by address) that an upload copies is hashed and compared with its previous upload:
//  first      no previous upload of the page seen;
//  unchanged  same bytes as the previous upload (fault-ahead, false sharing, rewritten equal);
//  settled    changed, and the recording of the previous upload had completed on the GPU;
//  in flight  changed while the recording that took the previous snapshot had not completed: GPU
//  work
//             recorded against the old bytes would read the new ones in place.
// One "Kyty upload classes" line every 10 s (pages and MiB, scaled by n).
struct UploadClassState {
	struct Seen {
		uint64_t hash = 0;
		uint64_t tick = 0;
	};
	std::unordered_map<uint64_t, Seen> pages; // GPU thread
	uint64_t                           count[4] {};
	uint64_t                           hot_pages = 0;
	uint64_t                           report_ns = 0;
};
UploadClassState& UploadClass() {
	static UploadClassState state;
	return state;
}
uint32_t UploadClassifyEvery() {
	static const uint32_t every = [] {
		const auto* value = std::getenv("KYTY_ZC_UPLOAD_CLASSIFY");
		return value != nullptr ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10)) : 0u;
	}();
	return every;
}
uint64_t HashPage(const uint8_t* bytes) noexcept {
	uint64_t hash = 0x9e3779b97f4a7c15ull;
	for (uint32_t i = 0; i < TRACKER_PAGE_SIZE; i += 8) {
		uint64_t word;
		std::memcpy(&word, bytes + i, 8);
		hash = (hash ^ word) * 0xff51afd7ed558ccdull;
		hash ^= hash >> 29;
	}
	return hash;
}

thread_local int t_write_drain_kind = 1; // nps5 faultwin: see ReadFaultPublishedState::write_drains

struct BurstWaitTimer {
	uint64_t start = 0;
	~BurstWaitTimer();
};

uint32_t ReadFaultPublishedBurst() {
	static const uint32_t burst = [] {
		const auto* value = std::getenv("KYTY_READ_FAULT_PUBLISHED_BURST");
		return value != nullptr ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10)) : 8u;
	}();
	return burst;
}

// nps5: KYTY_READ_FAULT_PUBLISHED_STRADDLE (default on; =0 off): a plain load whose bytes lie in
// two tracker pages is served too when each of the two pages is either plain CPU memory (no GPU
// owner: the backing holds its bytes) or qualifies like the faulting page.
bool ReadFaultPublishedStraddle() {
	static const bool on = [] {
		const auto* value = std::getenv("KYTY_READ_FAULT_PUBLISHED_STRADDLE");
		return value == nullptr || value[0] != '0';
	}();
	return on;
}

uint32_t ReadFaultPublishedFrames() {
	static const uint32_t frames = [] {
		const auto* value = std::getenv("KYTY_READ_FAULT_PUBLISHED_FRAMES");
		return value != nullptr ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10)) : 2u;
	}();
	return frames;
}

struct ReadFaultPublishedState {
	std::mutex                             mutex;
	std::unordered_map<uint64_t, uint32_t> published_frame; // tracker page -> guest frame
	std::atomic<uint64_t>                  served {0};
	std::atomic<uint64_t>                  undecodable {0};
	std::atomic<uint64_t>                  image {0};
	std::atomic<uint64_t>                  stale {0};
	std::atomic<uint64_t>                  other {0};
	std::atomic<uint64_t>                  burst {0};
	std::atomic<uint64_t>                  straddled {0};
	// nps5 faultwin: time guest threads spent in ReadMemory after a burst decision (the wait for
	// the page's publication), and pages KYTY_READ_RELAX made readable.
	std::atomic<uint64_t> burst_ns {0};
	std::atomic<uint64_t> burst_max_ns {0};
	std::atomic<uint64_t> relaxed_pages {0};
	std::atomic<uint64_t> relax_calls {0};
	// nps5 faultwin: write readbacks served by a drain, by kind: 0 explicit invalidation (not a
	// fault: host API, file reads, emulator), 1 fault not retried, 2 retried fault whose side copy
	// could not be issued, 3 on the GPU thread; count and wall time of the caller.
	std::atomic<uint64_t>                  write_drains[4] {};
	std::atomic<uint64_t>                  write_drain_ns[4] {};
	std::atomic<uint64_t>                  verify_equal {0};
	std::atomic<uint64_t>                  verify_different {0};
	std::atomic<uint64_t>                  report_ns {0};
	std::atomic<uint32_t>                  report_frame {0};
	std::atomic<uint32_t>                  logged {0};
};
ReadFaultPublishedState& ReadFaultPublished() {
	static ReadFaultPublishedState state;
	return state;
}

// While the switch is on: the guest frame in which a readback last published each page of `ranges`.
void NoteReadFaultPublication(std::span<const GuestRange> ranges, uint32_t frame) {
	auto&           state = ReadFaultPublished();
	std::lock_guard lock(state.mutex);
	for (const auto& range: ranges) {
		const auto last = range.address + range.size;
		for (auto page = Common::AlignDown(range.address, TRACKER_PAGE_SIZE); page < last;
		     page += TRACKER_PAGE_SIZE) {
			state.published_frame[page] = frame;
		}
	}
	if (state.published_frame.size() > 8192) {
		std::erase_if(state.published_frame,
		              [frame](const auto& entry) { return frame - entry.second > 600; });
	}
}

[[nodiscard]] bool ReadFaultPublishedRecently(uint64_t page, uint32_t frame) {
	auto&           state = ReadFaultPublished();
	std::lock_guard lock(state.mutex);
	const auto      found = state.published_frame.find(page);
	return found != state.published_frame.end() &&
	       frame - found->second <= ReadFaultPublishedFrames();
}

void ReportReadFaultPublished(uint32_t frame) {
	auto&      state = ReadFaultPublished();
	const auto now   = ReadFaultDiag::NowNs();
	auto       last  = state.report_ns.load(std::memory_order_relaxed);
	if (last == 0) {
		state.report_ns.compare_exchange_strong(last, now);
		state.report_frame.store(frame, std::memory_order_relaxed);
		return;
	}
	if (now - last < 10'000'000'000ull || !state.report_ns.compare_exchange_strong(last, now)) {
		return;
	}
	const auto frames = std::max<uint32_t>(1, frame - state.report_frame.exchange(frame));
	const auto take   = [](std::atomic<uint64_t>& counter) { return counter.exchange(0); };
	const auto served = take(state.served);
	std::printf(
	    "Kyty read-fault published 10s: %u frames, served %" PRIu64 " (%.2f/frame); waited: "
	    "not a plain load %" PRIu64 ", image %" PRIu64 ", not published recently %" PRIu64
	    ", other %" PRIu64 ", copy burst %" PRIu64 "; verify: equal %" PRIu64 ", different %" PRIu64
	    "; served across two pages %" PRIu64 "; burst waits %.1f ms total (max %.2f ms); "
	    "relaxed %" PRIu64 " pages (%" PRIu64 " calls)\n",
	    frames, served, static_cast<double>(served) / frames, take(state.undecodable),
	    take(state.image), take(state.stale), take(state.other), take(state.burst),
	    take(state.verify_equal), take(state.verify_different), take(state.straddled),
	    static_cast<double>(take(state.burst_ns)) / 1e6,
	    static_cast<double>(take(state.burst_max_ns)) / 1e6, take(state.relaxed_pages),
	    take(state.relax_calls));
	std::printf(
	    "Kyty write drains 10s: invalidate %" PRIu64 " (%.1f ms), fault-noretry %" PRIu64
	    " (%.1f ms), side-failed %" PRIu64 " (%.1f ms), gpu-thread %" PRIu64 " (%.1f ms)\n",
	    take(state.write_drains[0]), static_cast<double>(take(state.write_drain_ns[0])) / 1e6,
	    take(state.write_drains[1]), static_cast<double>(take(state.write_drain_ns[1])) / 1e6,
	    take(state.write_drains[2]), static_cast<double>(take(state.write_drain_ns[2])) / 1e6,
	    take(state.write_drains[3]), static_cast<double>(take(state.write_drain_ns[3])) / 1e6);
}

struct WriteDrainTimer {
	int      kind  = 0;
	uint64_t start = ReadFaultDiag::NowNs();
	~WriteDrainTimer() {
		auto& state = ReadFaultPublished();
		state.write_drains[kind].fetch_add(1, std::memory_order_relaxed);
		state.write_drain_ns[kind].fetch_add(ReadFaultDiag::NowNs() - start,
		                                     std::memory_order_relaxed);
	}
};

BurstWaitTimer::~BurstWaitTimer() {
	const auto ns    = ReadFaultDiag::NowNs() - start;
	auto&      state = ReadFaultPublished();
	state.burst_ns.fetch_add(ns, std::memory_order_relaxed);
	auto max = state.burst_max_ns.load(std::memory_order_relaxed);
	while (ns > max &&
	       !state.burst_max_ns.compare_exchange_weak(max, ns, std::memory_order_relaxed)) {
	}
}

// The number of regions copies (source ascending) give when a copy starting at most `gap` bytes
// after the previous region's source end joins it.
uint64_t CountMergedRegions(std::span<const vk::BufferCopy> copies, uint64_t gap) {
	uint64_t count = 0;
	uint64_t begin = 0;
	uint64_t end   = 0;
	for (const auto& copy: copies) {
		if (count == 0 || copy.srcOffset < begin || copy.srcOffset > end + gap) {
			count++;
			begin = copy.srcOffset;
			end   = copy.srcOffset + copy.size;
		} else {
			end = std::max(end, copy.srcOffset + copy.size);
		}
	}
	return count;
}

// KYTY_READBACK_REGION_LOG=1: every 10 s, the readback copy commands, their ranges and regions
// (and the regions a 64 KiB gap would give) per CP flip.
void RecordReadbackRegions(std::span<const vk::BufferCopy> ranges,
                           std::span<const vk::BufferCopy> regions) {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_READBACK_REGION_LOG");
		return value != nullptr && value[0] == '1';
	}();
	if (!enabled) {
		return;
	}
	struct Totals {
		std::atomic<uint64_t> commands {0};
		std::atomic<uint64_t> ranges {0};
		std::atomic<uint64_t> regions {0};
		std::atomic<uint64_t> regions_64k {0};
		std::atomic<uint64_t> range_bytes {0};
		std::atomic<uint64_t> region_bytes {0};
		std::atomic<int64_t>  last_ms {0};
		std::atomic<uint64_t> last_flip {0};
	};
	static Totals totals;
	uint64_t      range_bytes  = 0;
	uint64_t      region_bytes = 0;
	for (const auto& range: ranges) {
		range_bytes += range.size;
	}
	for (const auto& region: regions) {
		region_bytes += region.size;
	}
	totals.commands.fetch_add(1, std::memory_order_relaxed);
	totals.ranges.fetch_add(ranges.size(), std::memory_order_relaxed);
	totals.regions.fetch_add(regions.size(), std::memory_order_relaxed);
	totals.regions_64k.fetch_add(CountMergedRegions(ranges, 64 * 1024), std::memory_order_relaxed);
	totals.range_bytes.fetch_add(range_bytes, std::memory_order_relaxed);
	totals.region_bytes.fetch_add(region_bytes, std::memory_order_relaxed);
	const int64_t now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
	                           std::chrono::steady_clock::now().time_since_epoch())
	                           .count();
	auto last = totals.last_ms.load(std::memory_order_relaxed);
	if (last == 0) {
		if (totals.last_ms.compare_exchange_strong(last, now_ms, std::memory_order_relaxed)) {
			totals.last_flip.store(Live::Testing::CpFlips(), std::memory_order_relaxed);
		}
		return;
	}
	if (now_ms - last < 10000 ||
	    !totals.last_ms.compare_exchange_strong(last, now_ms, std::memory_order_relaxed)) {
		return;
	}
	const auto flip  = Live::Testing::CpFlips();
	const auto flips = std::max<uint64_t>(1, flip - totals.last_flip.exchange(flip));
	const auto per   = [flips](std::atomic<uint64_t>& value) {
        return static_cast<double>(value.exchange(0, std::memory_order_relaxed)) /
               static_cast<double>(flips);
	};
	const auto commands    = per(totals.commands);
	const auto range_count = per(totals.ranges);
	const auto region      = per(totals.regions);
	const auto region_64k  = per(totals.regions_64k);
	const auto bytes       = per(totals.range_bytes);
	const auto copied      = per(totals.region_bytes);
	std::printf("ReadbackRegions 10s: %" PRIu64 " flips, per flip: %.1f copy commands, %.1f ranges, %.1f "
	     "regions (%.1f with a 64 KiB gap), %.1f KiB written back, %.1f KiB copied\n",
	     flips, commands, range_count, region, region_64k, bytes / 1024.0, copied / 1024.0);
}

// The regions of copies whose source and destination advance together (a side-readback window):
// a copy starting at most `gap` bytes after the previous region's source end joins it.
std::vector<vk::BufferCopy> MergeLinearCopies(const std::vector<vk::BufferCopy>& copies,
                                              uint64_t gap) {
	if (gap == 0 || copies.size() < 2) {
		return copies;
	}
	std::vector<vk::BufferCopy> regions;
	regions.reserve(copies.size());
	for (const auto& copy: copies) {
		if (!regions.empty()) {
			auto&      last     = regions.back();
			const auto last_end = last.srcOffset + last.size;
			if (copy.srcOffset >= last.srcOffset && copy.srcOffset <= last_end + gap &&
			    copy.dstOffset >= last.dstOffset &&
			    copy.dstOffset - last.dstOffset == copy.srcOffset - last.srcOffset) {
				last.size = std::max(last_end, copy.srcOffset + copy.size) - last.srcOffset;
				continue;
			}
		}
		regions.push_back(copy);
	}
	return regions;
}

// Packs download parts (source ascending) into regions: a part starting at most `gap` bytes after
// the previous region's source end joins it. Each part's dstOffset becomes its place inside its
// region's packed bytes (regions 64-byte aligned, as before). Returns the packed size.
uint64_t PackDownloadRegions(std::vector<vk::BufferCopy>& parts, uint64_t gap,
                             std::vector<vk::BufferCopy>& regions) {
	regions.clear();
	regions.reserve(parts.size());
	uint64_t packed = 0;
	for (auto& part: parts) {
		if (gap != 0 && !regions.empty()) {
			auto&      last     = regions.back();
			const auto last_end = last.srcOffset + last.size;
			if (part.srcOffset >= last.srcOffset && part.srcOffset <= last_end + gap) {
				packed -= Common::AlignUp(last.size, uint64_t {64});
				last.size = std::max(last_end, part.srcOffset + part.size) - last.srcOffset;
				packed += Common::AlignUp(last.size, uint64_t {64});
				part.dstOffset = last.dstOffset + (part.srcOffset - last.srcOffset);
				continue;
			}
		}
		part.dstOffset = packed;
		regions.emplace_back(part.srcOffset, packed, part.size);
		packed += Common::AlignUp(part.size, uint64_t {64});
	}
	return packed;
}

uint64_t ParseEnvU64(const char* name, uint64_t fallback) {
	const auto* value = std::getenv(name);
	if (value == nullptr) {
		return fallback;
	}
	char*      end    = nullptr;
	const auto parsed = std::strtoull(value, &end, 10);
	return end == value || *end != '\0' ? fallback : parsed;
}

// Guest write-fault policy of the buffer tracker (RegionManager::MarkWriteFault):
//   KYTY_FAULT_AHEAD_KB   aligned window made CPU-dirty around a write-faulting page: a power of
//                         two, 8..256 KiB (default 32); 0 or 4 disables fault-ahead.
//   KYTY_HOT_PAGES        0 disables hot pages. A page write-faulting in KYTY_HOT_PAGE_FRAMES
//                         consecutive guest frames (default 3) stays CPU-dirty and writable; its
//                         uploads copy it only when its contents differ from the last copy
//                         uploaded (exact shadow compare). It returns to normal tracking after
//                         KYTY_HOT_PAGE_QUIET_FRAMES frames (default 8) without a change or
//                         without an upload, on any GPU-side write that can reach it, or when its
//                         buffer is untracked. At most KYTY_HOT_PAGE_MAX pages (default 1024,
//                         4 KiB shadow each) are hot.
//   KYTY_UPLOAD_COPY_OUTSIDE_LOCK  0 copies written uploads with their region locks held (the
//                         previous behaviour) instead of MemoryTracker::ForEachWrittenUploadRange.
MemoryTracker::FaultPolicy BufferFaultPolicy() {
	MemoryTracker::FaultPolicy policy;
	const auto ahead_kib = ParseEnvU64("KYTY_FAULT_AHEAD_KB", 32);
	if (ahead_kib >= 8 && ahead_kib <= 256 && (ahead_kib & (ahead_kib - 1)) == 0) {
		policy.ahead_pages = static_cast<uint32_t>(ahead_kib * 1024 / TRACKER_PAGE_SIZE);
	}
	if (ParseEnvU64("KYTY_HOT_PAGES", 1) != 0) {
		policy.hot_frames = static_cast<uint32_t>(
		    std::clamp<uint64_t>(ParseEnvU64("KYTY_HOT_PAGE_FRAMES", 3), 1, 255));
		policy.hot_max =
		    static_cast<uint32_t>(std::min<uint64_t>(ParseEnvU64("KYTY_HOT_PAGE_MAX", 1024), 65536));
		if (policy.hot_max == 0) {
			policy.hot_frames = 0;
		}
	}
	policy.copy_outside_lock = ParseEnvU64("KYTY_UPLOAD_COPY_OUTSIDE_LOCK", 1) != 0;
	return policy;
}

uint32_t HotPageQuietFrames() {
	return static_cast<uint32_t>(
	    std::clamp<uint64_t>(ParseEnvU64("KYTY_HOT_PAGE_QUIET_FRAMES", 8), 1, 1000));
}

// BufferCache::m_hot_check_limit (0 disables).
uint32_t HotPageCheckLimit() {
	return static_cast<uint32_t>(
	    std::min<uint64_t>(ParseEnvU64("KYTY_HOT_PAGE_CHECK_LIMIT", 64), 1u << 20u));
}

// BufferCache::RangeMemo (bufferCache.h).
bool RangeMemoEnabled() {
	return ParseEnvU64("KYTY_BUFFER_RANGE_MEMO", 1) != 0;
}

// The small-read stream decision of ObtainBuffer from one MemoryTracker::QueryDirty.
bool DirtyQueryCombinedEnabled() {
	static const bool enabled = ParseEnvU64("KYTY_BUFFER_DIRTY_QUERY_COMBINED", 1) != 0;
	return enabled;
}

// KYTY_STREAM_DIRECT_READ=1 (default off; upstream 37501b5a7): a small read binding copied into
// the stream buffer reads the guest bytes through the guest mapping instead of the backing store
// (TryReadBacking). Its pages are CPU-dirty and not GPU-dirty, so no protection stops the read.
BufferCache* g_defer_cache = nullptr; // the BufferCache whose stream copies are deferred
UploadDma*   g_write_drain_dma = nullptr; // KYTY_UPLOAD_DMA_WRITE_DRAIN: the cache's upload DMA

bool StreamCopyProbe() {
	static const bool enabled = ParseEnvU64("KYTY_STREAM_COPY_PROBE", 0) != 0;
	return enabled;
}

void StreamCopyProbeAdd(uint64_t size, uint64_t cycles, uint64_t vaddr) {
	static uint64_t count = 0, bytes = 0, total = 0, large = 0, start = 0, dup = 0, dup_bytes = 0;
	static uint64_t hist[8] = {};
	static std::unordered_map<uint64_t, uint64_t> last;
	{
		const auto key = vaddr ^ (size << 44u);
		const auto t   = __rdtsc();
		auto [it, inserted] = last.try_emplace(key, t);
		if (!inserted) {
			if (t - it->second < 33'600'000ull) { // 8 ms
				dup++;
				dup_bytes += size;
			}
			it->second = t;
		}
	}
	count++;
	bytes += size;
	total += cycles;
	hist[std::min<uint64_t>(std::bit_width(size) / 2u, 7u)]++;
	large += size > 4096 ? 1u : 0u;
	const auto now = __rdtsc();
	if (start == 0) {
		start = now;
	}
	if (now - start >= 42'000'000'000ull) {
		std::printf("StreamCopy 10s: dup8ms %" PRIu64 " dupbytes %" PRIu64 " keys %zu ", dup, dup_bytes, last.size());
		dup = dup_bytes = 0;
		last.clear();
		std::printf("copies %" PRIu64 " bytes %" PRIu64 " cycles %" PRIu64
		            " large %" PRIu64 " sizes(log4) %" PRIu64 " %" PRIu64 " %" PRIu64 " %" PRIu64
		            " %" PRIu64 " %" PRIu64 " %" PRIu64 " %" PRIu64 "\n",
		            count, bytes, total, large, hist[0], hist[1], hist[2], hist[3], hist[4], hist[5],
		            hist[6], hist[7]);
		std::fflush(stdout);
		count = bytes = total = large = 0;
		std::fill(std::begin(hist), std::end(hist), 0u);
		start = now;
	}
}

// KYTY_SNAPSHOT_CLEAN_READS (default on; =0 off): a small read binding (<= CACHING_PAGESIZE) that
// is not GPU-dirty is copied into the stream buffer even when none of its pages is CPU-dirty, so
// the draw reads the guest bytes as they are at bind time instead of the cache buffer. Astro Bot
// (PPSA21564) rotates its per-frame constants through three arenas; two of them are still
// CPU-dirty at bind (stream copies), the third is uploaded earlier by another consumer of the
// large cache buffer that holds it and so binds that buffer: with GC/VRAM pressure (and rarely
// without) every third frame then reads wrong helmet constants (period-3 visor-crack flicker,
// flightrec2 2026-11-18). The snapshot makes all three arenas take the same stream path.
bool SnapshotCleanReadsEnabled() {
	static const bool enabled = ParseEnvU64("KYTY_SNAPSHOT_CLEAN_READS", 1) != 0;
	return enabled;
}

bool StreamDirectReadEnabled() {
	static const bool enabled = ParseEnvU64("KYTY_STREAM_DIRECT_READ", 0) != 0;
	return enabled;
}

int RangeMemoVerifyMode() {
	static const int mode = [] {
		const auto* value = std::getenv("KYTY_BUFFER_RANGE_MEMO_VERIFY");
		if (value == nullptr || *value == '\0' || std::strcmp(value, "0") == 0) {
			return 0;
		}
		return std::strcmp(value, "exit") == 0 ? 2 : 1;
	}();
	return mode;
}

// KYTY_TRACKER_RELAXED_QUERIES (default on): on the GPU thread, the small-read stream decision
// and the "nothing to upload" test of read synchronizations use the tracker's lock-free dirty
// mirrors (MemoryTracker::QueryDirtyRelaxed) instead of the region locks.
bool RelaxedQueriesEnabled() {
	static const bool enabled = ParseEnvU64("KYTY_TRACKER_RELAXED_QUERIES", 1) != 0;
	return enabled;
}

// KYTY_BINDING_EPOCH_MEMO (default on; =0 off): read bindings reused within a sync epoch
// (BufferCache::BindingMemo).
bool BindingEpochMemoEnabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_BINDING_EPOCH_MEMO");
		return value == nullptr || std::strcmp(value, "0") != 0;
	}();
	return enabled;
}

// KYTY_BINDING_EPOCH_MEMO_VERIFY=1|exit: every hit also runs the normal path
// (BufferCache::VerifyBindingHit).
int BindingEpochMemoVerifyMode() {
	static const int mode = [] {
		const auto* value = std::getenv("KYTY_BINDING_EPOCH_MEMO_VERIFY");
		if (value == nullptr || *value == '\0' || std::strcmp(value, "0") == 0) {
			return 0;
		}
		return std::strcmp(value, "exit") == 0 ? 2 : 1;
	}();
	return mode;
}

// 0 when the variable is unset, empty or "0"; 2 for "exit"; 1 otherwise. Read per cache, so tests
// can build caches with either setting.
int EnvVerifyMode(const char* name) {
	const auto* value = std::getenv(name);
	if (value == nullptr || *value == '\0' || std::strcmp(value, "0") == 0) {
		return 0;
	}
	return std::strcmp(value, "exit") == 0 ? 2 : 1;
}

// KYTY_BINDING_MEMO_CROSS_EPOCH (default on; =0 off): a cache-buffer read-binding memo from an
// earlier sync epoch is reused once the range is proven to have no CPU-dirty page
// (BufferCache::ObtainReadBinding).
bool BindingMemoCrossEpochEnabled() {
	return ParseEnvU64("KYTY_BINDING_MEMO_CROSS_EPOCH", 1) != 0;
}

// ASTROHYBRID_COMMIT_EPOCH_FASTPATH (bufferCache.h): 0 off (default), 1 on, 2 verify, 3 exit.
int CommitEpochFastPathMode() {
	const char* value = std::getenv("ASTROHYBRID_COMMIT_EPOCH_FASTPATH");
	if (value == nullptr || value[0] == '\0' || std::strcmp(value, "0") == 0) {
		return 0;
	}
	if (std::strcmp(value, "verify") == 0) {
		return 2;
	}
	return std::strcmp(value, "exit") == 0 ? 3 : 1;
}

// KYTY_WRITTEN_SYNC_SKIP (default on; =0 off): a written synchronization of a range that is
// entirely GPU-owned does nothing and is skipped (BufferCache::SynchronizeBuffer).
// KYTY_WRITTEN_SYNC_SKIP_VERIFY=1|exit runs the normal path anyway and counts skips it would
// have made while that path found pages to upload (WrittenSyncSkipVerifyMismatches).
bool WrittenSyncSkipEnabled() {
	return ParseEnvU64("KYTY_WRITTEN_SYNC_SKIP", 1) != 0;
}

// KYTY_FALSE_SHARING_WRITES (default off; =1 on): write faults on GPU-owned pages at bytes the GPU
// never wrote release the page without draining the GPU (BufferCache::TryFalseSharingWrite).
bool FalseSharingWritesEnabled() {
	return ParseEnvU64("KYTY_FALSE_SHARING_WRITES", 0) != 0;
}

// KYTY_BDA_SYNC_EPOCH_VERIFY=1|exit: every skipped BDA pass runs anyway and counts the pages the
// skip would have missed that no guest write explains (BufferCache::VerifyBdaEpochSkip).
int BdaEpochVerifyMode() {
	static const int mode = [] {
		const auto* value = std::getenv("KYTY_BDA_SYNC_EPOCH_VERIFY");
		if (value == nullptr || *value == '\0' || std::strcmp(value, "0") == 0) {
			return 0;
		}
		return std::strcmp(value, "exit") == 0 ? 2 : 1;
	}();
	return mode;
}

// KYTY_TRACKER_RELAXED_VERIFY=1|exit: every relaxed answer is followed by the locked query. Only
// the transitions other threads can make in between (a page turning CPU-dirty, a GPU-dirty page
// published) may separate them, and only while the range's mutation serials moved.
int RelaxedVerifyMode() {
	static const int mode = [] {
		const auto* value = std::getenv("KYTY_TRACKER_RELAXED_VERIFY");
		if (value == nullptr || *value == '\0' || std::strcmp(value, "0") == 0) {
			return 0;
		}
		return std::strcmp(value, "exit") == 0 ? 2 : 1;
	}();
	return mode;
}

// Readback window of guest write faults on GPU-owned pages, in bytes: a power of two between
// 4 KiB (only the faulting page) and 512 KiB (default, the read-drain window). Every page of the
// window loses GPU ownership, so a smaller window downloads fewer bytes per fault but makes a
// CPU writer walking a larger GPU-written range drain the GPU once per window instead.
uint64_t WriteFaultWindow() {
	static const uint64_t window = []() -> uint64_t {
		constexpr uint64_t Default = 512 * 1024;
		const auto*        value   = std::getenv("KYTY_WRITE_FAULT_WINDOW_KB");
		if (value == nullptr) {
			return Default;
		}
		char*      end = nullptr;
		const auto kib = std::strtoull(value, &end, 10);
		if (end == value || *end != '\0' || kib < 4 || kib > 512 || (kib & (kib - 1)) != 0) {
			return Default;
		}
		return kib * 1024;
	}();
	return window;
}

// Eager readback publication (BufferCache::IssueEagerReadbacks) unless KYTY_READBACK_EAGER is
// "0". Read per cache (not cached in a static) so tests can build caches with either setting.
bool EagerReadbackEnabled() {
	const auto* value = std::getenv("KYTY_READBACK_EAGER");
	return value == nullptr || !(value[0] == '0' && value[1] == '\0');
}

// KYTY_READBACK_EAGER_PAGES (hot pages kept, default 64, at most 256),
// KYTY_READBACK_EAGER_IDLE_FRAMES (frames without a readback before a page stops being hot,
// default 600) and KYTY_READBACK_EAGER_FRAME_BUDGET (copies per page and frame, default 4).
EagerReadbackPages::Limits EagerReadbackLimits() {
	EagerReadbackPages::Limits limits;
	limits.capacity = static_cast<uint32_t>(
	    std::min<uint64_t>(ParseEnvU64("KYTY_READBACK_EAGER_PAGES", 64), 2048));
	limits.idle_frames = static_cast<uint32_t>(
	    std::clamp<uint64_t>(ParseEnvU64("KYTY_READBACK_EAGER_IDLE_FRAMES", 600), 1, 1000000));
	limits.frame_budget = static_cast<uint32_t>(
	    std::clamp<uint64_t>(ParseEnvU64("KYTY_READBACK_EAGER_FRAME_BUDGET", 4), 1, 1024));
	return limits;
}

// Early submissions per frame after a recorded writer of a page the GPU thread reads back
// (KYTY_READBACK_EAGER_FLUSHES, default 8; 0 disables them).
uint32_t EagerFlushBudget() {
	return static_cast<uint32_t>(
	    std::min<uint64_t>(ParseEnvU64("KYTY_READBACK_EAGER_FLUSHES", 8), 100000));
}

// Reads larger than this are bulk readbacks (e.g. DCC metadata): their pages do not become hot.
constexpr uint64_t EagerReadMaxBytes = 64 * 1024;

} // namespace

// One side-copy readback: the exact GPU-dirty bytes of a tracker-page-aligned window, copied
// by a command buffer outside the scheduler's recording. Completion (any thread, exactly once)
// publishes them to the backing and unprotects the window's pages no newer writer re-owned.
// An eager readback (KYTY_READBACK_EAGER) is the same record for one tracker page whose copy
// ends a scheduler recording instead: `value` is that recording's master tick, and the
// completion runner completes it (if no reader did first) once that tick is reached.
struct BufferCache::SideReadback {
	std::mutex              mutex;
	std::atomic<bool>       done {false};
	uint64_t                begin       = 0;
	uint64_t                end         = 0;
	uint64_t                value       = 0;
	uint32_t                slot        = 0;
	bool                    eager       = false;
	uint64_t                publication = 0;
	// Side copies: the master tick the copy waits for (KYTY_READ_FAULT_DIAG).
	uint64_t                producer    = 0;
	// HangTrace clock at issue (eager rows report issue-to-publication time).
	uint64_t                issue_ns    = 0;
	// Staged at (address - begin) within the slot.
	std::vector<GuestRange> ranges;
};

struct BufferCache::SideReadbackState {
	static constexpr uint32_t SlotCount = 16;
	// One tracker page each; a slot is busy from its eager copy's issue to its publication.
	static constexpr uint32_t EagerSlotCount = 64;
	struct Slot {
		vk::CommandBuffer command = nullptr;
		// Set by the GPU thread at issue, cleared after the slot's publication has read it.
		std::atomic<bool> busy {false};
	};

	SideReadbackState(GraphicContext& context, CommandScheduler& scheduler, uint64_t window_size,
	                  bool eager)
	    : graphics(context), window(window_size) {
		vk::CommandPoolCreateInfo pool_info {};
		pool_info.queueFamilyIndex = graphics.queue_family;
		pool_info.flags            = vk::CommandPoolCreateFlagBits::eTransient |
		                  vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
		RequireVulkanSuccess(graphics.device.createCommandPool(&pool_info, nullptr, &pool),
		                     "create side-readback command pool");
		std::array<vk::CommandBuffer, SlotCount> buffers {};
		vk::CommandBufferAllocateInfo            allocate {};
		allocate.commandPool        = pool;
		allocate.level              = vk::CommandBufferLevel::ePrimary;
		allocate.commandBufferCount = SlotCount;
		RequireVulkanSuccess(graphics.device.allocateCommandBuffers(&allocate, buffers.data()),
		                     "allocate side-readback command buffers");
		for (uint32_t index = 0; index < SlotCount; ++index) {
			slots[index].command = buffers[index];
		}
		vk::SemaphoreTypeCreateInfo type_info {};
		type_info.semaphoreType = vk::SemaphoreType::eTimeline;
		type_info.initialValue  = 0;
		vk::SemaphoreCreateInfo create_info {};
		create_info.pNext = &type_info;
		RequireVulkanSuccess(graphics.device.createSemaphore(&create_info, nullptr, &semaphore),
		                     "create side-readback timeline semaphore");
		staging = std::make_unique<Buffer>(graphics, scheduler, MemoryUsage::Download, 0,
		                                   vk::BufferUsageFlagBits::eTransferDst,
		                                   window * SlotCount);
		EXIT_IF(staging->Mapped().empty());
		SetVulkanObjectNameF(graphics.device, staging->Handle(), "Kyty.SideReadbackStaging");
		if (eager) {
			eager_staging = std::make_unique<Buffer>(graphics, scheduler, MemoryUsage::Download, 0,
			                                         vk::BufferUsageFlagBits::eTransferDst,
			                                         TRACKER_PAGE_SIZE * EagerSlotCount);
			EXIT_IF(eager_staging->Mapped().empty());
			SetVulkanObjectNameF(graphics.device, eager_staging->Handle(),
			                     "Kyty.EagerReadbackStaging");
		}
	}

	~SideReadbackState() {
		eager_staging.reset();
		staging.reset();
		if (semaphore != nullptr) {
			graphics.device.destroySemaphore(semaphore, nullptr);
		}
		if (pool != nullptr) {
			graphics.device.destroyCommandPool(pool, nullptr);
		}
	}

	KYTY_CLASS_NO_COPY(SideReadbackState);

	void Wait(uint64_t target) const {
		HangWatchdog::Scope wait("side-readback-gpu",
		                         reinterpret_cast<uint64_t>(static_cast<VkSemaphore>(semaphore)),
		                         target);
		uint64_t current = 0;
		RequireVulkanSuccess(graphics.device.getSemaphoreCounterValue(semaphore, &current),
		                     "query side-readback semaphore");
		if (current >= target) {
			return;
		}
		vk::SemaphoreWaitInfo wait_info {};
		wait_info.semaphoreCount = 1;
		wait_info.pSemaphores    = &semaphore;
		wait_info.pValues        = &target;
		RequireVulkanSuccess(graphics.device.waitSemaphores(&wait_info, UINT64_MAX),
		                     "wait for side readback");
	}

	GraphicContext&                            graphics;
	const uint64_t                             window;
	vk::CommandPool                            pool      = nullptr;
	vk::Semaphore                              semaphore = nullptr;
	// GPU thread only: the last signal value handed out.
	uint64_t                                   next_value = 0;
	std::array<Slot, SlotCount>                slots;
	std::unique_ptr<Buffer>                    staging;
	// Eager copies (null unless KYTY_READBACK_EAGER): EagerSlotCount tracker pages.
	std::unique_ptr<Buffer>                    eager_staging;
	std::array<std::atomic<bool>, EagerSlotCount> eager_busy {};
	mutable std::mutex                         pending_mutex;
	std::vector<std::shared_ptr<SideReadback>> pending;
	std::atomic<size_t>                        pending_count {0};
};

// KYTY_BDA_PAGETABLE_SPARSE: memory for the sparse page table, in chunks of zeroed device memory
// handed out one sparse block at a time. A chunk is zeroed (through a plain buffer over its memory)
// and that fill has completed before any of its blocks is bound, and every bind is waited for here:
// a block reads zero before its bind (residencyNonResidentStrict) and after it, so no GPU work, in
// flight or recorded later, can see anything but zero or an entry written after the bind.
struct BufferCache::SparsePageTable {
	static constexpr uint32_t ChunkBlocks = 32;

	struct Chunk {
		VmaAllocation  allocation = nullptr;
		vk::DeviceMemory memory   = nullptr;
		uint64_t       offset     = 0;
		uint32_t       used       = 0;
	};

	SparsePageTable(GraphicContext& context, vk::Buffer buffer, uint64_t size)
	    : graphics(context), table(buffer) {
		vk::MemoryRequirements requirements {};
		graphics.device.getBufferMemoryRequirements(table, &requirements);
		block_size       = requirements.alignment;
		memory_type_bits = requirements.memoryTypeBits;
		EXIT_IF(block_size == 0 || size % block_size != 0 || memory_type_bits == 0);
		bound.assign(static_cast<size_t>(size / block_size), 0);
		vk::CommandPoolCreateInfo pool_info {};
		pool_info.flags            = vk::CommandPoolCreateFlagBits::eTransient |
		                             vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
		pool_info.queueFamilyIndex = graphics.queue_family;
		RequireVulkanSuccess(graphics.device.createCommandPool(&pool_info, nullptr, &pool),
		                     "create BDA page table command pool");
		vk::FenceCreateInfo fence_info {};
		RequireVulkanSuccess(graphics.device.createFence(&fence_info, nullptr, &fence),
		                     "create BDA page table fence");
		LOGF("BDA page table: sparse residency, %zu blocks of %" PRIu64 " KiB (KYTY_BDA_PAGETABLE_SPARSE)\n",
		     bound.size(), block_size / 1024);
	}

	~SparsePageTable() {
		for (const auto& chunk: chunks) {
			VramStats::Note(VramStats::Kind::OtherBuffer, true,
			                -static_cast<int64_t>(block_size * ChunkBlocks));
			VramStats::UnregisterBuffer(chunk.allocation);
			vmaFreeMemory(graphics.allocator, chunk.allocation);
		}
		if (fence != nullptr) {
			graphics.device.destroyFence(fence, nullptr);
		}
		if (pool != nullptr) {
			graphics.device.destroyCommandPool(pool, nullptr);
		}
	}

	// One submission to the side queue (queue 0 without one), waited for on the host.
	template <typename Submit>
	void SubmitAndWait(Submit&& submit, const char* operation) {
		HangWatchdog::Scope wait("sparse-page-table",
		                         reinterpret_cast<uint64_t>(static_cast<VkFence>(fence)));
		vk::Result result {};
		if (graphics.side_queue != nullptr) {
			Common::LockGuard lock(graphics.side_queue_mutex);
			result = submit(graphics.side_queue);
		} else {
			Common::LockGuard lock(graphics.queue_mutex);
			result = submit(graphics.queue);
		}
		RequireVulkanSuccess(result, operation);
		RequireVulkanSuccess(graphics.device.waitForFences(1, &fence, VK_TRUE, UINT64_MAX), operation);
		RequireVulkanSuccess(graphics.device.resetFences(1, &fence), operation);
	}

	// Allocates a chunk of device memory the table can bind and zeroes it.
	void AddChunk() {
		const uint64_t     bytes = block_size * ChunkBlocks;
		vk::BufferCreateInfo zero_info {};
		zero_info.size  = bytes;
		zero_info.usage = vk::BufferUsageFlagBits::eTransferDst;
		vk::Buffer zero_buffer = nullptr;
		RequireVulkanSuccess(graphics.device.createBuffer(&zero_info, nullptr, &zero_buffer),
		                     "create BDA page table zero buffer");
		vk::MemoryRequirements zero_requirements {};
		graphics.device.getBufferMemoryRequirements(zero_buffer, &zero_requirements);
		VkMemoryRequirements requirements {};
		requirements.size           = bytes;
		requirements.alignment      = std::max<uint64_t>(block_size, zero_requirements.alignment);
		requirements.memoryTypeBits = memory_type_bits & zero_requirements.memoryTypeBits;
		EXIT_IF(requirements.memoryTypeBits == 0);
		// Own device memory, not dedicated to a resource: both buffers may bind it.
		VmaAllocationCreateInfo create {};
		create.flags         = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
		create.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
		Chunk             chunk;
		VmaAllocationInfo info {};
		RequireVulkanSuccess(static_cast<vk::Result>(vmaAllocateMemory(graphics.allocator, &requirements,
		                                                               &create, &chunk.allocation, &info)),
		                     "allocate BDA page table memory");
		chunk.memory = info.deviceMemory;
		chunk.offset = info.offset;
		EXIT_IF(chunk.offset % block_size != 0);
		RequireVulkanSuccess(static_cast<vk::Result>(vmaBindBufferMemory(graphics.allocator,
		                                                                 chunk.allocation, zero_buffer)),
		                     "bind BDA page table zero buffer");
		vk::CommandBufferAllocateInfo command_info {};
		command_info.commandPool        = pool;
		command_info.level              = vk::CommandBufferLevel::ePrimary;
		command_info.commandBufferCount = 1;
		vk::CommandBuffer command = nullptr;
		RequireVulkanSuccess(graphics.device.allocateCommandBuffers(&command_info, &command),
		                     "allocate BDA page table command buffer");
		vk::CommandBufferBeginInfo begin {};
		begin.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
		RequireVulkanSuccess(command.begin(&begin), "begin BDA page table zero fill");
		command.fillBuffer(zero_buffer, 0, VK_WHOLE_SIZE, 0);
		RequireVulkanSuccess(command.end(), "end BDA page table zero fill");
		vk::SubmitInfo submit {};
		submit.commandBufferCount = 1;
		submit.pCommandBuffers    = &command;
		SubmitAndWait([&](vk::Queue queue) { return queue.submit(1, &submit, fence); },
		              "zero BDA page table memory");
		graphics.device.freeCommandBuffers(pool, 1, &command);
		graphics.device.destroyBuffer(zero_buffer, nullptr);
		VramStats::Note(VramStats::Kind::OtherBuffer, true, static_cast<int64_t>(bytes));
		VramStats::RegisterBuffer(chunk.allocation, {bytes, 0, 0, true, false});
		chunks.push_back(chunk);
	}

	// Binds zeroed memory behind every block of table bytes [offset, offset + size) without any.
	void EnsureResident(uint64_t offset, uint64_t size) {
		const auto first = offset / block_size;
		const auto last  = (offset + size - 1) / block_size;
		std::vector<vk::SparseMemoryBind> binds;
		for (auto block = first; block <= last; ++block) {
			if (bound[block] != 0) {
				continue;
			}
			if (chunks.empty() || chunks.back().used == ChunkBlocks) {
				AddChunk();
			}
			auto&                chunk = chunks.back();
			vk::SparseMemoryBind bind {};
			bind.resourceOffset = block * block_size;
			bind.size           = block_size;
			bind.memory         = chunk.memory;
			bind.memoryOffset   = chunk.offset + uint64_t {chunk.used} * block_size;
			binds.push_back(bind);
			chunk.used++;
			bound[block] = 1;
		}
		if (binds.empty()) {
			return;
		}
		vk::SparseBufferMemoryBindInfo buffer_bind {};
		buffer_bind.buffer    = table;
		buffer_bind.bindCount = static_cast<uint32_t>(binds.size());
		buffer_bind.pBinds    = binds.data();
		vk::BindSparseInfo bind_info {};
		bind_info.bufferBindCount = 1;
		bind_info.pBufferBinds    = &buffer_bind;
		SubmitAndWait([&](vk::Queue queue) { return queue.bindSparse(1, &bind_info, fence); },
		              "bind BDA page table memory");
		bound_blocks += binds.size();
	}

	GraphicContext&      graphics;
	vk::Buffer           table;
	uint64_t             block_size       = 0;
	uint32_t             memory_type_bits = 0;
	std::vector<uint8_t> bound;
	std::vector<Chunk>   chunks;
	uint64_t             bound_blocks = 0;
	vk::CommandPool      pool  = nullptr;
	vk::Fence            fence = nullptr;
};

void BufferCache::EnsureBdaTableResident(uint64_t first_page, uint64_t page_count) {
	if (m_bda_sparse == nullptr || page_count == 0) {
		return;
	}
	m_bda_sparse->EnsureResident(first_page * sizeof(vk::DeviceAddress),
	                             page_count * sizeof(vk::DeviceAddress));
}

void BufferCache::WriteDataBuffer(Buffer& buffer, uint64_t address, const void* source,
                                  uint64_t size) {
	auto* bytes = static_cast<const uint8_t*>(source);
	while (size != 0) {
		const auto chunk  = std::min(size, m_staging_buffer.Size());
		const auto offset = m_staging_buffer.Copy(bytes, chunk, 4);
		buffer.CopyFrom(m_scheduler.Current(), m_staging_buffer, offset, buffer.Offset(address),
		                chunk, vk::AccessFlagBits::eHostWrite);
		bytes += chunk;
		address += chunk;
		size -= chunk;
	}
}

void BufferCache::Register(BufferId id) {
	ChangeRegister<true>(id);
}

void BufferCache::Unregister(BufferId id) {
	m_unregister_epoch++;
	ChangeRegister<false>(id);
}

// KYTY_BDA_PAGETABLE_MB=<n> (memtrim, default 0 = the whole 512 MiB table): the dense BDA page
// table holds only the entries of guest addresses below n MiB / 8 * 16 KiB (64 MiB: below 128 GiB;
// Astro's GPU mappings end at 0x10_c800_0000 = 67 GiB). A Metal heap is committed whole as soon as
// the GPU touches it (work/kyty/mem/mtltest/zero.m), so the zero fill of the 512 MiB table costs
// 512 MiB of footprint although the entries past the guest's GPU span are never written. Shaders
// keep indexing up to 2^26 entries: the table is bound with its real size and robustBufferAccess2
// (required on KosmicKrisp) reads 0 past it, i.e. "no buffer" (a fault-bitmap bit, as for any
// unmapped page). No shader or pipeline-cache change. A buffer registered above the covered range
// would be invisible to BDA reads: counted and reported (stderr, once) - raise n if that line ever
// appears.
static uint64_t BdaTableEntries() {
	static const uint64_t entries = [] {
		uint64_t all = BufferCache::CACHING_NUMPAGES;
		if (const auto* value = std::getenv("KYTY_BDA_PAGETABLE_MB"); value != nullptr) {
			const uint64_t mib = std::strtoull(value, nullptr, 10);
			if (mib != 0) {
				all = std::min<uint64_t>(all, (mib << 20u) / sizeof(vk::DeviceAddress));
				std::fprintf(
				    stderr,
				    "[memtrim] BDA page table %llu MiB: covers guest addresses below 0x%llx "
				    "(KYTY_BDA_PAGETABLE_MB)\n",
				    static_cast<unsigned long long>((all * sizeof(vk::DeviceAddress)) >> 20u),
				    static_cast<unsigned long long>(all << BufferCache::CACHING_PAGEBITS));
			}
		}
		return all;
	}();
	return entries;
}
static std::atomic<uint64_t> g_bda_pages_uncovered {0};
static void                  NoteBdaUncovered(uint64_t first, uint64_t last_exclusive) {
	const auto pages = memtrim::BdaUncovered(first, last_exclusive, BdaTableEntries());
	if (pages == 0) {
		return;
	}
	if (g_bda_pages_uncovered.fetch_add(pages, std::memory_order_relaxed) == 0) {
		std::fprintf(stderr,
		             "[memtrim] WARNING: guest buffer at 0x%llx is above the BDA page table "
		             "(KYTY_BDA_PAGETABLE_MB): BDA reads of it fault\n",
		             static_cast<unsigned long long>(first << BufferCache::CACHING_PAGEBITS));
	}
}

void BufferCache::EnsureBdaPageTableInitialized() {
	if (m_bda_pagetable_initialized) {
		return;
	}
	// A sparse table reads as zero wherever nothing is bound (KYTY_BDA_PAGETABLE_SPARSE).
	if (m_bda_pagetable_buffer.IsSparse()) {
		m_bda_pagetable_initialized = true;
		return;
	}
	// Recorded before the first table write or read, and ordered by Fill barriers.
	m_bda_pagetable_buffer.Fill(0, m_bda_pagetable_buffer.Size(), 0);
	m_bda_pagetable_initialized = true;
}

Buffer* BufferCache::GetBdaPageTableBuffer() {
	EnsureBdaPageTableInitialized();
	return &m_bda_pagetable_buffer;
}

template <bool insert>
void BufferCache::ChangeRegister(BufferId id) {
	InvalidateBdaSynchronization();
	EnsureBdaPageTableInitialized();
	auto& buffer = m_slot_buffers[id];
	PageTable::PageRange pages {};
	EXIT_IF(!PageTable::TryGetPageRange(buffer.CpuAddress(), buffer.Size(), pages));
	for (size_t page = pages.first; page < pages.last_exclusive; ++page) {
		if constexpr (insert) {
			m_page_table[page] = id;
		} else {
			m_page_table[page] = {};
		}
	}
	const auto size_pages = pages.last_exclusive - pages.first;
	if constexpr (insert) {
		const auto [it, inserted] = m_buffers.emplace(buffer.CpuAddress(), id);
		(void)it;
		EXIT_IF(!inserted);
		m_total_used_memory += buffer.Size();
		buffer.lru_id   = m_lru_cache.Insert(id, m_gc_tick);
		buffer.lru_tick = m_gc_tick;
		std::vector<vk::DeviceAddress> addresses;
		addresses.reserve(size_pages);
		for (uint64_t i = 0; i < size_pages; ++i) {
			addresses.push_back(buffer.BufferDeviceAddress() + (i << CACHING_PAGEBITS));
		}
		EnsureBdaTableResident(pages.first, size_pages);
		NoteBdaUncovered(pages.first, pages.last_exclusive);
		if (const auto covered = memtrim::BdaCovered(pages.first, size_pages, BdaTableEntries());
		    covered != 0) {
			WriteDataBuffer(m_bda_pagetable_buffer, pages.first * sizeof(vk::DeviceAddress),
			                addresses.data(), covered * sizeof(vk::DeviceAddress));
		}
	} else {
		const auto found = m_buffers.find(buffer.CpuAddress());
		EXIT_IF(found == m_buffers.end() || found->second != id);
		m_buffers.erase(found);
		// With VK_EXT_memory_budget the collector sets m_total_used_memory to the device-local
		// usage, which can be below the buffers' own bytes (allocations outside the device-local
		// heaps when VRAM runs short), so it saturates. Without it the counter is the buffers'
		// own bytes and must not underflow.
		EXIT_IF(buffer.Size() > m_total_used_memory && !m_graphics.CanReportMemoryUsage());
		m_total_used_memory -= std::min(m_total_used_memory, buffer.Size());
		m_lru_cache.Free(buffer.lru_id);
		if (const auto covered = memtrim::BdaCovered(pages.first, size_pages, BdaTableEntries());
		    covered != 0) {
			TilerTrace::Count("fill.who.bda_pagetable", 1, covered * sizeof(vk::DeviceAddress));
			m_bda_pagetable_buffer.Fill(pages.first * sizeof(vk::DeviceAddress),
			                            covered * sizeof(vk::DeviceAddress), 0);
		}
		buffer.is_deleted = true;
	}
}

// KYTY_BUFFER_LRU_SKIP=1 (default off): draws touch the same buffers many times per GC tick. The
// buffer mirrors its LRU item's tick (lru_tick, set by Insert and every Touch, the only writers of
// the item's tick), so a touch in a tick the item already holds, which LeastRecentlyUsedCache::Touch
// would return from at once, skips reading the scattered item. The LRU order is unchanged.
static bool BufferLruSkipEnabled() {
	static std::atomic<int64_t> enabled_once {Common::OnceUnset};
	const auto enabled = static_cast<bool>(Common::Once(enabled_once, [] {
		const auto* value = std::getenv("KYTY_BUFFER_LRU_SKIP");
		return value != nullptr && std::strcmp(value, "1") == 0;
	}));
	return enabled;
}

void BufferCache::TouchBuffer(const Buffer& buffer) {
	if (buffer.is_deleted) {
		return;
	}
	if (BufferLruSkipEnabled()) {
		if (buffer.lru_tick >= m_gc_tick) {
			return;
		}
		buffer.lru_tick = m_gc_tick;
	}
	m_lru_cache.Touch(buffer.lru_id, m_gc_tick);
}

void BufferCache::DeleteBuffer(BufferId id) {
	if (IsBufferInvalid(id)) {
		return;
	}
	{
		// A pending side copy reads this buffer outside the scheduler's timeline; the deferred
		// erase below is not ordered after it.
		const auto& buffer = m_slot_buffers[id];
		CompleteSideReadbacks(buffer.CpuAddress(), buffer.Size());
	}
	Unregister(id);
	if (m_scheduler.Active()) {
		m_scheduler.DeferOperation([this, id] { m_slot_buffers.erase(id); });
	} else {
		m_slot_buffers.erase(id);
	}
}

bool BufferCache::DownloadBufferMemory(Buffer& buffer, uint64_t vaddr, uint64_t size,
                                       const std::shared_ptr<EarlyReleasedDownload>& early) {
	// An older side publication of these pages must reach the backing (and settle its pages)
	// before this newer download is queued; otherwise it could overwrite newer bytes.
	CompleteSideReadbacks(vaddr, size);
	FlightRec::Rec(FlightRec::EvDownload, vaddr, size, buffer.CpuAddress());
	KYTY_GPU_OP_SITE("buffercache.download");
	std::vector<vk::BufferCopy> copies;
	uint64_t                    total_size     = 0;
	const auto                  buffer_address = buffer.CpuAddress();
	m_memory_tracker.ForEachDownloadRange<false>(
	    vaddr, size, [&](uint64_t address, uint64_t bytes) noexcept {
		    m_memory_tracker.ValidateGpuDirtyPages(m_gpu_modified_ranges, address, bytes,
		                                           "buffer download");
		    m_gpu_modified_ranges.ForEachInRange(address, bytes, [&](uint64_t start, uint64_t end) {
			    copies.emplace_back(start - buffer_address, total_size, end - start);
			    // Keep packed ranges on separate cache lines, as in shadPS4.
			    total_size += Common::AlignUp(end - start, 64);
		    });
		    // Ownership moves to the backing publication registered below (which bumps again).
		    CleanVerdict::Invalidate(address, bytes, Coherence::Source::BufferDirtySubtract);
		    m_gpu_range_removals.fetch_add(1, std::memory_order_acq_rel);
		    m_gpu_modified_ranges.Subtract(address, bytes);
	    });
	if (copies.empty()) {
		return false;
	}
	// Nearby ranges share one copy region (KYTY_READBACK_MERGE_GAP_KB); `copies` stay the exact
	// parts written back.
	std::vector<vk::BufferCopy> regions;
	total_size = PackDownloadRegions(copies, static_cast<uint64_t>(g_readback_merge_gap.Get()), regions);
	RecordReadbackRegions(copies, regions);
	Common::DebugCounters::Add(Common::DebugCounters::Counter::BufferDownloadBytes, total_size);

	auto [mapped, offset] = m_download_buffer.Map(total_size, 64);
	// A download larger than the staging ring gets a buffer of its own, released after the
	// publication below (upstream 18a5f04d0).
	std::unique_ptr<Buffer> temporary;
	if (mapped == nullptr) {
		temporary = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0,
		                                     vk::BufferUsageFlagBits::eTransferDst, total_size);
		mapped = temporary->Mapped().data();
	} else {
		m_download_buffer.Commit();
	}
	const auto& download = temporary ? *temporary : m_download_buffer;
	for (auto& copy: copies) {
		copy.dstOffset += offset;
	}
	for (auto& region: regions) {
		region.dstOffset += offset;
	}

	auto& command = m_scheduler.Current();
	command.EndRendering();
	const auto              native = command.Handle();
	vk::BufferMemoryBarrier before {};
	before.srcAccessMask       = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	before.dstAccessMask       = vk::AccessFlagBits::eTransferRead;
	before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.buffer              = buffer.Handle();
	before.offset              = 0;
	before.size                = buffer.Size();
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                       vk::PipelineStageFlagBits::eTransfer, {}, 0, nullptr, 1, &before, 0,
	                       nullptr);
	native.copyBuffer(buffer.Handle(), download.Handle(),
	                  static_cast<uint32_t>(regions.size()), regions.data());

	auto after          = before;
	after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask = vk::AccessFlagBits::eHostRead;
	after.buffer        = download.Handle();
	after.offset        = offset;
	after.size          = total_size;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                       vk::PipelineStageFlagBits::eAllCommands |
	                           vk::PipelineStageFlagBits::eHost,
	                       {}, 0, nullptr, 1, &after, 0, nullptr);
	std::vector<GuestRange> publication_ranges;
	publication_ranges.reserve(copies.size());
	for (const auto& copy: copies) {
		publication_ranges.push_back({buffer_address + copy.srcOffset, copy.size});
	}
	if (early != nullptr) {
		// The pages are released before this publication lands: remember its bytes (uploads skip
		// them meanwhile) and, verifying, their guest contents now.
		early->ranges = publication_ranges;
		if (early->verify) {
			for (const auto& range: early->ranges) {
				const auto at = early->snapshot.size();
				early->snapshot.resize(at + range.size);
				(void)Libs::LibKernel::Memory::TryReadBacking(range.address, early->snapshot.data() + at,
				                                              range.size);
			}
		}
	}
	const auto publication = BeginBackingPublication(publication_ranges, m_scheduler.CurrentTick());
	m_scheduler.DeferPriorityOperation([this, mapped, offset, total_size, buffer_address, publication,
	                                    early, copies = std::move(copies),
	                                    owner = std::move(temporary)] {
		(owner ? *owner : m_download_buffer).Invalidate(offset, total_size);
		if (early != nullptr && early->verify) {
			VerifyEarlyRelease(*early);
		}
		for (const auto& copy: copies) {
			Libs::LibKernel::Memory::WriteBacking(buffer_address + copy.srcOffset,
			                                      mapped + (copy.dstOffset - offset), copy.size);
		}
		EndBackingPublication(publication);
		if (early != nullptr) {
			early->published.store(true, std::memory_order_release);
		}
	});
	return true;
}

void BufferCache::VerifyEarlyRelease(const EarlyReleasedDownload& early) {
	// Priority worker, right before the publication writes these bytes: they must still hold what
	// they held when their pages were released, or a CPU write since is about to be overwritten.
	m_false_sharing_totals.verify_checks.fetch_add(1, std::memory_order_relaxed);
	Profiler::CountFrameEvent(Profiler::FrameEvent::FalseSharingVerifyChecks);
	std::vector<uint8_t> current;
	uint64_t             at        = 0;
	uint64_t             conflicts = 0;
	for (const auto& range: early.ranges) {
		current.resize(range.size);
		(void)Libs::LibKernel::Memory::TryReadBacking(range.address, current.data(), range.size);
		if (std::memcmp(current.data(), early.snapshot.data() + at, range.size) != 0) {
			conflicts++;
			static std::atomic<uint32_t> logged {0};
			if (logged.fetch_add(1, std::memory_order_relaxed) < 32) {
				std::fprintf(stderr,
				             "FalseSharingVerify: a CPU write to GPU-owned bytes of a released page "
				             "is overwritten by their publication: addr=0x%016" PRIx64
				             " size=0x%" PRIx64 "\n",
				             range.address, range.size);
			}
		}
		at += range.size;
	}
	if (conflicts == 0) {
		return;
	}
	m_false_sharing_totals.verify_conflicts.fetch_add(conflicts, std::memory_order_relaxed);
	Profiler::CountFrameEvent(Profiler::FrameEvent::FalseSharingVerifyConflicts, conflicts);
	if (m_false_sharing_verify == 2) {
		EXIT("FalseSharingVerify: a CPU write to GPU-owned bytes of a released page\n");
	}
}

void BufferCache::PruneEarlyReleased() {
	std::erase_if(m_early_released, [](const auto& entry) {
		return entry->published.load(std::memory_order_acquire);
	});
}

bool BufferCache::OverlapsUnpublished(uint64_t address, uint64_t size) const {
	const auto end = address + size;
	for (const auto& entry: m_early_released) {
		if (entry->published.load(std::memory_order_acquire)) {
			continue;
		}
		for (const auto& range: entry->ranges) {
			if (range.address < end && address < range.End()) {
				return true;
			}
		}
	}
	return false;
}

template <typename Emit>
void BufferCache::ForEachPublishedPart(uint64_t address, uint64_t size, Emit&& emit) const {
	// Only called while early releases are pending (a handful of ranges at most).
	const auto              end = address + size;
	std::vector<GuestRange> excluded;
	for (const auto& entry: m_early_released) {
		if (entry->published.load(std::memory_order_acquire)) {
			continue;
		}
		for (const auto& range: entry->ranges) {
			if (range.address < end && address < range.End()) {
				excluded.push_back(range);
			}
		}
	}
	if (excluded.empty()) {
		emit(address, size);
		return;
	}
	std::sort(excluded.begin(), excluded.end(),
	          [](const GuestRange& a, const GuestRange& b) { return a.address < b.address; });
	uint64_t cursor = address;
	for (const auto& range: excluded) {
		if (range.address > cursor) {
			emit(cursor, std::min(range.address, end) - cursor);
		}
		cursor = std::max(cursor, range.End());
		if (cursor >= end) {
			return;
		}
	}
	if (cursor < end) {
		emit(cursor, end - cursor);
	}
}

// KYTY_STREAM_RING_HOST=1 (default off): the 64 MiB stream ring (per-draw copies of small
// CPU-written buffers, flattened SRTs, shader data, index data) lives in cached host memory instead
// of the device-local host-visible heap (write-combined over the PCIe BAR), so the CPU's writes and
// the accesses after them do not stall on write-combining; the GPU reads the ring over the bus.
// Idea from BryanKAdams/KytyPS5 a25a2b9 (measured there on ReBAR; this machine has ReBAR off, so
// the ring sits in the 256 MiB BAR window today).
static bool StreamRingHostEnabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_STREAM_RING_HOST");
		return value != nullptr && std::strcmp(value, "1") == 0;
	}();
	return enabled;
}

BufferCache::BufferCache(GraphicContext& graphics, CommandScheduler& scheduler,
                         PageManager& page_manager, TextureCache& texture_cache)
    : m_graphics(graphics), m_scheduler(scheduler), m_fault_manager(graphics, scheduler, *this),
      m_gds_buffer(graphics, scheduler, MemoryUsage::Stream, 0, AllFlags, GdsBufferSize),
      m_bda_pagetable_buffer(graphics, scheduler, MemoryUsage::DeviceLocal, 0, AllFlags,
                             graphics.sparse_residency_buffer_enabled
                                 ? BDA_PAGETABLE_SIZE
                                 : BdaTableEntries() * sizeof(vk::DeviceAddress),
                             false, false, graphics.sparse_residency_buffer_enabled),
      m_bda_incremental_sync(IncrementalBdaSyncEnabled()),
      m_bda_hot_sync(m_bda_incremental_sync && BdaHotSyncEnabled()),
      m_memory_tracker(page_manager, m_bda_incremental_sync, BufferFaultPolicy()),
      m_hot_quiet_frames(HotPageQuietFrames()), m_hot_check_limit(HotPageCheckLimit()),
      m_range_memo(RangeMemoEnabled() ? std::make_unique<RangeMemo[]>(RangeMemoSlots) : nullptr),
      m_staging_buffer(graphics, scheduler, MemoryUsage::Upload, UploadRingSize(),
                       graphics.transfer_queue != nullptr),
      m_upload_dma(UploadDma::Create(graphics, scheduler)),
      m_stream_buffer(graphics, scheduler, MemoryUsage::Stream, 64 * MiB, false, {},
                      StreamRingHostEnabled()),
      m_download_buffer(graphics, scheduler, MemoryUsage::Download, 64 * MiB),
      m_device_buffer(graphics, scheduler, MemoryUsage::DeviceLocal, 128 * MiB),
      m_texture_cache(texture_cache) {
	m_range_memo_verify = m_range_memo != nullptr ? RangeMemoVerifyMode() : 0;
	// KYTY_TRACKER_LOCKFREE_DIRTY (memoryTracker.h): the GPU thread owns the GPU-dirty bits.
	MemoryTracker::SetLockFreeDirtyOwner(&GuestGpu::IsGpuThread);
	if (m_upload_dma != nullptr) {
		g_write_drain_dma = m_upload_dma.get(); // KYTY_UPLOAD_DMA_WRITE_DRAIN
	}
	m_relaxed_queries = RelaxedQueriesEnabled();
	m_bda_dirty_log   = m_bda_hot_sync && BdaDirtyLogEnabled();
	if (m_bda_dirty_log) {
		m_memory_tracker.EnableDirtiedLog();
		m_bda_log_verify = BdaDirtyLogVerifyMode();
	}
	m_bda_epoch_skip  = SyncEpoch::Enabled() && ParseEnvU64("KYTY_BDA_SYNC_EPOCH", 1) != 0;
	// The verify mode tells guest writes from missed pages by the fault epoch, which the tracker
	// keeps only with incremental BDA synchronization.
	m_bda_epoch_verify = m_bda_epoch_skip && m_bda_incremental_sync ? BdaEpochVerifyMode() : 0;
	if (SyncEpoch::Enabled() && BindingEpochMemoEnabled()) {
		const bool large      = CpCommit::Enabled(CpCommit::Part::BindSlots);
		m_binding_memo_shift  = large ? 49u : 53u;
		m_binding_memo        = std::make_unique<BindingMemo[]>(large ? BindingMemoSlotsLarge
		                                                              : BindingMemoSlots);
		m_binding_memo_verify = BindingEpochMemoVerifyMode();
		m_binding_memo_cross  = BindingMemoCrossEpochEnabled();
	}
	MemoryTracker::SetFaultAheadOverride(FaultAheadOverridePages());
	m_written_sync_skip = WrittenSyncSkipEnabled();
	if (m_written_sync_skip) {
		m_written_sync_skip_verify = EnvVerifyMode("KYTY_WRITTEN_SYNC_SKIP_VERIFY");
	}
	m_written_epoch_mode = CommitEpochFastPathMode();
	if (m_written_epoch_mode != 0 && m_written_sync_skip) {
		m_written_epoch = std::make_unique<WrittenEpochMemo[]>(WrittenEpochSlots);
	}
	m_false_sharing = FalseSharingWritesEnabled();
	if (m_false_sharing) {
		m_false_sharing_verify = EnvVerifyMode("KYTY_FALSE_SHARING_WRITES_VERIFY");
	}
	if (m_bda_pagetable_buffer.IsSparse()) {
		m_bda_sparse = std::make_unique<SparsePageTable>(m_graphics, m_bda_pagetable_buffer.Handle(),
		                                                 BDA_PAGETABLE_SIZE);
	}
	// Buffers have their own switches: shaders reach guest memory through BDA pointers without a
	// binding, which never touches a buffer's LRU entry, so "unused for N frames" can be wrong for a
	// buffer only BDA code reads (its draws then fault until the buffer is recreated). The image
	// switches (KYTY_VRAM_IDLE_FRAMES, KYTY_VRAM_PRESSURE_FRAMES) do not retire buffers.
	m_idle_frames     = ParseEnvU64("KYTY_VRAM_IDLE_BUFFER_FRAMES", 0);
	m_pressure_frames = ParseEnvU64("KYTY_VRAM_PRESSURE_BUFFER_FRAMES", 0);
	std::memset(m_gds_buffer.Mapped().data(), 0, static_cast<size_t>(m_gds_buffer.Size()));
	m_gds_buffer.Flush(0, m_gds_buffer.Size());
	if (m_upload_dma != nullptr) {
		m_scheduler.SetSubmitDependency(m_upload_dma.get(), 1);
	}
	SetVulkanObjectNameF(m_graphics.device, m_bda_pagetable_buffer.Handle(),
	                     "BDA Page Table Buffer");
	m_scheduler.BeginCommand();
	for (auto* buffer: {&m_bda_pagetable_buffer, m_fault_manager.GetFaultBuffer()}) {
		buffer->Fill(0, buffer->Size(), 0);
	}
	const auto null_id =
	    m_slot_buffers.insert(m_graphics, m_scheduler, MemoryUsage::DeviceLocal, 0, AllFlags, 16);
	EXIT_IF(null_id != NULL_BUFFER_ID);
	SetVulkanObjectNameF(m_graphics.device, GetBuffer(null_id).Handle(), "Kyty.NullBuffer");
	if (SideReadbackEnabled()) {
		// Eager copies reuse the side-readback registry, so they need it.
		const bool eager = EagerReadbackEnabled();
		m_side = std::make_unique<SideReadbackState>(m_graphics, m_scheduler, SideReadbackWindow(),
		                                             eager);
		if (eager) {
			m_eager_enabled      = true;
			m_eager              = EagerReadbackPages(EagerReadbackLimits());
			m_eager_flush_budget = EagerFlushBudget();
		}
	}
	if (!m_graphics.CanReportMemoryUsage()) {
		return;
	}
	constexpr int64_t GiB              = 1024ll * 1024 * 1024;
	constexpr int64_t target_threshold = 8 * GiB;
	const auto        budget =
	    static_cast<int64_t>(std::min<uint64_t>(m_graphics.GetTotalMemoryBudget(), INT64_MAX));
	const auto threshold = std::min(budget, target_threshold);
	const auto expected  = std::min(budget - 6 * threshold / 10, budget - GiB);
	const auto critical  = std::min(budget - 2 * threshold / 10, budget - GiB / 2);
	m_trigger_gc_memory  = static_cast<uint64_t>(std::max<int64_t>(expected, GiB));
	m_critical_gc_memory = static_cast<uint64_t>(std::max<int64_t>(critical, 2 * GiB));
}

BufferCache::~BufferCache() {
	if (g_defer_cache == this) {
		g_defer_cache = nullptr;
	}
	if (m_upload_dma != nullptr && g_write_drain_dma == m_upload_dma.get()) {
		g_write_drain_dma = nullptr;
	}
	if (m_upload_dma != nullptr) {
		// The scheduler has drained by now; no later submission may wait on the upload DMA.
		m_scheduler.SetSubmitDependency(nullptr, 1);
	}
	CompleteAllSideReadbacks();
	if (!m_gpu_modified_ranges.Empty()) {
		EXIT("BufferCache: destroyed with pending GPU-modified ranges\n");
	}
	for (const auto& [vaddr, id]: m_buffers) {
		(void)vaddr;
		const auto& buffer = m_slot_buffers[id];
		if (m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size())) {
			EXIT("BufferCache: destroyed with GPU-modified buffer\n");
		}
	}
	m_buffers.clear();
}

void BufferCache::InvalidateMemory(uint64_t vaddr, uint64_t size, bool write_fault) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid memory-invalidation range\n");
	}
	ForgetKnownFills(vaddr, size);
	const auto flush = [this, vaddr, size] {
		t_write_drain_kind = 0;
		ReadMemory(vaddr, size, true);
		t_write_drain_kind = 1;
	};
	if (!write_fault) {
		m_memory_tracker.InvalidateRegion(vaddr, size, flush);
		return;
	}
	// KYTY_READBACK_SIDE_WRITES: only a fault taken by the exception handler is retried.
	const auto fault_flush = [this, vaddr, size] {
		ReadMemory(vaddr, size, true, FaultCost::InRetriedFault());
	};
	// Pages fault-ahead opens take writes without faulting: forget their fills first, exactly as
	// for the faulting range (never GPU-dirty pages, so fills of GPU-owned ranges survive).
	m_memory_tracker.InvalidateRegionOnWriteFault(
	    vaddr, size, fault_flush,
	    [this](uint64_t address, uint64_t bytes) noexcept { ForgetKnownFills(address, bytes); });
}

BufferCache::UploadBatch::UploadBatch(BufferCache& cache): m_cache(cache) {
	m_cache.m_upload_batch_depth++;
}

BufferCache::UploadBatch::~UploadBatch() {
	if (--m_cache.m_upload_batch_depth == 0 && m_cache.m_scheduler.Active()) {
		auto& command = m_cache.m_scheduler.Current();
		if (command.IsInvalid()) {
			return;
		}
		// The scope exists to record its queued uploads behind one barrier pair. Barriers that
		// were pending for other reasons (a previous draw's shader-write barrier, deferrable image
		// transitions) wait for the next flush point as everywhere else: recording them here ended
		// the rendering instance before every draw that followed a storage-writing draw and so
		// defeated KYTY_DRAW_WRITE_SINK (u42 Sky Garden: ~745 instance splits per flip).
		if (UploadBatchScopedFlushEnabled() && !command.HasPendingUploads()) {
			if (command.HasPendingBarriers()) {
				Profiler::CountFrameEvent(Profiler::FrameEvent::UploadBatchFlushesDeferred);
			}
			return;
		}
		command.FlushBarriers();
	}
}

void BufferCache::AdvanceFrame() noexcept {
	m_memory_tracker.AdvanceFrame();
	// nps5 zerocopy4 (KYTY_ZEROCOPY_READ_OPEN): the per-frame check runs on the GPU thread before a
	// submission. If it did not run for 3 guest frames, nothing vouches for the open pages: close
	// all.
	if (auto& open = ReadOpen();
	    m_memory_tracker.Frame() - open.sweep_frame.load(std::memory_order_relaxed) > 3) {
		std::lock_guard lock(open.mutex);
		for (const auto& [page, frame]: open.pages) {
			(void)m_memory_tracker.CloseRead(page, TRACKER_PAGE_SIZE);
			open.closed_unchecked.fetch_add(1, std::memory_order_relaxed);
		}
		open.pages.clear();
	}
	FaultCost::AdvanceFrame();
	// KYTY_FAULT_AHEAD_ADAPT, applied once per guest flip (the cost model moves slowly).
	MemoryTracker::SetFaultAheadOverride(FaultAheadOverridePages());
}

void BufferCache::EraseHotShadows(uint64_t vaddr, uint64_t size) {
	if (m_hot_shadows.empty()) {
		return;
	}
	auto it = m_hot_shadows.lower_bound(Common::AlignDown(vaddr, TRACKER_PAGE_SIZE));
	while (it != m_hot_shadows.end() && it->first < vaddr + size) {
		it = m_hot_shadows.erase(it);
	}
}

void BufferCache::SettleHotPages(uint64_t vaddr, uint64_t size) {
	if (m_memory_tracker.HotPageCount() == 0) {
		return;
	}
	for (const auto page: m_memory_tracker.SettleHotPages(vaddr, size)) {
		// The page is write-protected now: its contents can no longer change unobserved. If they
		// differ from the last upload (or nothing was uploaded while hot), the CPU wrote since.
		const auto shadow = m_hot_shadows.find(page);
		if (shadow == m_hot_shadows.end() ||
		    std::memcmp(shadow->second.data.get(), LibKernel::Memory::GuestReadPtr(page, TRACKER_PAGE_SIZE),
		                TRACKER_PAGE_SIZE) != 0) {
			m_memory_tracker.MarkRegionAsCpuModified(page, TRACKER_PAGE_SIZE);
		}
		if (shadow != m_hot_shadows.end()) {
			m_hot_shadows.erase(shadow);
		}
	}
}

void BufferCache::MaintainHotPages() {
	const auto frame = m_memory_tracker.Frame();
	if (frame - m_hot_sweep_frame < 8) {
		return;
	}
	m_hot_sweep_frame = frame;
	m_memory_tracker.SweepHotPages(m_hot_quiet_frames);
	std::erase_if(m_hot_shadows, [this, frame](const auto& entry) {
		return frame - entry.second.last_use > m_hot_quiet_frames;
	});
}

void BufferCache::CollectHotPages(Buffer& buffer, std::span<const GuestRange> hot_ranges,
                                  std::vector<vk::BufferCopy>& copies, uint64_t& total_size,
                                  std::vector<uint64_t>& demote, std::vector<uint64_t>& settle) {
	uint64_t hot_bytes = 0;
	for (const auto& range: hot_ranges) {
		hot_bytes += range.size;
	}
	if (m_hot_scratch.size() < hot_bytes) {
		m_hot_scratch.resize(hot_bytes);
	}
	const auto frame     = m_memory_tracker.Frame();
	const auto max_pages = m_memory_tracker.GetFaultPolicy().hot_max;
	uint64_t   staged    = 0;
	uint64_t   visited   = 0;
	uint64_t   skipped   = 0;
	uint64_t   settled   = 0;
	for (const auto& range: hot_ranges) {
		// The shadows of a run's pages, in address order: one lookup per run instead of one per
		// page. `next` stays the first shadow above the current page (an emplace below inserts
		// before it and keeps it valid).
		auto next = m_hot_shadows.lower_bound(range.address);
		for (auto page = range.address; page < range.End(); page += TRACKER_PAGE_SIZE) {
			visited++;
			auto shadow = m_hot_shadows.end();
			if (next != m_hot_shadows.end() && next->first == page) {
				shadow = next++;
			}
			const auto unchanged = [&](const void* contents) {
				return shadow != m_hot_shadows.end() &&
				       std::memcmp(shadow->second.data.get(), contents, TRACKER_PAGE_SIZE) == 0;
			};
			auto* snapshot = m_hot_scratch.data() + staged;
			// KYTY_BDA_HOT_SYNC: nearly every visit finds the page unchanged, so compare the live
			// (hot, hence never GPU-dirty and readable) page first and snapshot only one that
			// differs. A write racing either compare is seen by the next upload, whose shadow still
			// holds the old contents, exactly as a write landing right after a snapshot.
			bool same = m_bda_hot_sync && unchanged(LibKernel::Memory::GuestReadPtr(page, TRACKER_PAGE_SIZE));
			if (!same) {
				// Snapshot the (writable) page once: the compare, the shadow and the upload all use
				// this copy, so the shadow always equals what the buffer receives.
				std::memcpy(snapshot, LibKernel::Memory::GuestReadPtr(page, TRACKER_PAGE_SIZE), TRACKER_PAGE_SIZE);
				same = unchanged(snapshot);
			}
			if (same) {
				skipped++;
				auto& state    = shadow->second;
				state.last_use = frame;
				if (frame - state.last_change > m_hot_quiet_frames) {
					demote.push_back(page);
				} else if (m_hot_check_limit != 0 && ++state.unchanged_checks >= m_hot_check_limit) {
					// Checked far more often than written: back to fault tracking, clean (the
					// buffer holds exactly the shadow, which SettleHotPages compares again once the
					// page is write-protected).
					settle.push_back(page);
					settled++;
				}
				continue;
			}
			if (!m_early_released.empty() && OverlapsUnpublished(page, TRACKER_PAGE_SIZE)) {
				// KYTY_FALSE_SHARING_WRITES: GPU-owned bytes of this page still wait for their
				// publication, and the buffer holds them: upload only the other bytes of the
				// snapshot, and return the page to normal tracking (no shadow of mixed contents).
				ForEachPublishedPart(page, TRACKER_PAGE_SIZE, [&](uint64_t part, uint64_t bytes) {
					copies.emplace_back(total_size + (part - page), buffer.Offset(part), bytes);
				});
				m_false_sharing_totals.upload_splits++;
				Profiler::CountFrameEvent(Profiler::FrameEvent::FalseSharingUploadSplits);
				demote.push_back(page);
				total_size += TRACKER_PAGE_SIZE;
				staged += TRACKER_PAGE_SIZE;
				continue;
			}
			if (shadow == m_hot_shadows.end()) {
				if (m_hot_shadows.size() < max_pages) {
					shadow = m_hot_shadows.emplace(page, HotShadow {}).first;
					shadow->second.data = std::make_unique<uint8_t[]>(TRACKER_PAGE_SIZE);
				} else {
					// No shadow to compare against: back to faulting on writes.
					demote.push_back(page);
				}
			}
			if (shadow != m_hot_shadows.end()) {
				std::memcpy(shadow->second.data.get(), snapshot, TRACKER_PAGE_SIZE);
				shadow->second.last_change      = frame;
				shadow->second.last_use         = frame;
				shadow->second.unchanged_checks = 0;
			}
			copies.emplace_back(total_size, buffer.Offset(page), TRACKER_PAGE_SIZE);
			total_size += TRACKER_PAGE_SIZE;
			staged += TRACKER_PAGE_SIZE;
		}
	}
	MemoryStats::Count(MemoryStats::Counter::HotUploadPages, visited);
	MemoryStats::Count(MemoryStats::Counter::HotUploadSkipped, skipped);
	if (settled != 0) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::HotPageCheckSettles, settled);
		m_range_memo_totals.settles += settled;
	}
}

namespace {

// KYTY_READ_FAULT_DIAG (readFaultDiag.h): one guest read fault of the diagnosed page, printed
// when ReadMemory returns.
class ReadFaultDiagScope {
public:
	ReadFaultDiagScope(uint64_t vaddr, MasterSemaphore& master) {
		m_fault.start_ns    = ReadFaultDiag::NowNs();
		m_fault.fault_vaddr = vaddr;
		auto& state         = ReadFaultDiag::Get();
		m_write_ns          = state.write_ns.load(std::memory_order_acquire);
		m_write_tick        = state.write_tick.load(std::memory_order_relaxed);
		m_write_caller      = state.write_caller.load(std::memory_order_relaxed);
		m_current           = master.CurrentTick();
		master.Refresh();
		m_gpu = master.KnownGpuTick();
		if (const auto& progress = master.GetSubmissionProgress(); progress != nullptr) {
			m_dispatched = progress->dispatched_tick.load(std::memory_order_acquire);
		}
		ReadFaultDiag::t_fault = &m_fault;
	}
	~ReadFaultDiagScope() {
		ReadFaultDiag::t_fault = nullptr;
		auto& state = ReadFaultDiag::Get();
		if (state.lines.fetch_add(1, std::memory_order_relaxed) >= MaxLines()) {
			return;
		}
		const auto end     = ReadFaultDiag::NowNs();
		const auto labels  = ReadFaultDiag::LabelsBetween(m_write_ns, m_fault.start_ns);
		const auto us      = [](uint64_t ns) { return static_cast<double>(ns) / 1e3; };
		const auto age     = [&](uint64_t ns) { return ns != 0 ? us(m_fault.start_ns - ns) : -1.0; };
		const auto publish = m_fault.phases_end_ns != 0 ? end - m_fault.phases_end_ns : 0;
		char       thread[64] {};
		FaultCost::DescribeThread(thread, sizeof(thread));
		const auto rip  = FaultCost::FaultInstruction();
		const auto base = FaultCost::ExeBase();
		{
			// The faulting guest instruction's bytes, once per instruction.
			static std::mutex                   seen_mutex;
			static std::unordered_set<uint64_t> seen;
			std::unique_lock                    seen_lock(seen_mutex);
			if (rip != 0 && seen.size() < 64 && seen.insert(rip).second) {
				seen_lock.unlock();
				uint8_t code[16] {};
				if (HangTrace::TryReadReadable(rip, code, sizeof(code))) {
					std::string hex;
					for (const auto byte: code) {
						hex += fmt::format("{:02x} ", byte);
					}
					std::printf("ReadFaultDiag code 0x%" PRIx64 ": %s\n", rip, hex.c_str());
				}
			}
		}
		std::printf(
		    "ReadFaultDiag t=%.3f thr='%s' rip=0x%" PRIx64 " path=%s off=0x%03" PRIx64
		    " dirty=%d waittick=%" PRIu64 " cur=%" PRIu64 " gpu=%" PRIu64 " disp=%" PRIu64
		    " | total %.1f sendsync %.1f dispatch %.1f gpu %.1f copy %.1f publish %.1f us"
		    " | write age %.1f us tick %" PRIu64 " caller kyty+0x%" PRIx64
		    " | labels after write %" PRIu64 ", first %s 0x%" PRIx64 "=0x%" PRIx64 " age %.1f us,"
		    " last %s 0x%" PRIx64 "=0x%" PRIx64 " age %.1f us, newest any age %.1f us\n",
		    us(m_fault.start_ns) / 1e6, thread, rip, m_path, m_fault.fault_vaddr - ReadFaultDiag::Page(),
		    m_fault.byte_known ? (m_fault.byte_dirty ? 1 : 0) : -1, m_fault.wait_tick, m_current, m_gpu,
		    m_dispatched, us(end - m_fault.start_ns), us(m_fault.sendsync_ns), us(m_fault.dispatch_ns),
		    us(m_fault.gpu_ns), us(m_fault.copy_ns), us(publish), age(m_write_ns), m_write_tick,
		    m_write_caller - std::min(m_write_caller, base), labels.count, labels.first.kind,
		    labels.first.address, labels.first.value, age(labels.first.ns), labels.last.kind,
		    labels.last.address, labels.last.value, age(labels.last.ns), age(labels.last_any_ns));
	}
	ReadFaultDiagScope(const ReadFaultDiagScope&)            = delete;
	ReadFaultDiagScope& operator=(const ReadFaultDiagScope&) = delete;

	void                  SetPath(const char* path) noexcept { m_path = path; }
	ReadFaultDiag::Fault& Fault() noexcept { return m_fault; }

private:
	static uint32_t MaxLines() {
		static const uint32_t lines = [] {
			const auto* value = std::getenv("KYTY_READ_FAULT_DIAG_LINES");
			return value != nullptr ? static_cast<uint32_t>(std::strtoul(value, nullptr, 0)) : 4000u;
		}();
		return lines;
	}

	ReadFaultDiag::Fault m_fault;
	const char*          m_path         = "other";
	uint64_t             m_write_ns     = 0;
	uint64_t             m_write_tick   = 0;
	uint64_t             m_write_caller = 0;
	uint64_t             m_current      = 0;
	uint64_t             m_gpu          = 0;
	uint64_t             m_dispatched   = 0;
};

} // namespace

void BufferCache::ReadMemory(uint64_t vaddr, uint64_t size, bool is_write, bool retried_write) {
	std::optional<ReadFaultDiagScope> diag;
	if (ReadFaultDiag::Covers(vaddr, size) && !is_write && !GuestGpu::IsGpuThread()) {
		diag.emplace(vaddr, m_scheduler.GetMasterSemaphore());
	}
	// KYTY_READ_FAULT_DIAG=auto: the first slow guest read fault (after the load) chooses the page.
	struct AutoChooser {
		uint64_t vaddr = 0;
		uint64_t start = 0;
		~AutoChooser() {
			if (start != 0 && ReadFaultDiag::NowNs() - start > 2'000'000) {
				ReadFaultDiag::AutoChoose(vaddr);
			}
		}
	} auto_chooser;
	if (!diag && !is_write && ReadFaultDiag::AutoPending() && !GuestGpu::IsGpuThread() &&
	    FaultCost::Frame() > 3000) {
		auto_chooser.vaddr = vaddr;
		auto_chooser.start = ReadFaultDiag::NowNs();
	}
	// KYTY_READ_FAULT_PUBLISHED: complete a guest load with the page's published bytes, or (verify)
	// remember what it would have read and compare after the wait below.
	struct PublishedVerify {
		GuestLoad::Load load;
		uint8_t         before[16] {};
		uint32_t        frame = 0;
		~PublishedVerify() {
			uint8_t after[16] {};
			if (!LibKernel::Memory::TryReadBacking(load.address, after, load.size)) {
				return;
			}
			auto&      state = ReadFaultPublished();
			const bool equal = std::memcmp(before, after, load.size) == 0;
			(equal ? state.verify_equal : state.verify_different).fetch_add(1, std::memory_order_relaxed);
			if (!equal && state.logged.fetch_add(1, std::memory_order_relaxed) < 32) {
				std::string text;
				for (uint32_t i = 0; i < load.size; i++) {
					text += fmt::format("{:02x}{:02x} ", before[i], after[i]);
				}
				std::printf("Kyty read-fault published verify: 0x%" PRIx64 " (%u bytes) served/waited "
				            "bytes differ: %s\n",
				            load.address, load.size, text.c_str());
			}
		}
	};
	std::optional<PublishedVerify> published_verify;
	std::optional<BurstWaitTimer>  burst_timer;
	if (const auto published_mode = g_read_fault_published.Get();
	    published_mode != 0 && !is_write && GuestLoad::t_native_context != nullptr &&
	    !GuestGpu::IsGpuThread()) {
		auto&      state = ReadFaultPublished();
		const auto frame = m_memory_tracker.Frame();
		ReportReadFaultPublished(frame);
		const auto      page = Common::AlignDown(vaddr, TRACKER_PAGE_SIZE);
		GuestLoad::Load load;
		uint8_t         bytes[16] {};
		// A copy loop over the page would fault on every load: after a burst, wait (which
		// publishes the page and makes it readable) instead.
		thread_local uint64_t burst_page  = 0;
		thread_local uint64_t burst_ns    = 0;
		thread_local uint32_t burst_count = 0;
		const auto            now_ns      = ReadFaultDiag::NowNs();
		burst_count = burst_page == page && now_ns - burst_ns < 1'000'000 ? burst_count + 1 : 0;
		burst_page  = page;
		burst_ns    = now_ns;
		// The other tracker page of a load that straddles two (0: the load lies in the faulting
		// page). 1: it holds an image's bytes; 2: its GPU-owned bytes were not published recently.
		uint64_t   other_page = 0;
		const auto other_page_state = [&]() -> int {
			if (m_texture_cache.IsRegionGpuModified(other_page, TRACKER_PAGE_SIZE)) {
				return 1;
			}
			if (IsRegionGpuModifiedRelaxed(other_page, TRACKER_PAGE_SIZE) &&
			    !ReadFaultPublishedRecently(other_page, frame)) {
				return 2;
			}
			return 0; // plain CPU memory, or published like the faulting page
		};
		const auto decoded = [&] {
			if (!GuestLoad::Decode(vaddr, load)) {
				return false;
			}
			const auto first = Common::AlignDown(load.address, TRACKER_PAGE_SIZE);
			const auto last  = Common::AlignDown(load.address + load.size - 1, TRACKER_PAGE_SIZE);
			if (first == page && last == page) {
				return true;
			}
			if (!ReadFaultPublishedStraddle() || last - first != TRACKER_PAGE_SIZE ||
			    (first != page && last != page)) {
				return false;
			}
			other_page = first == page ? last : first;
			return true;
		};
		if (const auto burst_limit = ReadFaultPublishedBurst();
		    (burst_limit != 0 && burst_count >= burst_limit) ||
		    ReadFaultFrameBurst::ShouldWait(ReadFaultFrameCounter(), ReadFaultPublishedFrameBurst(), page,
		                                    static_cast<uint32_t>(frame))) {
			state.burst.fetch_add(1, std::memory_order_relaxed);
			burst_timer.emplace();
			burst_timer->start = now_ns;
			if (g_fault_class.On()) {
				(void)FaultClasses().table.NoteRead(page, static_cast<uint32_t>(frame));
			}
		} else if (!decoded()) {
			state.undecodable.fetch_add(1, std::memory_order_relaxed);
		} else if (!IsRegionGpuModifiedRelaxed(page, TRACKER_PAGE_SIZE) ||
		           m_texture_cache.IsRegionGpuModified(page, TRACKER_PAGE_SIZE) ||
		           (other_page != 0 && other_page_state() == 1)) {
			// Not (only) buffer bytes the GPU owns: an image's contents are not in the backing.
			state.image.fetch_add(1, std::memory_order_relaxed);
		} else if (!ReadFaultPublishedRecently(page, frame) ||
		           (other_page != 0 && other_page_state() == 2)) {
			state.stale.fetch_add(1, std::memory_order_relaxed);
		} else if (!LibKernel::Memory::TryReadBacking(load.address, bytes, load.size)) {
			state.other.fetch_add(1, std::memory_order_relaxed);
		} else if (published_mode == 2) {
			published_verify.emplace();
			published_verify->load  = load;
			published_verify->frame = frame;
			std::memcpy(published_verify->before, bytes, load.size);
		} else if (GuestLoad::Apply(load, bytes)) {
			state.served.fetch_add(1, std::memory_order_relaxed);
			if (const auto relax_after = ReadRelaxAfter();
			    relax_after != 0 && other_page == 0 &&
			    ReadRelaxCounter().Bump(page, static_cast<uint32_t>(frame)) >= relax_after) {
				state.relax_calls.fetch_add(1, std::memory_order_relaxed);
				// The host protects 16 KiB pages (KYTY_READ_RELAX_SPAN_KB, default 16): relaxing
				// one 4 KiB tracker page of a host page whose neighbours stay access-watched only
				// turns its faults into alias emulations. Relax the span when every GPU-dirty page
				// of it would be served too (buffer bytes only, published recently); else the page
				// alone.
				const uint64_t span       = ReadRelaxSpan();
				const auto     span_begin = Common::AlignDown(page, span);
				bool           whole      = span > TRACKER_PAGE_SIZE &&
				                            !m_texture_cache.IsRegionGpuModified(span_begin, span);
				for (uint64_t p = span_begin; whole && p < span_begin + span;
				     p += TRACKER_PAGE_SIZE) {
					whole = p == page || !IsRegionGpuModifiedRelaxed(p, TRACKER_PAGE_SIZE) ||
					        ReadFaultPublishedRecently(p, frame);
				}
				state.relaxed_pages.fetch_add(
				    whole ? m_memory_tracker.RelaxRead(span_begin, span)
				          : m_memory_tracker.RelaxRead(page, TRACKER_PAGE_SIZE),
				    std::memory_order_relaxed);
			}
			// nps5 zerocopy4 (KYTY_ZEROCOPY_READ_OPEN): the served page (a buffer page, no image,
			// published recently) stays readable from now on; the per-frame check keeps or closes
			// it. nps5 faultclass (KYTY_FAULT_CLASS): the same, but only for a page read in several
			// frames.
			if (m_eager_enabled &&
			    (g_read_open.On() ||
			     (g_fault_class.On() && other_page == 0 &&
			      FaultClasses().table.NoteRead(page, static_cast<uint32_t>(frame))))) {
				auto&           open = ReadOpen();
				std::lock_guard lock(open.mutex);
				if (open.pages.emplace(page, static_cast<uint32_t>(frame)).second) {
					(void)m_memory_tracker.OpenRead(page, TRACKER_PAGE_SIZE);
					open.opened.fetch_add(1, std::memory_order_relaxed);
					FlightRec::ZcSkip(page, TRACKER_PAGE_SIZE, 5); // reason 5 = read-open
				}
			}
			// Keep the page read-hot so its eager copy keeps publishing the GPU's new bytes.
			(void)m_scheduler.Context().GetGpu().TrySendCommand(
			    [this, page] { NoteEagerRead(page, TRACKER_PAGE_SIZE, false); });
			if (other_page != 0) {
				state.straddled.fetch_add(1, std::memory_order_relaxed);
				(void)m_scheduler.Context().GetGpu().TrySendCommand(
				    [this, other_page] { NoteEagerRead(other_page, TRACKER_PAGE_SIZE, false); });
			}
			if (diag) {
				diag->SetPath("published");
			}
			return;
		} else {
			state.other.fetch_add(1, std::memory_order_relaxed);
		}
	}
	const auto diag_path = [&diag](const char* path) {
		if (diag) {
			diag->SetPath(path);
		}
	};
	HangWatchdog::Scope       read("buffer-readback", vaddr, size, 0, 0, is_write);
	Profiler::ScopedFrameWait frame_wait(Profiler::FrameWait::ReadMemory);
	if (!GuestGpu::IsGpuThread() && CommandScheduler::InDeferredOperation()) {
		EXIT("unsupported buffer readback from an asynchronous GPU completion, "
		     "addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
		     vaddr, size);
	}
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid readback range\n");
	}
	const auto      trace_start = HangTrace::Enabled() ? HangTrace::NowNs() : 0;
	ReadMemoryTrace trace;
	const auto      record = [&](std::optional<HangTrace::ReadbackKind> kind) {
		if (!HangTrace::Enabled()) {
			return;
		}
		const auto previous = HangTrace::GetReadbackKind();
		if (kind) {
			HangTrace::SetReadbackKind(*kind);
		}
		HangTrace::RecordReadback(vaddr, size, trace.begin, trace.size, trace.downloaded,
		                          HangTrace::NowNs() - trace_start);
		HangTrace::SetReadbackKind(previous);
	};
	auto&      gpu        = m_scheduler.Context().GetGpu();
	const auto page_begin = Common::AlignDown(vaddr, TRACKER_PAGE_SIZE);
	const auto page_end   = Common::AlignUp(vaddr + size, TRACKER_PAGE_SIZE);

	// Writes need the CPU-dirty transition of the drain path (or, KYTY_READBACK_SIDE_WRITES, of
	// a second fault). Guest-thread reads use side copies; GPU-thread reads do too
	// (KYTY_READBACK_SIDE_GPU_THREAD, default on) and wait for the copy themselves: it waits only
	// for the producing recording, not for the current one, and the current recording is neither
	// split nor submitted.
	const bool gpu_thread = GuestGpu::IsGpuThread();
	// nps5 faultclass: a guest store on a read-open page that waits here (for a pending publication
	// or a drain) is a store conflict: with the read watch on, the reader's own wait would have
	// cleaned the page before the store. The table counts them; the per-frame check closes a page
	// it made Exact (never here: this can run inside the tracker's write-fault callback).
	struct StoreConflict {
		uint64_t page  = 0;
		uint64_t start = 0;
		uint32_t frame = 0;
		~StoreConflict() {
			const auto waited = ReadFaultDiag::NowNs() - start;
			if (waited >= 500'000) {
				auto& cls = FaultClasses();
				cls.store_conflict_ns.fetch_add(waited, std::memory_order_relaxed);
				(void)cls.table.NoteStoreConflict(page, frame);
			}
		}
	};
	std::optional<StoreConflict> store_conflict;
	if (is_write && !gpu_thread && g_fault_class.On()) {
		auto& open   = ReadOpen();
		bool  opened = false;
		{
			std::lock_guard lock(open.mutex);
			opened = open.pages.contains(page_begin);
		}
		if (opened) {
			store_conflict.emplace();
			store_conflict->page  = page_begin;
			store_conflict->start = ReadFaultDiag::NowNs();
			store_conflict->frame = static_cast<uint32_t>(m_memory_tracker.Frame());
		}
	}
	const bool side_write = is_write && retried_write && !gpu_thread && m_side != nullptr &&
	                        g_readback_side_writes.On();
	const bool side_path  = m_side != nullptr && (!is_write || side_write) &&
	                       (!gpu_thread || SideReadbackGpuThreadEnabled());
	if (OverlapsPendingSideReadback(page_begin, page_end)) {
		// Another fault already copies these pages (or an eager copy publishes them): wait for
		// (or finish) its publication instead of copying again. Writes and GPU-thread reads must
		// also be ordered after it.
		uint32_t eager = 0;
		{
			Profiler::ScopedFrameWait side_wait(Profiler::FrameWait::ReadbackSideWait);
			eager = CompleteSideReadbacks(page_begin, page_end - page_begin);
		}
		if (eager != 0) {
			Profiler::CountFrameEvent(Profiler::FrameEvent::ReadbackEagerWaits);
		}
		// A GPU-thread read continues below: bytes re-dirtied since that copy was issued still
		// need their own readback before the caller reads.
		if (side_path && !gpu_thread) {
			diag_path(eager != 0 ? "pending-eager" : "pending-side");
			// If a newer writer re-dirtied the page meanwhile it stays protected, and the
			// retried access faults into a fresh readback.
			if (eager == 0) {
				Profiler::CountFrameEvent(Profiler::FrameEvent::ReadbackSideDuplicateWaits);
			}
			record(eager != 0 ? HangTrace::ReadbackKind::FaultReadEager
			                  : HangTrace::ReadbackKind::FaultReadDuplicate);
			return;
		}
		if (side_path && eager != 0 &&
		    !m_memory_tracker.IsRegionGpuModified(page_begin, page_end - page_begin)) {
			// The eager copy published every GPU-owned byte of these pages: nothing is left.
			record(std::nullopt);
			return;
		}
	}
	if (!side_path) {
		diag_path("drain-nonside");
		std::optional<WriteDrainTimer> drain_timer;
		if (is_write) {
			drain_timer.emplace();
			drain_timer->kind = gpu_thread                ? 3
			                    : t_write_drain_kind == 0 ? 0
			                    : retried_write           ? 2
			                                              : 1;
		}
		gpu.SendCommandSync([&, this, vaddr, size, is_write, gpu_thread] {
			if (!is_write || g_readback_eager_writes.On()) {
				NoteEagerRead(vaddr, size, gpu_thread);
			}
			CpGaps::Note(is_write ? CpGaps::Event::DrainWrite : CpGaps::Event::DrainOther);
			ReadMemoryDrain(vaddr, size, is_write, trace);
		});
		record(std::nullopt);
		return;
	}

	if (gpu_thread) {
		NoteEagerRead(vaddr, size, true);
		if (FaultPagesCurrent(page_begin, page_end)) { // KYTY_CLEAN_PAGE
			record(std::nullopt);
			return;
		}
		std::shared_ptr<SideReadback> issued;
		const auto                    result = TryIssueSideReadback(vaddr, size, issued);
		if (result == SideIssueResult::Issued) {
			{
				Profiler::ScopedFrameWait side_wait(Profiler::FrameWait::GpuWaitSideCopy);
				CompleteSideReadback(*issued);
			}
			Profiler::CountFrameEvent(Profiler::FrameEvent::ReadbackGpuThreadSideCopies);
			trace.begin      = issued->begin;
			trace.size       = issued->end - issued->begin;
			trace.downloaded = true;
			record(std::nullopt);
			return;
		}
		switch (result) {
			case SideIssueResult::CurrentWriter:
				Profiler::CountFrameEvent(Profiler::FrameEvent::ReadbackSideFallbackCurrentWriter);
				break;
			case SideIssueResult::Unbounded:
				Profiler::CountFrameEvent(Profiler::FrameEvent::ReadbackSideFallbackUnbounded);
				break;
			default: Profiler::CountFrameEvent(Profiler::FrameEvent::ReadbackSideFallbackOther); break;
		}
		Profiler::CountFrameEvent(Profiler::FrameEvent::ReadbackGpuThreadDrains);
		CpGaps::Note(CpGaps::Event::DrainGpuThread);
		ReadMemoryDrain(vaddr, size, false, trace);
		record(std::nullopt);
		return;
	}

	std::shared_ptr<SideReadback> issued;
	auto                          result    = SideIssueResult::Other;
	const auto                    sync_mark = diag ? ReadFaultDiag::NowNs() : 0;
	const auto                    write_mark = is_write ? ReadFaultDiag::NowNs() : 0;
	bool                          current    = false; // KYTY_CLEAN_PAGE
	gpu.SendCommandSync([&, this, vaddr, size, is_write] {
		if (!is_write || g_readback_eager_writes.On()) {
			NoteEagerRead(vaddr, size, false);
		}
		// KYTY_CLEAN_PAGE: another thread's read made the faulting pages current meanwhile; a
		// side copy would only take the window's other GPU-written ranges, and this thread would
		// wait for it.
		if (FaultPagesCurrent(page_begin, page_end)) {
			if (is_write) {
				m_memory_tracker.MarkRegionAsCpuModified(vaddr, size);
			}
			current = true;
			return;
		}
		result = TryIssueSideReadback(vaddr, size, issued);
		if ((result == SideIssueResult::CurrentWriter || result == SideIssueResult::Unbounded) &&
		    (g_readback_flush_side.On() || is_write)) {
			// KYTY_READBACK_FLUSH_SIDE (reads) / KYTY_READBACK_SIDE_WRITES: submit the writing
			// recording, then copy aside.
			CpGaps::Note(CpGaps::Event::SideFlushes);
			(void)m_scheduler.FlushObservable();
			result = TryIssueSideReadback(vaddr, size, issued);
			if (result == SideIssueResult::Issued) {
				CpGaps::Note(CpGaps::Event::SideFlushCopies);
				g_readback_flush_side_copies.fetch_add(1, std::memory_order_relaxed);
			}
		}
		if (result == SideIssueResult::Issued && is_write) {
			g_readback_side_writes_copies.fetch_add(1, std::memory_order_relaxed);
		}
		if (result != SideIssueResult::Issued && result != SideIssueResult::Pending) {
			CpGaps::Note(is_write                                   ? CpGaps::Event::DrainWrite
			             : result == SideIssueResult::CurrentWriter ? CpGaps::Event::DrainCurrentWriter
			             : result == SideIssueResult::Unbounded     ? CpGaps::Event::DrainUnbounded
			                                                        : CpGaps::Event::DrainOther);
			ReadMemoryDrain(vaddr, size, is_write, trace);
		}
	});
	if (is_write && result != SideIssueResult::Issued && result != SideIssueResult::Pending) {
		auto& state = ReadFaultPublished();
		state.write_drains[2].fetch_add(1, std::memory_order_relaxed);
		state.write_drain_ns[2].fetch_add(ReadFaultDiag::NowNs() - write_mark,
		                                  std::memory_order_relaxed);
	}
	if (current) {
		diag_path("current");
		record(std::nullopt);
		return;
	}
	if (diag) {
		diag->Fault().sendsync_ns = ReadFaultDiag::NowNs() - sync_mark;
		switch (result) {
			case SideIssueResult::Issued: diag_path("side-issued"); break;
			case SideIssueResult::Pending: diag_path("pending-after-issue"); break;
			case SideIssueResult::CurrentWriter: diag_path("drain-current-writer"); break;
			case SideIssueResult::Unbounded: diag_path("drain-unbounded"); break;
			case SideIssueResult::Other: diag_path("drain-other"); break;
		}
	}
	switch (result) {
		case SideIssueResult::Issued: {
			// The GPU thread returned right after submitting; this guest thread waits.
			{
				Profiler::ScopedFrameWait side_wait(Profiler::FrameWait::ReadbackSideWait);
				CompleteSideReadback(*issued);
			}
			trace.begin      = issued->begin;
			trace.size       = issued->end - issued->begin;
			trace.downloaded = true;
			record(HangTrace::ReadbackKind::FaultReadSide);
			return;
		}
		case SideIssueResult::Pending: {
			// Another thread's side copy (or an eager copy) of the faulting page was issued after
			// the check above.
			uint32_t eager = 0;
			{
				Profiler::ScopedFrameWait side_wait(Profiler::FrameWait::ReadbackSideWait);
				eager = CompleteSideReadbacks(page_begin, page_end - page_begin);
			}
			Profiler::CountFrameEvent(eager != 0 ? Profiler::FrameEvent::ReadbackEagerWaits
			                                     : Profiler::FrameEvent::ReadbackSideDuplicateWaits);
			record(eager != 0 ? HangTrace::ReadbackKind::FaultReadEager
			                  : HangTrace::ReadbackKind::FaultReadDuplicate);
			return;
		}
		case SideIssueResult::CurrentWriter:
			Profiler::CountFrameEvent(Profiler::FrameEvent::ReadbackSideFallbackCurrentWriter);
			break;
		case SideIssueResult::Unbounded:
			Profiler::CountFrameEvent(Profiler::FrameEvent::ReadbackSideFallbackUnbounded);
			break;
		case SideIssueResult::Other:
			Profiler::CountFrameEvent(Profiler::FrameEvent::ReadbackSideFallbackOther);
			break;
	}
	record(std::nullopt);
}

uint64_t BufferCache::ReadbackFlushSideCopies() noexcept {
	return g_readback_flush_side_copies.load(std::memory_order_relaxed);
}

uint64_t BufferCache::ReadbackSideWriteCopies() noexcept {
	return g_readback_side_writes_copies.load(std::memory_order_relaxed);
}

bool BufferCache::TryLiftFaultPages(uint64_t page_begin, uint64_t page_end) {
	const auto bytes = page_end - page_begin;
	// Still GPU-written for the tracker, but no byte left to download or to wait for.
	if (!m_memory_tracker.IsRegionGpuModified(page_begin, bytes) ||
	    HasGpuDirtyBytes(page_begin, bytes) || HasPendingBackingPublication(page_begin, bytes) ||
	    OverlapsPendingSideReadback(page_begin, page_end) ||
	    m_texture_cache.IsRegionGpuModified(page_begin, bytes)) {
		return false;
	}
	// An address-writing shader still running may write bytes of these pages nobody tracks yet.
	if (m_unbounded_write_tick != 0 && !m_scheduler.IsFree(m_unbounded_write_tick)) {
		return false;
	}
	m_memory_tracker.UnmarkRegionAsGpuModified(page_begin, bytes);
	static std::atomic<uint64_t> lifted {0};
	if (const auto n = lifted.fetch_add(1, std::memory_order_relaxed) + 1;
	    (n & (n - 1)) == 0 && n >= 64) {
		LOGF("ReadMemory: lifted the protection of %" PRIu64 " faulting page ranges without GPU-"
		     "written bytes, no download (KYTY_FAULT_PAGE_LIFT)\n",
		     n);
	}
	return true;
}

bool BufferCache::FaultPagesCurrent(uint64_t page_begin, uint64_t page_end) {
	// KYTY_FAULT_PAGE_LIFT (default 0; 1 on, read once): see TryLiftFaultPages.
	static const bool lift    = ParseEnvU64("KYTY_FAULT_PAGE_LIFT", 0) != 0;
	static const bool enabled = ParseEnvU64("KYTY_CLEAN_PAGE", 0) != 0;
	if (page_end <= page_begin) {
		return false;
	}
	if (lift && TryLiftFaultPages(page_begin, page_end)) {
		return true;
	}
	if (!enabled) {
		return false;
	}
	const auto bytes = page_end - page_begin;
	if (m_memory_tracker.IsRegionGpuModified(page_begin, bytes) ||
	    HasGpuDirtyBytes(page_begin, bytes) || HasPendingBackingPublication(page_begin, bytes) ||
	    OverlapsPendingSideReadback(page_begin, page_end) ||
	    m_texture_cache.IsRegionGpuModified(page_begin, bytes)) {
		return false;
	}
	static std::atomic<uint64_t> current {0};
	if (const auto n = current.fetch_add(1, std::memory_order_relaxed) + 1;
	    (n & (n - 1)) == 0 && n >= 64) {
		LOGF("ReadMemory: %" PRIu64 " faults on pages another read had made current went on at "
		     "once (KYTY_CLEAN_PAGE)\n",
		     n);
	}
	return true;
}

void BufferCache::ReadMemoryDrain(uint64_t vaddr, uint64_t size, bool is_write,
                                  ReadMemoryTrace& trace) {
	EXIT_IF(!GuestGpu::IsGpuThread());
	HangWatchdog::Scope drain("buffer-readback-drain", vaddr, size, 0, 0, is_write);
	if (is_write && !IsRegionRegistered(vaddr, size)) {
		return;
	}
	// KYTY_CLEAN_PAGE: nothing of these pages is left to download.
	if (FaultPagesCurrent(Common::AlignDown(vaddr, TRACKER_PAGE_SIZE),
	                      Common::AlignUp(vaddr + size, TRACKER_PAGE_SIZE))) {
		if (is_write) {
			m_memory_tracker.MarkRegionAsCpuModified(vaddr, size);
		}
		return;
	}
	if (is_write && m_false_sharing && TryFalseSharingWrite(vaddr, size, trace)) {
		return;
	}
	auto& buffer = m_slot_buffers[FindBuffer(vaddr, size)];

	// Widen nearby CPU reads so they share one GPU drain.
	const uint64_t     WindowSize   = is_write ? WriteFaultWindow() : 512 * 1024;
	const auto         buffer_begin = buffer.CpuAddress();
	const auto         buffer_end   = buffer_begin + buffer.Size();
	const auto window_begin = std::max(Common::AlignDown(vaddr, WindowSize), buffer_begin);
	const auto window_end = std::min(std::max(window_begin + WindowSize, vaddr + size), buffer_end);

	trace.begin = window_begin;
	trace.size  = window_end - window_begin;
	if (DownloadBufferMemory(buffer, window_begin, window_end - window_begin)) {
		trace.downloaded = true;
		const auto                    tick = m_scheduler.CurrentTick();
		Profiler::ScopedGpuWaitReason wait_reason(Profiler::FrameWait::GpuWaitDrain);
		m_scheduler.Wait(tick);
		m_scheduler.WaitPriorityOperations(tick);
		m_memory_tracker.UnmarkRegionAsGpuModified(window_begin, window_end - window_begin);
	}
	if (is_write) {
		m_memory_tracker.MarkRegionAsCpuModified(vaddr, size);
	}
}

bool BufferCache::TryFalseSharingWrite(uint64_t vaddr, uint64_t size, ReadMemoryTrace& trace) {
	// KYTY_FALSE_SHARING_WRITES (bufferCache.h). The written bytes must be none the GPU owns...
	if (m_gpu_modified_ranges.Intersects(vaddr, size) || OverlapsUnpublished(vaddr, size)) {
		return false;
	}
	// ...and no address-writing shader may still be writing bytes nobody tracks.
	if (m_unbounded_write_tick != 0 && !m_scheduler.IsFree(m_unbounded_write_tick)) {
		return false;
	}
	// Never create a buffer here: GPU-dirty bytes always live in a registered buffer, and one
	// buffer owns every caching page it overlaps, hence the whole tracker page.
	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	if (owner == nullptr || !*owner || IsBufferInvalid(*owner)) {
		return false;
	}
	auto&      buffer     = m_slot_buffers[*owner];
	const auto page_begin = Common::AlignDown(vaddr, TRACKER_PAGE_SIZE);
	const auto page_end   = Common::AlignUp(vaddr + size, TRACKER_PAGE_SIZE);
	if (!buffer.IsInBounds(page_begin, page_end - page_begin)) {
		return false;
	}
	// An image over the pages would refresh from guest memory once invalidated by this fault.
	{
		std::scoped_lock lock {m_texture_cache.m_lock};
		if (!m_texture_cache.FindImagesInRegion(page_begin, page_end - page_begin, true).empty()) {
			return false;
		}
	}
	// Pages the GPU owns without exact bytes (none to preserve) take the ordinary path, which
	// then releases them without waiting.
	if (!m_gpu_modified_ranges.Intersects(page_begin, page_end - page_begin)) {
		return false;
	}
	auto early    = std::make_shared<EarlyReleasedDownload>();
	early->verify = m_false_sharing_verify != 0;
	if (!DownloadBufferMemory(buffer, page_begin, page_end - page_begin, early)) {
		return false;
	}
	// Release the pages now: the download above is recorded behind every GPU writer of their
	// bytes, and its publication writes only those bytes. Uploads skip them until it lands.
	m_memory_tracker.UnmarkRegionAsGpuModified(page_begin, page_end - page_begin);
	m_memory_tracker.MarkRegionAsCpuModified(vaddr, size);
	uint64_t bytes = 0;
	for (const auto& range: early->ranges) {
		bytes += range.size;
	}
	m_early_released.push_back(std::move(early));
	// Submit the recording that copies them soon (within the eager-flush budget): the sooner it
	// completes, the shorter the time guest reads of those bytes see their old contents.
	if (m_eager_flush_budget != 0) {
		m_eager_flush = true;
	}
	m_false_sharing_totals.writes++;
	m_false_sharing_totals.bytes += bytes;
	Profiler::CountFrameEvent(Profiler::FrameEvent::FalseSharingWrites);
	Profiler::CountFrameEvent(Profiler::FrameEvent::FalseSharingBytes, bytes);
	trace.begin      = page_begin;
	trace.size       = page_end - page_begin;
	trace.downloaded = true;
	return true;
}

BufferCache::SideIssueResult BufferCache::TryIssueSideReadback(
    uint64_t vaddr, uint64_t size, std::shared_ptr<SideReadback>& issued) {
	EXIT_IF(!GuestGpu::IsGpuThread() || m_side == nullptr);
	FlightRec::Rec(FlightRec::EvSideReadback, vaddr, size, m_scheduler.CurrentTick());
	auto&      side    = *m_side;
	const auto current = m_scheduler.CurrentTick();
	// An address-writing shader in the unsubmitted recording may write any buffer byte.
	if (m_unbounded_write_tick == current) {
		return SideIssueResult::Unbounded;
	}
	// Why a read is refused as Other (ReadbackSideOther* counters).
	const auto other = [](Profiler::FrameEvent cause) {
		Profiler::CountFrameEvent(cause);
		return SideIssueResult::Other;
	};
	// Never create a buffer here: GPU-dirty bytes always live in a registered buffer.
	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	if (owner == nullptr || !*owner || IsBufferInvalid(*owner)) {
		return other(Profiler::FrameEvent::ReadbackSideOtherOwner);
	}
	auto& buffer = m_slot_buffers[*owner];
	if (!buffer.IsInBounds(vaddr, size)) {
		return other(Profiler::FrameEvent::ReadbackSideOtherOwner);
	}
	const auto buffer_begin = buffer.CpuAddress();
	const auto buffer_end   = buffer_begin + buffer.Size();
	const auto page_begin   = std::max(Common::AlignDown(vaddr, TRACKER_PAGE_SIZE), buffer_begin);
	const auto page_end = std::min(Common::AlignUp(vaddr + size, TRACKER_PAGE_SIZE), buffer_end);
	const auto aligned  = Common::AlignDown(vaddr, side.window);
	const auto window_begin = std::max(aligned, buffer_begin);
	const auto window_end   = std::min(aligned + side.window, buffer_end);
	if (((page_begin | page_end | window_begin | window_end) % TRACKER_PAGE_SIZE) != 0) {
		return other(Profiler::FrameEvent::ReadbackSideOtherAlignment);
	}

	// Prefer the aligned window (neighbouring polled values share one copy); fall back to the
	// faulting pages when a window byte is not eligible (e.g. written by the current recording).
	struct Candidate {
		uint64_t begin;
		uint64_t end;
	};
	const std::array<Candidate, 2> candidates {{{window_begin, window_end}, {page_begin, page_end}}};
	std::vector<GuestRange>        dirty;
	std::optional<Candidate>       chosen;
	uint64_t                       producer = 0;
	auto                           reason   = SideIssueResult::Other;
	// The cause of an Other result: no candidate fits the window unless one says otherwise.
	auto other_cause = Profiler::FrameEvent::ReadbackSideOtherWindow;
	for (size_t index = 0; index < candidates.size(); ++index) {
		const auto& candidate = candidates[index];
		if (index != 0 && candidate.begin == candidates[0].begin &&
		    candidate.end == candidates[0].end) {
			break;
		}
		if (candidate.begin > page_begin || candidate.end < page_end ||
		    candidate.end - candidate.begin > side.window) {
			continue;
		}
		if (OverlapsPendingSideReadback(candidate.begin, candidate.end)) {
			reason = SideIssueResult::Pending;
			continue;
		}
		// A queued drain/texture publication decides these bytes' final backing contents;
		// publishing next to it could reorder the backing writes.
		if (HasPendingBackingPublication(candidate.begin, candidate.end - candidate.begin)) {
			reason      = SideIssueResult::Other;
			other_cause = Profiler::FrameEvent::ReadbackSideOtherPublication;
			continue;
		}
		dirty.clear();
		uint64_t newest = m_write_tick_floor;
		m_gpu_modified_ranges.ForEachInRange(
		    candidate.begin, candidate.end - candidate.begin, [&](uint64_t start, uint64_t end) {
			    dirty.push_back({start, end - start});
			    newest = std::max(newest, m_write_ticks.MaxTick(start, end - start));
		    });
		if (dirty.empty()) {
			reason      = SideIssueResult::Other;
			other_cause = Profiler::FrameEvent::ReadbackSideOtherNoDirty;
			continue;
		}
		if (newest >= current) {
			reason = SideIssueResult::CurrentWriter;
			continue;
		}
		chosen   = candidate;
		producer = newest;
		break;
	}
	if (!chosen && reason == SideIssueResult::Other) {
		Profiler::CountFrameEvent(other_cause);
	}
	if (!chosen) {
		return reason;
	}

	uint32_t slot_index = SideReadbackState::SlotCount;
	for (uint32_t index = 0; index < SideReadbackState::SlotCount; ++index) {
		if (!side.slots[index].busy.load(std::memory_order_acquire)) {
			slot_index = index;
			break;
		}
	}
	if (slot_index == SideReadbackState::SlotCount) {
		return other(Profiler::FrameEvent::ReadbackSideOtherSlot);
	}
	auto&      slot         = side.slots[slot_index];
	const auto staging_base = uint64_t {slot_index} * side.window;

	auto readback    = std::make_shared<SideReadback>();
	readback->begin  = chosen->begin;
	readback->end    = chosen->end;
	readback->slot   = slot_index;
	readback->ranges = dirty;
	std::vector<vk::BufferCopy> copies;
	copies.reserve(dirty.size());
	uint64_t bytes = 0;
	for (const auto& range: dirty) {
		copies.emplace_back(buffer.Offset(range.address),
		                    staging_base + (range.address - chosen->begin), range.size);
		bytes += range.size;
	}

	// Ownership of the exact dirty bytes moves to the publication registered below, exactly as
	// in DownloadBufferMemory. The tracker pages stay GPU-owned (protected) until completion.
	for (const auto& range: dirty) {
		CleanVerdict::Invalidate(range.address, range.size, Coherence::Source::BufferDirtySubtract);
	}
	for (const auto& range: dirty) {
		m_gpu_range_removals.fetch_add(1, std::memory_order_acq_rel);
		m_gpu_modified_ranges.Subtract(range.address, range.size);
	}
	m_memory_tracker.MarkReadbackPending(chosen->begin, chosen->end - chosen->begin);
	readback->publication = BeginBackingPublication(dirty, producer);

	const auto                 command = slot.command;
	vk::CommandBufferBeginInfo begin_info {};
	begin_info.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
	RequireVulkanSuccess(command.begin(&begin_info), "begin side-readback command buffer");
	if (GpuBreadcrumbs::Active()) {
		GpuBreadcrumbs::Write(command, GpuBreadcrumbs::Queue::Side,
		                      GpuBreadcrumbs::Note(GpuBreadcrumbs::Queue::Side,
		                                           {.op   = GpuBreadcrumbs::OpSideReadback,
		                                            .arg0 = static_cast<uint32_t>(bytes),
		                                            .arg4 = chosen->begin,
		                                            .tick = producer}));
	}
	// The timeline wait below orders the producer (and, on the side queue, the newest unbounded
	// writer). On a shared queue this barrier's first scope additionally covers every earlier
	// submission, so the copy observes all submitted work. On the side queue later queue-0 work
	// can overwrite copied bytes while the copy runs; such a writer re-adds them as GPU-dirty
	// and keeps their pages protected (UnmarkReadbackPending retains them), so the possibly
	// torn published bytes are never read before a newer readback replaces them.
	vk::BufferMemoryBarrier before {};
	before.srcAccessMask       = vk::AccessFlagBits::eMemoryWrite;
	before.dstAccessMask       = vk::AccessFlagBits::eTransferRead;
	before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.buffer              = buffer.Handle();
	before.offset              = buffer.Offset(chosen->begin);
	before.size                = chosen->end - chosen->begin;
	command.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                        vk::PipelineStageFlagBits::eTransfer, {}, 0, nullptr, 1, &before, 0,
	                        nullptr);
	// Nearby dirty ranges share one region (KYTY_READBACK_MERGE_GAP_KB): the staging window mirrors
	// the guest window, and completion writes back only readback->ranges.
	const auto regions =
	    MergeLinearCopies(copies, static_cast<uint64_t>(g_readback_merge_gap.Get()));
	RecordReadbackRegions(copies, regions);
	command.copyBuffer(buffer.Handle(), side.staging->Handle(), static_cast<uint32_t>(regions.size()),
	                   regions.data());
	vk::BufferMemoryBarrier after = before;
	after.srcAccessMask           = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask           = vk::AccessFlagBits::eHostRead;
	after.buffer                  = side.staging->Handle();
	after.offset                  = staging_base;
	after.size                    = side.window;
	command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost,
	                        {}, 0, nullptr, 1, &after, 0, nullptr);
	if (GpuBreadcrumbs::Active()) {
		GpuBreadcrumbs::Write(command, GpuBreadcrumbs::Queue::Side,
		                      GpuBreadcrumbs::Note(GpuBreadcrumbs::Queue::Side,
		                                           {.op = GpuBreadcrumbs::OpSideReadbackEnd, .tick = producer}));
	}
	RequireVulkanSuccess(command.end(), "end side-readback command buffer");

	const auto value = ++side.next_value;
	readback->value  = value;
	slot.busy.store(true, std::memory_order_relaxed);

	// On the separate side queue nothing but the timeline wait orders the copy after queue-0
	// work, so it must also cover the newest unbounded (address) writer, whose bytes carry no
	// writer tick. It is older than `current` here (checked above), hence already submitted.
	const bool side_queue = m_graphics.side_queue != nullptr;
	if (side_queue) {
		producer = std::max(producer, m_unbounded_write_tick);
	}
	readback->producer = producer;
	const auto                      master     = m_scheduler.GetMasterSemaphore().Handle();
	const uint64_t                  wait_value = producer;
	const vk::PipelineStageFlags    wait_stage = vk::PipelineStageFlagBits::eTransfer;
	const uint32_t                  wait_count = producer != 0 ? 1u : 0u;
	vk::TimelineSemaphoreSubmitInfo timeline {};
	timeline.waitSemaphoreValueCount   = wait_count;
	timeline.pWaitSemaphoreValues      = &wait_value;
	timeline.signalSemaphoreValueCount = 1;
	timeline.pSignalSemaphoreValues    = &value;
	vk::SubmitInfo submit {};
	submit.pNext                = &timeline;
	submit.waitSemaphoreCount   = wait_count;
	submit.pWaitSemaphores      = &master;
	submit.pWaitDstStageMask    = &wait_stage;
	submit.commandBufferCount   = 1;
	submit.pCommandBuffers      = &command;
	submit.signalSemaphoreCount = 1;
	submit.pSignalSemaphores    = &side.semaphore;
	vk::Result submit_result;
	// KYTY_CP_RECORDER: older ticks may still be in the recorder's ring. Hand them to the queue or
	// the broker first (without queue_mutex, which the recorder may need): the producer before a
	// side-queue wait on it, every older tick before a copy on the shared queue.
	m_scheduler.WaitRecorded(side_queue ? producer : current - 1);
	if (side_queue) {
		// Every tick older than the current recording was handed to queue 0 or to the submission
		// broker. Drain the broker only if the producer has not reached the driver yet, so this
		// wait is never submitted ahead of its signal; only up to the producer, after waiting
		// without queue_mutex for the work its batches read (SubmitDependency).
		const auto& progress = m_scheduler.GetMasterSemaphore().GetSubmissionProgress();
		if (progress != nullptr &&
		    progress->dispatched_tick.load(std::memory_order_acquire) < producer) {
			m_graphics.submission_queue.WaitPendingDependencies(progress.get(), producer);
			Common::LockGuard lock(m_graphics.queue_mutex);
			m_graphics.submission_queue.DrainThroughLocked(progress.get(), producer);
		}
		Common::LockGuard lock(m_graphics.side_queue_mutex);
		HangWatchdog::Scope native(
		    "vkQueueSubmit-side-readback",
		    reinterpret_cast<uint64_t>(static_cast<VkQueue>(m_graphics.side_queue)), value,
		    producer);
		NoteWatchdogSubmit(m_graphics.side_queue, submit);
		submit_result = m_graphics.side_queue.submit(1, &submit, nullptr);
		Profiler::CountFrameEvent(Profiler::FrameEvent::ReadbackSideQueueCopies);
	} else {
		// Every tick older than the current recording was handed to the queue or to the
		// submission broker (drained here first), so the producer is submitted before this
		// copy waits on it. The current recording follows this copy in submission order.
		Common::LockGuard lock(m_graphics.queue_mutex);
		m_graphics.submission_queue.DrainPendingLocked();
		HangWatchdog::Scope native(
		    "vkQueueSubmit-shared-readback",
		    reinterpret_cast<uint64_t>(static_cast<VkQueue>(m_graphics.queue)), value, producer);
		NoteWatchdogSubmit(m_graphics.queue, submit);
		submit_result = m_graphics.queue.submit(1, &submit, nullptr);
	}
	RequireVulkanSuccess(submit_result, "submit side readback");

	{
		std::lock_guard lock(side.pending_mutex);
		side.pending.push_back(readback);
		side.pending_count.store(side.pending.size(), std::memory_order_release);
	}
	Profiler::CountFrameEvent(Profiler::FrameEvent::ReadbackSideCopies);
	Profiler::CountFrameEvent(Profiler::FrameEvent::ReadbackSideCopyBytes, bytes);
	issued = std::move(readback);
	return SideIssueResult::Issued;
}

// KYTY_READ_FAULT_DIAG: what a publication of the diagnosed page wrote (its first 16).
static void DiagPrintPublication(const std::vector<GuestRange>& ranges, uint64_t begin, bool eager,
                                 const uint8_t* staged) {
	auto& state = ReadFaultDiag::Get();
	if (state.publishes.fetch_add(1, std::memory_order_relaxed) >= 16) {
		return;
	}
	const auto  page = ReadFaultDiag::Page();
	std::string text;
	uint64_t    bytes = 0;
	for (const auto& range: ranges) {
		const auto first = std::max(range.address, page);
		const auto last  = std::min(range.address + range.size, page + ReadFaultDiag::RangeSize());
		if (first >= last) {
			continue;
		}
		bytes += last - first;
		if (text.size() < 400) {
			text += fmt::format(" +0x{:x}:{}", first - page, last - first);
		}
	}
	std::string hex;
	for (const auto& range: ranges) {
		const auto first = std::max(range.address, page);
		const auto last  = std::min(range.address + range.size, page + ReadFaultDiag::RangeSize());
		for (auto a = first; a < last && hex.size() < 3 * 64; a++) {
			hex += fmt::format("{:02x} ", staged[a - begin]);
		}
		if (!hex.empty()) {
			break;
		}
	}
	std::printf("ReadFaultDiag publish %s: %" PRIu64 " page bytes in%s | first bytes %s\n",
	            eager ? "eager" : "side", bytes, text.c_str(), hex.c_str());
}

bool BufferCache::CompleteSideReadback(SideReadback& readback) {
	FlightRec::Rec(FlightRec::EvSideDone, readback.begin, readback.end, readback.eager ? 1 : 0);
	HangWatchdog::Scope wait("readback-publication", readback.begin, readback.value, readback.end,
	                         0, readback.eager);
	HangWatchdog::DebugDelay("readback", readback.begin);
	if (auto* diag = ReadFaultDiag::t_fault;
	    diag != nullptr && !diag->waited && !readback.done.load(std::memory_order_acquire)) {
		// KYTY_READ_FAULT_DIAG: split this reader's wait by polling each condition in turn.
		diag->waited    = true;
		auto&      master = m_scheduler.GetMasterSemaphore();
		const auto tick   = readback.eager ? readback.value : readback.producer;
		diag->wait_tick   = tick;
		diag->byte_known  = true;
		for (const auto& range: readback.ranges) {
			diag->byte_dirty |= diag->fault_vaddr >= range.address &&
			                    diag->fault_vaddr < range.address + range.size;
		}
		auto mark = ReadFaultDiag::NowNs();
		const auto lap = [&](uint64_t& part) {
			const auto now = ReadFaultDiag::NowNs();
			part += now - mark;
			mark = now;
		};
		if (const auto& progress = master.GetSubmissionProgress(); progress != nullptr) {
			while (progress->dispatched_tick.load(std::memory_order_acquire) < tick &&
			       !readback.done.load(std::memory_order_acquire)) {
				std::this_thread::yield();
			}
		}
		lap(diag->dispatch_ns);
		while (!readback.done.load(std::memory_order_acquire)) {
			master.Refresh();
			if (master.KnownGpuTick() >= tick) {
				break;
			}
			std::this_thread::yield();
		}
		lap(diag->gpu_ns);
		if (!readback.eager && !readback.done.load(std::memory_order_acquire)) {
			m_side->Wait(readback.value);
		}
		lap(diag->copy_ns);
		diag->phases_end_ns = mark;
	}
	std::scoped_lock lock(readback.mutex);
	if (readback.done.load(std::memory_order_acquire)) {
		return false;
	}
	auto&    side         = *m_side;
	Buffer*  staging      = nullptr;
	uint64_t staging_base = 0;
	if (readback.eager) {
		// The copy ends the recording of master tick `value`, behind all of its commands. A GPU
		// thread (CP) reader waiting here waits for that recording, never the current one.
		Profiler::ScopedGpuWaitReason wait_reason(Profiler::FrameWait::GpuWaitSideCopy);
		m_scheduler.GetMasterSemaphore().Wait(readback.value);
		staging      = side.eager_staging.get();
		staging_base = uint64_t {readback.slot} * TRACKER_PAGE_SIZE;
	} else {
		side.Wait(readback.value);
		staging      = side.staging.get();
		staging_base = uint64_t {readback.slot} * side.window;
	}
	staging->Invalidate(staging_base, readback.end - readback.begin);
	const auto* staged = staging->Mapped().data() + staging_base;
	for (const auto& range: readback.ranges) {
		Libs::LibKernel::Memory::WriteBacking(range.address,
		                                      staged + (range.address - readback.begin), range.size);
	}
	if (ReadFaultDiag::Covers(readback.begin, readback.end - readback.begin)) {
		DiagPrintPublication(readback.ranges, readback.begin, readback.eager, staged);
	}
	if (g_read_fault_published.On()) {
		NoteReadFaultPublication(readback.ranges, m_memory_tracker.Frame());
	}
	EndBackingPublication(readback.publication);
	// Only pages whose pending mark survived lose GPU ownership: a newer recorded writer (or any
	// other GPU transition) since the issue keeps its page protected for a new readback.
	const auto unmark =
	    m_memory_tracker.UnmarkReadbackPending(readback.begin, readback.end - readback.begin);
	if (readback.eager) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::ReadbackEagerPagesUnmarked,
		                          unmark.unmarked_pages);
		Profiler::CountFrameEvent(Profiler::FrameEvent::ReadbackEagerPagesRetained,
		                          unmark.retained_pages);
		side.eager_busy[readback.slot].store(false, std::memory_order_release);
	} else {
		Profiler::CountFrameEvent(Profiler::FrameEvent::ReadbackSidePagesUnmarked,
		                          unmark.unmarked_pages);
		Profiler::CountFrameEvent(Profiler::FrameEvent::ReadbackSidePagesRetained,
		                          unmark.retained_pages);
		side.slots[readback.slot].busy.store(false, std::memory_order_release);
	}
	readback.done.store(true, std::memory_order_release);
	{
		std::lock_guard pending_lock(side.pending_mutex);
		const auto      found =
		    std::find_if(side.pending.begin(), side.pending.end(),
		                 [&readback](const auto& entry) { return entry.get() == &readback; });
		EXIT_IF(found == side.pending.end());
		side.pending.erase(found);
		side.pending_count.store(side.pending.size(), std::memory_order_release);
	}
	if (readback.eager && HangTrace::Enabled()) {
		uint64_t bytes = 0;
		for (const auto& range: readback.ranges) {
			bytes += range.size;
		}
		const auto previous = HangTrace::GetReadbackKind();
		HangTrace::SetReadbackKind(HangTrace::ReadbackKind::EagerPublish);
		HangTrace::RecordReadback(readback.begin, bytes, readback.begin,
		                          readback.end - readback.begin, unmark.unmarked_pages != 0,
		                          HangTrace::NowNs() - readback.issue_ns);
		HangTrace::SetReadbackKind(previous);
	}
	return true;
}

bool BufferCache::OverlapsPendingSideReadback(uint64_t begin, uint64_t end) const {
	if (m_side == nullptr || m_side->pending_count.load(std::memory_order_acquire) == 0) {
		return false;
	}
	std::lock_guard lock(m_side->pending_mutex);
	return std::any_of(m_side->pending.begin(), m_side->pending.end(),
	                   [begin, end](const auto& entry) {
		                   return entry->begin < end && begin < entry->end;
	                   });
}

uint32_t BufferCache::CompleteSideReadbacks(uint64_t vaddr, uint64_t size) {
	if (m_side == nullptr || m_side->pending_count.load(std::memory_order_acquire) == 0 ||
	    !GuestRange {vaddr, size}.Valid()) {
		return 0;
	}
	std::vector<std::shared_ptr<SideReadback>> overlapping;
	{
		std::lock_guard lock(m_side->pending_mutex);
		for (const auto& entry: m_side->pending) {
			if (entry->begin < vaddr + size && vaddr < entry->end) {
				overlapping.push_back(entry);
			}
		}
	}
	// Pending entries never overlap each other (issue skips overlapping windows and pages), and
	// are in issue order, so completing them in this order keeps publications in submission
	// order.
	uint32_t eager = 0;
	for (const auto& entry: overlapping) {
		CompleteSideReadback(*entry);
		eager += entry->eager ? 1u : 0u;
	}
	return eager;
}

void BufferCache::CompleteAllSideReadbacks() {
	if (m_side == nullptr || m_side->pending_count.load(std::memory_order_acquire) == 0) {
		return;
	}
	std::vector<std::shared_ptr<SideReadback>> all;
	{
		std::lock_guard lock(m_side->pending_mutex);
		all = m_side->pending;
	}
	for (const auto& entry: all) {
		CompleteSideReadback(*entry);
	}
}

void BufferCache::NoteBufferContentWrite(uint64_t vaddr, uint64_t size) {
	// KYTY_ALIAS_REBUILD_PARTIAL: images whose native bytes equalled the old canonical ones.
	m_texture_cache.NoteCanonicalWrite(vaddr, size);
	if (m_side == nullptr) {
		return;
	}
	m_write_ticks.Assign(vaddr, size, m_scheduler.CurrentTick());
	ReadFaultDiag::NoteGpuWrite(vaddr, size, m_scheduler.CurrentTick(),
	                            reinterpret_cast<uint64_t>(__builtin_return_address(0)));
	if (m_write_ticks.Size() >= m_write_tick_prune_size) {
		// Completed writers need no entry: their ranges read back as the prune floor.
		const auto completed = m_scheduler.GetMasterSemaphore().KnownGpuTick();
		m_write_ticks.Prune(completed);
		m_write_tick_floor      = std::max(m_write_tick_floor, completed);
		m_write_tick_prune_size = std::max<size_t>(1024, m_write_ticks.Size() * 2);
	}
	if (m_eager_enabled && !m_eager.Empty() && m_eager.NoteWrite(vaddr, size) &&
	    m_eager_flush_budget != 0) {
		// A page the command processor reads back: submit this recording once the writer is
		// recorded, so the read after it finds a submitted (ideally finished) producer.
		m_eager_flush = true;
	}
}

bool BufferCache::ShaderWriteRetickEnabled() {
	return g_shader_write_retick.On();
}

void BufferCache::RetagShaderWrite(uint64_t vaddr, uint64_t size, uint64_t preparation_tick) {
	if (vaddr != 0 && size != 0 && preparation_tick != m_scheduler.CurrentTick()) {
		{
			const TilerTrace::Reason canon_reason("retag");
			NoteBufferContentWrite(vaddr, size);
		}
	}
}

void BufferCache::NoteEagerRead(uint64_t vaddr, uint64_t size, bool gpu_thread_reader) {
	if (!m_eager_enabled || size == 0 || size > EagerReadMaxBytes ||
	    !GuestRange {vaddr, size}.Valid()) {
		return;
	}
	const auto frame = m_memory_tracker.Frame();
	const auto first = Common::AlignDown(vaddr, TRACKER_PAGE_SIZE);
	const auto last  = Common::AlignUp(vaddr + size, TRACKER_PAGE_SIZE);
	for (auto page = first; page < last; page += TRACKER_PAGE_SIZE) {
		if (!m_eager.IsHot(page)) {
			Profiler::CountFrameEvent(Profiler::FrameEvent::ReadbackEagerHotPages);
		}
		m_eager.NoteRead(page, frame, gpu_thread_reader);
	}
}

void BufferCache::IssueEagerReadbacks() {
	// Called by the command processor right before it submits the current recording, between
	// packets, never from inside another cache operation: a Submit can also happen inside one
	// (e.g. a staging-ring wrap during a written upload, with tracker locks held), where issuing
	// would re-enter the tracker. Submissions without this call simply carry no eager copies.
	if (!m_eager_enabled || !GuestGpu::IsGpuThread() || CommandScheduler::InDeferredOperation() ||
	    !m_scheduler.Active() || m_scheduler.Current().IsInvalid()) {
		return;
	}
	// This submission is the early flush a recorded writer may have asked for.
	m_eager_flush = false;
	const auto frame = m_memory_tracker.Frame();
	// nps5 zerocopy4 (KYTY_ZEROCOPY_READ_OPEN): the per-frame check of the read-open pages.
	if (auto& open = ReadOpen(); open.sweep_frame.load(std::memory_order_relaxed) != frame) {
		open.sweep_frame.store(frame, std::memory_order_relaxed);
		const auto now  = ReadFaultDiag::NowNs();
		size_t     live = 0;
		{
			// nps5 faultclass: KYTY_FAULT_CLASS keeps the check running like the read-open switch;
			// a page its table made Exact (guest stores waited for unpublished GPU bytes) is
			// closed.
			const bool            classes = g_fault_class.On();
			const bool            on      = g_read_open.On() || classes;
			const uint64_t        span    = ReadRelaxSpan();
			std::vector<uint64_t> add;
			std::lock_guard       lock(open.mutex);
			for (auto it = open.pages.begin(); it != open.pages.end();) {
				const auto page  = it->first;
				const bool image = m_texture_cache.IsRegionGpuModified(page, TRACKER_PAGE_SIZE);
				const bool stale = !ReadFaultPublishedRecently(page, frame);
				const bool old   = frame - it->second > ReadOpenMaxFrames();
				const bool exact =
				    classes && FaultClasses().table.Get(page, frame) == FaultClass::Class::Exact;
				if (!on || image || stale || old || exact) {
					(void)m_memory_tracker.CloseRead(page, TRACKER_PAGE_SIZE);
					(!on     ? open.closed_off
					 : image ? open.closed_image
					 : stale ? open.closed_stale
					         : open.closed_expired)
					    .fetch_add(1, std::memory_order_relaxed);
					// reason 10 = faultclass: demoted to exact (counted with "expired" in the
					// read-open line)
					FlightRec::ZcSkip(page, TRACKER_PAGE_SIZE,
					                  !on     ? 9
					                  : image ? 8
					                  : stale ? 6
					                  : exact ? 10
					                          : 7);
					// An expired page keeps its class: its next served load re-opens it.
					if (classes && !exact && !old) {
						FaultClasses().table.NoteClosed(page);
					}
					it = open.pages.erase(it);
					continue;
				}
				// No fault tells the eager set that the page is still read: keep it hot here.
				NoteEagerRead(page, TRACKER_PAGE_SIZE, false);
				const auto begin = Common::AlignDown(page, span);
				for (auto other = begin; other < begin + span; other += TRACKER_PAGE_SIZE) {
					if (other == page || open.pages.contains(other) ||
					    !IsRegionGpuModifiedRelaxed(other, TRACKER_PAGE_SIZE) ||
					    m_texture_cache.IsRegionGpuModified(other, TRACKER_PAGE_SIZE) ||
					    (classes &&
					     FaultClasses().table.Get(other, frame) == FaultClass::Class::Exact)) {
						continue;
					}
					if (ReadFaultPublishedRecently(other, frame)) {
						add.push_back(other);
					} else if (!m_eager.IsHot(other)) {
						NoteEagerRead(other, TRACKER_PAGE_SIZE, false);
						open.promoted.fetch_add(1, std::memory_order_relaxed);
					} else {
						NoteEagerRead(other, TRACKER_PAGE_SIZE, false);
					}
				}
				++it;
			}
			for (const auto page: add) {
				if (open.pages.emplace(page, static_cast<uint32_t>(frame)).second) {
					(void)m_memory_tracker.OpenRead(page, TRACKER_PAGE_SIZE);
					open.neighbours.fetch_add(1, std::memory_order_relaxed);
					FlightRec::ZcSkip(page, TRACKER_PAGE_SIZE, 5);
					if (classes) {
						FaultClasses().table.SetLate(page); // its store conflicts count too
					}
				}
			}
			// nps5 faultclass: seeds (pages open in an earlier run). GPU-owned buffer pages among
			// them are kept read-hot, and opened once published recently, as a neighbour is.
			if (classes && m_eager_enabled) {
				auto&                        cls = FaultClasses();
				static std::vector<uint64_t> seeds; // GPU thread
				cls.table.PendingSeeds(seeds, 256);
				for (const auto page: seeds) {
					if (open.pages.contains(page)) {
						cls.table.NoteSeedOpened(page);
						continue;
					}
					if (!IsRegionGpuModifiedRelaxed(page, TRACKER_PAGE_SIZE) ||
					    m_texture_cache.IsRegionGpuModified(page, TRACKER_PAGE_SIZE)) {
						continue;
					}
					if (!ReadFaultPublishedRecently(page, frame)) {
						NoteEagerRead(page, TRACKER_PAGE_SIZE, false);
						cls.seed_hot.fetch_add(1, std::memory_order_relaxed);
						continue;
					}
					if (open.pages.emplace(page, static_cast<uint32_t>(frame)).second) {
						(void)m_memory_tracker.OpenRead(page, TRACKER_PAGE_SIZE);
						open.opened.fetch_add(1, std::memory_order_relaxed);
						FlightRec::ZcSkip(page, TRACKER_PAGE_SIZE, 5);
						cls.table.NoteSeedOpened(page);
					}
				}
			}
			live = open.pages.size();
		}
		if (g_fault_class.On()) {
			auto& cls = FaultClasses();
			if (cls.report_ns == 0) {
				cls.report_ns = now;
			} else if (now - cls.report_ns >= 10'000'000'000ull) {
				cls.report_ns       = now;
				const auto counters = cls.table.TakeCounters();
				const auto late     = cls.table.LatePages();
				std::printf("Kyty fault class 10s: %zu pages late (%zu open, %zu tracked); "
				            "promoted %" PRIu64 ", demoted to exact %" PRIu64 " (%" PRIu64
				            " store conflicts, %.1f ms waited); seeds: "
				            "opened %" PRIu64 ", kept read-hot %" PRIu64 " times\n",
				            late.size(), live, cls.table.Size(), counters.promoted,
				            counters.demoted, counters.conflicts,
				            static_cast<double>(cls.store_conflict_ns.exchange(0)) / 1e6,
				            counters.seed_hits, cls.seed_hot.exchange(0));
				if (!cls.profile_out.empty() && late.size() != cls.saved && !late.empty()) {
					cls.saved = late.size();
					(void)FaultClass::SaveProfile(cls.profile_out, late, TRACKER_PAGE_SIZE);
				}
			}
		}
		if (open.report_ns == 0) {
			open.report_ns = now;
		} else if (now - open.report_ns >= 10'000'000'000ull) {
			open.report_ns    = now;
			const auto take   = [](std::atomic<uint64_t>& counter) { return counter.exchange(0); };
			const auto opened = take(open.opened);
			const auto closed = take(open.closed_stale);
			if (g_read_open.On() || live != 0 || opened != 0 || closed != 0) {
				std::printf("Kyty read-open 10s: %zu pages open; opened %" PRIu64 " (+%" PRIu64
				            " neighbours, %" PRIu64
				            " neighbours made read-hot); closed: not published "
				            "recently %" PRIu64 ", image %" PRIu64 ", expired %" PRIu64
				            ", switched off %" PRIu64 ", check missing %" PRIu64 "\n",
				            live, opened, take(open.neighbours), take(open.promoted), closed,
				            take(open.closed_image), take(open.closed_expired),
				            take(open.closed_off), take(open.closed_unchecked));
			}
			if (UploadClassifyEvery() != 0) {
				auto&        cls   = UploadClass();
				const double scale = static_cast<double>(UploadClassifyEvery());
				const auto   mib   = [scale](uint64_t pages) {
					return static_cast<double>(pages) * scale * TRACKER_PAGE_SIZE /
					       (1024.0 * 1024.0);
				};
				std::printf(
				    "Kyty upload classes 10s (every %u. page, scaled): first %.1f MiB, unchanged "
				    "%.1f "
				    "MiB, changed after its last upload completed %.1f MiB, changed in flight %.1f "
				    "MiB; hot pages (rewritten every frame, compared by shadow) %.1f MiB\n",
				    UploadClassifyEvery(), mib(cls.count[0]), mib(cls.count[1]), mib(cls.count[2]),
				    mib(cls.count[3]),
				    static_cast<double>(cls.hot_pages) * TRACKER_PAGE_SIZE / (1024.0 * 1024.0));
				cls.count[0] = cls.count[1] = cls.count[2] = cls.count[3] = 0;
				cls.hot_pages                                             = 0;
				if (cls.pages.size() > 400000) {
					cls.pages.clear();
				}
			}
		}
	}
	if (m_eager.Empty()) {
		return;
	}
	m_eager.Sweep(frame);
	if (m_eager.Candidates() == 0) {
		return;
	}
	const auto tick = m_scheduler.CurrentTick();
	m_eager.IssueCandidates(frame, [this, tick](uint64_t page) {
		const auto result = TryIssueEagerReadback(page, tick);
		if (result == EagerReadbackPages::IssueResult::Retry) {
			Profiler::CountFrameEvent(Profiler::FrameEvent::ReadbackEagerRetries);
		}
		return result;
	});
}

EagerReadbackPages::IssueResult BufferCache::TryIssueEagerReadback(uint64_t page, uint64_t tick) {
	FlightRec::Rec(FlightRec::EvEagerReadback, page, tick);
	using Result = EagerReadbackPages::IssueResult;
	auto&      side = *m_side;
	const auto end  = page + TRACKER_PAGE_SIZE;
	const auto* owner = m_page_table.Find(page >> PageTable::kPageBits);
	if (owner == nullptr || !*owner || IsBufferInvalid(*owner) ||
	    !m_slot_buffers[*owner].IsInBounds(page, TRACKER_PAGE_SIZE)) {
		return Result::Drop;
	}
	auto& buffer = m_slot_buffers[*owner];
	// One publication per byte at a time, published in issue order (as for side readbacks).
	if (OverlapsPendingSideReadback(page, end) ||
	    HasPendingBackingPublication(page, TRACKER_PAGE_SIZE)) {
		return Result::Retry;
	}
	std::vector<GuestRange> dirty;
	uint64_t                newest = m_write_tick_floor;
	uint64_t                bytes  = 0;
	m_gpu_modified_ranges.ForEachInRange(page, TRACKER_PAGE_SIZE, [&](uint64_t start, uint64_t stop) {
		dirty.push_back({start, stop - start});
		newest = std::max(newest, m_write_ticks.MaxTick(start, stop - start));
		bytes += stop - start;
	});
	if (dirty.empty()) {
		return Result::Drop;
	}
	// Only bytes whose writers are already submitted: a writer registered by this recording may
	// record its shader command after this submission (a submit in the middle of a draw's
	// preparation), so its bytes wait for the next submission. That is the side-readback rule;
	// here the copy additionally follows the whole of this recording in queue order.
	if (newest >= tick) {
		return Result::Retry;
	}
	// Dirty bytes keep their tracker page GPU-owned (MemoryTracker validates the pairing).
	if (!m_memory_tracker.IsRegionGpuModified(page, TRACKER_PAGE_SIZE)) {
		return Result::Drop;
	}
	uint32_t slot = SideReadbackState::EagerSlotCount;
	for (uint32_t index = 0; index < SideReadbackState::EagerSlotCount; ++index) {
		if (!side.eager_busy[index].load(std::memory_order_acquire)) {
			slot = index;
			break;
		}
	}
	if (slot == SideReadbackState::EagerSlotCount) {
		return Result::Retry;
	}
	const auto staging_base = uint64_t {slot} * TRACKER_PAGE_SIZE;

	auto readback      = std::make_shared<SideReadback>();
	readback->begin    = page;
	readback->end      = end;
	readback->value    = tick;
	readback->slot     = slot;
	readback->eager    = true;
	readback->issue_ns = HangTrace::Enabled() ? HangTrace::NowNs() : 0;
	readback->ranges   = dirty;
	std::vector<vk::BufferCopy> copies;
	copies.reserve(dirty.size());
	for (const auto& range: dirty) {
		copies.emplace_back(buffer.Offset(range.address), staging_base + (range.address - page),
		                    range.size);
	}

	// Recorded last in this recording: the barrier's first scope covers every earlier command on
	// the queue, including all writers of these bytes (checked above to be submitted already).
	auto& command = m_scheduler.Current();
	command.EndRendering();
	// Encoded with the recorder (no drain): the completion runner reads the staging only after
	// this tick completed, which the recorder's submission of it precedes.
	const auto              native = command.HelperSink();
	vk::BufferMemoryBarrier before {};
	before.srcAccessMask       = vk::AccessFlagBits::eMemoryWrite;
	before.dstAccessMask       = vk::AccessFlagBits::eTransferRead;
	before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.buffer              = buffer.Handle();
	before.offset              = buffer.Offset(page);
	before.size                = TRACKER_PAGE_SIZE;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                       vk::PipelineStageFlagBits::eTransfer, {}, 0, nullptr, 1, &before, 0,
	                       nullptr);
	const auto regions =
	    MergeLinearCopies(copies, static_cast<uint64_t>(g_readback_merge_gap.Get()));
	RecordReadbackRegions(copies, regions);
	native.copyBuffer(buffer.Handle(), side.eager_staging->Handle(),
	                  static_cast<uint32_t>(regions.size()), regions.data());
	vk::BufferMemoryBarrier after = before;
	after.srcAccessMask           = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask           = vk::AccessFlagBits::eHostRead;
	after.buffer                  = side.eager_staging->Handle();
	after.offset                  = staging_base;
	after.size                    = TRACKER_PAGE_SIZE;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost,
	                       {}, 0, nullptr, 1, &after, 0, nullptr);

	// Ownership of the exact dirty bytes moves to the publication registered below, exactly as in
	// TryIssueSideReadback; the page stays GPU-owned (protected) until completion.
	for (const auto& range: dirty) {
		CleanVerdict::Invalidate(range.address, range.size, Coherence::Source::BufferDirtySubtract);
	}
	for (const auto& range: dirty) {
		m_gpu_range_removals.fetch_add(1, std::memory_order_acq_rel);
		m_gpu_modified_ranges.Subtract(range.address, range.size);
	}
	m_memory_tracker.MarkReadbackPending(page, TRACKER_PAGE_SIZE);
	readback->publication = BeginBackingPublication(dirty, tick);
	side.eager_busy[slot].store(true, std::memory_order_relaxed);
	{
		std::lock_guard lock(side.pending_mutex);
		side.pending.push_back(readback);
		side.pending_count.store(side.pending.size(), std::memory_order_release);
	}
	// Diagnostic only: allow a guest reader to acquire this publication before the CP flushes
	// its producer tick, opening the issue-to-submit race window without changing ordering.
	HangWatchdog::DebugDelay("eager-issue", page);
	// Queued for this recording's tick: the completion runner publishes it once the recording
	// has finished, unless a reader (guest fault, CP read) completed it first.
	m_scheduler.DeferPriorityOperation(
	    [this, readback = std::move(readback)] { CompleteSideReadback(*readback); });
	Profiler::CountFrameEvent(Profiler::FrameEvent::ReadbackEagerCopies);
	Profiler::CountFrameEvent(Profiler::FrameEvent::ReadbackEagerCopyBytes, bytes);
	return Result::Issued;
}

bool BufferCache::TakeEagerFlushRequest(bool in_rendering) {
	if (!m_eager_flush || in_rendering) {
		return false;
	}
	m_eager_flush    = false;
	const auto frame = m_memory_tracker.Frame();
	if (frame != m_eager_flush_frame) {
		m_eager_flush_frame = frame;
		m_eager_flushes     = 0;
	}
	if (m_eager_flushes >= m_eager_flush_budget) {
		return false;
	}
	m_eager_flushes++;
	Profiler::CountFrameEvent(Profiler::FrameEvent::ReadbackEagerFlushes);
	return true;
}

BufferId BufferCache::FindBuffer(uint64_t vaddr, uint64_t size) {
	if (vaddr == 0) {
		return NULL_BUFFER_ID;
	}
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid buffer discovery request\n");
	}
	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	if (owner != nullptr && *owner) {
		auto& buffer = m_slot_buffers[*owner];
		if (buffer.IsInBounds(vaddr, size)) {
			return *owner;
		}
	}
	return CreateBuffer(vaddr, size);
}

BufferCache::OverlapResult BufferCache::ResolveOverlaps(uint64_t vaddr, uint64_t size) {
	static constexpr int      StreamLeapThreshold = 16;
	static constexpr uint64_t StreamLeapSize      = CACHING_PAGESIZE * 128;

	auto       begin      = vaddr;
	auto       end        = vaddr + size;
	const auto find_first = [&](uint64_t address) {
		auto first = m_buffers.lower_bound(address);
		if (first != m_buffers.begin()) {
			const auto  previous = std::prev(first);
			const auto& buffer   = m_slot_buffers[previous->second];
			if (buffer.CpuAddress() + buffer.Size() > address) {
				first = previous;
			}
		}
		return first;
	};
	auto first           = find_first(begin);
	auto last            = first;
	int  stream_score    = 0;
	bool has_stream_leap = false;
	for (; last != m_buffers.end() && last->first < end; ++last) {
		const auto& buffer        = m_slot_buffers[last->second];
		const auto  buffer_begin  = buffer.CpuAddress();
		const auto  buffer_end    = buffer_begin + buffer.Size();
		const bool  expands_left  = buffer_begin < begin;
		const bool  expands_right = buffer_end > end;
		begin                     = std::min(begin, buffer_begin);
		end                       = std::max(end, buffer_end);
		if (!has_stream_leap && (stream_score += buffer.StreamScore()) > StreamLeapThreshold) {
			has_stream_leap = true;
			// Reserve space in the incoming stream's direction of growth.
			// The old buffer extending left of the request predicts growth to the right, and vice versa.
			if (expands_left) {
				end += std::min(StreamLeapSize, PageTable::kAddressSpaceSize - end);
			}
			if (expands_right) {
				const auto minimum = CACHING_PAGESIZE * 2;
				if (begin > minimum) {
					begin -= std::min(StreamLeapSize, begin - minimum);
				}
				first = find_first(begin);
				begin = std::min(begin, first->first);
			}
		}
	}
	return {first, last, begin, end, has_stream_leap};
}

void BufferCache::JoinOverlap(BufferId new_id, BufferId overlap_id, bool accumulate_stream_score) {
	auto& new_buffer = m_slot_buffers[new_id];
	auto& overlap    = m_slot_buffers[overlap_id];
	if (accumulate_stream_score) {
		new_buffer.IncreaseStreamScore(overlap.StreamScore() + 1);
	}
	new_buffer.CopyFrom(m_scheduler.Current(), overlap, 0,
	                    overlap.CpuAddress() - new_buffer.CpuAddress(), overlap.Size());
	DeleteBuffer(overlap_id);
}

BufferId BufferCache::CreateBuffer(uint64_t vaddr, uint64_t size) {
	EXIT_IF(m_scheduler.Current().IsInvalid());
	const BufferLookupStats::PartScope stats_scope(BufferLookupStats::Part::CreateBuffer);
	const auto end = Common::AlignUp(vaddr + size, CACHING_PAGESIZE);
	vaddr = Common::AlignDown(vaddr, CACHING_PAGESIZE);
	size               = end - vaddr;
	const auto overlap = ResolveOverlaps(vaddr, size);

	const auto id = m_slot_buffers.insert(
	    m_graphics, m_scheduler, MemoryUsage::DeviceLocal, overlap.begin,
	    AllFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress, overlap.end - overlap.begin);
	const auto& buffer = m_slot_buffers[id];
	SetVulkanObjectNameF(m_graphics.device, buffer.Handle(),
	                     "Kyty.GameBuffer[guest=0x{:016x} size=0x{:x}]", overlap.begin,
	                     overlap.end - overlap.begin);
	const bool joined = overlap.first != overlap.last;
	for (auto it = overlap.first; it != overlap.last;) {
		const auto old_id = (it++)->second;
		JoinOverlap(id, old_id, !overlap.has_stream_leap);
	}
	if (joined) {
		// The joined contents (including GPU-dirty bytes) are copied by this recording.
		{
			const TilerTrace::Reason canon_reason("create");
			NoteBufferContentWrite(overlap.begin, overlap.end - overlap.begin);
		}
	}
	FlightRec::Rec(FlightRec::EvBufCreate, vaddr, size,
	               (static_cast<uint64_t>(id.index) << 32u) | id.generation);
	Register(id);
	return id;
}

bool BufferCache::SynchronizeBuffer(Buffer& buffer, uint64_t vaddr, uint64_t size, bool is_written,
                                    bool is_texel_buffer, BdaSyncStats* stats,
                                    const char* upload_reason) {
	KYTY_GPU_OP_SITE("buffercache.upload");
	// KYTY_TRACKER_RELAXED_QUERIES: a read-only synchronization of a range without a CPU-dirty
	// page (hot pages are CPU-dirty too) collects and uploads nothing. Texel reads also download
	// GPU-written images; the BDA hot-pass verification scans in full.
	if (!is_written && !is_texel_buffer && (stats == nullptr || stats->verify_fault_epoch == 0) &&
	    RelaxedNothingToUpload(vaddr, size)) {
		if (FlightRecDraw::Keyed()) [[unlikely]] {
			FlightRec::Rec(FlightRec::EvSyncSkip, vaddr, size, FlightRec::SkipRelaxedClean);
		}
		return false;
	}
	// KYTY_WRITTEN_SYNC_SKIP (bufferCache.h): a range the GPU already owns entirely. (Only texel
	// READS do more than the tracker work below.)
	bool written_skip_verify = false;
	if (is_written && stats == nullptr && m_written_sync_skip && GuestGpu::IsGpuThread() &&
	    m_memory_tracker.IsRangeGpuOwned(vaddr, size)) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::WrittenSyncSkips);
		m_written_sync_totals.skips++;
		if (m_written_sync_skip_verify == 0) {
			if (FlightRecDraw::Keyed()) [[unlikely]] {
				FlightRec::Rec(FlightRec::EvSyncSkip, vaddr, size, FlightRec::SkipWrittenOwned);
			}
			return false;
		}
		written_skip_verify = true;
	}
	// KYTY_BUFFER_RANGE_MEMO: a read-only synchronization of a range that is still Clean does
	// nothing (bufferCache.h). Texel reads also download GPU-written images (not tracker state);
	// the BDA hot-pass verification scans in full.
	uint64_t memo_signature = 0;
	uint64_t memo_dirtied   = 0; // verify mode: dirtying serials before the lookup
	bool     memo_verify    = false;
	const bool memo_applies = !is_written && !is_texel_buffer && m_range_memo != nullptr &&
	                          (stats == nullptr || stats->verify_fault_epoch == 0);
	if (memo_applies) {
		if (m_range_memo_verify != 0) {
			memo_dirtied = m_memory_tracker.RangeDirtiedSignature(vaddr, size);
		}
		memo_signature = m_memory_tracker.RangeSignature(vaddr, size);
		const auto& memo = RangeMemoSlot(vaddr, size);
		if (memo_signature != 0 && memo.signature == memo_signature && memo.vaddr == vaddr &&
		    memo.size == size && memo.fact == RangeFact::Clean) {
			Profiler::CountFrameEvent(Profiler::FrameEvent::BufferRangeMemoCleanHits);
			m_range_memo_totals.clean_hits++;
			if (m_range_memo_verify == 0) {
				if (FlightRecDraw::Keyed()) [[unlikely]] {
					FlightRec::Rec(FlightRec::EvSyncSkip, vaddr, size, FlightRec::SkipRangeMemo);
				}
				return false;
			}
			memo_verify = true;
			RunRangeMemoVerifyHook(vaddr, size);
		}
	}
	std::vector<vk::BufferCopy> copies;
	uint64_t                    total_size = 0;
	vk::Buffer                  source;
	// KYTY_FALSE_SHARING_WRITES: bytes an early release left to their publication are the GPU's
	// in the buffer; CPU-dirty page runs over them upload only the other bytes until it lands.
	if (!m_early_released.empty()) {
		PruneEarlyReleased();
	}
	const bool exclude_unpublished = !m_early_released.empty();
	const auto append_copy = [&](std::vector<vk::BufferCopy>& list, uint64_t& list_size,
	                             uint64_t address, uint64_t bytes) noexcept {
		if (!exclude_unpublished) {
			list.emplace_back(list_size, buffer.Offset(address), bytes);
			list_size += bytes;
			return;
		}
		uint64_t emitted = 0;
		ForEachPublishedPart(address, bytes, [&](uint64_t part, uint64_t part_bytes) {
			list.emplace_back(list_size, buffer.Offset(part), part_bytes);
			list_size += part_bytes;
			emitted += part_bytes;
		});
		if (emitted != bytes) {
			m_false_sharing_totals.upload_splits++;
			Profiler::CountFrameEvent(Profiler::FrameEvent::FalseSharingUploadSplits);
		}
	};
	uint8_t* reserved = nullptr;
	uint64_t reserved_offset = 0;
	uint64_t reserved_size = 0;
	if (Common::RendererBatchEnabled() && is_written && size <= MiB &&
	    m_memory_tracker.IsRegionCpuModified(vaddr, size)) {
		// Reserve before entering writable tracker locks. The dirty set is collected
		// again under those locks, so a concurrent CPU write cannot be missed.
		const auto begin = Common::AlignDown(vaddr, CACHING_PAGESIZE);
		reserved_size = Common::AlignUp(vaddr + size, CACHING_PAGESIZE) - begin;
		copies.reserve(static_cast<size_t>(reserved_size / CACHING_PAGESIZE));
		std::tie(reserved, reserved_offset) = m_staging_buffer.Map(reserved_size, 4);
	}
	// Hot pages (MemoryTracker) stay CPU-dirty: they are reported separately and copied only when
	// they differ from the shadow of the last copy this buffer received (CollectHotPages).
	std::vector<GuestRange> hot_ranges;
	std::vector<uint64_t>   demote_hot;
	std::vector<uint64_t>   settle_hot;
	size_t                guest_copies = 0;
	uint64_t              host_base    = 0;
	// Written uploads with KYTY_UPLOAD_COPY_OUTSIDE_LOCK: pages re-dirtied by a racing guest write
	// while the main copy ran unlocked, copied again under the tracker locks.
	std::vector<vk::BufferCopy> late_copies;
	uint64_t                    late_size = 0;
	vk::Buffer                  late_source;
	bool                        reserved_committed = false;
	const auto collect = [&](uint64_t address, uint64_t bytes, bool hot) noexcept {
		if (hot) {
			hot_ranges.push_back({address, bytes});
			if (stats != nullptr && stats->hot_ranges != nullptr) {
				// A full BDA pass remembers where its hot pages are (KYTY_BDA_HOT_SYNC).
				stats->hot_ranges->push_back({stats->buffer_id, address, bytes});
			}
			return;
		}
		if (stats != nullptr && stats->verify_fault_epoch != 0 &&
		    m_memory_tracker.FaultMutationEpoch() == stats->verify_fault_epoch &&
		    m_bda_structure_epoch.load(std::memory_order_acquire) == stats->verify_structure_epoch) {
			// Called under the region lock: a transition that dirtied this page published its
			// epoch change before the lock was released, so it would be visible here.
			stats->verify_mismatch_pages += bytes / TRACKER_PAGE_SIZE;
		}
		append_copy(copies, total_size, address, bytes);
	};
	// KYTY_UPLOAD_DMA_HOST_COPY (uploadDma.h): a read upload large enough for the copy engine
	// leaves its guest bytes to the DMA worker. The pages are already clean and write-protected
	// when upload() runs (ForEachUploadRange), as for the copy made here.
	std::vector<UploadHostCopy> host_copies;
	// KYTY_DEFER_PROTECTS (pageManager.h): the collection of a read upload that may go to the DMA
	// worker defers the write protections of the pages it cleans. The worker applies them before
	// it copies the pages; every copy of their guest bytes made on this thread applies them first
	// (below when the upload stays on this thread, UploadCopies, StageUploadDma), and the scope's
	// end applies what is left.
	const int  defer_mode     = PageManager::DeferProtectsMode();
	const bool defer_protects = defer_mode != 0 && !is_written && UploadBatchEnabled() &&
	                            m_upload_dma != nullptr && UploadDmaHostCopyEnabled() &&
	                            !UploadDmaVerify() && m_staging_buffer.IsCoherent();
	const PageManager::DeferProtectsScope defer_scope(defer_protects);
	bool                                  force_dma = false;
	const auto upload = [&]() noexcept {
		if (defer_protects) {
			PageManager::DeferProtectsScope::Stop(); // the collection is over
		}
		// A normal upload replaces whatever a hot page shadow described.
		for (const auto& copy: copies) {
			EraseHotShadows(buffer.CpuAddress() + copy.dstOffset, copy.size);
		}
		if (const auto every = UploadClassifyEvery(); every != 0 && GuestGpu::IsGpuThread())
		    [[unlikely]] {
			// nps5 zerocopy4 (KYTY_ZC_UPLOAD_CLASSIFY): see UploadClassState.
			auto&      cls  = UploadClass();
			const auto tick = m_scheduler.CurrentTick();
			for (const auto& copy: copies) {
				const auto begin = buffer.CpuAddress() + copy.dstOffset;
				for (auto page = Common::AlignUp(begin, TRACKER_PAGE_SIZE);
				     page + TRACKER_PAGE_SIZE <= begin + copy.size; page += TRACKER_PAGE_SIZE) {
					if ((page / TRACKER_PAGE_SIZE) % every != 0) {
						continue;
					}
					uint8_t bytes[TRACKER_PAGE_SIZE];
					if (!LibKernel::Memory::TryReadBacking(page, bytes, TRACKER_PAGE_SIZE)) {
						continue;
					}
					const auto hash = HashPage(bytes);
					auto&      seen = cls.pages[page];
					const int  kind = seen.tick == 0                  ? 0
					                  : seen.hash == hash             ? 1
					                  : m_scheduler.IsFree(seen.tick) ? 2
					                                                  : 3;
					cls.count[kind]++;
					seen.hash = hash;
					seen.tick = tick;
				}
			}
			for (const auto& range: hot_ranges) {
				cls.hot_pages += range.size / TRACKER_PAGE_SIZE;
			}
		}
		guest_copies = copies.size();
		host_base    = total_size;
		if (!hot_ranges.empty()) {
			CollectHotPages(buffer, hot_ranges, copies, total_size, demote_hot, settle_hot);
		}
		if (reserved != nullptr && total_size <= reserved_size && guest_copies == copies.size()) {
			for (auto& copy: copies) {
				std::memcpy(reserved + copy.srcOffset,
				            LibKernel::Memory::GuestReadPtr(buffer.CpuAddress() + copy.dstOffset, copy.size),
				            copy.size);
				copy.srcOffset += reserved_offset;
			}
			if (!copies.empty()) source = m_staging_buffer.Handle();
		} else {
			reserved             = nullptr;
			// KYTY_DEFER_PROTECTS=all: an upload with deferred protections goes to the worker
			// whatever its size.
			force_dma = defer_protects && (defer_mode & 2) != 0 && PageManager::HasPendingProtects();
			const bool defer_host = !is_written && UploadBatchEnabled() && m_upload_dma != nullptr &&
			                        UploadDmaHostCopyEnabled() && !UploadDmaVerify() &&
			                        m_staging_buffer.IsCoherent() &&
			                        (total_size >= m_upload_dma->MinBytes() || force_dma);
			if (!defer_host) {
				PageManager::ApplyPendingProtects(); // this thread copies the pages next
			}
			source = UploadCopies(buffer, copies, total_size, guest_copies, m_hot_scratch.data(),
			                      host_base, defer_host ? &host_copies : nullptr);
		}
	};
	if (is_written && m_memory_tracker.GetFaultPolicy().copy_outside_lock) {
		m_memory_tracker.ForEachWrittenUploadRange(
		    vaddr, size,
		    [&](uint64_t address, uint64_t bytes) noexcept { collect(address, bytes, false); },
		    upload,
		    [&](uint64_t address, uint64_t bytes) noexcept {
			    append_copy(late_copies, late_size, address, bytes);
		    },
		    [&]() noexcept {
			    if (late_copies.empty()) {
				    return;
			    }
			    if (reserved != nullptr && source) {
				    // A staging Map() reuses the pending reservation until it is committed.
				    m_staging_buffer.Commit();
				    reserved_committed = true;
			    }
			    late_source = UploadCopies(buffer, late_copies, late_size);
		    });
	} else {
		m_memory_tracker.ForEachUploadRange(vaddr, size, is_written, collect, upload);
	}
	if (written_skip_verify) {
		// Nothing a CPU access could have changed since the check: it waits for this thread.
		Profiler::CountFrameEvent(Profiler::FrameEvent::WrittenSyncSkipVerifyChecks);
		m_written_sync_totals.verify_checks++;
		if (!copies.empty() || !hot_ranges.empty() || !late_copies.empty()) {
			Profiler::CountFrameEvent(Profiler::FrameEvent::WrittenSyncSkipVerifyMismatches);
			m_written_sync_totals.verify_mismatches++;
			static std::atomic<uint32_t> logged {0};
			if (logged.fetch_add(1, std::memory_order_relaxed) < 32) {
				std::fprintf(stderr,
				             "WrittenSyncSkipVerify: a GPU-owned range had pages to upload: "
				             "addr=0x%016" PRIx64 " size=0x%" PRIx64 "\n",
				             vaddr, size);
			}
			if (m_written_sync_skip_verify == 2) {
				EXIT("WrittenSyncSkipVerify: a GPU-owned range had pages to upload: addr=0x%016" PRIx64
				     " size=0x%" PRIx64 "\n",
				     vaddr, size);
			}
		}
	}
	if (memo_applies) {
		// Nothing collected: no page of the range was CPU-dirty (hot pages are CPU-dirty too).
		const bool collected = !copies.empty() || !hot_ranges.empty();
		if (memo_verify) {
			Profiler::CountFrameEvent(Profiler::FrameEvent::BufferRangeMemoVerifyChecks);
			m_range_memo_totals.verify_checks++;
			if (collected) {
				ClassifyRangeMemoDifference("a skipped synchronization found pages to upload",
				                            vaddr, size, memo_dirtied);
			}
		} else if (!collected && memo_signature != 0 &&
		           m_memory_tracker.RangeSignature(vaddr, size) == memo_signature) {
			RecordRangeFact(vaddr, size, memo_signature, RangeFact::Clean);
		}
	}
	for (const auto page: demote_hot) {
		m_memory_tracker.DemoteHotPages(page, TRACKER_PAGE_SIZE);
		EraseHotShadows(page, TRACKER_PAGE_SIZE);
	}
	for (const auto page: settle_hot) {
		SettleHotPages(page, TRACKER_PAGE_SIZE);
	}
	if (reserved != nullptr && source && !reserved_committed) {
		// Source copying and GPU ownership publication stayed consistent (under the tracker locks,
		// or unlocked with the late pass). Flush and ring bookkeeping need no tracker lock and
		// finish before native copy recording.
		m_staging_buffer.Commit();
		Profiler::CountFrameEvent(Profiler::FrameEvent::UploadReservationsOutsideLocks);
	}
	if (FlightRec::On()) [[unlikely]] {
		if (source || late_source) {
			FlightRec::Rec(FlightRec::EvUpload, buffer.CpuAddress(),
			               copies.empty() ? vaddr : buffer.CpuAddress() + copies.front().dstOffset,
			               total_size + late_size,
			               static_cast<uint64_t>(copies.size() & 0xffff) |
			                   (static_cast<uint64_t>(hot_ranges.size() & 0xff) << 16u) |
			                   (static_cast<uint64_t>(late_copies.size() & 0xff) << 24u) |
			                   (UploadBatchEnabled() && !late_source ? 1ull << 32u : 0) |
			                   (FlightRecDraw::Keyed() ? 1ull << 33u : 0) |
			                   (stats != nullptr ? 1ull << 34u : 0));
			if (total_size + late_size >= (256u << 10u)) {
				FlightRec::Rec(FlightRec::EvUploadBig, buffer.CpuAddress(), vaddr,
				               total_size + late_size,
				               static_cast<uint64_t>(copies.size() & 0xffff));
			}
		} else if (FlightRecDraw::Keyed()) {
			FlightRec::Rec(FlightRec::EvSyncSkip, vaddr, size,
			               FlightRec::SkipNothing |
			                   (static_cast<uint64_t>(hot_ranges.size()) << 8u));
		}
	}
	if (source || late_source) {
		if (stats != nullptr) {
			stats->upload_bytes += total_size + late_size;
			stats->upload_copies += copies.size() + late_copies.size();
		}
		Profiler::CountFrameEvent(Profiler::FrameEvent::BufferUploadBytes, total_size + late_size);
		if (HangTrace::Enabled()) {
			const char* reason = upload_reason;
			if (reason == nullptr) {
				reason = stats != nullptr    ? "bda-sync"
				         : is_written        ? "written-binding"
				         : is_texel_buffer   ? "texel-read"
				                             : "read-binding";
			}
			// Address is the first uploaded page run; span_bytes the requested range.
			HangTrace::RecordTransfer(HangTrace::TransferKind::BufferUpload, reason, "",
			                          copies.empty() ? vaddr
			                                         : buffer.CpuAddress() + copies.front().dstOffset,
			                          0, static_cast<uint32_t>(copies.size()), 0, total_size, size);
		}
		auto& command = m_scheduler.Current();
		if (UploadBatchEnabled() && !late_source) {
			// One barrier pair per flush for all queued uploads instead of one per upload. Inside
			// an UploadBatch scope the copy waits for the scope end (or an earlier flush point).
			source = StageUploadDma(source, copies, &host_copies, force_dma);
			command.RequestUploadCopy(source, buffer.Handle(), copies);
			buffer.MarkContentWritten();
			if (m_upload_batch_depth == 0) {
				command.FlushBarriers();
			}
			if (is_texel_buffer && !is_written) {
				return SynchronizeBufferFromImage(buffer, vaddr, size);
			}
			return false;
		}
		if (command.ActiveRenderingSerial() != 0) {
			MemoryStats::Count(MemoryStats::Counter::UploadRenderSplits);
		}
		const bool both = source && late_source;
		MemoryStats::Count(MemoryStats::Counter::UploadCopies, both ? 2 : 1);
		MemoryStats::Count(MemoryStats::Counter::UploadBarriers, both ? 3 : 2);
		command.EndRendering();
		const auto native = command.Handle();
		vk::BufferMemoryBarrier before {};
		before.srcAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite |
		                       vk::AccessFlagBits::eTransferRead |
		                       vk::AccessFlagBits::eTransferWrite;
		before.dstAccessMask       = vk::AccessFlagBits::eTransferWrite;
		before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.buffer              = buffer.Handle();
		before.offset              = 0;
		before.size                = buffer.Size();
		native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
		                       vk::PipelineStageFlagBits::eTransfer,
		                       vk::DependencyFlagBits::eByRegion, 0, nullptr, 1, &before, 0, nullptr);
		if (source) {
			native.copyBuffer(source, buffer.Handle(), static_cast<uint32_t>(copies.size()),
			                  copies.data());
		}
		if (late_source) {
			if (source) {
				// The late copy rewrites pages of the first one: order the two writes.
				auto between          = before;
				between.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
				native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
				                       vk::PipelineStageFlagBits::eTransfer,
				                       vk::DependencyFlagBits::eByRegion, 0, nullptr, 1, &between,
				                       0, nullptr);
			}
			native.copyBuffer(late_source, buffer.Handle(),
			                  static_cast<uint32_t>(late_copies.size()), late_copies.data());
		}
		buffer.MarkContentWritten();
		auto after          = before;
		after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
		after.dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
		native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
		                       vk::PipelineStageFlagBits::eAllCommands,
		                       vk::DependencyFlagBits::eByRegion, 0, nullptr, 1, &after, 0, nullptr);
	}
	if (is_texel_buffer && !is_written) {
		return SynchronizeBufferFromImage(buffer, vaddr, size);
	}
	return false;
}

vk::Buffer BufferCache::StageUploadDma(vk::Buffer source, std::span<vk::BufferCopy> copies,
                                       std::vector<UploadHostCopy>* host_copies, bool allow_small) {
	// Staging bytes UploadCopies left to the DMA worker, written here when the copy is not
	// staged after all (the graphics copies then read the staging ring directly).
	const auto write_host_copies = [host_copies] {
		PageManager::ApplyPendingProtects(); // KYTY_DEFER_PROTECTS: before the bytes are read
		if (host_copies != nullptr) {
			for (const auto& copy: *host_copies) {
				std::memcpy(copy.destination, copy.source, static_cast<size_t>(copy.size));
			}
			host_copies->clear();
		}
	};
	if (m_upload_dma == nullptr || source != m_staging_buffer.Handle() || copies.empty()) {
		write_host_copies();
		return source;
	}
	// The staged bytes of one upload are one packed range of the ring (SynchronizeBuffer and
	// UploadCopies place them back to back); the transfer copies that whole range.
	uint64_t begin = UINT64_MAX;
	uint64_t end   = 0;
	for (const auto& copy: copies) {
		begin = std::min(begin, copy.srcOffset);
		end   = std::max(end, copy.srcOffset + copy.size);
	}
	const auto size        = end - begin;
	// KYTY_DEFER_PROTECTS: the deferred protections go to the worker with the host copies.
	auto       protects    = PageManager::TakePendingProtects();
	const auto ring_offset = m_upload_dma->Stage(source, begin, size, host_copies, &protects,
	                                             allow_small && !protects.empty());
	if (!ring_offset.has_value()) {
		PageManager::ApplyProtects(protects, false);
		write_host_copies();
		return source;
	}
	for (auto& copy: copies) {
		copy.srcOffset = copy.srcOffset - begin + *ring_offset;
	}
	if (UploadDmaVerify()) {
		// The ring bytes as this recording's graphics copies read them (the whole command buffer
		// runs after the transfer, and the ring range is not reused before this tick completes).
		auto snapshot = std::make_shared<std::vector<uint8_t>>(
		    m_staging_buffer.Mapped().data() + begin, m_staging_buffer.Mapped().data() + end);
		auto readback = std::make_shared<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0,
		                                         vk::BufferUsageFlagBits::eTransferDst, size);
		auto               native = m_scheduler.Current().Handle();
		const vk::BufferCopy copy {*ring_offset, 0, size};
		native.copyBuffer(m_upload_dma->RingHandle(), readback->Handle(), 1, &copy);
		vk::MemoryBarrier to_host {};
		to_host.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
		to_host.dstAccessMask = vk::AccessFlagBits::eHostRead;
		native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
		                       vk::PipelineStageFlagBits::eHost, {}, 1, &to_host, 0, nullptr, 0,
		                       nullptr);
		m_scheduler.DeferOperation([snapshot, readback, size] {
			readback->Invalidate(0, size);
			Profiler::CountFrameEvent(Profiler::FrameEvent::UploadDmaVerifyChecks);
			if (std::memcmp(readback->Mapped().data(), snapshot->data(), size) != 0) {
				Profiler::CountFrameEvent(Profiler::FrameEvent::UploadDmaVerifyMismatches);
				static std::atomic<uint32_t> logged {0};
				if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
					std::printf("Upload DMA verify: %" PRIu64 " staged bytes differ in the ring\n",
					            size);
				}
			}
		});
	}
	return m_upload_dma->RingHandle();
}

vk::Buffer BufferCache::UploadCopies(Buffer& buffer, std::span<vk::BufferCopy> copies,
                                     uint64_t total_size, size_t guest_copies,
                                     const uint8_t* host_data, uint64_t host_base,
                                     std::vector<UploadHostCopy>* deferred) {
	if (copies.empty()) {
		return nullptr;
	}
	// The first guest_copies read guest memory at their destination; the rest read host_data
	// at (srcOffset - host_base).
	const auto source_of = [&](size_t index, const vk::BufferCopy& copy) -> const void* {
		if (index < guest_copies) {
			return LibKernel::Memory::GuestReadPtr(buffer.CpuAddress() + copy.dstOffset, copy.size);
		}
		return host_data + (copy.srcOffset - host_base);
	};
	Common::DebugCounters::Add(Common::DebugCounters::Counter::BufferUploadBytes, total_size);

	auto [mapped, base_offset] = m_staging_buffer.Map(total_size, 4);
	if (mapped != nullptr) {
		for (size_t index = 0; index < copies.size(); index++) {
			auto& copy = copies[index];
			const void* alias = nullptr;
			if (deferred != nullptr && index < guest_copies) {
				// KYTY_UPLOAD_DMA_HOST_COPY: the DMA worker copies from the backing alias.
				alias = LibKernel::Memory::GuestBackingAlias(buffer.CpuAddress() + copy.dstOffset,
				                                             copy.size);
			}
			if (alias == nullptr && index < guest_copies) {
				PageManager::ApplyPendingProtects(); // KYTY_DEFER_PROTECTS: copied here
			}
			if (alias != nullptr) {
				deferred->push_back({mapped + copy.srcOffset, static_cast<const uint8_t*>(alias),
				                     copy.size, buffer.CpuAddress() + copy.dstOffset});
			} else {
				std::memcpy(mapped + copy.srcOffset, source_of(index, copy), copy.size);
			}
			copy.srcOffset += base_offset;
		}
		m_staging_buffer.Commit();
		return m_staging_buffer.Handle();
	}

	auto temporary = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Upload, 0,
	                                         vk::BufferUsageFlagBits::eTransferSrc, total_size);
	PageManager::ApplyPendingProtects(); // KYTY_DEFER_PROTECTS: copied here
	for (size_t index = 0; index < copies.size(); index++) {
		const auto& copy = copies[index];
		std::memcpy(temporary->Mapped().data() + copy.srcOffset, source_of(index, copy),
		            copy.size);
	}
	temporary->Flush(0, total_size);
	const auto handle = temporary->Handle();
	m_scheduler.DeferOperation([owner = std::move(temporary)]() mutable { owner.reset(); });
	return handle;
}

void BufferCache::RecordRangeFact(uint64_t vaddr, uint64_t size, uint64_t signature,
                                  RangeFact fact) {
	auto& memo     = RangeMemoSlot(vaddr, size);
	memo.vaddr     = vaddr;
	memo.size      = size;
	memo.signature = signature;
	memo.fact      = fact;
	Profiler::CountFrameEvent(Profiler::FrameEvent::BufferRangeMemoRecords);
	m_range_memo_totals.records++;
}

void BufferCache::ClassifyRangeMemoDifference(const char* what, uint64_t vaddr, uint64_t size,
                                              uint64_t dirtied_before) {
	// Every transition that turns a page CPU-dirty advances its region's dirtying serial under the
	// region lock before the bit changes, and the hit's signature was read after `dirtied_before`.
	// Unchanged serials therefore mean the page was already dirty when the hit took place.
	if (dirtied_before == 0 ||
	    m_memory_tracker.RangeDirtiedSignature(vaddr, size) != dirtied_before) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::BufferRangeMemoVerifyRaces);
		m_range_memo_totals.verify_races++;
		return;
	}
	ReportRangeMemoMismatch(what, vaddr, size);
}

void BufferCache::ReportRangeMemoMismatch(const char* what, uint64_t vaddr, uint64_t size) {
	Profiler::CountFrameEvent(Profiler::FrameEvent::BufferRangeMemoVerifyMismatches);
	m_range_memo_totals.verify_mismatches++;
	static std::atomic<uint32_t> logged {0};
	if (logged.fetch_add(1, std::memory_order_relaxed) < 32) {
		std::fprintf(stderr,
		             "BufferRangeMemoVerify: %s: addr=0x%016" PRIx64 " size=0x%" PRIx64 "\n", what,
		             vaddr, size);
	}
	if (RangeMemoVerifyMode() == 2) {
		EXIT("BufferRangeMemoVerify: %s: addr=0x%016" PRIx64 " size=0x%" PRIx64 "\n", what, vaddr,
		     size);
	}
}

bool BufferCache::RelaxedDirtySnapshot(uint64_t vaddr, uint64_t size,
                                       MemoryTracker::DirtyState& state) {
	if (!m_relaxed_queries || !GuestGpu::IsGpuThread()) {
		return false;
	}
	const bool verify    = RelaxedVerifyMode() != 0;
	const auto signature = verify ? m_memory_tracker.RangeSignature(vaddr, size) : 0;
	if (!m_memory_tracker.QueryDirtyRelaxed(vaddr, size, state)) {
		return false;
	}
	Profiler::CountFrameEvent(Profiler::FrameEvent::TrackerRelaxedQueries);
	m_relaxed_totals.queries++;
	if (verify) {
		// Every region exists, so the locked query creates none.
		const auto locked = m_memory_tracker.QueryDirty(vaddr, size);
		VerifyRelaxedSnapshot(vaddr, size, state, locked, signature);
	}
	return true;
}

bool BufferCache::QueryUploadSnapshot(const MemoryTracker& tracker, uint64_t vaddr, uint64_t size,
                                      MemoryTracker::DirtyState& state, bool& cpu_only) {
	cpu_only = g_cpu_only_query.On();
	if (!cpu_only) {
		return tracker.QueryDirtyRelaxed(vaddr, size, state);
	}
	state = {};
	return tracker.QueryCpuDirtyRelaxed(vaddr, size, state.cpu);
}

bool BufferCache::RelaxedNothingToUpload(uint64_t vaddr, uint64_t size) {
	if (!m_relaxed_queries || !GuestGpu::IsGpuThread()) {
		return false;
	}
	const bool                verify    = RelaxedVerifyMode() != 0;
	const auto                signature = verify ? m_memory_tracker.RangeSignature(vaddr, size) : 0;
	MemoryTracker::DirtyState state;
	bool                     cpu_only = false;
	if (!QueryUploadSnapshot(m_memory_tracker, vaddr, size, state, cpu_only) || state.cpu) {
		return false;
	}
	if (verify) {
		const auto locked = m_memory_tracker.QueryDirty(vaddr, size);
		if (!VerifyRelaxedSnapshot(vaddr, size, state, locked, signature, cpu_only) || locked.cpu) {
			// A page turned CPU-dirty in between: synchronize it now, as the locked path would.
			return false;
		}
	}
	Profiler::CountFrameEvent(Profiler::FrameEvent::TrackerRelaxedSyncSkips);
	m_relaxed_totals.sync_skips++;
	return true;
}

bool BufferCache::VerifyRelaxedSnapshot(uint64_t vaddr, uint64_t size,
                                        const MemoryTracker::DirtyState& relaxed,
                                        const MemoryTracker::DirtyState& locked,
                                        uint64_t                         signature, bool cpu_only) {
	Profiler::CountFrameEvent(Profiler::FrameEvent::TrackerRelaxedVerifyChecks);
	if (relaxed.cpu == locked.cpu && (cpu_only || relaxed.gpu == locked.gpu)) {
		return true;
	}
	// Other threads only make pages CPU-dirty and publish GPU-dirty ones, and every such change
	// advances the range's mutation serials first.
	const bool forbidden = (relaxed.cpu && !locked.cpu) ||
	                       (!cpu_only && !relaxed.gpu && locked.gpu);
	const bool quiet =
	    signature != 0 && m_memory_tracker.RangeSignature(vaddr, size) == signature;
	if (!forbidden && !quiet) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::TrackerRelaxedVerifyRaces);
		return true;
	}
	Profiler::CountFrameEvent(Profiler::FrameEvent::TrackerRelaxedVerifyMismatches);
	m_relaxed_totals.mismatches++;
	static std::atomic<uint32_t> logged {0};
	if (logged.fetch_add(1, std::memory_order_relaxed) < 32) {
		std::fprintf(stderr,
		             "TrackerRelaxedVerify: addr=0x%016" PRIx64 " size=0x%" PRIx64
		             " relaxed cpu=%d gpu=%d, locked cpu=%d gpu=%d, serials %s\n",
		             vaddr, size, relaxed.cpu, relaxed.gpu, locked.cpu, locked.gpu,
		             quiet ? "unchanged" : "moved");
	}
	if (RelaxedVerifyMode() == 2) {
		EXIT("TrackerRelaxedVerify: a lock-free dirty snapshot differs from the locked query\n");
	}
	return false;
}

std::pair<Buffer*, uint64_t> BufferCache::ObtainBuffer(uint64_t vaddr, uint64_t size,
                                                       bool is_written, bool is_texel_buffer,
                                                       BufferId id) {
	auto& command = m_scheduler.Current();
	if (command.IsInvalid() || !GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: buffer request requires a recording command buffer\n");
	}
	const auto plain = [&]() -> std::pair<Buffer*, uint64_t> {
		if (!is_written && !is_texel_buffer && m_binding_memo != nullptr &&
		    size < BindingMemoMaxSize && GuestGpu::IsGpuThread()) {
			return ObtainReadBinding(vaddr, size, id);
		}
		if (is_written && !is_texel_buffer && m_written_epoch != nullptr &&
		    GuestGpu::IsGpuThread()) {
			return ObtainWrittenEpoch(vaddr, size, {}, false, id);
		}
		return ObtainBufferNow(vaddr, size, is_written, is_texel_buffer, id, nullptr);
	};
	if (!FlightRecDraw::Keyed()) [[likely]] {
		return plain();
	}
	// KYTY_FLIGHTREC keyed draws: the range's state before the lookup and the lookup's result.
	const auto signature = m_memory_tracker.RangeSignature(vaddr, size);
	uint64_t   flags     = 0;
	flags |= m_memory_tracker.IsRegionCpuModified(vaddr, size) ? FlightRec::BindCpuDirty : 0;
	flags |= m_gpu_modified_ranges.Intersects(vaddr, size) ? FlightRec::BindGpuModified : 0;
	flags |= m_memory_tracker.IsRangeGpuOwned(vaddr, size) ? FlightRec::BindGpuOwned : 0;
	flags |= m_side != nullptr && OverlapsPendingSideReadback(vaddr, vaddr + size)
	             ? FlightRec::BindSideReadback
	             : 0;
	flags |= is_written ? FlightRec::BindWritten : 0;
	flags |= is_texel_buffer ? FlightRec::BindTexel : 0;
	flags |= signature == 0 ? FlightRec::BindNoRegion : 0;
	FlightRec::Rec(FlightRec::EvBindState, vaddr, size, flags, signature);
	m_last_read_hit   = false;
	const auto result = plain();
	const bool stream = result.first == &m_stream_buffer;
	FlightRec::Rec(FlightRec::EvBindResult, vaddr, size,
	               stream || result.first == nullptr ? 0 : result.first->CpuAddress(),
	               result.second | (m_last_read_hit ? 1ull << 63u : 0));
	return result;
}

namespace {
uint64_t WrittenRangesKey(std::span<const GuestRange> written) {
	uint64_t hash = 0xcbf29ce484222325ull;
	for (const auto& range: written) {
		hash = (hash ^ range.address) * 0x100000001b3ull;
		hash = (hash ^ range.size) * 0x100000001b3ull;
	}
	return hash ^ written.size();
}
} // namespace

std::pair<Buffer*, uint64_t> BufferCache::ObtainWrittenEpoch(uint64_t vaddr, uint64_t size,
                                                             std::span<const GuestRange> written,
                                                             bool narrowed, BufferId id) {
	const auto full = [&](BufferId* obtained) {
		return narrowed ? ObtainWrittenBufferNow(vaddr, size, written, id, obtained)
		                : ObtainBufferNow(vaddr, size, true, false, id, obtained);
	};
	auto& totals = m_written_epoch_totals;
	if ((++totals.lookups & 0xfffu) == 0) {
		PrintWrittenEpochSummary();
	}
	const uint64_t key  = narrowed ? WrittenRangesKey(written) : 0;
	const auto     hash = (vaddr >> 4u) * 0x9e3779b97f4a7c15ull ^ size * 0xc2b2ae3d27d4eb4full ^
	                  key * 0xff51afd7ed558ccdull;
	auto& slot = m_written_epoch[static_cast<size_t>(hash >> 52u) & (WrittenEpochSlots - 1)];
	// Loaded before the full path, so a record taken from them is never newer than its binding.
	const auto structure = m_bda_structure_epoch.load(std::memory_order_acquire);
	const auto removals  = m_gpu_range_removals.load(std::memory_order_acquire);
	const bool key_match = slot.valid && slot.vaddr == vaddr && slot.size == size &&
	                       slot.ranges == key && slot.narrowed == narrowed && slot.caller == id;
	if (key_match && structure != UINT64_MAX && slot.structure == structure &&
	    slot.removals == removals && !ImageWritebackOnGpuWriteEnabled()) {
		const auto signature = m_memory_tracker.RangeSignature(vaddr, size);
		if (signature != 0 && signature == slot.signature && !IsBufferInvalid(slot.id) &&
		    m_slot_buffers[slot.id].IsInBounds(vaddr, size)) {
			totals.hits++;
			if (m_written_epoch_mode == 1) {
				return ApplyWrittenEpochHit(slot.id, vaddr, size, written, narrowed);
			}
			// verify / exit: the full path decides; the accepted binding must equal it.
			totals.verify_checks++;
			auto&      expected_buffer = m_slot_buffers[slot.id];
			const std::pair<Buffer*, uint64_t> expected {&expected_buffer,
			                                             expected_buffer.Offset(vaddr)};
			bool contained = true;
			if (narrowed) {
				for (const auto& range: written) {
					contained = contained && m_gpu_modified_ranges.Contains(range.address, range.size);
				}
			} else {
				contained = m_gpu_modified_ranges.Contains(vaddr, size);
			}
			// The invariant the record proved, under the region locks now: no CPU-dirty or hot
			// page in the range and every written range GPU-owned. Failing while the signature
			// still reads the recorded value is a transition that did not advance a serial; with
			// the signature moved, a transition landed after the acceptance (a race).
			bool clean = !m_memory_tracker.IsRegionCpuModified(vaddr, size);
			if (narrowed) {
				for (const auto& range: written) {
					clean = clean && m_memory_tracker.IsRangeGpuOwned(range.address, range.size);
				}
			} else {
				clean = clean && m_memory_tracker.IsRangeGpuOwned(vaddr, size);
			}
			const bool  quiet    = m_memory_tracker.RangeSignature(vaddr, size) == slot.signature;
			BufferId    obtained {};
			const auto  fresh    = full(&obtained);
			bool        raced    = false;
			const char* problem  = nullptr;
			if (fresh != expected) {
				problem = "a different buffer or offset";
			} else if (!contained) {
				problem = "a written range outside the GPU-modified ranges";
			} else if (!clean) {
				problem = "the range is CPU-dirty or not GPU-owned";
				raced   = !quiet;
			}
			if (problem != nullptr && raced) {
				totals.verify_races++;
			} else if (problem != nullptr) {
				totals.verify_mismatches++;
				static std::atomic<uint32_t> logged {0};
				if (logged.fetch_add(1, std::memory_order_relaxed) < 32) {
					std::printf("CommitEpochFastVerify: %s: addr=0x%016" PRIx64 " size=0x%" PRIx64
					            " narrowed=%d\n",
					            problem, vaddr, size, narrowed ? 1 : 0);
					std::fflush(stdout);
				}
				if (m_written_epoch_mode == 3) {
					EXIT("CommitEpochFastVerify: %s: addr=0x%016" PRIx64 " size=0x%" PRIx64 "\n",
					     problem, vaddr, size);
				}
			}
			RecordWrittenEpoch(slot, vaddr, size, key, narrowed, id, obtained, structure, removals,
			                   written);
			return fresh;
		}
	}
	BufferId   obtained {};
	const auto result = full(&obtained);
	RecordWrittenEpoch(slot, vaddr, size, key, narrowed, id, obtained, structure, removals,
	                   written);
	return result;
}

std::pair<Buffer*, uint64_t> BufferCache::ApplyWrittenEpochHit(BufferId id, uint64_t vaddr,
                                                               uint64_t size,
                                                               std::span<const GuestRange> written,
                                                               bool narrowed) {
	// The full path's order without its no-op synchronizations, image walk and dirty-range add.
	auto& buffer = m_slot_buffers[id];
	TouchBuffer(buffer);
	buffer.MarkContentWritten();
	if (!narrowed) {
		{
			const TilerTrace::Reason canon_reason("epoch");
			NoteBufferContentWrite(vaddr, size);
		}
		ForgetKnownFills(vaddr, size);
		HangTrace::NoteGpuWrite(vaddr, size);
	} else {
		for (const auto& range: written) {
			{
				const TilerTrace::Reason canon_reason("epoch");
				NoteBufferContentWrite(range.address, range.size);
			}
			ForgetKnownFills(range.address, range.size);
			HangTrace::NoteGpuWrite(range.address, range.size);
		}
	}
	return {&buffer, buffer.Offset(vaddr)};
}

void BufferCache::RecordWrittenEpoch(WrittenEpochMemo& slot, uint64_t vaddr, uint64_t size,
                                     uint64_t key, bool narrowed, BufferId caller,
                                     BufferId obtained, uint64_t structure, uint64_t removals,
                                     std::span<const GuestRange> written) {
	slot.valid = false;
	if (structure == UINT64_MAX || IsBufferInvalid(obtained) ||
	    !m_slot_buffers[obtained].IsInBounds(vaddr, size) || ImageWritebackOnGpuWriteEnabled()) {
		return;
	}
	const auto signature = m_memory_tracker.RangeSignature(vaddr, size);
	if (signature == 0) {
		return;
	}
	// Nothing the read synchronization would upload (no CPU-dirty or hot page), and every written
	// range GPU-owned (a written synchronization skips) and already in the GPU-modified set.
	MemoryTracker::DirtyState state;
	if (!m_memory_tracker.QueryDirtyRelaxed(vaddr, size, state) || state.cpu) {
		return;
	}
	const auto owned = [&](uint64_t address, uint64_t bytes) {
		return m_memory_tracker.IsRangeGpuOwned(address, bytes) &&
		       m_gpu_modified_ranges.Contains(address, bytes);
	};
	if (narrowed) {
		for (const auto& range: written) {
			if (!owned(range.address, range.size)) {
				return;
			}
		}
	} else if (!owned(vaddr, size)) {
		return;
	}
	if (m_memory_tracker.RangeSignature(vaddr, size) != signature) {
		return;
	}
	slot.vaddr     = vaddr;
	slot.size      = size;
	slot.ranges    = key;
	slot.signature = signature;
	slot.structure = structure;
	slot.removals  = removals;
	slot.caller    = caller;
	slot.id        = obtained;
	slot.narrowed  = narrowed;
	slot.valid     = true;
	m_written_epoch_totals.records++;
}

void BufferCache::PrintWrittenEpochSummary() {
	auto&      totals = m_written_epoch_totals;
	const auto now    = std::chrono::duration_cast<std::chrono::nanoseconds>(
                         std::chrono::steady_clock::now().time_since_epoch())
                         .count();
	if (totals.printed_ns == 0) {
		totals.printed_ns = now;
		return;
	}
	if (now - totals.printed_ns < 10'000'000'000LL) {
		return;
	}
	const double lookups = static_cast<double>(std::max<uint64_t>(1, totals.lookups));
	std::printf("CommitEpochFast 10s: written bindings %llu, accepted %llu (%.1f%%), records %llu | "
	            "verify %llu checks, %llu mismatches, %llu races | read cross-epoch from clean memo "
	            "%llu, verify %llu checks, %llu mismatches, %llu races\n",
	            static_cast<unsigned long long>(totals.lookups),
	            static_cast<unsigned long long>(totals.hits),
	            100.0 * static_cast<double>(totals.hits) / lookups,
	            static_cast<unsigned long long>(totals.records),
	            static_cast<unsigned long long>(totals.verify_checks),
	            static_cast<unsigned long long>(totals.verify_mismatches),
	            static_cast<unsigned long long>(totals.verify_races),
	            static_cast<unsigned long long>(totals.read_fast),
	            static_cast<unsigned long long>(totals.read_verify_checks),
	            static_cast<unsigned long long>(totals.read_verify_mismatches),
	            static_cast<unsigned long long>(totals.read_verify_races));
	std::fflush(stdout);
	totals            = {};
	totals.printed_ns = now;
}

void BufferCache::PrefetchReadBinding(uint64_t vaddr, uint64_t size, bool at_find) const noexcept {
	if (m_binding_memo == nullptr || g_binding_batch_prefetch.Get() != (at_find ? 2 : 1)) {
		return;
	}
	const auto* slot =
	    &m_binding_memo[static_cast<size_t>(BindingMemoHash(vaddr, size) >> m_binding_memo_shift)];
#if defined(__clang__) || defined(__GNUC__)
	__builtin_prefetch(slot, 0, 3);
#elif defined(_M_X64) || defined(_M_IX86)
	_mm_prefetch(reinterpret_cast<const char*>(slot), _MM_HINT_T0);
#endif
}

// KYTY_BINDING_MEMO_STATS=1 (diagnostic, default off): ObtainReadBinding outcomes, one console line
// every 10 s (GPU thread only).
namespace {
struct BindingMemoStats {
	uint64_t calls = 0, no_region = 0, empty = 0, other_key = 0, signature = 0, guard = 0,
	         epoch = 0, stream_hits = 0, cached_hits = 0, slow_stream = 0;
	std::chrono::steady_clock::time_point last {};
};
bool BindingMemoStatsEnabled() {
	static std::atomic<int64_t> on_once {Common::OnceUnset};
	const auto on = static_cast<bool>(Common::Once(on_once, [] {
		const char* value = std::getenv("KYTY_BINDING_MEMO_STATS");
		return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
	}));
	return on;
}
BindingMemoStats& GetBindingMemoStats() {
	static BindingMemoStats stats;
	return stats;
}
void PrintBindingMemoStats() {
	auto& st = GetBindingMemoStats();
	if ((st.calls & 0xffffu) != 0) {
		return;
	}
	const auto now = std::chrono::steady_clock::now();
	if (st.last.time_since_epoch().count() == 0) {
		st.last = now;
		return;
	}
	if (now - st.last < std::chrono::seconds(10)) {
		return;
	}
	const double c = static_cast<double>(std::max<uint64_t>(1, st.calls)) / 100.0;
	std::printf("BindingMemoStats 10s: calls %llu hits stream %.1f%% cached %.1f%% | misses: no-region "
	            "%.1f%% empty %.1f%% other-key %.1f%% signature %.1f%% guard %.1f%% epoch %.1f%% | slow "
	            "path stream result %.1f%%\n",
	            static_cast<unsigned long long>(st.calls), static_cast<double>(st.stream_hits) / c,
	            static_cast<double>(st.cached_hits) / c, static_cast<double>(st.no_region) / c,
	            static_cast<double>(st.empty) / c, static_cast<double>(st.other_key) / c,
	            static_cast<double>(st.signature) / c, static_cast<double>(st.guard) / c,
	            static_cast<double>(st.epoch) / c, static_cast<double>(st.slow_stream) / c);
	std::fflush(stdout);
	st      = {};
	st.last = now;
}
} // namespace

// KYTY_CP_COMMIT_STATS=steps: the memo lookup's time by outcome (diagnostic only).
std::pair<Buffer*, uint64_t> BufferCache::ObtainReadBinding(uint64_t vaddr, uint64_t size,
                                                            BufferId id) {
	if (!CommitStats::StepsEnabled()) [[likely]] {
		return ObtainReadBindingMemo(vaddr, size, id);
	}
	m_last_read_hit   = false;
	const auto start  = CommitStats::Now();
	const auto result = ObtainReadBindingMemo(vaddr, size, id);
	const bool hit    = m_last_read_hit;
	CommitStats::AddSub(hit ? CommitStats::Sub::ReadHit : CommitStats::Sub::ReadMiss,
	                    CommitStats::Now() - start);
	CommitStats::AddItems(hit ? CommitStats::Item::ReadHits : CommitStats::Item::ReadMisses, 1);
	return result;
}

std::pair<Buffer*, uint64_t> BufferCache::ObtainReadBindingMemo(uint64_t vaddr, uint64_t size,
                                                                BufferId id) {
	// KYTY_BINDING_EPOCH_MEMO (bufferCache.h).
	// KYTY_CP_BINDING_MEMO_PREFETCH (default off, live): issue the read hint before the tracker
	// query, so its region/serial loads can overlap the random memo-line fetch. This is a CPU cache
	// hint only: every key, signature, epoch, tick and structure check below still runs.
	m_binding_hot_enabled = g_binding_hot_memo.On();
	if (m_binding_hot_enabled) {
		const auto generation = g_binding_hot_generation.load(std::memory_order_relaxed);
		if (generation == UINT64_MAX || m_binding_hot_generation != generation) {
			for (auto& entry: m_binding_hot_memo) {
				entry.kind = BindingMemoKind::Empty;
			}
			m_binding_hot_generation = generation;
		}
	}
	auto& full = BindingMemoSlot(vaddr, size);
	auto& hot  = BindingHotMemoSlot(vaddr, size);
	const bool hot_key = m_binding_hot_enabled && hot.kind != BindingMemoKind::Empty &&
	                     hot.vaddr == vaddr && hot.size == size;
	auto& memo = hot_key ? hot : full;
	if (!hot_key && g_binding_memo_prefetch.On()) {
#if defined(__clang__) || defined(__GNUC__)
		__builtin_prefetch(&memo, 0, 3);
#elif defined(_M_X64) || defined(_M_IX86)
		_mm_prefetch(reinterpret_cast<const char*>(&memo), _MM_HINT_T0);
#endif
	}
	const auto sig_start = CommitStats::StepsEnabled() ? CommitStats::Now() : 0;
	const auto epoch  = SyncEpoch::Current();
	const auto before = m_memory_tracker.RangeSignature(vaddr, size);
	if (sig_start != 0) {
		CommitStats::AddSub(CommitStats::Sub::ReadSig, CommitStats::Now() - sig_start);
	}
	const bool key    = memo.kind != BindingMemoKind::Empty && memo.vaddr == vaddr && memo.size == size;
	BindingMemoStats* stats = nullptr;
	if (BindingMemoStatsEnabled()) [[unlikely]] {
		stats = &GetBindingMemoStats();
		stats->calls++;
	}
	using LookupOutcome = BufferLookupStats::Outcome;
	using LookupMiss    = BufferLookupStats::Miss;
	if (BufferLookupStats::Enabled() && BufferLookupStats::KeysEnabled()) [[unlikely]] {
		BufferLookupStats::NoteKey(vaddr, size, m_memory_tracker.Frame());
	}
	if (before == 0 || !key) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::BindingEpochMemoMissSlot);
		if (stats != nullptr) {
			(before == 0 ? stats->no_region
			             : memo.kind == BindingMemoKind::Empty ? stats->empty : stats->other_key)++;
		}
		BufferLookupStats::AddMiss(before == 0                            ? LookupMiss::NoRegion
		                           : memo.kind == BindingMemoKind::Empty ? LookupMiss::Empty
		                                                                  : LookupMiss::OtherKey);
	} else if (memo.signature != before) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::BindingEpochMemoMissSignature);
		if (stats != nullptr) {
			stats->signature++;
		}
		BufferLookupStats::AddMiss(LookupMiss::Signature);
	} else if (const bool stream = memo.kind == BindingMemoKind::Stream;
	           memo.guard != (stream ? m_scheduler.CurrentTick()
	                                 : m_bda_structure_epoch.load(std::memory_order_acquire))) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::BindingEpochMemoMissGuard);
		if (stats != nullptr) {
			stats->guard++;
		}
		BufferLookupStats::AddMiss(LookupMiss::Guard);
	} else {
		bool cross = false;
		if (memo.epoch != epoch) {
			// KYTY_BINDING_MEMO_CROSS_EPOCH (bufferCache.h): only a cache-buffer memo, and only
			// once no page of the range is CPU-dirty while the signature still holds. The relaxed
			// mirror answers "CPU-dirty" exactly on this thread (only it clears those bits); the
			// locked query also waits for a transition that advanced the serial but has not
			// changed its bits yet.
			MemoryTracker::DirtyState relaxed;
			bool                     cpu_only = false;
			// ASTROHYBRID_COMMIT_EPOCH_FASTPATH (BindingMemo::clean): the signature equals
			// memo.signature (checked above), under which the locked query below once passed.
			const bool fast = !stream && m_binding_memo_cross && memo.clean &&
			                  m_written_epoch_mode != 0;
			if (fast && m_written_epoch_mode == 1) {
				cross = true;
				m_written_epoch_totals.read_fast++;
			} else {
				cross = !stream && m_binding_memo_cross &&
				        QueryUploadSnapshot(m_memory_tracker, vaddr, size, relaxed, cpu_only) &&
				        !relaxed.cpu && !m_memory_tracker.IsRegionCpuModified(vaddr, size) &&
				        m_memory_tracker.RangeSignature(vaddr, size) == before;
				if (fast) {
					auto& totals = m_written_epoch_totals;
					totals.read_fast++;
					totals.read_verify_checks++;
					if (!cross) {
						if (m_memory_tracker.RangeSignature(vaddr, size) != before) {
							totals.read_verify_races++;
						} else {
							totals.read_verify_mismatches++;
							static std::atomic<uint32_t> logged {0};
							if (logged.fetch_add(1, std::memory_order_relaxed) < 32) {
								std::printf("CommitEpochFastVerify: a clean read memo has CPU-dirty "
								            "pages: addr=0x%016" PRIx64 " size=0x%" PRIx64 "\n",
								            vaddr, size);
								std::fflush(stdout);
							}
							if (m_written_epoch_mode == 3) {
								EXIT("CommitEpochFastVerify: a clean read memo has CPU-dirty pages\n");
							}
						}
					}
				}
				if (cross && m_written_epoch_mode != 0) {
					memo.clean = true;
				}
			}
			if (!cross) {
				if (stats != nullptr) {
					stats->epoch++;
				}
				BufferLookupStats::AddMiss(LookupMiss::Epoch);
				if (!stream && m_binding_memo_cross) {
					Profiler::CountFrameEvent(Profiler::FrameEvent::BindingEpochMemoCrossRejects);
					m_binding_memo_totals.cross_rejects++;
				} else {
					Profiler::CountFrameEvent(Profiler::FrameEvent::BindingEpochMemoMissEpoch);
				}
			} else {
				Profiler::CountFrameEvent(Profiler::FrameEvent::BindingEpochMemoCrossHits);
				m_binding_memo_totals.cross_hits++;
				memo.epoch = epoch;
			}
		}
		if (memo.epoch == epoch) {
			m_last_read_hit = true;
			if (m_binding_hot_enabled) {
				if (hot_key) {
					m_binding_memo_totals.hot_hits++;
				} else {
					hot = memo;
				}
			}
			std::pair<Buffer*, uint64_t> hit {nullptr, memo.offset};
			if (stats != nullptr) {
				(stream ? stats->stream_hits : stats->cached_hits)++;
				PrintBindingMemoStats();
			}
			if (stream) {
				Profiler::CountFrameEvent(Profiler::FrameEvent::BindingEpochMemoStreamHits);
				m_binding_memo_totals.stream_hits++;
				hit.first = &m_stream_buffer;
				BufferLookupStats::SetOutcome(LookupOutcome::MemoStream);
			} else {
				Profiler::CountFrameEvent(Profiler::FrameEvent::BindingEpochMemoCachedHits);
				m_binding_memo_totals.cached_hits++;
				const BufferLookupStats::PartScope touch_scope(BufferLookupStats::Part::ReadTouch);
				hit.first = memo.buffer;
				if (!g_binding_memo_touch.On()) {
					TouchBuffer(*hit.first);
				} else {
					// KYTY_BINDING_MEMO_TOUCH (bufferCache.h).
					if (memo.touched != m_gc_tick) {
						TouchBuffer(*hit.first);
						memo.touched = m_gc_tick;
					} else {
						m_binding_memo_totals.touch_skips++;
					}
				}
				BufferLookupStats::SetOutcome(cross ? LookupOutcome::MemoCross : LookupOutcome::MemoHit);
			}
			if (m_binding_memo_verify != 0) {
				const auto copy = memo;
				return VerifyBindingHit(copy, hit, id, cross);
			}
			return hit;
		}
	}
	if (std::pair<Buffer*, uint64_t> lean {};
	    g_binding_lean_miss.On() && TryLeanReadMiss(vaddr, size, epoch, before, id, lean)) {
		return lean;
	}
	BufferId                     obtained {};
	BufferLookupStats::PartScope now_scope(BufferLookupStats::Part::ReadNow);
	const auto                   result = ObtainBufferNow(vaddr, size, false, false, id, &obtained);
	now_scope.Stop();
	if (stats != nullptr) {
		stats->slow_stream += result.first == &m_stream_buffer ? 1u : 0u;
		PrintBindingMemoStats();
	}
	BufferLookupStats::SetOutcome(result.first == &m_stream_buffer ? LookupOutcome::MissStream
	                                                               : LookupOutcome::MissCached);
	const BufferLookupStats::PartScope record_scope(BufferLookupStats::Part::ReadRecord);
	RecordBinding(vaddr, size, epoch, before, result, obtained);
	return result;
}

bool BufferCache::TryLeanReadMiss(uint64_t vaddr, uint64_t size, uint64_t epoch, uint64_t before,
                                  BufferId id, std::pair<Buffer*, uint64_t>& result) {
	if (before == 0 || !m_relaxed_queries || IsBufferInvalid(id) ||
	    !m_slot_buffers[id].IsInBounds(vaddr, size) ||
	    (SnapshotCleanReadsEnabled() && size <= CACHING_PAGESIZE)) {
		// KYTY_SNAPSHOT_CLEAN_READS: the normal path copies such a range into the stream buffer.
		return false;
	}
	const BufferLookupStats::PartScope lean_scope(BufferLookupStats::Part::ReadNow);
	bool cpu_dirty = true;
	if (!m_memory_tracker.QueryCpuDirtyRelaxed(vaddr, size, cpu_dirty) || cpu_dirty ||
	    m_memory_tracker.RangeSignature(vaddr, size) != before) {
		return false;
	}
	const auto structure = m_bda_structure_epoch.load(std::memory_order_acquire);
	if (structure == UINT64_MAX) {
		return false;
	}
	auto& buffer = m_slot_buffers[id];
	result       = {&buffer, buffer.Offset(vaddr)};
	m_binding_memo_totals.lean_misses++;
	BufferLookupStats::SetOutcome(BufferLookupStats::Outcome::MissLean);
	if (g_binding_lean_miss.Get() >= 2) {
		// The normal path decides (and synchronizes) the binding.
		m_binding_memo_totals.lean_verify_checks++;
		BufferId   obtained {};
		const auto normal = ObtainBufferNow(vaddr, size, false, false, id, &obtained);
		if (normal != result) {
			if (m_memory_tracker.RangeSignature(vaddr, size) != before) {
				m_binding_memo_totals.lean_verify_races++;
			} else {
				m_binding_memo_totals.lean_verify_mismatches++;
				static std::atomic<uint32_t> logged {0};
				if (logged.fetch_add(1, std::memory_order_relaxed) < 32) {
					std::fprintf(stderr,
					             "BindingLeanMissVerify: the normal path bound another buffer or "
					             "offset: addr=0x%016" PRIx64 " size=0x%" PRIx64 "\n",
					             vaddr, size);
				}
				if (g_binding_lean_miss.Get() == 3) {
					EXIT("BindingLeanMissVerify: the normal path bound another buffer or offset: "
					     "addr=0x%016" PRIx64 " size=0x%" PRIx64 "\n",
					     vaddr, size);
				}
			}
		}
		result = normal;
		RecordBinding(vaddr, size, epoch, before, normal, obtained);
		return true;
	}
	TouchBuffer(buffer);
	// RecordBinding's cache-buffer entry, with the signature the decision was taken under.
	BindingMemo entry;
	entry.vaddr     = vaddr;
	entry.size      = static_cast<uint32_t>(size);
	entry.epoch     = epoch;
	entry.offset    = result.second;
	entry.signature = before;
	entry.guard     = structure;
	entry.buffer    = &buffer;
	entry.touched   = m_gc_tick;
	entry.kind      = BindingMemoKind::Cached;
	BindingMemoSlot(vaddr, size) = entry;
	if (m_binding_hot_enabled) {
		BindingHotMemoSlot(vaddr, size) = entry;
	}
	Profiler::CountFrameEvent(Profiler::FrameEvent::BindingEpochMemoRecords);
	m_binding_memo_totals.records++;
	return true;
}

void BufferCache::RecordBinding(uint64_t vaddr, uint64_t size, uint64_t epoch, uint64_t before,
                                const std::pair<Buffer*, uint64_t>& result, BufferId id) {
	if (before == 0) {
		return; // a tracker region did not exist yet (its creation made the range CPU-dirty)
	}
	BindingMemo entry;
	entry.vaddr  = vaddr;
	entry.size   = static_cast<uint32_t>(size);
	entry.epoch  = epoch;
	entry.offset = result.second;
	entry.buffer = result.first;
	if (result.first == &m_stream_buffer) {
		// The decision and the copy saw one tracker state: no transition since `before`.
		if (m_memory_tracker.RangeSignature(vaddr, size) != before) {
			return;
		}
		entry.signature = before;
		entry.guard     = m_scheduler.CurrentTick();
		entry.kind      = BindingMemoKind::Stream;
	} else {
		const auto structure = m_bda_structure_epoch.load(std::memory_order_acquire);
		const auto after     = m_memory_tracker.RangeSignature(vaddr, size);
		if (structure == UINT64_MAX || after == 0 || IsBufferInvalid(id) ||
		    &m_slot_buffers[id] != result.first) {
			return;
		}
		if (size <= CACHING_PAGESIZE) {
			// The next small read decides from the bits `after` pins: record only when that is no
			// stream copy (not after a stream decision whose ring allocation failed, nor with hot
			// pages left CPU-dirty and nothing GPU-dirty).
			MemoryTracker::DirtyState state;
			// KYTY_SNAPSHOT_CLEAN_READS: any small read without GPU-dirty pages is a stream copy.
			if (!m_memory_tracker.QueryDirtyRelaxed(vaddr, size, state) ||
			    ((state.cpu || SnapshotCleanReadsEnabled()) && !state.gpu) ||
			    m_memory_tracker.RangeSignature(vaddr, size) != after) {
				return;
			}
		}
		entry.signature = after;
		entry.guard     = structure;
		// The normal path touched the buffer in this tick (ObtainBufferNow).
		entry.touched   = m_gc_tick;
		entry.kind      = BindingMemoKind::Cached;
	}
	BindingMemoSlot(vaddr, size) = entry;
	if (m_binding_hot_enabled) {
		BindingHotMemoSlot(vaddr, size) = entry;
	}
	Profiler::CountFrameEvent(Profiler::FrameEvent::BindingEpochMemoRecords);
	m_binding_memo_totals.records++;
}

std::pair<Buffer*, uint64_t> BufferCache::VerifyBindingHit(const BindingMemo&           memo,
                                                           std::pair<Buffer*, uint64_t> hit,
                                                           BufferId id, bool cross) {
	Profiler::CountFrameEvent(Profiler::FrameEvent::BindingEpochMemoVerifyChecks);
	m_binding_memo_totals.verify_checks++;
	const auto vaddr = memo.vaddr;
	const uint64_t size = memo.size;
	// The tracker bits the hit relied on and the decision the normal path takes from them. A
	// transition since the hit's signature read (a guest write fault) raced this check.
	const auto state          = m_memory_tracker.QueryDirty(vaddr, size);
	const bool quiet          = m_memory_tracker.RangeSignature(vaddr, size) == memo.signature;
	const bool decides_stream =
	    size <= CACHING_PAGESIZE && (state.cpu || SnapshotCleanReadsEnabled()) && !state.gpu;
	BufferId   obtained {};
	const auto fresh = ObtainBufferNow(vaddr, size, false, false, id, &obtained);
	const char* problem = nullptr;
	bool        race    = false;
	if (memo.kind == BindingMemoKind::Stream) {
		if (!decides_stream) {
			problem = "a stream-copy hit whose range is no longer copied";
		} else if (fresh.first == &m_stream_buffer) {
			race = std::memcmp(m_stream_buffer.Mapped().data() + hit.second,
			                   m_stream_buffer.Mapped().data() + fresh.second,
			                   static_cast<size_t>(size)) != 0;
		}
		// Otherwise the ring could not take the copy without waiting: the normal path bound the
		// cache buffer instead, as it may.
	} else if (decides_stream) {
		problem = "a cache-buffer hit whose range is now a stream copy";
	} else if (fresh != hit) {
		problem = "a cache-buffer hit bound a different buffer or offset";
	} else if (cross && state.cpu) {
		// The cross-epoch check found no CPU-dirty page under the region locks with this
		// signature: one now needs a transition since, which `quiet` rules out below.
		problem = "a cross-epoch cache-buffer hit whose range has CPU-dirty pages";
	} else {
		// Pages the synchronization uploaded: hot pages written since, or pages a racing fault
		// dirtied before the hit's signature was taken.
		race = state.cpu;
	}
	if (problem != nullptr && !quiet) {
		problem = nullptr;
		race    = true;
	}
	if (race) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::BindingEpochMemoVerifyRaces);
		m_binding_memo_totals.verify_races++;
	}
	if (problem != nullptr) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::BindingEpochMemoVerifyMismatches);
		m_binding_memo_totals.verify_mismatches++;
		static std::atomic<uint32_t> logged {0};
		if (logged.fetch_add(1, std::memory_order_relaxed) < 32) {
			std::fprintf(stderr,
			             "BindingEpochMemoVerify: %s: addr=0x%016" PRIx64 " size=0x%" PRIx64 "\n",
			             problem, vaddr, size);
		}
		if (m_binding_memo_verify == 2) {
			EXIT("BindingEpochMemoVerify: %s: addr=0x%016" PRIx64 " size=0x%" PRIx64 "\n", problem,
			     vaddr, size);
		}
	}
	return fresh;
}

bool StreamCopyDefer::Enabled() {
	static const bool enabled = ParseEnvU64("KYTY_STREAM_COPY_DEFER", 0) != 0;
	return enabled;
}

void BufferCache::PruneDeferredCopies(const CommandRecorder& recorder) {
	const auto consumed = recorder.RingConsumed();
	while (m_deferred_head < m_deferred.size() && m_deferred[m_deferred_head].position <= consumed) {
		m_deferred_head++;
	}
	if (m_deferred_head == m_deferred.size()) {
		m_deferred.clear();
		m_deferred_head = 0;
		m_deferred_lo   = UINT64_MAX;
		m_deferred_hi   = 0;
	}
}

bool BufferCache::TryDeferStreamCopy(uint8_t* mapped, uint64_t vaddr, uint64_t size) {
	if (!StreamCopyDefer::Enabled() || m_range_memo_verify != 0 || m_binding_memo_verify != 0 ||
	    !m_stream_buffer.IsCoherent() || !CpCommit::Enabled(CpCommit::Part::StreamRead) ||
	    !GuestGpu::IsGpuThread()) {
		return false;
	}
	const auto& command  = m_scheduler.Current();
	auto*       recorder = command.Recorder();
	if (recorder == nullptr || recorder->GetMode() != CommandRecorder::Mode::Thread) {
		return false;
	}
	if (m_deferred_recorder != recorder) {
		// A different recorder (never in practice): the pending copies of the old one drain first.
		if (m_deferred_recorder != nullptr && !m_deferred.empty()) {
			m_deferred_recorder->Drain(nullptr, false);
		}
		m_deferred.clear();
		m_deferred_head     = 0;
		m_deferred_lo       = UINT64_MAX;
		m_deferred_hi       = 0;
		m_deferred_recorder = recorder;
	}
	if (!CommandSink(command).hostCopy(mapped, vaddr, size)) {
		return false; // a direct window: the recorder is drained, copy now
	}
	g_defer_cache = this;
	PruneDeferredCopies(*recorder);
	m_deferred.push_back({vaddr, vaddr + size, recorder->RingWritePosition()});
	m_deferred_lo = std::min(m_deferred_lo, vaddr);
	m_deferred_hi = std::max(m_deferred_hi, vaddr + size);
	return true;
}

void StreamCopyDefer::NoteGpuOnlyWrite(uint64_t address, uint64_t size) {
	if (auto* dma = g_write_drain_dma; dma != nullptr && size != 0 && UploadDmaWriteDrainEnabled() &&
	                                   GuestGpu::IsGpuThread()) {
		(void)dma->DrainHostCopies(address, size, false);
	}
	NoteGuestWriteStream(address, size);
}

void StreamCopyDefer::NoteGuestWrite(uint64_t address, uint64_t size) {
	// KYTY_UPLOAD_DMA_WRITE_DRAIN (uploadDma.h): the guest may observe this write (a label) and
	// then rewrite memory an earlier upload still has to read, so every queued host copy is made
	// first.
	if (auto* dma = g_write_drain_dma; dma != nullptr && size != 0 && UploadDmaWriteDrainEnabled() &&
	                                   GuestGpu::IsGpuThread()) {
		(void)dma->DrainHostCopies(address, size, true);
	}
	NoteGuestWriteStream(address, size);
}

void StreamCopyDefer::NoteGuestWriteStream(uint64_t address, uint64_t size) {
	auto* cache = g_defer_cache;
	if (cache == nullptr || cache->m_deferred.empty() || size == 0 || !GuestGpu::IsGpuThread()) {
		return;
	}
	const auto end = address + size;
	if (end <= cache->m_deferred_lo || address >= cache->m_deferred_hi) {
		return;
	}
	auto* recorder = cache->m_deferred_recorder;
	cache->PruneDeferredCopies(*recorder);
	bool overlap = false;
	for (size_t i = cache->m_deferred_head; i < cache->m_deferred.size() && !overlap; i++) {
		const auto& copy = cache->m_deferred[i];
		overlap          = address < copy.end && copy.begin < end;
	}
	if (!overlap) {
		return;
	}
	static std::atomic<uint64_t> drains {0};
	if (drains.fetch_add(1, std::memory_order_relaxed) < 8) {
		std::printf("StreamCopyDefer: a CP write [0x%016" PRIx64 ", +0x%" PRIx64
		            ") overlaps a pending stream copy: recorder drained\n",
		            address, size);
		std::fflush(stdout);
	}
	recorder->Drain(nullptr, false);
	cache->m_deferred.clear();
	cache->m_deferred_head = 0;
	cache->m_deferred_lo   = UINT64_MAX;
	cache->m_deferred_hi   = 0;
}

std::pair<Buffer*, uint64_t> BufferCache::ObtainBufferNow(uint64_t vaddr, uint64_t size,
                                                          bool is_written, bool is_texel_buffer,
                                                          BufferId id, BufferId* obtained) {
	if (!is_written && size <= CACHING_PAGESIZE) {
		// A small read of a CPU-dirty range that is not GPU-dirty is copied into the stream buffer.
		// KYTY_BUFFER_RANGE_MEMO: the decision depends only on the range's tracker bits, so a
		// Stream or Clean fact recorded under the current signature decides it without the two
		// locked queries (the bytes are copied again every time).
		// KYTY_BUFFER_DIRTY_QUERY_COMBINED (default on): both tracker queries with each region
		// lock taken once (MemoryTracker::QueryDirty), the same decision.
		const auto decide = [&] {
			if (DirtyQueryCombinedEnabled()) {
				Profiler::CountFrameEvent(Profiler::FrameEvent::BufferDirtyQueriesCombined);
				const auto state = m_memory_tracker.QueryDirty(vaddr, size);
				return !state.gpu && state.cpu;
			}
			return !m_memory_tracker.IsRegionGpuModified(vaddr, size) &&
			       m_memory_tracker.IsRegionCpuModified(vaddr, size);
		};
		bool                      stream = false;
		MemoryTracker::DirtyState relaxed;
		if (RelaxedDirtySnapshot(vaddr, size, relaxed)) {
			// KYTY_TRACKER_RELAXED_QUERIES: the same decision from the lock-free mirrors.
			stream = !relaxed.gpu && relaxed.cpu;
		} else if (m_range_memo != nullptr) {
			// Verify mode: dirtying serials before the lookup (ClassifyRangeMemoDifference).
			const auto  dirtied = m_range_memo_verify != 0
			                          ? m_memory_tracker.RangeDirtiedSignature(vaddr, size)
			                          : uint64_t {0};
			const auto  signature = m_memory_tracker.RangeSignature(vaddr, size);
			const auto& memo      = RangeMemoSlot(vaddr, size);
			if (signature != 0 && memo.signature == signature && memo.vaddr == vaddr &&
			    memo.size == size) {
				// Clean: no page is CPU-dirty, so not a stream copy.
				stream = memo.fact == RangeFact::Stream;
				if (stream) {
					Profiler::CountFrameEvent(Profiler::FrameEvent::BufferRangeMemoStreamHits);
					m_range_memo_totals.stream_hits++;
				}
				if (m_range_memo_verify != 0) {
					Profiler::CountFrameEvent(Profiler::FrameEvent::BufferRangeMemoVerifyChecks);
					m_range_memo_totals.verify_checks++;
					RunRangeMemoVerifyHook(vaddr, size);
					const bool decided = decide();
					if (decided != stream) {
						if (!stream) {
							// Clean, CPU-dirty now: a page may have turned dirty since the lookup.
							ClassifyRangeMemoDifference(
							    "a small read binding changed its stream decision", vaddr, size,
							    dirtied);
						} else {
							// Only this thread clears CPU-dirty pages and sets GPU-dirty ones.
							ReportRangeMemoMismatch(
							    "a small read binding changed its stream decision", vaddr, size);
						}
						stream = decided;
					}
				}
			} else {
				stream = decide();
				if (stream && signature != 0 &&
				    m_memory_tracker.RangeSignature(vaddr, size) == signature) {
					RecordRangeFact(vaddr, size, signature, RangeFact::Stream);
				}
			}
		} else {
			stream = decide();
		}
		if (!stream && SnapshotCleanReadsEnabled()) {
			// KYTY_SNAPSHOT_CLEAN_READS: clean (not GPU-dirty) small reads are copied as well.
			MemoryTracker::DirtyState state;
			if (!RelaxedDirtySnapshot(vaddr, size, state)) {
				state = m_memory_tracker.QueryDirty(vaddr, size);
			}
			stream = !state.gpu;
		}
		if (stream) {
			const auto alignment = std::max<uint64_t>(
			    m_graphics.physical_device_properties.limits.minUniformBufferOffsetAlignment, 1);
			auto [mapped, offset] = m_stream_buffer.Map(size, alignment, false);
			bool read = false;
			if (mapped != nullptr && StreamDirectReadEnabled()) {
				std::memcpy(mapped, LibKernel::Memory::GuestReadPtr(vaddr, size), size);
				FlightRec::Rec(FlightRec::EvSnapRead, vaddr, size, 1);
				read = true;
			} else if (mapped != nullptr && TryDeferStreamCopy(mapped, vaddr, size)) {
				read = true; // KYTY_STREAM_COPY_DEFER: the recorder copies the bytes
			} else if (mapped != nullptr) {
				// KYTY_CP_COMMIT=streamread: the reservation is committed only after a successful
				// read, so the direct read may leave partial bytes in it when it fails.
				const uint64_t probe_t0 = StreamCopyProbe() ? __rdtsc() : 0;
				read = CpCommit::Enabled(CpCommit::Part::StreamRead)
				           ? Libs::LibKernel::Memory::TryReadBackingDirect(vaddr, mapped, size)
				           : Libs::LibKernel::Memory::TryReadBacking(vaddr, mapped, size);
				if (probe_t0 != 0) {
					StreamCopyProbeAdd(size, __rdtsc() - probe_t0, vaddr);
				}
			}
			if (read) {
				m_stream_buffer.Commit();
				return {&m_stream_buffer, offset};
			}
		}
	}

	if (IsBufferInvalid(id) || !m_slot_buffers[id].IsInBounds(vaddr, size)) {
		id = FindBuffer(vaddr, size);
	}
	if (obtained != nullptr) {
		*obtained = id;
	}
	auto& buffer = m_slot_buffers[id];
	TouchBuffer(buffer);
	(void)SynchronizeBuffer(buffer, vaddr, size, is_written, is_texel_buffer);
	if (is_written) {
		PreserveImagesForGpuWrite(id, vaddr, size);
		// Writable descriptors reserve a new version before recording their shader commands.
		buffer.MarkContentWritten();
		// An Add that changes nothing cannot stale a cached clean page, because pages with
		// dirty bytes are never cached clean. The tracker GPU bits that SynchronizeBuffer just
		// set are only ever set together with this Add; clean-read verdicts never read them.
		// A range already contained is left as it is (the Add would change nothing).
		if (!m_gpu_modified_ranges.Contains(vaddr, size)) {
			CleanVerdict::Invalidate(vaddr, size, Coherence::Source::BufferDirtyAdd);
			m_gpu_modified_ranges.Add(vaddr, size);
		}
		{
			const TilerTrace::Reason canon_reason("obtain");
			NoteBufferContentWrite(vaddr, size);
		}
		ForgetKnownFills(vaddr, size);
		HangTrace::NoteGpuWrite(vaddr, size);
	}
	return {&buffer, buffer.Offset(vaddr)};
}

std::pair<Buffer*, uint64_t> BufferCache::ObtainWrittenBuffer(uint64_t vaddr, uint64_t size,
                                                              std::span<const GuestRange> written,
                                                              BufferId                    id) {
	auto& command = m_scheduler.Current();
	if (command.IsInvalid() || !GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: buffer request requires a recording command buffer\n");
	}
	for (const auto& range: written) {
		if (!range.Valid() || range.address < vaddr || range.End() > vaddr + size) {
			EXIT("BufferCache: written range 0x%016" PRIx64 "+0x%" PRIx64
			     " is outside its binding 0x%016" PRIx64 "+0x%" PRIx64 "\n",
			     range.address, range.size, vaddr, size);
		}
	}
	if (m_written_epoch != nullptr && GuestGpu::IsGpuThread()) {
		return ObtainWrittenEpoch(vaddr, size, written, true, id);
	}
	return ObtainWrittenBufferNow(vaddr, size, written, id, nullptr);
}

std::pair<Buffer*, uint64_t> BufferCache::ObtainWrittenBufferNow(uint64_t vaddr, uint64_t size,
                                                                 std::span<const GuestRange> written,
                                                                 BufferId id, BufferId* obtained) {
	if (IsBufferInvalid(id) || !m_slot_buffers[id].IsInBounds(vaddr, size)) {
		id = FindBuffer(vaddr, size);
	}
	if (obtained != nullptr) {
		*obtained = id;
	}
	auto& buffer = m_slot_buffers[id];
	TouchBuffer(buffer);
	// Bytes the shader cannot write only need the upload a read binding gets. Each written range
	// then uploads anything dirtied meanwhile and becomes GPU-owned under the same tracker locks,
	// exactly as a whole writable binding does.
	(void)SynchronizeBuffer(buffer, vaddr, size, false, false, nullptr, "written-binding");
	// Writable descriptors reserve a new version before recording their shader commands.
	buffer.MarkContentWritten();
	for (const auto& range: written) {
		(void)SynchronizeBuffer(buffer, range.address, range.size, true, false, nullptr,
		                        "written-binding");
		PreserveImagesForGpuWrite(id, range.address, range.size);
		if (!m_gpu_modified_ranges.Contains(range.address, range.size)) {
			CleanVerdict::Invalidate(range.address, range.size, Coherence::Source::BufferDirtyAdd);
			m_gpu_modified_ranges.Add(range.address, range.size);
		}
		{
			const TilerTrace::Reason canon_reason("obtain_written");
			NoteBufferContentWrite(range.address, range.size);
		}
		ForgetKnownFills(range.address, range.size);
		HangTrace::NoteGpuWrite(range.address, range.size);
	}
	return {&buffer, buffer.Offset(vaddr)};
}

std::pair<Buffer*, uint64_t> BufferCache::ObtainImageStagingBuffer(uint64_t size) {
	auto [mapped, offset] = m_staging_buffer.Map(size, 16);
	if (mapped != nullptr) return {&m_staging_buffer, offset};
	// The ring (fixed small ring or the default one) must not impose a maximum guest image size
	// (upstream 9bcf80bf6: a 576 MiB image exceeded the default 512 MiB ring). Keep this one-off
	// source alive until the submission using it has completed, as UploadCopies does for buffers.
	auto temporary = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Upload, 0,
	                                         AllFlags, size);
	auto* source = temporary.get();
	m_scheduler.DeferOperation([owner = std::move(temporary)]() mutable { owner.reset(); });
	return {source, 0};
}

std::pair<Buffer*, uint64_t> BufferCache::ObtainBufferForImage(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid image source\n");
	}
	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	if (owner != nullptr && *owner) {
		auto& buffer = m_slot_buffers[*owner];
		if (buffer.IsInBounds(vaddr, size)) {
			TouchBuffer(buffer);
			(void)SynchronizeBuffer(buffer, vaddr, size, false, false, nullptr, "image-source");
			return {&buffer, buffer.Offset(vaddr)};
		}
	}
	if (IsRegionGpuModified(vaddr, size)) {
		return ObtainBuffer(vaddr, size, false, false);
	}

	auto [source, stage_offset] = ObtainImageStagingBuffer(size);
	if (source == nullptr) EXIT("BufferCache: failed to read mapped guest image backing\n");
	auto* staging = source->Mapped().data() + stage_offset;
	if (!Libs::LibKernel::Memory::TryReadBacking(vaddr, staging, size) &&
	    !Libs::LibKernel::Memory::TryReadSparseBacking(vaddr, staging, size)) {
		EXIT("BufferCache: failed to read mapped guest image backing\n");
	}
	if (source == &m_staging_buffer) m_staging_buffer.Commit();
	else source->Flush(0, size);
	return {source, stage_offset};
}

void BufferCache::FillBuffer(uint64_t vaddr, uint64_t size, uint32_t value, bool is_gds) {
	if ((vaddr & 3u) != 0 || size == 0 || (size & 3u) != 0 || size > UINT64_MAX - vaddr) {
		EXIT("BufferCache: fill range must be dword aligned\n");
	}
	if (is_gds) {
		if (vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - vaddr) {
			EXIT("BufferCache: GDS fill range is out of bounds\n");
		}
		TilerTrace::Count("fill.who.gds", 1, size);
		m_gds_buffer.Fill(vaddr, size, value);
		return;
	}
	if (vaddr == 0) {
		EXIT("BufferCache: invalid fill memory address\n");
	}
	(void)m_texture_cache.ClearMeta(vaddr);
	if (!IsRegionGpuModified(vaddr, size)) {
		// Access the guest mapping so write faults invalidate cached buffers and images.
		FlightRec::Rec(FlightRec::EvCpWrite, FlightRec::CpDmaFillCpu, vaddr, size, value);
		TilerTrace::Count("fill.who.guest_cpu(no-gpu-fill)", 1, size);
		auto* destination = Guest::Ptr<uint32_t>(vaddr);
		std::fill(destination, destination + size / sizeof(uint32_t), value);
		return;
	}

	// Obtaining the destination moves overlapping GPU-modified images into it first (their
	// bytes outside the fill survive); only then do those images lose GPU ownership.
	const bool preserve_images = ImageWritebackOnGpuWriteEnabled();
	if (!preserve_images) {
		{ // KYTY_TILER_TRACE: who marks images buffer-modified
			const TilerTrace::Reason bwrite_reason("fill");
			m_texture_cache.InvalidateMemoryFromGPU(vaddr, size);
		}
	}
	HangTrace::ScopedGpuWriteKind trace_kind(HangTrace::GpuWriteKind::Fill);
	auto [dst, dst_offset] = ObtainBuffer(vaddr, size, true, true);
	if (preserve_images) {
		{ // KYTY_TILER_TRACE: who marks images buffer-modified
			const TilerTrace::Reason bwrite_reason("fill");
			m_texture_cache.InvalidateMemoryFromGPU(vaddr, size);
		}
	}
	TilerTrace::Count("fill.who.guest_gpu", 1, size);
	dst->Fill(dst_offset, size, value);
	RecordKnownFill(vaddr, size, value);
}

bool BufferCache::TryWriteDataGpu(uint64_t vaddr, const uint32_t* data, uint64_t size) {
	if (vaddr == 0 || data == nullptr || size == 0 || (vaddr & 3u) != 0 || (size & 3u) != 0 ||
	    size > 65536 || !GuestRange {vaddr, size}.Valid()) {
		return false;
	}
	// Only bytes owned by recorded GPU work need ordering on the GPU timeline; everything else
	// keeps the CPU write (which faults into the usual invalidation when tracked).
	if (!HasGpuDirtyBytes(vaddr, size) && !HasPendingBackingPublication(vaddr, size)) {
		return false;
	}
	KYTY_GPU_OP_SITE("buffercache.write_data");
	// As for fills and copies: obtaining the destination first moves overlapping GPU-modified
	// images into it (KYTY_IMAGE_WRITEBACK_ON_GPU_WRITE); only then do they lose GPU ownership.
	const bool preserve_images = ImageWritebackOnGpuWriteEnabled();
	if (!preserve_images) {
		{ // KYTY_TILER_TRACE: who marks images buffer-modified
			const TilerTrace::Reason bwrite_reason("write_data");
			m_texture_cache.InvalidateMemoryFromGPU(vaddr, size);
		}
	}
	HangTrace::ScopedGpuWriteKind trace_kind(HangTrace::GpuWriteKind::Copy);
	auto [buffer, offset] = ObtainBuffer(vaddr, size, true, true);
	if (preserve_images) {
		{ // KYTY_TILER_TRACE: who marks images buffer-modified
			const TilerTrace::Reason bwrite_reason("write_data");
			m_texture_cache.InvalidateMemoryFromGPU(vaddr, size);
		}
	}
	auto& command         = m_scheduler.Current();
	command.EndRendering();
	const auto              native = command.Handle();
	vk::BufferMemoryBarrier before {};
	before.srcAccessMask       = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	before.dstAccessMask       = vk::AccessFlagBits::eTransferWrite;
	before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.buffer              = buffer->Handle();
	before.offset              = offset;
	before.size                = size;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                       vk::PipelineStageFlagBits::eTransfer, {}, 0, nullptr, 1, &before, 0,
	                       nullptr);
	native.updateBuffer(buffer->Handle(), offset, size, data);
	auto after          = before;
	after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                       vk::PipelineStageFlagBits::eAllCommands, {}, 0, nullptr, 1, &after, 0,
	                       nullptr);
	return true;
}

void BufferCache::RecordKnownFill(uint64_t vaddr, uint64_t size, uint32_t value) {
	if (size == 0) {
		return;
	}
	std::scoped_lock lock(m_known_fill_mutex);
	ForgetKnownFillsLocked(vaddr, size);
	if (m_known_fills.size() >= 64) {
		m_known_fills.erase(m_known_fills.begin());
	}
	m_known_fills.push_back({vaddr, size, value});
	m_has_known_fills.store(true, std::memory_order_release);
	m_known_fill_generation.fetch_add(1, std::memory_order_acq_rel);
}

std::optional<uint32_t> BufferCache::KnownFill(uint64_t vaddr, uint64_t size) const {
	std::scoped_lock lock(m_known_fill_mutex);
	return KnownFillLocked(vaddr, size);
}

std::optional<uint32_t> BufferCache::KnownFill(uint64_t vaddr, uint64_t size,
                                               uint64_t& generation) const {
	std::scoped_lock lock(m_known_fill_mutex);
	generation = m_known_fill_generation.load(std::memory_order_relaxed);
	return KnownFillLocked(vaddr, size);
}

std::optional<uint32_t> BufferCache::KnownFillLocked(uint64_t vaddr, uint64_t size) const {
	// The range may be covered by several adjacent fills (e.g. per-slice consumption); all of
	// them must carry the same value.
	const uint64_t          end    = vaddr + size;
	uint64_t                cursor = vaddr;
	std::optional<uint32_t> value;
	while (cursor < end) {
		const auto covering =
		    std::find_if(m_known_fills.begin(), m_known_fills.end(), [cursor](const KnownFillRange& fill) {
			    return cursor >= fill.address && cursor < fill.address + fill.size;
		    });
		if (covering == m_known_fills.end() || (value && *value != covering->value)) {
			return std::nullopt;
		}
		value  = covering->value;
		cursor = covering->address + covering->size;
	}
	return value;
}

void BufferCache::ForgetKnownFills(uint64_t vaddr, uint64_t size) {
	// Every guest write fault and writable binding lands here; almost always nothing is known.
	// A fill recorded concurrently with this check is ordered after this write either way.
	if (!m_has_known_fills.load(std::memory_order_acquire)) {
		return;
	}
	std::scoped_lock lock(m_known_fill_mutex);
	ForgetKnownFillsLocked(vaddr, size);
}

void BufferCache::ForgetKnownFillsLocked(uint64_t vaddr, uint64_t size) {
	// Keep the parts of each fill outside the written range.
	const uint64_t end      = vaddr + size;
	const bool     overlaps = std::any_of(
        m_known_fills.begin(), m_known_fills.end(), [vaddr, end](const KnownFillRange& fill) {
            return vaddr < fill.address + fill.size && fill.address < end;
        });
	if (!overlaps) {
		return;
	}
	std::vector<KnownFillRange> kept;
	kept.reserve(m_known_fills.size() + 1);
	for (const auto& fill: m_known_fills) {
		const uint64_t fill_end = fill.address + fill.size;
		if (end <= fill.address || fill_end <= vaddr) {
			kept.push_back(fill);
			continue;
		}
		if (fill.address < vaddr) {
			kept.push_back({fill.address, vaddr - fill.address, fill.value});
		}
		if (end < fill_end) {
			kept.push_back({end, fill_end - end, fill.value});
		}
	}
	m_known_fills.swap(kept);
	m_has_known_fills.store(!m_known_fills.empty(), std::memory_order_release);
	m_known_fill_generation.fetch_add(1, std::memory_order_acq_rel);
	if (!GuestGpu::IsGpuThread()) {
		m_known_fill_foreign.fetch_add(1, std::memory_order_acq_rel);
	}
}

void BufferCache::CopyBuffer(uint64_t dst_vaddr, uint64_t src_vaddr, uint64_t size, bool dst_gds,
                             bool src_gds) {
	const bool dst_memory = !dst_gds;
	const bool src_memory = !src_gds;
	if ((dst_memory && dst_vaddr == 0) || (src_memory && src_vaddr == 0) || size == 0 ||
	    ((dst_gds || src_gds) && ((dst_vaddr | src_vaddr | size) & 3u) != 0) ||
	    size > UINT64_MAX - dst_vaddr || size > UINT64_MAX - src_vaddr || (dst_gds && src_gds) ||
	    (dst_gds && (dst_vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - dst_vaddr)) ||
	    (src_gds && (src_vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - src_vaddr))) {
		EXIT("BufferCache: invalid copy range, src=0x%016" PRIx64 " dst=0x%016" PRIx64
		     " size=0x%016" PRIx64 " src_gds=%d dst_gds=%d\n",
		     src_vaddr, dst_vaddr, size, static_cast<int>(src_gds), static_cast<int>(dst_gds));
	}
	// KYTY_ALIAS_BYTES: bytes a GPU-modified image owns without starting at the source (or at the
	// destination) are not in guest memory either; the GPU path moves them into the buffer.
	if (src_memory && dst_memory && !IsRegionGpuModified(dst_vaddr, size) &&
	    !IsRegionGpuModified(src_vaddr, size) && !m_texture_cache.FindImageFromRange(src_vaddr, size) &&
	    (!TextureCache::AliasBytesEnabled() ||
	     (!m_texture_cache.IsRegionGpuModified(src_vaddr, size) &&
	      !m_texture_cache.IsRegionGpuModified(dst_vaddr, size)))) {
		FlightRec::Rec(FlightRec::EvCpWrite, FlightRec::CpDmaCopyCpu, dst_vaddr, size, src_vaddr);
		if (!g_cpu_copy_page_skip.On()) {
			std::memcpy(Guest::Ptr<void>(dst_vaddr), Guest::Ptr<const void>(src_vaddr), size);
		} else if ((dst_vaddr >= src_vaddr ? dst_vaddr - src_vaddr : src_vaddr - dst_vaddr) < size) {
			// Page-by-page copying would corrupt later source bytes of an overlapping copy.
			std::memmove(Guest::Ptr<void>(dst_vaddr), Guest::Ptr<const void>(src_vaddr), size);
		} else {
			for (uint64_t at = 0; at < size;) {
				const auto bytes = std::min(TRACKER_PAGE_SIZE - (dst_vaddr + at) % TRACKER_PAGE_SIZE, size - at);
				auto* to = Guest::Ptr<void>(dst_vaddr + at);
				const auto* from = Guest::Ptr<const void>(src_vaddr + at);
				// Only a successful clean-backing comparison can suppress a write. Dirty GPU
				// ownership is checked above; an unmapped/failed proof takes the original copy.
				if (LibKernel::Memory::CompareGpuCleanBacking(dst_vaddr + at, from, bytes) !=
				    LibKernel::Memory::BackingCompare::Equal) {
					std::memcpy(to, from, bytes);
				}
				at += bytes;
			}
		}
		return;
	}

	auto& command = m_scheduler.Current();
	// The source synchronizes from a GPU-modified image at its address and the destination moves
	// overlapping GPU-modified images into itself before those images lose GPU ownership.
	const bool preserve_images = ImageWritebackOnGpuWriteEnabled();
	if (dst_memory && !preserve_images) {
		{ // KYTY_TILER_TRACE: who marks images buffer-modified
			const TilerTrace::Reason bwrite_reason("copy");
			m_texture_cache.InvalidateMemoryFromGPU(dst_vaddr, size);
		}
	}
	const auto src_id      = src_memory ? FindBuffer(src_vaddr, size) : BufferId {};
	const auto dst_id      = dst_memory ? FindBuffer(dst_vaddr, size) : BufferId {};
	auto [src, src_offset] = src_memory ? ObtainBuffer(src_vaddr, size, false, true, src_id)
	                                    : std::pair {&m_gds_buffer, src_vaddr};
	HangTrace::ScopedGpuWriteKind trace_kind(HangTrace::GpuWriteKind::Copy);
	auto [dst, dst_offset] = dst_memory ? ObtainBuffer(dst_vaddr, size, true, true, dst_id)
	                                    : std::pair {&m_gds_buffer, dst_vaddr};
	if (dst_memory && preserve_images) {
		{ // KYTY_TILER_TRACE: who marks images buffer-modified
			const TilerTrace::Reason bwrite_reason("copy");
			m_texture_cache.InvalidateMemoryFromGPU(dst_vaddr, size);
		}
	}
	if (src_memory && src->is_deleted) {
		// Obtaining the destination created a buffer that joined (and deleted) the source's. The
		// deleted buffer is freed once the current tick completes, and that synchronization may
		// already have submitted the tick the join was recorded in: copy from the joined buffer.
		std::tie(src, src_offset) = ObtainBuffer(src_vaddr, size, false, true, BufferId {});
		EXIT_IF(src->is_deleted);
	}
	dst->CopyFrom(command, *src, src_offset, dst_offset, size);
}

Buffer& BufferCache::PrepareImageBytes(const RangeSet& ranges, Buffer* target) {
	uint64_t first = UINT64_MAX;
	uint64_t last  = 0;
	ranges.ForEach([&](uint64_t begin, uint64_t end) {
		first = std::min(first, begin);
		last  = std::max(last, end);
	});
	EXIT_IF(first >= last);
	Buffer* buffer = target;
	if (buffer == nullptr || buffer->is_deleted || !buffer->IsInBounds(first, last - first)) {
		buffer = &m_slot_buffers[FindBuffer(first, last - first)];
	}
	TouchBuffer(*buffer);
	ranges.ForEach([&](uint64_t begin, uint64_t end) {
		(void)SynchronizeBuffer(*buffer, begin, end - begin, true, false, nullptr, "image-bytes");
	});
	return *buffer;
}

void BufferCache::CommitImageBytes(Buffer& buffer, const RangeSet& ranges) {
	buffer.MarkContentWritten();
	ranges.ForEach([&](uint64_t begin, uint64_t end) {
		const auto size = end - begin;
		if (!m_gpu_modified_ranges.Contains(begin, size)) {
			CleanVerdict::Invalidate(begin, size, Coherence::Source::BufferDirtyAdd);
		}
		m_gpu_modified_ranges.Add(begin, size);
		{
			const TilerTrace::Reason canon_reason("commit_image");
			NoteBufferContentWrite(begin, size);
		}
		ForgetKnownFills(begin, size);
		HangTrace::NoteGpuWrite(begin, size);
	});
}

bool BufferCache::IsRegionRegistered(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid registered-region query\n");
	}
	// Cached buffers are ordered and non-overlapping. The last buffer beginning before the query
	// end is therefore the only possible intersection.
	const auto candidate = m_buffers.lower_bound(vaddr + size);
	if (candidate == m_buffers.begin()) {
		return false;
	}
	const auto& [address, id] = *std::prev(candidate);
	return address + m_slot_buffers[id].Size() > vaddr;
}

bool BufferCache::IsRegionGpuModified(uint64_t vaddr, uint64_t size) {
	return m_memory_tracker.IsRegionGpuModified(vaddr, size);
}

bool BufferCache::IsRegionGpuModifiedRelaxed(uint64_t vaddr, uint64_t size) const {
	return m_memory_tracker.IsRegionGpuModifiedRelaxed(vaddr, size);
}

BufferCache::PageStates BufferCache::CountPageStates(uint64_t vaddr, uint64_t size) const {
	PageStates states;
	if (!GuestRange {vaddr, size}.Valid()) {
		return states;
	}
	const auto end = Common::AlignUp(vaddr + size, TRACKER_PAGE_SIZE);
	for (auto page = Common::AlignDown(vaddr, TRACKER_PAGE_SIZE); page < end;
	     page += TRACKER_PAGE_SIZE) {
		MemoryTracker::DirtyState state;
		if (!m_memory_tracker.QueryDirtyRelaxed(page, TRACKER_PAGE_SIZE, state)) {
			states.untracked++;
		} else if (state.gpu) {
			states.gpu_dirty++;
		} else if (!state.cpu) {
			states.clean++;
		}
	}
	return states;
}

bool BufferCache::GpuDirtyMirrorMatches(uint64_t vaddr, uint64_t size) {
	return m_memory_tracker.GpuMirrorMatches(vaddr, size);
}

bool BufferCache::HasGpuDirtyBytes(uint64_t vaddr, uint64_t size) {
	if (m_gpu_modified_ranges.Intersects(vaddr, size)) {
		return true;
	}
	// KYTY_FALSE_SHARING_WRITES: released before their publication landed.
	return !m_early_released.empty() && OverlapsUnpublished(vaddr, size);
}

std::optional<BufferContentRevision> BufferCache::GetContentRevision(uint64_t vaddr,
	                                                                uint64_t size) {
	EXIT_IF(!GuestGpu::IsGpuThread());
	if (!GuestRange {vaddr, size}.Valid() ||
	    m_memory_tracker.IsRegionCpuModified(vaddr, size) ||
	    HasPendingBackingPublication(vaddr, size)) {
		return std::nullopt;
	}
	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	if (owner == nullptr || IsBufferInvalid(*owner)) {
		return std::nullopt;
	}
	const auto& buffer = m_slot_buffers[*owner];
	if (!buffer.IsInBounds(vaddr, size)) {
		return std::nullopt;
	}
	return BufferContentRevision {*owner, buffer.ContentRevision(), m_content_revision_epoch};
}

void BufferCache::InvalidateContentRevisions() {
	EXIT_IF(!GuestGpu::IsGpuThread() || m_content_revision_epoch == UINT64_MAX);
	// Unbounded GPU writes follow; retire clean-read verdicts along with the revisions.
	CleanVerdict::Invalidate(0, UINT64_MAX, Coherence::Source::ContentRevisions);
	++m_content_revision_epoch;
	// They may change any buffer byte, including bytes of hot pages.
	SettleHotPages(0, 0);
	// Their bytes carry no writer tick, so no readback may skip this recording.
	m_unbounded_write_tick = m_scheduler.CurrentTick();
	// Nor do they clear the GPU ownership of images they may overwrite: order them against image
	// writes instead (TextureCache::SupersedesGpuDirtyBytes).
	m_unbounded_write_serial = Image::NextContentSerial();
}

uint64_t BufferCache::BeginBackingPublication(std::span<const GuestRange> ranges, uint64_t tick) {
	EXIT_IF(!GuestGpu::IsGpuThread() || ranges.empty());
	for (const auto& range: ranges) {
		EXIT_IF(!range.Valid());
	}
	std::lock_guard lock(m_backing_publication_mutex);
	const auto token = ++m_next_backing_publication_token;
	EXIT_IF(token == 0);
	// Pending ranges are not clean for backing reads; retire verdicts before publishing them.
	for (const auto& range: ranges) {
		CleanVerdict::Invalidate(range.address, range.size, Coherence::Source::PublicationBegin);
	}
	m_backing_publications.push_back({token, tick, {ranges.begin(), ranges.end()}});
	RewritePublicationSnapshot();
	m_backing_publication_count.store(m_backing_publications.size(), std::memory_order_release);
	return token;
}

void BufferCache::EndBackingPublication(uint64_t token) {
	std::lock_guard lock(m_backing_publication_mutex);
	const auto found = std::find_if(m_backing_publications.begin(), m_backing_publications.end(),
	                                [token](const auto& entry) { return entry.token == token; });
	EXIT_IF(found == m_backing_publications.end());
	// Backing authority changes here (runs on the priority worker). Bump before the entry is
	// erased so a verdict evaluated after the erase is tagged with the new generation.
	for (const auto& range: found->ranges) {
		CleanVerdict::Invalidate(range.address, range.size, Coherence::Source::PublicationEnd);
	}
	m_backing_publications.erase(found);
	RewritePublicationSnapshot();
	// Publish completion only after the callback has written every registered backing range.
	m_backing_publication_count.store(m_backing_publications.size(), std::memory_order_release);
}

// KYTY_PUBLICATION_SNAPSHOT (live, default 1): HasPendingBackingPublication reads the seqlock copy
// of the pending ranges instead of scanning the list under m_backing_publication_mutex. Every
// recorded read of the draw-prep workers asks it (the read gate's pending-publication hint), and so
// do the command processor's clean predicates; with publications almost always pending, 6 workers
// and the command processor serialized on the mutex (Sky Garden sampling: ~30% of the workers'
// preparation time in the lock). The answer is the locked scan's at one instant inside the call,
// as before. 0 off (the locked scan); verify: both, counts answers that differ twice in a row (a
// registration or completion between two scans can make one pair differ legitimately); exit:
// stops at the first such difference.
static Live::Switch g_publication_snapshot("KYTY_PUBLICATION_SNAPSHOT", [](const char* value) -> int64_t {
	if (value == nullptr || value[0] == '\0') {
		return 1;
	}
	if (std::strcmp(value, "verify") == 0) {
		return 2;
	}
	if (std::strcmp(value, "exit") == 0) {
		return 3;
	}
	return std::strcmp(value, "0") == 0 ? 0 : 1;
});
static std::atomic<uint64_t> g_publication_snapshot_checks {0};
static std::atomic<uint64_t> g_publication_snapshot_mismatches {0};
static std::atomic<uint64_t> g_publication_snapshot_locked {0}; // verify mode: scans that took the mutex

void BufferCache::RewritePublicationSnapshot() {
	uint64_t count = 0;
	for (const auto& entry: m_backing_publications) {
		count += entry.ranges.size();
	}
	// Seqlock writer (serialized by the mutex): odd while the copy changes.
	const auto seq = m_publication_seq.load(std::memory_order_relaxed);
	m_publication_seq.store(seq + 1, std::memory_order_relaxed);
	std::atomic_thread_fence(std::memory_order_release);
	if (count > PublicationSnapshotRanges) {
		m_publication_snapshot_count.store(PublicationSnapshotOverflow, std::memory_order_relaxed);
	} else {
		uint32_t index = 0;
		for (const auto& entry: m_backing_publications) {
			for (const auto& range: entry.ranges) {
				m_publication_begin[index].store(range.address, std::memory_order_relaxed);
				m_publication_end[index].store(range.End(), std::memory_order_relaxed);
				index++;
			}
		}
		m_publication_snapshot_count.store(index, std::memory_order_relaxed);
	}
	m_publication_seq.store(seq + 2, std::memory_order_release);
}

// The seqlock read: false when the copy overflowed or kept changing (the caller scans locked).
bool BufferCache::SnapshotPendingBackingPublication(uint64_t vaddr, uint64_t size,
                                                    bool& answer) const {
	const auto end = vaddr + size;
	for (uint32_t attempt = 0; attempt < 64; attempt++) {
		const auto seq = m_publication_seq.load(std::memory_order_acquire);
		if ((seq & 1u) != 0) {
			continue;
		}
		const auto count = m_publication_snapshot_count.load(std::memory_order_relaxed);
		if (count == PublicationSnapshotOverflow) {
			return false;
		}
		bool found = false;
		for (uint32_t i = 0; i < count && !found; i++) {
			found = m_publication_begin[i].load(std::memory_order_relaxed) < end &&
			        vaddr < m_publication_end[i].load(std::memory_order_relaxed);
		}
		std::atomic_thread_fence(std::memory_order_acquire);
		if (m_publication_seq.load(std::memory_order_relaxed) == seq) {
			answer = found;
			return true;
		}
	}
	return false;
}

bool BufferCache::HasPendingBackingPublication(uint64_t vaddr, uint64_t size) const {
	const auto mode = g_publication_snapshot.Get();
	if (mode == 0) {
		return PendingBackingPublicationTick(vaddr, size).has_value();
	}
	const GuestRange query {vaddr, size};
	EXIT_IF(!query.Valid());
	if (m_backing_publication_count.load(std::memory_order_acquire) == 0) {
		return false;
	}
	const auto locked = [&] { return PendingBackingPublicationTick(vaddr, size).has_value(); };
	bool answer = false;
	if (!SnapshotPendingBackingPublication(vaddr, size, answer)) {
		answer = locked();
		if (mode >= 2) {
			g_publication_snapshot_locked.fetch_add(1, std::memory_order_relaxed);
		}
	}
	if (mode >= 2) [[unlikely]] {
		const auto checks = g_publication_snapshot_checks.fetch_add(1, std::memory_order_relaxed) + 1;
		if (answer != locked()) {
			bool again = false;
			if (!SnapshotPendingBackingPublication(vaddr, size, again)) {
				again = locked();
			}
			if (again != locked()) {
				const auto count =
				    g_publication_snapshot_mismatches.fetch_add(1, std::memory_order_relaxed) + 1;
				if (count <= 16) {
					std::printf("PublicationSnapshot: snapshot and locked scan differ for [0x%" PRIx64
					            ", +0x%" PRIx64 ")\n",
					            vaddr, size);
				}
				if (mode == 3) {
					EXIT("PublicationSnapshot: snapshot and locked scan differ\n");
				}
			}
		}
		if ((checks & 0xffffffu) == 0) {
			std::printf("PublicationSnapshot verify: %" PRIu64 " checks, %" PRIu64
			            " mismatches, %" PRIu64 " locked scans\n",
			            checks, g_publication_snapshot_mismatches.load(std::memory_order_relaxed),
			            g_publication_snapshot_locked.load(std::memory_order_relaxed));
		}
	}
	return answer;
}

std::optional<uint64_t> BufferCache::PendingBackingPublicationTick(uint64_t vaddr,
	                                                               uint64_t size) const {
	const GuestRange query {vaddr, size};
	EXIT_IF(!query.Valid());
	if (m_backing_publication_count.load(std::memory_order_acquire) == 0) {
		return std::nullopt;
	}
	std::lock_guard lock(m_backing_publication_mutex);
	std::optional<uint64_t> latest;
	for (const auto& entry: m_backing_publications) {
		for (const auto& range: entry.ranges) {
			if (range.address < query.End() && query.address < range.End()) {
				latest = latest ? std::max(*latest, entry.tick) : entry.tick;
				break;
			}
		}
	}
	return latest;
}

bool BufferCache::IsRegionCpuModified(uint64_t vaddr, uint64_t size) {
	return m_memory_tracker.IsRegionCpuModified(vaddr, size);
}

void BufferCache::RetireUnusedBuffers(uint64_t frame, uint64_t min_age) {
	if (frame <= min_age) {
		return;
	}
	// The GC tick the newest recorded frame up to frame - min_age began at: an LRU item below it
	// was last used before that frame (LRU ticks are GC ticks).
	const auto limit_frame = frame - min_age;
	const auto newer       = std::upper_bound(m_frame_ticks.begin(), m_frame_ticks.end(), limit_frame,
	                                          [](uint64_t value, const auto& entry) { return value < entry.first; });
	if (newer == m_frame_ticks.begin()) {
		return;
	}
	const auto            limit_tick = std::prev(newer)->second;
	std::vector<BufferId> candidates;
	size_t                scanned = 0;
	m_lru_cache.ForEachItemBelow(limit_tick, [&](BufferId id) {
		const auto& buffer = m_slot_buffers[id];
		if (!buffer.is_deleted && !m_gpu_modified_ranges.Intersects(buffer.CpuAddress(), buffer.Size())) {
			candidates.push_back(id);
		}
		return ++scanned >= 4096 || candidates.size() >= 64;
	});
	if (candidates.empty()) {
		return;
	}
	// As the collector below: settle pending side readbacks before the ownership checks.
	CompleteAllSideReadbacks();
	for (const auto id: candidates) {
		if (IsBufferInvalid(id)) {
			continue;
		}
		auto& buffer = m_slot_buffers[id];
		m_memory_tracker.ValidateGpuDirtyOwnership(m_gpu_modified_ranges, buffer.CpuAddress(),
		                                           buffer.Size(), "idle retirement");
		if (m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size())) {
			continue;
		}
		m_idle_freed_bytes += buffer.Size();
		m_idle_freed++;
		m_memory_tracker.UntrackMemory(buffer.CpuAddress(), buffer.Size());
		FlightRec::Rec(FlightRec::EvIdleRetire, buffer.CpuAddress(), buffer.Size(),
		               (static_cast<uint64_t>(id.index) << 32u) | id.generation);
		DeleteBuffer(id);
	}
}

void BufferCache::RunGarbageCollector() {
	// A pass slower than 2 ms records EvGcPass (a3 = duration in us); the retires are
	// EvGcRetire/EvIdleRetire.
	struct GcFlight {
		uint64_t t0 = FlightRec::Detail::Now();
		~GcFlight() {
			const uint64_t us = FlightRec::ToNs(FlightRec::Detail::Now() - t0) / 1000ull;
			if (us > 2000) {
				FlightRec::Rec(FlightRec::EvGcPass, 0, 0, 0, us);
			}
		}
	} gc_flight;
	MaintainHotPages();
	const auto tick = m_gc_tick++;
	if (m_graphics.CanReportMemoryUsage()) {
		m_total_used_memory = m_graphics.GetDeviceMemoryUsage();
	}
	const uint64_t frame = m_memory_tracker.Frame();
	if (m_idle_frames != 0 || m_pressure_frames != 0) {
		if (m_frame_ticks.empty() || m_frame_ticks.back().first != frame) {
			m_frame_ticks.emplace_back(frame, tick);
		}
		const auto keep = std::max(m_idle_frames, m_pressure_frames) + 64;
		while (m_frame_ticks.size() > keep) {
			m_frame_ticks.pop_front();
		}
	}
	if (m_idle_frames != 0 && frame >= m_idle_next_frame) {
		m_idle_next_frame = frame + 32;
		RetireUnusedBuffers(frame, m_idle_frames);
	}
	// KYTY_VRAM_GC_BUDGET (vramBudget.h): the marks follow the budget as it is now; the collector
	// below (blind to BDA reads) only runs above the planning budget itself.
	auto trigger  = m_trigger_gc_memory;
	auto critical = m_critical_gc_memory;
	if (VramBudget::GcEnabled() && m_graphics.CanReportMemoryUsage()) {
		// The budget once per frame (this runs on the command-processor thread after every
		// completed submission).
		if (frame != m_budget_frame) {
			m_budget_frame    = frame;
			const auto budget = m_graphics.GetTotalMemoryBudget();
			m_budget_trigger  = VramBudget::BufferTrigger(budget);
			m_budget_critical = VramBudget::BufferCritical(budget);
		}
		trigger  = m_budget_trigger;
		critical = m_budget_critical;
	}
	auto& gc_stats = g_buffer_gc_stats;
	gc_stats.passes++;
	PrintBufferGcStats();
	if (m_total_used_memory < trigger) {
		gc_stats.below++;
		return;
	}
	if (m_pressure_frames != 0) {
		// KYTY_VRAM_PRESSURE_BUFFER_FRAMES: once per frame, buffers unused for that many
		// frames (a quarter, at least two, above the critical mark) without GPU-dirty bytes, instead
		// of the submission-age collection below (no downloads).
		if (frame != m_pressure_frame) {
			m_pressure_frame = frame;
			RetireUnusedBuffers(frame, m_total_used_memory >= critical
			                               ? std::max<uint64_t>(m_pressure_frames / 4, 2)
			                               : m_pressure_frames);
		}
		return;
	}
	const bool     aggressive = m_total_used_memory >= critical;
	// KYTY_GC_SIDE_SETTLE_LAZY (VramBudget::PlanBufferPass): below the critical mark, a pass that
	// would wait for the GPU to settle the pending side readbacks is put off.
	// (State kept here, not in the class: one BufferCache, this runs on the GPU thread only.)
	static uint64_t gc_settle_deferred = 0;
	const bool      lazy               = g_gc_side_settle_lazy.On();
	bool            settle_waits       = false;
	const uint64_t  age                = std::min<uint64_t>(aggressive ? 80 : 160, tick);
	gc_stats.critical += aggressive ? 1u : 0u;
	if (g_gc_settle_skip_empty.On()) {
		// KYTY_GC_SETTLE_SKIP_EMPTY: the loop below visits nothing.
		bool candidate = false;
		m_lru_cache.ForEachItemBelow(tick - age, [&](BufferId) {
			candidate = true;
			return true;
		});
		if (!candidate) {
			gc_settle_deferred = 0; // as after a pass that ran and retired nothing
			gc_stats.empty++;
			return;
		}
	}
	const bool selective     = g_gc_settle_selective.On();
	const auto lazy_critical = aggressive && !selective
	                               ? static_cast<uint64_t>(g_gc_side_settle_lazy_critical.Get())
	                               : uint64_t {0};
	if (!selective && ((lazy && !aggressive) || lazy_critical != 0) && m_side != nullptr &&
	    m_side->pending_count.load(std::memory_order_acquire) != 0) {
		// CompleteAllSideReadbacks does not wait when every pending readback is published or is
		// an eager copy whose recording (master tick `value`) has completed.
		auto& master = m_scheduler.GetMasterSemaphore();
		master.Refresh();
		std::lock_guard lock(m_side->pending_mutex);
		for (const auto& entry: m_side->pending) {
			if (!entry->done.load(std::memory_order_acquire) &&
			    (!entry->eager || !master.IsFree(entry->value))) {
				settle_waits = true;
				break;
			}
		}
	}
	const auto pass =
	    lazy_critical != 0
	        ? VramBudget::PlanBufferPassLazyCritical(settle_waits, gc_settle_deferred,
	                                                 lazy_critical)
	        : VramBudget::PlanBufferPass(lazy, aggressive, settle_waits, gc_settle_deferred);
	gc_settle_deferred = pass.deferred;
	if (!pass.run) {
		g_gc_settle_deferred_passes.fetch_add(1, std::memory_order_relaxed);
		gc_stats.deferred++;
		return;
	}
	// Pending side readbacks keep tracker pages GPU-owned without exact dirty bytes; settle
	// them so the ownership checks and downloads below see a consistent state.
	gc_stats.ran++;
	if (selective) {
		// KYTY_GC_SETTLE_SELECTIVE: no settle; the loop skips candidates under pending readbacks.
	} else if (CpGaps::Enabled() && m_side != nullptr &&
	           m_side->pending_count.load(std::memory_order_acquire) != 0) {
		const auto settle_start = std::chrono::steady_clock::now();
		CompleteAllSideReadbacks();
		const auto settle_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
		                           std::chrono::steady_clock::now() - settle_start)
		                           .count();
		if (settle_ns >= 50'000) {
			gc_stats.waited++;
			gc_stats.wait_ns += static_cast<uint64_t>(settle_ns);
		}
	} else {
		CompleteAllSideReadbacks();
	}

	const size_t   limit      = static_cast<size_t>(pass.limit);

	std::vector<BufferId> dirty_buffers;
	size_t                retire_count = 0;
	size_t                scanned      = 0;
	m_lru_cache.ForEachItemBelow(tick - age, [&](BufferId id) {
		auto& buffer = m_slot_buffers[id];
		EXIT_IF(buffer.is_deleted);
		if (selective) {
			const auto first = Common::AlignDown(buffer.CpuAddress(), TRACKER_PAGE_SIZE);
			const auto last =
			    Common::AlignUp(buffer.CpuAddress() + buffer.Size(), TRACKER_PAGE_SIZE);
			if (OverlapsPendingSideReadback(first, last)) {
				gc_stats.busy++;
				return ++scanned >= 4096;
			}
			++scanned;
		}
		m_memory_tracker.ValidateGpuDirtyOwnership(m_gpu_modified_ranges, buffer.CpuAddress(),
		                                           buffer.Size(), "garbage collection");
		const bool dirty = m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size());
		if (dirty && !aggressive) {
			return false;
		}
		if (dirty) {
			EXIT_IF(!DownloadBufferMemory(buffer, buffer.CpuAddress(), buffer.Size()));
			dirty_buffers.push_back(id);
			gc_stats.downloads++;
		} else {
			m_memory_tracker.UntrackMemory(buffer.CpuAddress(), buffer.Size());
			FlightRec::Rec(FlightRec::EvGcRetire, buffer.CpuAddress(), buffer.Size(),
			               (static_cast<uint64_t>(id.index) << 32u) | id.generation);
			DeleteBuffer(id);
			Common::DebugCounters::Add(Common::DebugCounters::Counter::BufferEvictions);
		}
		gc_stats.retired++;
		return ++retire_count == limit;
	});
	if (dirty_buffers.empty()) {
		return;
	}

	// Publish all queued downloads before releasing their tracked pages and owners.
	const auto completion_tick = m_scheduler.CurrentTick();
	m_scheduler.Wait(completion_tick);
	m_scheduler.WaitPriorityOperations(completion_tick);
	for (const auto id: dirty_buffers) {
		auto& buffer = m_slot_buffers[id];
		m_memory_tracker.UnmarkRegionAsGpuModified(buffer.CpuAddress(), buffer.Size());
		if (m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size()) ||
		    m_gpu_modified_ranges.Intersects(buffer.CpuAddress(), buffer.Size())) {
			EXIT("BufferCache: garbage collection retained GPU ownership\n");
		}
		m_memory_tracker.UntrackMemory(buffer.CpuAddress(), buffer.Size());
		Unregister(id);
		m_slot_buffers.erase(id);
	}
}

void BufferCache::ReportVram() {
	using VramStats::ToMiB;
	struct Bucket {
		uint64_t bytes = 0;
		uint64_t count = 0;
		void     Add(uint64_t value) {
            bytes += value;
            ++count;
		}
	};
	struct Entry {
		uint64_t bytes   = 0;
		uint64_t address = 0;
		uint64_t age     = 0;
		bool     dirty   = false;
	};
	// GC ticks (completed submissions) per presented frame since the last report, to express the
	// LRU ages in frames.
	static uint64_t last_tick  = 0;
	static uint64_t last_frame = 0;
	const auto      frame      = static_cast<uint64_t>(m_memory_tracker.Frame());
	const auto      tick       = m_gc_tick;
	const double    per_frame  = frame > last_frame ? static_cast<double>(tick - last_tick) /
	                                                   static_cast<double>(frame - last_frame)
	                                                : 0.0;
	last_tick  = tick;
	last_frame = frame;
	static constexpr std::array<uint64_t, 4>    age_limits {1, 10, 60, 600};
	static constexpr std::array<const char*, 5> age_names {"<=1", "2-10", "11-60", "61-600", ">600"};
	std::array<Bucket, age_names.size()> by_age {};
	Bucket             all;
	Bucket             dirty;
	uint64_t           lowest  = UINT64_MAX;
	uint64_t           highest = 0;
	std::vector<Entry> entries;
	entries.reserve(m_buffers.size());
	for (const auto& [address, id]: m_buffers) {
		const auto& buffer    = m_slot_buffers[id];
		const auto  age_ticks = tick - std::min<uint64_t>(tick, m_lru_cache.TickOf(buffer.lru_id));
		const auto  age       = per_frame > 0.0
		                            ? static_cast<uint64_t>(static_cast<double>(age_ticks) / per_frame)
		                            : age_ticks;
		const bool  gpu_dirty = m_gpu_modified_ranges.Intersects(buffer.CpuAddress(), buffer.Size());
		size_t      bucket    = 0;
		while (bucket < age_limits.size() && age > age_limits[bucket]) {
			++bucket;
		}
		by_age[bucket].Add(buffer.Size());
		all.Add(buffer.Size());
		if (gpu_dirty) {
			dirty.Add(buffer.Size());
		}
		lowest  = std::min(lowest, address);
		highest = std::max(highest, address + buffer.Size());
		entries.push_back({buffer.Size(), address, age, gpu_dirty});
	}
	std::string ages;
	for (size_t index = 0; index < age_names.size(); ++index) {
		char part[64];
		std::snprintf(part, sizeof(part), " %s:%.1f(%llu)", age_names[index], ToMiB(by_age[index].bytes),
		              static_cast<unsigned long long>(by_age[index].count));
		ages += part;
	}
	VramStats::Line("guest buffers: %.1f MiB (%llu), with GPU-dirty bytes %.1f MiB (%llu), guest span "
	                "0x%" PRIx64 "..0x%" PRIx64 " | frames since use (%.1f GC ticks/frame):%s",
	                ToMiB(all.bytes), static_cast<unsigned long long>(all.count), ToMiB(dirty.bytes),
	                static_cast<unsigned long long>(dirty.count), lowest == UINT64_MAX ? 0 : lowest,
	                highest, per_frame, ages.c_str());
	const size_t top = std::min<size_t>(entries.size(), 10);
	std::partial_sort(entries.begin(), entries.begin() + static_cast<std::ptrdiff_t>(top),
	                  entries.end(), [](const Entry& a, const Entry& b) { return a.bytes > b.bytes; });
	for (size_t index = 0; index < top; ++index) {
		VramStats::Line("  guest top %.1f MiB guest=0x%" PRIx64 " unused=%llu frames%s",
		                ToMiB(entries[index].bytes), entries[index].address,
		                static_cast<unsigned long long>(entries[index].age),
		                entries[index].dirty ? " gpu-dirty" : "");
	}
	auto buffers = VramStats::BufferSnapshot();
	std::erase_if(buffers, [](const VramStats::BufferEntry& entry) { return entry.cpu_address != 0; });
	std::sort(buffers.begin(), buffers.end(),
	          [](const auto& a, const auto& b) { return a.bytes > b.bytes; });
	static constexpr std::array<const char*, 4> usage_names {"device-local", "upload", "download",
	                                                         "stream"};
	std::string line;
	for (size_t index = 0; index < std::min<size_t>(buffers.size(), 14); ++index) {
		char part[96];
		std::snprintf(part, sizeof(part), " %.1f:%s/%s%s", ToMiB(buffers[index].bytes),
		              buffers[index].usage < usage_names.size() ? usage_names[buffers[index].usage] : "?",
		              buffers[index].device_local ? "vram" : "sysmem",
		              buffers[index].host_cached ? "/cached" : "");
		line += part;
	}
	VramStats::Line("non-guest buffers (%zu), largest MiB:%s", buffers.size(), line.c_str());
	if (m_bda_sparse != nullptr) {
		VramStats::Line("BDA page table: sparse, %llu of %zu blocks bound (%.1f of %.1f MiB) in %zu chunks",
		                static_cast<unsigned long long>(m_bda_sparse->bound_blocks),
		                m_bda_sparse->bound.size(),
		                ToMiB(m_bda_sparse->bound_blocks * m_bda_sparse->block_size),
		                ToMiB(BDA_PAGETABLE_SIZE), m_bda_sparse->chunks.size());
	} else {
		VramStats::Line(
		    "BDA page table: dense, %.1f MiB (KYTY_BDA_PAGETABLE_MB; pages above it %llu)",
		    ToMiB(m_bda_pagetable_buffer.Size()),
		    static_cast<unsigned long long>(g_bda_pages_uncovered.load(std::memory_order_relaxed)));
	}
	VramStats::Line("buffer GC: usage %.1f MiB, trigger %.1f critical %.1f MiB, GC tick %llu | idle "
	                "retirement (KYTY_VRAM_IDLE_BUFFER_FRAMES=%llu) deleted %.1f MiB (%llu) in total",
	                ToMiB(m_total_used_memory), ToMiB(m_trigger_gc_memory), ToMiB(m_critical_gc_memory),
	                static_cast<unsigned long long>(m_gc_tick),
	                static_cast<unsigned long long>(m_idle_frames), ToMiB(m_idle_freed_bytes),
	                static_cast<unsigned long long>(m_idle_freed));
}

void BufferCache::ProcessFaultBuffer() {
	m_fault_manager.ProcessFaultBuffer();
}

void BufferCache::InvalidateBdaSynchronization() noexcept {
	// Also without incremental synchronization: KYTY_BDA_SYNC_EPOCH skips a pass only while the
	// registered buffers and GPU mappings are the ones its last pass scanned.
	auto epoch = m_bda_structure_epoch.load(std::memory_order_relaxed);
	while (epoch != UINT64_MAX &&
	       !m_bda_structure_epoch.compare_exchange_weak(epoch, epoch + 1,
	                                                     std::memory_order_release,
	                                                     std::memory_order_relaxed)) {}
}

void BufferCache::SynchronizeBdaBuffers(const RangeSet& mapped_ranges) {
	// KYTY_BDA_SYNC_EPOCH (syncEpoch.h): once per synchronization epoch. After a completed pass,
	// memory the pass left clean can only need an upload again within the same epoch through a
	// guest CPU write, which races the draws that follow (the first pass of the next epoch uploads
	// it), or a change of the registered buffers or GPU mappings, which moves the BDA structure
	// epoch. Everything the command processor orders before later draws (packets writing memory,
	// waits, cache invalidations, service commands) advances the epoch first.
	const auto sync_epoch = SyncEpoch::Current();
	const auto submission = SyncEpoch::CurrentSubmission();
	const auto structure  = m_bda_structure_epoch.load(std::memory_order_acquire);
	if (m_bda_epoch_skip && sync_epoch == m_bda_synced_epoch &&
	    structure == m_bda_synced_structure) {
		BufferLookupStats::AddPart(BufferLookupStats::Part::BdaSkip, 0);
		Profiler::CountFrameEvent(Profiler::FrameEvent::BdaSyncEpochSkips);
		m_bda_epoch_totals.skips++;
		if (m_bda_epoch_verify != 0) {
			VerifyBdaEpochSkip(mapped_ranges);
		}
		return;
	}
	// KYTY_BDA_SYNC_PER_SUBMISSION=1 (default off, live; upstream KytyPS5 309ba4f5, Senaxx
	// a329b69a8): at most one pass per guest submission, instead of one per epoch. What the game
	// wrote before submitting reaches every BDA read of the submission, as on the console. A CPU
	// write made while the submission runs reaches BDA reads only in the next submission, even
	// behind a fence that orders it (a WAIT_REG_MEM on a CPU-written label): the epoch pass would
	// have uploaded it. New buffers and GPU mapping changes still run the pass (structure epoch): a
	// new buffer's pages start CPU-dirty and have never been uploaded. KYTY_BDA_SYNC_PER_SUBMISSION=
	// verify runs those passes and counts what they upload (the bytes 1 would defer).
	const auto per_submission  = g_bda_sync_per_submission.Get();
	const bool same_submission = per_submission != 0 && submission == m_bda_synced_submission &&
	                             structure == m_bda_synced_structure;
	if (same_submission && per_submission == 1) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::BdaSyncSubmissionSkips);
		m_bda_epoch_totals.submission_skips++;
		ReportBdaSubmissionGate();
		return;
	}
	const auto fault_epoch     = m_memory_tracker.FaultMutationEpoch();
	const auto uploaded_before = m_bda_pass_upload_bytes;
	m_bda_count_uploads        = same_submission;
	const auto pass_start      = BufferLookupStats::Now();
	SynchronizeBdaBuffersNow(mapped_ranges);
	if (pass_start != 0) {
		BufferLookupStats::AddPart(static_cast<BufferLookupStats::Part>(m_bda_pass_kind),
		                           BufferLookupStats::Now() - pass_start);
	}
	m_bda_count_uploads = false;
	m_bda_epoch_totals.passes++;
	if (same_submission) {
		m_bda_epoch_totals.submission_verify_passes++;
		m_bda_epoch_totals.submission_deferred_bytes += m_bda_pass_upload_bytes - uploaded_before;
	}
	if (per_submission != 0) {
		ReportBdaSubmissionGate();
	}
	m_bda_synced_epoch      = sync_epoch;
	m_bda_synced_submission = submission;
	m_bda_synced_structure  = structure;
	m_bda_synced_fault      = fault_epoch;
}

void BufferCache::ReportBdaSubmissionGate() {
	// Every 10 s while the gate is on: passes, skips and (verify) the bytes a skip would defer.
	if ((++m_bda_gate_report_calls & 1023u) != 0) {
		return;
	}
	const auto now = std::chrono::steady_clock::now();
	if (m_bda_gate_report_time == std::chrono::steady_clock::time_point {}) {
		m_bda_gate_report_time = now;
		m_bda_gate_reported    = m_bda_epoch_totals;
		return;
	}
	const auto elapsed = std::chrono::duration<double>(now - m_bda_gate_report_time).count();
	if (elapsed < 10.0) {
		return;
	}
	const auto& totals = m_bda_epoch_totals;
	const auto& last   = m_bda_gate_reported;
	std::printf("BdaSyncPerSubmission %.0fs (%s): %" PRIu64 " passes, %" PRIu64
	            " skipped by the submission gate, %" PRIu64 " epoch skips; verify %" PRIu64
	            " gated passes run, %" PRIu64 " bytes they uploaded\n",
	            elapsed, g_bda_sync_per_submission.Get() == 2 ? "verify" : "on",
	            totals.passes - last.passes, totals.submission_skips - last.submission_skips,
	            totals.skips - last.skips,
	            totals.submission_verify_passes - last.submission_verify_passes,
	            totals.submission_deferred_bytes - last.submission_deferred_bytes);
	std::fflush(stdout);
	m_bda_gate_report_time = now;
	m_bda_gate_reported    = totals;
}

void BufferCache::VerifyBdaEpochSkip(const RangeSet& mapped_ranges) {
	// The full scan the skip replaced. It uploads whatever it finds, so this mode stays correct.
	// A normal page it finds CPU-dirty while the fault epoch is still the one taken before the
	// last pass is a page that pass should have uploaded (no page turned CPU-dirty since): a
	// mismatch. Hot pages, written without faults, are compared with their shadows and uploaded
	// when changed: guest writes racing the draws, as the skip assumes.
	BdaSyncStats verify;
	verify.verify_fault_epoch     = m_bda_synced_fault;
	verify.verify_structure_epoch = m_bda_synced_structure;
	{
		const UploadBatch upload_batch(*this);
		mapped_ranges.ForEach([this, &verify](uint64_t start, uint64_t end) {
			SynchronizeBuffersInRange(start, end - start, &verify);
		});
	}
	Profiler::CountFrameEvent(Profiler::FrameEvent::BdaSyncEpochVerifyChecks);
	m_bda_epoch_totals.verify_checks++;
	if (verify.verify_mismatch_pages == 0) {
		return;
	}
	Profiler::CountFrameEvent(Profiler::FrameEvent::BdaSyncEpochVerifyMismatches,
	                          verify.verify_mismatch_pages);
	m_bda_epoch_totals.verify_mismatch_pages += verify.verify_mismatch_pages;
	static std::atomic<uint32_t> logged {0};
	if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
		std::fprintf(stderr,
		             "BdaSyncEpochVerify: a skipped pass would have missed %" PRIu64
		             " CPU-dirty page(s) that turned dirty before the last pass\n",
		             verify.verify_mismatch_pages);
	}
	if (m_bda_epoch_verify == 2) {
		EXIT("BdaSyncEpochVerify: a skipped BDA pass missed CPU-dirty pages\n");
	}
}

void BufferCache::SynchronizeBdaBuffersNow(const RangeSet& mapped_ranges) {
	const bool collect = Profiler::AggregateEnabled();
	// Read these before scanning: a fault to an already scanned page must force the NEXT
	// pass, even if its dirty transition completed before this pass finished uploading.
	// CpuMutationEpoch() is saturated while any hot page exists (hot pages are written without
	// faults), which used to make every pass a full scan. KYTY_BDA_HOT_SYNC uses the fault epoch
	// instead and, while it and the structure epoch hold, re-examines exactly the pages the fault
	// epoch does not cover: the hot page runs the last full pass found in the scanned buffers.
	// Every other page of those buffers was uploaded (clean and write-protected) by that pass and
	// can only become CPU-dirty again through a transition that changes one of the epochs,
	// including the write fault that promotes a page to hot and a hot page's demotion.
	// KYTY_BDA_DIRTY_LOG: the fault epoch comes with the ranges logged up to it, taken together.
	uint64_t fault_epoch  = 0;
	bool     log_complete = false;
	if (m_bda_dirty_log) {
		log_complete = m_memory_tracker.TakeDirtiedRanges(m_bda_dirtied, fault_epoch);
		if (!log_complete) {
			m_bda_log_totals.overflows++;
			Profiler::CountFrameEvent(Profiler::FrameEvent::BdaSyncLogOverflows);
		}
	}
	const auto cpu_epoch = !m_bda_incremental_sync ? 0
	                       : m_bda_dirty_log       ? fault_epoch
	                       : m_bda_hot_sync        ? m_memory_tracker.FaultMutationEpoch()
	                                               : m_memory_tracker.CpuMutationEpoch();
	const auto structure_epoch =
	    m_bda_incremental_sync ? m_bda_structure_epoch.load(std::memory_order_acquire) : 0;
	const bool structure_holds = m_bda_incremental_sync && cpu_epoch != UINT64_MAX &&
	                             structure_epoch != UINT64_MAX &&
	                             structure_epoch == m_bda_scanned_structure_epoch;
	using LookupPart = BufferLookupStats::Part;
	m_bda_pass_kind  = static_cast<uint8_t>(LookupPart::BdaFull);
	if (structure_holds && cpu_epoch == m_bda_scanned_cpu_epoch) {
		m_bda_pass_kind = static_cast<uint8_t>(LookupPart::BdaNone);
		if (m_bda_hot_ranges.empty()) {
			if (collect) {
				Profiler::CountFrameEvent(Profiler::FrameEvent::BdaSyncSkips);
			}
			FaultCost::NoteBdaPass(0, 0);
			return;
		}
		BdaSyncStats stats;
		m_bda_pass_kind = static_cast<uint8_t>(LookupPart::BdaHot);
		if (SynchronizeBdaHotRanges(stats)) {
			FaultCost::NoteBdaPass(1, stats.upload_bytes);
			m_bda_pass_upload_bytes += stats.upload_bytes;
			if (collect) {
				Profiler::CountFrameEvent(Profiler::FrameEvent::BdaSyncHotPasses);
				Profiler::CountFrameEvent(Profiler::FrameEvent::BdaSyncHotRanges,
				                          m_bda_hot_ranges.size());
				Profiler::CountFrameEvent(Profiler::FrameEvent::BdaSyncUploadBytes, stats.upload_bytes);
				Profiler::CountFrameEvent(Profiler::FrameEvent::BdaSyncUploadCopies,
				                          stats.upload_copies);
			}
			if (BdaHotSyncVerifyEnabled()) {
				VerifyBdaHotPass(mapped_ranges, cpu_epoch, structure_epoch);
			}
			return;
		}
		// A recorded buffer is gone although the structure epoch held: scan everything.
		m_bda_pass_kind = static_cast<uint8_t>(LookupPart::BdaFull);
	} else if (structure_holds && m_bda_dirty_log && log_complete && m_bda_log_baseline &&
	           (m_bda_pass_kind = static_cast<uint8_t>(LookupPart::BdaLog),
	            SynchronizeBdaDirtied(mapped_ranges))) {
		FaultCost::NoteBdaPass(2, m_bda_last_pass_bytes);
		m_bda_scanned_cpu_epoch = cpu_epoch;
		if (m_bda_log_verify != 0) {
			m_bda_log_totals.verify_checks++;
			Profiler::CountFrameEvent(Profiler::FrameEvent::BdaSyncLogVerifyChecks);
			const auto missed = VerifyBdaFullScan(mapped_ranges, cpu_epoch, structure_epoch);
			if (missed != 0) {
				m_bda_log_totals.verify_mismatch_pages += missed;
				Profiler::CountFrameEvent(Profiler::FrameEvent::BdaSyncLogVerifyMismatches, missed);
				static std::atomic<uint32_t> logged {0};
				if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
					std::fprintf(stderr,
					             "BdaDirtyLogVerify: a dirty-log pass missed %" PRIu64
					             " CPU-dirty page(s) (fault epoch %" PRIu64 ")\n",
					             missed, cpu_epoch);
				}
				if (m_bda_log_verify == 2) {
					EXIT("BdaDirtyLogVerify: a dirty-log pass missed CPU-dirty pages\n");
				}
			}
		}
		return;
	}

	BdaSyncStats stats;
	m_bda_pass_kind = static_cast<uint8_t>(
	    m_bda_incremental_sync && structure_epoch != m_bda_scanned_structure_epoch
	        ? LookupPart::BdaFullNew
	        : LookupPart::BdaFull);
	if (m_bda_hot_sync) {
		m_bda_hot_ranges.clear();
		stats.hot_ranges = &m_bda_hot_ranges;
	}
	const bool keep_stats = collect || m_bda_hot_sync || m_bda_count_uploads;
	{
		// Only uploads are recorded while scanning: all of them share one barrier pair.
		const UploadBatch upload_batch(*this);
		mapped_ranges.ForEach([this, keep_stats, &stats](uint64_t start, uint64_t end) {
			SynchronizeBuffersInRange(start, end - start, keep_stats ? &stats : nullptr);
		});
	}
	m_bda_pass_upload_bytes += stats.upload_bytes;
	if (m_bda_incremental_sync) {
		m_bda_scanned_cpu_epoch       = cpu_epoch;
		m_bda_scanned_structure_epoch = structure_epoch;
	}
	// The log taken above holds nothing this scan did not cover; later transitions log again.
	m_bda_log_baseline = m_bda_dirty_log;
	FaultCost::NoteBdaPass(3, stats.upload_bytes);
	if (collect) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::BdaSyncPasses);
		Profiler::CountFrameEvent(Profiler::FrameEvent::BdaSyncScannedBuffers, stats.scanned_buffers);
		Profiler::CountFrameEvent(Profiler::FrameEvent::BdaSyncUploadBytes, stats.upload_bytes);
		Profiler::CountFrameEvent(Profiler::FrameEvent::BdaSyncUploadCopies, stats.upload_copies);
	}
}

bool BufferCache::SynchronizeBdaDirtied(const RangeSet& mapped_ranges) {
	// A hot run grows at most a few entries per pass; past this, rebuild the list in a full scan.
	constexpr size_t MaxHotRanges = 4096;
	if (m_bda_hot_ranges.size() > MaxHotRanges) {
		return false;
	}
	for (const auto& range: m_bda_hot_ranges) {
		const auto* buffer = m_slot_buffers.try_get(range.id);
		if (buffer == nullptr || buffer->is_deleted || !buffer->IsInBounds(range.address, range.size)) {
			return false;
		}
	}
	// The logged ranges report the hot runs they turn up (new hot pages, which a write fault
	// logged when it promoted them); the recorded runs are re-examined as in a hot pass.
	std::vector<BdaHotRange> found;
	BdaSyncStats             stats;
	stats.hot_ranges = &found;
	uint64_t logged  = 0;
	{
		const UploadBatch upload_batch(*this);
		m_bda_dirtied.ForEach([&](uint64_t begin, uint64_t end) {
			logged++;
			mapped_ranges.ForEachInRange(begin, end - begin, [&](uint64_t start, uint64_t finish) {
				SynchronizeBuffersInRange(start, finish - start, &stats);
			});
		});
		BdaSyncStats hot_stats;
		for (const auto& range: m_bda_hot_ranges) {
			(void)SynchronizeBuffer(m_slot_buffers[range.id], range.address, range.size, false,
			                        false, &hot_stats);
		}
		stats.upload_bytes += hot_stats.upload_bytes;
		stats.upload_copies += hot_stats.upload_copies;
	}
	m_bda_pass_upload_bytes += stats.upload_bytes;
	for (const auto& range: found) {
		const bool known =
		    std::any_of(m_bda_hot_ranges.begin(), m_bda_hot_ranges.end(), [&](const auto& other) {
			    return other.id == range.id && other.address == range.address &&
			           other.size == range.size;
		    });
		if (!known) {
			m_bda_hot_ranges.push_back(range);
		}
	}
	m_bda_log_totals.passes++;
	m_bda_log_totals.ranges += logged;
	m_bda_last_pass_bytes = stats.upload_bytes;
	if (Profiler::AggregateEnabled()) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::BdaSyncLogPasses);
		Profiler::CountFrameEvent(Profiler::FrameEvent::BdaSyncLogRanges, logged);
		Profiler::CountFrameEvent(Profiler::FrameEvent::BdaSyncScannedBuffers, stats.scanned_buffers);
		Profiler::CountFrameEvent(Profiler::FrameEvent::BdaSyncUploadBytes, stats.upload_bytes);
		Profiler::CountFrameEvent(Profiler::FrameEvent::BdaSyncUploadCopies, stats.upload_copies);
	}
	return true;
}

uint64_t BufferCache::VerifyBdaFullScan(const RangeSet& mapped_ranges, uint64_t fault_epoch,
                                        uint64_t structure_epoch) {
	// It uploads whatever it finds, so a verify mode stays correct when it reports a mismatch.
	BdaSyncStats verify;
	verify.verify_fault_epoch     = fault_epoch;
	verify.verify_structure_epoch = structure_epoch;
	{
		const UploadBatch upload_batch(*this);
		mapped_ranges.ForEach([this, &verify](uint64_t start, uint64_t end) {
			SynchronizeBuffersInRange(start, end - start, &verify);
		});
	}
	return verify.verify_mismatch_pages;
}

void BufferCache::VerifyBdaHotPass(const RangeSet& mapped_ranges, uint64_t fault_epoch,
                                   uint64_t structure_epoch) {
	// The full scan the hot pass replaced.
	const auto missed = VerifyBdaFullScan(mapped_ranges, fault_epoch, structure_epoch);
	Profiler::CountFrameEvent(Profiler::FrameEvent::BdaSyncHotVerifyChecks);
	if (missed == 0) {
		return;
	}
	Profiler::CountFrameEvent(Profiler::FrameEvent::BdaSyncHotVerifyMismatches, missed);
	static std::atomic<uint32_t> logged {0};
	if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
		std::fprintf(stderr,
		             "BdaHotSyncVerify: a hot pass would have missed %" PRIu64
		             " CPU-dirty page(s) (fault epoch %" PRIu64 ", structure epoch %" PRIu64 ")\n",
		             missed, fault_epoch, structure_epoch);
	}
	if (BdaHotSyncVerifyMode() == 2) {
		EXIT("BdaHotSyncVerify: a hot pass missed CPU-dirty pages\n");
	}
}

bool BufferCache::SynchronizeBdaHotRanges(BdaSyncStats& stats) {
	for (const auto& range: m_bda_hot_ranges) {
		const auto* buffer = m_slot_buffers.try_get(range.id);
		if (buffer == nullptr || buffer->is_deleted || !buffer->IsInBounds(range.address, range.size)) {
			return false;
		}
	}
	// A hot run that is no longer hot is synchronized like any range: a demoted page (still
	// CPU-dirty) is uploaded and protected, a settled or re-owned one is left alone.
	const UploadBatch upload_batch(*this);
	for (const auto& range: m_bda_hot_ranges) {
		(void)SynchronizeBuffer(m_slot_buffers[range.id], range.address, range.size, false, false,
		                        &stats);
	}
	return true;
}

void BufferCache::SynchronizeBuffersInRange(uint64_t vaddr, uint64_t size, BdaSyncStats* stats) {
	const auto end = vaddr + size;
	auto       it  = m_buffers.upper_bound(vaddr);
	if (it != m_buffers.begin()) {
		--it;
	}
	for (; it != m_buffers.end() && it->first < end; ++it) {
		auto&      buffer = m_slot_buffers[it->second];
		const auto start  = std::max(buffer.CpuAddress(), vaddr);
		const auto finish = std::min(buffer.CpuAddress() + buffer.Size(), end);
		if (start < finish) {
			if (stats != nullptr) {
				++stats->scanned_buffers;
				stats->buffer_id = it->second;
			}
			(void)SynchronizeBuffer(buffer, start, finish - start, false, false, stats);
		}
	}
}

namespace {

enum class BdaWritesMode { Off, On, Verify };

BdaWritesMode GetBdaWritesMode() {
	static const BdaWritesMode mode = [] {
		const auto* value = std::getenv("KYTY_BDA_WRITES");
		// On by default, like CodegenOptions::bda_writes (KYTY_BDA_WRITES=0 turns both off).
		if (value == nullptr || value[0] == '\0') {
			return BdaWritesMode::On;
		}
		if (std::strcmp(value, "0") == 0) {
			return BdaWritesMode::Off;
		}
		return std::strcmp(value, "verify") == 0 ? BdaWritesMode::Verify : BdaWritesMode::On;
	}();
	return mode;
}

// One console line per shader for each kind of BDA-write anomaly.
bool FirstBdaWriteNote(uint64_t shader_hash, uint32_t kind) {
	static std::mutex                   mutex;
	static std::unordered_set<uint64_t> noted;
	std::scoped_lock                    lock(mutex);
	return noted.insert(shader_hash ^ (static_cast<uint64_t>(kind) << 62u)).second;
}

} // namespace

bool BdaWritesEnabled() {
	return GetBdaWritesMode() != BdaWritesMode::Off;
}

bool BdaWritesVerify() {
	return GetBdaWritesMode() == BdaWritesMode::Verify;
}

void BufferCache::PrepareBdaWrites() {
	EXIT_IF(!BdaWritesEnabled());
	m_fault_manager.PrepareBdaWrites();
	// The dispatch may overwrite any known fill.
	ForgetKnownFills(0, uint64_t {1} << 40u);
	// A GPU-modified image over a cache buffer holds bytes its buffer lacks. A BDA write into that
	// buffer and the settle's image invalidation would drop them: move them into the buffer first
	// when image writebacks are on, exactly as a writable binding over the range does (only pages
	// with a cache buffer can receive a BDA write).
	std::vector<std::pair<BufferId, GuestRange>> overlaps;
	{
		std::scoped_lock lock {m_texture_cache.m_lock};
		m_texture_cache.m_slot_images.ForEach([&](ImageId, const Image& image) {
			if (!image.IsGpuModified()) {
				return;
			}
			const auto begin = image.info.data.address;
			const auto end   = image.info.data.End();
			auto       it    = m_buffers.upper_bound(begin);
			if (it != m_buffers.begin()) {
				--it;
			}
			for (; it != m_buffers.end() && it->first < end; ++it) {
				const auto& buffer = m_slot_buffers[it->second];
				const auto  start  = std::max(buffer.CpuAddress(), begin);
				const auto  finish = std::min(buffer.CpuAddress() + buffer.Size(), end);
				if (start < finish) {
					overlaps.push_back({it->second, GuestRange {start, finish - start}});
				}
			}
		});
	}
	if (overlaps.empty()) {
		return;
	}
	Profiler::CountFrameEvent(Profiler::FrameEvent::BdaAliasedImages, overlaps.size());
	if (!ImageWritebackOnGpuWriteEnabled()) {
		// As for a writable binding with writebacks off: the image contents are not moved.
		return;
	}
	for (const auto& [id, range]: overlaps) {
		if (!IsBufferInvalid(id) && m_slot_buffers[id].IsInBounds(range.address, range.size)) {
			PreserveImagesForGpuWrite(id, range.address, range.size);
		}
	}
}

// KYTY_BDA_STORE_LOG_STATS=1: "BdaSettle 10s: ..." console lines while BDA-writing dispatches
// settle (none printed: none settled). GPU thread.
static void NoteBdaSettleStats(size_t logged, uint32_t dropped) {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_BDA_STORE_LOG_STATS");
		return value != nullptr && std::strcmp(value, "1") == 0;
	}();
	if (!enabled) {
		return;
	}
	static uint64_t settles = 0;
	static uint64_t stores  = 0;
	static uint64_t drops   = 0;
	static uint64_t total   = 0;
	static auto     start   = std::chrono::steady_clock::now();
	settles++;
	total++;
	stores += logged;
	drops += dropped;
	const auto now = std::chrono::steady_clock::now();
	const auto ms  = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
	if (ms < 10000) {
		return;
	}
	std::printf("BdaSettle 10s: %" PRIu64 " settles, %" PRIu64 " logged stores, %" PRIu64
	            " dropped writes in %.1f s; total %" PRIu64 " settles\n",
	            settles, stores, drops, static_cast<double>(ms) / 1000.0, total);
	std::fflush(stdout);
	settles = stores = drops = 0;
	start = now;
}

void BufferCache::SettleBdaWrites(uint64_t shader_hash) {
	EXIT_IF(!BdaWritesEnabled());
	Profiler::ScopedFrameWait wait(Profiler::FrameWait::BdaSettle);
	// Records the compaction after the dispatch, submits and waits: the pages it wrote.
	auto writes = m_fault_manager.CollectBdaWrites();
	Profiler::CountFrameEvent(Profiler::FrameEvent::BdaSettles);
	if (!writes.stores.empty()) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::BdaLoggedStores, writes.stores.size());
		writes.dropped += ApplyBdaStoreLog(writes.stores, shader_hash);
		if (FirstBdaWriteNote(shader_hash, 5)) {
			Log::WriteToConsoleAndLog(fmt::format(
			    "KYTY_BDA_STORE_LOG: CS shader 0x{:016x} stored {} dword(s) to pages without a cache "
			    "buffer; the settle wrote them into guest memory.\n",
			    shader_hash, writes.stores.size()));
		}
	}
	NoteBdaSettleStats(writes.stores.size(), writes.dropped);
	if (writes.dropped != 0) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::BdaDroppedWrites, writes.dropped);
		if (FirstBdaWriteNote(shader_hash, 0)) {
			Log::WriteToConsoleAndLog(fmt::format(
			    "KYTY_BDA_WRITES: CS shader 0x{:016x} dropped {} write(s) to pages without a cache "
			    "buffer (their faults create one for later dispatches).\n",
			    shader_hash, writes.dropped));
		}
		if (BdaWritesVerify()) {
			EXIT("KYTY_BDA_WRITES=verify: shader 0x%016" PRIx64 " dropped %u BDA write(s)\n",
			     shader_hash, writes.dropped);
		}
	}
	RangeSet written;
	if (writes.overflow) {
		// More pages than the list holds (the rest of the bits are gone): every page that could
		// have received a write, i.e. every cache buffer.
		for (const auto& [vaddr, id]: m_buffers) {
			written.Add(vaddr, m_slot_buffers[id].Size());
		}
		if (FirstBdaWriteNote(shader_hash, 1)) {
			Log::WriteToConsoleAndLog(fmt::format(
			    "KYTY_BDA_WRITES: CS shader 0x{:016x} wrote more pages than one settle lists; every "
			    "cache buffer is taken into GPU ownership.\n",
			    shader_hash));
		}
	} else {
		for (const auto page: writes.pages) {
			written.Add(page, CACHING_PAGESIZE);
		}
	}
	uint64_t settled_pages = 0;
	written.ForEach([&](uint64_t start, uint64_t end) {
		SettleBdaWrittenRange(start, end - start, shader_hash, settled_pages);
	});
	Profiler::CountFrameEvent(Profiler::FrameEvent::BdaSettlePages, settled_pages);
}

void BufferCache::SettleBdaWrittenRange(uint64_t vaddr, uint64_t size, uint64_t shader_hash,
                                        uint64_t& settled_pages) {
	// Collect first: settling a part may create or merge buffers (not here, but keep the loop
	// independent of the map).
	std::vector<std::pair<BufferId, GuestRange>> parts;
	const auto                                   end = vaddr + size;
	auto                                         it  = m_buffers.upper_bound(vaddr);
	if (it != m_buffers.begin()) {
		--it;
	}
	for (; it != m_buffers.end() && it->first < end; ++it) {
		const auto& buffer = m_slot_buffers[it->second];
		const auto  start  = std::max(buffer.CpuAddress(), vaddr);
		const auto  finish = std::min(buffer.CpuAddress() + buffer.Size(), end);
		if (start < finish) {
			parts.push_back({it->second, GuestRange {start, finish - start}});
		}
	}
	for (const auto& [id, range]: parts) {
		const auto start = range.address;
		const auto bytes = range.size;
		settled_pages += (bytes + CACHING_PAGESIZE - 1) / CACHING_PAGESIZE;
		// An image that owned bytes here missed their move into the buffer before the write
		// (writebacks off, or a GPU-modified image this dispatch made): its bytes are lost now.
		if (m_texture_cache.IsRegionGpuModified(start, bytes)) {
			Profiler::CountFrameEvent(Profiler::FrameEvent::BdaAliasHits);
			if (FirstBdaWriteNote(shader_hash, 2)) {
				Log::WriteToConsoleAndLog(fmt::format(
				    "KYTY_BDA_WRITES: CS shader 0x{:016x} wrote 0x{:x}+0x{:x}, under a GPU-modified "
				    "image.\n",
				    shader_hash, start, bytes));
			}
			if (BdaWritesVerify()) {
				EXIT("KYTY_BDA_WRITES=verify: shader 0x%016" PRIx64 " wrote 0x%016" PRIx64
				     "+0x%" PRIx64 " under a GPU-modified image\n",
				     shader_hash, start, bytes);
			}
		}
		// PrepareBda uploaded every CPU-dirty page before the dispatch, and CommitBindings then
		// returned every hot page (CPU-dirty by construction) to clean tracking, or to CPU-dirty
		// when its shadow no longer matched (has_address_writes: InvalidateContentRevisions). A
		// written page that is CPU-dirty now was written by the guest while the dispatch was being
		// recorded or ran: the upload below then puts the guest's page over the dispatch's bytes
		// (hardware would keep both writers' bytes).
		if (m_memory_tracker.IsRegionCpuModified(start, bytes)) {
			uint64_t dirty_pages = 0;
			for (uint64_t page = Common::AlignDown(start, CACHING_PAGESIZE); page < start + bytes;
			     page += CACHING_PAGESIZE) {
				const auto first = std::max(page, start);
				const auto last  = std::min(page + CACHING_PAGESIZE, start + bytes);
				dirty_pages += m_memory_tracker.IsRegionCpuModified(first, last - first) ? 1u : 0u;
			}
			Profiler::CountFrameEvent(Profiler::FrameEvent::BdaSettleCpuDirtyPages, dirty_pages);
			if (FirstBdaWriteNote(shader_hash, 3)) {
				Log::WriteToConsoleAndLog(fmt::format(
				    "KYTY_BDA_WRITES: CS shader 0x{:016x} wrote 0x{:x}+0x{:x}, which the guest also "
				    "wrote during the dispatch ({} page(s)).\n",
				    shader_hash, start, bytes, dirty_pages));
			}
			if (BdaWritesVerify()) {
				EXIT("KYTY_BDA_WRITES=verify: shader 0x%016" PRIx64 " wrote 0x%016" PRIx64
				     "+0x%" PRIx64 ", which the guest wrote during the dispatch\n",
				     shader_hash, start, bytes);
			}
		}
		// What a writable binding over the range records (ObtainBuffer), after the fact: the
		// dispatch has completed, so the pages become GPU-owned before anything else is recorded
		// or read.
		auto& buffer = m_slot_buffers[id];
		(void)SynchronizeBuffer(buffer, start, bytes, true, false, nullptr, "bda-write");
		buffer.MarkContentWritten();
		if (!m_gpu_modified_ranges.Contains(start, bytes)) {
			CleanVerdict::Invalidate(start, bytes, Coherence::Source::BufferDirtyAdd);
		}
		m_gpu_modified_ranges.Add(start, bytes);
		{
			const TilerTrace::Reason canon_reason("bda");
			NoteBufferContentWrite(start, bytes);
		}
		ForgetKnownFills(start, bytes);
		HangTrace::NoteGpuWrite(start, bytes);
		// Overlapping images are rebuilt from the buffer (the Water agent's rule for BDA writes).
		{ // KYTY_TILER_TRACE: who marks images buffer-modified
			const TilerTrace::Reason bwrite_reason("bda");
			m_texture_cache.InvalidateMemoryFromGPU(start, bytes);
		}
	}
}

uint32_t BufferCache::ApplyBdaStoreLog(std::span<const FaultManager::BdaStore> stores,
                                       uint64_t shader_hash) {
	// The guest-write transition once per dword run, before any byte lands.
	RangeSet dwords;
	for (const auto& store: stores) {
		dwords.Add(store.Address() & ~uint64_t {3}, sizeof(uint32_t));
	}
	RangeSet refused;
	dwords.ForEach([&](uint64_t start, uint64_t end) {
		const auto bytes = end - start;
		if (HasGpuDirtyBytes(start, bytes) || HasPendingBackingPublication(start, bytes) ||
		    m_texture_cache.IsRegionGpuModified(start, bytes)) {
			refused.Add(start, bytes);
			return;
		}
		InvalidateMemory(start, bytes);
		m_texture_cache.InvalidateMemory(start, bytes);
	});
	uint32_t dropped = 0;
	for (const auto& store: stores) {
		const auto address = store.Address() & ~uint64_t {3};
		uint32_t   word    = 0;
		if (refused.Intersects(address, sizeof(word)) ||
		    !Libs::LibKernel::Memory::TryReadBacking(address, &word, sizeof(word))) {
			dropped++;
			continue;
		}
		word = (word & ~store.mask) | (store.value & store.mask);
		if (!Libs::LibKernel::Memory::TryWriteBacking(address, &word, sizeof(word))) {
			dropped++;
			continue;
		}
		HangTrace::NoteGpuWrite(address, sizeof(word));
	}
	// A memory write the command processor orders before later work (syncEpoch.h).
	SyncEpoch::Advance();
	if (!refused.Empty() && FirstBdaWriteNote(shader_hash, 4)) {
		Log::WriteToConsoleAndLog(fmt::format(
		    "KYTY_BDA_STORE_LOG: CS shader 0x{:016x} stored to bytes without a cache buffer that the "
		    "GPU still owns; those stores are dropped.\n",
		    shader_hash));
	}
	return dropped;
}

} // namespace Libs::Graphics
