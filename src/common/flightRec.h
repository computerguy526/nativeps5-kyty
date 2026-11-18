#ifndef EMULATOR_SRC_COMMON_FLIGHTREC_H_
#define EMULATOR_SRC_COMMON_FLIGHTREC_H_

// KYTY_FLIGHTREC (nps5 flightrec; port of NXNative's DM_NVN_FLIGHTREC, !ns_go
// sw2_tools/pr/out/flightrec-2026-11-07.md): an in-memory flight recorder. Every thread that
// records owns a ring of fixed 48-byte binary records (timestamp, guest flip, event id, 4 u64
// arguments); one writer per ring, no locks, no formatting, no I/O on the hot path. Signal
// handlers (page-protection faults) record into one shared ring (atomic slot claim). The rings are
// written to <dir>/flightrec_<pid>_<n>_<why>.bin only when something asks for it:
//   process exit (KYTY_FLIGHTREC_EXIT, default 1 when KYTY_FLIGHTREC is set explicitly),
//   SIGSEGV/SIGBUS that Kyty's fault handler leaves unresolved, SIGABRT (async-signal-safe
//   writer), SIGINFO (`kill -INFO <pid>`; SIGUSR1 is Kyty's guest-signal transport), SIGUSR2
//   (the driver's GPU watchdog; the previous handler still runs), the self memory limit exit
//   (code 86), KYTY_FLIGHTREC_TRIGGER=<flip>[,<flip>..] / p<present>, the control file
//   KYTY_FLIGHTREC_CTL (its existence is polled once per flip; it is removed after the dump) and
//   FlightRec::Trigger() from code.
// Decoder: work/tools/flightrec/decode.py (text / TSV, --flips, --ev filters).
//
//   KYTY_FLIGHTREC=0|1|2        0 off; unset or 1 = LIGHT (always-on release mode: low-rate events + counters,
//                               16384 records/thread, <= 16 rings, ~13 MB total, dumps only on triggers);
//                               2 = VERBOSE (+ every draw/dispatch/upload/bind event, 65536 records/thread, exit dump)
//   KYTY_FLIGHTREC_N=<records>  per-thread ring size (power of 2; light default 16384 = 768 KiB)
//   KYTY_FLIGHTREC_DIR=<dir>    output folder (default ./flightrec)
//   KYTY_FLIGHTREC_TAG=<tag>    file name <dir>/<tag>.<n>.<why>.flightrec (default flightrec_<pid>)
//   KYTY_FLIGHTREC_WINDOW_S=<s> a dump holds the last <s> seconds per ring (light 12, verbose 0 = everything)
//   KYTY_FLIGHTREC_MAXDUMPS=<n> default 12
// Dump triggers (light mode too): GPU watchdog (SIGUSR2 from kyty-safe / the driver), any frame > 2 s, present
// gap > 2 s and > 10 s (monitor thread), main-thread heartbeat gap > 2 s (with the main thread's backtrace),
// self memory limit, SIGTERM, SIGABRT, unresolved SEGV/BUS, process exit.
// A dump holds addresses, ids, sizes and timings only (no game bytes, no paths: files are recorded as hashes).
//   KYTY_FLIGHTREC_KEYS=<hash prefix>[,..]  draws/dispatches of these shaders record their full
//                               inputs (graphics/host_gpu/renderer/flightRecDraw.h)

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace FlightRec {

// Event ids (decode.py EVENTS mirrors them; append only).
enum Ev : uint16_t {
	EvNone = 0,
	EvFlip,         // a0 flip index, a1 video out index, a2 buffer index, a3 flip arg
	EvPresent,      // a0 present index
	EvDraw,         // a0 vs hash, a1 ps hash, a2 (draw index in flip << 32) | (keyed << 31) | colours, a3 scheduler tick
	EvDispatch,     // a0 cs hash, a1 0, a2 dispatch index in flip << 32 | keyed << 31, a3 tick
	EvBind,         // keyed: a0 guest addr, a1 size | stage << 56 | slot << 48, a2 buffer id (index << 32 | gen) | stream << 31 | written << 30, a3 FNV hash of the guest bytes (first KYTY_FLIGHTREC_HASH_KB)
	EvBindState,    // keyed, before the lookup: a0 addr, a1 size, a2 flags (BindFlag), a3 tracker range signature
	EvTarget,       // keyed: a0 guest addr, a1 slot << 48 | image index << 16 | image gen, a2 format << 32 | w << 16 | h, a3 mip << 16 | layer
	EvTexture,      // keyed: a0 guest addr, a1 stage << 56 | storage << 55 | binding << 48 | image index << 16 | gen, a2 format << 32 | w << 16 | h, a3 level << 16 | layer
	EvUpload,       // a0 buffer cpu addr, a1 first copy's guest addr, a2 bytes, a3 copies | hot << 16 | late << 24 | batch << 32 | keyed << 33 | dma << 34
	EvSyncSkip,     // keyed: a0 addr, a1 size, a2 reason (SkipWhy)
	EvDownload,     // a0 addr, a1 size, a2 buffer cpu addr
	EvSideReadback, // a0 addr, a1 size, a2 tick
	EvSideDone,     // a0 begin, a1 end, a2 eager
	EvEagerReadback,// a0 page, a1 tick, a2 result
	EvGcRetire,     // a0 addr, a1 size, a2 buffer id
	EvIdleRetire,   // a0 addr, a1 size, a2 buffer id
	EvFault,        // shared ring: a0 fault addr, a1 write
	EvRehash,       // a0 addr, a1 size, a2 hash at bind, a3 hash now (only when they differ); flags = age in flips
	EvBufCreate,    // a0 addr, a1 size, a2 buffer id
	EvTrigger,      // a0 dump number
	EvFrameSum,     // a0 flip, a1 draws, a2 keyed draws, a3 rehash checks
	EvTables,       // reserved
	EvBindResult,   // keyed, after the lookup: a0 addr, a1 size, a2 buffer cpu addr (0 = stream ring), a3 offset in that buffer | read-hit << 63
	// --- always-on (light) events, flightrec4 ---
	EvSubmit,       // guest submit to the CP: a0 queue, a1 dwords, a2 constant dwords, a3 scheduler tick
	EvNativeSubmit, // driver vkQueueSubmit (GpuSafety::NoteSubmit): a0 last tick, a1 guest submit id, a2 debug op, a3 debug arg0
	EvCpWrite,      // CP packet that writes guest memory on the CPU path: a0 CpWriteKind, a1 guest addr, a2 bytes, a3 first value / src addr
	EvIndirect,     // indirect draw/dispatch args as read: a0 kind (1 dispatch, 2 draw, 3 drawIndexed, 4 count), a1 args guest addr, a2 x|y<<32 (or count), a3 z|stride<<32
	EvSnapRead,     // a read binding served as a stream snapshot copy (high rate: counted in light mode): a0 addr, a1 size, a2 reason
	EvUploadBig,    // an upload >= 256 KiB: a0 buffer cpu addr, a1 guest addr, a2 bytes, a3 flags
	EvTexUpload,    // texture upload or skip: a0 guest addr, a1 bytes, a2 decision (TexDecision), a3 format << 32 | w << 16 | h
	EvPipeCompile,  // a0 kind (1 program, 2 pipeline, 3 cache hit, 4 stall), a1 hash/id, a2 duration us, a3 extra
	EvGcPass,       // VRAM GC pass: a0 reason, a1 buffers retired, a2 bytes, a3 duration us
	EvFileOpen,     // a kernel file open > 50 ms: a0 path hash, a1 ms, a2 result, a3 flags
	EvHeartbeat,    // main-thread run-loop heartbeat sampled by the monitor: a0 gap ms (only recorded when the gap grows > 250 ms)
	EvFrame,        // per present: a0 frame ms, a1 minflt delta, a2 majflt delta, a3 present index
	EvMem,          // 1 Hz: a0 footprint MB << 32 | resident MB, a1 compressed MB << 32 | swap used MB, a2 majflt, a3 minflt
	EvMemExtra,     // 1 Hz: a0 direct allocated MB, a1 direct mapped MB, a2 flexible MB, a3 host provider value
	EvCounters,     // 1 Hz deltas: a0 draws+dispatches, a1 uploads | skips << 32, a2 faults | snapreads << 32, a3 cp writes | compiles << 32
	EvStall,        // monitor: a0 kind (1 present gap, 2 main-thread gap, 3 frame > 2 s), a1 gap ms
	EvBacktrace,    // monitor: flags = group index | kind << 5 (1 present gap, 2 main-thread gap); a0..a3 = 4 frame addresses of the main thread (pc, lr, ..; 0 ends)
	EvZcImport,     // zero-copy hook: a0 guest addr, a1 size, a2 range id, a3 flags
	EvZcUnmap,      // zero-copy hook: a0 guest addr, a1 size, a2 range id, a3 reason
	EvZcSkip,       // zero-copy hook: readback skipped / upload skipped: a0 guest addr, a1 size, a2 reason, a3 range id
	EvZcEvent,      // zero-copy hook, free form: a0 tag, a1..a3 args
	EvNote,         // free form: a0 tag, a1..a3 args
	EvCount
};

enum BindFlag : uint64_t {
	BindCpuDirty      = 1u << 0, // a page of the range is CPU-modified (an upload is owed)
	BindGpuModified   = 1u << 1, // the range intersects the GPU-modified ranges
	BindGpuOwned      = 1u << 2, // the whole range is GPU-owned
	BindSideReadback  = 1u << 3, // a side readback over the range is pending
	BindWritten       = 1u << 4, // writable binding
	BindTexel         = 1u << 5, // formatted read
	BindNoRegion      = 1u << 6, // the tracker has no region for it (signature 0)
};

enum CpWriteKind : uint64_t {
	CpWriteData = 1,   // WRITE_DATA fallback on the CPU path (no GPU-timeline write possible)
	CpWriteDataDeferred = 2,
	CpDmaFillCpu = 3,  // DMA_DATA fill by the CPU
	CpDmaCopyCpu = 4,  // DMA_DATA copy by the CPU
	CpLabelCpu = 5,    // label/EOP write on the CPU
	CpSnapshotRead = 6,
};

enum TexDecision : uint64_t { TexUpload = 1, TexSkipClean = 2, TexSkipOwned = 3, TexUploadPartial = 4 };

enum SkipWhy : uint64_t {
	SkipRelaxedClean = 1, // RelaxedNothingToUpload
	SkipWrittenOwned = 2, // KYTY_WRITTEN_SYNC_SKIP
	SkipRangeMemo    = 3, // KYTY_BUFFER_RANGE_MEMO clean hit
	SkipNothing      = 4, // the full scan collected nothing
};

struct Record {
	uint64_t t;    // mach_absolute_time
	uint32_t flip; // guest flips reserved so far
	uint16_t ev;
	uint8_t  tid;  // ring index
	uint8_t  flags;
	uint64_t a[4];
};
static_assert(sizeof(Record) == 48);

namespace Detail {
struct Ring {
	Record*               rec  = nullptr;
	uint64_t              mask = 0;
	std::atomic<uint64_t> head {0};
	uint64_t              os_tid = 0;
	uint32_t              index  = 0;
	uint32_t              shared = 0;
	char                  name[32] {};
};
inline std::atomic<int>      g_on {-1};
inline std::atomic<int>      g_level {1}; // 1 light, 2 verbose (set by Init)
// Per-event totals (relaxed, racy by design: a lost increment under contention is fine); the 1 Hz
// sampler records deltas, so the high-rate events of light mode are still visible as rates.
inline std::atomic<uint32_t> g_count[64];
// Events that fire per draw / per binding: counted only in light mode.
constexpr uint64_t Bit(Ev e) noexcept {
	return 1ull << static_cast<unsigned>(e);
}
constexpr uint64_t kHighRateMask = Bit(EvDraw) | Bit(EvDispatch) | Bit(EvBind) | Bit(EvBindState) | Bit(EvTarget) |
                                   Bit(EvTexture) | Bit(EvUpload) | Bit(EvSyncSkip) | Bit(EvBindResult) |
                                   Bit(EvSnapRead) | Bit(EvTexUpload);
static_assert(EvCount <= 64);
inline std::atomic<uint32_t> g_flip {0};
inline std::atomic<uint32_t> g_present {0};
inline thread_local Ring*    t_ring = nullptr;
bool                         Init() noexcept;
Ring*                        NewThreadRing() noexcept;
uint64_t                     Now() noexcept;
} // namespace Detail

// Async-signal-safe (no allocation, no locks, no TLS): the shared ring. No-op before Init.
void RecShared(uint16_t ev, uint64_t a0 = 0, uint64_t a1 = 0, uint64_t a2 = 0, uint64_t a3 = 0) noexcept;

[[nodiscard]] inline bool On() noexcept {
	const int v = Detail::g_on.load(std::memory_order_relaxed);
	if (v >= 0) [[likely]] {
		return v != 0;
	}
	return Detail::Init();
}

// Records on this thread's ring (no-op when off).
inline void Rec(uint16_t ev, uint64_t a0 = 0, uint64_t a1 = 0, uint64_t a2 = 0, uint64_t a3 = 0,
                uint8_t flags = 0) noexcept {
	if (!On()) [[likely]] {
		return;
	}
	auto& cnt = Detail::g_count[ev & 63u];
	cnt.store(cnt.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
	if (((Detail::kHighRateMask >> (ev & 63u)) & 1u) != 0 && Detail::g_level.load(std::memory_order_relaxed) < 2)
	    [[likely]] {
		return;
	}
	Detail::Ring* r = Detail::t_ring;
	if (r == nullptr) [[unlikely]] {
		r = Detail::NewThreadRing();
		if (r == nullptr) {
			RecShared(ev, a0, a1, a2, a3); // out of rings (thread cap): the shared ring
			return;
		}
	}
	const uint64_t h = r->head.load(std::memory_order_relaxed);
	Record&        x = r->rec[h & r->mask];
	x.t              = Detail::Now();
	x.flip           = Detail::g_flip.load(std::memory_order_relaxed);
	x.ev             = ev;
	x.tid            = static_cast<uint8_t>(r->index);
	x.flags          = flags;
	x.a[0]           = a0;
	x.a[1]           = a1;
	x.a[2]           = a2;
	x.a[3]           = a3;
	r->head.store(h + 1, std::memory_order_release);
}


// A guest flip was reserved (any thread): advances the flip counter, records EvFlip, checks the
// flip triggers and the control file.
void NoteFlip(uint64_t video_out = 0, uint64_t buffer = 0, uint64_t arg = 0) noexcept;
void NotePresent() noexcept;

// True in VERBOSE mode (KYTY_FLIGHTREC=2): per-draw / per-binding events are recorded, not only counted.
[[nodiscard]] inline bool Verbose() noexcept {
	return Detail::g_level.load(std::memory_order_relaxed) >= 2;
}

// Main-thread run-loop heartbeat (call once per window-loop step). The monitor thread dumps when it
// stops for > 2 s, with the main thread's backtrace.
void Heartbeat() noexcept;
// Called by the present path with the wall-clock frame time (ms); > 2000 triggers a dump.
void NoteFrameTime(uint32_t frame_ms) noexcept;

struct MemSample {
	uint64_t footprint = 0, resident = 0, compressed = 0, swap_used = 0; // bytes
	uint64_t majflt = 0, minflt = 0;
};
// task_info + vm.swapusage + getrusage (the same numbers as KYTY_MEMLOG's [mem] line).
void ReadMem(MemSample& out) noexcept;
// A host-provided extra 1 Hz sample (direct memory etc., MB): called from the monitor thread.
struct MemExtra {
	uint64_t direct_alloc_mb = 0, direct_mapped_mb = 0, flexible_mb = 0, value = 0;
};
void SetMemExtraProvider(void (*fn)(MemExtra&)) noexcept;
// Timebase ticks (FlightRec::Detail::Now) to nanoseconds.
[[nodiscard]] uint64_t ToNs(uint64_t ticks) noexcept;
// 64-bit hash of a C string (file paths are recorded as hashes only).
[[nodiscard]] uint64_t HashStr(const char* s) noexcept;
// Zero-copy hook API (zerocopy3): thin wrappers, always cheap, never allocate.
inline void ZcImport(uint64_t addr, uint64_t size, uint64_t range_id = 0, uint64_t flags = 0) noexcept {
	Rec(EvZcImport, addr, size, range_id, flags);
}
inline void ZcUnmap(uint64_t addr, uint64_t size, uint64_t range_id = 0, uint64_t reason = 0) noexcept {
	Rec(EvZcUnmap, addr, size, range_id, reason);
}
inline void ZcSkip(uint64_t addr, uint64_t size, uint64_t reason, uint64_t range_id = 0) noexcept {
	Rec(EvZcSkip, addr, size, reason, range_id);
}

// Dump on a helper thread (returns at once). why: short [a-z0-9_] tag.
void Trigger(const char* why) noexcept;
// Synchronous dump (sig != 0: from a signal handler; only async-signal-safe calls).
void DumpNow(const char* why, int sig) noexcept;
// The self memory limit exit (code 86): synchronous dump when on.
void BeforeMemoryLimitExit() noexcept;
// A fatal SIGSEGV/SIGBUS that no handler resolved (hostException.cpp), before the default action.
void OnUnresolvedFault(int sig) noexcept;

// FNV-1a 64 over n bytes, 8 at a time (not cryptographic; a content fingerprint).
[[nodiscard]] uint64_t Hash(const void* data, size_t n) noexcept;

} // namespace FlightRec

#endif /* EMULATOR_SRC_COMMON_FLIGHTREC_H_ */
