#include "common/flightRec.h"

#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>

#if defined(__APPLE__)
#include <mach-o/dyld.h>
#include <mach/mach.h>
#include <mach/mach_time.h>
#include <sys/resource.h>
#include <sys/sysctl.h>
#endif
#include <dlfcn.h>
#include <fcntl.h>
#include <libgen.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace FlightRec {

namespace {

// Compiled in, OFF by default in the public builds (v1.1): KYTY_FLIGHTREC=1 turns the light
// recorder on, =2 the verbose one. Memory bound (light mode): ring 0 (shared) + 16 thread rings of
// 16384 x 48 B = 768 KiB = 13 MB; threads past the cap record into the shared ring. Verbose mode
// (KYTY_FLIGHTREC=2) uses 65536-record rings.
constexpr uint32_t kMaxRings    = 17;
constexpr uint32_t kMaxTriggers = 16;
constexpr uint32_t kMaxImages   = 192;
uint64_t           g_shared_n   = 1u << 14;

Detail::Ring          g_rings[kMaxRings];
std::atomic<uint32_t> g_ring_count {1}; // ring 0 = the shared ring
uint64_t              g_n         = 1u << 14;
uint64_t              g_window_ns = 12ull * 1000000000ull;
char                  g_tag[96]   = "";
char                  g_dir[512]  = "flightrec";
char                  g_ctl[512]  = "";
int                   g_max_dumps = 12;
bool                  g_exit_dump = true;
uint32_t              g_trig_flip[kMaxTriggers];
uint32_t              g_trig_flip_n = 0;
uint32_t              g_trig_present[kMaxTriggers];
uint32_t              g_trig_present_n = 0;
std::atomic<int>      g_dumps {0};
std::atomic<int>      g_busy {0};
uint32_t              g_tb_numer = 1;
uint32_t              g_tb_denom = 1;
thread_local bool     t_failed   = false;

struct sigaction g_old_abrt;
struct sigaction g_old_info;
struct sigaction g_old_usr2;
struct sigaction g_old_term;
bool             g_old_term_valid = false;
bool             g_old_abrt_valid = false;
bool             g_old_info_valid = false;
bool             g_old_usr2_valid = false;

void* MapZeroed(size_t bytes) noexcept {
	void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
	return p == MAP_FAILED ? nullptr : p;
}

uint64_t ParseU64(const char* s, uint64_t fallback) noexcept {
	if (s == nullptr || *s == '\0') {
		return fallback;
	}
	char*      end = nullptr;
	const auto v   = std::strtoull(s, &end, 0);
	return end == s ? fallback : v;
}

void ParseTriggers(const char* s) noexcept {
	while (s != nullptr && *s != '\0') {
		const bool present = *s == 'p';
		if (present) {
			s++;
		}
		char*      end = nullptr;
		const auto v   = std::strtoull(s, &end, 10);
		if (end == s) {
			break;
		}
		if (present && g_trig_present_n < kMaxTriggers) {
			g_trig_present[g_trig_present_n++] = static_cast<uint32_t>(v);
		} else if (!present && g_trig_flip_n < kMaxTriggers) {
			g_trig_flip[g_trig_flip_n++] = static_cast<uint32_t>(v);
		}
		s = end;
		while (*s == ',' || *s == ' ') {
			s++;
		}
	}
}

// --- async-signal-safe formatting/writing ---
struct Out {
	char   buf[640];
	size_t len = 0;
	void   Str(const char* s) noexcept {
		while (s != nullptr && *s != '\0' && len + 1 < sizeof(buf)) {
			buf[len++] = *s++;
		}
		buf[len] = '\0';
	}
	void U64(uint64_t v) noexcept {
		char   tmp[24];
		size_t n = 0;
		do {
			tmp[n++] = static_cast<char>('0' + v % 10);
			v /= 10;
		} while (v != 0 && n < sizeof(tmp));
		while (n > 0 && len + 1 < sizeof(buf)) {
			buf[len++] = tmp[--n];
		}
		buf[len] = '\0';
	}
};

bool WriteAll(int fd, const void* p, size_t n) noexcept {
	const auto* b = static_cast<const uint8_t*>(p);
	while (n > 0) {
		const ssize_t w = write(fd, b, n > (1u << 24) ? (1u << 24) : n);
		if (w < 0 && errno == EINTR) {
			continue;
		}
		if (w <= 0) {
			return false;
		}
		b += w;
		n -= static_cast<size_t>(w);
	}
	return true;
}

// File layout (little endian; decode.py):
//   FileHeader, then per ring: RingHeader, `count` Records (oldest first), u64 head after the copy.
struct FileHeader {
	char     magic[8]; // "KYFLREC1"
	uint32_t version;
	uint32_t record_size;
	uint32_t rings;
	int32_t  sig;
	int32_t  pid;
	uint32_t flip;
	uint32_t present;
	uint32_t tb_numer;
	uint32_t tb_denom;
	uint32_t dump;
	uint64_t t_dump;
	char     why[32];
	uint64_t window_ns; // v2: records older than t_dump - window_ns were left out (0 = all)
	uint32_t images;    // v2: image table (KYIMGS01) after the rings
	uint32_t level;     // 1 light, 2 verbose
};
struct RingHeader {
	uint32_t index;
	uint32_t shared;
	uint64_t os_tid;
	uint64_t capacity;
	uint64_t head; // records ever written (before the copy)
	uint64_t count;
	char     name[32];
};

void SetWhy(char* dst, size_t cap, const char* why) noexcept {
	size_t i = 0;
	for (; why != nullptr && why[i] != '\0' && i + 1 < cap; i++) {
		const char c = why[i];
		dst[i]       = (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_' ? c : '_';
	}
	dst[i] = '\0';
}

void ChainSignal(const struct sigaction& old, bool valid, int sig, siginfo_t* info,
                 void* ctx) noexcept {
	if (!valid) {
		return;
	}
	if ((old.sa_flags & SA_SIGINFO) != 0) {
		if (old.sa_sigaction != nullptr) {
			old.sa_sigaction(sig, info, ctx);
		}
	} else if (old.sa_handler != SIG_DFL && old.sa_handler != SIG_IGN) {
		old.sa_handler(sig);
	}
}

void OnAbort(int sig, siginfo_t* info, void* ctx) {
	const int saved = errno;
	DumpNow("abort", sig);
	if (g_old_abrt_valid) {
		ChainSignal(g_old_abrt, true, sig, info, ctx);
	}
	// Returning lets abort() restore the default action and raise again.
	errno = saved;
}

void OnInfo(int sig, siginfo_t* info, void* ctx) {
	const int saved = errno;
	DumpNow("siginfo", sig);
	ChainSignal(g_old_info, g_old_info_valid, sig, info, ctx);
	errno = saved;
}

void OnUsr2(int sig, siginfo_t* info, void* ctx) {
	const int saved = errno;
	DumpNow("watchdog", sig);
	ChainSignal(g_old_usr2, g_old_usr2_valid, sig, info, ctx);
	errno = saved;
}

void Install(int sig, void (*fn)(int, siginfo_t*, void*), struct sigaction& old,
             bool& valid) noexcept {
	struct sigaction prev {};
	if (sigaction(sig, nullptr, &prev) == 0) {
		old   = prev;
		valid = (prev.sa_flags & SA_SIGINFO) != 0
		            ? prev.sa_sigaction != nullptr
		            : (prev.sa_handler != SIG_DFL && prev.sa_handler != SIG_IGN);
	}
	struct sigaction sa {};
	sa.sa_sigaction = fn;
	sa.sa_flags     = SA_SIGINFO | SA_RESTART | SA_ONSTACK;
	sigemptyset(&sa.sa_mask);
	sigaction(sig, &sa, nullptr);
}

void OnTerm(int sig, siginfo_t* info, void* ctx) {
	const int saved = errno;
	DumpNow("sigterm", sig);
	if (g_old_term_valid) {
		ChainSignal(g_old_term, true, sig, info, ctx);
	} else {
		// Nobody else handles it: restore the default action and re-raise (the dump is written).
		struct sigaction dfl {};
		dfl.sa_handler = SIG_DFL;
		sigemptyset(&dfl.sa_mask);
		sigaction(sig, &dfl, nullptr);
		raise(sig);
	}
	errno = saved;
}

void AtExit() {
	DumpNow("exit", 0);
}

// --- monitor thread: heartbeat / present-gap detection, 1 Hz samples, main-thread backtrace ---
std::atomic<uint64_t> g_present_t {0}; // Now() of the last present
std::atomic<uint64_t> g_hb_t {0};      // Now() of the last main-loop heartbeat
std::atomic<uint32_t> g_main_port {0}; // mach port of the heartbeating thread
void (*g_mem_extra)(MemExtra&) = nullptr;
struct ImageEntry {
	uint64_t base;
	char     name[56];
};
ImageEntry            g_images[kMaxImages];
std::atomic<uint32_t> g_image_n {0};
std::atomic<int>      g_reason_count[8];

uint64_t NsToTicks(uint64_t ns) noexcept {
	return ns * g_tb_denom / g_tb_numer;
}
uint64_t TicksToNs(uint64_t t) noexcept {
	return t * g_tb_numer / g_tb_denom;
}

// Not async-signal-safe (dyld calls): monitor / dump helper threads only.
void RefreshImages() noexcept {
#if defined(__APPLE__)
	const uint32_t n = _dyld_image_count();
	uint32_t       k = 0;
	for (uint32_t i = 0; i < n && k < kMaxImages; i++) {
		const char* nm = _dyld_get_image_name(i);
		if (nm == nullptr) {
			continue;
		}
		const char* base = std::strrchr(nm, '/'); // basename only: no user paths in a dump
		base             = base != nullptr ? base + 1 : nm;
		g_images[k].base = reinterpret_cast<uint64_t>(_dyld_get_image_header(i));
		std::snprintf(g_images[k].name, sizeof(g_images[k].name), "%s", base);
		k++;
	}
	g_image_n.store(k, std::memory_order_release);
#endif
}

// Walks the frame-pointer chain of a thread (suspended for the walk) into out[]. No allocation
// while it is suspended; reads go through vm_read_overwrite so a bad frame pointer cannot fault us.
uint32_t Backtrace(uint32_t port, uint64_t* out, uint32_t cap) noexcept {
#if defined(__APPLE__) && defined(__arm64__)
	if (port == 0 || cap < 2) {
		return 0;
	}
	const mach_port_t th = static_cast<mach_port_t>(port);
	if (thread_suspend(th) != KERN_SUCCESS) {
		return 0;
	}
	uint32_t               n = 0;
	arm_thread_state64_t   st {};
	mach_msg_type_number_t cnt = ARM_THREAD_STATE64_COUNT;
	if (thread_get_state(th, ARM_THREAD_STATE64, reinterpret_cast<thread_state_t>(&st), &cnt) ==
	    KERN_SUCCESS) {
		constexpr uint64_t mask = 0x00007fffffffffffull;
		out[n++]                = reinterpret_cast<uint64_t>(arm_thread_state64_get_pc(st)) & mask;
		out[n++]                = reinterpret_cast<uint64_t>(arm_thread_state64_get_lr(st)) & mask;
		uint64_t fp             = reinterpret_cast<uint64_t>(arm_thread_state64_get_fp(st)) & mask;
		for (; n < cap && fp != 0 && (fp & 15u) == 0;) {
			uint64_t  frame[2] = {0, 0};
			vm_size_t got      = 0;
			if (vm_read_overwrite(mach_task_self(), fp, sizeof(frame),
			                      reinterpret_cast<vm_address_t>(frame), &got) != KERN_SUCCESS ||
			    got != sizeof(frame)) {
				break;
			}
			const uint64_t next = frame[0] & mask;
			out[n++]            = frame[1] & mask;
			if (next <= fp) {
				break;
			}
			fp = next;
		}
	}
	thread_resume(th);
	return n;
#else
	(void)port;
	(void)out;
	(void)cap;
	return 0;
#endif
}

void RecordBacktrace(uint32_t kind, uint32_t port) noexcept {
	uint64_t       pcs[32] = {};
	const uint32_t n       = Backtrace(port, pcs, 32);
	for (uint32_t i = 0, g = 0; i < n; i += 4, g++) {
		Rec(EvBacktrace, pcs[i], pcs[i + 1], pcs[i + 2], pcs[i + 3],
		    static_cast<uint8_t>(g | (kind << 5)));
	}
}

bool ReasonAllowed(int idx, int cap) noexcept {
	return g_reason_count[idx].fetch_add(1, std::memory_order_relaxed) < cap;
}

void MonitorSample(uint64_t& prev_maj, uint64_t& prev_min, uint32_t (&prev_cnt)[64]) noexcept {
	MemSample m;
	ReadMem(m);
	constexpr uint64_t MB = 1048576;
	Rec(EvMem, (m.footprint / MB) << 32 | (m.resident / MB),
	    (m.compressed / MB) << 32 | (m.swap_used / MB), m.majflt, m.minflt);
	if (g_mem_extra != nullptr) {
		MemExtra e;
		g_mem_extra(e);
		Rec(EvMemExtra, e.direct_alloc_mb, e.direct_mapped_mb, e.flexible_mb, e.value);
	}
	uint32_t d[64];
	for (uint32_t i = 0; i < 64; i++) {
		const uint32_t c = Detail::g_count[i].load(std::memory_order_relaxed);
		d[i]             = c - prev_cnt[i];
		prev_cnt[i]      = c;
	}
	auto u = [&](uint32_t e) -> uint64_t { return d[e]; };
	Rec(EvCounters, u(EvDraw) + u(EvDispatch), u(EvUpload) + u(EvUploadBig) | (u(EvSyncSkip) << 32),
	    u(EvFault) | ((u(EvSnapRead)) << 32), u(EvCpWrite) | (u(EvPipeCompile) << 32));
	(void)prev_maj;
	(void)prev_min;
}

void* MonitorMain(void*) {
#if defined(__APPLE__)
	pthread_setname_np("flightrec-mon");
#endif
	Rec(EvNote, 1); // creates this thread's ring
	RefreshImages();
	uint64_t prev_maj = 0, prev_min = 0;
	uint32_t prev_cnt[64]  = {};
	uint64_t last_sample   = Detail::Now();
	uint32_t armed_present = 0; // bit0: 2 s dumped, bit1: 10 s dumped
	uint32_t armed_main    = 0;
	uint64_t seen_present = 0, seen_hb = 0;
	for (;;) {
		usleep(100 * 1000);
		const uint64_t now = Detail::Now();
		const uint64_t pt  = g_present_t.load(std::memory_order_relaxed);
		if (pt != seen_present) {
			seen_present  = pt;
			armed_present = 0;
		}
		if (pt != 0 && Detail::g_present.load(std::memory_order_relaxed) >= 2) {
			const uint64_t gap_ms = TicksToNs(now > pt ? now - pt : 0) / 1000000ull; // pt may be newer than now (stored after the read): no wrap
			if (gap_ms > 2000 && (armed_present & 1u) == 0) {
				armed_present |= 1u;
				Rec(EvStall, 1, gap_ms);
				RecordBacktrace(1, g_main_port.load(std::memory_order_relaxed));
				if (ReasonAllowed(1, 3)) {
					RefreshImages();
					DumpNow("stall", 0);
				}
			} else if (gap_ms > 10000 && (armed_present & 2u) == 0) {
				armed_present |= 2u;
				Rec(EvStall, 1, gap_ms);
				RecordBacktrace(1, g_main_port.load(std::memory_order_relaxed));
				if (ReasonAllowed(2, 3)) {
					RefreshImages();
					DumpNow("stall10", 0);
				}
			}
		}
		const uint64_t hb = g_hb_t.load(std::memory_order_relaxed);
		if (hb != seen_hb) {
			seen_hb    = hb;
			armed_main = 0;
		}
		if (hb != 0) {
			const uint64_t gap_ms = TicksToNs(now > hb ? now - hb : 0) / 1000000ull; // hb may be newer than now: an unsigned wrap gave false "mainhang" dumps (earlysub sky2, gap 6148914691236 ms)
			if (gap_ms > 250 && gap_ms % 500 < 100) {
				Rec(EvHeartbeat, gap_ms);
			}
			if (gap_ms > 2000 && (armed_main & 1u) == 0) {
				armed_main |= 1u;
				Rec(EvStall, 2, gap_ms);
				RecordBacktrace(2, g_main_port.load(std::memory_order_relaxed));
				if (ReasonAllowed(3, 3)) {
					RefreshImages();
					DumpNow("mainhang", 0);
				}
			} else if (gap_ms > 10000 && (armed_main & 2u) == 0) {
				armed_main |= 2u;
				Rec(EvStall, 2, gap_ms);
				RecordBacktrace(2, g_main_port.load(std::memory_order_relaxed));
				if (ReasonAllowed(4, 3)) {
					RefreshImages();
					DumpNow("mainhang10", 0);
				}
			}
		}
		if (now - last_sample >= NsToTicks(1000000000ull)) {
			last_sample = now;
			MonitorSample(prev_maj, prev_min, prev_cnt);
		}
	}
	return nullptr;
}

void StartMonitor() noexcept {
	pthread_attr_t attr;
	pthread_attr_init(&attr);
	pthread_attr_setstacksize(&attr, 256 * 1024);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
	pthread_t th;
	pthread_create(&th, &attr, MonitorMain, nullptr);
	pthread_attr_destroy(&attr);
}

bool InitImpl() noexcept {
	const char* env = std::getenv("KYTY_FLIGHTREC");
	if (env == nullptr || (std::strcmp(env, "1") != 0 && std::strcmp(env, "2") != 0)) {
		return false;
	}
	const bool verbose = env != nullptr && std::strcmp(env, "2") == 0;
	Detail::g_level.store(verbose ? 2 : 1, std::memory_order_relaxed);
	uint64_t n = ParseU64(std::getenv("KYTY_FLIGHTREC_N"), verbose ? (1u << 16) : (1u << 14));
	n          = n < 1024 ? 1024 : (n > (1u << 24) ? (1u << 24) : n);
	uint64_t p = 1;
	while (p < n) {
		p <<= 1u;
	}
	g_n        = p;
	g_shared_n = verbose ? (1u << 16) : p;
	g_window_ns =
	    ParseU64(std::getenv("KYTY_FLIGHTREC_WINDOW_S"), verbose ? 0 : 12) * 1000000000ull;
	if (const char* d = std::getenv("KYTY_FLIGHTREC_DIR"); d != nullptr && *d != '\0') {
		std::snprintf(g_dir, sizeof(g_dir), "%s", d);
	}
	if (const char* t = std::getenv("KYTY_FLIGHTREC_TAG"); t != nullptr && *t != '\0') {
		size_t i = 0; // [A-Za-z0-9_.-] only: no path separators
		for (; t[i] != '\0' && i + 1 < sizeof(g_tag); i++) {
			const char c = t[i];
			g_tag[i] = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
			                   c == '_' || c == '-'
			               ? c
			               : '_';
		}
		g_tag[i] = '\0';
	}
	if (const char* c = std::getenv("KYTY_FLIGHTREC_CTL"); c != nullptr && *c != '\0') {
		std::snprintf(g_ctl, sizeof(g_ctl), "%s", c);
	}
	g_max_dumps = static_cast<int>(ParseU64(std::getenv("KYTY_FLIGHTREC_MAXDUMPS"), 12));
	g_exit_dump = ParseU64(std::getenv("KYTY_FLIGHTREC_EXIT"), 1) != 0;
	ParseTriggers(std::getenv("KYTY_FLIGHTREC_TRIGGER"));
#if defined(__APPLE__)
	mach_timebase_info_data_t tb {};
	if (mach_timebase_info(&tb) == KERN_SUCCESS) {
		g_tb_numer = tb.numer;
		g_tb_denom = tb.denom;
	}
#endif
	auto& shared = g_rings[0];
	shared.rec   = static_cast<Record*>(MapZeroed(g_shared_n * sizeof(Record)));
	if (shared.rec == nullptr) {
		return false;
	}
	shared.mask   = g_shared_n - 1;
	shared.shared = 1;
	std::snprintf(shared.name, sizeof(shared.name), "shared");
	Install(SIGABRT, OnAbort, g_old_abrt, g_old_abrt_valid);
#if defined(SIGINFO)
	Install(SIGINFO, OnInfo, g_old_info, g_old_info_valid);
#endif
	Install(SIGUSR2, OnUsr2, g_old_usr2, g_old_usr2_valid);
	Install(SIGTERM, OnTerm, g_old_term, g_old_term_valid);
	if (g_exit_dump) {
		std::atexit(AtExit);
	}
	Detail::g_on.store(1, std::memory_order_release); // the monitor thread records
	StartMonitor();
	std::fprintf(stderr,
	             "[flightrec] on (%s): %llu records/thread (%llu KiB), <= %u rings, dir %s, window "
	             "%llu s, exit dump %d, "
	             "%u flip / %u present triggers%s%s\n",
	             verbose ? "verbose" : "light", static_cast<unsigned long long>(g_n),
	             static_cast<unsigned long long>(g_n * sizeof(Record) / 1024), kMaxRings, g_dir,
	             static_cast<unsigned long long>(g_window_ns / 1000000000ull), g_exit_dump ? 1 : 0,
	             g_trig_flip_n, g_trig_present_n, g_ctl[0] != '\0' ? ", ctl " : "", g_ctl);
	return true;
}

} // namespace

bool Detail::Init() noexcept {
	static const bool on = InitImpl();
	g_on.store(on ? 1 : 0, std::memory_order_release);
	return on;
}

uint64_t Detail::Now() noexcept {
#if defined(__APPLE__)
	return mach_absolute_time();
#else
	timespec ts {};
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return static_cast<uint64_t>(ts.tv_sec) * 1000000000ull + static_cast<uint64_t>(ts.tv_nsec);
#endif
}

Detail::Ring* Detail::NewThreadRing() noexcept {
	if (t_failed || g_on.load(std::memory_order_acquire) != 1) {
		return nullptr;
	}
	const uint32_t index = g_ring_count.fetch_add(1, std::memory_order_acq_rel);
	if (index >= kMaxRings) {
		g_ring_count.store(kMaxRings, std::memory_order_relaxed);
		t_failed = true;
		return nullptr;
	}
	auto& r   = g_rings[index];
	auto* rec = static_cast<Record*>(MapZeroed(g_n * sizeof(Record)));
	if (rec == nullptr) {
		t_failed = true;
		return nullptr;
	}
	r.mask       = g_n - 1;
	r.index      = index;
	uint64_t tid = 0;
#if defined(__APPLE__)
	pthread_threadid_np(nullptr, &tid);
	pthread_getname_np(pthread_self(), r.name, sizeof(r.name));
#endif
	r.os_tid = tid;
	if (r.name[0] == '\0') {
		std::snprintf(r.name, sizeof(r.name), "thread%u", index);
	}
	// Published last: the dump reads rec only for rings whose rec is set.
	std::atomic_thread_fence(std::memory_order_release);
	r.rec  = rec;
	t_ring = &r;
	return &r;
}

void RecShared(uint16_t ev, uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3) noexcept {
	if (Detail::g_on.load(std::memory_order_relaxed) != 1) {
		return;
	}
	{
		auto& cnt = Detail::g_count[ev & 63u];
		cnt.store(cnt.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
	}
	auto&          r = g_rings[0];
	const uint64_t h = r.head.fetch_add(1, std::memory_order_relaxed);
	Record&        x = r.rec[h & r.mask];
	x.ev             = 0; // a reader seeing ev 0 skips the slot (torn)
	x.t              = Detail::Now();
	x.flip           = Detail::g_flip.load(std::memory_order_relaxed);
	x.tid            = 0;
	x.flags          = 0;
	x.a[0]           = a0;
	x.a[1]           = a1;
	x.a[2]           = a2;
	x.a[3]           = a3;
	std::atomic_thread_fence(std::memory_order_release);
	x.ev = ev;
}

void NoteFlip(uint64_t video_out, uint64_t buffer, uint64_t arg) noexcept {
	if (!On()) {
		return;
	}
	const uint32_t f = Detail::g_flip.fetch_add(1, std::memory_order_relaxed) + 1;
	Rec(EvFlip, f, video_out, buffer, arg);
	for (uint32_t i = 0; i < g_trig_flip_n; i++) {
		if (g_trig_flip[i] == f) {
			Trigger("flip");
		}
	}
	if (g_ctl[0] != '\0' && access(g_ctl, F_OK) == 0) {
		unlink(g_ctl);
		Trigger("ctl");
	}
}

void NotePresent() noexcept {
	if (!On()) {
		return;
	}
	const uint32_t p    = Detail::g_present.fetch_add(1, std::memory_order_relaxed) + 1;
	const uint64_t now  = Detail::Now();
	const uint64_t prev = g_present_t.exchange(now, std::memory_order_relaxed);
	Rec(EvPresent, p);
	if (prev != 0) {
		const uint64_t  ms       = TicksToNs(now - prev) / 1000000ull;
		static uint64_t last_min = 0, last_maj = 0;
#if defined(__APPLE__)
		rusage ru {};
		getrusage(RUSAGE_SELF, &ru);
		const uint64_t dmin = static_cast<uint64_t>(ru.ru_minflt) - last_min,
		               dmaj = static_cast<uint64_t>(ru.ru_majflt) - last_maj;
		last_min            = static_cast<uint64_t>(ru.ru_minflt);
		last_maj            = static_cast<uint64_t>(ru.ru_majflt);
#else
		const uint64_t dmin = 0, dmaj = 0;
#endif
		Rec(EvFrame, ms, dmin, dmaj, p);
		if (ms > 2000 && ReasonAllowed(0, 3)) {
			Trigger("slowframe");
		}
	}
	for (uint32_t i = 0; i < g_trig_present_n; i++) {
		if (g_trig_present[i] == p) {
			Trigger("present");
		}
	}
}

void Trigger(const char* why) noexcept {
	if (!On() || g_dumps.load(std::memory_order_relaxed) >= g_max_dumps) {
		return;
	}
	char tag[32];
	SetWhy(tag, sizeof(tag), why);
	std::thread([w = std::string(tag)] { DumpNow(w.c_str(), 0); }).detach();
}

void DumpNow(const char* why, int sig) noexcept {
	if (Detail::g_on.load(std::memory_order_acquire) != 1) {
		return;
	}
	if (g_busy.exchange(1, std::memory_order_acq_rel) != 0) {
		if (sig != 0) {
			return; // a crash during a dump: skipped
		}
		int i = 0;
		for (; i < 4000 && g_busy.exchange(1, std::memory_order_acq_rel) != 0; i++) {
			usleep(1000);
		}
		if (i == 4000) {
			return;
		}
	}
	const int k = g_dumps.fetch_add(1, std::memory_order_relaxed);
	if (k >= g_max_dumps) {
		g_busy.store(0, std::memory_order_release);
		return;
	}
	if (sig == 0) {
		Rec(EvTrigger, static_cast<uint64_t>(k)); // not in a signal handler: TLS / ring allocation
	}
	char tag[32];
	SetWhy(tag, sizeof(tag), why);
	mkdir(g_dir, 0755);
	Out path;
	path.Str(g_dir);
	if (g_tag[0] != '\0') {
		path.Str("/");
		path.Str(g_tag);
	} else {
		path.Str("/flightrec_");
		path.U64(static_cast<uint64_t>(getpid()));
	}
	path.Str(".");
	path.U64(static_cast<uint64_t>(k));
	path.Str(".");
	path.Str(tag);
	path.Str(".flightrec");
	const int fd    = open(path.buf, O_WRONLY | O_CREAT | O_TRUNC, 0644);
	bool      ok    = fd >= 0;
	uint32_t  rings = g_ring_count.load(std::memory_order_acquire);
	rings           = rings > kMaxRings ? kMaxRings : rings;
	if (ok) {
		FileHeader h {};
		std::memcpy(h.magic, "KYFLREC1", 8);
		h.version     = 2;
		h.record_size = sizeof(Record);
		h.rings       = rings;
		h.sig         = sig;
		h.pid         = static_cast<int32_t>(getpid());
		h.flip        = Detail::g_flip.load(std::memory_order_relaxed);
		h.present     = Detail::g_present.load(std::memory_order_relaxed);
		h.tb_numer    = g_tb_numer;
		h.tb_denom    = g_tb_denom;
		h.dump        = static_cast<uint32_t>(k);
		h.t_dump      = Detail::Now();
		std::memcpy(h.why, tag, sizeof(tag));
		h.window_ns = g_window_ns;
		h.level     = static_cast<uint32_t>(Detail::g_level.load(std::memory_order_relaxed));
		h.images    = sig == 0 ? 0 : 0;
		const uint32_t nimg   = g_image_n.load(std::memory_order_acquire);
		h.images              = nimg;
		const uint64_t cutoff = g_window_ns == 0 || h.t_dump < NsToTicks(g_window_ns)
		                            ? 0
		                            : h.t_dump - NsToTicks(g_window_ns);
		ok                    = WriteAll(fd, &h, sizeof(h));
		for (uint32_t i = 0; ok && i < rings; i++) {
			auto&      r   = g_rings[i];
			Record*    rec = r.rec;
			RingHeader rh {};
			rh.index    = i;
			rh.shared   = r.shared;
			rh.os_tid   = r.os_tid;
			rh.capacity = rec != nullptr ? r.mask + 1 : 0;
			rh.head     = r.head.load(std::memory_order_acquire);
			rh.count    = rec == nullptr ? 0 : (rh.head < rh.capacity ? rh.head : rh.capacity);
			// Window: keep only records newer than the cutoff (scan back from the newest; a torn
			// slot has ev 0).
			if (cutoff != 0 && rh.count > 0) {
				uint64_t keep = 0;
				while (keep < rh.count) {
					const Record& x = rec[(rh.head - 1 - keep) & r.mask];
					if (x.ev != 0 && x.t < cutoff) {
						break;
					}
					keep++;
				}
				rh.count = keep;
			}
			std::memcpy(rh.name, r.name, sizeof(rh.name));
			ok = WriteAll(fd, &rh, sizeof(rh));
			if (ok && rh.count > 0) {
				// Oldest first: [head - count, head).
				const uint64_t first = rh.head - rh.count;
				const uint64_t start = first & r.mask;
				const uint64_t tail =
				    rh.capacity - start < rh.count ? rh.capacity - start : rh.count;
				ok = WriteAll(fd, rec + start, tail * sizeof(Record)) &&
				     (tail == rh.count || WriteAll(fd, rec, (rh.count - tail) * sizeof(Record)));
			}
			const uint64_t after = r.head.load(std::memory_order_acquire);
			ok                   = ok && WriteAll(fd, &after, sizeof(after));
		}
		for (uint32_t i = 0; ok && i < nimg;
		     i++) { // image table: base + basename (symbolise frame addresses)
			ok = WriteAll(fd, &g_images[i], sizeof(ImageEntry));
		}
		close(fd);
	}
	Out msg;
	msg.Str(ok ? "[flightrec] wrote " : "[flightrec] FAILED ");
	msg.Str(path.buf);
	msg.Str(" (");
	msg.Str(tag);
	msg.Str(", flip ");
	msg.U64(Detail::g_flip.load(std::memory_order_relaxed));
	msg.Str(")\n");
	(void)!write(2, msg.buf, msg.len);
	g_busy.store(0, std::memory_order_release);
}

void BeforeMemoryLimitExit() noexcept {
	DumpNow("memlimit", 0);
}

void OnUnresolvedFault(int sig) noexcept {
	DumpNow(sig == SIGBUS ? "sigbus" : "sigsegv", sig);
}

uint64_t Hash(const void* data, size_t n) noexcept {
	const auto* p = static_cast<const uint8_t*>(data);
	uint64_t    h = 0xcbf29ce484222325ull;
	size_t      i = 0;
	for (; i + 8 <= n; i += 8) {
		uint64_t w;
		std::memcpy(&w, p + i, 8);
		h = (h ^ w) * 0x100000001b3ull;
		h ^= h >> 29u;
	}
	for (; i < n; i++) {
		h = (h ^ p[i]) * 0x100000001b3ull;
	}
	return h ^ n;
}

void Heartbeat() noexcept {
	if (!On()) {
		return;
	}
#if defined(__APPLE__)
	static thread_local bool port_set = false;
	if (!port_set) {
		port_set = true;
		g_main_port.store(pthread_mach_thread_np(pthread_self()), std::memory_order_relaxed);
	}
#endif
	g_hb_t.store(Detail::Now(), std::memory_order_relaxed);
}

void ReadMem(MemSample& out) noexcept {
#if defined(__APPLE__)
	task_vm_info_data_t    vi {};
	mach_msg_type_number_t cnt = TASK_VM_INFO_COUNT;
	if (task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&vi), &cnt) ==
	    KERN_SUCCESS) {
		out.footprint  = vi.phys_footprint;
		out.resident   = vi.resident_size;
		out.compressed = vi.compressed;
	}
	xsw_usage sw {};
	size_t    swl = sizeof(sw);
	if (sysctlbyname("vm.swapusage", &sw, &swl, nullptr, 0) == 0) {
		out.swap_used = sw.xsu_used;
	}
	rusage ru {};
	getrusage(RUSAGE_SELF, &ru);
	out.majflt = static_cast<uint64_t>(ru.ru_majflt);
	out.minflt = static_cast<uint64_t>(ru.ru_minflt);
#else
	out = MemSample {};
#endif
}

void SetMemExtraProvider(void (*fn)(MemExtra&)) noexcept {
	g_mem_extra = fn;
}

uint64_t ToNs(uint64_t ticks) noexcept {
	return TicksToNs(ticks);
}

uint64_t HashStr(const char* s) noexcept {
	uint64_t h = 0xcbf29ce484222325ull;
	for (; s != nullptr && *s != '\0'; s++) {
		h = (h ^ static_cast<uint8_t>(*s)) * 0x100000001b3ull;
	}
	return h;
}

} // namespace FlightRec
