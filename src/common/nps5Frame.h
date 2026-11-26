// nps5 cachesave: frame-time probe shared by the present path (writer) and background savers
// (readers).
#pragma once
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <mutex>

#if defined(__APPLE__) || defined(__linux__)
#include <dlfcn.h>
#endif

namespace Nps5Frame {
// Wall-clock ns of the last Present() and the ms length of the last frame (0 until two frames were
// seen).
inline std::atomic<uint64_t> g_last_present_ns {0};
inline std::atomic<uint32_t> g_last_frame_ms {0};
// Steady ns at which the frame time was last above 100 ms (0 = never).
inline std::atomic<uint64_t> g_slow_frame_ns {0};

// transfx: periodic cache saves (Vulkan pipeline cache ~285 MB, program cache ~180 MB) wait out
// level loads. KYTY_SAVE_LOAD_QUIET_S=N (default 0 = off, behaviour unchanged): a periodic save is
// not started until N s after the last sign of a level load:
//   - a frame above 100 ms (g_slow_frame_ns),
//   - the game's VFS read counter (nps5_vfs_bytes_read, exported by the runtime) growing by at
//   least
//     KYTY_SAVE_LOAD_VFS_MBS (default 48) MB/s between two checks (the savers check about once a
//     second),
//   - a new level opened (nps5_vfs_phase).
// A save is held back at most KYTY_SAVE_LOAD_MAX_DEFER_S (default 180) s in a row, so a long stream
// of loads cannot postpone it forever. Exit, SIGTERM and window-close saves are never held back.
inline uint64_t EnvU64Or(const char* name, uint64_t def) {
	const char* v = std::getenv(name);
	if (v == nullptr || *v == '\0') return def;
	char*      end    = nullptr;
	const auto parsed = std::strtoull(v, &end, 10);
	return end != v ? parsed : def;
}

inline uint64_t SaveLoadQuietNs() {
	static const uint64_t v = EnvU64Or("KYTY_SAVE_LOAD_QUIET_S", 0) * 1'000'000'000ull;
	return v;
}

// Steady ns of the last load sign seen up to `now` (0 = none yet). Thread-safe; samples the runtime
// counters on every call.
inline uint64_t LastLoadSignNs(uint64_t now) {
	// Never destroyed: the cache savers (PipelineCacheSaver, ProgramCacheSaver) call this from
	// their own threads and can still be running while the process exits. A function-local static
	// mutex was destroyed by exit(), the next lock threw std::system_error ("mutex lock failed:
	// Invalid argument"), std::terminate aborted, and the crash handler then waited forever: the
	// process never ended after a window close / SIGTERM (v1.4 gate, laptop playtest).
	static std::mutex& m       = *new std::mutex;
	static uint64_t   last_ns = 0, last_bytes = 0, last_sign = 0;
	static int        last_phase = -1;
#if defined(__APPLE__) || defined(__linux__)
	static const auto* bytes = reinterpret_cast<const std::atomic<unsigned long long>*>(
	    dlsym(RTLD_DEFAULT, "nps5_vfs_bytes_read"));
	static const auto* phase =
	    reinterpret_cast<const std::atomic<int>*>(dlsym(RTLD_DEFAULT, "nps5_vfs_phase"));
#else
	static const std::atomic<unsigned long long>* bytes = nullptr;
	static const std::atomic<int>*                phase = nullptr;
#endif
	static const uint64_t mbs = EnvU64Or("KYTY_SAVE_LOAD_VFS_MBS", 48);
	std::scoped_lock      lock(m);
	if (bytes != nullptr) {
		const uint64_t b = bytes->load(std::memory_order_relaxed);
		if (last_ns != 0 && now > last_ns && b >= last_bytes) {
			const double rate =
			    static_cast<double>(b - last_bytes) * 1e9 / static_cast<double>(now - last_ns);
			if (rate >= static_cast<double>(mbs) * 1048576.0) last_sign = now;
		}
		last_ns    = now;
		last_bytes = b;
	}
	if (phase != nullptr) {
		const int p = phase->load(std::memory_order_relaxed);
		if (last_phase >= 0 && p != last_phase) last_sign = now;
		last_phase = p;
	}
	const uint64_t slow = g_slow_frame_ns.load(std::memory_order_relaxed);
	if (slow != 0 && slow <= now && slow > last_sign) last_sign = slow;
	return last_sign;
}

// True: hold this periodic save back (a level load is in progress or just ended). `since` is the
// caller's own state (steady ns its save was first held back, 0 = not held).
inline bool SaveHeldForLoad(uint64_t now, uint64_t& since) {
	const uint64_t quiet = SaveLoadQuietNs();
	if (quiet == 0) return false;
	static const uint64_t max_defer =
	    EnvU64Or("KYTY_SAVE_LOAD_MAX_DEFER_S", 180) * 1'000'000'000ull;
	const uint64_t sign = LastLoadSignNs(now);
	if (sign == 0 || now - sign >= quiet) {
		since = 0;
		return false;
	}
	if (since == 0) since = now;
	return now - since < max_defer;
}
} // namespace Nps5Frame
