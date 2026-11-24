// naniteshade (2026-11-24): uncapped per-interval counters for the Nanite shading-bin path (SAROS
// PPSA07631, UE5.5 r.Nanite.Bundle.Shading=1). Diagnostic only, no behaviour change.
// Opt-in: KYTY_SAROS_NANITE_SHADE_STATS=1 (default off; =<seconds> sets the report interval, default 5).
//
// Why: the existing evidence was capped/partial (the "[bundle] compact" line stops after 16, "[nanite] CS-ind
// bind" logs each shader once, the skip-empty log stops after 64, and RenderExecutor::DispatchIndirect returns
// silently when the pipeline is not ready). This counts, per interval: compactions (records in / kept) and
// every DISPATCH_INDIRECT by outcome, split into "bin" (args inside the args range of a recent compaction =
// a Nanite shading bin) and "other", plus the distinct bin CS hashes executed. One line per interval:
//   [naniteshade] 5.0s: compact 45 (in 2295 kept 495 = 11.0/c) | bin exec 495 empty 0 noprog 0 notready 0
//   nocs 0 (11.0/c, 37 cs) | other exec ... ; flips are not known here, "/c" = per compaction (one per frame).
// Standalone-testable: clang++ -std=c++20 -O2 -Isrc tests/NaniteShadeStatsTests.cpp -o /tmp/nst && /tmp/nst
// CMake target: nanite_shade_stats_tests.
#ifndef EMULATOR_SRC_GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_NANITESHADESTATS_H_
#define EMULATOR_SRC_GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_NANITESHADESTATS_H_

#include <array>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_set>

namespace Libs::Graphics::NaniteShadeStats {

enum class Outcome : uint32_t {
	Executed = 0, // vkCmdDispatchIndirect recorded
	SkipEmpty,    // KYTY_SKIP_EMPTY_INDIRECT: CPU-visible zero group count
	NoProgram,    // the CS did not translate
	NotReady,     // pipeline not built yet / RT hold
	NoShader,     // COMPUTE_PGM address 0
	Count
};
inline constexpr uint32_t kOutcomes = static_cast<uint32_t>(Outcome::Count);

// Args ranges of the most recent compactions (the per-bin args pool rotates between a few transient
// buffers, so keep a small ring).
inline constexpr uint32_t kRanges = 8;

struct Snapshot {
	uint64_t compactions = 0, records_in = 0, records_kept = 0;
	uint64_t bin[kOutcomes] {}, other[kOutcomes] {};
	uint64_t bin_shaders = 0;
};

struct State {
	std::atomic<uint64_t> compactions {0}, records_in {0}, records_kept {0};
	std::atomic<uint64_t> bin[kOutcomes] {}, other[kOutcomes] {};
	std::atomic<uint64_t> range_lo[kRanges] {}, range_hi[kRanges] {};
	std::atomic<uint32_t> range_next {0};
	std::mutex                   shaders_mutex;
	std::unordered_set<uint64_t> bin_shaders;
	std::atomic<int64_t>         last_report_ms {-1};

	void NoteCompaction(uint64_t args_base, uint64_t args_bytes, uint32_t n, uint32_t kept) {
		compactions.fetch_add(1, std::memory_order_relaxed);
		records_in.fetch_add(n, std::memory_order_relaxed);
		records_kept.fetch_add(kept, std::memory_order_relaxed);
		if (args_base == 0 || args_bytes == 0) {
			return;
		}
		for (uint32_t i = 0; i < kRanges; i++) {
			if (range_lo[i].load(std::memory_order_relaxed) == args_base) {
				uint64_t hi = range_hi[i].load(std::memory_order_relaxed);
				while (hi < args_base + args_bytes &&
				       !range_hi[i].compare_exchange_weak(hi, args_base + args_bytes, std::memory_order_relaxed)) {
				}
				return;
			}
		}
		const uint32_t slot = range_next.fetch_add(1, std::memory_order_relaxed) % kRanges;
		range_hi[slot].store(0, std::memory_order_relaxed);
		range_lo[slot].store(args_base, std::memory_order_relaxed);
		range_hi[slot].store(args_base + args_bytes, std::memory_order_relaxed);
	}

	[[nodiscard]] bool IsBinArgs(uint64_t args_addr) const {
		for (uint32_t i = 0; i < kRanges; i++) {
			const uint64_t lo = range_lo[i].load(std::memory_order_relaxed);
			const uint64_t hi = range_hi[i].load(std::memory_order_relaxed);
			if (lo != 0 && args_addr >= lo && args_addr < hi) {
				return true;
			}
		}
		return false;
	}

	void NoteIndirect(uint64_t args_addr, Outcome outcome, uint64_t shader_hash) {
		const auto k   = static_cast<uint32_t>(outcome);
		const bool is_bin = IsBinArgs(args_addr);
		(is_bin ? bin : other)[k].fetch_add(1, std::memory_order_relaxed);
		if (is_bin && outcome == Outcome::Executed && shader_hash != 0) {
			std::scoped_lock lock(shaders_mutex);
			bin_shaders.insert(shader_hash);
		}
	}

	// Takes the interval's counts and resets them (the args ranges stay).
	Snapshot Take() {
		Snapshot s;
		s.compactions  = compactions.exchange(0, std::memory_order_relaxed);
		s.records_in   = records_in.exchange(0, std::memory_order_relaxed);
		s.records_kept = records_kept.exchange(0, std::memory_order_relaxed);
		for (uint32_t k = 0; k < kOutcomes; k++) {
			s.bin[k]   = bin[k].exchange(0, std::memory_order_relaxed);
			s.other[k] = other[k].exchange(0, std::memory_order_relaxed);
		}
		std::scoped_lock lock(shaders_mutex);
		s.bin_shaders = bin_shaders.size();
		bin_shaders.clear();
		return s;
	}

	// True once per interval (the caller then prints Take()); elapsed_ms = the interval's real length.
	// now_ms: any monotonic clock in ms.
	bool Due(int64_t now_ms, int64_t interval_ms, int64_t* elapsed_ms) {
		int64_t last = last_report_ms.load(std::memory_order_relaxed);
		if (last < 0) {
			last_report_ms.compare_exchange_strong(last, now_ms, std::memory_order_relaxed);
			return false;
		}
		if (now_ms - last < interval_ms ||
		    !last_report_ms.compare_exchange_strong(last, now_ms, std::memory_order_relaxed)) {
			return false;
		}
		*elapsed_ms = now_ms - last;
		return true;
	}
};

[[nodiscard]] inline std::string Format(const Snapshot& s, double seconds) {
	const double per_c = s.compactions != 0 ? 1.0 / static_cast<double>(s.compactions) : 0.0;
	char         line[512];
	std::snprintf(line, sizeof(line),
	              "[naniteshade] %.1fs: compact %" PRIu64 " (in %" PRIu64 " kept %" PRIu64 " = %.1f/c) | bin exec %" PRIu64
	              " empty %" PRIu64 " noprog %" PRIu64 " notready %" PRIu64 " nocs %" PRIu64 " (%.1f/c, %" PRIu64
	              " cs) | other exec %" PRIu64 " empty %" PRIu64 " noprog %" PRIu64 " notready %" PRIu64 " nocs %" PRIu64,
	              seconds, s.compactions, s.records_in, s.records_kept, static_cast<double>(s.records_kept) * per_c,
	              s.bin[0], s.bin[1], s.bin[2], s.bin[3], s.bin[4], static_cast<double>(s.bin[0]) * per_c, s.bin_shaders,
	              s.other[0], s.other[1], s.other[2], s.other[3], s.other[4]);
	return line;
}

// Report interval in ms from KYTY_SAROS_NANITE_SHADE_STATS; 0 = off.
[[nodiscard]] inline int64_t ParseIntervalMs(const char* value) {
	if (value == nullptr || value[0] == '\0' || std::strcmp(value, "0") == 0) {
		return 0;
	}
	const double s = std::strtod(value, nullptr);
	return s > 1.5 ? static_cast<int64_t>(s * 1000.0) : 5000; // "1" = on with the default interval
}

inline int64_t IntervalMs() {
	static const int64_t ms = ParseIntervalMs(std::getenv("KYTY_SAROS_NANITE_SHADE_STATS"));
	return ms;
}
inline bool Enabled() { return IntervalMs() != 0; }

inline State& Global() {
	static State state;
	return state;
}

inline void NoteCompaction(uint64_t args_base, uint64_t args_bytes, uint32_t n, uint32_t kept) {
	if (Enabled()) [[unlikely]] {
		Global().NoteCompaction(args_base, args_bytes, n, kept);
	}
}

inline void NoteIndirect(uint64_t args_addr, Outcome outcome, uint64_t shader_hash) {
	if (!Enabled()) [[likely]] {
		return;
	}
	auto&         g      = Global();
	g.NoteIndirect(args_addr, outcome, shader_hash);
	const int64_t now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
	                           std::chrono::steady_clock::now().time_since_epoch())
	                           .count();
	int64_t elapsed_ms = 0;
	if (g.Due(now_ms, IntervalMs(), &elapsed_ms)) {
		std::fprintf(stderr, "%s\n", Format(g.Take(), static_cast<double>(elapsed_ms) / 1000.0).c_str());
	}
}

} // namespace Libs::Graphics::NaniteShadeStats

#endif
