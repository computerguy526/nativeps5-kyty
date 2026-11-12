#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PASSTRACE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PASSTRACE_H_

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

// ASTROHYBRID_PASS_TRACE=<seconds> (diagnostic, default off): starting that many seconds after the
// first traced pass, the next 3 guest flips are traced to stdout: every draw's and dispatch's
// shader stages (shader hash), the images each stage binds (sampled T / storage S: guest address,
// Vulkan format, extent) and every draw's colour targets (RT), one "PassTrace" line per pass, and a
// "PassTrace flip" line per flip. Used to find the game's depth, motion vectors and resolve pass
// for DLSS. Run with KYTY_DRAW_RUN=0 so every draw resolves its own images.
namespace Libs::Graphics::PassTrace {

namespace Detail {
inline std::atomic<int> g_state {0}; // 0 unset, 1 waiting, 2 tracing, 3 done / off
inline std::atomic<int64_t> g_start_ns {0};
inline std::atomic<uint32_t> g_flips {0};
inline std::atomic<uint64_t> g_seq {0};
inline int64_t NowNs() {
	return std::chrono::duration_cast<std::chrono::nanoseconds>(
	           std::chrono::steady_clock::now().time_since_epoch())
	    .count();
}
} // namespace Detail

// Whether this pass is traced (GPU thread).
[[nodiscard]] inline bool Active() {
	using namespace Detail;
	auto state = g_state.load(std::memory_order_relaxed);
	if (state == 2) {
		return true;
	}
	if (state == 3) {
		return false;
	}
	if (state == 0) {
		const char* value = std::getenv("ASTROHYBRID_PASS_TRACE");
		if (value == nullptr || value[0] == '\0' || value[0] == '0') {
			g_state.store(3, std::memory_order_relaxed);
			return false;
		}
		g_start_ns.store(NowNs() + static_cast<int64_t>(std::strtoll(value, nullptr, 10)) * 1'000'000'000LL,
		                 std::memory_order_relaxed);
		g_state.store(1, std::memory_order_relaxed);
		return false;
	}
	return false;
}

// A guest flip was reserved (any thread): starts tracing when due, stops after 3 flips.
inline void NoteFlip() {
	using namespace Detail;
	auto state = g_state.load(std::memory_order_relaxed);
	if (state == 0) {
		(void)Active();
		state = g_state.load(std::memory_order_relaxed);
	}
	if (state == 1 && NowNs() >= g_start_ns.load(std::memory_order_relaxed)) {
		g_state.store(2, std::memory_order_relaxed);
		std::printf("PassTrace flip start\n");
		return;
	}
	if (state == 2) {
		const auto flips = g_flips.fetch_add(1, std::memory_order_relaxed) + 1;
		std::printf("PassTrace flip %u\n", flips);
		if (flips >= 3) {
			g_state.store(3, std::memory_order_relaxed);
			std::fflush(stdout);
		}
	}
}

[[nodiscard]] inline uint64_t NextSeq() {
	return Detail::g_seq.fetch_add(1, std::memory_order_relaxed);
}

} // namespace Libs::Graphics::PassTrace

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PASSTRACE_H_
