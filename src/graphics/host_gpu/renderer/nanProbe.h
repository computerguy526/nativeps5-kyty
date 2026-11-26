#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_NANPROBE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_NANPROBE_H_

// nps5 nanite (SAROS PPSA07631): one-shot image probes around a compute dispatch / pixel-shader
// draw and a log of texture-cache events over one guest range. Diagnostic only, every switch
// default off.
//
//   KYTY_CS_PROBE=<hex hash>        at the first dispatch of that CS after the time trigger, copy
//   each
//                                   bound image (level 0, layer 0) to the host before the dispatch
//                                   ("pre") and each written image after it ("post"); raw dumps +
//                                   stats lines "[nanprobe] ...", files nanprobe_*.raw/.bmp in
//                                   KYTY_DUMP_DIR.
//   KYTY_DRAW_PROBE=<hex hash>      a draw with a stage of that hash (e.g. the Nanite raster PS)
//   after
//                                   the trigger arms a probe of its storage images, taken at the
//                                   next direct dispatch (outside the render pass) as "after-draw".
//   KYTY_PROBE_AFTER_S=<s>          time trigger, seconds after the first direct dispatch (default
//   0). KYTY_PROBE_IDS=<hex>[,<hex>...] after the trigger, log the host image identity (slot id,
//   VkImage,
//                                   state) of every image bound by a dispatch or draw with one of
//                                   these stage hashes (first KYTY_PROBE_LINES lines).
//   KYTY_PROBE_WATCH=<hex addr>[:<hex size>]  after the trigger, log inserts, frees (with reason),
//                                   uploads and ownership losses of cache images overlapping the
//                                   range (default size 1 MiB).
//   KYTY_PROBE_LINES=<n>            line budget for the IDS and WATCH logs (default 400).

#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace Libs::Graphics::NanProbe {

struct Config {
	uint64_t              cs_hash = 0;
	std::vector<uint64_t> cs_hashes; // saroslight: KYTY_CS_PROBE=<hex>[,<hex>...]
	uint64_t              draw_hash = 0;
	double                after_s   = 0;
	std::vector<uint64_t> id_hashes;
	uint64_t              watch_addr = 0;
	uint64_t              watch_size = 0;
	int64_t               lines      = 400;
	bool                  any        = false;
};

inline const Config& Get() {
	static const Config config = [] {
		Config     c;
		const auto hex = [](const char* name) -> uint64_t {
			const char* v = std::getenv(name);
			return v != nullptr && *v != '\0' ? std::strtoull(v, nullptr, 16) : 0;
		};
		c.cs_hash = hex("KYTY_CS_PROBE");
		for (const char* v = std::getenv("KYTY_CS_PROBE"); v != nullptr && *v != '\0';) {
			char*          end = nullptr;
			const uint64_t h   = std::strtoull(v, &end, 16);
			if (end == v) break;
			c.cs_hashes.push_back(h);
			v = (*end == ',') ? end + 1 : nullptr;
		}
		c.draw_hash = hex("KYTY_DRAW_PROBE");
		if (const char* v = std::getenv("KYTY_PROBE_AFTER_S"); v != nullptr) {
			c.after_s = std::strtod(v, nullptr);
		}
		for (const char* v = std::getenv("KYTY_PROBE_IDS"); v != nullptr && *v != '\0';) {
			char*          end = nullptr;
			const uint64_t h   = std::strtoull(v, &end, 16);
			if (end == v) break;
			c.id_hashes.push_back(h);
			v = (*end == ',') ? end + 1 : nullptr;
		}
		if (const char* v = std::getenv("KYTY_PROBE_WATCH"); v != nullptr && *v != '\0') {
			char* end    = nullptr;
			c.watch_addr = std::strtoull(v, &end, 16);
			c.watch_size =
			    (end != nullptr && *end == ':') ? std::strtoull(end + 1, nullptr, 16) : 0x100000;
		}
		if (const char* v = std::getenv("KYTY_PROBE_LINES"); v != nullptr) {
			c.lines = std::strtoll(v, nullptr, 10);
		}
		c.any =
		    !c.cs_hashes.empty() || c.draw_hash != 0 || !c.id_hashes.empty() || c.watch_size != 0;
		if (c.any) {
			std::fprintf(stderr,
			             "[nanprobe] cs 0x%016" PRIx64 " draw 0x%016" PRIx64
			             " after %.1f s ids %zu watch 0x%" PRIx64 "+0x%" PRIx64 "\n",
			             c.cs_hash, c.draw_hash, c.after_s, c.id_hashes.size(), c.watch_addr,
			             c.watch_size);
		}
		return c;
	}();
	return config;
}

inline std::atomic<bool>& ArmedFlag() {
	static std::atomic<bool> armed {false};
	return armed;
}

// Called at every direct dispatch while a probe is configured: starts the clock at the first one.
inline bool TickArm() {
	const auto& c = Get();
	if (!c.any) return false;
	if (ArmedFlag().load(std::memory_order_relaxed)) return true;
	static const auto start = std::chrono::steady_clock::now();
	if (std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() >=
	    c.after_s) {
		if (!ArmedFlag().exchange(true)) {
			std::fprintf(stderr, "[nanprobe] armed (%.1f s after the first dispatch)\n", c.after_s);
		}
		return true;
	}
	return false;
}

inline bool Armed() {
	return Get().any && ArmedFlag().load(std::memory_order_relaxed);
}

inline bool TakeLine() {
	static std::atomic<int64_t> used {0};
	return used.fetch_add(1, std::memory_order_relaxed) < Get().lines;
}

inline bool IdHash(uint64_t hash) {
	for (const auto h: Get().id_hashes) {
		if (h == hash) return true;
	}
	return false;
}

inline bool Watched(uint64_t address, uint64_t size) {
	const auto& c = Get();
	return c.watch_size != 0 && Armed() && address < c.watch_addr + c.watch_size &&
	       c.watch_addr < address + size;
}

// The hash of the last direct dispatch that started preparing (watch-log context only).
inline std::atomic<uint64_t>& LastCs() {
	static std::atomic<uint64_t> last {0};
	return last;
}

inline void WatchLog(const char* event, const char* detail, uint64_t address, uint64_t size,
                     uint32_t index, uint32_t generation, const char* format, uint32_t width,
                     uint32_t height, int gpu_modified) {
	if (!Watched(address, size) || !TakeLine()) return;
	std::fprintf(stderr,
	             "[nanprobe] watch %s%s%s slot %u:%u guest 0x%010" PRIx64 "+0x%" PRIx64
	             " %ux%u %s gpu-mod %d last-cs "
	             "0x%016" PRIx64 "\n",
	             event, detail != nullptr ? " " : "", detail != nullptr ? detail : "", index,
	             generation, address, size, width, height, format, gpu_modified,
	             LastCs().load(std::memory_order_relaxed));
}

// Draw -> next dispatch hand-off (GPU thread only).
struct DrawPending {
	bool                  pending = false;
	bool                  done    = false;
	uint64_t              hash    = 0;
	std::vector<uint32_t> index; // ImageId index
	std::vector<uint32_t> gen;   // ImageId generation
	std::vector<uint64_t> addr;
	std::vector<uint32_t> tex_index; // sarosclay: the draw's sampled textures
	std::vector<uint32_t> tex_gen;
};
inline DrawPending& Draw() {
	static DrawPending d;
	return d;
}

} // namespace Libs::Graphics::NanProbe

#endif
