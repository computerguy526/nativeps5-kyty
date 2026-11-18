// nps5: MetalFX in place of Astro Bot's temporal upscale pass (metalfx.h).
#include "graphics/host_gpu/renderer/metalfx.h"

#include "common/guestAddress.h"
#include "graphics/host_gpu/renderer/astroTileGrid.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/image/image.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"

#if defined(__APPLE__)
#include <cerrno>
#include <dlfcn.h>
#include <mach-o/dyld.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace Libs::Graphics {

namespace MetalFx {

namespace {

// The driver command's argument (the patched KosmicKrisp driver, src/kosmickrisp/vulkan/kk_cmd_fx.c).
struct KkMetalFxInfo {
	uint32_t size;
	uint32_t mode; // 1 temporal, 2 spatial
	VkImage  color, depth, motion, output;
	uint32_t in_w, in_h;
	float    jitter[2];
	float    mv_scale[2];
	float    mv_bias[2];
	uint32_t reset;
	uint32_t depth_reversed;
	uint32_t auto_exposure;
	uint32_t depth_mode; // 0: the depth image; 1: colour.w is linear view depth (near / w, reversed)
	float    depth_near;
	uint32_t reserved;
};
using KkCmdMetalFx = int (*)(VkCommandBuffer, const KkMetalFxInfo*);

enum class Mode { Off, Temporal, Spatial };

enum class Fallback : uint8_t { NoDriver, Bindings, Sizes, NoDepth, NoMotionVectors, Refused, NoUpscale, Count };
constexpr std::array<const char*, static_cast<size_t>(Fallback::Count)> FALLBACK_NAMES {
    "no-driver-command", "bindings", "sizes", "no-depth", "no-mv", "metalfx-refused", "not-an-upscale"};

struct DepthPair {
	ImageId color;
	ImageId depth;
};

// 2936e5eca99d75e7: the pass in 01.019 (ASTROHYBRID_DLSS).
// Pass profiles: pixel shader hash + constant layout (the shape is checked against the draw).
//   V1019 (2936e5eca99d75e7, ASTROHYBRID_DLSS): jitter UV dwords 0-3, out 4-5, in 6-7, frame 11,
//     motion used 13; images motion, colour; depth = the attachment drawn with the motion target.
//   V1007 (5a34b303576ee324): out 0-1, in 2-3, in-1 4-5, 6 == 1 "copy, no history" (camera cut),
//     frame 7, motion valid 9; images motion (.xy UV, previous = uv - mv), colour (.w = linear view
//     depth), history. No projection jitter. Its history target is read by other passes, so the
//     game's pass draws and MetalFX then overwrites target 0 ("after").
// KYTY_METALFX_PASS=<hash>[:1019|:1007] names another pass (the title profile's value).
enum class Layout : uint8_t { V1019, V1007 };
struct PassProfile {
	uint64_t hash;
	Layout   layout;
};
constexpr std::array<PassProfile, 2> KNOWN_PASSES {
    {{0x2936e5eca99d75e7ull, Layout::V1019}, {0x5a34b303576ee324ull, Layout::V1007}}};

struct Pending {
	bool     valid = false;
	bool     after = false;
	bool     temporal = false;
	Layout   layout = Layout::V1019;
	ImageId  color, motion, depth, output;
	uint32_t in_w = 0, in_h = 0, out_w = 0, out_h = 0, frame = 0;
	bool     cut = false;
	float    jitter_uv[4] {};
};

std::atomic<uint64_t> g_flips {0};

struct State {
	Mode     mode           = Mode::Off;
	bool     find           = false;
	bool     find_all       = false;
	bool     pass_auto      = false;
	uint64_t pass_hash      = 0; // 0: KNOWN_PASSES
	Layout   pass_layout    = Layout::V1007;
	bool     depth_attachment = false; // V1007: the depth attachment instead of colour.w
	float    depth_near     = 0.1f;
	float    mv_sign        = 1.0f;
	uint32_t max_in_w       = 1920;
	Pending  pending;
	std::array<ImageId, 8> recent_depths {};
	uint32_t               recent_depth_next = 0;
	uint64_t pass_draws = 0;
	bool     jitter         = false;
	float    jitter_sign_y  = 1.0f;
	uint64_t    cycle_every = 0;
	uint64_t    cycle_from  = 1;
	std::string cycle_list;
	int         cycle_index = -1;
	// A compute pass that scales the scene colour to the output size (TryMetalFxCompute); 0 = none.
	uint64_t    compute_hash = 0x4bf3b92602fbe696ull; // ASTRO BOT 01.007 levels
	uint64_t    win_draws = 0, win_scene = 0; // pass draws of the current cycle window; those with a scene-size input
	uint32_t    win_in_w = 0, win_in_h = 0;
	uint32_t jitter_index   = 0;
	float    jitter_px[2] {}; // the viewport offset of the frame being drawn
	float    jitter_sign    = 1.0f;
	float    mv_jitter      = 1.0f;
	bool     depth_reversed = false;
	std::array<DepthPair, 16> depth_pairs {};
	uint32_t                  depth_next = 0;
	bool                      lookup_tried = false;
	KkCmdMetalFx              command      = nullptr;
	uint32_t last_frame = 0;
	uint32_t in_w = 0, in_h = 0, out_w = 0, out_h = 0;
	float    last_jitter[2] {};
	uint64_t evaluations = 0;
	uint64_t resets      = 0;
	std::array<uint64_t, static_cast<size_t>(Fallback::Count)> fallbacks {};
	int64_t  print_ns = 0;
	std::array<uint64_t, 64> logged_hashes {};
	uint32_t                 logged_count = 0;
	uint32_t                 find_cb_logged = 0;
};

State& Get() {
	static State state = [] {
		State s;
		if (const char* value = std::getenv("KYTY_METALFX"); value != nullptr) {
			s.mode = (std::strcmp(value, "1") == 0 || std::strcmp(value, "temporal") == 0) ? Mode::Temporal
			         : std::strcmp(value, "spatial") == 0                                  ? Mode::Spatial
			                                                                               : Mode::Off;
		}
		if (const char* value = std::getenv("KYTY_METALFX_FIND"); value != nullptr) {
			s.find     = std::strcmp(value, "1") == 0 || std::strcmp(value, "2") == 0;
			s.find_all = std::strcmp(value, "2") == 0;
		}
		if (const char* value = std::getenv("KYTY_METALFX_PASS"); value != nullptr) {
			if (std::strcmp(value, "auto") == 0) {
				s.pass_auto = true;
			} else {
				char* end   = nullptr;
				s.pass_hash = std::strtoull(value, &end, 16);
				s.pass_layout = (end != nullptr && std::strcmp(end, ":1019") == 0) ? Layout::V1019 : Layout::V1007;
				for (const auto& known: KNOWN_PASSES) {
					if (known.hash == s.pass_hash && (end == nullptr || *end == '\0')) {
						s.pass_layout = known.layout;
					}
				}
			}
		}
		if (const char* value = std::getenv("KYTY_METALFX_JITTER_SIGN"); value != nullptr) {
			s.jitter_sign = std::strtof(value, nullptr) < 0 ? -1.0f : 1.0f;
		}
		if (const char* value = std::getenv("KYTY_METALFX_MV_JITTER"); value != nullptr) {
			const float v = std::strtof(value, nullptr);
			s.mv_jitter   = v < 0 ? -1.0f : v > 0 ? 1.0f : 0.0f;
		}
		if (const char* value = std::getenv("KYTY_METALFX_JITTER"); value != nullptr) {
			s.jitter = std::strcmp(value, "0") != 0 && s.mode == Mode::Temporal;
		}
		if (const char* value = std::getenv("KYTY_METALFX_JITTER_SIGN_Y"); value != nullptr) {
			s.jitter_sign_y = std::strtof(value, nullptr) < 0 ? -1.0f : 1.0f;
		}
		if (const char* value = std::getenv("KYTY_METALFX_CYCLE"); value != nullptr && s.mode != Mode::Off) {
			char* end     = nullptr;
			s.cycle_every = std::strtoull(value, &end, 10);
			s.cycle_list  = end != nullptr && *end == ',' ? end + 1 : "gst";
			for (const char c: s.cycle_list) {
				if (std::strchr("gstabcd", c) == nullptr) {
					s.cycle_every = 0;
				}
				s.jitter = s.jitter || (c >= 'a' && c <= 'd');
			}
		}
		if (const char* value = std::getenv("KYTY_METALFX_COMPUTE_PASS"); value != nullptr) {
			s.compute_hash = std::strtoull(value, nullptr, 16);
		}
		if (const char* value = std::getenv("KYTY_METALFX_CYCLE_FROM"); value != nullptr) {
			s.cycle_from = std::strtoull(value, nullptr, 10);
		}
		if (const char* value = std::getenv("KYTY_METALFX_DEPTH"); value != nullptr) {
			s.depth_attachment = std::strcmp(value, "attachment") == 0;
		}
		if (const char* value = std::getenv("KYTY_METALFX_DEPTH_NEAR"); value != nullptr) {
			s.depth_near = std::strtof(value, nullptr);
		}
		if (const char* value = std::getenv("KYTY_METALFX_MAX_IN_W"); value != nullptr) {
			s.max_in_w = static_cast<uint32_t>(std::atoi(value));
		}
		if (const char* value = std::getenv("KYTY_METALFX_MV_SIGN"); value != nullptr) {
			s.mv_sign = std::strtof(value, nullptr) < 0 ? -1.0f : 1.0f;
		}
		if (const char* value = std::getenv("KYTY_METALFX_DEPTH_REVERSED"); value != nullptr) {
			s.depth_reversed = std::strcmp(value, "0") != 0;
		}
		return s;
	}();
	return state;
}

int64_t NowNs() {
	return std::chrono::duration_cast<std::chrono::nanoseconds>(
	           std::chrono::steady_clock::now().time_since_epoch())
	    .count();
}

void Summary(State& s) {
	const auto now = NowNs();
	if (s.print_ns == 0) {
		s.print_ns = now;
	}
	if (now - s.print_ns < 10'000'000'000LL) {
		return;
	}
	s.print_ns = now;
	std::string text;
	for (size_t i = 0; i < s.fallbacks.size(); i++) {
		if (s.fallbacks[i] != 0) {
			text += " " + std::string(FALLBACK_NAMES[i]) + "=" + std::to_string(s.fallbacks[i]);
		}
	}
	std::printf("MetalFX 10s: %s evaluations %llu of %llu pass draws, resets %llu (%ux%u -> %ux%u, jitter %.3f %.3f px) | game picture kept:%s\n",
	            s.cycle_every != 0 ? "cycle" : s.mode == Mode::Spatial ? "spatial" : "temporal",
	            static_cast<unsigned long long>(s.evaluations),
	            static_cast<unsigned long long>(s.pass_draws),
	            static_cast<unsigned long long>(s.resets), s.in_w, s.in_h, s.out_w, s.out_h,
	            static_cast<double>(s.last_jitter[0]), static_cast<double>(s.last_jitter[1]),
	            text.empty() ? " none" : text.c_str());
	std::fflush(stdout);
	s.evaluations = 0;
	s.pass_draws  = 0;
	s.resets      = 0;
	s.fallbacks   = {};
}

bool Fail(State& s, Fallback reason) {
	s.fallbacks[static_cast<size_t>(reason)]++;
	Summary(s);
	return false;
}

// The command is an exported symbol of the loaded KosmicKrisp driver, not a Vulkan entry point.
KkCmdMetalFx Lookup() {
#if defined(__APPLE__)
	const uint32_t count = _dyld_image_count();
	for (uint32_t i = 0; i < count; i++) {
		const char* name = _dyld_get_image_name(i);
		if (name == nullptr || std::strstr(name, "libvulkan_kosmickrisp") == nullptr) {
			continue;
		}
		if (void* handle = dlopen(name, RTLD_NOLOAD | RTLD_LAZY); handle != nullptr) {
			auto* symbol = dlsym(handle, "nps5_kk_cmd_metalfx");
			std::printf("MetalFX: driver %s %s nps5_kk_cmd_metalfx\n", name,
			            symbol != nullptr ? "has" : "has NO");
			std::fflush(stdout);
			return reinterpret_cast<KkCmdMetalFx>(symbol);
		}
	}
	std::printf("MetalFX: no KosmicKrisp driver loaded; the game's own upscale pass stays\n");
	std::fflush(stdout);
#endif
	return nullptr;
}

bool LogOnce(State& s, uint64_t hash) {
	for (uint32_t i = 0; i < s.logged_count; i++) {
		if (s.logged_hashes[i] == hash) {
			return false;
		}
	}
	if (s.logged_count == s.logged_hashes.size()) {
		return false;
	}
	s.logged_hashes[s.logged_count++] = hash;
	return true;
}

} // namespace

bool Requested() {
	static const bool requested = Get().mode != Mode::Off || Get().find || FindWide();
	return requested;
}

bool JitterRequested() {
	static const bool requested = Get().jitter;
	return requested;
}

bool ViewportJitter(uint32_t width, uint32_t height, bool has_depth, uint32_t vertices, float& x, float& y) {
	const auto& s = Get();
	// (an indirect draw has no vertex count here: scene geometry)
	if (!has_depth || (vertices != 0 && vertices <= 4) || width != s.in_w || height != s.in_h || s.in_w == 0 ||
	    (s.jitter_px[0] == 0.0f && s.jitter_px[1] == 0.0f)) {
		return false;
	}
	x = s.jitter_px[0];
	y = s.jitter_px[1];
	return true;
}

// KYTY_ASTRO_TILE_GRID=<tiles x>x<tiles y>|auto (default off; ASTRO BOT 01.007 at a forced internal
// resolution below its DRS level 0): the game takes the 16-px light-tile grid of the scene
// constants (words 0x29b0 / 0x29b4 of the 17904-byte scene buffer: the row pitch forward-lit
// transparents and particles index the tile light lists with) from two per-DRS-level tables in its
// read-only data, tiles y {68, 86, 86, 117, 135} at guest 0x8be984c and tiles x {120, 152, 152,
// 208, 240} at 0x8be9860, while the light culling pass sizes its grid from the real target (80 x 45
// at 1280x720). The level-0 entries are set to the given grid (720p: 80x45, 900p: 100x57) at the
// first flips; anything but the expected tables leaves the game untouched.
void AstroTileGridPatch() {
	static int      state = 0; // 0 not tried, 1 done or refused
	static uint32_t tries = 0;
	if (state != 0) {
		return;
	}
	const char* value = std::getenv("KYTY_ASTRO_TILE_GRID");
	unsigned    tx    = 0;
	unsigned    ty    = 0;
	if (value == nullptr) {
		state = 1;
		return;
	}
	// "auto": ceil(MetalFX input size / 16) per axis (1280x720 -> 80x45, 1600x900 -> 100x57,
	// 1920x1080 -> 120x68 = the game's own level 0, nothing to change), known after the first
	// MetalFX evaluation; an invalid spec ends it.
	const uint32_t in_w = Get().in_w;
	const uint32_t in_h = Get().in_h;
	if (!AstroTileGrid::Resolve(value, in_w, in_h, tx, ty)) {
		// auto waits for the first MetalFX evaluation; an invalid spec or a grid beyond the game's
		// largest level (240x135, its 2160p entry) ends it.
		if (std::strcmp(value, "auto") != 0 || (in_w != 0 && in_h != 0)) {
			state = 1;
			std::printf("AstroTileGrid: KYTY_ASTRO_TILE_GRID=%s (input %ux%u) is not a grid of at "
			            "most %ux%u tiles, "
			            "nothing changed\n",
			            value, in_w, in_h, AstroTileGrid::MAX_TILES_X, AstroTileGrid::MAX_TILES_Y);
			std::fflush(stdout);
		}
		return;
	}
	if (tx == 120 && ty == 68) {
		state = 1;
		return;
	}
	static constexpr uint64_t kTable      = 0x8be984c;
	static constexpr uint32_t kExpect[10] = {68, 86, 86, 117, 135, 120, 152, 152, 208, 240};
	auto*                     words       = Guest::Ptr<uint32_t>(kTable);
	if (std::memcmp(words, kExpect, sizeof(kExpect)) != 0) {
		if (++tries >= 600) {
			state = 1;
			std::printf("AstroTileGrid: tables not found at 0x%llx, nothing changed\n",
			            static_cast<unsigned long long>(kTable));
			std::fflush(stdout);
		}
		return;
	}
	const auto page  = static_cast<uintptr_t>(getpagesize());
	const auto begin = reinterpret_cast<uintptr_t>(words) & ~(page - 1);
	const auto end =
	    (reinterpret_cast<uintptr_t>(words) + sizeof(kExpect) + page - 1) & ~(page - 1);
	if (mprotect(reinterpret_cast<void*>(begin), end - begin, PROT_READ | PROT_WRITE) != 0) {
		state = 1;
		std::printf("AstroTileGrid: mprotect failed (%d), nothing changed\n", errno);
		std::fflush(stdout);
		return;
	}
	words[0] = ty;
	words[5] = tx;
	mprotect(reinterpret_cast<void*>(begin), end - begin, PROT_READ);
	state = 1;
	std::printf("AstroTileGrid: level-0 light tile grid 120x68 -> %ux%u (KYTY_ASTRO_TILE_GRID)\n",
	            tx, ty);
	std::fflush(stdout);
}

void NoteFlip() {
	g_flips.fetch_add(1, std::memory_order_relaxed);
	AstroTileGridPatch();
}

bool FindWide() {
	static const bool wide = [] {
		const char* value = std::getenv("KYTY_METALFX_FIND");
		return value != nullptr && std::strcmp(value, "3") == 0;
	}();
	return wide;
}

void FindLog(const char* kind, const PreparedBindings& bindings, TextureCache& cache, ImageId output,
             uint32_t color_count) {
	static const uint32_t width = [] {
		const char* value = std::getenv("KYTY_METALFX_FIND_W");
		return value != nullptr ? static_cast<uint32_t>(std::atoi(value)) : 2432u;
	}();
	if (bindings.runtime == nullptr || !*bindings.runtime) {
		return;
	}
	const Image* out   = output ? cache.TryGetImage(output) : nullptr;
	bool         wanted = out != nullptr && out->info.extent.width == width;
	if (out == nullptr) {
		for (const auto& binding: bindings.images) {
			const Image* image = binding.image_id ? cache.TryGetImage(binding.image_id) : nullptr;
			wanted = wanted || (image != nullptr && image->info.extent.width == width);
		}
	}
	auto& s = Get();
	const auto hash = bindings.runtime->program->shader_hash;
	// KYTY_METALFX_FIND_RESET_S=<s>: the once-per-shader list starts again that many seconds after
	// the first call (the steady scene instead of the loading screens), and from then the pass
	// KYTY_METALFX_PASS=<hash> has its images logged and every buffer of up to 64 KiB written to
	// KYTY_DUMP_DIR/mfxcb_<call>_<index>_<address>.bin for its next 6 calls (finding jitter words).
	static const int64_t reset_s = [] {
		const char* value = std::getenv("KYTY_METALFX_FIND_RESET_S");
		return value != nullptr ? static_cast<int64_t>(std::atoi(value)) : 0;
	}();
	static int64_t  first_ns   = 0;
	static bool     reset_done = false;
	static uint32_t pass_calls = 0;
	const auto      now        = NowNs();
	if (first_ns == 0) {
		first_ns = now;
	}
	if (reset_s > 0 && !reset_done && now - first_ns > reset_s * 1'000'000'000LL) {
		reset_done     = true;
		s.logged_count = 0;
		std::printf("MetalFX find3: ---- list restarted after %lld s ----\n", static_cast<long long>(reset_s));
	}
	if (reset_done && s.pass_hash != 0 && hash == s.pass_hash && pass_calls < 6) {
		const uint32_t call = pass_calls++;
		std::string    text;
		for (const auto& binding: bindings.images) {
			if (const Image* image = binding.image_id ? cache.TryGetImage(binding.image_id) : nullptr;
			    image != nullptr) {
				text += fmt::format(" {}x{}:{}@{:x}", image->info.extent.width, image->info.extent.height,
				                    vk::to_string(image->backing.format), image->info.data.address);
			} else {
				text += " -";
			}
		}
		std::printf("MetalFX find3: pass call %u output %llx images:%s\n", call,
		            static_cast<unsigned long long>(out != nullptr ? out->info.data.address : 0), text.c_str());
		uint32_t index = 0;
		for (const auto& source: bindings.buffer_sources) {
			if (source.address != 0 && source.size >= 4 && source.size <= 65536) {
				const char* dir  = std::getenv("KYTY_DUMP_DIR");
				const auto  path = fmt::format("{}/mfxcb_{}_{}_{:x}.bin", dir != nullptr ? dir : ".", call, index,
				                               source.address);
				if (auto* file = std::fopen(path.c_str(), "wb"); file != nullptr) {
					std::fwrite(Guest::Ptr<const uint8_t>(source.address), 1, source.size, file);
					std::fclose(file);
				}
			}
			index++;
		}
		std::fflush(stdout);
	}
	if (!wanted || !LogOnce(s, hash ^ 0x3333333333333333ull)) {
		return;
	}
	std::string text;
	for (const auto& binding: bindings.images) {
		if (const Image* image = binding.image_id ? cache.TryGetImage(binding.image_id) : nullptr;
		    image != nullptr) {
			text += fmt::format(" {}x{}:{}@{:x}", image->info.extent.width, image->info.extent.height,
			                    vk::to_string(image->backing.format), image->info.data.address);
		} else {
			text += " -";
		}
	}
	std::printf("MetalFX find3: %s shader %016llx targets %u output %s images:%s\n", kind,
	            static_cast<unsigned long long>(hash), color_count,
	            out != nullptr ? fmt::format("{}x{}:{}@{:x}", out->info.extent.width, out->info.extent.height,
	                                         vk::to_string(out->backing.format), out->info.data.address)
	                                 .c_str()
	                           : "-",
	            text.c_str());
	for (const auto& source: bindings.buffer_sources) {
		if (source.address == 0 || source.size < 16 || source.size > 512) {
			continue;
		}
		const auto* w = Guest::Ptr<const uint32_t>(source.address);
		const auto* f = reinterpret_cast<const float*>(w);
		std::string words;
		for (uint32_t i = 0; i < 24 && (i + 1) * 4 <= source.size; i++) {
			words += fmt::format(" {}:{:x}({:.5g})", i, w[i], static_cast<double>(f[i]));
		}
		std::printf("MetalFX find3:   buffer %llx+%u:%s\n", static_cast<unsigned long long>(source.address),
		            static_cast<unsigned>(source.size), words.c_str());
	}
	std::fflush(stdout);
}

namespace {

struct ResAuditConfig {
	int      level      = 0;
	int64_t  from_ns    = 0;
	uint32_t max_lines  = 4000;
	uint32_t base_w     = 1920;
	uint32_t base_h     = 1080;
	uint32_t internal_w = 1280;
	uint32_t internal_h = 720;
};

const ResAuditConfig& ResAuditGet() {
	static const ResAuditConfig config = [] {
		ResAuditConfig c;
		const auto     size = [](const char* name, uint32_t& w, uint32_t& h) {
			if (const char* value = std::getenv(name); value != nullptr) {
				unsigned a = 0;
				unsigned b = 0;
				if (std::sscanf(value, "%ux%u", &a, &b) == 2 && a != 0 && b != 0) {
					w = a;
					h = b;
				}
			}
		};
		if (const char* value = std::getenv("KYTY_RES_AUDIT"); value != nullptr) {
			c.level = std::atoi(value);
		}
		if (const char* value = std::getenv("KYTY_RES_AUDIT_FROM_S"); value != nullptr) {
			c.from_ns = static_cast<int64_t>(std::atoi(value)) * 1'000'000'000LL;
		}
		if (const char* value = std::getenv("KYTY_RES_AUDIT_MAX"); value != nullptr) {
			c.max_lines = static_cast<uint32_t>(std::max(std::atoi(value), 1));
		}
		size("KYTY_RES_AUDIT_BASE", c.base_w, c.base_h);
		size("KYTY_RES_AUDIT_INTERNAL", c.internal_w, c.internal_h);
		if (c.level > 0) {
			std::printf(
			    "ResAudit: on (level %d, base %ux%u, internal %ux%u, from %lld s, max %u lines)\n",
			    c.level, c.base_w, c.base_h, c.internal_w, c.internal_h,
			    static_cast<long long>(c.from_ns / 1'000'000'000LL), c.max_lines);
			std::fflush(stdout);
		}
		return c;
	}();
	return config;
}

// k when (w, h) is (base_w, base_h) / 2^k with the division rounded either way (k = 0..6), else -1.
int DerivedLevel(uint32_t w, uint32_t h, uint32_t base_w, uint32_t base_h) {
	for (int k = 0; k <= 6; k++) {
		const uint32_t d  = 1u << static_cast<uint32_t>(k);
		const uint32_t wf = base_w / d;
		const uint32_t wc = (base_w + d - 1) / d;
		const uint32_t hf = base_h / d;
		const uint32_t hc = (base_h + d - 1) / d;
		if ((w == wf || w == wc) && (h == hf || h == hc)) {
			return k;
		}
	}
	return -1;
}

// k when a grid of `groups` x `threads` covers (base / 2^k) with less than one group to spare.
int GridLevel(const uint32_t* groups, const uint32_t* threads, uint32_t base_w, uint32_t base_h) {
	const uint32_t tx = std::max(threads[0], 1u);
	const uint32_t ty = std::max(threads[1], 1u);
	for (int k = 0; k <= 6; k++) {
		const uint32_t d = 1u << static_cast<uint32_t>(k);
		const uint32_t w = (base_w + d - 1) / d;
		const uint32_t h = (base_h + d - 1) / d;
		if (groups[0] == (w + tx - 1) / tx && groups[1] == (h + ty - 1) / ty &&
		    (groups[0] > 1 || groups[1] > 1)) {
			return k;
		}
	}
	return -1;
}

} // namespace

bool ResAuditEnabled() {
	static const bool enabled = ResAuditGet().level > 0;
	return enabled;
}

void ResAudit(const char* kind, const PreparedBindings& bindings, TextureCache& cache,
              ImageId output, uint32_t color_count, const uint32_t* groups, const uint32_t* threads,
              uint64_t code_address) {
	const auto& c = ResAuditGet();
	if (c.level <= 0 || bindings.runtime == nullptr || !*bindings.runtime) {
		return;
	}
	static std::mutex                   mutex;
	static std::unordered_set<uint64_t> seen;
	static uint32_t                     lines    = 0;
	static int64_t                      first_ns = 0;
	std::lock_guard                     lock(mutex);
	const auto                          now = NowNs();
	if (first_ns == 0) {
		first_ns = now;
	}
	if (now - first_ns < c.from_ns || lines >= c.max_lines) {
		return;
	}
	const uint64_t hash = bindings.runtime->program->shader_hash;
	// KYTY_RES_AUDIT_SHADERS=<hash>[,<hash>...] [KYTY_RES_AUDIT_DUMP_W=<output width>]
	// (diagnostic): for these shaders (draws: only into a first target of that width), the
	// program's code up to s_endpgm goes to KYTY_DUMP_DIR/resaudit_<hash>_code.bin once, and calls
	// 0, 1, 2000, 2001, 8000, 8001, 20000, 20001 log "ResAudit: buffers" (address:size of every
	// bound buffer) and write each buffer of up to 4 MiB to
	// resaudit_<hash>_<call>_<index>_<address>.bin.
	{
		static const std::vector<uint64_t> wanted = [] {
			std::vector<uint64_t> list;
			if (const char* value = std::getenv("KYTY_RES_AUDIT_SHADERS"); value != nullptr) {
				const char* at = value;
				while (*at != 0) {
					char*      end = nullptr;
					const auto h   = std::strtoull(at, &end, 16);
					if (end == at) {
						break;
					}
					list.push_back(h);
					at = *end == ',' ? end + 1 : end;
				}
			}
			return list;
		}();
		static const uint32_t dump_w = [] {
			const char* value = std::getenv("KYTY_RES_AUDIT_DUMP_W");
			return value != nullptr ? static_cast<uint32_t>(std::atoi(value)) : 0u;
		}();
		if (!wanted.empty() && std::find(wanted.begin(), wanted.end(), hash) != wanted.end()) {
			const Image* target = output ? cache.TryGetImage(output) : nullptr;
			if (dump_w == 0 || std::strcmp(kind, "draw") != 0 ||
			    (target != nullptr && target->info.extent.width == dump_w)) {
				static std::unordered_map<uint64_t, uint32_t> calls;
				const uint32_t                                call = calls[hash]++;
				const char*                                   dir  = std::getenv("KYTY_DUMP_DIR");
				if (dir == nullptr) {
					dir = ".";
				}
				if (call == 0 && code_address != 0) {
					const auto* words = Guest::Ptr<const uint32_t>(code_address);
					uint32_t    n     = 0;
					while (n < 0x20000 && words[n] != 0xbf810000u) {
						n++;
					}
					if (auto* file = std::fopen(
					        fmt::format("{}/resaudit_{:016x}_code.bin", dir, hash).c_str(), "wb");
					    file != nullptr) {
						std::fwrite(words, 4, n + 1, file);
						std::fclose(file);
					}
				}
				if (call < 2 || call == 2000 || call == 2001 || call == 8000 || call == 8001 ||
				    call == 20000 || call == 20001) {
					std::string list;
					uint32_t    index = 0;
					for (const auto& source: bindings.buffer_sources) {
						list += fmt::format(" {:x}:{}", source.address, source.size);
						if (source.address != 0 && source.size >= 4 && source.size <= (4u << 20u)) {
							if (auto* file =
							        std::fopen(fmt::format("{}/resaudit_{:016x}_{}_{}_{:x}.bin",
							                               dir, hash, call, index, source.address)
							                       .c_str(),
							                   "wb");
							    file != nullptr) {
								std::fwrite(Guest::Ptr<const uint8_t>(source.address), 1,
								            source.size, file);
								std::fclose(file);
							}
						}
						index++;
					}
					std::string imgs;
					for (const auto& binding: bindings.images) {
						const Image* image =
						    binding.image_id ? cache.TryGetImage(binding.image_id) : nullptr;
						imgs +=
						    image != nullptr
						        ? fmt::format(" {}x{}x{}@{:x}", image->info.extent.width,
						                      image->info.extent.height, image->info.extent.depth,
						                      image->info.data.address)
						        : std::string(" -");
					}
					std::printf("ResAudit: buffers %s shader %016llx code %llx call %u out %ux%u "
					            "buffers:%s images:%s\n",
					            kind, static_cast<unsigned long long>(hash),
					            static_cast<unsigned long long>(code_address), call,
					            target != nullptr ? target->info.extent.width : 0,
					            target != nullptr ? target->info.extent.height : 0, list.c_str(),
					            imgs.c_str());
					std::fflush(stdout);
				}
			}
		}
	}
	uint64_t    sig = hash * 0x9e3779b97f4a7c15ull ^ (std::strcmp(kind, "draw") == 0 ? 1u : 2u);
	const auto  mix = [&sig](uint64_t v) { sig = (sig ^ v) * 0x100000001b3ull; };
	std::string flags;
	std::string text;
	int         img_k   = -1;
	bool        has_720 = false;
	uint32_t    max_w   = 0;
	uint32_t    max_h   = 0;
	const auto  note    = [&](const Image* image, std::string& out) {
		const uint32_t w = image->info.extent.width;
		const uint32_t h = image->info.extent.height;
		mix((static_cast<uint64_t>(w) << 32u) | h);
		const int k = DerivedLevel(w, h, c.base_w, c.base_h);
		if (k >= 0 && (img_k < 0 || k < img_k)) {
			img_k = k;
		}
		has_720 = has_720 || DerivedLevel(w, h, c.internal_w, c.internal_h) >= 0;
		max_w   = std::max(max_w, w);
		max_h   = std::max(max_h, h);
		out += fmt::format(" {}x{}{}:{}@{:x}", w, h,
		                   k >= 0 ? fmt::format("[1080/{}]", 1u << static_cast<uint32_t>(k)) : "",
		                   vk::to_string(image->backing.format), image->info.data.address);
	};
	const Image* out      = output ? cache.TryGetImage(output) : nullptr;
	std::string  out_text = "-";
	if (out != nullptr) {
		out_text.clear();
		note(out, out_text);
	}
	for (const auto& binding: bindings.images) {
		if (const Image* image = binding.image_id ? cache.TryGetImage(binding.image_id) : nullptr;
		    image != nullptr) {
			note(image, text);
		} else {
			text += " -";
		}
	}
	int         grid_k    = -1;
	std::string grid_text = "-";
	if (groups != nullptr && threads != nullptr) {
		mix((static_cast<uint64_t>(groups[0]) << 40u) ^ (static_cast<uint64_t>(groups[1]) << 20u) ^
		    groups[2]);
		grid_k = GridLevel(groups, threads, c.base_w, c.base_h);
		grid_text =
		    fmt::format("{}x{}x{} groups of {}x{}x{} = {}x{}", groups[0], groups[1], groups[2],
		                threads[0], threads[1], threads[2], groups[0] * std::max(threads[0], 1u),
		                groups[1] * std::max(threads[1], 1u));
		const uint32_t cover_w = groups[0] * std::max(threads[0], 1u);
		const uint32_t cover_h = groups[1] * std::max(threads[1], 1u);
		if (max_w != 0 && (cover_w >= max_w + std::max(threads[0], 1u) ||
		                   cover_h >= max_h + std::max(threads[1], 1u))) {
			flags += " grid>img";
		}
	}
	if (img_k >= 0) {
		flags += fmt::format(" img1080/{}", 1u << static_cast<uint32_t>(img_k));
	}
	if (grid_k >= 0) {
		flags += fmt::format(" grid1080/{}", 1u << static_cast<uint32_t>(grid_k));
	}
	if ((img_k >= 0 || grid_k >= 0) && has_720) {
		flags += " mixed";
	}
	if (c.level < 2 && img_k < 0 && grid_k < 0) {
		return;
	}
	if (!seen.insert(sig).second) {
		return;
	}
	lines++;
	std::printf(
	    "ResAudit: %s shader %016llx code %llx targets %u grid %s flags%s output %s images:%s\n",
	    kind, static_cast<unsigned long long>(hash), static_cast<unsigned long long>(code_address),
	    color_count, grid_text.c_str(), flags.empty() ? " -" : flags.c_str(), out_text.c_str(),
	    text.c_str());
	std::fflush(stdout);
}

} // namespace MetalFx

void RenderExecutor::NoteMetalFxTargets(std::span<const RenderColorInfo> colors, ImageId depth) {
	auto& s = MetalFx::Get();
	// Called for every draw before TryMetalFxUpscale: an upscale planned for a draw that was then
	// not recorded (its pipeline not built yet) is dropped.
	s.pending.valid = false;
	if (!depth) {
		return;
	}
	bool  recent = false;
	for (const auto id: s.recent_depths) {
		recent = recent || id == depth;
	}
	if (!recent) {
		s.recent_depths[s.recent_depth_next] = depth;
		s.recent_depth_next                  = (s.recent_depth_next + 1) % s.recent_depths.size();
	}
	for (const auto& color: colors) {
		if (!color.image_id) {
			continue;
		}
		bool found = false;
		for (auto& pair: s.depth_pairs) {
			if (pair.color == color.image_id) {
				pair.depth = depth;
				found      = true;
				break;
			}
		}
		if (!found) {
			s.depth_pairs[s.depth_next] = {color.image_id, depth};
			s.depth_next                = (s.depth_next + 1) % s.depth_pairs.size();
		}
	}
}

namespace MetalFx {
namespace {

bool SameSize(const Image& image, uint32_t w, uint32_t h) {
	return image.info.extent.width == w && image.info.extent.height == h && image.backing.image != nullptr &&
	       image.info.samples == 1;
}

// Records the upscale of a decoded pass (outside rendering). False: nothing was recorded.
bool Execute(State& s, TextureCache& cache, CommandBuffer& buffer, const Pending& p) {
	const auto alive = [&cache](ImageId id) { return id && cache.TryGetImage(id) != nullptr; };
	if (!alive(p.color) || !alive(p.output) || (p.temporal && !alive(p.motion)) || (p.depth && !alive(p.depth))) {
		return Fail(s, Fallback::Bindings);
	}
	auto&  color = cache.GetImage(p.color);
	auto&  out   = cache.GetImage(p.output);
	Image* mv    = p.temporal ? &cache.GetImage(p.motion) : nullptr;
	Image* depth = p.temporal && p.depth ? &cache.GetImage(p.depth) : nullptr;
	if (!SameSize(color, p.in_w, p.in_h) || !SameSize(out, p.out_w, p.out_h) ||
	    (mv != nullptr && !SameSize(*mv, p.in_w, p.in_h)) ||
	    (depth != nullptr && (!SameSize(*depth, p.in_w, p.in_h) || !depth->info.IsDepth()))) {
		return Fail(s, Fallback::Sizes);
	}

	// Native recording: outside rendering, the recorder drained (CommandBuffer::Handle).
	buffer.EndRendering();
	const auto cmd = buffer.Handle();
	color.Transit(vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits2::eShaderSampledRead, {}, cmd);
	if (mv != nullptr) {
		mv->Transit(vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits2::eShaderSampledRead, {}, cmd);
	}
	if (depth != nullptr) {
		depth->Transit(vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits2::eShaderSampledRead, {},
		               cmd);
	}
	out.Transit(vk::ImageLayout::eGeneral,
	            vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite, {}, cmd);

	KkMetalFxInfo info {};
	info.size   = sizeof(info);
	info.mode   = p.temporal ? 1 : 2;
	info.color  = static_cast<VkImage>(color.backing.image);
	info.depth  = depth != nullptr ? static_cast<VkImage>(depth->backing.image) : VK_NULL_HANDLE;
	info.motion = mv != nullptr ? static_cast<VkImage>(mv->backing.image) : VK_NULL_HANDLE;
	info.output = static_cast<VkImage>(out.backing.image);
	info.in_w   = p.in_w;
	info.in_h   = p.in_h;
	// Jitter in input pixels; motion vectors in UV from previous to current (previous = uv - mv):
	// MetalFX wants pixels pointing at the previous position, without the jitter.
	info.jitter[0]   = s.jitter_sign * p.jitter_uv[0] * static_cast<float>(p.in_w);
	info.jitter[1]   = s.jitter_sign * p.jitter_uv[1] * static_cast<float>(p.in_h);
	info.mv_scale[0] = -s.mv_sign * static_cast<float>(p.in_w);
	info.mv_scale[1] = -s.mv_sign * static_cast<float>(p.in_h);
	info.mv_bias[0]  = s.mv_jitter * (p.jitter_uv[0] - p.jitter_uv[2]);
	info.mv_bias[1]  = s.mv_jitter * (p.jitter_uv[1] - p.jitter_uv[3]);
	// A camera cut, a gap in the pass's frame counter or a new size (the game's dynamic resolution
	// selects another scaler in the driver, with an old history): the history starts again.
	info.reset = (p.cut || (s.last_frame != 0 && p.frame != s.last_frame + 1) || p.in_w != s.in_w ||
	              p.in_h != s.in_h || p.out_w != s.out_w || p.out_h != s.out_h)
	                 ? 1
	                 : 0;
	info.depth_reversed = s.depth_reversed ? 1 : 0;
	info.auto_exposure  = 1;
	info.depth_mode     = p.temporal && depth == nullptr ? 1 : 0;
	info.depth_near     = s.depth_near;
	const int result    = s.command(static_cast<VkCommandBuffer>(cmd), &info);
	buffer.ForgetBoundState();
	if (result != 1) {
		// Nothing was encoded: the game's picture stays.
		return Fail(s, Fallback::Refused);
	}
	s.last_frame     = p.frame;
	s.last_jitter[0] = info.jitter[0];
	s.last_jitter[1] = info.jitter[1];
	s.in_w           = p.in_w;
	s.in_h           = p.in_h;
	s.out_w          = p.out_w;
	s.out_h          = p.out_h;
	// The output now holds MetalFX's image: a GPU write of the target, as the draw's would be.
	// (the caller commits it: TextureCache::CommitGpuWrite)
	out.NoteContentWrite();
	s.resets += info.reset;
	s.evaluations++;
	Summary(s);
	return true;
}

// The configuration of this frame: g the game's picture, s spatial, t temporal, a-d temporal with
// the viewport jitter (KYTY_METALFX_CYCLE); changed: a new cycle window starts with this pass.
char CycleConfig(State& s, bool& changed) {
	char config = s.mode == Mode::Spatial ? 's' : s.jitter ? 'a' : 't';
	changed     = false;
	if (s.cycle_every != 0) {
		const auto flips = g_flips.load(std::memory_order_relaxed);
		const int  index = flips < s.cycle_from
		                       ? 0
		                       : static_cast<int>(((flips - s.cycle_from) / s.cycle_every) % s.cycle_list.size());
		config = s.cycle_list[static_cast<size_t>(index)];
		if (index != s.cycle_index) {
			// "prev": the window that just ended (pass draws, those upscaling from a scene size, the last
			// input size), so an analysis can tell level windows from loading screens.
			std::printf("MetalFX cycle: flip %llu t_us %lld config %c prev draws %llu scene %llu in %ux%u\n",
			            static_cast<unsigned long long>(flips), static_cast<long long>(NowNs() / 1000), config,
			            static_cast<unsigned long long>(s.win_draws), static_cast<unsigned long long>(s.win_scene),
			            s.win_in_w, s.win_in_h);
			s.win_draws = 0;
			s.win_scene = 0;
			std::fflush(stdout);
			s.cycle_index = index;
			changed       = true;
		}
	}
	return config;
}

} // namespace
} // namespace MetalFx

// True: the pass was replaced (V1019), the draw is done. False: the draw runs; when the pass is one
// that keeps its own draw (V1007) FinishMetalFx records the upscale after it.
bool RenderExecutor::TryMetalFxUpscale(CommandBuffer& buffer, PreparedBindings& pixel, ImageId output,
                                       uint32_t color_count) {
	using namespace MetalFx;
	auto& s         = Get();
	s.pending.valid = false;
	if (s.mode == Mode::Off || pixel.runtime == nullptr || !*pixel.runtime) {
		return false;
	}
	const auto hash   = pixel.runtime->program->shader_hash;
	bool       known  = false;
	Layout     layout = Layout::V1007;
	if (s.pass_hash != 0) {
		known  = hash == s.pass_hash;
		layout = s.pass_layout;
	} else {
		for (const auto& pass: KNOWN_PASSES) {
			if (hash == pass.hash) {
				known  = true;
				layout = pass.layout;
			}
		}
	}
	if (!known) {
		return false;
	}
	s.pass_draws++;
	const bool v1007 = layout == Layout::V1007;
	// The configuration of this frame: g the game's picture, s spatial, t temporal, a-d temporal
	// with the viewport jitter and MetalFX jitter signs ++, --, +-, -+ (KYTY_METALFX_CYCLE).
	bool changed = false;
	char config  = CycleConfig(s, changed);
	if (s.cycle_every != 0) {
		s.win_draws++;
		if (v1007 && !pixel.buffer_sources.empty() && pixel.buffer_sources[0].address != 0 &&
		    pixel.buffer_sources[0].size >= 40u) {
			const auto* sizes = Guest::Ptr<const uint32_t>(pixel.buffer_sources[0].address);
			s.win_in_w        = sizes[2];
			s.win_in_h        = sizes[3];
			if (sizes[2] != 0 && sizes[2] < sizes[0] && sizes[3] < sizes[1] && sizes[2] <= s.max_in_w) {
				s.win_scene++;
			}
		}
	}
	const bool jittered = config >= 'a' && config <= 'd';
	if (config == 'g') {
		s.jitter_px[0] = 0.0f;
		s.jitter_px[1] = 0.0f;
		Summary(s);
		return false;
	}
	// The profile's shape: V1007 two colour targets (picture, history) and three images.
	if (!output || pixel.images.size() < (v1007 ? 3u : 2u) || (v1007 && color_count != 2) ||
	    pixel.buffer_sources.empty() || pixel.buffer_sources[0].address == 0 ||
	    pixel.buffer_sources[0].size < (v1007 ? 40u : 56u)) {
		return Fail(s, Fallback::Bindings);
	}
	if (!s.lookup_tried) {
		s.lookup_tried = true;
		s.command      = Lookup();
	}
	if (s.command == nullptr) {
		return Fail(s, Fallback::NoDriver);
	}
	// nps5: guest bytes are at guest + bias (dlss.cpp reads the guest address itself: Windows identity map).
	const auto* cb = Guest::Ptr<const uint32_t>(pixel.buffer_sources[0].address);
	Pending     p;
	p.temporal = config != 's';
	p.layout   = layout;
	p.output   = output;
	p.motion   = pixel.images[0].image_id;
	p.color    = pixel.images[1].image_id;
	p.after    = v1007;
	bool mv_valid = false;
	if (v1007) {
		p.out_w = cb[0];
		p.out_h = cb[1];
		p.in_w  = cb[2];
		p.in_h  = cb[3];
		if (p.in_w == 0 || p.in_h == 0 || cb[4] != p.in_w - 1 || cb[5] != p.in_h - 1) {
			return Fail(s, Fallback::Sizes);
		}
		p.cut    = cb[6] == 1;
		p.frame  = cb[7];
		mv_valid = cb[9] != 0;
		p.cut    = p.cut || changed;
		if (s.jitter) {
			// The offset the frame's scene draws had (ViewportJitter), then the next frame's.
			static constexpr float SIGNS[4][2] {{1, 1}, {-1, -1}, {1, -1}, {-1, 1}};
			const auto* sign    = SIGNS[jittered ? config - 'a' : 0];
			const bool  applied = jittered && p.in_w == s.in_w && p.in_h == s.in_h;
			p.jitter_uv[0] = applied ? sign[0] * s.jitter_px[0] / static_cast<float>(p.in_w) : 0.0f;
			p.jitter_uv[1] = applied ? sign[1] * s.jitter_sign_y * s.jitter_px[1] / static_cast<float>(p.in_h) : 0.0f;
			p.jitter_uv[2] = p.jitter_uv[0];
			p.jitter_uv[3] = p.jitter_uv[1];
			const auto halton = [](uint32_t index, uint32_t base) {
				float f = 1.0f, r = 0.0f;
				for (; index > 0; index /= base) {
					f /= static_cast<float>(base);
					r += f * static_cast<float>(index % base);
				}
				return r;
			};
			const float ratio  = static_cast<float>(p.out_w) / static_cast<float>(p.in_w);
			const auto  phases = std::max<uint32_t>(8, static_cast<uint32_t>(8.0f * ratio * ratio + 0.999f));
			s.jitter_index     = (s.jitter_index + 1) % phases;
			s.jitter_px[0]     = jittered ? halton(s.jitter_index + 1, 2) - 0.5f : 0.0f;
			s.jitter_px[1]     = jittered ? halton(s.jitter_index + 1, 3) - 0.5f : 0.0f;
		}
	} else {
		std::memcpy(p.jitter_uv, cb, sizeof(p.jitter_uv));
		p.out_w  = cb[4];
		p.out_h  = cb[5];
		p.in_w   = cb[6];
		p.in_h   = cb[7];
		p.frame  = cb[11];
		mv_valid = cb[13] != 0;
	}
	// Only a real upscale from a scene size (not the 1:1 and near-4K passes of the loading screens'
	// dynamic-resolution levels, each of which would cost a scaler of its own).
	if (p.in_w >= p.out_w || p.in_h >= p.out_h || p.in_w > s.max_in_w) {
		return Fail(s, Fallback::NoUpscale);
	}
	if (p.temporal && !mv_valid) {
		return Fail(s, Fallback::NoMotionVectors);
	}
	auto& cache = m_context.GetTextureCache();
	if (p.temporal && (!v1007 || s.depth_attachment)) {
		for (const auto& pair: s.depth_pairs) {
			if (pair.color == p.motion) {
				p.depth = pair.depth;
			}
		}
		// V1007: else the last depth attachment of the input size.
		for (uint32_t i = 0; v1007 && !p.depth && i < s.recent_depths.size(); i++) {
			const auto id = s.recent_depths[(s.recent_depth_next + s.recent_depths.size() - 1 - i) %
			                                s.recent_depths.size()];
			if (id && cache.m_slot_images.try_get(id) != nullptr) {
				const auto& image = cache.GetImage(id);
				if (image.info.extent.width == p.in_w && image.info.extent.height == p.in_h) {
					p.depth = id;
				}
			}
		}
		if (!p.depth) {
			return Fail(s, Fallback::NoDepth);
		}
	}
	if (p.after) {
		s.pending       = p;
		s.pending.valid = true;
		return false;
	}
	if (!Execute(s, cache, buffer, p)) {
		return false;
	}
	cache.CommitGpuWrite(cache.GetImage(p.output));
	return true;
}

// After the draw of a pass that keeps its own draw: true when the upscale was recorded (rendering
// was ended; no draw-run record may be continued across it).
bool RenderExecutor::FinishMetalFx(CommandBuffer& buffer) {
	using namespace MetalFx;
	auto& s = Get();
	if (!s.pending.valid) {
		return false;
	}
	s.pending.valid = false;
	auto& cache = m_context.GetTextureCache();
	if (!Execute(s, cache, buffer, s.pending)) {
		return false;
	}
	cache.CommitGpuWrite(cache.GetImage(s.pending.output));
	return true;
}

// After a compute dispatch: when it is the pass that scales the scene colour to the output size
// (State::compute_hash, KYTY_METALFX_COMPUTE_PASS=<hex hash>, 0 = off; ASTRO BOT 01.007 levels:
// images[0] the colour at the scene size, images[1] the output, both RGBA16F), MetalFX overwrites
// the output. Always the SPATIAL scaler: the pass has no motion vectors or depth of its own. The
// game's dispatch has run (its bookkeeping stays as it is); true when the upscale was recorded.
bool RenderExecutor::TryMetalFxCompute(CommandBuffer& buffer, const PreparedBindings& bindings) {
	using namespace MetalFx;
	auto& s = Get();
	if (s.mode == Mode::Off || s.compute_hash == 0 || bindings.runtime == nullptr || !*bindings.runtime ||
	    bindings.runtime->program->shader_hash != s.compute_hash) {
		return false;
	}
	s.pass_draws++;
	auto& cache = m_context.GetTextureCache();
	if (bindings.images.size() < 2 || !bindings.images[0].image_id || !bindings.images[1].image_id ||
	    cache.TryGetImage(bindings.images[0].image_id) == nullptr ||
	    cache.TryGetImage(bindings.images[1].image_id) == nullptr) {
		return Fail(s, Fallback::Bindings);
	}
	Pending p;
	p.temporal = false;
	p.layout   = Layout::V1007;
	p.color    = bindings.images[0].image_id;
	p.output   = bindings.images[1].image_id;
	{
		const auto& in  = cache.GetImage(p.color);
		const auto& out = cache.GetImage(p.output);
		p.in_w          = in.info.extent.width;
		p.in_h          = in.info.extent.height;
		p.out_w         = out.info.extent.width;
		p.out_h         = out.info.extent.height;
	}
	const bool upscale = p.in_w < p.out_w && p.in_h < p.out_h && p.in_w <= s.max_in_w;
	if (s.cycle_every != 0) {
		s.win_draws++;
		s.win_in_w = p.in_w;
		s.win_in_h = p.in_h;
		s.win_scene += upscale ? 1 : 0;
	}
	bool       changed = false;
	const char config  = CycleConfig(s, changed);
	if (config == 'g') {
		Summary(s);
		return false;
	}
	if (!upscale) {
		return Fail(s, Fallback::NoUpscale);
	}
	if (!s.lookup_tried) {
		s.lookup_tried = true;
		s.command      = Lookup();
	}
	if (s.command == nullptr) {
		return Fail(s, Fallback::NoDriver);
	}
	p.frame = s.last_frame + 1;
	if (!Execute(s, cache, buffer, p)) {
		return false;
	}
	cache.CommitGpuWrite(cache.GetImage(p.output));
	return true;
}

} // namespace Libs::Graphics
