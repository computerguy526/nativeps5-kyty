// nps5: MetalFX in place of Astro Bot's temporal upscale pass (metalfx.h).
#include "graphics/host_gpu/renderer/metalfx.h"
#include "graphics/host_gpu/renderer/metalfxUe.h"

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
enum class Layout : uint8_t { V1019, V1007, Blit, Ue };
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
	float    mv_mul[2] {1.0f, 1.0f}; // Ue: the upscaler's own motion vector scale
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
	uint32_t    compute_in = 0, compute_out = 1; // image indices of that pass (mfxlayer: <hash>:<in>:<out>)
	bool        compute_replace = false;         // KYTY_METALFX_COMPUTE_REPLACE
	int         blit_index = -1;                 // KYTY_METALFX_PASS=<hash>:blit:<index>, -1 = auto
	bool        blit_auto  = false;              // KYTY_METALFX_PASS=auto:blit (no hash: by shape)
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
			} else if (std::strcmp(value, "auto:blit") == 0) {
				s.blit_auto = true;
			} else {
				char* end   = nullptr;
				s.pass_hash = std::strtoull(value, &end, 16);
				s.pass_layout = (end != nullptr && std::strcmp(end, ":1019") == 0) ? Layout::V1019 : Layout::V1007;
				if (end != nullptr && std::strncmp(end, ":blit", 5) == 0) {
					s.pass_layout = Layout::Blit;
					s.blit_index  = end[5] == ':' ? std::atoi(end + 6) : -1;
				}
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
			char* end      = nullptr;
			s.compute_hash = std::strtoull(value, &end, 16);
			if (end != nullptr && *end == ':') {
				s.compute_in = static_cast<uint32_t>(std::strtoul(end + 1, &end, 10));
				if (end != nullptr && *end == ':') {
					s.compute_out = static_cast<uint32_t>(std::strtoul(end + 1, nullptr, 10));
				}
			}
		}
		if (const char* value = std::getenv("KYTY_METALFX_COMPUTE_REPLACE"); value != nullptr) {
			s.compute_replace = std::strcmp(value, "1") == 0;
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
	static const bool requested = Get().mode != Mode::Off || Get().find || FindWide() || UeEnabled();
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

bool ComputeReplace(uint64_t shader_hash) {
	const auto& s = Get();
	return s.compute_replace && s.mode != Mode::Off && s.compute_hash != 0 && s.compute_hash == shader_hash;
}

namespace {
struct KoConfig {
	std::vector<uint64_t> hashes;
	uint64_t              cycle   = 0;
	int64_t               from_ns = 90'000'000'000LL;
};
const KoConfig& Ko() {
	static const KoConfig config = [] {
		KoConfig c;
		if (const char* value = std::getenv("KYTY_CS_KO"); value != nullptr) {
			for (const char* at = value; *at != '\0';) {
				char*      end  = nullptr;
				const auto hash = std::strtoull(at, &end, 16);
				if (end == at) {
					break;
				}
				if (hash != 0) {
					c.hashes.push_back(hash);
				}
				at = *end == ',' ? end + 1 : end;
				if (*end != ',' ) {
					break;
				}
			}
		}
		if (const char* value = std::getenv("KYTY_CS_KO_CYCLE"); value != nullptr) {
			c.cycle = std::strtoull(value, nullptr, 10);
		}
		if (const char* value = std::getenv("KYTY_CS_KO_FROM_S"); value != nullptr) {
			c.from_ns = static_cast<int64_t>(std::atoi(value)) * 1'000'000'000LL;
		}
		return c;
	}();
	return config;
}
} // namespace

bool CsKnockoutEnabled() {
	static const bool enabled = !Ko().hashes.empty();
	return enabled;
}

// Called on the command-processor thread for every compute dispatch while enabled.
bool CsKnockout(uint64_t shader_hash) {
	const auto&     c = Ko();
	static int64_t  first_ns   = 0;
	static uint64_t base_flip  = 0;
	static int      state      = -1; // -1 before from_s, 0 off window, 1 on window
	static uint64_t skipped    = 0;
	const auto      now        = NowNs();
	if (first_ns == 0) {
		first_ns = now;
	}
	if (now - first_ns < c.from_ns) {
		return false;
	}
	const auto flips = g_flips.load(std::memory_order_relaxed);
	if (state < 0) {
		base_flip = flips;
	}
	// The first window is OFF (a control window after from_s), then on, off, ...
	const int want = c.cycle == 0 ? 1 : static_cast<int>(((flips - base_flip) / c.cycle) % 2);
	if (want != state) {
		std::printf("CsKo cycle: flip %llu t_us %lld %s skipped %llu\n", static_cast<unsigned long long>(flips),
		            static_cast<long long>(now / 1000), want != 0 ? "on" : "off",
		            static_cast<unsigned long long>(skipped));
		std::fflush(stdout);
		state   = want;
		skipped = 0;
	}
	if (want == 0 || std::find(c.hashes.begin(), c.hashes.end(), shader_hash) == c.hashes.end()) {
		return false;
	}
	skipped++;
	return true;
}

bool FindWide() {
	static const bool wide = [] {
		const char* value = std::getenv("KYTY_METALFX_FIND");
		return value != nullptr && std::strcmp(value, "3") == 0;
	}();
	return wide;
}

namespace {
// KYTY_METALFX_FIND_ANCHOR (metalfx.h): the passes around one dispatch (a UE upscaler pass).
struct AnchorRec {
	struct Img {
		uint32_t w, h;
		vk::Format format;
		uint64_t address;
	};
	struct Buf {
		uint64_t address;
		uint32_t size;
	};
	uint64_t hash = 0;
	char     kind = '?';
	uint32_t image_count = 0, buffer_count = 0;
	Img      out {};
	std::array<Img, 16> images {};
	std::array<Buf, 8>  buffers {};
};

std::string AnchorText(const AnchorRec& r) {
	std::string text = fmt::format("{} shader {:016x} output {} images:", r.kind == 'd' ? "draw" : "dispatch", r.hash,
	                               r.out.address != 0 ? fmt::format("{}x{}:{}@{:x}", r.out.w, r.out.h,
	                                                                vk::to_string(r.out.format), r.out.address)
	                                                  : std::string("-"));
	for (uint32_t i = 0; i < r.image_count; i++) {
		const auto& m = r.images[i];
		text += m.address != 0 ? fmt::format(" {}x{}:{}@{:x}", m.w, m.h, vk::to_string(m.format), m.address) : " -";
	}
	text += " buffers:";
	for (uint32_t i = 0; i < r.buffer_count; i++) {
		text += fmt::format(" {:x}+{}", r.buffers[i].address, r.buffers[i].size);
	}
	return text;
}

bool AnchorShares(const AnchorRec& a, const AnchorRec& b) {
	const auto has = [&b](uint64_t address) {
		if (address == 0) {
			return false;
		}
		if (b.out.address == address) {
			return true;
		}
		for (uint32_t i = 0; i < b.image_count; i++) {
			if (b.images[i].address == address) {
				return true;
			}
		}
		return false;
	};
	// 1x1 and other tiny images are shared dummies, not data flow
	if (a.out.w > 8 && has(a.out.address)) {
		return true;
	}
	for (uint32_t i = 0; i < a.image_count; i++) {
		if (a.images[i].w > 8 && has(a.images[i].address)) {
			return true;
		}
	}
	return false;
}

void AnchorWords(const AnchorRec& r) {
	for (uint32_t i = 0; i < r.buffer_count; i++) {
		const auto& b = r.buffers[i];
		if (b.address == 0 || b.size < 16 || b.size > 16384) {
			continue;
		}
		const auto* w = Guest::Ptr<const uint32_t>(b.address);
		for (uint32_t at = 0; at * 4 < b.size; at += 32) {
			std::string words;
			for (uint32_t k = at; k < at + 32 && (k + 1) * 4 <= b.size; k++) {
				words += fmt::format(" {:08x}", w[k]);
			}
			std::printf("MetalFX anchor:   words %016llx %llx+%u @%u:%s\n", static_cast<unsigned long long>(r.hash),
			            static_cast<unsigned long long>(b.address), b.size, at, words.c_str());
		}
	}
}

void FindAnchor(const char* kind, const PreparedBindings& bindings, TextureCache& cache, const Image* out,
                uint64_t hash, bool started) {
	static const uint64_t anchor = [] {
		const char* value = std::getenv("KYTY_METALFX_FIND_ANCHOR");
		return value != nullptr ? std::strtoull(value, nullptr, 16) : 0ull;
	}();
	if (anchor == 0) {
		return;
	}
	static int frames_left = [] {
		const char* value = std::getenv("KYTY_METALFX_FIND_ANCHOR_FRAMES");
		return value != nullptr ? std::atoi(value) : 2;
	}();
	static std::array<AnchorRec, 64> ring;
	static uint32_t  ring_next = 0, ring_count = 0;
	static AnchorRec anchor_rec;
	static uint64_t  anchor_flip = ~0ull;
	static uint32_t  after_count = 0;

	AnchorRec r;
	r.hash = hash;
	r.kind = kind[1] == 'r' ? 'd' : 'c'; // "draw" / "dispatch"
	if (out != nullptr) {
		r.out = {out->info.extent.width, out->info.extent.height, out->backing.format, out->info.data.address};
	}
	for (const auto& binding: bindings.images) {
		if (r.image_count == r.images.size()) {
			break;
		}
		const Image* image = binding.image_id ? cache.TryGetImage(binding.image_id) : nullptr;
		r.images[r.image_count++] = image != nullptr ? AnchorRec::Img {image->info.extent.width, image->info.extent.height,
		                                                                image->backing.format, image->info.data.address}
		                                             : AnchorRec::Img {};
	}
	for (const auto& source: bindings.buffer_sources) {
		if (r.buffer_count == r.buffers.size()) {
			break;
		}
		r.buffers[r.buffer_count++] = {source.address, static_cast<uint32_t>(source.size)};
	}
	const auto flips = g_flips.load(std::memory_order_relaxed);
	if (anchor_flip == flips && hash != anchor && after_count < 48 && AnchorShares(anchor_rec, r)) {
		std::printf("MetalFX anchor: +%u %s\n", ++after_count, AnchorText(r).c_str());
		std::fflush(stdout);
	}
	if (hash == anchor && started && frames_left > 0 && anchor_flip != flips) {
		frames_left--;
		anchor_flip = flips;
		anchor_rec  = r;
		after_count = 0;
		std::printf("MetalFX anchor: ==== flip %llu %s\n", static_cast<unsigned long long>(flips), AnchorText(r).c_str());
		AnchorWords(r);
		std::array<uint8_t, 64> level {};
		for (uint32_t k = 0; k < ring_count; k++) {
			level[k] = AnchorShares(r, ring[(ring_next + ring.size() - 1 - k) % ring.size()]) ? 1 : 0;
		}
		for (uint32_t k = 0; k < ring_count; k++) {
			const auto& rk = ring[(ring_next + ring.size() - 1 - k) % ring.size()];
			for (uint32_t j = 0; level[k] == 0 && j < ring_count; j++) {
				if (level[j] == 1 && AnchorShares(ring[(ring_next + ring.size() - 1 - j) % ring.size()], rk)) {
					level[k] = 2;
				}
			}
		}
		for (uint32_t k = 0; k < ring_count; k++) {
			const auto& rk = ring[(ring_next + ring.size() - 1 - k) % ring.size()];
			std::printf("MetalFX anchor: -%u L%u %s\n", k + 1, static_cast<unsigned>(level[k]), AnchorText(rk).c_str());
			if (level[k] == 1 && rk.kind == 'c') {
				AnchorWords(rk);
			}
		}
		std::fflush(stdout);
	}
	ring[ring_next] = r;
	ring_next       = (ring_next + 1) % ring.size();
	ring_count      = std::min<uint32_t>(ring_count + 1, static_cast<uint32_t>(ring.size()));
}
} // namespace

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
	FindAnchor(kind, bindings, cache, out, hash, reset_s == 0 || reset_done);
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
	    (depth != nullptr && (!SameSize(*depth, p.in_w, p.in_h) ||
	                          (!depth->info.IsDepth() &&
	                           !(p.layout == Layout::Ue && depth->backing.format == vk::Format::eR32Sfloat))))) {
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
	info.mv_scale[0] = -s.mv_sign * static_cast<float>(p.in_w) * p.mv_mul[0];
	info.mv_scale[1] = -s.mv_sign * static_cast<float>(p.in_h) * p.mv_mul[1];
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


struct UeState {
	bool     log = false, adapter = false, skip = true;
	uint64_t flip = ~0ull;
	uint32_t index = 0;
	ImageId  depth, depth_r32, motion, replaced;
	uint32_t healthy = 0;
	uint64_t cool_until = 0, last_eval_flip = 0;
	uint64_t skipped = 0, kept = 0;
	int64_t  print_ns = 0;
};
UeState& Ue() {
	static UeState state = [] {
		UeState u;
		if (const char* value = std::getenv("KYTY_METALFX_UE"); value != nullptr) {
			u.log     = std::strcmp(value, "log") == 0 || std::strcmp(value, "fsr3") == 0;
			u.adapter = std::strcmp(value, "fsr3") == 0;
		}
		if (const char* value = std::getenv("KYTY_METALFX_UE_SKIP"); value != nullptr) {
			u.skip = std::strcmp(value, "0") != 0;
		}
		return u;
	}();
	return state;
}

} // namespace

bool UeEnabled() {
	static const bool enabled = Ue().log;
	return enabled;
}

ImageId UeOutput() {
	return Ue().replaced;
}

int UeDispatch(CommandBuffer& buffer, const PreparedBindings& bindings, TextureCache& cache) {
	if (bindings.runtime == nullptr || !*bindings.runtime) {
		return 0;
	}
	Fsr3Frame f;
	bool      found = false;
	for (const auto& source: bindings.buffer_sources) {
		if (source.address != 0 && source.size >= 96 && source.size <= 256 &&
		    ParseFsr3Constants(Guest::Ptr<const uint32_t>(source.address), source.size, f)) {
			found = true;
			break;
		}
	}
	if (!found) {
		return 0;
	}
	auto&      s     = Get();
	auto&      u     = Ue();
	const auto flips = g_flips.load(std::memory_order_relaxed);
	if (u.flip != flips) {
		u.flip      = flips;
		u.index     = 0;
		u.depth     = {};
		u.depth_r32 = {};
		u.motion    = {};
	}
	const uint32_t index = u.index++;
	const auto     hash  = bindings.runtime->program->shader_hash;
	ImageId        color {}, output {};
	uint32_t       out_colors = 0;
	for (const auto& binding: bindings.images) {
		const Image* image = binding.image_id ? cache.TryGetImage(binding.image_id) : nullptr;
		if (image == nullptr) {
			continue;
		}
		const uint32_t w = image->info.extent.width, h = image->info.extent.height;
		const auto     format     = image->backing.format;
		const bool     render     = w == f.in_w && h == f.in_h;
		const bool     upscaled   = w == f.out_w && h == f.out_h;
		const bool     color_like = format == vk::Format::eB10G11R11UfloatPack32 || format == vk::Format::eR16G16B16A16Sfloat;
		if (render && image->info.IsDepth() && !u.depth) {
			u.depth = binding.image_id;
		} else if (render && format == vk::Format::eR32Sfloat && !u.depth_r32) {
			u.depth_r32 = binding.image_id;
		} else if (render && format == vk::Format::eR16G16Sfloat) {
			u.motion = binding.image_id;
		} else if (render && color_like && !upscaled) {
			color = binding.image_id;
		} else if (upscaled && color_like) {
			output = binding.image_id;
			out_colors++;
		}
	}
	const bool last = color && out_colors >= 2;
	if (LogOnce(s, hash ^ 0x7777777777777777ull)) {
		std::printf("MetalFX ue: fsr3 #%u shader %016llx role %s render %ux%u (max %ux%u) upscale %ux%u jitter %.4f %.4f mv scale %.3f %.3f | depth %d r32 %d motion %d colour %d outputs %u\n",
		            index, static_cast<unsigned long long>(hash), last ? "last" : index == 0 ? "first" : "mid", f.in_w,
		            f.in_h, f.max_w, f.max_h, f.out_w, f.out_h, static_cast<double>(f.jitter[0]),
		            static_cast<double>(f.jitter[1]), static_cast<double>(f.mv_scale[0]),
		            static_cast<double>(f.mv_scale[1]), u.depth ? 1 : 0, u.depth_r32 ? 1 : 0, u.motion ? 1 : 0,
		            color ? 1 : 0, out_colors);
		std::fflush(stdout);
	}
	if (!u.adapter || s.mode != Mode::Temporal) {
		return 0;
	}
	bool       changed = false;
	const char config  = CycleConfig(s, changed);
	if (config == 'g') {
		u.healthy = 0;
		return 0;
	}
	if (const auto now = NowNs(); now - u.print_ns > 10'000'000'000LL) {
		if (u.print_ns != 0) {
			std::printf("MetalFX ue 10s: fsr3 dispatches skipped %llu kept %llu healthy %u\n",
			            static_cast<unsigned long long>(u.skipped), static_cast<unsigned long long>(u.kept), u.healthy);
			std::fflush(stdout);
		}
		u.print_ns = now;
		u.skipped  = 0;
		u.kept     = 0;
	}
	if (!last) {
		// The first dispatch (PrepareInputs) writes the dilated motion vectors MetalFX reads.
		if (index != 0 && u.skip && u.healthy >= 8 && flips >= u.cool_until) {
			u.skipped++;
			return 1;
		}
		u.kept++;
		return 0;
	}
	s.pass_draws++;
	if (s.cycle_every != 0) {
		s.win_draws++;
		s.win_scene++;
		s.win_in_w = f.in_w;
		s.win_in_h = f.in_h;
	}
	const auto fail = [&](Fallback reason) {
		u.healthy    = 0;
		u.cool_until = flips + 300;
		u.kept++;
		Fail(s, reason);
		return 0;
	};
	const ImageId depth = u.depth ? u.depth : u.depth_r32;
	if (!u.motion) {
		return fail(Fallback::NoMotionVectors);
	}
	if (!depth) {
		return fail(Fallback::NoDepth);
	}
	if (f.in_w >= f.out_w || f.in_h >= f.out_h || f.in_w > s.max_in_w || f.out_w > 3 * f.in_w || f.out_h > 3 * f.in_h) {
		return fail(Fallback::NoUpscale);
	}
	if (!s.lookup_tried) {
		s.lookup_tried = true;
		s.command      = Lookup();
	}
	if (s.command == nullptr) {
		return fail(Fallback::NoDriver);
	}
	Pending p;
	p.temporal = true;
	p.layout   = Layout::Ue;
	p.color    = color;
	p.motion   = u.motion;
	p.depth    = depth;
	p.output   = output;
	p.in_w     = f.in_w;
	p.in_h     = f.in_h;
	p.out_w    = f.out_w;
	p.out_h    = f.out_h;
	p.frame    = s.last_frame + 1;
	p.cut      = changed || u.last_eval_flip + 1 != flips;
	p.jitter_uv[0] = f.jitter[0] / static_cast<float>(f.in_w);
	p.jitter_uv[1] = s.jitter_sign_y * f.jitter[1] / static_cast<float>(f.in_h);
	p.jitter_uv[2] = p.jitter_uv[0];
	p.jitter_uv[3] = p.jitter_uv[1];
	p.mv_mul[0]    = f.mv_scale[0];
	p.mv_mul[1]    = f.mv_scale[1];
	// mfxlexec: cycle letters a/b/c flip the jitter sign, the motion sign, both (KYTY_METALFX_CYCLE=80,tabc).
	const float saved_jitter_sign = s.jitter_sign, saved_mv_sign = s.mv_sign;
	if (config == 'a' || config == 'c') {
		s.jitter_sign = -s.jitter_sign;
	}
	if (config == 'b' || config == 'c') {
		s.mv_sign = -s.mv_sign;
	}
	const bool executed = Execute(s, cache, buffer, p);
	s.jitter_sign       = saved_jitter_sign;
	s.mv_sign           = saved_mv_sign;
	if (!executed) {
		u.healthy    = 0;
		u.cool_until = flips + 300;
		u.kept++;
		return 0;
	}
	u.replaced = p.output; // the caller commits the GPU write (UeOutput)
	u.healthy++;
	u.last_eval_flip = flips;
	u.skipped++;
	return 2;
}

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
	if (!known && s.blit_auto) {
		known  = true;
		layout = Layout::Blit;
	}
	if (!known) {
		return false;
	}
	s.pass_draws++;
	if (layout == Layout::Blit) {
		// mfxlayer: UE's secondary upscale draw (one bound image scaled to a larger single target).
		auto&        blit_cache = m_context.GetTextureCache();
		const Image* target     = output ? blit_cache.TryGetImage(output) : nullptr;
		if (s.blit_auto) {
			// By shape (experiments only): one colour target at least 2560 wide and exactly one bound
			// image larger than 8 texels, which is smaller than the target and has its aspect.
			uint32_t big = 0, fit = 0;
			for (const auto& binding: pixel.images) {
				const Image* image = binding.image_id ? blit_cache.TryGetImage(binding.image_id) : nullptr;
				if (image == nullptr || image->info.extent.width <= 8 || target == nullptr) {
					continue;
				}
				big++;
				const uint32_t w = image->info.extent.width, h = image->info.extent.height;
				const int64_t  skew = static_cast<int64_t>(w) * target->info.extent.height -
				                      static_cast<int64_t>(h) * target->info.extent.width;
				fit += (w < target->info.extent.width && h < target->info.extent.height && w >= 960 &&
				        std::llabs(skew) <= static_cast<int64_t>(target->info.extent.width + target->info.extent.height))
				           ? 1
				           : 0;
			}
			if (color_count != 1 || target == nullptr || target->info.extent.width < 2560 || big != 1 || fit != 1) {
				s.pass_draws--;
				return false;
			}
			if (LogOnce(s, hash ^ 0x5555555555555555ull)) {
				std::printf("MetalFX blit: auto shader %016llx target %ux%u:%s\n", static_cast<unsigned long long>(hash),
				            target->info.extent.width, target->info.extent.height,
				            vk::to_string(target->backing.format).c_str());
				std::fflush(stdout);
			}
		}
		bool       blit_changed = false;
		const char blit_config  = CycleConfig(s, blit_changed);
		if (color_count != 1 || target == nullptr) {
			return Fail(s, Fallback::Bindings);
		}
		Pending p;
		p.temporal = false;
		p.layout   = layout;
		p.output   = output;
		p.out_w    = target->info.extent.width;
		p.out_h    = target->info.extent.height;
		for (size_t i = 0; i < pixel.images.size(); i++) {
			const Image* image = pixel.images[i].image_id ? blit_cache.TryGetImage(pixel.images[i].image_id) : nullptr;
			if (image == nullptr || (s.blit_index >= 0 && static_cast<size_t>(s.blit_index) != i)) {
				continue;
			}
			const uint32_t w = image->info.extent.width, h = image->info.extent.height;
			// the target's aspect within one texel, smaller than the target, the widest such image
			const int64_t skew = static_cast<int64_t>(w) * p.out_h - static_cast<int64_t>(h) * p.out_w;
			if (w < p.out_w && h < p.out_h && w > p.in_w && std::llabs(skew) <= static_cast<int64_t>(p.out_w + p.out_h)) {
				p.color = pixel.images[i].image_id;
				p.in_w  = w;
				p.in_h  = h;
			}
		}
		if (s.cycle_every != 0) {
			s.win_draws++;
			s.win_in_w = p.in_w;
			s.win_in_h = p.in_h;
			s.win_scene += p.color ? 1 : 0;
		}
		if (blit_config == 'g') {
			Summary(s);
			return false;
		}
		if (!p.color || p.in_w > s.max_in_w) {
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
		p.cut   = blit_changed;
		if (!Execute(s, blit_cache, buffer, p)) {
			return false;
		}
		blit_cache.CommitGpuWrite(blit_cache.GetImage(p.output));
		return true;
	}
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
	const uint32_t ci = s.compute_in, co = s.compute_out;
	if (bindings.images.size() <= std::max(ci, co) || ci == co || !bindings.images[ci].image_id ||
	    !bindings.images[co].image_id || cache.TryGetImage(bindings.images[ci].image_id) == nullptr ||
	    cache.TryGetImage(bindings.images[co].image_id) == nullptr) {
		return Fail(s, Fallback::Bindings);
	}
	Pending p;
	p.temporal = false;
	p.layout   = Layout::V1007;
	p.color    = bindings.images[ci].image_id;
	p.output   = bindings.images[co].image_id;
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
