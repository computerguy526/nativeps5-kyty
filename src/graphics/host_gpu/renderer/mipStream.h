#ifndef KYTY_RENDERER_MIPSTREAM_H_
#define KYTY_RENDERER_MIPSTREAM_H_

// nps5 mipstream: adaptive mip residency driven by the game's own texture-streaming feedback.
//
// Astro Bot streams textures itself: its streamer gives up to 16 buckets x 256 textures a T#
// mip-statistics counter (MipStatsCntEn / MipStatsCntId), reads one bucket's GET_LOD_STATS report
// per frame (finest mip sampled + "MipClamp" count, lodStatsReport.h) and moves each texture's
// T# MIN_LOD: low resolution until the clamp is hit, then full resolution. Kyty already makes
// only the levels a view can sample resident (KYTY_TEXTURE_RESIDENT_TRIM, RequestedFirstLevel).
//
// 1) KYTY_MIP_STATS_VIEW_SHIFT=1 (on with KYTY_TEX_MIP_AUTO): exact feedback for the game.
//    Without VK_EXT_image_view_min_lod (KosmicKrisp) a sampled view starts at its whole MIN_LOD
//    level (imageViewMinLod.h), and a trimmed native image cuts a view below native_first
//    (KYTY_TEX_MIP_SKIP, this module). The instrumented shader queries the LOD on that host view,
//    i.e. relative to the host view's first level, but only added the T# BASE_LEVEL: every
//    measured level was too fine by (host base - BASE_LEVEL), so the streamer saw MipClamp hits
//    that hardware would not report and promoted textures that were far away. The field now
//    carries the host view's absolute base level instead (ViewShiftBase).
// 2) KYTY_TEX_MIP_AUTO=1: per texture (guest address), the finest level the game's own
//    instrumented draws actually sampled (absolute guest level, from the GET_LOD_STATS copy at
//    GPU completion, mapped counter -> textures by the draws recorded since the last reset)
//    becomes the floor of RequestedFirstLevel: a large sampled 2D texture keeps only the levels
//    the camera needs (at the render resolution actually used). It starts at
//    KYTY_TEX_MIP_AUTO_START levels skipped; a measurement that needs finer levels grows it at the
//    next binding (EnsureResidency, the existing trim grow path; memo hits recheck the level).
//    A coarser measurement is held for KYTY_TEX_MIP_AUTO_HOLD frames (hysteresis).
//    Textures that never carry a counter (characters, UI, effects: the game keeps them resident)
//    get the "static" floor once KYTY_TEX_MIP_AUTO_STATIC_FRAMES frames passed without one.
// 3) KYTY_TEX_MIP_SKIP_CLASS="stream=2,static=0" (without AUTO): a fixed skip per class, the
//    class being "the game streams it" (a counter was seen on its T#) or not. Unclassified
//    textures use the "new" value until classified (default: the stream value).
//
// State is one lossy lock-free table keyed by guest address (never cleared; a texture at a reused
// address inherits the old record until measured again, which only changes how sharp it starts).

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <utility>
#include <vector>

namespace Libs::Graphics::MipStream {

struct Config {
	bool     view_shift     = false;
	bool     auto_on        = false;
	bool     class_on       = false;
	bool     census         = false;
	uint32_t auto_max       = 3;  // most levels the measurement may drop
	uint32_t auto_start     = 2;  // levels dropped before the first measurement
	uint32_t auto_bias      = 0;  // levels finer than measured to keep
	uint32_t auto_hold      = 90; // frames a finer measurement is kept
	uint32_t static_frames  = 48; // frames without a counter before a texture counts as static
	uint32_t class_stream   = 2;
	uint32_t class_static   = 0;
	uint32_t class_new      = 2;
	uint32_t grow_budget_mb = 0; // per frame, 0 = unlimited
	bool     proxy          = false; // KYTY_TEX_MIP_AUTO_PROXY: no-counter textures measured through a co-sampled counter texture
	uint32_t proxy_bias     = 1;     // levels finer than the proxy estimate to keep
};

inline uint32_t EnvU32(const char* name, uint32_t fallback) {
	const char* value = std::getenv(name);
	return value != nullptr && value[0] != 0 ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10))
	                                         : fallback;
}
inline bool EnvOn(const char* name) {
	const char* value = std::getenv(name);
	return value != nullptr && value[0] != 0 && std::strcmp(value, "0") != 0;
}

inline const Config& Get() {
	static const Config config = [] {
		Config c;
		c.auto_on        = EnvOn("KYTY_TEX_MIP_AUTO");
		c.view_shift     = c.auto_on || EnvOn("KYTY_MIP_STATS_VIEW_SHIFT");
		c.census         = EnvOn("KYTY_TEX_MIP_CENSUS") || c.auto_on;
		c.auto_max       = EnvU32("KYTY_TEX_MIP_AUTO_MAX", 3);
		c.auto_start     = EnvU32("KYTY_TEX_MIP_AUTO_START", 2);
		c.auto_bias      = EnvU32("KYTY_TEX_MIP_AUTO_BIAS", 0);
		c.auto_hold      = EnvU32("KYTY_TEX_MIP_AUTO_HOLD", 90);
		c.static_frames  = EnvU32("KYTY_TEX_MIP_AUTO_STATIC_FRAMES", 48);
		c.grow_budget_mb = EnvU32("KYTY_TEX_MIP_AUTO_GROW_MB", 0);
		c.class_static   = EnvU32("KYTY_TEX_MIP_AUTO_STATIC", 0);
		c.proxy          = EnvOn("KYTY_TEX_MIP_AUTO_PROXY");
		c.proxy_bias     = EnvU32("KYTY_TEX_MIP_AUTO_PROXY_BIAS", 1);
		if (const char* cls = std::getenv("KYTY_TEX_MIP_SKIP_CLASS"); cls != nullptr && cls[0] != 0) {
			c.class_on = true;
			bool new_set = false;
			// "stream=2,static=0,new=2"
			for (const char* p = cls; *p != 0;) {
				const char* eq = std::strchr(p, '=');
				if (eq == nullptr) break;
				const auto  len   = static_cast<size_t>(eq - p);
				const auto  value = static_cast<uint32_t>(std::strtoul(eq + 1, nullptr, 10));
				if (len == 6 && std::strncmp(p, "stream", 6) == 0) c.class_stream = value;
				if (len == 6 && std::strncmp(p, "static", 6) == 0) c.class_static = value;
				if (len == 3 && std::strncmp(p, "new", 3) == 0) {
					c.class_new = value;
					new_set     = true;
				}
				const char* comma = std::strchr(eq, ',');
				if (comma == nullptr) break;
				p = comma + 1;
			}
			if (!new_set) c.class_new = c.class_stream;
			c.census = true;
		}
		if (c.auto_on) c.class_on = false;
		if (c.view_shift || c.auto_on || c.class_on || c.census) {
			std::fprintf(stderr,
			             "[mipstream] view_shift %d auto %d (max %u start %u bias %u hold %u static %u/%u "
			             "grow %u MiB/frame proxy %d bias %u) class %d (stream %u static %u new %u) census %d\n",
			             c.view_shift, c.auto_on, c.auto_max, c.auto_start, c.auto_bias, c.auto_hold,
			             c.static_frames, c.class_static, c.grow_budget_mb, c.proxy, c.proxy_bias, c.class_on, c.class_stream,
			             c.class_static, c.class_new, c.census);
		}
		return c;
	}();
	return config;
}

[[nodiscard]] inline bool Active() {
	const auto& c = Get();
	return c.auto_on || c.class_on || c.view_shift || c.census;
}

// Presented-frame clock (TextureCache's frame counter, set by its constructor).
inline std::atomic<const std::atomic<uint64_t>*> g_frame_source {nullptr};
[[nodiscard]] inline uint32_t Now() {
	const auto* source = g_frame_source.load(std::memory_order_relaxed);
	return source != nullptr ? static_cast<uint32_t>(source->load(std::memory_order_relaxed) & 0xffffffu) : 0u;
}
[[nodiscard]] inline uint32_t Age(uint32_t now, uint32_t stamp) { return (now - stamp) & 0xffffffu; }

// Record bits.
constexpr uint64_t FinestMask   = 0xfu;          // 0..3 finest measured level (absolute)
constexpr uint64_t MeasuredBit  = 1u << 4u;
constexpr uint64_t StreamedBit  = 1u << 5u;
constexpr uint32_t NativeShift  = 6u;            // 6..9 native_first
constexpr uint64_t SeenBit      = 1u << 10u;     // first_seen valid
constexpr uint64_t ProxyBit     = 1u << 15u;     // measured through a co-sampled counter texture
constexpr uint32_t GameShift    = 11u;           // 11..14 the game's own first level (view + MIN_LOD)
constexpr uint32_t StampShift   = 16u;           // 16..39 measurement stamp
constexpr uint32_t SeenShift    = 40u;           // 40..63 first seen

struct Slot {
	std::atomic<uint64_t> key {0};
	std::atomic<uint64_t> val {0};
};
constexpr uint32_t Slots  = 1u << 15u;
constexpr uint32_t Probes = 16;
inline Slot        g_table[Slots];

[[nodiscard]] inline Slot* Find(uint64_t address, bool create) {
	if (address == 0) return nullptr;
	uint64_t h = (address >> 8u) * 0x9e3779b97f4a7c15ull;
	auto     i = static_cast<uint32_t>(h >> 49u);
	for (uint32_t probe = 0; probe < Probes; probe++, i = (i + 1u) & (Slots - 1u)) {
		auto&    slot = g_table[i];
		uint64_t key  = slot.key.load(std::memory_order_acquire);
		if (key == address) return &slot;
		if (key == 0) {
			if (!create) return nullptr;
			if (slot.key.compare_exchange_strong(key, address, std::memory_order_acq_rel) || key == address) {
				return &slot;
			}
		}
	}
	return nullptr;
}

template <class F>
inline void Update(uint64_t address, F&& f) {
	auto* slot = Find(address, true);
	if (slot == nullptr) return;
	uint64_t old = slot->val.load(std::memory_order_relaxed);
	while (!slot->val.compare_exchange_weak(old, f(old), std::memory_order_relaxed)) {
	}
}

[[nodiscard]] inline uint64_t Read(uint64_t address) {
	const auto* slot = Find(address, false);
	return slot != nullptr ? slot->val.load(std::memory_order_relaxed) : 0u;
}

inline void NoteStreamed(uint64_t address) {
	if (const auto* slot = Find(address, false);
	    slot != nullptr && (slot->val.load(std::memory_order_relaxed) & StreamedBit) != 0) {
		return;
	}
	Update(address, [](uint64_t v) { return v | StreamedBit; });
}

inline void NoteNative(uint64_t address, uint32_t native_first) {
	if (!Active()) return;
	Update(address, [native_first](uint64_t v) {
		return (v & ~(uint64_t {0xf} << NativeShift)) | (uint64_t {native_first & 0xfu} << NativeShift);
	});
}

inline void NoteSeen(uint64_t address, uint32_t game_first) {
	game_first &= 0xfu;
	if (const auto* slot = Find(address, false); slot != nullptr) {
		const auto v = slot->val.load(std::memory_order_relaxed);
		if ((v & SeenBit) != 0 && ((v >> GameShift) & 0xfu) == game_first) return;
	}
	const uint64_t now = Now();
	Update(address, [now, game_first](uint64_t v) {
		v = (v & ~(uint64_t {0xf} << GameShift)) | (uint64_t {game_first} << GameShift);
		return (v & SeenBit) != 0 ? v : (v | SeenBit | (now << SeenShift));
	});
}

inline std::atomic<uint64_t> g_measurements {0};
inline std::atomic<uint64_t> g_finer {0};
inline std::atomic<uint64_t> g_coarser {0};
inline std::atomic<uint64_t> g_grows {0};
inline std::atomic<uint64_t> g_grow_bytes {0};

// A measured finest level (absolute guest level) with hysteresis: finer at once, coarser only
// after `hold` frames without a finer one.
inline void NoteMeasured(uint64_t address, uint32_t finest, bool proxy = false) {
	const uint64_t now  = Now();
	const uint32_t hold = Get().auto_hold;
	finest              = finest > 14u ? 14u : finest;
	g_measurements.fetch_add(1, std::memory_order_relaxed);
	Update(address, [&](uint64_t v) {
		const uint32_t old   = static_cast<uint32_t>(v & FinestMask);
		const uint32_t stamp = static_cast<uint32_t>((v >> StampShift) & 0xffffffu);
		const bool     had   = (v & MeasuredBit) != 0;
		// A real counter measurement wins over a proxy estimate.
		if (proxy && (v & StreamedBit) != 0) {
			return v;
		}
		if (had && finest > old && Age(static_cast<uint32_t>(now), stamp) <= hold) {
			return v; // coarser, still held
		}
		if (!had || finest < old) {
			g_finer.fetch_add(1, std::memory_order_relaxed);
		} else if (finest > old) {
			g_coarser.fetch_add(1, std::memory_order_relaxed);
		}
		v &= ~(FinestMask | (uint64_t {0xffffff} << StampShift));
		return v | MeasuredBit | (proxy ? ProxyBit : StreamedBit) | finest | (now << StampShift);
	});
}

// The absolute guest level the host view of a sampled T# starts at: BASE_LEVEL, raised to the
// whole MIN_LOD level (no VK_EXT_image_view_min_lod) and to the native image's first level.
[[nodiscard]] inline uint32_t ViewShiftBase(const uint32_t* tsharp) {
	const uint32_t base    = (tsharp[3] >> 12u) & 0xfu;
	const uint32_t last    = (tsharp[3] >> 16u) & 0xfu;
	const uint32_t min_lod = (tsharp[1] >> 8u) & 0xfffu;
	uint32_t       host    = base;
	if (last > base) {
		const uint32_t whole = min_lod >> 8u;
		if (whole > host) host = whole < last ? whole : last;
		const uint64_t address = ((tsharp[0] | (static_cast<uint64_t>(tsharp[1]) << 32u)) & 0xFFFFFFFFFFull) << 8u;
		const uint32_t native  = static_cast<uint32_t>((Read(address) >> NativeShift) & 0xfu);
		if (native > host) host = native < last ? native : last;
	}
	return host;
}

[[nodiscard]] inline uint64_t TsharpAddress(const uint32_t* tsharp) {
	return ((tsharp[0] | (static_cast<uint64_t>(tsharp[1]) << 32u)) & 0xFFFFFFFFFFull) << 8u;
}

// The skip floor (absolute level) a policy wants for an eligible texture, before the per-texture
// limit (MipSkipLevels with `max` as the request). Returns the requested skip.
[[nodiscard]] inline uint32_t PolicySkip(uint64_t address) {
	const auto& c   = Get();
	const auto  v   = Read(address);
	const auto  now = Now();
	if (c.auto_on) {
		if ((v & MeasuredBit) != 0) {
			const uint32_t finest = static_cast<uint32_t>(v & FinestMask);
			const uint32_t want   = finest > c.auto_bias ? finest - c.auto_bias : 0u;
			return want < c.auto_max ? want : c.auto_max;
		}
		if ((v & StreamedBit) == 0 && (v & SeenBit) != 0 &&
		    Age(now, static_cast<uint32_t>((v >> SeenShift) & 0xffffffu)) > c.static_frames) {
			return c.class_static;
		}
		return c.auto_start;
	}
	if (c.class_on) {
		if ((v & StreamedBit) != 0) return c.class_stream;
		if ((v & SeenBit) != 0 && Age(now, static_cast<uint32_t>((v >> SeenShift) & 0xffffffu)) > c.static_frames) {
			return c.class_static;
		}
		return c.class_new;
	}
	return 0;
}

// Counter -> texture association of the draws recorded since the last counter reset (render
// thread, command order), snapshotted by each GET_LOD_STATS copy.
inline std::mutex                                 g_assoc_mutex;
inline std::vector<std::pair<uint32_t, uint64_t>> g_assoc;

// `counter` bits 0..7: counter id; bit 12: proxy, bits 8..11: level offset + 8 (proxy only).
inline void NoteAssociation(uint32_t counter, uint64_t address) {
	std::scoped_lock lock(g_assoc_mutex);
	// Draws reuse the same pairs many times per frame; a short look-back keeps the list small.
	const size_t n = g_assoc.size();
	for (size_t i = n > 8 ? n - 8 : 0; i < n; i++) {
		if (g_assoc[i].first == counter && g_assoc[i].second == address) return;
	}
	if (g_assoc.size() < 65536) g_assoc.emplace_back(counter, address);
}

} // namespace Libs::Graphics::MipStream

#endif // KYTY_RENDERER_MIPSTREAM_H_
