#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_WINDOW_PRESENTGEOMETRY_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_WINDOW_PRESENTGEOMETRY_H_

// winresize: pure window/present geometry (no SDL, no Vulkan) so it can be unit-tested.
//  - Letterbox: where the game frame lands inside a swapchain of a different aspect.
//  - ClampExtent: swapchain extent from the drawable pixel size and the surface limits.
//  - ParseResizeScript: KYTY_TEST_RESIZE='t1:WxH,t2:max,t3:fs,t4:restore' (test hook).

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <string_view>
#include <vector>

namespace Libs::Graphics::PresentGeometry {

struct Rect {
	int32_t                      x                                      = 0;
	int32_t                      y                                      = 0;
	uint32_t                     w                                      = 0;
	uint32_t                     h                                      = 0;
	[[nodiscard]] constexpr bool operator==(const Rect&) const noexcept = default;
};

// Largest rectangle with the source aspect that fits in dst, centred. Rounds to nearest. A result
// within one pixel of the full target on both axes snaps to the full target (exact blit, no
// one-pixel black edge). Zero sizes give the full target.
[[nodiscard]] constexpr Rect Letterbox(uint32_t src_w, uint32_t src_h, uint32_t dst_w,
                                       uint32_t dst_h) noexcept {
	if (dst_w == 0 || dst_h == 0 || src_w == 0 || src_h == 0) {
		return {0, 0, dst_w, dst_h};
	}
	uint64_t w = dst_w;
	uint64_t h = dst_h;
	// compare src_w/src_h with dst_w/dst_h without division
	if (static_cast<uint64_t>(src_w) * dst_h >= static_cast<uint64_t>(src_h) * dst_w) {
		h = (static_cast<uint64_t>(dst_w) * src_h + src_w / 2) /
		    src_w; // wider than target: bars top/bottom
	} else {
		w = (static_cast<uint64_t>(dst_h) * src_w + src_h / 2) /
		    src_h; // taller than target: bars left/right
	}
	w = std::clamp<uint64_t>(w, 1, dst_w);
	h = std::clamp<uint64_t>(h, 1, dst_h);
	if (dst_w - w <= 1 && dst_h - h <= 1) {
		return {0, 0, dst_w, dst_h};
	}
	return {static_cast<int32_t>((dst_w - w) / 2), static_cast<int32_t>((dst_h - h) / 2),
	        static_cast<uint32_t>(w), static_cast<uint32_t>(h)};
}

[[nodiscard]] constexpr bool IsFull(const Rect& r, uint32_t dst_w, uint32_t dst_h) noexcept {
	return r.x == 0 && r.y == 0 && r.w == dst_w && r.h == dst_h;
}

struct Extent {
	uint32_t                     w                                        = 0;
	uint32_t                     h                                        = 0;
	[[nodiscard]] constexpr bool operator==(const Extent&) const noexcept = default;
};

// Swapchain extent for a drawable of the given pixel size: clamped into the surface's
// [min, max] image extent. A surface that reports "any size" passes 0xFFFFFFFF as current.
[[nodiscard]] constexpr Extent ClampExtent(Extent drawable, Extent current, Extent min_extent,
                                           Extent max_extent) noexcept {
	if (current.w != 0xFFFFFFFFu && current.h != 0xFFFFFFFFu) {
		return current; // the surface dictates the size
	}
	const uint32_t max_w = max_extent.w == 0 ? 0xFFFFFFFFu : max_extent.w;
	const uint32_t max_h = max_extent.h == 0 ? 0xFFFFFFFFu : max_extent.h;
	return {std::clamp(drawable.w, min_extent.w, std::max(min_extent.w, max_w)),
	        std::clamp(drawable.h, min_extent.h, std::max(min_extent.h, max_h))};
}

// A swapchain image of this extent in this many images: bytes (4 bytes per pixel), for the
// memory budget check (the 18944 MiB self-limit is never raised; a resize must stay modest).
[[nodiscard]] constexpr uint64_t SwapchainBytes(Extent e, uint32_t image_count) noexcept {
	return static_cast<uint64_t>(e.w) * e.h * 4ull * image_count;
}

// Limits a requested window pixel size so that the swapchain stays below max_bytes (never
// grows past it; keeps the aspect). Returns the input when it fits.
[[nodiscard]] constexpr Extent FitToBudget(Extent want, uint32_t image_count,
                                           uint64_t max_bytes) noexcept {
	if (want.w == 0 || want.h == 0 || SwapchainBytes(want, image_count) <= max_bytes) {
		return want;
	}
	// scale both axes by sqrt(max/needed) using integer search on the width
	uint32_t lo = 1;
	uint32_t hi = want.w;
	while (lo < hi) {
		const uint32_t mid = lo + (hi - lo + 1) / 2;
		const uint64_t h   = std::max<uint64_t>(1, static_cast<uint64_t>(want.h) * mid / want.w);
		if (SwapchainBytes({mid, static_cast<uint32_t>(h)}, image_count) <= max_bytes) {
			lo = mid;
		} else {
			hi = mid - 1;
		}
	}
	return {lo, static_cast<uint32_t>(
	                std::max<uint64_t>(1, static_cast<uint64_t>(want.h) * lo / want.w))};
}

// Backing pixels from window points with a pixel density (HiDPI), rounded to nearest.
[[nodiscard]] constexpr Extent PointsToPixels(Extent points, float density) noexcept {
	return {static_cast<uint32_t>(static_cast<float>(points.w) * density + 0.5f),
	        static_cast<uint32_t>(static_cast<float>(points.h) * density + 0.5f)};
}

// ---- KYTY_TEST_RESIZE ------------------------------------------------------------------------
enum class StepKind : uint8_t { Size, Maximize, Fullscreen, Restore };

struct Step {
	double             at_s = 0; // seconds after the first presented frame
	StepKind           kind = StepKind::Size;
	int32_t            w    = 0; // window points for Size
	int32_t            h    = 0;
	[[nodiscard]] bool operator==(const Step&) const noexcept = default;
};

// 't1:WxH,t2:max,t3:fs,t4:restore' (also 'maximize', 'fullscreen'). Malformed steps are skipped;
// the result is sorted by time (stable).
[[nodiscard]] inline std::vector<Step> ParseResizeScript(std::string_view text) {
	std::vector<Step> steps;
	size_t            pos = 0;
	while (pos <= text.size()) {
		size_t end = text.find(',', pos);
		if (end == std::string_view::npos) {
			end = text.size();
		}
		const std::string item(text.substr(pos, end - pos));
		pos                = end + 1;
		const size_t colon = item.find(':');
		if (colon == std::string::npos || colon == 0) {
			continue;
		}
		char*             stop = nullptr;
		const std::string tstr = item.substr(0, colon);
		const double      t    = std::strtod(tstr.c_str(), &stop);
		if (stop == tstr.c_str() || *stop != '\0' || t < 0.0 || t > 1.0e6) {
			continue;
		}
		const std::string what = item.substr(colon + 1);
		Step              s {};
		s.at_s = t;
		if (what == "max" || what == "maximize") {
			s.kind = StepKind::Maximize;
		} else if (what == "fs" || what == "fullscreen") {
			s.kind = StepKind::Fullscreen;
		} else if (what == "restore") {
			s.kind = StepKind::Restore;
		} else {
			const size_t x = what.find('x');
			if (x == std::string::npos || x == 0) {
				continue;
			}
			char*             e1 = nullptr;
			char*             e2 = nullptr;
			const std::string ws = what.substr(0, x);
			const std::string hs = what.substr(x + 1);
			const long        w  = std::strtol(ws.c_str(), &e1, 10);
			const long        h  = std::strtol(hs.c_str(), &e2, 10);
			if (e1 == ws.c_str() || *e1 != '\0' || e2 == hs.c_str() || *e2 != '\0' || w < 16 ||
			    h < 16 || w > 16384 || h > 16384) {
				continue;
			}
			s.kind = StepKind::Size;
			s.w    = static_cast<int32_t>(w);
			s.h    = static_cast<int32_t>(h);
		}
		steps.push_back(s);
	}
	std::stable_sort(steps.begin(), steps.end(),
	                 [](const Step& a, const Step& b) { return a.at_s < b.at_s; });
	return steps;
}

} // namespace Libs::Graphics::PresentGeometry

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_WINDOW_PRESENTGEOMETRY_H_
