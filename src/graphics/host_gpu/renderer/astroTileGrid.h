#pragma once
// ASTRO BOT light-tile grid for a forced internal resolution (pure helpers; see metalfx.cpp
// AstroTileGridPatch). The light culling pass sizes its grid as ceil(extent / 16) per axis.
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace Libs::Graphics::AstroTileGrid {

constexpr uint32_t TILE_PX = 16;
// The grid of the game's largest DRS level (3840x2160); nothing larger is written into its tables.
constexpr uint32_t MAX_TILES_X = 240;
constexpr uint32_t MAX_TILES_Y = 135;

[[nodiscard]] constexpr uint32_t TilesFor(uint32_t pixels) {
	return (pixels + TILE_PX - 1) / TILE_PX;
}

// spec: "<tx>x<ty>" (explicit) or "auto" (from the MetalFX input size in_w x in_h, 0 = not known yet).
// Returns false when the spec is invalid, (auto) the size is not known yet, or the grid is larger than the
// game's largest level (MAX_TILES_X x MAX_TILES_Y).
[[nodiscard]] inline bool Resolve(const char* spec, uint32_t in_w, uint32_t in_h, unsigned& tx, unsigned& ty) {
	if (spec == nullptr) {
		return false;
	}
	unsigned x = 0;
	unsigned y = 0;
	if (std::strcmp(spec, "auto") == 0) {
		if (in_w == 0 || in_h == 0) {
			return false;
		}
		x = TilesFor(in_w);
		y = TilesFor(in_h);
	} else if (spec[0] < '0' || spec[0] > '9' || std::sscanf(spec, "%ux%u", &x, &y) != 2) {
		return false;
	}
	if (x == 0 || y == 0 || x > MAX_TILES_X || y > MAX_TILES_Y) {
		return false;
	}
	tx = x;
	ty = y;
	return true;
}

} // namespace Libs::Graphics::AstroTileGrid
