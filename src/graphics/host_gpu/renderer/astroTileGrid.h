#pragma once
// ASTRO BOT light-tile grid for a forced internal resolution (pure helpers; see metalfx.cpp
// AstroTileGridPatch). The light culling pass sizes its grid as ceil(extent / 16) per axis.
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace Libs::Graphics::AstroTileGrid {

constexpr uint32_t TILE_PX = 16;

[[nodiscard]] constexpr uint32_t TilesFor(uint32_t pixels) {
	return (pixels + TILE_PX - 1) / TILE_PX;
}

// spec: "<tx>x<ty>" (explicit) or "auto" (from the MetalFX input size in_w x in_h, 0 = not known yet).
// Returns false when the spec is invalid or (auto) the size is not known yet.
[[nodiscard]] inline bool Resolve(const char* spec, uint32_t in_w, uint32_t in_h, unsigned& tx, unsigned& ty) {
	if (spec == nullptr) {
		return false;
	}
	if (std::strcmp(spec, "auto") == 0) {
		if (in_w == 0 || in_h == 0) {
			return false;
		}
		tx = TilesFor(in_w);
		ty = TilesFor(in_h);
		return true;
	}
	return std::sscanf(spec, "%ux%u", &tx, &ty) == 2 && tx != 0 && ty != 0;
}

} // namespace Libs::Graphics::AstroTileGrid
