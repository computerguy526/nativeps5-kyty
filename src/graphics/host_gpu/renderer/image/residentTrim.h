#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMAGE_RESIDENTTRIM_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMAGE_RESIDENTTRIM_H_

#include <cstdint>

// nps5 kmem: KYTY_TEXTURE_RESIDENT_TRIM=1 (default off): a partially resident texture (Image::
// resident_first > 0, KYTY_TEXTURE_RESIDENT_MIPS) gets a native image that holds only levels
// [native_first, levels) of the guest chain: its level 0 is guest level native_first.
//
// Why: Metal commits a texture's whole storage at the first write to any of its levels (measured:
// 32 unwritten 4096^2 BC7 chains cost 0 MiB of footprint, the same chains with only levels >= 2
// written cost all 683 MiB), and KosmicKrisp has no sparse residency images, so the levels no
// view samples were real memory: 2.9 GiB of the 4.8 GiB of textures in shuriken_ninja_castle.
//
// Why it changes nothing a shader sees: a device without VK_EXT_image_view_min_lod (KosmicKrisp)
// already starts every sampled view at its whole minimum LOD level (imageViewMinLod.h), and
// resident_first is never above that level (TextureCache::RequestedFirstLevel), so every view
// keeps its guest levels and extents; only its base level index in the native image moves.
// With the extension the clamp is absolute to the image's levels, so the trim stays off there.
//
// Everything outside Image keeps guest level numbers; these helpers are the only translation
// (Image::FindView, Image::GetBarriers, Image::Upload, TileManager::ImageTransferEligible).
namespace Libs::Graphics::ResidentTrim {

// The extent of guest level `first` (one axis): the native image's level 0.
[[nodiscard]] constexpr uint32_t LevelExtent(uint32_t extent, uint32_t first) noexcept {
	const uint32_t shifted = first < 32u ? extent >> first : 0u;
	return shifted != 0u ? shifted : 1u;
}

// Levels the native image holds.
[[nodiscard]] constexpr uint32_t NativeLevels(uint32_t guest_levels, uint32_t native_first) noexcept {
	return guest_levels > native_first ? guest_levels - native_first : 0u;
}

// Whether the native image holds guest level `level`.
[[nodiscard]] constexpr bool HasLevel(uint32_t level, uint32_t guest_levels,
                                      uint32_t native_first) noexcept {
	return level >= native_first && level < guest_levels;
}

// The first level a trimmed image may start at for a request of `resident_first`: the request
// itself while it leaves at least one level, 0 (no trim) otherwise.
[[nodiscard]] constexpr uint32_t TrimFirst(uint32_t resident_first, uint32_t guest_levels) noexcept {
	return resident_first < guest_levels ? resident_first : 0u;
}

struct ViewLevels {
	uint32_t base    = 0;     // native base level
	uint32_t count   = 0;     // 0: the view has no level in the native image's range
	bool     clamped = false; // the guest range started below native_first
};

// A view of guest levels [base, base + count) on a native image that starts at native_first.
// A range that starts below native_first is cut to the levels the image holds (the coarser ones;
// at least the image's first level): no such view exists while resident_first covers every
// view's first level, the cut keeps a stray one inside the image.
[[nodiscard]] constexpr ViewLevels NativeView(uint32_t base, uint32_t count, uint32_t guest_levels,
                                              uint32_t native_first) noexcept {
	if (count == 0u || base >= guest_levels || count > guest_levels - base) {
		return {};
	}
	if (base >= native_first) {
		return {base - native_first, count, false};
	}
	const uint32_t cut = native_first - base;
	if (native_first >= guest_levels) {
		return {};
	}
	return {0u, count > cut ? count - cut : 1u, true};
}

// Texels of guest levels [first, levels) over texels of the whole chain, as native bytes scale
// (2D, one layer): what part of a whole-chain allocation a trimmed image needs.
[[nodiscard]] constexpr double KeptFraction(uint32_t width, uint32_t height, uint32_t levels,
                                            uint32_t first) noexcept {
	double all  = 0.0;
	double kept = 0.0;
	for (uint32_t level = 0; level < levels; ++level) {
		const double texels = static_cast<double>(LevelExtent(width, level)) *
		                      static_cast<double>(LevelExtent(height, level));
		all += texels;
		if (level >= first) {
			kept += texels;
		}
	}
	return all > 0.0 ? kept / all : 1.0;
}

// nps5 mipskip: KYTY_TEX_MIP_SKIP=1|2 (needs KYTY_TEXTURE_RESIDENT_TRIM=1): the finest `skip` levels of a large
// read-only sampled 2D texture are never made resident, so its native image starts at guest level `skip`
// (the trim above), whatever the guest's views ask for. Views keep guest numbering: a view whose base level is
// below native_first is cut to the image's first level (NativeView), implicit-LOD sampling stays right
// (normalised coordinates on a half-size image pick level L-1 = guest level L); the picture only loses the
// finest levels. Any non-sampling use (target, storage, depth) makes the whole chain resident again.
// Returns the levels to skip: 0 when the texture is not eligible.
[[nodiscard]] constexpr uint32_t MipSkipLevels(uint32_t requested, uint32_t min_dim, uint32_t width, uint32_t height,
                                               uint32_t levels, uint32_t bytes_per_block, bool block_compressed) noexcept {
	if (requested == 0u || levels < 4u) return 0u;
	const uint32_t longest = width > height ? width : height;
	if (longest < min_dim) return 0u;
	if (!block_compressed && bytes_per_block < 2u) return 0u; // R8 masks and font atlases stay whole
	uint32_t skip = requested;
	if (skip > levels - 3u) skip = levels - 3u;               // at least three levels remain
	while (skip > 0u && ((width >> skip) == 0u || (height >> skip) == 0u)) --skip;
	// Block compressed: the skipped level's extent must still be a whole number of 4x4 blocks.
	if (block_compressed) {
		while (skip > 0u && (((width >> skip) & 3u) != 0u || ((height >> skip) & 3u) != 0u)) --skip;
	}
	return skip;
}

// The first level a sampled view may need once the skip floor applies.
[[nodiscard]] constexpr uint32_t SkipFloor(uint32_t view_first, uint32_t skip, uint32_t guest_levels) noexcept {
	const uint32_t first = view_first > skip ? view_first : skip;
	return first < guest_levels ? first : (guest_levels != 0u ? guest_levels - 1u : 0u);
}

} // namespace Libs::Graphics::ResidentTrim

#endif
