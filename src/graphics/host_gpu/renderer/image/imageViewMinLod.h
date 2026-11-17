#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMAGE_IMAGEVIEWMINLOD_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMAGE_IMAGEVIEWMINLOD_H_

#include <cstdint>

namespace Libs::Graphics::ImageViewOps {

// A device without VK_EXT_image_view_min_lod (KosmicKrisp) cannot clamp a view's minimum LOD. The
// whole-level part of the clamp moves into the view instead: the view starts at the clamp level,
// so the host computes every LOD relative to it, which clamps exactly at that level (fetches of
// finer levels are out of range, as with the extension). The fraction of a fractional clamp is
// dropped (trilinear blends are then taken from the whole level). 3D slice views keep their
// range (their depth halves per level) and fall back to no clamp. Returns true when it changed
// the view.
template <class View>
inline bool ApplyMinLodWithoutExtension(View& view, bool slice_view) noexcept {
	if (view.min_lod == 0) {
		return false;
	}
	const uint32_t whole = view.min_lod >> 8u;
	view.min_lod         = 0;
	if (slice_view || whole == 0 || view.level_count == 0) {
		return true;
	}
	const uint32_t shift = whole < view.level_count ? whole : view.level_count - 1u;
	view.base_level += shift;
	view.level_count -= shift;
	return true;
}

} // namespace Libs::Graphics::ImageViewOps

#endif
