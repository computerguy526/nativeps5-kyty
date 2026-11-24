#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMAGE_COPY_RESOLVE_PLAN_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMAGE_COPY_RESOLVE_PLAN_H_

#include <algorithm>
#include <cstdint>

// Pure decisions of two always-on upstream fixes (Jetsku 245879c2 D16 copy, 6e541a47 MSAA resolve), split out so
// they are testable without a Vulkan device. Shared by TextureCache::CopyD16 and Image::Resolve.
namespace Libs::Graphics::CopyResolvePlan {

// CopyD16: guest D16 is held either as a four-byte host depth aspect (D24/D32, stenciled; `wide`) or as a two-byte
// colour / native-D16-depth image. Only a wide <-> narrow pair converts; wide <-> wide and narrow <-> narrow are
// left to the raw copy.
struct D16Pair {
	bool convert;      // the pair is handled by CopyD16
	bool source_wide;  // the source holds the four-byte depth aspect (the destination is then the narrow image)
};

constexpr D16Pair ClassifyD16Pair(uint32_t source_transfer_bytes, uint32_t destination_transfer_bytes) {
	const bool source_wide      = source_transfer_bytes == sizeof(uint32_t);
	const bool destination_wide = destination_transfer_bytes == sizeof(uint32_t);
	return {source_wide != destination_wide, source_wide};
}

struct Extent2D {
	uint32_t width;
	uint32_t height;
	constexpr bool operator==(const Extent2D&) const = default;
};

// Image::Resolve: the image's own extent when it fits both levels (as before the fix), otherwise the area both
// levels have (this used to abort).
constexpr Extent2D ResolveExtent(Extent2D image, Extent2D source_level, Extent2D destination_level) {
	const bool fits = image.width <= source_level.width && image.height <= source_level.height &&
	                  image.width <= destination_level.width && image.height <= destination_level.height;
	if (fits) {
		return image;
	}
	return {std::min(source_level.width, destination_level.width), std::min(source_level.height, destination_level.height)};
}

} // namespace Libs::Graphics::CopyResolvePlan

#endif
