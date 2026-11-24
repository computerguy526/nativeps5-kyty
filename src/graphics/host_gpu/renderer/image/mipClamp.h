#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMAGE_MIPCLAMP_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMAGE_MIPCLAMP_H_

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstdlib>

// nps5 mipclamp: KYTY_CLAMP_MIP_TO_EXTENT (default ON, =0 disables). A guest T# can name more mip
// levels than its extent holds (SAROS T720: 10 levels on 128x64). Metal asserts on such a
// descriptor ("requests 10 mipmap levels, but the dimensions (128,64,1) can only support a maximum
// of 8") and Vulkan forbids it, so no legal texture ever has them. The clamp touches only
// levels > floor(log2(max(w,h,d)))+1, so every legal texture is unchanged bit for bit.
// Applied where the texture Info is derived (BuildTextureDescription) so the image, its views and
// its mip_layout agree.
namespace Libs::Graphics::MipClamp {

[[nodiscard]] inline bool Enabled() noexcept {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_CLAMP_MIP_TO_EXTENT");
		return value == nullptr || value[0] != '0';
	}();
	return enabled;
}

// floor(log2(max(w,h,d)))+1. `depth` is the volume depth (1 for 2D and array textures: array
// layers do not add levels).
[[nodiscard]] constexpr uint32_t MaxLevels(uint32_t width, uint32_t height,
                                           uint32_t depth) noexcept {
	const uint32_t m = std::max({width, height, depth, 1u});
	return static_cast<uint32_t>(std::bit_width(m));
}

[[nodiscard]] constexpr uint32_t ClampLevels(uint32_t levels, uint32_t width, uint32_t height,
                                             uint32_t depth) noexcept {
	return std::min(levels, MaxLevels(width, height, depth));
}

// A view of [base, base+count) mapped into a chain of `max_levels`: a base past the end becomes
// the last valid level, the count shrinks to what remains (at least 1).
constexpr void ClampView(uint32_t max_levels, uint32_t& base, uint32_t& count) noexcept {
	if (max_levels == 0) {
		return;
	}
	base  = std::min(base, max_levels - 1u);
	count = std::max(1u, std::min(count, max_levels - base));
}

} // namespace Libs::Graphics::MipClamp

#endif
