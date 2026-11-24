#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_METALFXUE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_METALFXUE_H_

// nps5 mfxlayer: the constant block every FidelityFX FSR 3.1 upscaler dispatch binds (UE's FSR3
// plugin = the engine's third-party temporal upscaler slot; SAROS PPSA07631, runs t2_d / mp2).
// 32-bit words: 0-1 render size, 2-3 previous render size, 4-5 upscale size, 6-7 previous upscale
// size, 8-9 max render size, 10-11 max upscale size, 12-15 device-to-view depth, 16-17 jitter in
// render pixels, 18-19 previous jitter, 20-21 motion vector scale (1 = the motion texture is in
// UV units), 22-23 downscale factor (render / upscale). Pure: unit-testable without a GPU.
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>

namespace Libs::Graphics::MetalFx {

struct Fsr3Frame {
	uint32_t in_w = 0, in_h = 0, out_w = 0, out_h = 0, max_w = 0, max_h = 0;
	float    jitter[2] {};
	float    mv_scale[2] {};
};

[[nodiscard]] inline bool ParseFsr3Constants(const uint32_t* w, size_t bytes, Fsr3Frame& out) {
	if (w == nullptr || bytes < 96 || bytes > 256) {
		return false;
	}
	for (const int i: {0, 1, 4, 5, 8, 9, 10, 11}) {
		if (w[i] == 0 || w[i] > 8192) {
			return false;
		}
	}
	if (w[0] > w[8] || w[1] > w[9] || w[4] != w[10] || w[5] != w[11] || w[0] > w[4] || w[1] > w[5]) {
		return false;
	}
	float f[24];
	std::memcpy(f, w, sizeof(f));
	if (!std::isfinite(f[16]) || !std::isfinite(f[17]) || std::fabs(f[16]) > 1.0f || std::fabs(f[17]) > 1.0f ||
	    !std::isfinite(f[20]) || !std::isfinite(f[21]) || f[20] == 0.0f || f[21] == 0.0f) {
		return false;
	}
	if (std::fabs(f[22] - static_cast<float>(w[0]) / static_cast<float>(w[4])) > 0.01f ||
	    std::fabs(f[23] - static_cast<float>(w[1]) / static_cast<float>(w[5])) > 0.01f) {
		return false;
	}
	out.in_w  = w[0];
	out.in_h  = w[1];
	out.out_w = w[4];
	out.out_h = w[5];
	out.max_w = w[8];
	out.max_h = w[9];
	out.jitter[0]   = f[16];
	out.jitter[1]   = f[17];
	out.mv_scale[0] = f[20];
	out.mv_scale[1] = f[21];
	return true;
}

} // namespace Libs::Graphics::MetalFx

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_METALFXUE_H_
