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
	float    prev_jitter[2] {};
	float    mv_scale[2] {};
	uint32_t word31 = 0; // FSR 3.1 frameIndex (unverified: Fsr3ResetTracker only trusts it once it counts up)
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
	out.prev_jitter[0] = f[18];
	out.prev_jitter[1] = f[19];
	out.word31         = bytes >= 128 ? w[31] : 0;
	out.mv_scale[0] = f[20];
	out.mv_scale[1] = f[21];
	return true;
}

// ---- mfxreact: reactive mask + history reset (pure, unit-tested; metalfx.cpp wires them) ----

// What an FSR3 dispatch image binding looks like to the mask picker.
enum class MaskFormat : uint8_t { Other, R8, RG8, RGBA8 };
struct MaskCandidate {
	uint32_t   binding = 0; // index in the dispatch's image bindings
	uint32_t   width = 0, height = 0;
	MaskFormat format = MaskFormat::Other;
	bool       written = false;
};

// The 8-bit unorm image of the render size the dispatch only reads: FSR3's dilated reactive masks
// (RGBA8 or RG8 in the Accumulate pass; its other 8-bit images are 4K lock status). `override_binding`
// >= 0 forces that binding index (KYTY_METALFX_REACTIVE_IMG). Returns the candidate index or -1; with
// several the best format wins (RGBA8 > RG8 > R8), the first of equals.
[[nodiscard]] inline int PickReactiveMask(const MaskCandidate* c, size_t n, uint32_t in_w, uint32_t in_h,
                                          int override_binding) {
	int best = -1;
	for (size_t i = 0; i < n; i++) {
		if (c[i].format == MaskFormat::Other || c[i].written || c[i].width != in_w || c[i].height != in_h) {
			continue;
		}
		if (override_binding >= 0) {
			if (static_cast<int>(c[i].binding) == override_binding) {
				return static_cast<int>(i);
			}
			continue;
		}
		if (best < 0 || static_cast<int>(c[i].format) > static_cast<int>(c[static_cast<size_t>(best)].format)) {
			best = static_cast<int>(i);
		}
	}
	return best;
}

// Channel the single-channel MetalFX mask takes: 0..3 = r g b a, 4 = max(r, g) (reactive and
// transparency&composition both mean "do not trust history"). R8 sources always use channel 0.
[[nodiscard]] inline uint32_t ReactiveChannel(const char* name, MaskFormat format) {
	if (format == MaskFormat::R8) {
		return 0;
	}
	if (name == nullptr || name[0] == '\0' || std::strcmp(name, "max") == 0) {
		return 4;
	}
	switch (name[0]) {
		case 'r': return 0;
		case 'g': return format == MaskFormat::RG8 || format == MaskFormat::RGBA8 ? 1u : 0u;
		case 'b': return format == MaskFormat::RGBA8 ? 2u : 4u;
		case 'a': return format == MaskFormat::RGBA8 ? 3u : 4u;
		default: return 4;
	}
}

// CPU model of the prep pass in KK (mtl_fx.m fx_react): saturate(channel * gain).
[[nodiscard]] inline float ReactiveValue(const float rgba[4], uint32_t channel, float gain) {
	const float v = channel < 4 ? rgba[channel] : (rgba[0] > rgba[1] ? rgba[0] : rgba[1]);
	const float m = v * gain;
	return m < 0.0f ? 0.0f : m > 1.0f ? 1.0f : m;
}

// History restart from the FSR3 block: the plugin's "previous jitter" (words 18-19) must be the jitter we
// saw on the previous frame, and frameIndex (word 31) must count up by one once it has been seen doing
// so 8 times in a row (an unverified word is never trusted: a wrong guess would reset every frame).
struct Fsr3ResetTracker {
	bool     have_last = false;
	float    last_jitter[2] {};
	uint32_t last_index = 0;
	uint32_t index_run  = 0; // consecutive +1 steps seen
	uint64_t index_breaks = 0, jitter_breaks = 0;

	[[nodiscard]] bool Observe(const Fsr3Frame& f, bool consecutive_frame) {
		bool reset = false;
		if (have_last) {
			if (consecutive_frame) {
				if (std::fabs(f.prev_jitter[0] - last_jitter[0]) > 1e-3f ||
				    std::fabs(f.prev_jitter[1] - last_jitter[1]) > 1e-3f) {
					reset = true;
					jitter_breaks++;
				}
				if (f.word31 == last_index + 1) {
					index_run++;
				} else {
					if (index_run >= 8) {
						reset = true;
						index_breaks++;
					}
					index_run = 0;
				}
			} else {
				index_run = 0;
			}
		}
		have_last      = true;
		last_jitter[0] = f.jitter[0];
		last_jitter[1] = f.jitter[1];
		last_index     = f.word31;
		return reset;
	}
};

} // namespace Libs::Graphics::MetalFx

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_METALFXUE_H_
