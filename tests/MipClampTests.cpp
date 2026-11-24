#include "graphics/host_gpu/renderer/image/mipClamp.h"

#include <cstdio>
#include <cstdlib>

using namespace Libs::Graphics::MipClamp;

static void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "%s\n", message);
		std::exit(1);
	}
}

int main() {
	// SAROS T720: 10 levels on 128x64 -> 8.
	Check(MaxLevels(128, 64, 1) == 8, "128x64 holds 8 levels");
	Check(ClampLevels(10, 128, 64, 1) == 8, "128x64 with 10 levels clamps to 8");
	// Legal counts untouched, for every extent up to 4096 (square, wide, tall, odd).
	for (uint32_t w = 1; w <= 4096; w = w * 2 + (w & 1u)) {
		for (uint32_t h = 1; h <= 4096; h = h * 2 + (h & 1u)) {
			const uint32_t max = MaxLevels(w, h, 1);
			for (uint32_t l = 1; l <= max; l++) {
				Check(ClampLevels(l, w, h, 1) == l, "legal level count changed");
			}
			Check(ClampLevels(max + 1, w, h, 1) == max, "excess level count not clamped");
		}
	}
	Check(ClampLevels(13, 4096, 4096, 1) == 13 && ClampLevels(14, 4096, 4096, 1) == 13,
	      "4096^2 holds exactly 13 levels");
	Check(ClampLevels(1, 1, 1, 1) == 1 && ClampLevels(5, 1, 1, 1) == 1, "1x1 holds 1 level");
	Check(ClampLevels(8, 129, 64, 1) == 8 && ClampLevels(9, 129, 64, 1) == 8,
	      "non power of two uses floor(log2)+1");
	// Block-compressed extents are texel extents; same rule (4x4 -> 3 levels).
	Check(ClampLevels(6, 4, 4, 1) == 3, "4x4 holds 3 levels");
	// 3D: depth counts; 2D array: layers do not.
	Check(ClampLevels(10, 16, 16, 256) == 9, "16x16x256 volume holds 9 levels");
	Check(ClampLevels(10, 16, 16, 1) == 5, "16x16 array/2D (depth 1) holds 5 levels");
	Check(ClampLevels(5, 16, 16, 4) == 5, "volume depth below width changes nothing");
	// Views.
	uint32_t base = 0, count = 10;
	ClampView(8, base, count);
	Check(base == 0 && count == 8, "view 0..10 on 8 levels -> 0..8");
	base = 2; count = 3;
	ClampView(8, base, count);
	Check(base == 2 && count == 3, "legal view unchanged");
	base = 6; count = 4;
	ClampView(8, base, count);
	Check(base == 6 && count == 2, "view tail trimmed to the chain");
	base = 9; count = 2;
	ClampView(8, base, count);
	Check(base == 7 && count == 1, "view past the end maps to the last valid level");
	base = 5; count = 0;
	ClampView(8, base, count);
	Check(count == 1, "view count never zero");
	return 0;
}
