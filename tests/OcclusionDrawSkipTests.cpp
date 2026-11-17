#include "graphics/host_gpu/renderer/occlusionDrawSkip.h"

#include <cstdio>

using namespace Libs::Graphics::OcclusionDrawSkip;

static int g_failed = 0;
#define CHECK(cond)                                                       \
	do {                                                                  \
		if (!(cond)) {                                                    \
			std::printf("FAILED %s:%d: %s\n", __FILE__, __LINE__, #cond); \
			g_failed++;                                                   \
		}                                                                 \
	} while (0)

int main() {
	// A depth-test-only box with nothing written: the only skippable draw.
	CHECK(Classify({}) == Verdict::Skip);
	// Each write, alone, keeps the draw.
	CHECK(Classify({.occlusion_accurate = true}) == Verdict::Accurate);
	CHECK(Classify({.mesh = true}) == Verdict::Mesh);
	CHECK(Classify({.color_count = 1}) == Verdict::Color);
	CHECK(Classify({.color_count = 8}) == Verdict::Color);
	CHECK(Classify({.depth_write = true}) == Verdict::Depth);
	CHECK(Classify({.depth_clear = true}) == Verdict::Depth);
	CHECK(Classify({.stencil_write = true}) == Verdict::Stencil);
	CHECK(Classify({.stencil_clear = true}) == Verdict::Stencil);
	CHECK(Classify({.shader_writes = true}) == Verdict::ShaderWrites);
	// Exhaustive: of all 2^7 combinations (colour 0/1) exactly the all-false one skips.
	int skipped = 0;
	for (int m = 0; m < 128; m++) {
		DrawFacts f;
		f.occlusion_accurate = m & 1;
		f.mesh               = m & 2;
		f.color_count        = (m & 4) ? 1 : 0;
		f.depth_write        = m & 8;
		f.depth_clear        = m & 16;
		f.stencil_write      = (m & 32);
		f.shader_writes      = m & 64;
		const auto v = Classify(f);
		CHECK((v == Verdict::Skip) == (m == 0));
		skipped += v == Verdict::Skip;
	}
	CHECK(skipped == 1);
	// Accurate occlusion always wins (counts samples), whatever else is set.
	CHECK(Classify({.occlusion_accurate = true, .shader_writes = true}) == Verdict::Accurate);
	// Default off.
	CHECK(!Enabled());
	std::printf(g_failed ? "FAIL (%d)\n" : "PASS\n", g_failed);
	return g_failed ? 1 : 0;
}
