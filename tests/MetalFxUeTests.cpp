// mfxlayer: the FSR 3.1 constant block detector (metalfxUe.h). Vectors: SAROS run t2_d (box 3). Standalone:
//   clang++ -std=c++20 -O2 -Isrc tests/MetalFxUeTests.cpp -o /tmp/mfxue && /tmp/mfxue
// CMake target: metalfx_ue_tests.
#include "graphics/host_gpu/renderer/metalfxUe.h"

#include <cstdio>

using Libs::Graphics::MetalFx::Fsr3Frame;
using Libs::Graphics::MetalFx::ParseFsr3Constants;
using Libs::Graphics::MetalFx::MaskCandidate;
using Libs::Graphics::MetalFx::MaskFormat;
using Libs::Graphics::MetalFx::PickReactiveMask;
using Libs::Graphics::MetalFx::ReactiveChannel;
using Libs::Graphics::MetalFx::ReactiveValue;
using Libs::Graphics::MetalFx::Fsr3ResetTracker;

namespace {
// 862d9f3ce00c9c80, gameplay (words 0-23; the rest of the 144 bytes is zero here)
constexpr uint32_t kBlock[36] = {0x8d3, 0x4f7, 0x8d3, 0x4f7, 0xf00, 0x870, 0xf00, 0x870, 0x8d3, 0x4f7, 0xf00, 0x870,
                                 0xb4000000, 0x41200000, 0x3f4f40f2, 0x3ee937c0, 0x3ed00000, 0x3eb425ee, 0xbdc00000,
                                 0x3c97b440, 0x3f800000, 0x3f800000, 0x3f16999a, 0x3f16a315};
// the 64-byte block of the downsample pass 3074088f69a155d0
constexpr uint32_t kOther[16] = {0x403e44a0, 0x11, 0x2, 0, 0xc2efbe60, 0x1e, 0xc2efa860, 0x1e,
                                 0x800ccf60, 0x10, 0x320, 0x1c8, 0xf00, 0x870, 0xf00, 0x870};
} // namespace

static int g_checks = 0, g_failed = 0;
#define CHECK(c)                                                                                   \
	do {                                                                                           \
		g_checks++;                                                                                \
		if (!(c)) {                                                                                \
			g_failed++;                                                                            \
			std::printf("FAILED line %d: %s\n", __LINE__, #c);                                     \
		}                                                                                          \
	} while (0)

int main() {
	Fsr3Frame f;
	CHECK(ParseFsr3Constants(kBlock, 144, f));
	CHECK(f.in_w == 2259u && f.in_h == 1271u && f.out_w == 3840u && f.out_h == 2160u);
	CHECK(f.max_w == 2259u && f.max_h == 1271u);
	CHECK(f.jitter[0] == 0.40625f && f.jitter[1] > 0.3518f && f.jitter[1] < 0.3519f);
	CHECK(f.mv_scale[0] == 1.0f && f.mv_scale[1] == 1.0f);
	CHECK(!ParseFsr3Constants(kOther, 64, f));
	uint32_t zero[36] {};
	CHECK(!ParseFsr3Constants(zero, 144, f));
	uint32_t bad[36];
	std::memcpy(bad, kBlock, sizeof(bad));
	bad[10] = 2560; // upscale size != max upscale size
	CHECK(!ParseFsr3Constants(bad, 144, f));
	std::memcpy(bad, kBlock, sizeof(bad));
	bad[16] = 0x40400000; // jitter 3 px
	CHECK(!ParseFsr3Constants(bad, 144, f));
	CHECK(!ParseFsr3Constants(kBlock, 64, f));
	CHECK(!ParseFsr3Constants(nullptr, 144, f));
	// mfxreact: previous jitter and the (unverified) frame index word are parsed from the block
	{
		Fsr3Frame g;
		uint32_t  b[36];
		std::memcpy(b, kBlock, sizeof(b));
		b[31] = 77;
		CHECK(ParseFsr3Constants(b, 144, g));
		CHECK(g.prev_jitter[0] == -0.09375f && g.word31 == 77u);
		CHECK(ParseFsr3Constants(b, 96, g) && g.word31 == 0u); // a short block carries no word 31
	}
	// the mask picker: the render-size 8-bit image the dispatch only reads (Accumulate: RGBA8 render size,
	// the 4K R8 lock images and written images are no candidates)
	{
		const MaskCandidate c[] = {{0, 1152, 648, MaskFormat::Other, false}, {2, 1280, 720, MaskFormat::RGBA8, false},
		                           {6, 3840, 2160, MaskFormat::R8, false},   {11, 3840, 2160, MaskFormat::R8, true},
		                           {5, 1280, 720, MaskFormat::R8, true},     {7, 1280, 720, MaskFormat::R8, false}};
		CHECK(PickReactiveMask(c, 6, 1280, 720, -1) == 1); // RGBA8 beats R8
		CHECK(PickReactiveMask(c, 6, 1280, 720, 7) == 5);  // override picks binding 7
		CHECK(PickReactiveMask(c, 6, 1280, 720, 5) == -1); // an override that is written / not a candidate: none
		CHECK(PickReactiveMask(c, 6, 1920, 1080, -1) == -1);
		CHECK(PickReactiveMask(c, 0, 1280, 720, -1) == -1);
		CHECK(PickReactiveMask(nullptr, 0, 1280, 720, -1) == -1);
	}
	// channel choice and the single-channel conversion (the model of KK's fx_react kernel)
	CHECK(ReactiveChannel(nullptr, MaskFormat::RGBA8) == 4u && ReactiveChannel("max", MaskFormat::RG8) == 4u);
	CHECK(ReactiveChannel("r", MaskFormat::RGBA8) == 0u && ReactiveChannel("g", MaskFormat::RGBA8) == 1u);
	CHECK(ReactiveChannel("b", MaskFormat::RGBA8) == 2u && ReactiveChannel("a", MaskFormat::RGBA8) == 3u);
	CHECK(ReactiveChannel("g", MaskFormat::R8) == 0u && ReactiveChannel("a", MaskFormat::RG8) == 4u); // R8 has one channel
	{
		const float px[4] = {0.25f, 0.75f, 0.1f, 0.9f};
		CHECK(ReactiveValue(px, 0, 1.0f) == 0.25f && ReactiveValue(px, 1, 1.0f) == 0.75f);
		CHECK(ReactiveValue(px, 4, 1.0f) == 0.75f);           // max(r, g)
		CHECK(ReactiveValue(px, 3, 2.0f) == 1.0f);            // saturates
		CHECK(ReactiveValue(px, 0, 0.5f) == 0.125f);
		const float neg[4] = {-0.5f, 0, 0, 0};
		CHECK(ReactiveValue(neg, 0, 1.0f) == 0.0f);
	}
	// history restart from the block
	{
		Fsr3ResetTracker t;
		Fsr3Frame        fr;
		fr.jitter[0] = 0.1f; fr.jitter[1] = 0.2f; fr.prev_jitter[0] = 0.0f; fr.prev_jitter[1] = 0.0f; fr.word31 = 5;
		CHECK(!t.Observe(fr, false)); // first frame: nothing to compare
		// consecutive frames with a matching previous jitter and a counting index: no reset; the index is
		// only trusted after 8 steps
		for (uint32_t i = 0; i < 9; i++) {
			Fsr3Frame nx;
			nx.prev_jitter[0] = fr.jitter[0]; nx.prev_jitter[1] = fr.jitter[1];
			nx.jitter[0] = 0.01f * static_cast<float>(i); nx.jitter[1] = -0.02f * static_cast<float>(i);
			nx.word31 = fr.word31 + 1;
			CHECK(!t.Observe(nx, true));
			fr = nx;
		}
		CHECK(t.index_run == 9u);
		Fsr3Frame cut;  // camera cut: the plugin restarts its frame counter
		cut.prev_jitter[0] = fr.jitter[0]; cut.prev_jitter[1] = fr.jitter[1]; cut.word31 = 0;
		CHECK(t.Observe(cut, true));
		CHECK(t.index_breaks == 1u && t.index_run == 0u);
		Fsr3Frame jb;   // previous jitter not the one we saw: a frame was missing
		jb.prev_jitter[0] = 0.5f; jb.prev_jitter[1] = 0.5f; jb.word31 = cut.word31 + 1;
		CHECK(t.Observe(jb, true) && t.jitter_breaks == 1u);
		Fsr3Frame gap;  // not the next flip: the caller resets (p.cut); the tracker just re-bases
		gap.prev_jitter[0] = 9.0f; gap.word31 = 1000;
		CHECK(!t.Observe(gap, false));
	}
	{
		// a word 31 that never counts is never trusted: constant garbage must not reset every frame
		Fsr3ResetTracker t;
		Fsr3Frame        fr;
		fr.word31 = 0x3f800000;
		bool any = false;
		for (int i = 0; i < 50; i++) {
			any = any || t.Observe(fr, true);
		}
		CHECK(!any && t.index_breaks == 0u);
		// a word that counts up but wraps once trusted does reset once
		Fsr3ResetTracker u;
		Fsr3Frame        g;
		for (uint32_t i = 0; i < 12; i++) {
			g.word31 = 100 + i;
			CHECK(!u.Observe(g, true));
		}
		g.word31 = 3;
		CHECK(u.Observe(g, true));
	}
	std::printf("metalfx_ue_tests: %d checks, %d failed\n", g_checks, g_failed);
	return g_failed == 0 ? 0 : 1;
}
