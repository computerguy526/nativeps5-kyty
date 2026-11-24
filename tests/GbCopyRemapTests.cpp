// nps5 decalfix: KYTY_GBCOPY_REMAP pair parsing, patch-choice derivation and remap (gbCopyRemap.h), synthetic choices, no GPU.
#include "graphics/host_gpu/renderer/image/gbCopyRemap.h"

#include <cstdio>

using namespace Libs::Graphics;
using namespace Libs::Graphics::GbCopyChoices;

static int fails = 0;
#define CHECK(c)                                                    \
	do {                                                            \
		if (!(c)) {                                                 \
			std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); \
			fails++;                                                \
		}                                                           \
	} while (0)

static uint64_t Size(uint32_t w, uint32_t h) { return uint64_t {w} | (uint64_t {h} << 32u); }

static uint64_t g_sizes[3]  = {Size(1920, 1080), Size(1280, 720), Size(960, 540)};
static uint64_t g_levels[3] = {Size(2432, 1368), Size(1920, 1080), Size(1280, 720)};
static uint64_t g_junk[3]   = {Size(8, 8), Size(1280, 720), Size(960, 540)};
static Value    g_values[2] = {{0, 0, 0x1000, g_sizes}, {1, 0, 0x2000, g_levels}};
static Value    g_float[1]  = {{0, 1, 0x3000, g_sizes}};
static Value    g_bad[1]    = {{0, 0, 0x4000, g_junk}};
static Desc     g_descs[4]  = {
    {"render_scale", "t", "d", "e", 3, 0, 1, nullptr, nullptr, nullptr, g_values},
    {"upscaling", "t", "d", "e", 3, 0, 1, nullptr, nullptr, nullptr, g_values + 1},
    {"other", "t", "d", "e", 3, 0, 1, nullptr, nullptr, nullptr, g_values},
    {"render_scale", "t", "d", "e", 3, 0, 1, nullptr, nullptr, nullptr, g_float},
};
static int g_choice[4] = {1, 1, 2, 1};
static int Count() { return 3; } // render_scale, upscaling, other
static const Desc* DescAt(int i) { return &g_descs[i]; }
static int Get(int i) { return g_choice[i]; }

int main() {
	// Explicit pairs.
	std::vector<GbCopyRemap> p;
	CHECK(ParseGbCopyPairs("1920x1080=1280x720,2432x1368=960x540", p) && p.size() == 2);
	CHECK(p[0].from_w == 1920 && p[0].to_h == 720 && p[1].from_h == 1368 && p[1].to_w == 960);
	std::vector<GbCopyRemap> bad;
	CHECK(!ParseGbCopyPairs("1920x1080=1280x720,garbage", bad) && bad.size() == 1); // stops at the bad pair
	std::vector<GbCopyRemap> up;
	CHECK(!ParseGbCopyPairs("1280x720=1920x1080", up) && up.empty());              // a larger target is refused
	std::vector<GbCopyRemap> zero;
	CHECK(!ParseGbCopyPairs("1920x1080=0x720", zero) && zero.empty());
	// Remap.
	uint32_t w = 1920, h = 1080;
	CHECK(ApplyGbCopyRemap(p, w, h) && w == 1280 && h == 720);
	w = 2432; h = 1368;
	CHECK(ApplyGbCopyRemap(p, w, h) && w == 960 && h == 540);
	w = 1280; h = 720;
	CHECK(!ApplyGbCopyRemap(p, w, h) && w == 1280 && h == 720); // the G-buffer size itself is left alone
	w = 1920; h = 1081;
	CHECK(!ApplyGbCopyRemap(p, w, h));
	// auto: render_scale option 1 (1280x720 of 1920x1080) and upscaling option 1 (1920x1080 of 2432x1368);
	// "other" is ignored, a float value site is skipped.
	std::vector<GbCopyRemap> a;
	DeriveFrom(Count, DescAt, Get, a);
	CHECK(a.size() == 2);
	CHECK(a.size() == 2 && a[0].from_w == 1920 && a[0].from_h == 1080 && a[0].to_w == 1280 && a[0].to_h == 720);
	CHECK(a.size() == 2 && a[1].from_w == 2432 && a[1].to_w == 1920 && a[1].to_h == 1080);
	// The native choice (option 0) gives no pair.
	g_choice[0] = 0; g_choice[1] = 0;
	std::vector<GbCopyRemap> none;
	DeriveFrom(Count, DescAt, Get, none);
	CHECK(none.empty());
	// Option 2 of render_scale: 960x540.
	g_choice[0] = 2;
	std::vector<GbCopyRemap> two;
	DeriveFrom(Count, DescAt, Get, two);
	CHECK(two.size() == 1 && two[0].to_w == 960 && two[0].to_h == 540);
	// A value site with an implausible size (8x8) is refused as a whole; missing API -> off.
	g_descs[0].values = g_bad;
	std::vector<GbCopyRemap> junk;
	DeriveFrom(Count, DescAt, Get, junk);
	CHECK(junk.empty());
	std::vector<GbCopyRemap> noapi;
	DeriveFrom(nullptr, nullptr, nullptr, noapi);
	CHECK(noapi.empty());
	std::printf("gbcopy_remap_test: %s (%d fails)\n", fails == 0 ? "PASS" : "FAIL", fails);
	return fails == 0 ? 0 : 1;
}
