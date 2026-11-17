#include "graphics/presentation/window/presentGeometry.h"

#include <cstdio>
#include <random>

using namespace Libs::Graphics::PresentGeometry;

static int g_failed = 0;
static int g_checks = 0;
#define CHECK(cond)                                                          \
	do {                                                                     \
		g_checks++;                                                          \
		if (!(cond)) {                                                       \
			std::printf("FAILED %s:%d: %s\n", __FILE__, __LINE__, #cond);    \
			g_failed++;                                                      \
		}                                                                    \
	} while (0)

int main() {
	// ---- Letterbox: known cases
	CHECK((Letterbox(1920, 1080, 1920, 1080) == Rect {0, 0, 1920, 1080}));
	CHECK((Letterbox(1920, 1080, 960, 540) == Rect {0, 0, 960, 540}));
	CHECK((Letterbox(1920, 1080, 1000, 1000) == Rect {0, 218, 1000, 563})); // pillar-free, bars top/bottom
	CHECK((Letterbox(1920, 1080, 2560, 1600) == Rect {0, 80, 2560, 1440}));
	CHECK((Letterbox(1920, 1080, 800, 1200) == Rect {0, 375, 800, 450}));
	CHECK((Letterbox(1920, 1080, 3440, 1440) == Rect {440, 0, 2560, 1440})); // ultrawide: bars left/right
	CHECK((Letterbox(1920, 1080, 1000, 1001).w == 1000));
	CHECK((Letterbox(1920, 1080, 1, 1) == Rect {0, 0, 1, 1}));
	// one-pixel slack snaps to the full target (exact blit, no sliver)
	CHECK((Letterbox(1920, 1080, 1281, 721) == Rect {0, 0, 1281, 721}));
	CHECK((Letterbox(1920, 1080, 996, 560) == Rect {0, 0, 996, 560}));
	// degenerate inputs give the full target
	CHECK((Letterbox(0, 1080, 100, 50) == Rect {0, 0, 100, 50}));
	CHECK((Letterbox(1920, 0, 100, 50) == Rect {0, 0, 100, 50}));
	CHECK((Letterbox(1920, 1080, 0, 50) == Rect {0, 0, 0, 50}));
	CHECK((Letterbox(1920, 1080, 100, 0) == Rect {0, 0, 100, 0}));
	CHECK(IsFull({0, 0, 5, 6}, 5, 6) && !IsFull({1, 0, 4, 6}, 5, 6));

	// ---- Letterbox: invariants over random inputs (fits, centred, aspect kept to 1 px, touches 2 edges)
	std::mt19937_64 rng(12345);
	for (int i = 0; i < 400000; i++) {
		const uint32_t sw = 16 + static_cast<uint32_t>(rng() % 8000);
		const uint32_t sh = 16 + static_cast<uint32_t>(rng() % 5000);
		const uint32_t dw = 1 + static_cast<uint32_t>(rng() % 9000);
		const uint32_t dh = 1 + static_cast<uint32_t>(rng() % 6000);
		const Rect     r  = Letterbox(sw, sh, dw, dh);
		CHECK(r.x >= 0 && r.y >= 0 && r.w >= 1 && r.h >= 1);
		CHECK(static_cast<uint64_t>(r.x) + r.w <= dw && static_cast<uint64_t>(r.y) + r.h <= dh);
		CHECK(r.w == dw || r.h == dh); // one axis always fills
		// centred within one pixel
		CHECK(std::abs(r.x - static_cast<int64_t>(dw - r.w - r.x)) <= 1);
		CHECK(std::abs(r.y - static_cast<int64_t>(dh - r.h - r.y)) <= 1);
		// aspect: |r.w*sh - r.h*sw| <= max(sw, sh) (one pixel of rounding on the free axis), unless snapped (<= 1 px off)
		const int64_t err = std::abs(static_cast<int64_t>(r.w) * sh - static_cast<int64_t>(r.h) * sw);
		CHECK(err <= static_cast<int64_t>(std::max(sw, sh)) * 2);
		// never larger than a full-fit
		CHECK(IsFull(r, dw, dh) == (r.w == dw && r.h == dh));
	}
	// same aspect at any scale is exactly full
	for (uint32_t k = 1; k <= 200; k++) {
		CHECK(IsFull(Letterbox(16, 9, 16 * k, 9 * k), 16 * k, 9 * k));
		CHECK(IsFull(Letterbox(1920, 1080, 16 * k, 9 * k), 16 * k, 9 * k));
	}

	// ---- ClampExtent
	CHECK((ClampExtent({800, 600}, {0xFFFFFFFFu, 0xFFFFFFFFu}, {1, 1}, {4096, 4096}) == Extent {800, 600}));
	CHECK((ClampExtent({9000, 600}, {0xFFFFFFFFu, 0xFFFFFFFFu}, {1, 1}, {4096, 4096}) == Extent {4096, 600}));
	CHECK((ClampExtent({0, 0}, {0xFFFFFFFFu, 0xFFFFFFFFu}, {1, 1}, {4096, 4096}) == Extent {1, 1}));
	CHECK((ClampExtent({800, 600}, {1000, 700}, {1, 1}, {4096, 4096}) == Extent {1000, 700})); // surface dictates
	CHECK((ClampExtent({800, 600}, {0xFFFFFFFFu, 0xFFFFFFFFu}, {1, 1}, {0, 0}) == Extent {800, 600})); // max unset
	CHECK((ClampExtent({800, 600}, {0xFFFFFFFFu, 0xFFFFFFFFu}, {2000, 2000}, {1000, 1000}) == Extent {2000, 2000})); // min wins

	// ---- memory budget
	CHECK(SwapchainBytes({2560, 1600}, 3) == 2560ull * 1600 * 4 * 3);
	CHECK((FitToBudget({1000, 500}, 3, 1ull << 30) == Extent {1000, 500}));
	CHECK((FitToBudget({0, 500}, 3, 10) == Extent {0, 500}));
	for (int i = 0; i < 20000; i++) {
		const Extent   e {16 + static_cast<uint32_t>(rng() % 16000), 16 + static_cast<uint32_t>(rng() % 9000)};
		const uint32_t n = 2 + static_cast<uint32_t>(rng() % 4);
		const uint64_t budget = 1ull << (22 + rng() % 8);
		const Extent   f = FitToBudget(e, n, budget);
		CHECK(f.w >= 1 && f.h >= 1 && f.w <= e.w && f.h <= e.h);
		if (SwapchainBytes(e, n) <= budget) {
			CHECK(f == e);
		} else if (budget >= SwapchainBytes({16, 16}, n)) {
			CHECK(SwapchainBytes(f, n) <= budget);
			// aspect kept within rounding
			CHECK(std::abs(static_cast<int64_t>(f.w) * e.h - static_cast<int64_t>(f.h) * e.w) <= static_cast<int64_t>(e.w) + e.h);
			// and close to the budget (not pessimistic by more than one width step)
			const Extent bigger {f.w + 1, static_cast<uint32_t>(static_cast<uint64_t>(e.h) * (f.w + 1) / e.w)};
			CHECK(f.w == e.w || SwapchainBytes(bigger, n) > budget);
		}
	}

	// ---- HiDPI
	CHECK((PointsToPixels({498, 280}, 2.0f) == Extent {996, 560}));
	CHECK((PointsToPixels({498, 280}, 1.0f) == Extent {498, 280}));
	CHECK((PointsToPixels({1000, 563}, 1.5f) == Extent {1500, 845}));

	// ---- KYTY_TEST_RESIZE parser
	{
		const auto s = ParseResizeScript("20:1280x720,40:max,60:fs,80:restore");
		CHECK(s.size() == 4);
		CHECK((s[0] == Step {20, StepKind::Size, 1280, 720}));
		CHECK(s[1].kind == StepKind::Maximize && s[1].at_s == 40);
		CHECK(s[2].kind == StepKind::Fullscreen);
		CHECK(s[3].kind == StepKind::Restore && s[3].at_s == 80);
	}
	{
		const auto s = ParseResizeScript("t1:100x100,5:maximize,2.5:fullscreen,9:bogus,x:max,3:0x0,4:99999x10,,7:12x,8:widthxheight,6:700x400");
		// kept: 5 max, 2.5 fs, 6 700x400 (sorted by time)
		CHECK(s.size() == 3);
		if (s.size() == 3) {
			CHECK(s[0].at_s == 2.5 && s[0].kind == StepKind::Fullscreen);
			CHECK(s[1].at_s == 5 && s[1].kind == StepKind::Maximize);
			CHECK(s[2].at_s == 6 && s[2].kind == StepKind::Size && s[2].w == 700 && s[2].h == 400);
		}
	}
	CHECK(ParseResizeScript("").empty());
	CHECK(ParseResizeScript(",,,").empty());
	CHECK(ParseResizeScript("5").empty());
	CHECK(ParseResizeScript(":max").empty());
	CHECK(ParseResizeScript("-1:max").empty());
	{
		const auto s = ParseResizeScript("5:fs,5:max,5:restore"); // equal times keep script order
		CHECK(s.size() == 3 && s[0].kind == StepKind::Fullscreen && s[1].kind == StepKind::Maximize &&
		      s[2].kind == StepKind::Restore);
	}

	std::printf("presentgeometry: %d checks, %d failed\n", g_checks, g_failed);
	return g_failed == 0 ? 0 : 1;
}
