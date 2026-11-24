// uppick4: unit tests of the pure decisions behind the picked upstream fixes 245879c2 (D16 <-> D24/D32 copy) and
// 6e541a47 (MSAA resolve at levels smaller than the image extent). Standalone:
//   clang++ -std=c++20 -O2 -Isrc tests/CopyResolvePlanTests.cpp -o /tmp/copy_resolve_plan_tests && /tmp/copy_resolve_plan_tests
// CMake target: copy_resolve_plan_tests.
#include "graphics/host_gpu/renderer/image/copyResolvePlan.h"

#include <cstdio>

using namespace Libs::Graphics::CopyResolvePlan;

static int g_checks = 0, g_failed = 0;
#define CHECK(c)                                                \
  do {                                                          \
    g_checks++;                                                 \
    if (!(c)) {                                                 \
      g_failed++;                                               \
      std::printf("FAILED %s:%d %s\n", __FILE__, __LINE__, #c); \
    }                                                           \
  } while (0)

int main() {
  // D16 pair matrix over the depth-aspect transfer bytes: 4 = D24/D32 (wide), 2 = native D16, 0 = colour (no aspect).
  const uint32_t kinds[] = {0, 2, 4};
  for (uint32_t s : kinds)
    for (uint32_t d : kinds) {
      const auto p = ClassifyD16Pair(s, d);
      CHECK(p.convert == ((s == 4) != (d == 4)));
      CHECK(p.source_wide == (s == 4));
    }
  CHECK(ClassifyD16Pair(4, 2).convert && ClassifyD16Pair(4, 2).source_wide);   // D32 -> native D16: converts now
  CHECK(ClassifyD16Pair(2, 4).convert && !ClassifyD16Pair(2, 4).source_wide);  // native D16 -> D32
  CHECK(ClassifyD16Pair(4, 0).convert && ClassifyD16Pair(0, 4).convert);       // the pair that converted before
  CHECK(!ClassifyD16Pair(4, 4).convert && !ClassifyD16Pair(2, 0).convert);     // left to the raw copy

  // Resolve extent: fitting extents resolve the image's extent exactly as before.
  CHECK((ResolveExtent({1920, 1080}, {1920, 1080}, {1920, 1080}) == Extent2D{1920, 1080}));
  CHECK((ResolveExtent({960, 540}, {1920, 1080}, {1920, 1080}) == Extent2D{960, 540}));
  // A level smaller than the image (mip of a larger placed image): the common area, no abort.
  CHECK((ResolveExtent({1920, 1080}, {960, 540}, {1920, 1080}) == Extent2D{960, 540}));
  CHECK((ResolveExtent({1920, 1080}, {1920, 1080}, {480, 270}) == Extent2D{480, 270}));
  CHECK((ResolveExtent({1920, 1080}, {960, 1080}, {1920, 270}) == Extent2D{960, 270}));  // per-axis minimum
  CHECK((ResolveExtent({64, 64}, {1, 1}, {1, 1}) == Extent2D{1, 1}));
  // Only one axis too large still takes the common area of both axes.
  CHECK((ResolveExtent({128, 32}, {64, 64}, {64, 64}) == Extent2D{64, 64}));

  std::printf("copy_resolve_plan_tests: %d checks, %d failed\n", g_checks, g_failed);
  return g_failed ? 1 : 0;
}
