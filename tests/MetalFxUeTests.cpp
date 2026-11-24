// mfxlayer: the FSR 3.1 constant block detector (metalfxUe.h). Vectors: SAROS
// run t2_d (box 3). Standalone:
//   clang++ -std=c++20 -O2 -Isrc tests/MetalFxUeTests.cpp -o /tmp/mfxue &&
//   /tmp/mfxue
// CMake target: metalfx_ue_tests.
#include "graphics/host_gpu/renderer/metalfxUe.h"

#include <cstdio>

using Libs::Graphics::MetalFx::Fsr3Frame;
using Libs::Graphics::MetalFx::ParseFsr3Constants;

namespace {
// 862d9f3ce00c9c80, gameplay (words 0-23; the rest of the 144 bytes is zero
// here)
constexpr uint32_t kBlock[36] = {
    0x8d3,      0x4f7,      0x8d3,      0x4f7,      0xf00,      0x870,
    0xf00,      0x870,      0x8d3,      0x4f7,      0xf00,      0x870,
    0xb4000000, 0x41200000, 0x3f4f40f2, 0x3ee937c0, 0x3ed00000, 0x3eb425ee,
    0xbdc00000, 0x3c97b440, 0x3f800000, 0x3f800000, 0x3f16999a, 0x3f16a315};
// the 64-byte block of the downsample pass 3074088f69a155d0
constexpr uint32_t kOther[16] = {
    0x403e44a0, 0x11, 0x2,   0,     0xc2efbe60, 0x1e,  0xc2efa860, 0x1e,
    0x800ccf60, 0x10, 0x320, 0x1c8, 0xf00,      0x870, 0xf00,      0x870};
} // namespace

static int g_checks = 0, g_failed = 0;
#define CHECK(c)                                                               \
  do {                                                                         \
    g_checks++;                                                                \
    if (!(c)) {                                                                \
      g_failed++;                                                              \
      std::printf("FAILED line %d: %s\n", __LINE__, #c);                       \
    }                                                                          \
  } while (0)

int main() {
  Fsr3Frame f;
  CHECK(ParseFsr3Constants(kBlock, 144, f));
  CHECK(f.in_w == 2259u && f.in_h == 1271u && f.out_w == 3840u &&
        f.out_h == 2160u);
  CHECK(f.max_w == 2259u && f.max_h == 1271u);
  CHECK(f.jitter[0] == 0.40625f && f.jitter[1] > 0.3518f &&
        f.jitter[1] < 0.3519f);
  CHECK(f.mv_scale[0] == 1.0f && f.mv_scale[1] == 1.0f);
  CHECK(!ParseFsr3Constants(kOther, 64, f));
  uint32_t zero[36]{};
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
  std::printf("metalfx_ue_tests: %d checks, %d failed\n", g_checks, g_failed);
  return g_failed == 0 ? 0 : 1;
}
