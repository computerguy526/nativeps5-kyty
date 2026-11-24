// nps5 onereg: unit test of the KYTY_PREDICATION_NO_DRAIN decision (predicationDrain.h). Standalone:
//   clang++ -std=c++20 -O2 -Isrc tests/PredicationDrainTests.cpp -o /tmp/predication_drain_tests && /tmp/predication_drain_tests
// CMake target: predication_drain_tests.
#include "graphics/guest_gpu/command_processor/predicationDrain.h"

#include <cstdio>

using namespace Libs::Graphics::PredicationDrain;

static int g_checks = 0, g_failed = 0;
#define CHECK(c)                                                  \
  do {                                                            \
    g_checks++;                                                   \
    if (!(c)) {                                                   \
      g_failed++;                                                 \
      std::printf("FAILED %s:%d %s\n", __FILE__, __LINE__, #c);   \
    }                                                             \
  } while (0)

int main() {
  // Mode parsing: only "1" and "2" switch it on.
  CHECK(ParseMode(nullptr) == 0);
  CHECK(ParseMode("") == 0);
  CHECK(ParseMode("0") == 0);
  CHECK(ParseMode("1") == 1);
  CHECK(ParseMode("2") == 2);
  CHECK(ParseMode("3") == 0);
  CHECK(ParseMode("on") == 0);
  CHECK(ParseMode("12") == 0);

  // No wait selector: never drains, whatever the mode.
  for (int mode = 0; mode <= 2; mode++)
    for (int l = 0; l < 2; l++)
      for (int c = 0; c < 2; c++) CHECK(!ShouldDrain(mode, false, l != 0, c != 0));

  // Mode 0: always drains when waiting.
  for (int l = 0; l < 2; l++)
    for (int c = 0; c < 2; c++) CHECK(ShouldDrain(0, true, l != 0, c != 0));

  // Mode 1: never drains.
  for (int l = 0; l < 2; l++)
    for (int c = 0; c < 2; c++) CHECK(!ShouldDrain(1, true, l != 0, c != 0));

  // Mode 2: drains exactly when a pending writer can overlap (deferred label or not clean).
  CHECK(!ShouldDrain(2, true, false, true));  // clean, no label: read without the drain
  CHECK(ShouldDrain(2, true, true, true));    // deferred label covers the predicate
  CHECK(ShouldDrain(2, true, true, false));   // label and dirty
  CHECK(ShouldDrain(2, true, false, false));  // GPU-dirty / pending publication

  // Unknown mode values behave as mode 0 (the safe full drain).
  CHECK(ShouldDrain(7, true, false, true));
  CHECK(ShouldDrain(-1, true, false, true));

  std::printf("predication_drain_tests: %d checks, %d failed\n", g_checks, g_failed);
  return g_failed == 0 ? 0 : 1;
}
