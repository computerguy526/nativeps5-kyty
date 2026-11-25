// nps5 indirectfix: unit test of the SET_BASE state and draw verdicts (indirectBase.h). Standalone:
//   clang++ -std=c++20 -O2 -Isrc tests/IndirectBaseTests.cpp -o /tmp/indirect_base_tests && /tmp/indirect_base_tests
// CMake target: indirect_base_tests.
#include "graphics/guest_gpu/command_processor/indirectBase.h"

#include <cstdio>

using namespace Libs::Graphics::IndirectBase;

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
  State cp;
  // Base set in a previous command buffer; the next buffer starts with a CP reset and draws
  // without a SET_BASE: the base persists, the draw is drawn (rc65 regression).
  cp.Set(0, 0x7000123000ull);
  cp.Set(1, 0x7000456000ull);
  cp.OnReset();
  CHECK(cp.draw_base == 0x7000123000ull);
  CHECK(cp.dispatch_base == 0x7000456000ull);
  CHECK(ClassifyDraw(cp.draw_base, 2u, false) == DrawVerdict::Draw);
  CHECK(ClassifyDraw(cp.draw_base, 0u, true) == DrawVerdict::Draw);
  CHECK(ClassifyDraw(cp.draw_base, 0x22u, false) == DrawVerdict::Draw);
  // A second reset and a new SET_BASE in the new buffer replace it.
  cp.OnReset();
  cp.Set(0, 0x7000789000ull);
  CHECK(cp.draw_base == 0x7000789000ull);
  CHECK(cp.dispatch_base == 0x7000456000ull);
  CHECK(cp.set_count == 3 && cp.last_set_type == 0 && cp.last_set_addr == 0x7000789000ull);

  // Base never set (0): skip, no exit; unsupported initiator: skip as well.
  State fresh;
  fresh.OnReset();
  CHECK(ClassifyDraw(fresh.draw_base, 2u, false) == DrawVerdict::SkipNoBase);
  CHECK(ClassifyDraw(fresh.draw_base, 0u, true) == DrawVerdict::SkipNoBase);
  CHECK(ClassifyDraw(0x1000, 1u, false) == DrawVerdict::SkipBadInitiator);
  CHECK(ClassifyDraw(0x1000, 2u, true) == DrawVerdict::SkipBadInitiator);

  // Rate limit: first 8 lines, then powers of two.
  uint64_t logged = 0;
  for (uint64_t i = 1; i <= 1000; i++) logged += ShouldLogSkip(i) ? 1 : 0;
  CHECK(logged == 8 + 6);  // 1..8, then 16..512 (6 powers of two)
  CHECK(ShouldLogSkip(1) && ShouldLogSkip(8) && !ShouldLogSkip(9) && ShouldLogSkip(16));

  std::printf("indirect_base_tests: %d checks, %d failed\n", g_checks, g_failed);
  return g_failed == 0 ? 0 : 1;
}
