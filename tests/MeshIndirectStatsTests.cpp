// KYTY_CP_OP_STATS "MeshIndirect 10s" line (meshIndirectStats.h). Standalone:
//   clang++ -std=c++20 -O2 -Isrc tests/MeshIndirectStatsTests.cpp -o /tmp/mist
//   && /tmp/mist
// CMake target: mesh_indirect_stats_tests. Synthetic data, no GPU.
#include "graphics/host_gpu/renderer/meshIndirectStats.h"

#include <cstdio>
#include <cstring>

using namespace MeshIndirectStats;

static int g_checks = 0, g_failed = 0;
#define CHECK(c)                                                               \
  do {                                                                         \
    g_checks++;                                                                \
    if (!(c)) {                                                                \
      g_failed++;                                                              \
      std::printf("FAILED %s:%d %s\n", __FILE__, __LINE__, #c);                \
    }                                                                          \
  } while (0)

static bool Has(const char *s, const char *part) {
  return std::strstr(s, part) != nullptr;
}

int main() {
  char line[512];

  // Delta: moved counters subtract, unmoved ones read 0, a reset reads 0.
  Snapshot before;
  before.draws = 100;
  before.checks = 50;
  before.declined_off = 7;
  before.stale = 3;
  Snapshot now = before;
  now.draws = 160;
  now.checks = 110;
  now.always_empty = 4;
  now.declined_multi = 2;
  Snapshot d = Delta(now, before);
  CHECK(d.draws == 60 && d.checks == 60 && d.always_empty == 4 &&
        d.declined_multi == 2);
  CHECK(d.declined_off == 0 && d.stale == 0 && d.mismatches == 0);
  Snapshot reset;
  d = Delta(reset, before);
  CHECK(d.draws == 0 && d.checks == 0 && d.declined_off == 0);

  // All-zero delta.
  const int len0 = Format(line, sizeof line, Snapshot{}, "off");
  CHECK(len0 == static_cast<int>(std::strlen(line)));
  CHECK(std::strcmp(line, "MeshIndirect 10s: mode=off draws=0 empty=0 "
                          "declined(off=0 multi=0 restart=0 clear=0 "
                          "program=0) verify(checks=0 mismatches=0) "
                          "status(overflow=0 limits=0 index_range=0) "
                          "stale=0") == 0);

  // A clean verify window.
  Snapshot v;
  v.draws = 240;
  v.checks = 239;
  Format(line, sizeof line, v, "verify");
  CHECK(Has(line, "mode=verify draws=240 empty=0"));
  CHECK(Has(line, "verify(checks=239 mismatches=0)"));
  CHECK(Has(line, "status(overflow=0 limits=0 index_range=0)"));

  // Each status bit and the mismatch counter land in their own field.
  Snapshot s;
  s.status_overflow = 1;
  Format(line, sizeof line, s, "on");
  CHECK(Has(line, "status(overflow=1 limits=0 index_range=0)"));
  s = {};
  s.status_limits = 2;
  Format(line, sizeof line, s, "on");
  CHECK(Has(line, "status(overflow=0 limits=2 index_range=0)"));
  s = {};
  s.status_index_range = 3;
  s.mismatches = 5;
  s.stale = 9;
  Format(line, sizeof line, s, "exit");
  CHECK(Has(line, "status(overflow=0 limits=0 index_range=3)"));
  CHECK(Has(line, "mismatches=5") && Has(line, "stale=9") &&
        Has(line, "mode=exit"));

  // Each decline reason.
  Snapshot r;
  r.declined_off = 11;
  Format(line, sizeof line, r, "off");
  CHECK(Has(line, "declined(off=11 multi=0 restart=0 clear=0 program=0)"));
  r = {};
  r.declined_multi = 1;
  Format(line, sizeof line, r, "on");
  CHECK(Has(line, "declined(off=0 multi=1 restart=0 clear=0 program=0)"));
  r = {};
  r.declined_restart = 2;
  Format(line, sizeof line, r, "on");
  CHECK(Has(line, "declined(off=0 multi=0 restart=2 clear=0 program=0)"));
  r = {};
  r.declined_clear = 3;
  Format(line, sizeof line, r, "on");
  CHECK(Has(line, "declined(off=0 multi=0 restart=0 clear=3 program=0)"));
  r = {};
  r.declined_program = 4;
  r.always_empty = 8;
  Format(line, sizeof line, r, "empty");
  CHECK(Has(line, "declined(off=0 multi=0 restart=0 clear=0 program=4)"));
  CHECK(Has(line, "mode=empty draws=0 empty=8"));

  // Large values do not truncate at 64 bit.
  Snapshot big;
  big.draws = 18446744073709551615ull;
  Format(line, sizeof line, big, "on");
  CHECK(Has(line, "draws=18446744073709551615"));

  // Truncation: a small buffer gets a terminated prefix and the clamped length.
  char small[16];
  std::memset(small, 'x', sizeof small);
  const int n = Format(small, sizeof small, v, "verify");
  CHECK(n == 15 && small[15] == '\0' &&
        std::strncmp(small, "MeshIndirect 10", 15) == 0);
  CHECK(Format(small, 0, v, "verify") == 0);
  CHECK(Format(nullptr, 8, v, "verify") == 0);
  char one[1] = {'x'};
  CHECK(Format(one, 1, v, "verify") == 0 && one[0] == '\0');

  // Null mode name is tolerated.
  Format(line, sizeof line, Snapshot{}, nullptr);
  CHECK(Has(line, "mode=?"));

  // Decline counters are process-wide and start at zero in this process.
  auto &dc = GetDeclines();
  CHECK(dc.off.load() == 0 && dc.program.load() == 0);
  dc.multi.fetch_add(2, std::memory_order_relaxed);
  CHECK(GetDeclines().multi.load() == 2);

  std::printf("mesh_indirect_stats_tests: %d checks, %d failed\n", g_checks,
              g_failed);
  return g_failed == 0 ? 0 : 1;
}
