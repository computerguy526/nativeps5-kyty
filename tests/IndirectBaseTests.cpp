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

// nps5 indcrash (2026-11-28): a model of the command processor's indirect-draw decisions on a
// synthetic PM4 stream, both halves of the pipeline: the front (CpOpSetBase decode +
// IndirectDrawOp's ClassifyDraw, which decides whether an op is emitted) and the back
// (ExecDrawIndirect / ExecDrawIndirectMulti, which read records on the CPU only when
// ArgsReadable). The DrawPrep path (KYTY_DRAW_PREP_INDIRECT) emits the same op with the slot
// offered: its commit's inner call is the plain single-record path, so both share the back check.
namespace {

constexpr uint32_t kSetBaseDraw    = 0xc0021100u; // SET_BASE, shader_type 0 (header bit 1 clear)
constexpr uint32_t kDrawIndexIndir = 0xc0032500u; // DRAW_INDEX_INDIRECT
constexpr uint32_t kIndexedRecord  = 20u;         // DrawIndexedIndirectArgs
constexpr uint32_t kAutoRecord     = 16u;         // DrawIndirectArgs

enum class Path { Plain, DrawPrep, Run, Multi };

struct Outcome {
  bool emitted = false; // the front emitted an op
  bool read    = false; // the back read records (a CPU read or the native path's bind)
  uint64_t args = 0;
};

// One synthetic stream: SET_BASE (word1, word2) then DRAW_INDEX_INDIRECT (data_offset, initiator 0).
Outcome Simulate(uint32_t set_base_w1, uint32_t set_base_w2, uint32_t data_offset, Path path,
                 uint32_t records = 1) {
  const uint32_t pm4[] = {kSetBaseDraw, 1u, set_base_w1, set_base_w2,
                          kDrawIndexIndir, data_offset, 0u, 0u, 0u};
  State cp;
  cp.Set(0, DecodeSetBaseAddr(pm4[2], pm4[3]));
  Outcome o;
  if (ClassifyDraw(cp.draw_base, pm4[8], pm4[4] == kDrawIndexIndir) != DrawVerdict::Draw) {
    return o;  // IndirectDrawOp returned false: no op
  }
  o.emitted = true;
  o.args    = cp.draw_base + pm4[5];
  // The back: a run/multi checks the whole extent first; every single record (plain, DrawPrep's
  // inner call, a run's replay) checks its own record.
  if (path == Path::Run || path == Path::Multi) {
    if (!ArgsReadable(o.args, static_cast<uint64_t>(records) * kIndexedRecord)) {
      if (path == Path::Multi) return o;
      for (uint32_t i = 0; i < records; i++) {
        o.read |= ArgsReadable(o.args + i * kIndexedRecord, kIndexedRecord);
      }
      return o;
    }
  }
  o.read = ArgsReadable(o.args, kIndexedRecord);
  return o;
}

void IndcrashChecks() {
  const Path paths[] = {Path::Plain, Path::DrawPrep, Path::Run, Path::Multi};
  for (const auto path: paths) {
    // The incident's first draw: SET_BASE 0, data_offset 0: skipped at the front.
    auto o = Simulate(0u, 0u, 0u, path, 3);
    CHECK(!o.emitted && !o.read);
    // The crash: guest 0x20 = a null object's member as the base (passed `base == 0`).
    o = Simulate(0x20u, 0u, 0u, path, 3);
    CHECK(!o.emitted && !o.read);
    // Base 0 with an offset cannot reach the back through the front; if it ever does (another
    // route), the back still does not read 0x20.
    CHECK(!ArgsReadable(0x0ull + 0x20u, kIndexedRecord));
    CHECK(!ArgsReadable(0x8ull, kAutoRecord));
    // A real base draws on every path.
    o = Simulate(0x00123000u, 0x7u, 0x40u, path, 3);
    CHECK(o.emitted && o.read && o.args == 0x700123040ull);
  }
  // The null-page edge, the 8-byte alignment of the decode, wrap-around.
  CHECK(!BaseUsable(0) && !BaseUsable(0x20) && !BaseUsable(kGuestNullPageEnd - 8));
  CHECK(BaseUsable(kGuestNullPageEnd) && BaseUsable(0x400000000ull));
  CHECK(DecodeSetBaseAddr(0x27u, 0x0u) == 0x20u);
  CHECK(DecodeSetBaseAddr(0x00123007u, 0xabcd0007u) == 0x700123000ull);
  CHECK(!ArgsReadable(~0ull - 4u, kIndexedRecord));
  CHECK(ArgsReadable(kGuestNullPageEnd, kIndexedRecord));
  CHECK(ArgsReadable(0x400000000ull, 0));
  // The old verdicts stand: a usable base persists across a reset and draws.
  CHECK(ClassifyDraw(0x20u, 0u, true) == DrawVerdict::SkipNoBase);
  CHECK(ClassifyDraw(0x20u, 1u, false) == DrawVerdict::SkipBadInitiator);
}

} // namespace

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

  IndcrashChecks();

  std::printf("indirect_base_tests: %d checks, %d failed\n", g_checks, g_failed);
  return g_failed == 0 ? 0 : 1;
}
