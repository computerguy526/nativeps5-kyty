// KYTY_ALIAS_REBUILD_PARTIAL plan
// (graphics/host_gpu/renderer/image/ownedRebuildPlan.h).
#include "graphics/host_gpu/renderer/image/imageCoherence.h"
#include "graphics/host_gpu/renderer/image/ownedRebuildPlan.h"

#include <cstdio>
#include <cstdlib>
#include <set>
#include <utility>
#include <vector>

namespace {
namespace R = Libs::Graphics::OwnedRebuild;

void Check(bool c, const char *m) {
  if (!c) {
    std::fprintf(stderr, "OwnedRebuildPlanTests: %s\n", m);
    std::abort();
  }
}

// gpu_tiler_render_target.comp, ELEMENT_BYTES = 8 (block 128 x 64), and
// gpu_tiler_common.inc's address of element p of a tile whose first block is at
// tiled_base with blocks_per_row columns.
uint32_t RtOffset8(uint32_t x, uint32_t y) {
  return ((y << 4) & 0x010u) ^ ((y << 6) & 0x080u) ^ ((y << 5) & 0xf00u) ^
         ((y << 10) & 0x5000u) ^ ((x << 3) & 0x008u) ^ ((x << 4) & 0x460u) ^
         ((x << 5) & 0x300u) ^ ((x << 6) & 0x800u) ^ ((x << 10) & 0x2000u) ^
         ((x << 9) & 0x8000u);
}
// All rt_offset variants (same order as the shader's switch) and their block
// extents.
uint32_t RtOffset(uint32_t e, uint32_t x, uint32_t y) {
  switch (e) {
  case 1:
    return ((y << 2) & 0x008u) ^ ((y << 4) & 0x010u) ^ ((y << 3) & 0x0a0u) ^
           ((y << 5) & 0xf00u) ^ ((y << 6) & 0x1000u) ^ ((y << 7) & 0x4000u) ^
           (x & 7u) ^ ((x << 3) & 0x040u) ^ ((x << 5) & 0x300u) ^
           ((x << 4) & 0x400u) ^ ((x << 6) & 0x800u) ^ ((x << 7) & 0x2000u) ^
           ((x << 8) & 0x8000u);
  case 2:
    return ((y << 4) & 0x070u) ^ ((y << 5) & 0xf00u) ^ ((y << 8) & 0x5000u) ^
           ((x << 1) & 0x00eu) ^ ((x << 4) & 0x480u) ^ ((x << 5) & 0x300u) ^
           ((x << 6) & 0x800u) ^ ((x << 7) & 0x2000u) ^ ((x << 8) & 0x8000u);
  case 4:
    return ((y << 4) & 0x070u) ^ ((y << 5) & 0xf00u) ^ ((y << 9) & 0x1000u) ^
           ((y << 8) & 0x4000u) ^ ((x << 2) & 0x00cu) ^ ((x << 5) & 0x380u) ^
           ((x << 4) & 0x400u) ^ ((x << 6) & 0x800u) ^ ((x << 9) & 0xa000u);
  case 8:
    return RtOffset8(x, y);
  default:
    return ((x << 4) & 0x410u) ^ ((x << 5) & 0x340u) ^ ((x << 6) & 0x800u) ^
           ((x << 11) & 0xa000u) ^ ((y << 5) & 0xf20u) ^ ((y << 6) & 0x080u) ^
           ((y << 10) & 0x1000u) ^ ((y << 11) & 0x4000u);
  }
}

// RowAlign matches the shader: the swizzle depends only on (x mod block width,
// y mod (block height * RowAlign)), and with RowAlign 1 where that is enough.
void TestRowAlign() {
  for (uint32_t e : {1u, 2u, 4u, 8u, 16u}) {
    const uint32_t bw = e <= 2 ? 256u : (e <= 8 ? 128u : 64u);
    const uint32_t bh = 65536u / (bw * e);
    const uint32_t ry = R::RowAlign(e), rx = R::ColumnAlign(e);
    bool row_parity = false, column_parity = false;
    for (uint32_t y = 0; y < bh * 4; ++y) {
      for (uint32_t x = 0; x < bw * 4; ++x) {
        Check(RtOffset(e, x, y) == RtOffset(e, x % (bw * rx), y % (bh * ry)),
              "swizzle period");
        Check(RtOffset(e, x, y) < 65536u, "inside a block");
        row_parity = row_parity || RtOffset(e, x, y) != RtOffset(e, x, y % bh);
        column_parity =
            column_parity || RtOffset(e, x, y) != RtOffset(e, x % bw, y);
      }
    }
    Check(row_parity == (ry == 2) && column_parity == (rx == 2),
          "aligns are the smallest periods");
  }
}

// 2432 x 1368, 8 bytes per element: 19 columns x 22 rows of 64 KiB blocks.
R::Surface Surface2432() {
  R::Surface s;
  s.base = 0x5397e0000ull;
  s.width = 2432;
  s.height = 1368;
  s.block_width = 128;
  s.block_height = 64;
  s.block_size = 65536;
  s.columns = 19;
  s.rows = 22;
  return s;
}

// Owned bytes [owned_begin, owned_end) relative to base.
auto RangeOwner(const R::Surface &s, uint64_t owned_begin, uint64_t owned_end) {
  return [=](uint64_t address, uint64_t size) {
    const uint64_t b = address - s.base, e = b + size;
    if (b >= owned_begin && e <= owned_end)
      return R::BlockState::Owned;
    if (e <= owned_begin || b >= owned_end)
      return R::BlockState::NotOwned;
    return R::BlockState::Mixed;
  };
}

void TestAllOwned() {
  const auto s = Surface2432();
  R::Plan p;
  Check(R::Build(s, RangeOwner(s, 0, 19ull * 22 * 65536), p, 2),
        "all owned plans");
  Check(p.segments.empty() && p.refresh_blocks == 0, "all owned: nothing");
}

void TestNoneOwned() {
  const auto s = Surface2432();
  R::Plan p;
  Check(
      R::Build(
          s, [](uint64_t, uint64_t) { return R::BlockState::NotOwned; }, p, 2),
      "none owned");
  Check(p.segments.size() == 1 && p.segments[0].row_count == 22 &&
            p.segments[0].column_count == 19,
        "none owned: one whole-surface segment");
  Check(p.refresh_blocks == 19u * 22u && p.preserve_blocks == 0,
        "none owned: every block");
}

// A smaller alias at the same base (1280x720 RGBA16F = 7.4 MB) took the first
// bytes: the first 112 blocks = 5 whole rows + 17 columns of row 5.
void TestPrefixTaken() {
  const auto s = Surface2432();
  const uint64_t taken = 112ull * 65536;
  R::Plan p;
  // Row align 1 (4-byte texels): whole rows 0..4, then columns 0..16 of row 5.
  Check(R::Build(s, RangeOwner(s, taken, 19ull * 22 * 65536), p, 1),
        "prefix plans");
  Check(p.segments.size() == 2, "prefix: two segments");
  Check(p.segments[0].first_row == 0 && p.segments[0].row_count == 5 &&
            p.segments[0].column_count == 19,
        "prefix: whole rows 0..4");
  Check(p.segments[1].first_row == 5 && p.segments[1].row_count == 1 &&
            p.segments[1].first_column == 0 && p.segments[1].column_count == 17,
        "prefix: 17 columns of row 5");
  Check(p.refresh_blocks == 112 && p.preserve_blocks == 0, "prefix counts");
  Check(R::SegmentOffset(s, p.segments[1]) == 95ull * 65536 &&
            R::SegmentBytes(s, p.segments[1]) == 17ull * 65536,
        "prefix: segment bytes");
  // Row align 2 (8-byte texels): row 5 is odd, so rows 4..5 whole: one segment
  // rows 0..5, the two owned blocks of row 5 preserved.
  Check(R::Build(s, RangeOwner(s, taken, 19ull * 22 * 65536), p, 2),
        "prefix plans (align 2)");
  Check(p.segments.size() == 1 && p.segments[0].first_row == 0 &&
            p.segments[0].row_count == 6 && p.segments[0].column_count == 19,
        "prefix align 2: rows 0..5");
  Check(p.refresh_blocks == 6u * 19u && p.preserve_blocks == 2,
        "prefix align 2 counts");
}

void TestMixed() {
  const auto s = Surface2432();
  R::Plan p;
  // Bytes [1000, end) owned: block 0 is mixed, refreshed with its owned bytes
  // preserved.
  Check(R::Build(s, RangeOwner(s, 1000, 19ull * 22 * 65536), p, 2),
        "mixed plans");
  Check(p.segments.size() == 1 && p.segments[0].first_row == 0 &&
            p.segments[0].column_count == 1 && p.preserve_blocks == 1 &&
            p.refresh_blocks == 1,
        "mixed block refreshed and preserved");
}

void TestNonContiguousRows() {
  const auto s = Surface2432();
  // Rows 2 and 3 not owned, row 7 not owned, columns 4..6 of row 10 not owned.
  auto st = [&](uint64_t address, uint64_t) {
    const uint64_t block = (address - s.base) / 65536;
    const uint64_t row = block / 19, col = block % 19;
    const bool not_owned =
        row == 2 || row == 3 || row == 7 || (row == 10 && col >= 4 && col <= 6);
    return not_owned ? R::BlockState::NotOwned : R::BlockState::Owned;
  };
  R::Plan p;
  Check(R::Build(s, st, p, 1), "rows plan");
  Check(p.segments.size() == 3, "rows: three segments");
  Check(p.segments[0].first_row == 2 && p.segments[0].row_count == 2,
        "rows 2..3 merged");
  Check(p.segments[1].first_row == 7 && p.segments[1].row_count == 1, "row 7");
  Check(p.segments[2].first_row == 10 && p.segments[2].first_column == 4 &&
            p.segments[2].column_count == 3,
        "row 10 columns 4..6");
  Check(R::Build(s, st, p, 2), "rows plan align 2");
  Check(p.segments.size() == 3 && p.segments[1].first_row == 6 &&
            p.segments[1].row_count == 2 && p.preserve_blocks == 19,
        "align 2: row 7 becomes rows 6..7, row 6 preserved");
  for (const auto &g : p.segments) {
    Check(g.first_row % 2 == 0, "align 2: aligned starts");
  }
  // Too many segments: a checkerboard.
  auto checker = [&](uint64_t address, uint64_t) {
    const uint64_t block = (address - s.base) / 65536;
    return (block % 2) != 0 ? R::BlockState::NotOwned : R::BlockState::Owned;
  };
  Check(!R::Build(s, checker, p, 1, 1, 64),
        "checkerboard exceeds the segment cap");
  Check(R::Build(s, checker, p, 1, 1, 1000),
        "checkerboard within a larger cap");
}

void TestClip() {
  // 1280 x 720, 8 bytes: pitch 1280 -> 10 columns, 12 rows (720 / 64 = 11.25),
  // last row partial.
  R::Surface s;
  s.width = 1280;
  s.height = 720;
  s.block_width = 128;
  s.block_height = 64;
  s.block_size = 65536;
  s.columns = 10;
  s.rows = 12;
  const R::Segment last{11, 1, 0, 10};
  const auto r = R::SegmentRect(s, last);
  Check(r.y == 704 && r.height == 16 && r.width == 1280,
        "last row clipped to the height");
  s.width = 2400;
  s.columns = 19;
  Check(R::SegmentRect(s, {0, 1, 18, 1}).width == 2400 - 18 * 128,
        "last column clipped");
  s.width = 2300;
  Check(R::SegmentRect(s, {0, 1, 18, 1}).width == 0, "padding column empty");
  R::Plan p;
  s.columns = 5; // fewer columns than the width needs
  Check(!R::Build(
            s, [](uint64_t, uint64_t) { return R::BlockState::NotOwned; }, p),
        "bad surface");
  // Odd row count with align 2: the last group has one row.
  s.width = 1280;
  s.columns = 10;
  s.rows = 11;
  s.height = 700;
  auto last_row = [&](uint64_t address, uint64_t) {
    return (address - s.base) / 65536 / 10 == 10 ? R::BlockState::NotOwned
                                                 : R::BlockState::Owned;
  };
  Check(R::Build(s, last_row, p, 2) && p.segments.size() == 1 &&
            p.segments[0].first_row == 10 && p.segments[0].row_count == 1,
        "odd last group");
}

// The tiler addresses each segment as its own sub-surface (first block at
// SegmentOffset, columns = column_count, positions relative to the segment):
// every texel of the segment must map to the same tiled byte as in the whole
// surface, and the segments must cover every texel of every block that is not
// wholly owned.
void TestSubSurfaceAddressing() {
  for (uint32_t e : {1u, 2u, 4u, 8u, 16u}) {
    R::Surface s;
    s.base = 0x5397e0000ull;
    s.block_size = 65536;
    s.block_width = e <= 2 ? 256u : (e <= 8 ? 128u : 64u);
    s.block_height = 65536u / (s.block_width * e);
    s.width = 19 * s.block_width - 37;
    s.height = 22 * s.block_height - 9;
    s.columns = 19;
    s.rows = 22;
    auto st = [&](uint64_t address, uint64_t) {
      const uint64_t block = (address - s.base) / 65536;
      const uint64_t row = block / 19, col = block % 19;
      if (row == 13 && col == 2)
        return R::BlockState::Mixed;
      return (row < 5 || (row == 5 && col < 17) || row == 21 ||
              (row == 9 && col >= 3 && col < 11) || (row == 16 && col == 18))
                 ? R::BlockState::NotOwned
                 : R::BlockState::Owned;
    };
    R::Plan p;
    Check(R::Build(s, st, p, R::RowAlign(e), R::ColumnAlign(e)),
          "addressing plan");
    std::set<std::pair<uint32_t, uint32_t>> covered;
    for (const auto &g : p.segments) {
      Check(g.first_row % R::RowAlign(e) == 0 &&
                g.first_column % R::ColumnAlign(e) == 0,
            "aligned segment");
      Check(g.row_count == 1 || g.column_count == s.columns,
            "multi-row segments span all columns");
      const auto r = R::SegmentRect(s, g);
      const uint64_t base = R::SegmentOffset(s, g);
      for (uint32_t y = 0; y < r.height; ++y) {
        for (uint32_t x = 0; x < r.width; ++x) {
          const uint32_t sx = r.x + x, sy = r.y + y;
          const uint64_t sub = base +
                               (uint64_t{y / s.block_height} * g.column_count +
                                x / s.block_width) *
                                   65536 +
                               RtOffset(e, x, y);
          const uint64_t whole =
              (uint64_t{sy / s.block_height} * s.columns + sx / s.block_width) *
                  65536 +
              RtOffset(e, sx, sy);
          Check(sub == whole, "segment texel maps to the surface's tiled byte");
          Check(sub >= base && sub < base + R::SegmentBytes(s, g),
                "inside the segment bytes");
          Check(covered.insert({sx, sy}).second, "texel covered once");
        }
      }
    }
    for (uint32_t y = 0; y < s.height; ++y) {
      for (uint32_t x = 0; x < s.width; ++x) {
        const uint64_t a =
            (uint64_t{y / s.block_height} * s.columns + x / s.block_width) *
            65536;
        if (st(s.base + a, 65536) != R::BlockState::Owned) {
          Check(covered.count({x, y}) != 0,
                "every texel of a not wholly owned block");
        }
      }
    }
  }
}

// KYTY_ALIAS_REBUILD_PRESERVE_TIGHT (PreserveSegments): for random ownership, the parts
// are valid segments inside their segment (aligned, non-overlapping, multi-row
// ones spanning it), they cover every block of the segment that holds owned
// bytes, and tiling the image through the parts writes, in every such block,
// exactly the bytes tiling it through the whole segment writes (simulated on
// element indices: tiled byte -> the texel it receives).
void TestPreserveSegments() {
  uint64_t seed = 0x9e3779b97f4a7c15ull;
  const auto rnd = [&]() {
    seed ^= seed << 13;
    seed ^= seed >> 7;
    seed ^= seed << 17;
    return seed;
  };
  uint64_t parts_total = 0, saved_blocks = 0, kept_blocks = 0;
  for (uint32_t e : {1u, 2u, 4u, 8u, 16u}) {
    for (int round = 0; round < 60; ++round) {
      R::Surface s;
      s.base = 0x5397e0000ull;
      s.block_size = 65536;
      s.block_width = e <= 2 ? 256u : (e <= 8 ? 128u : 64u);
      s.block_height = 65536u / (s.block_width * e);
      s.columns = 1 + rnd() % 7;
      s.rows = 1 + rnd() % 9;
      s.width = s.columns * s.block_width - rnd() % s.block_width;
      s.height = s.rows * s.block_height - rnd() % s.block_height;
      // 0 not owned, 1 mixed (owns some bytes), 2 owned; density varies by round.
      std::vector<uint8_t> own(size_t{s.columns} * s.rows);
      const uint32_t mode = round % 4;
      for (auto &o : own) {
        const uint32_t v = rnd() % 100;
        o = mode == 0   ? (v < 6 ? 1 : v < 12 ? 2 : 0)
            : mode == 1 ? (v < 30 ? 2 : v < 40 ? 1 : 0)
            : mode == 2 ? (v < 3 ? 1 : 0)
                        : (v < 70 ? 2 : v < 80 ? 1 : 0);
      }
      const auto at = [&](uint64_t address) {
        return own[(address - s.base) / 65536];
      };
      R::Plan p;
      Check(R::Build(
                s,
                [&](uint64_t a, uint64_t) {
                  return at(a) == 2   ? R::BlockState::Owned
                         : at(a) == 1 ? R::BlockState::Mixed
                                      : R::BlockState::NotOwned;
                },
                p, R::RowAlign(e), R::ColumnAlign(e), 1u << 20),
            "preserve plan");
      for (const auto &g : p.segments) {
        std::vector<R::Segment> parts;
        R::PreserveSegments(
            s, g, [&](uint64_t a, uint64_t) { return at(a) != 0; },
            R::RowAlign(e), R::ColumnAlign(e), parts);
        std::set<std::pair<uint32_t, uint32_t>> in_parts;
        for (const auto &q : parts) {
          parts_total++;
          Check(q.row_count != 0 && q.column_count != 0, "non-empty part");
          Check(q.first_row % R::RowAlign(e) == 0 &&
                    q.first_column % R::ColumnAlign(e) == 0,
                "aligned part");
          Check(q.row_count == 1 || q.column_count == s.columns,
                "multi-row parts span all columns");
          Check(q.first_row >= g.first_row &&
                    q.first_row + q.row_count <= g.first_row + g.row_count &&
                    q.first_column >= g.first_column &&
                    q.first_column + q.column_count <=
                        g.first_column + g.column_count,
                "part inside its segment");
          for (uint32_t r = 0; r < q.row_count; ++r) {
            for (uint32_t c = 0; c < q.column_count; ++c) {
              Check(in_parts.insert({q.first_row + r, q.first_column + c})
                        .second,
                    "parts do not overlap");
            }
          }
          // Every texel of the part: same tiled byte as through the segment
          // (both equal the whole surface's, TestSubSurfaceAddressing).
          const auto r = R::SegmentRect(s, q);
          const auto gr = R::SegmentRect(s, g);
          const uint64_t qb = R::SegmentOffset(s, q), gb = R::SegmentOffset(s, g);
          for (uint32_t y = 0; y < r.height; y += 1 + rnd() % 3) {
            for (uint32_t x = 0; x < r.width; x += 1 + rnd() % 5) {
              const uint32_t sx = r.x + x, sy = r.y + y;
              const uint32_t gx = sx - gr.x, gy = sy - gr.y;
              const uint64_t via_part =
                  qb + (uint64_t{y / s.block_height} * q.column_count +
                        x / s.block_width) * 65536 + RtOffset(e, x, y);
              const uint64_t via_segment =
                  gb + (uint64_t{gy / s.block_height} * g.column_count +
                        gx / s.block_width) * 65536 + RtOffset(e, gx, gy);
              Check(via_part == via_segment,
                    "part texel maps to the segment's tiled byte");
            }
          }
        }
        for (uint32_t r = 0; r < g.row_count; ++r) {
          for (uint32_t c = 0; c < g.column_count; ++c) {
            const uint32_t row = g.first_row + r, col = g.first_column + c;
            const bool held = own[size_t{row} * s.columns + col] != 0;
            const bool in = in_parts.count({row, col}) != 0;
            Check(!held || in, "every block with owned bytes is in a part");
            saved_blocks += !in;
            kept_blocks += in;
          }
        }
      }
    }
  }
  std::printf("PreserveSegments: %llu parts, blocks kept %llu, not tiled %llu\n",
              static_cast<unsigned long long>(parts_total),
              static_cast<unsigned long long>(kept_blocks),
              static_cast<unsigned long long>(saved_blocks));
  Check(parts_total > 300 && saved_blocks > 300 && kept_blocks > 300,
        "the random cases exercise the parts");
}

// ImageCoherence (KYTY_ALIAS_REBUILD_PARTIAL): the events TextureCache maps
// onto it, in the order a Sky Garden frame produces them, and the block
// decision the rebuild takes from it.
namespace C = Libs::Graphics;
constexpr uint64_t kBase = 0x5397e0000ull, kBlock = 65536,
                   kSize = 19ull * 22 * kBlock;

void TestCoherenceAlgebra() {
  C::ImageCoherence c;
  Check(c.Empty() && !c.Contains(kBase, 1) && c.Contains(kBase, 0),
        "starts empty");
  // Whole refresh of an image owning blocks 0..9 (InitializeImage).
  C::RangeSet owned;
  owned.Add(kBase, 10 * kBlock);
  c.SetAllExcept(kBase, kSize, &owned, false);
  Check(!c.Contains(kBase, kBlock) &&
            c.Contains(kBase + 10 * kBlock, kSize - 10 * kBlock),
        "refresh: all but the owned bytes");
  c.SetAllExcept(kBase, kSize, nullptr, true);
  Check(c.Empty(), "refresh while owning everything: nothing coherent");
  c.SetAllExcept(kBase, kSize, nullptr, false);
  Check(c.Contains(kBase, kSize),
        "refresh of an image owning nothing: everything");
  // CoveredWith: owned or coherent throughout.
  c.Clear();
  c.Add(kBase + kBlock, kBlock / 2);
  owned.Clear();
  owned.Add(kBase + kBlock + kBlock / 2, kBlock / 2);
  Check(c.CoveredWith(&owned, false, kBase + kBlock, kBlock),
        "half coherent + half owned");
  Check(!c.CoveredWith(nullptr, false, kBase + kBlock, kBlock),
        "half coherent alone");
  Check(c.CoveredWith(nullptr, true, kBase, kSize), "owns all");
  owned.Clear();
  owned.Add(kBase + kBlock + 100,
            50); // a hole in the middle, the rest only partly coherent
  Check(!c.CoveredWith(&owned, false, kBase + kBlock, kBlock),
        "gap after the owned piece");
}

// The frame that cost 976 MB/s (the 2432x1368 target over a 1280x720 one at the
// same address).
void TestCoherenceFrame() {
  C::ImageCoherence big;
  C::RangeSet owned; // the big target's owned bytes (Image::m_owned)
  const auto covered = [&](uint64_t block) {
    return big.CoveredWith(&owned, false, kBase + block * kBlock, kBlock);
  };
  // 1. Whole refresh, then a draw claims the scissor blocks 100..199
  // (CommitGpuWrite subtracts
  //    the claim from coherence: an owned byte is never coherent).
  big.SetAllExcept(kBase, kSize, nullptr, false);
  owned.Add(kBase + 100 * kBlock, 100 * kBlock);
  big.Subtract(kBase + 100 * kBlock, 100 * kBlock);
  Check(covered(0) && covered(150) && covered(300),
        "after the draw every block is skippable");
  // 2. A compute storage write of the 1280x720 surface's bytes (blocks 0..112)
  // as a buffer
  //    (InvalidateMemoryFromGPU: subtract + disown).
  big.Subtract(kBase, 113 * kBlock);
  owned.Subtract(kBase, 113 * kBlock);
  Check(!covered(0) && !covered(112) && covered(113) && covered(300),
        "storage write: its blocks stale");
  // 3. Another image renders over blocks 113..120 (CommitGpuWrite: takes the
  // big target's ownership,
  //    coherence unchanged: neither its native bytes nor the canonical ones
  //    changed).
  owned.Subtract(kBase + 113 * kBlock, 8 * kBlock);
  Check(!covered(115), "taken owned bytes were never coherent: stale");
  Check(covered(300), "elsewhere still skippable");
  // 4. The big target's blocks 121..199 move to the buffer
  // (MaterializeOwnedBytes: the buffer
  //    write removes coherence from all, then the owner adds what it moved).
  big.Subtract(kBase + 121 * kBlock, 79 * kBlock);
  owned.Subtract(kBase + 121 * kBlock, 79 * kBlock);
  big.Add(kBase + 121 * kBlock, 79 * kBlock);
  Check(covered(150), "materialized bytes are coherent");
  // 5. Another image's bytes reach the buffer over blocks 180..189
  // (NoteBufferContentWrite).
  big.Subtract(kBase + 180 * kBlock, 10 * kBlock);
  Check(!covered(185) && covered(179) && covered(190),
        "foreign bytes in the buffer: stale");
  // 6. The rebuild refreshes the stale blocks and adds them (segments minus
  // owned).
  big.Add(kBase, 121 * kBlock);
  big.Add(kBase + 180 * kBlock, 10 * kBlock);
  for (uint64_t b = 0; b < 19 * 22; ++b) {
    Check(covered(b), "after the rebuild every block is skippable");
  }
  // 7. A CPU write, a residency change, an unbounded GPU write
  // (MarkImageGpuModified without a
  //    claim), a copy into the image or the loss of GPU ownership: Clear().
  big.Clear();
  Check(!covered(300), "cleared: nothing skippable unless owned");
  Check(big.CoveredWith(&owned, false, kBase + 250 * kBlock, 0),
        "zero-size range");
}
} // namespace

int main() {
  TestAllOwned();
  TestNoneOwned();
  TestPrefixTaken();
  TestMixed();
  TestRowAlign();
  TestNonContiguousRows();
  TestClip();
  TestSubSurfaceAddressing();
  TestPreserveSegments();
  TestCoherenceAlgebra();
  TestCoherenceFrame();
  std::puts("OwnedRebuildPlanTests: ok");
  return 0;
}
