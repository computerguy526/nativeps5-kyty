// nps5 findbuf: the dirty walk of a BDA pass (BdaScanWalk::CollectDirtyVisits)
// visits exactly the buffers of the reference walk (ForEachOverlapping) whose
// clipped range intersects a dirty run, in the same order and with the same
// ranges, for random disjoint buffer maps (BufferCache's buffers never overlap:
// CreateBuffer joins every overlap), random walk ranges and random runs.
// Overlapping maps (not a BufferCache state) only check that the dirty walk
// never visits a buffer the reference does not, in order, with its range
// (KYTY_FINDBUF_VERIFY reports a miss at run time).
#include "graphics/host_gpu/bitRuns.h"
#include "graphics/host_gpu/renderer/cache/bdaScanWalk.h"

#include <array>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <random>
#include <utility>
#include <vector>

namespace {

using Libs::Graphics::BdaScanWalk::CollectDirtyVisits;
using Libs::Graphics::BdaScanWalk::ForEachOverlapping;
using Libs::Graphics::BdaScanWalk::Visit;

struct Buf {
  uint64_t begin = 0;
  uint64_t end = 0;
};
using Map = std::map<uint64_t, Buf>;

int g_failures = 0;

void Check(bool value, const char *text) {
  if (!value) {
    std::fprintf(stderr, "FAIL: %s\n", text);
    if (++g_failures > 20) {
      std::exit(1);
    }
  }
}

const auto kBounds = [](const Buf &b) {
  return std::pair<uint64_t, uint64_t>{b.begin, b.end};
};

std::vector<Visit>
Reference(const Map &buffers, uint64_t vaddr, uint64_t end,
          const std::vector<std::pair<uint64_t, uint64_t>> &runs) {
  std::vector<Visit> out;
  ForEachOverlapping(buffers, vaddr, end, kBounds,
                     [&](auto it, uint64_t start, uint64_t finish) {
                       for (const auto &[b, e] : runs) {
                         if (b < finish && start < e) {
                           out.push_back({it->first, start, finish});
                           return;
                         }
                       }
                     });
  return out;
}

std::vector<Visit>
Fast(const Map &buffers, uint64_t vaddr, uint64_t end,
     const std::vector<std::pair<uint64_t, uint64_t>> &runs) {
  std::vector<Visit> out;
  CollectDirtyVisits(
      buffers, vaddr, end, kBounds,
      [&](auto &&callback) {
        for (const auto &[b, e] : runs) {
          callback(b, e);
        }
      },
      out);
  return out;
}

Map RandomMap(std::mt19937_64 &rng, bool overlapping) {
  Map map;
  uint64_t at = 0x1000 * (rng() % 8);
  const int count = static_cast<int>(rng() % 40);
  for (int i = 0; i < count; i++) {
    const uint64_t size = 0x1000 * (1 + rng() % 64);
    if (overlapping && (rng() % 3) == 0 && !map.empty()) {
      // A buffer starting inside an earlier one, possibly reaching beyond it.
      const auto &last = std::prev(map.end())->second;
      const auto begin =
          last.begin + 0x1000 * (rng() % ((last.end - last.begin) / 0x1000));
      if (map.count(begin) == 0) {
        map[begin] = {begin, begin + size};
      }
      continue;
    }
    at += 0x1000 * (rng() % 6); // gaps
    map[at] = {at, at + size};
    at += size;
  }
  return map;
}

std::vector<std::pair<uint64_t, uint64_t>>
RandomRuns(std::mt19937_64 &rng, uint64_t vaddr, uint64_t end) {
  std::vector<std::pair<uint64_t, uint64_t>> runs;
  uint64_t at = vaddr;
  while (at < end) {
    at += (rng() % 4 == 0)
              ? 0
              : 0x1000 * (rng() % 24) + (rng() % 2 ? 0 : rng() % 0x1000);
    if (at >= end) {
      break;
    }
    const uint64_t len = 1 + rng() % (0x1000 * 8);
    const uint64_t finish = std::min(end, at + len);
    if (!runs.empty() && runs.back().second >= at) {
      at = runs.back().second +
           1; // keep runs disjoint and not adjacent (merged)
      continue;
    }
    runs.emplace_back(at, finish);
    at = finish + 1;
  }
  return runs;
}

bool Same(const std::vector<Visit> &a, const std::vector<Visit> &b) {
  if (a.size() != b.size()) {
    return false;
  }
  for (size_t i = 0; i < a.size(); i++) {
    if (!(a[i] == b[i])) {
      return false;
    }
  }
  return true;
}

void TestFixed() {
  Map map;
  map[0x1000] = {0x1000, 0x5000};
  map[0x8000] = {0x8000, 0x9000};
  map[0xa000] = {0xa000, 0x20000};
  // No runs: no visits.
  Check(Fast(map, 0, 0x30000, {}).empty(),
        "a walk without runs visited a buffer");
  // A run in the gap between buffers: no visits.
  Check(Fast(map, 0, 0x30000, {{0x6000, 0x7000}}).empty(),
        "a run in a gap visited a buffer");
  // Many runs inside one buffer: one visit with the buffer's clipped range.
  const auto many =
      Fast(map, 0xb000, 0x30000,
           {{0xb000, 0xb100}, {0xc000, 0xc100}, {0x1f000, 0x1f001}});
  Check(many.size() == 1 && many[0].key == 0xa000 && many[0].start == 0xb000 &&
            many[0].finish == 0x20000,
        "runs inside one buffer did not give one clipped visit");
  // The walk starts inside the predecessor buffer.
  const auto pred =
      Fast(map, 0x2000, 0x8800, {{0x2000, 0x3000}, {0x8400, 0x8500}});
  Check(pred.size() == 2 && pred[0].key == 0x1000 && pred[0].start == 0x2000 &&
            pred[1].key == 0x8000 && pred[1].finish == 0x8800,
        "the walk from inside a buffer lost the predecessor or the clip");
  // Empty map and empty range.
  Check(Fast(Map{}, 0, 0x1000, {{0, 0x1000}}).empty(), "an empty map visited");
  Check(Fast(map, 0x2000, 0x2000, {}).empty(), "an empty range visited");
}

void TestRandom(bool overlapping) {
  std::mt19937_64 rng(overlapping ? 0xb0bull : 0xf1dbull);
  uint64_t visits = 0;
  for (int round = 0; round < 20000; round++) {
    const auto map = RandomMap(rng, overlapping);
    uint64_t vaddr = 0x800 * (rng() % 256) + (rng() % 2 ? 0 : rng() % 0x800);
    uint64_t end = vaddr + 1 + rng() % 0x100000;
    const auto runs = RandomRuns(rng, vaddr, end);
    const auto ref = Reference(map, vaddr, end, runs);
    const auto fast = Fast(map, vaddr, end, runs);
    visits += ref.size();
    if (overlapping) {
      size_t j = 0;
      for (const auto &v : fast) {
        while (j < ref.size() && !(ref[j] == v)) {
          j++;
        }
        Check(j < ref.size(), "overlapping map: the dirty walk visited a "
                              "buffer the reference does not");
        j++;
      }
      continue;
    }
    if (!Same(ref, fast)) {
      std::fprintf(stderr,
                   "round %d: ref %zu visits, fast %zu (range 0x%" PRIx64
                   "+0x%" PRIx64 ")\n",
                   round, ref.size(), fast.size(), vaddr, end - vaddr);
      Check(false,
            overlapping
                ? "overlapping map: dirty walk differs from the reference"
                : "disjoint map: dirty walk differs from the reference");
    }
    // All-dirty: the dirty walk equals the plain reference walk (every
    // overlapping buffer).
    std::vector<Visit> all;
    ForEachOverlapping(map, vaddr, end, kBounds,
                       [&](auto it, uint64_t s, uint64_t f) {
                         all.push_back({it->first, s, f});
                       });
    Check(Same(all, Fast(map, vaddr, end, {{vaddr, end}})),
          "an all-dirty range did not visit every overlapping buffer");
  }
  Check(visits > 1000,
        "the random cases visited too few buffers to mean anything");
}

// ForEachSetBitRun: exactly the set bits of [first, last), as ascending runs
// per word.
void TestSetBitRuns() {
  std::mt19937_64 rng(0x5e7b175ull);
  for (int round = 0; round < 20000; round++) {
    std::array<uint64_t, 4> words{};
    for (auto &w : words) {
      const auto kind = rng() % 4;
      w = kind == 0 ? 0 : kind == 1 ? ~uint64_t{0} : rng() & rng();
    }
    const size_t first = rng() % 257;
    const size_t last = first + rng() % (257 - first);
    std::vector<bool> seen(256, false);
    size_t prev_end = 0;
    bool ordered = true;
    Libs::Graphics::ForEachSetBitRun(
        [&](size_t w) { return words[w]; }, first, last,
        [&](size_t b, size_t e) {
          ordered = ordered && b >= prev_end && b < e && b >= first &&
                    e <= last && (b / 64) == ((e - 1) / 64);
          prev_end = e;
          for (size_t i = b; i < e; i++) {
            seen[i] = true;
          }
        });
    Check(ordered, "set-bit runs are not ascending, in range and per word");
    for (size_t i = 0; i < 256; i++) {
      const bool set =
          i >= first && i < last && ((words[i / 64] >> (i % 64)) & 1u) != 0;
      if (seen[i] != set) {
        Check(false, "set-bit runs differ from the bits");
        break;
      }
    }
  }
}

// A model tracker: regions of 16 pages of 0x100 bytes (one 64-bit word holds a
// region's bits), some missing. Query(s, f) answers as
// MemoryTracker::QueryCpuDirtyRelaxed: "dirty or undecided" when a page
// touching [s, f) is set or a region touching it is missing.
struct ModelTracker {
  static constexpr uint64_t kPage = 0x100;
  static constexpr uint64_t kRegion = 0x1000;
  std::vector<std::unique_ptr<uint64_t>> regions;

  bool Query(uint64_t s, uint64_t f) const {
    for (uint64_t r = s / kRegion; r <= (f - 1) / kRegion; r++) {
      if (!regions[r]) {
        return true;
      }
    }
    for (uint64_t page = s / kPage; page <= (f - 1) / kPage; page++) {
      if (((*regions[page / 16]) >> (page % 16)) & 1u) {
        return true;
      }
    }
    return false;
  }
  template <typename Func>
  void Runs(uint64_t vaddr, uint64_t size, Func &&func) const {
    Libs::Graphics::ForEachDirtyOrMissingRun(
        vaddr, size, kRegion, kPage,
        [this](uint64_t index) -> const uint64_t * {
          return regions[index].get();
        },
        [](const uint64_t *bits, size_t w) {
          return w == 0 ? *bits : uint64_t{0};
        },
        func);
  }
};

ModelTracker RandomTracker(std::mt19937_64 &rng, size_t count) {
  ModelTracker tracker;
  for (size_t i = 0; i < count; i++) {
    if (rng() % 5 == 0) {
      tracker.regions.emplace_back();
      continue;
    }
    const auto kind = rng() % 4;
    tracker.regions.emplace_back(
        std::make_unique<uint64_t>((kind == 0   ? 0
                                    : kind == 1 ? 0xffffu
                                                : (rng() & rng() & 0xffffu))));
  }
  return tracker;
}

// The tracker walk: merged, clipped runs that a sub-range intersects exactly
// when the model query says "dirty or undecided".
void TestDirtyOrMissingRuns() {
  std::mt19937_64 rng(0xd17ull);
  for (int round = 0; round < 3000; round++) {
    const auto tracker = RandomTracker(rng, 12);
    const uint64_t span = 12 * ModelTracker::kRegion;
    for (int q = 0; q < 40; q++) {
      uint64_t a = rng() % span;
      uint64_t b = rng() % span;
      if (a > b)
        std::swap(a, b);
      if (a == b)
        b = a + 1;
      std::vector<std::pair<uint64_t, uint64_t>> runs;
      tracker.Runs(a, b - a,
                   [&](uint64_t x, uint64_t y) { runs.emplace_back(x, y); });
      for (size_t i = 0; i < runs.size(); i++) {
        Check(runs[i].first < runs[i].second && runs[i].first >= a &&
                  runs[i].second <= b &&
                  (i == 0 || runs[i - 1].second < runs[i].first),
              "tracker runs are not merged, ascending and clipped");
      }
      for (int k = 0; k < 8; k++) {
        const uint64_t s = a + rng() % (b - a);
        const uint64_t f = s + 1 + rng() % (b - s);
        bool hit = false;
        for (const auto &[x, y] : runs) {
          hit = hit || (x < f && s < y);
        }
        Check(hit == tracker.Query(s, f),
              "tracker runs disagree with the relaxed query");
      }
    }
  }
}

// End to end: random disjoint buffers over the model tracker; the dirty walk's
// visits are the reference walk's visits whose relaxed query does not skip them
// (RelaxedNothingToUpload).
void TestWalkOverTracker() {
  std::mt19937_64 rng(0xe2eull);
  uint64_t visits = 0;
  for (int round = 0; round < 5000; round++) {
    const auto tracker = RandomTracker(rng, 16);
    Map map;
    uint64_t at = 0x40 * (rng() % 8);
    while (true) {
      at += 0x40 * (rng() % 40);
      const uint64_t size = 0x40 * (1 + rng() % 200);
      if (at + size > 16 * ModelTracker::kRegion) {
        break;
      }
      map[at] = {at, at + size};
      at += size;
    }
    uint64_t a = rng() % (16 * ModelTracker::kRegion);
    uint64_t b = rng() % (16 * ModelTracker::kRegion);
    if (a > b)
      std::swap(a, b);
    if (a == b)
      b = a + 1;
    std::vector<Visit> ref;
    ForEachOverlapping(map, a, b, kBounds,
                       [&](auto it, uint64_t s, uint64_t f) {
                         if (tracker.Query(s, f)) {
                           ref.push_back({it->first, s, f});
                         }
                       });
    std::vector<Visit> fast;
    CollectDirtyVisits(
        map, a, b, kBounds,
        [&](auto &&callback) { tracker.Runs(a, b - a, callback); }, fast);
    visits += ref.size();
    Check(
        Same(ref, fast),
        "the dirty walk over the tracker differs from the skipping reference");
  }
  Check(visits > 1000,
        "the tracker walk cases visited too few buffers to mean anything");
}

} // namespace

int main() {
  TestFixed();
  TestRandom(false);
  TestRandom(true);
  TestSetBitRuns();
  TestDirtyOrMissingRuns();
  TestWalkOverTracker();
  if (g_failures != 0) {
    std::fprintf(stderr, "BdaScanWalkTests: %d failure(s)\n", g_failures);
    return 1;
  }
  std::puts("BdaScanWalkTests: all cases passed");
  return 0;
}
