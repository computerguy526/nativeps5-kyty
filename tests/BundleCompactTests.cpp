// sarosbundle: unit test of the shader-bundle compaction HLE (bundleCompact.h).
// Standalone:
//   clang++ -std=c++20 -O2 -Isrc tests/BundleCompactTests.cpp -o
//   /tmp/bundle_compact_tests && /tmp/bundle_compact_tests
// CMake target: bundle_compact_tests.
#include "graphics/guest_gpu/command_processor/bundleCompact.h"

#include <cstdio>

using namespace Libs::Graphics::BundleCompact;

static int g_checks = 0, g_failed = 0;
#define CHECK(c)                                                               \
  do {                                                                         \
    g_checks++;                                                                \
    if (!(c)) {                                                                \
      g_failed++;                                                              \
      std::printf("FAILED %s:%d %s\n", __FILE__, __LINE__, #c);                \
    }                                                                          \
  } while (0)

int main() {
  // Kernel identity: the exact bytes match, one flipped bit does not.
  uint8_t code[kKernelCodeSize];
  std::memcpy(code, kKernelCode, sizeof(code));
  CHECK(MatchesKernel(code));
  code[0x100] ^= 1u;
  CHECK(!MatchesKernel(code));
  CHECK(kKernelCode[0] == 0x03 && kKernelCode[1] == 0x00 &&
        kKernelCode[2] == 0xa0 && kKernelCode[3] == 0xbf);

  // User data decode: a layout like the SAROS frame (108 x 76, trailer 56, args
  // offset 0 stride 16).
  uint32_t s[16] = {
      108,        76,     56, 0x00100000, 0x10000000, 0x00001ec2, 0, 0,
      0xaaaa0000, 0x1111, 0,  0,          0xbbbb0000, 0x2222,     0, 0};
  s[1] |= 0xabcd0000u; // the high half of s1 is not part of R
  const auto p = Decode(s);
  CHECK(p.n == 108 && p.r == 76 && p.trailer == 56 && p.args_offset == 0 &&
        p.args_stride == 16);
  CHECK(p.args == (0x10000000ull | (0x1ec2ull << 32)));
  CHECK(p.input == (0xaaaa0000ull | (0x1111ull << 32)));
  CHECK(p.output == (0xbbbb0000ull | (0x2222ull << 32)));
  CHECK(Valid(p));
  CHECK(InputBytes(p) == 108u * 4u + 108u * 76u);
  CHECK(OutputBytes(p) == 8u + 108u * 76u + 56u);

  // Synthetic: N = 70 (two waves), R = 12, trailer 8. Bins 0..69; index = bin
  // except -1 for every 5th; args (stride 16, offset 4): bins divisible by 3
  // are empty (one zero dim).
  Params q;
  q.n = 70;
  q.r = 12;
  q.trailer = 8;
  q.args_offset = 4;
  q.args_stride = 16;
  std::vector<uint32_t> in_dw(q.n + q.n * q.r / 4);
  std::vector<uint8_t> args(4 + 16 * 70 + 16, 0xee);
  std::vector<uint32_t> expect_bins;
  for (uint32_t i = 0; i < q.n; i++) {
    in_dw[i] = i % 5 == 4 ? 0xffffffffu : i;
    for (uint32_t w = 0; w < 3; w++)
      in_dw[q.n + i * 3 + w] = 0x1000u * (i + 1) + w;
    const uint32_t dims[3] = {i % 3 == 0 ? 0u : 2u, 3u, 4u};
    std::memcpy(args.data() + 4 + 16u * i, dims, 12);
    if (in_dw[i] != 0xffffffffu && i % 3 != 0)
      expect_bins.push_back(i);
  }
  const auto *input = reinterpret_cast<const uint8_t *>(in_dw.data());
  CHECK(ArgsExtent(q, input) ==
        4u + 16u * 68u + 12u); // highest used index: 68 (69 % 5 == 4)
  std::vector<uint8_t> out(OutputBytes(q), 0xff);
  const uint8_t trailer[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  std::memcpy(out.data() + 8 + q.n * q.r, trailer, 8);
  const uint32_t k = Compact(q, input, args.data(), out.data());
  CHECK(k == expect_bins.size());
  uint32_t total = 0, waves = 0;
  std::memcpy(&total, out.data(), 4);
  std::memcpy(&waves, out.data() + 4, 4);
  CHECK(total == k * q.r);
  CHECK(waves == 2);
  for (uint32_t j = 0; j < k; j++) {
    uint32_t rec[3];
    std::memcpy(rec, out.data() + 8 + j * q.r, 12);
    const uint32_t bin = expect_bins[j];
    CHECK(rec[0] == 0x1000u * (bin + 1) && rec[1] == rec[0] + 1 &&
          rec[2] == rec[0] + 2);
  }
  CHECK(std::memcmp(out.data() + 8 + k * q.r, trailer, 8) == 0);
  // A trailer moved onto records it overlaps (k < N) survives; an all-survivor
  // layout leaves it in place.
  std::vector<uint8_t> full_out(OutputBytes(q), 0);
  Params all = q;
  std::vector<uint32_t> in2(q.n + q.n * 3);
  std::vector<uint8_t> args2(4 + 16 * 70, 0);
  for (uint32_t i = 0; i < q.n; i++) {
    in2[i] = i;
    const uint32_t dims[3] = {1, 1, 1};
    std::memcpy(args2.data() + 4 + 16u * i, dims, 12);
    in2[q.n + i * 3] = i;
  }
  std::memcpy(full_out.data() + 8 + q.n * q.r, trailer, 8);
  CHECK(Compact(all, reinterpret_cast<const uint8_t *>(in2.data()),
                args2.data(), full_out.data()) == 70u);
  CHECK(std::memcmp(full_out.data() + 8 + q.n * q.r, trailer, 8) == 0);
  // Nothing survives: the trailer lands right after the header.
  std::vector<uint8_t> none_out(OutputBytes(q), 0xff);
  std::memcpy(none_out.data() + 8 + q.n * q.r, trailer, 8);
  std::vector<uint32_t> in3(q.n + q.n * 3, 0xffffffffu);
  CHECK(Compact(q, reinterpret_cast<const uint8_t *>(in3.data()), args.data(),
                none_out.data()) == 0u);
  CHECK(std::memcmp(none_out.data() + 8, trailer, 8) == 0);
  uint32_t t0 = 1;
  std::memcpy(&t0, none_out.data(), 4);
  CHECK(t0 == 0);

  // Invalid layouts are rejected.
  Params bad = q;
  bad.args = bad.input = bad.output = 0x1000;
  CHECK(Valid(bad));
  bad.n = 0;
  CHECK(!Valid(bad));
  bad.n = 70;
  bad.r = 0;
  CHECK(!Valid(bad));
  std::printf("%d checks, %d failed\n", g_checks, g_failed);
  return g_failed == 0 ? 0 : 1;
}
