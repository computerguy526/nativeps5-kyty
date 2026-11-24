// nps5 onereg: unit test of the KYTY_DCC_KNOWN_FILL_ALIASED decision (dccKnownFill.h). Standalone:
//   clang++ -std=c++20 -O2 -Isrc tests/DccKnownFillTests.cpp -o /tmp/dcc_known_fill_tests && /tmp/dcc_known_fill_tests
// CMake target: dcc_known_fill_tests.
#include "graphics/host_gpu/renderer/cache/dccKnownFill.h"

#include <cstdio>

using namespace Libs::Graphics::DccKnownFill;

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
  CHECK(ParseMode(nullptr) == Mode::Off);
  CHECK(ParseMode("") == Mode::Off);
  CHECK(ParseMode("0") == Mode::Off);
  CHECK(ParseMode("1") == Mode::Take);
  CHECK(ParseMode("verify") == Mode::Verify);
  CHECK(ParseMode("Verify") == Mode::Off);
  CHECK(ParseMode("2") == Mode::Off);

  const Mode modes[] = {Mode::Off, Mode::Take, Mode::Verify};

  // No record (or a stale one, which the buffer cache reports as no record): nothing is decided.
  for (Mode m : modes)
    for (int u = 0; u < 2; u++) {
      const auto d = Decide(m, false, 0x01010101u, u != 0);
      CHECK(!d.uniform_known && !d.take_known && !d.verify);
    }

  // Uniform record, unaliased range: taken in every mode (the default path), never verified.
  for (Mode m : modes) {
    const auto d = Decide(m, true, 0x0A0A0A0Au, true);
    CHECK(d.uniform_known && d.take_known && !d.verify && d.known_byte == 0x0A);
  }

  // Uniform record, aliased range: taken only with =1 (keep the contents instead of reading back);
  // =verify keeps the CPU fallback and asks for the comparison; off does neither.
  {
    const auto off = Decide(Mode::Off, true, 0xFFFFFFFFu, false);
    CHECK(off.uniform_known && !off.take_known && !off.verify && off.known_byte == 0xFF);
    const auto take = Decide(Mode::Take, true, 0xFFFFFFFFu, false);
    CHECK(take.uniform_known && take.take_known && !take.verify);
    const auto ver = Decide(Mode::Verify, true, 0xFFFFFFFFu, false);
    CHECK(ver.uniform_known && !ver.take_known && ver.verify && ver.known_byte == 0xFF);
  }

  // Non-uniform record: never taken, never verified, whatever the mode or aliasing.
  const uint32_t mixed[] = {0x01010102u, 0x00010101u, 0xFF00FF00u, 0x12345678u};
  for (uint32_t w : mixed)
    for (Mode m : modes)
      for (int u = 0; u < 2; u++) {
        const auto d = Decide(m, true, w, u != 0);
        CHECK(!d.uniform_known && !d.take_known && !d.verify);
      }

  // A zero fill is a valid uniform record (byte 0).
  {
    const auto d = Decide(Mode::Take, true, 0u, false);
    CHECK(d.uniform_known && d.take_known && d.known_byte == 0);
  }

  std::printf("dcc_known_fill_tests: %d checks, %d failed\n", g_checks, g_failed);
  return g_failed == 0 ? 0 : 1;
}
