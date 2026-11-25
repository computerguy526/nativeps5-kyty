// v1.4 review M15: KYTY_SAROS_PROGRAM_CACHE / SarosCacheBypass / SarosCacheVariant (sarosProgramCache.h), synthetic hashes, no GPU.
// The switches are read once per process: argv[1] picks the state ("default", "cache" = KYTY_SAROS_PROGRAM_CACHE=1, "off").
#include "graphics/shader/recompiler/sarosProgramCache.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace Libs::Graphics::ShaderRecompiler;

static int fails = 0;
#define CHECK(c)                                                    \
	do {                                                            \
		if (!(c)) {                                                 \
			std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); \
			fails++;                                                \
		}                                                           \
	} while (0)

int main(int argc, char** argv) {
	const char* mode = argc > 1 ? argv[1] : "default";
	constexpr uint64_t kHand = 0x9fba2edffc549531ull; // first SarosHandShader list entry
	constexpr uint64_t kRaw  = 0xda18f5091bbf4843ull; // built-in Nanite-raw pixel shader
	constexpr uint64_t kPlain = 0x1234ull;
	if (std::strcmp(mode, "off") == 0) {
		// no SAROS switch: every program has variant 0 and may use the cache, bypass or not
		CHECK(SarosCacheVariant(kHand, false) == 0 && SarosCacheVariant(kRaw, false) == 0 && SarosCacheVariant(kPlain, true) == 0);
		CHECK(SarosCacheBypass());
		CHECK(SarosProgramMayUseDiskCache(kHand, false) && SarosProgramMayUseDiskCache(kPlain, true));
	} else {
		setenv("KYTY_SAROS_HANDSHADERS", "1", 1);
		setenv("KYTY_SAROS_NANITE_RAW", "1", 1);
		if (std::strcmp(mode, "cache") == 0) setenv("KYTY_SAROS_PROGRAM_CACHE", "1", 1);
		// the switches change the translation: non-zero variant for the listed hand shader, the raw PS and every mesh program
		const auto hand = SarosCacheVariant(kHand, false), raw = SarosCacheVariant(kRaw, false), mesh = SarosCacheVariant(kPlain, true);
		CHECK(hand != 0 && raw != 0 && mesh != 0);
		CHECK(hand != raw && raw != mesh && hand != mesh);
		CHECK(SarosCacheVariant(kPlain, false) == 0); // an unlisted pixel/compute program keeps key bytes unchanged
		CHECK(hand == SarosCacheVariant(kHand, false));  // deterministic
		if (std::strcmp(mode, "cache") == 0) {
			CHECK(!SarosCacheBypass());
			CHECK(SarosProgramMayUseDiskCache(kHand, false) && SarosProgramMayUseDiskCache(kRaw, false) && SarosProgramMayUseDiskCache(kPlain, true));
		} else {
			CHECK(SarosCacheBypass());
			CHECK(!SarosProgramMayUseDiskCache(kHand, false) && !SarosProgramMayUseDiskCache(kRaw, false) && !SarosProgramMayUseDiskCache(kPlain, true));
			CHECK(SarosProgramMayUseDiskCache(kPlain, false)); // a program no switch touches still uses the cache
		}
	}
	std::printf("saros_program_cache_test %s: %s (%d fails)\n", mode, fails == 0 ? "PASS" : "FAIL", fails);
	return fails == 0 ? 0 : 1;
}
