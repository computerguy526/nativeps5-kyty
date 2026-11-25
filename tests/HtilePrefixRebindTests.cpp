// v1.4 review M6: the retain_cached_layout guard of the HTile prefix rebind pick (6506ef9f6), gated default off. No GPU.
#include "graphics/host_gpu/renderer/cache/htilePrefixRebind.h"

#include <cstdio>
#include <cstdlib>

using namespace Libs::Graphics;

static int fails = 0;
#define CHECK(c)                                                    \
	do {                                                            \
		if (!(c)) {                                                 \
			std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); \
			fails++;                                                \
		}                                                           \
	} while (0)

int main(int argc, char**) {
	// Gate off (the shipped default): the v1.3.1 checks. Same address, equal bytes per block, cached layers, no metadata -> rebind.
	CHECK(PrefixLayoutGuard(false, true, true, true, true, false));
	// ... a different base address (a suffix/middle slice) is NOT bound to the cached layout (the M6 failure)
	CHECK(!PrefixLayoutGuard(false, true, false, true, true, false));
	CHECK(!PrefixLayoutGuard(false, false, true, true, true, false)); // other bytes per block
	CHECK(!PrefixLayoutGuard(false, true, true, false, true, false)); // cached side has no layers
	CHECK(!PrefixLayoutGuard(false, true, true, true, false, true));  // an HTile request is not admitted
	CHECK(!PrefixLayoutGuard(false, true, true, true, false, false)); // DCC/other metadata
	// Gate on (upstream 6dcdcd44): address / bpp / layers no longer matter, HTile is admitted, other metadata is not.
	CHECK(PrefixLayoutGuard(true, true, true, true, false, true));
	CHECK(PrefixLayoutGuard(true, false, false, false, true, false));
	CHECK(!PrefixLayoutGuard(true, true, true, true, false, false));
	// The switch: unset or anything but "1" is off (read once per process; argv[1] present = "1" in the environment).
	if (argc > 1) {
		setenv("KYTY_HTILE_PREFIX_REBIND", "1", 1);
		CHECK(HtilePrefixRebindEnabled());
	} else {
		unsetenv("KYTY_HTILE_PREFIX_REBIND");
		CHECK(!HtilePrefixRebindEnabled());
	}
	std::printf("htile_prefix_rebind_tests %s: %s (%d fails)\n", argc > 1 ? "on" : "default", fails == 0 ? "PASS" : "FAIL", fails);
	return fails == 0 ? 0 : 1;
}
