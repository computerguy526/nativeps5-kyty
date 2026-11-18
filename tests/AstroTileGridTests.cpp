// Light-tile grid math for a forced internal resolution (graphics/host_gpu/renderer/astroTileGrid.h).
#include "graphics/host_gpu/renderer/astroTileGrid.h"

#include <cstdio>
#include <cstdlib>

namespace {
namespace G = Libs::Graphics::AstroTileGrid;

void Check(bool c, const char* m) {
	if (!c) {
		std::fprintf(stderr, "AstroTileGridTests: %s\n", m);
		std::abort();
	}
}

void Expect(uint32_t w, uint32_t h, unsigned ex, unsigned ey) {
	unsigned tx = 0, ty = 0;
	Check(G::Resolve("auto", w, h, tx, ty), "auto resolves");
	Check(tx == ex && ty == ey, "auto grid");
}
} // namespace

int main() {
	Expect(1280, 720, 80, 45);
	Expect(1600, 900, 100, 57); // 900 / 16 = 56.25 -> 57
	Expect(1920, 1080, 120, 68); // 67.5 -> 68 (the game's level 0)
	Expect(1368, 770, 86, 49);
	unsigned tx = 7, ty = 7;
	Check(!G::Resolve("auto", 0, 0, tx, ty), "auto needs a size");
	Check(!G::Resolve(nullptr, 1280, 720, tx, ty), "null");
	Check(!G::Resolve("0x45", 0, 0, tx, ty) && !G::Resolve("junk", 0, 0, tx, ty), "invalid");
	Check(G::Resolve("80x45", 0, 0, tx, ty) && tx == 80 && ty == 45, "explicit");
	std::puts("AstroTileGridTests: ok");
	return 0;
}
