// nps5 sarosnanite: KYTY_SAROS_NANITE_RAW switch parsing (sarosHandShaders.h), synthetic hashes, no GPU.
#include "graphics/shader/recompiler/sarosHandShaders.h"

#include <cstdio>
#include <cstdlib>

using namespace Libs::Graphics::ShaderRecompiler;

static int fails = 0;
#define CHECK(c)                                                         \
	do {                                                                 \
		if (!(c)) {                                                      \
			std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c);      \
			fails++;                                                     \
		}                                                                \
	} while (0)

int main(int argc, char** argv) {
	if (argc > 1) { // "default": the switch on with =1 (built-in four pixel shaders, default masks)
		setenv("KYTY_SAROS_NANITE_RAW", "1", 1);
		CHECK(SarosNaniteRawPs(0xda18f5091bbf4843ull) && SarosNaniteRawPs(0x865a72566b298994ull));
		CHECK(!SarosNaniteRawPs(1));
		CHECK(!SarosNaniteRawAnd(0) && SarosNaniteRawAnd(1) && SarosNaniteRawVerts(2) && !SarosNaniteRawVerts(1));
		std::printf("saros_nanite_raw_test default: %s (%d fails)\n", fails == 0 ? "PASS" : "FAIL", fails);
		return fails == 0 ? 0 : 1;
	}
	// Unset: everything is off (the switch is read once per process, so each state is a process).
	// Without arguments this binary runs with the list form; with one argument, the =1 form.
	setenv("KYTY_SAROS_NANITE_RAW", "1234abcd,ff", 1);
	setenv("KYTY_SAROS_NANITE_RAW_AND", "0x1", 1);
	setenv("KYTY_SAROS_NANITE_RAW_VERTS", "0x6", 1);
	CHECK(SarosNaniteRawMesh());
	CHECK(SarosNaniteRawPs(0x1234abcdull));
	CHECK(SarosNaniteRawPs(0xffull));
	CHECK(!SarosNaniteRawPs(0xda18f5091bbf4843ull)); // a list replaces the built-in four
	CHECK(!SarosNaniteRawPs(0x12ull));
	CHECK(SarosNaniteRawAnd(0) && !SarosNaniteRawAnd(1) && !SarosNaniteRawAnd(40));
	CHECK(!SarosNaniteRawVerts(0) && SarosNaniteRawVerts(1) && SarosNaniteRawVerts(2) && !SarosNaniteRawVerts(3));
	CHECK(SarosNaniteRawLocation(1) == 17u && SarosNaniteRawLocation(5) == 17u);
	CHECK(SarosNaniteRawVertexLocation(0, 0) == 20u && SarosNaniteRawVertexLocation(1, 2) == 25u &&
	      SarosNaniteRawVertexLocation(4, 1) == 21u);
	std::printf("saros_nanite_raw_test: %s (%d fails)\n", fails == 0 ? "PASS" : "FAIL", fails);
	return fails == 0 ? 0 : 1;
}
