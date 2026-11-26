// c2acct: KYTY_NATIVE_INDIRECT_MESH decides CodegenOptions::mesh_indirect_params, which the codegen
// fingerprint hashes. The ASTRO profile sets =1; a profile (or a player) that set 0 or "empty" would
// change the fingerprint and start from a cold pipeline cache. No GPU.
// Usage: mesh_indirect_env_tests <unset|VALUE> <expect_mesh_indirect_params 0|1>
// (the options are read once per process, so one value per invocation).
#include "graphics/shader/recompiler/CodegenOptions.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace Libs::Graphics::ShaderRecompiler;

int main(int argc, char** argv) {
	if (argc != 3) {
		std::printf("usage: %s <unset|VALUE> <0|1>\n", argv[0]);
		return 2;
	}
	if (std::strcmp(argv[1], "unset") == 0) {
		unsetenv("KYTY_NATIVE_INDIRECT_MESH");
	} else {
		setenv("KYTY_NATIVE_INDIRECT_MESH", argv[1], 1);
	}
	const bool expect = std::strcmp(argv[2], "1") == 0;
	const bool got    = GetCodegenOptions().mesh_indirect_params;
	std::printf("mesh_indirect_env_tests: KYTY_NATIVE_INDIRECT_MESH=%s mesh_indirect_params=%d expected=%d\n",
	            argv[1], got, expect);
	return got == expect ? 0 : 1;
}
