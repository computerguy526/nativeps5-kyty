// nps5 kytyone: one Kyty for ASTRO (PPSA21564), SAROS (PPSA07631) and Sons of Sparta (PPSA28997).
// Header-only checks run on every merge step (tests/one/run_one_tests.sh):
//  (a) the seam export table (hybrid/kytyGpuNids.inc) has every graphics NID each title imports;
//  (b) title switches are off when their variable is unset, empty or "0";
//  (e) title-specific pure functions (SAROS hand-picked shader list) without a GPU.
// A title's list is enforced once its line is merged: KYTY_ONE_REQUIRE_<TITLE>=1 below.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <sys/wait.h>
#include <unistd.h>

#ifndef KYTY_ONE_REQUIRE_SAROS
#define KYTY_ONE_REQUIRE_SAROS 1
#endif
#ifndef KYTY_ONE_REQUIRE_SPARTA
#define KYTY_ONE_REQUIRE_SPARTA 1
#endif

#if __has_include("graphics/shader/recompiler/sarosHandShaders.h")
#include "graphics/shader/recompiler/sarosHandShaders.h"
#define KYTY_ONE_HAVE_SAROS_HANDSHADERS 1
#endif

namespace {

int g_failures = 0;

void Expect(bool condition, const std::string& what) {
	if (!condition) {
		std::printf("KytyOneTests: failed: %s\n", what.c_str());
		g_failures++;
	}
}

struct Row {
	const char* nid;
	const char* library;
	const char* function;
};

#define KYTY_GPU_NID(index, nid, library, function, binding) {nid, library, function},
constexpr Row kExports[] = {
#include "hybrid/kytyGpuNids.inc"
};
#undef KYTY_GPU_NID

struct Need {
	const char* nid;
	const char* name;
};

constexpr Need kAstro[] = {
#include "one/astro_nids.inc"
};
constexpr Need kSaros[] = {
#include "one/saros_nids.inc"
};
constexpr Need kSparta[] = {
#include "one/sparta_nids.inc"
};

template <size_t N>
int CheckTitle(const char* title, const Need (&needs)[N], bool required) {
	std::set<std::string> exported;
	for (const auto& row: kExports) {
		exported.insert(row.nid);
	}
	int missing = 0;
	for (const auto& need: needs) {
		if (exported.count(need.nid) == 0) {
			missing++;
			if (required) {
				Expect(false, std::string(title) + " needs " + need.nid + " (" + need.name + "), not in kytyGpuNids.inc");
			}
		}
	}
	std::printf("KytyOneTests: %s: %zu of %zu graphics NIDs exported%s\n", title, N - missing, N, required ? "" : " (not required yet)");
	return missing;
}

void TestExportTable() {
	std::set<std::string> seen;
	for (const auto& row: kExports) {
		Expect(seen.insert(row.nid).second, std::string("duplicate export ") + row.nid);
		Expect(std::strlen(row.nid) == 11, std::string("NID length ") + row.nid);
	}
	CheckTitle("ASTRO PPSA21564", kAstro, true);
	CheckTitle("SAROS PPSA07631", kSaros, KYTY_ONE_REQUIRE_SAROS != 0);
	CheckTitle("Sparta PPSA28997", kSparta, KYTY_ONE_REQUIRE_SPARTA != 0);
}

// Runs fn in a child with only `name=value` set (nullptr = unset), so function-local statics that
// cache getenv start fresh. Returns the child's bool result.
template <typename Fn>
bool InChild(const char* name, const char* value, Fn fn) {
	std::fflush(stdout);
	const pid_t pid = fork();
	if (pid == 0) {
		if (value == nullptr) {
			unsetenv(name);
		} else {
			setenv(name, value, 1);
		}
		_exit(fn() ? 1 : 0);
	}
	int status = 0;
	waitpid(pid, &status, 0);
	return WIFEXITED(status) && WEXITSTATUS(status) == 1;
}

void TestSarosHandShaders() {
#ifdef KYTY_ONE_HAVE_SAROS_HANDSHADERS
	using Libs::Graphics::ShaderRecompiler::SarosHandShader;
	constexpr uint64_t listed   = 0x9fba2edffc549531ull;
	constexpr uint64_t unlisted = 0x0123456789abcdefull;
	auto               on       = [](uint64_t h) { return [h] { return SarosHandShader(h); }; };
	Expect(!InChild("KYTY_SAROS_HANDSHADERS", nullptr, on(listed)), "SAROS hand shaders off when unset");
	Expect(!InChild("KYTY_SAROS_HANDSHADERS", "", on(listed)), "SAROS hand shaders off when empty");
	Expect(!InChild("KYTY_SAROS_HANDSHADERS", "0", on(listed)), "SAROS hand shaders off when 0");
	Expect(InChild("KYTY_SAROS_HANDSHADERS", "1", on(listed)), "SAROS hand shaders on for a listed hash");
	Expect(!InChild("KYTY_SAROS_HANDSHADERS", "1", on(unlisted)), "SAROS hand shaders never reach another title's hash");
	Expect(InChild("KYTY_SAROS_HANDSHADERS", "9fba2edffc549531", on(listed)), "subset list selects the hash");
	Expect(!InChild("KYTY_SAROS_HANDSHADERS", "a10a7fd8c1f0999c", on(listed)), "subset list excludes other hashes");
	std::printf("KytyOneTests: SAROS hand shader switch checked\n");
#else
	Expect(KYTY_ONE_REQUIRE_SAROS == 0, "sarosHandShaders.h missing after the SAROS merge");
#endif
}

} // namespace

int main() {
	TestExportTable();
	TestSarosHandShaders();
	if (g_failures != 0) {
		std::printf("KytyOneTests: %d failure(s)\n", g_failures);
		return 1;
	}
	std::printf("KytyOneTests: all passed\n");
	return 0;
}
