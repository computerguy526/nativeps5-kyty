// ovguard: the fast-math deny list guards the Metal override keys (KosmicKrisp keys of the SPIR-V Kyty emits move when a shader gets fast
// math). CodegenOptions only, no GPU: the list must load or be reported loudly, and a listed shader never gets fast math.
#include "graphics/shader/recompiler/CodegenOptions.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

using namespace Libs::Graphics::ShaderRecompiler;

namespace {
int g_failures = 0;
void Expect(bool condition, const char* what) {
	if (!condition) {
		std::printf("FastMathDenyTests: failed: %s\n", what);
		g_failures++;
	}
}
} // namespace

int main() {
	const auto dir  = std::filesystem::temp_directory_path() / ("ovguard_deny_" + std::to_string(getpid()));
	std::filesystem::create_directories(dir);
	const auto err  = (dir / "stderr.txt").string();
	const auto good = (dir / "good.txt").string();
	{
		std::ofstream(good) << "# list\n0123456789abcdef\nfedcba9876543210 # tail\n";
	}

	// Start-up (first use of the options): a missing deny file under KYTY_GPU_ACCURACY=low prints one ERROR line with deny=missing.
	setenv("KYTY_GPU_ACCURACY", "low", 1);
	setenv("KYTY_FAST_MATH_DENY", "/no/such/ovguard/deny.txt", 1);
	std::fflush(stderr);
	const int saved_stderr = dup(2);
	std::freopen(err.c_str(), "w", stderr);
	CodegenOptions options = GetCodegenOptions();
	std::fflush(stderr);
	dup2(saved_stderr, 2);
	close(saved_stderr);
	std::ostringstream log;
	log << std::ifstream(err).rdbuf();
	Expect(log.str().find("[fastmath] ERROR") != std::string::npos && log.str().find("deny=missing") != std::string::npos,
	       "missing deny file under ACCURACY=low logs the ERROR line");
	Expect(options.fast_math_deny.empty(), "missing deny file leaves an empty list");
	Expect(options.gpu_accuracy == 2, "accuracy low parsed");

	std::vector<uint64_t> hashes;
	Expect(FastMathDenyProblem(nullptr) == nullptr, "unset is fine");
	Expect(FastMathDenyProblem("") != nullptr, "empty value reported");
	Expect(FastMathDenyProblem("/no/such/ovguard/deny.txt") != nullptr, "missing file reported");
	{
		std::ofstream(dir / "empty.txt") << "# comments only\n\n";
	}
	Expect(FastMathDenyProblem((dir / "empty.txt").c_str()) != nullptr, "file without a hash reported");
	Expect(FastMathDenyProblem(good.c_str(), &hashes) == nullptr && hashes.size() == 2, "good list loads");

	// Key stability (logic level; the SPIR-V level is CheckFastMathDenyKeyStability in shader_recompiler_compute_tests): under low accuracy a
	// listed shader never gets fast math, an unlisted one does, at both stages that carry overrides.
	options.fast_math          = 0;
	options.fast_math_stages   = 3;
	options.gpu_accuracy       = 2;
	options.fast_math_allow_set = false;
	options.fast_math_deny     = hashes;
	SetCodegenOptions(options);
	const uint64_t listed = 0x0123456789abcdefull, other = 0x1111111111111111ull;
	Expect(!AllFastApplies(listed, "cs") && !AllFastApplies(listed, "fs"), "listed shader keeps exact math (low)");
	Expect(AllFastApplies(other, "cs") && AllFastApplies(other, "fs"), "unlisted shader gets fast math (low): negative control");
	options.gpu_accuracy = 0;
	SetCodegenOptions(options);
	Expect(!AllFastApplies(other, "cs"), "high accuracy: nothing gets fast math");

	std::filesystem::remove_all(dir);
	if (g_failures == 0) {
		std::printf("FastMathDenyTests: ok\n");
	}
	return g_failures == 0 ? 0 : 1;
}
