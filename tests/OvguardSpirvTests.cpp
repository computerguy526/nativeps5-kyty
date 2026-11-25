// ovguard: Metal override keys are KosmicKrisp's hash of the MSL of the SPIR-V Kyty emits. KYTY_GPU_ACCURACY=low gives every shader NOT in the
// fast-math deny list fast math, which changes its SPIR-V and so its key. A deny-listed shader must emit byte-identical SPIR-V at low and
// high accuracy; an unlisted one must differ (negative control: the guard can fail). A trivial synthetic compute program is enough: the
// float-controls execution mode that fast math drops is declared for every compute program. Builds and runs on arm64 macOS (no emulator).
#include "graphics/shader/recompiler/CodegenOptions.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <vector>

using namespace Libs::Graphics;
namespace SR = Libs::Graphics::ShaderRecompiler;

namespace {
int g_failures = 0;
void Expect(bool condition, const char* what) {
	if (!condition) {
		std::printf("OvguardSpirvTests: failed: %s\n", what);
		g_failures++;
	}
}

std::vector<uint32_t> Spirv(uint8_t accuracy, uint64_t hash, const std::vector<uint64_t>& deny) {
	auto options = SR::GetCodegenOptions();
	options.relaxed_float       = false;
	options.fast_math           = 0;
	options.fast_math_stages    = 3;
	options.gpu_accuracy        = accuracy; // 0 high, 1 normal, 2 low
	options.fast_math_allow_set = false;
	options.fast_math_deny      = deny;
	SR::SetCodegenOptions(options);

	const std::vector<uint32_t> code = {0xbf810000u}; // s_endpgm
	std::array<uint32_t, 64>    user_data {};
	user_data[3]  = 3u << 28u; // as MakeNativeUserData in the compute tests
	user_data[50] = 1u << 20u;
	user_data[51] = 3u << 28u;
	SR::CompileOptions          copt;
	copt.stage       = ShaderType::Compute;
	copt.shader_hash = hash;
	ShaderComputeInputInfo compute_info {};
	compute_info.host_subgroup_size = 64;
	copt.input_info.compute         = &compute_info;
	copt.user_data                  = user_data;
	auto translated                 = SR::TranslateProgram(code, copt);
	SR::IR::ResourceSpecialization specialization;
	return SR::CompileProgram(std::move(translated), copt, specialization).spirv;
}
} // namespace

int main() {
	constexpr uint64_t listed = 0x1122334455667788ull, other = 0x99aabbccddeeff00ull;
	const auto         saved  = SR::GetCodegenOptions();
	const auto         high_listed = Spirv(0, listed, {listed}), high_other = Spirv(0, other, {listed});
	Expect(!high_listed.empty(), "SPIR-V emitted");
	Expect(high_listed == high_other, "high accuracy: SPIR-V does not depend on the hash");
	Expect(Spirv(2, listed, {listed}) == high_listed, "low accuracy, deny-listed shader: SPIR-V == high (its override key is stable)");
	Expect(Spirv(2, other, {listed}) != high_other, "low accuracy, unlisted shader: SPIR-V differs (negative control)");
	SR::SetCodegenOptions(saved);
	if (g_failures == 0) {
		std::printf("OvguardSpirvTests: ok\n");
	}
	return g_failures == 0 ? 0 : 1;
}
