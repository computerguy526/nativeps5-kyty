// nps5: ShaderRestoreVertexAliases (shaderVertexAliases.h) against BuildStageStaticKey: a vertex input
// info cut to its stored bytes (as shader journals and pipeline lists keep it) gets its PS parameter
// aliases back from the static key, so the replay derives the recorded key again.
#include "graphics/shader/shaderCompiler.h"
#include "graphics/shader/shaderVertexAliases.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

using namespace Libs::Graphics;

// The two assert handlers BuildStageStaticKey links against (the test links no common library).
namespace Common {
int DbgExitIfHandler(const char* expression, const char* file, int line) { std::printf("EXIT_IF %s at %s:%d\n", expression, file, line); return 0; }
void DbgExit(int code) { std::exit(code); }
} // namespace Common

namespace {
int g_checks = 0, g_failed = 0;
void Check(bool ok, const char* what, int n) {
	g_checks++;
	if (!ok) { g_failed++; std::printf("FAILED %s (case %d)\n", what, n); }
}
// What a journal keeps of an info, loaded back as the replay loads it.
std::unique_ptr<ShaderVertexInputInfo> Stored(const ShaderVertexInputInfo& info) {
	auto out = std::make_unique<ShaderVertexInputInfo>();
	std::memcpy(static_cast<void*>(out.get()), static_cast<const void*>(&info), ShaderStoredInputInfoSize<ShaderVertexInputInfo>());
	return out;
}
} // namespace

int main() {
	std::mt19937 rng(20261116);
	for (int n = 0; n < 4000; n++) {
		auto info = std::make_unique<ShaderVertexInputInfo>();
		info->resources_num       = static_cast<int>(rng() % 8);
		info->wave_size           = (rng() & 1) != 0 ? 32 : 64;
		info->scratch_size_dwords = rng() % 5;
		info->pa_cl_vs_out_cntl   = rng();
		info->lds_size_dwords     = (rng() % 3) == 0 ? 1 + rng() % 1000 : 0;
		info->user_clip_planes    = (rng() % 4) == 0 ? 1 + rng() % 63 : 0;
		info->clip_space.enabled  = (rng() & 1) != 0;
		// no aliases, copies (source only), field aliases (upper bits), every destination, edge words
		const int kind = n % 5;
		if (kind != 0) {
			info->param_alias_mask = kind == 3 ? 0xffffffffu : kind == 4 ? (1u << (rng() % 32)) : rng();
			if (info->param_alias_mask == 0) info->param_alias_mask = 0x80000001u;
			for (uint32_t d = 0; d < 32; d++) {
				if ((info->param_alias_mask & (1u << d)) == 0) continue;
				const uint32_t source = rng() % 32;
				info->param_alias_source[d] = kind == 1 ? source : kind == 4 ? (n % 2 != 0 ? 0x7A11A5u : 0xffffffffu) : (source | (rng() << 5));
			}
		}
		std::vector<uint32_t> key, again;
		BuildStageStaticKey(*info, key);
		auto stored = Stored(*info);
		stored->param_alias_mask = 0x1234u; // stale words of an earlier use must not survive
		stored->param_alias_source.fill(0xdeadbeefu);
		Check(ShaderRestoreVertexAliases(*stored, key), "restore accepts a key BuildStageStaticKey wrote", n);
		Check(stored->param_alias_mask == info->param_alias_mask, "mask", n);
		Check(stored->param_alias_source == info->param_alias_source, "sources (field codes in the upper bits)", n);
		BuildStageStaticKey(*stored, again);
		Check(again == key, "the restored info derives the recorded key", n);
		// The old replay (aliases cleared) derives another key exactly when there are aliases.
		auto cleared = Stored(*info);
		cleared->param_alias_mask = 0;
		BuildStageStaticKey(*cleared, again);
		Check((again == key) == (info->param_alias_mask == 0), "cleared aliases differ only for alias keys", n);
		// Truncated alias words: refused, info left without aliases.
		if (info->param_alias_mask != 0) {
			const size_t at = 8u + (info->lds_size_dwords != 0 ? 1u : 0u) + (info->user_clip_planes != 0 ? 1u : 0u);
			for (size_t cut: {at + 1, at + 2, at + 1 + static_cast<size_t>(std::popcount(info->param_alias_mask))}) {
				std::vector<uint32_t> shorter(key.begin(), key.begin() + static_cast<std::ptrdiff_t>(cut));
				auto bad = Stored(*info);
				Check(!ShaderRestoreVertexAliases(*bad, shorter) && bad->param_alias_mask == 0, "truncated key refused", n);
			}
			std::vector<uint32_t> zero_mask = key;
			zero_mask[at + 1] = 0;
			auto bad = Stored(*info);
			Check(!ShaderRestoreVertexAliases(*bad, zero_mask) && bad->param_alias_mask == 0, "tag with an empty mask refused", n);
		}
	}
	// Empty and short keys: no aliases, accepted.
	auto info = std::make_unique<ShaderVertexInputInfo>();
	info->param_alias_mask = 5;
	Check(ShaderRestoreVertexAliases(*info, {}) && info->param_alias_mask == 0, "empty key", -1);
	std::printf("shader_vertex_alias_tests: %d checks, %d failed\n", g_checks, g_failed);
	return g_failed == 0 ? 0 : 1;
}
