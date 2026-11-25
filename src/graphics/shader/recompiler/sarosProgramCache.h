#pragma once

// sarosperf (restored on v14, v1.4 review M15): the SAROS Nanite-raw / sample-fract switches change a translation, so such
// a program used to bypass the persistent program cache. With KYTY_SAROS_PROGRAM_CACHE=1 it is cacheable when the switch
// state is part of the key: SarosCacheVariant is 0 (key bytes unchanged) unless one of the switches applies to the
// program. Default off = the bypass (only reachable with the SAROS switches set). Header only, no GPU; the switches are
// read once per process (tests/SarosProgramCacheTests.cpp runs one process per state).

#include "graphics/shader/recompiler/sarosHandShaders.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace Libs::Graphics::ShaderRecompiler {

// true = programs with a non-zero variant bypass the cache (the default); KYTY_SAROS_PROGRAM_CACHE=1 turns the bypass off.
inline bool SarosCacheBypass() {
	static const bool bypass = [] {
		const auto* v = std::getenv("KYTY_SAROS_PROGRAM_CACHE");
		return v == nullptr || std::strcmp(v, "1") != 0;
	}();
	return bypass;
}

// Bump kSarosVariantVersion when the hand-shader / nanite-raw translation changes without a codegen-fingerprint change
// (a clean build already gets a new cache header).
inline uint64_t SarosCacheVariant(uint64_t shader_hash, bool is_mesh_stage) {
	constexpr uint64_t kSarosVariantVersion = 1;
	uint64_t           variant              = 0;
	const auto         mix                  = [&variant](uint64_t v) { variant = (variant ^ v) * 0x100000001b3ull + 0x9e3779b97f4a7c15ull; };
	// KYTY_SAROS_HANDSHADERS / SAMPLE_LOD0 / SAMPLE_DBG / KEYSTRIDE / MAP_PAD have no codegen effect on v14 (HANDSHADERS only logs,
	// ResourceMaterialization.cpp): they are not part of the variant, so those programs stay cacheable (v14rev2 v14fix #1).
	if (SarosSampleFract(shader_hash)) {
		mix(8);
		mix(kSarosVariantVersion);
	}
	if (SarosNaniteRawPs(shader_hash)) {
		mix(2);
		mix(kSarosVariantVersion);
	}
	if (SarosNaniteRawMesh() && is_mesh_stage) {
		static const uint64_t masks = [] {
			uint64_t m = 0;
			for (uint32_t i = 0; i < 32; i++) {
				m |= (SarosNaniteRawAnd(i) ? 1ull : 0ull) << i;
				m |= (SarosNaniteRawVerts(i) ? 1ull : 0ull) << (32 + i);
			}
			return m;
		}();
		mix(3);
		mix(kSarosVariantVersion);
		mix(masks);
	}
	return variant;
}

// The cache gate for one program (pipelineCache.cpp DiskEnabled, minus the dump switches).
inline bool SarosProgramMayUseDiskCache(uint64_t shader_hash, bool is_mesh_stage) {
	return !(SarosCacheBypass() && SarosCacheVariant(shader_hash, is_mesh_stage) != 0);
}

} // namespace Libs::Graphics::ShaderRecompiler
