// nps5: with accurate occlusion queries off (KYTY_GPU_OCCLUSION=0, the guest's ZPASS dumps
// are answered synthetically), a draw that can write nothing is a query box and need not be
// recorded. Shuriken Ninja Castle issues ~10,500 such depth-test-only draws in each of three
// frames every 33 frames (3 x ~200 ms hitches).
//
// KYTY_OCCLUSION_QUERY_DRAW_SKIP (default 0 = off; 1 = on). Per draw every write is checked: no
// colour target, no depth write or clear, no stencil write or clear, no buffer/image/GDS write
// from any stage, no mesh shader. Anything else records as before.
#pragma once

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace Libs::Graphics::OcclusionDrawSkip {

struct DrawFacts {
	bool     occlusion_accurate = false; // KYTY_GPU_OCCLUSION=1: the draw's samples are counted
	bool     mesh               = false;
	uint32_t color_count        = 0;     // colour targets the draw writes
	bool     depth_write        = false;
	bool     depth_clear        = false;
	bool     stencil_write      = false; // stencil test enabled with a nonzero write mask
	bool     stencil_clear      = false;
	bool     shader_writes      = false; // any stage writes a buffer/image, atomics, addresses, GDS
};

enum class Verdict : uint8_t { Skip, Accurate, Mesh, Color, Depth, Stencil, ShaderWrites };

inline Verdict Classify(const DrawFacts& f) {
	if (f.occlusion_accurate) return Verdict::Accurate;
	if (f.mesh) return Verdict::Mesh;
	if (f.color_count != 0) return Verdict::Color;
	if (f.depth_write || f.depth_clear) return Verdict::Depth;
	if (f.stencil_write || f.stencil_clear) return Verdict::Stencil;
	if (f.shader_writes) return Verdict::ShaderWrites;
	return Verdict::Skip;
}

inline bool Enabled() {
	static const bool on = [] {
		const auto* v = std::getenv("KYTY_OCCLUSION_QUERY_DRAW_SKIP");
		return v != nullptr && v[0] != '\0' && std::strcmp(v, "0") != 0;
	}();
	return on;
}

struct Counters {
	std::atomic<uint64_t> n[7] {};
};
inline Counters& GetCounters() {
	static Counters c;
	return c;
}

} // namespace Libs::Graphics::OcclusionDrawSkip
