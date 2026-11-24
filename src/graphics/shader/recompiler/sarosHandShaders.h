#pragma once

// sarosshade: KYTY_SAROS_HANDSHADERS=1 (bring-up, default off). Hand-picked per-shader handling for
// the bindless lighting kernels of SAROS (PPSA07631, UE5.5), keyed by the exact guest shader hash
// so no other title can reach it. These programs take 5-8 T# per light/probe record from the
// bindless descriptor heap (a V# in user data), the heap index read from a per-record constant
// buffer inside a scalar loop (record = CB[i * 0xa0]). The general tracker refuses them ("computes
// a descriptor at runtime that has no BDA path"), so their dispatches are dropped and indirect
// diffuse stays black. With the switch, the listed hashes
//  - take the indirect-image plan (ResourceTracking TryMakeIndirectImage) with a loop-counter
//    selector: the host reads every record's heap index before the dispatch and binds those images,
//  - bypass the persistent program cache (pipelineCache.cpp), so the stored "skip" verdict is not
//    used and nothing translated under the switch is written back.
// KYTY_SAROS_HANDSHADERS=<hex,hex,...> restricts the switch to a subset of the list (bisect aid).

#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace Libs::Graphics::ShaderRecompiler {

inline bool SarosHandShader(uint64_t hash) {
	static constexpr uint64_t List[] = {
	    0x9fba2edffc549531ull, 0xa10a7fd8c1f0999cull, 0x97ee21b58604dbceull, 0x0564916e428c0919ull,
	    0xbd8b409b73ca2ee2ull, 0x9b855d35b0a8701eull, 0x905b2e2c09c61a5dull, 0x2be7cda6416b3708ull,
	};
	static const char* const value = std::getenv("KYTY_SAROS_HANDSHADERS");
	if (value == nullptr || value[0] == '\0' || std::strcmp(value, "0") == 0) return false;
	bool listed = false;
	for (const auto entry: List)
		listed = listed || entry == hash;
	if (!listed) return false;
	if (std::strcmp(value, "1") == 0) return true;
	for (const char* cursor = value; *cursor != '\0';) {
		char* end = nullptr;
		if (std::strtoull(cursor, &end, 16) == hash && end != cursor) return true;
		if (end == cursor) break;
		cursor = *end == ',' ? end + 1 : end;
		if (*end != ',') break;
	}
	return false;
}

} // namespace Libs::Graphics::ShaderRecompiler

namespace Libs::Graphics::ShaderRecompiler {

// sarosperf2: KYTY_SAROS_HS_DYNIDX=1 (default off; needs KYTY_SAROS_HANDSHADERS). The indirect
// image tables of the listed hand shaders get one canonical layout, so a kernel has one SPIR-V
// permutation instead of one per (distinct-key count, child count) of its roots: every root keeps
// its key table at a fixed 2^7-entry tier (constant mapping offset / 7 search iterations), roots of
// equal image class share one candidate group, and the group is padded to a fixed number of image
// slots (64 in all), the key table maps a key to the slot's position in the group, and the shader
// picks the slot with a dynamic descriptor-array index instead of a per-root constant switch.
inline bool SarosHsDynIdx() {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_SAROS_HS_DYNIDX");
		return v != nullptr && std::strcmp(v, "1") == 0;
	}();
	return on;
}

// sarosnanite: KYTY_SAROS_NANITE_RAW=1 (bring-up, default off). SAROS's Nanite hardware raster
// pixel shaders (mesh pipeline) find their triangle without SV_PrimitiveID: every vertex carries a
// 128-bit mask of the cluster triangles that use it (mesh export param 1, built with DS_OR), and
// the pixel shader ANDs the three raw vertices' masks (V_INTERP_MOV P0/P10/P20) and takes the
// lowest set bit (V_FFBL) as the triangle index it writes into the visibility buffer
// (IMAGE_ATOMIC_UMAX, 64-bit). KosmicKrisp has no per-vertex fragment inputs and Kyty's unpack
// replacement only links after an ordinary vertex shader ("reads unpacked parameter fields after a
// mesh program: undefined"), so the triangle index was garbage: the material pass then shaded each
// pixel with another triangle of the cluster (per-triangle mosaic of normals / UVs / base colour).
// With the switch:
//  - every mesh program also writes, per primitive, the AND of its three vertices' parameter
//    dwords (uvec4, PerPrimitiveEXT) at location SarosNaniteRawLocation(parameter location);
//  - the listed pixel shaders keep their raw-vertex reads (no unpack pass) and read all three
//    vertices of a raw parameter from that per-primitive input. For these shaders that is exact:
//    they only AND the three reads (param 1) or read a cluster-uniform value at P0 (param 0).
//  - both bypass the persistent program cache.
// KYTY_SAROS_NANITE_RAW=<hex,...> lists other pixel shader hashes instead of the built-in four.
inline const char* SarosNaniteRawValue() {
	static const char* const value = std::getenv("KYTY_SAROS_NANITE_RAW");
	return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0 ? value : nullptr;
}
inline bool SarosNaniteRawMesh() {
	return SarosNaniteRawValue() != nullptr;
}
inline bool SarosNaniteRawPs(uint64_t hash) {
	static constexpr uint64_t List[] = {
	    0xda18f5091bbf4843ull,
	    0x034b248b0927c4a2ull,
	    0x75bce95a35765ef0ull,
	    0x865a72566b298994ull,
	};
	const char* value = SarosNaniteRawValue();
	if (value == nullptr) return false;
	if (std::strcmp(value, "1") == 0) {
		for (const auto entry: List) {
			if (entry == hash) return true;
		}
		return false;
	}
	for (const char* cursor = value; *cursor != '\0';) {
		char* end = nullptr;
		if (std::strtoull(cursor, &end, 16) == hash && end != cursor) return true;
		if (end == cursor || *end != ',') break;
		cursor = end + 1;
	}
	return false;
}
// Which parameter locations a mesh program forwards per primitive: as the AND of the three vertices
// (KYTY_SAROS_NANITE_RAW_AND, bit mask, default 0x2 = param 1, the triangle masks) and as the three
// vertices themselves (KYTY_SAROS_NANITE_RAW_VERTS, default 0x4 = param 2, the vertex indices the
// masked-material raster PS 865a7256 reads at P0/P10/P20). Metal bounds the mesh output size, so
// only what the listed pixel shaders read is forwarded.
inline uint32_t SarosNaniteRawEnvMask(const char* name, uint32_t fallback) {
	const char* value = std::getenv(name);
	return value != nullptr && value[0] != '\0'
	           ? static_cast<uint32_t>(std::strtoul(value, nullptr, 0))
	           : fallback;
}
inline bool SarosNaniteRawAnd(uint32_t location) {
	static const uint32_t mask = SarosNaniteRawEnvMask("KYTY_SAROS_NANITE_RAW_AND", 0x2u);
	return SarosNaniteRawValue() != nullptr && location < 32u &&
	       (mask >> location & 1u) != 0u; // only with KYTY_SAROS_NANITE_RAW
}
inline bool SarosNaniteRawVerts(uint32_t location) {
	static const uint32_t mask = SarosNaniteRawEnvMask("KYTY_SAROS_NANITE_RAW_VERTS", 0x4u);
	return SarosNaniteRawValue() != nullptr && location < 32u &&
	       (mask >> location & 1u) != 0u; // only with KYTY_SAROS_NANITE_RAW
}
// Per-primitive locations (spare: guest parameters use the low ones): the AND at 16 + location,
// vertex k at 20 + 3 * (location & 3) + k.
inline uint32_t SarosNaniteRawLocation(uint32_t location) {
	return 16u + (location & 3u);
}
inline uint32_t SarosNaniteRawVertexLocation(uint32_t location, uint32_t vertex) {
	return 20u + 3u * (location & 3u) + vertex;
}

} // namespace Libs::Graphics::ShaderRecompiler

namespace Libs::Graphics::ShaderRecompiler {

// sarosmip: KYTY_SAROS_SAMPLE_LOD0=<hex,hex,...>|all (bring-up experiment, default off). The listed
// shaders' IMAGE_SAMPLE_D (explicit gradient) samples use an explicit LOD 0 instead of the
// gradients; "all" applies to every shader. Discriminates "gradient/LOD picks the smallest mip"
// from "UV or texture content is flat".
inline bool SarosSampleLod0(uint64_t hash) {
	static const char* const value = std::getenv("KYTY_SAROS_SAMPLE_LOD0");
	if (value == nullptr || value[0] == '\0' || std::strcmp(value, "0") == 0) return false;
	if (std::strcmp(value, "all") == 0) return true;
	for (const char* cursor = value; *cursor != '\0';) {
		char* end = nullptr;
		if (std::strtoull(cursor, &end, 16) == hash && end != cursor) return true;
		if (end == cursor || *end != ',') break;
		cursor = end + 1;
	}
	return false;
}

// sarosuv: KYTY_SAROS_SAMPLE_DBG=<hex,hex,...>|all (bring-up experiment, default off): the listed
// pixel shaders' 2D float image samples return (coord.x, coord.y, 0, 1) instead of the texel
// (spirvEmitterImage.cpp).
inline bool SarosSampleDbg(uint64_t hash) {
	static const char* const value = std::getenv("KYTY_SAROS_SAMPLE_DBG");
	if (value == nullptr || value[0] == '\0' || std::strcmp(value, "0") == 0) return false;
	if (std::strcmp(value, "all") == 0) return true;
	for (const char* cursor = value; *cursor != '\0';) {
		char* end = nullptr;
		if (std::strtoull(cursor, &end, 16) == hash && end != cursor) return true;
		if (end == cursor || *end != ',') break;
		cursor = end + 1;
	}
	return false;
}

// sarosaddr: KYTY_SAROS_SAMPLE_FRACT=<hex,..|all> (bring-up experiment, default off): the listed
// pixel shaders' 2D float samples use fract(coord) (software wrap) to tell a host sampler wrap
// failure from wrong coordinates.
inline bool SarosSampleFract(uint64_t hash) {
	static const char* const value = std::getenv("KYTY_SAROS_SAMPLE_FRACT");
	if (value == nullptr || value[0] == '\0' || std::strcmp(value, "0") == 0) return false;
	if (std::strcmp(value, "all") == 0) return true;
	for (const char* cursor = value; *cursor != '\0';) {
		char* end = nullptr;
		if (std::strtoull(cursor, &end, 16) == hash && end != cursor) return true;
		if (end == cursor || *end != ',') break;
		cursor = end + 1;
	}
	return false;
}

} // namespace Libs::Graphics::ShaderRecompiler
