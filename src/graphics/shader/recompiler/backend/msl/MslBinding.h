// T2 msl-mem: resource binding model -> Metal argument buffer + JSON manifest (schema v1, see
// work/mslnative/STATUS.md "T2 msl-mem"). Plain structs so the emitters, the converter from
// IR::ShaderInfo (MslBindingFromIr.cpp) and the tests share them without pulling in the IR.
#pragma once

#include "graphics/shader/recompiler/backend/msl/MslMemCommon.h"

#include <array>
#include <map>
#include <optional>

namespace Libs::Graphics::ShaderRecompiler::Msl::Mem {

enum class ImgDim : uint8_t { D1, D1Array, D2, D2Array, D2Ms, D2MsArray, D3, Cube };
enum class Numeric : uint8_t { Float, Uint, Sint };
enum class Access : uint8_t { Read, Write, ReadWrite, Atomic, Sample };

struct BufferBinding {
	uint32_t id      = 0;
	bool     read    = true;
	bool     written = false;
	bool     atomic  = false;
	uint32_t bytes   = 65536; // harness allocation hint (max_byte_extent when known)
	uint32_t packed_stride =
	    0; // IR packed stride word: stride:0-13 swizzle:14 idx_stride:16-17 add_tid:20
	uint32_t    format       = 0;      // Prospero::BufferFormat of the V# (0 = raw)
	std::string init         = "seed"; // manifest init: seed|zero|seedf
	bool        bounds_check = true;   // emit explicit `offset+size <= b<i>_size` guard
};

struct ImageBinding {
	uint32_t    id      = 0;
	ImgDim      dim     = ImgDim::D2;
	Numeric     numeric = Numeric::Float;
	Access      access  = Access::Sample;
	bool        depth   = false;
	uint32_t    mips    = 1;
	uint32_t    width = 64, height = 64, depth_or_layers = 1;
	std::string pixel_format = "rgba32float";
	uint32_t    shader_swizzle =
	    4u | (5u << 3) | (6u << 6) |
	    (7u << 9); // GCN dst_sel per source channel (4+component); identity default
};

struct SamplerBinding {
	uint32_t    id      = 0;
	bool        compare = false;
	bool        linear  = true;
	std::string address = "clamp";
};

struct BindingModel {
	std::string             stage = "compute";
	std::string             entry = "main_entrypoint";
	std::array<uint32_t, 3> threadgroup {64, 1, 1};
	uint32_t                lds_bytes  = 0; // threadgroup memory (0 = none)
	uint32_t                push_bytes = 0; // 0 = no push constants
	uint32_t gds_bytes = 0; // GDS counters (DS_APPEND/CONSUME on GDS): device buffer, 0 = unused
	uint32_t srt_bytes = 0; // flattened SRT dwords (ReadConst), host-filled; 0 = unused
	std::vector<BufferBinding>  buffers;
	std::vector<ImageBinding>   images;
	std::vector<SamplerBinding> samplers;
};

inline constexpr uint32_t RootBufferIndex = 0;
inline constexpr uint32_t PushBufferIndex = 1;

std::string BufName(uint32_t id);     // "r.b3"   (device uchar*)
std::string BufSizeName(uint32_t id); // "r.b3_size"
std::string TexName(uint32_t id);     // "r.t2"
std::string SampName(uint32_t id);    // "r.s1"
const char* MslTextureType(
    const ImageBinding& img); // "texture2d<float, access::sample>" (storage in static buf)
std::string TextureDecl(const ImageBinding& img);

// Byte offsets of each field inside the root block `Resources` (natural layout).
struct RootLayout {
	std::map<uint32_t, uint32_t> buffer_ptr, buffer_size, texture, sampler;
	uint32_t                     srt_ptr = UINT32_MAX, gds_ptr = UINT32_MAX;
	uint32_t                     bytes = 8;
};
RootLayout ComputeRootLayout(const BindingModel& m);

// `struct Resources { ... };` bound as `constant Resources& r [[buffer(0)]]` (plain struct, no
// [[id]]: the harness patches pointers / resource ids at ComputeRootLayout offsets).
std::string ArgumentBufferStruct(const BindingModel& m);
// Entry-point parameter list tail: resources, push constants, threadgroup LDS (no thread-id
// params).
std::string EntryParams(const BindingModel& m);
// Manifest JSON (schema 1). convention is "msl-native-v1", "kk-bda" or "custom".
std::string ManifestJson(const BindingModel& m, const std::string& convention = "msl-native-v1");

} // namespace Libs::Graphics::ShaderRecompiler::Msl::Mem
