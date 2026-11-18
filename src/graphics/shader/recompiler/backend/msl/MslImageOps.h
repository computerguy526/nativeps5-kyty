// T2 msl-mem: GCN image instructions (IMAGE_SAMPLE*/GATHER4*/LOAD/STORE/ATOMIC/GET_RESINFO/GET_LOD)
// -> Metal texture methods. Mirrors the SPIR-V emitter's address layout (Decoder::
// ImageAddressComponentLayout, sample-flag order offset, bias, dref, grads, coord, lod, clamp).
// Cubes are bound as 2D-array views (6 faces per cube): GCN cube coords are face-UV (x-1, y-1)
// plus a face index whose guest padding (8 slots per 6 faces) is removed.
// 1D / 1D-array images are bound as 2D / 2D-array textures with height 1 (Metal texture1d has no
// level/bias/gradient sampling); the manifest marks them "emulated_2d":true.
// Unsupported (reported via MemOut::error): converted-format images, mip-stats, indirect image
// descriptors, 1D-array gathers, GATHER horizontal, image cmpswap/inc/dec atomics.
#pragma once

#include "graphics/shader/recompiler/backend/msl/MslBinding.h"

#include <optional>

namespace Libs::Graphics::ShaderRecompiler::Msl::Mem {

namespace SampleFlag { // == Decoder::ImageSampleFlag
inline constexpr uint32_t Lod = 1u << 0, Bias = 1u << 1, Derivative = 1u << 2, Compare = 1u << 3,
                          Offset = 1u << 4, LevelZero = 1u << 5, LodClamp = 1u << 6, A16 = 1u << 7,
                          Cd = 1u << 8, GatherHorizontal = 1u << 9, Adjust = 1u << 10;
}

struct ImageOp {
	const ImageBinding*   img          = nullptr;
	const SamplerBinding* samp         = nullptr;
	uint32_t              sample_flags = 0;
	uint32_t              dmask        = 0xf;
	uint32_t              data_dwords  = 4;
	uint32_t              data_bits    = 32; // 16: results packed as half pairs
	uint32_t address_components = 0; // components present in the address (incl. lod/sample index)
	bool     has_mip            = false;
	bool     dynamic_mip        = false; // ImageResource::mip_mode == Dynamic
	bool     pixel_stage        = true;  // implicit LOD only valid in pixel shaders
	bool     lod_clamp_enabled  = true;
	bool     converted          = false; // conversion_format != invalid (unsupported)
	std::optional<uint32_t>
	    const_offset; // packed 6-bit signed x|y<<8|z<<16 when the address dword is an immediate
};

struct AddrLayout {
	uint32_t bit_offset, bit_width;
};
AddrLayout ImageAddressComponentLayout(uint32_t flags, uint32_t component);
uint32_t   ImageCoordComponents(ImgDim d); // coordinate comps incl. layer, excl. sample index
uint32_t   ImageSpatialComponents(ImgDim d);

// addr = the packed address dwords (uint expressions).
// mslinject kk-bda: sampler LOD bias from KK's sampled-image descriptor (+12, fp16) added to every
// LOD (`kk_sbias<id>` locals).
void SetKkSamplerBias(bool on);
// mslrest3 kk-bda: GET_LOD_STATS feedback of sampling pixel shaders (Kyty KYTY_LOD_STATS_MODE=gpu).
// Needs locals `push` (shader data), `kk_ms_a` (ulong MipStats buffer address) and `kk_ms_n` (its
// byte size). on=false: no records.
struct MipStatsConfig {
	bool     on           = false;
	uint32_t offset_dword = 0;
	uint32_t count        = 0;
};
void        SetMipStats(const MipStatsConfig& c);
std::string ImageSample(MemOut& o, const ImageOp& op, const std::vector<std::string>& addr,
                        bool gather);
std::string ImageLoad(MemOut& o, const ImageOp& op, const std::vector<std::string>& addr,
                      const std::string& active = "true");
void        ImageStore(MemOut& o, const ImageOp& op, const std::vector<std::string>& addr,
                       const std::string& data_uint4, const std::string& active = "true");
std::string ImageAtomic32(MemOut& o, const ImageOp& op, const std::vector<std::string>& addr,
                          int atomic_op_kind, const std::string& value,
                          const std::string& active = "true");
std::string ImageQueryDims(MemOut& o, const ImageOp& op, const std::vector<std::string>& addr);
std::string ImageQueryLod(MemOut& o, const ImageOp& op, const std::vector<std::string>& addr);

} // namespace Libs::Graphics::ShaderRecompiler::Msl::Mem
