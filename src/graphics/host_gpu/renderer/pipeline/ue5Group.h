#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_UE5GROUP_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_UE5GROUP_H_

#include "graphics/host_gpu/renderer/pipeline/pipelineList.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <span>
#include <vector>
#include <xxhash.h>

// Splits a decoded UE5 shader library group into the shader codes the guest later passes to
// AgcCreateShader.
//
// A group holds, per shader: a preamble, the AGC shader header (the `Shader` struct of
// graphics/shader/shader.h: the bytes "1234" and version 0x18, then the header's fields), the ISA,
// padding and a footer. The marker is the header's first 8 bytes; the code sits `*(u32*)(marker -
// 0x14)` bytes after the marker (the offset the library tooling records ahead of the header) and is
// `*(u32*)(marker - 0x0c)` bytes long (validated against real groups). The header's own shader_size
// (marker + 0x44) is only a cross-check: when it differs, the -0x0c value is used and the shader is
// counted in Stats::size_mismatch. The code is hashed as the guest
// hashes what it passes to AgcCreateShader (XXH3-128 over the bytes, XXH3-64 of the first min(64,
// size) bytes) and so is the key of a KYPLST `<runtime>` shader.
//
// A group the library lists with another shader count than it has markers (the ray tracing layouts)
// is split all the same, with the stage unknown: the order of its shaders cannot be matched to the
// library's entries.
namespace Libs::Graphics::Ue5Group {

inline constexpr uint8_t Marker[8] = {'1', '2', '3', '4', 0x18, 0, 0, 0};
inline constexpr size_t  CodeOffsetBack =
    0x14;                                // the u32 code offset sits this far before the marker
inline constexpr size_t SizeBack = 0x0c; // the u32 code size sits this far before the marker
inline constexpr size_t ShaderSizeAt =
    0x44; // the header's shader_size, from the marker (cross-check)
inline constexpr uint32_t UnknownStage = 0xFFFFFFFFu;

// The per-UE-version knobs of the split, as data: adding a UE version is one row here and one
// fixture in the tests. `name` is the word the profile gives (KYTY_UE5_LAYOUT; unset = the first
// row); an unknown name turns the feature off with one log line (ue5CodeWarm.h).
struct Layout {
	const char* name;
	uint8_t     marker[8];        // the AGC shader header's first 8 bytes
	size_t      code_offset_back; // the u32 code offset sits this far before the marker
	size_t      size_back;        // the u32 code size sits this far before the marker
	size_t      shader_size_at;   // the header's shader_size, from the marker (cross-check)
};
inline constexpr Layout Layouts[] = {
    {"agc18", {'1', '2', '3', '4', 0x18, 0, 0, 0}, CodeOffsetBack, SizeBack, ShaderSizeAt},
};
[[nodiscard]] inline const Layout* FindLayout(const char* name) {
	if (name == nullptr || *name == '\0') return &Layouts[0];
	for (const auto& layout: Layouts)
		if (std::strcmp(layout.name, name) == 0) return &layout;
	return nullptr;
}

struct Shader {
	uint32_t                 stage = UnknownStage; // the library entry's frequency (low 4 bits)
	uint32_t                 size  = 0;            // code bytes
	PipelineList::Hash128    hash;                 // XXH3-128 of the code
	uint64_t                 prefix = 0;           // XXH3-64 of the first min(64, size) bytes
	std::span<const uint8_t> code;                 // inside the group
};

struct Stats {
	uint32_t markers       = 0;     // headers found
	uint32_t shaders       = 0;     // yielded
	uint32_t out_of_bounds = 0;     // a code outside the group (dropped)
	uint32_t bad_size      = 0;     // size 0 or not a multiple of 4 (dropped)
	uint32_t size_mismatch = 0;     // the header's shader_size differs from the size used (kept)
	bool     matched       = false; // markers == the library's shader count: stages known
};

namespace Detail {
template <typename T>
inline T Get(std::span<const uint8_t> bytes, size_t at) {
	T value;
	std::memcpy(&value, bytes.data() + at, sizeof(value));
	return value;
}
} // namespace Detail

// `frequencies`: the library's per-shader stage values in group order
// (KytyGpuUe5Group::frequencies); may be empty. The result's spans point into `group`.
[[nodiscard]] inline std::vector<Shader> Split(std::span<const uint8_t>  group,
                                               std::span<const uint32_t> frequencies,
                                               Stats*                    stats  = nullptr,
                                               const Layout*             layout = nullptr) {
	if (layout == nullptr) layout = &Layouts[0];
	const auto& marker_          = layout->marker;
	const auto  CodeOffsetBack_ = layout->code_offset_back;
	const auto  SizeBack_       = layout->size_back;
	const auto  ShaderSizeAt_   = layout->shader_size_at;
	Stats               local;
	std::vector<Shader> found;
	std::vector<size_t> markers,
	    ordinal;           // header offsets; per yielded shader the index of its header
	size_t skip_until = 0; // markers inside an accepted code are not headers
	for (size_t at = 0; at + sizeof(marker_) <= group.size();) {
		const auto* hit = static_cast<const uint8_t*>(
		    std::memchr(group.data() + at, marker_[0], group.size() - sizeof(marker_) + 1 - at));
		if (hit == nullptr) break;
		at = static_cast<size_t>(hit - group.data());
		if (at >= skip_until && std::memcmp(hit, marker_, sizeof(marker_)) == 0) {
			markers.push_back(at);
			if (at >= CodeOffsetBack_ && at + sizeof(marker_) <= group.size()) {
				const uint64_t start = at + Detail::Get<uint32_t>(group, at - CodeOffsetBack_);
				const uint32_t size  = Detail::Get<uint32_t>(group, at - SizeBack_);
				if (at + ShaderSizeAt_ + 4 <= group.size() &&
				    Detail::Get<uint32_t>(group, at + ShaderSizeAt_) != size) {
					local.size_mismatch++;
				}
				if (size == 0 || size % 4u != 0) {
					local.bad_size++;
				} else if (start + size > group.size()) {
					local.out_of_bounds++;
				} else {
					Shader shader;
					shader.size  = size;
					shader.code  = group.subspan(static_cast<size_t>(start), size);
					const auto h = XXH3_128bits(shader.code.data(), shader.code.size());
					shader.hash  = {h.low64, h.high64};
					shader.prefix =
					    XXH3_64bits(shader.code.data(), std::min<size_t>(64, shader.code.size()));
					found.push_back(shader);
					ordinal.push_back(markers.size() - 1);
					skip_until = static_cast<size_t>(start + size);
					at += sizeof(marker_);
					continue;
				}
			} else {
				local.out_of_bounds++;
			}
			at += sizeof(marker_);
			continue;
		}
		at++;
	}
	local.markers = static_cast<uint32_t>(markers.size());
	local.matched = !frequencies.empty() && markers.size() == frequencies.size();
	if (local.matched) {
		// Stages follow the headers (a dropped header still takes its library entry).
		for (size_t i = 0; i < found.size(); i++)
			found[i].stage = frequencies[ordinal[i]] & 0xFu;
	}
	local.shaders = static_cast<uint32_t>(found.size());
	if (stats != nullptr) *stats = local;
	return found;
}

} // namespace Libs::Graphics::Ue5Group

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_UE5GROUP_H_
