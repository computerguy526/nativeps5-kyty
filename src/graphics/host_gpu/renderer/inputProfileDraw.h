#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_INPUTPROFILEDRAW_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_INPUTPROFILEDRAW_H_

// KYTY_INPUT_PROFILE=1 draw/dispatch hooks (common/inputProfile.h): LOCAL-ONLY research capture of
// the values real draws read, for the offline shader validator. Default off: one inline bool test.

#include "common/inputProfile.h"

#include <cstdint>
#include <span>

namespace Libs::Graphics {
class BufferCache;
struct PreparedBindings;
struct ShaderVertexInputInfo;
} // namespace Libs::Graphics

namespace Libs::Graphics::InputProfileDraw {

void NoteImpl(BufferCache& cache, std::span<PreparedBindings* const> stages) noexcept;
void NoteVertexImpl(BufferCache& cache, const ShaderVertexInputInfo& info) noexcept;
void NoteMeshImpl(BufferCache& cache, const ShaderVertexInputInfo& info, uint32_t index_count,
                  uint32_t instances, uint64_t index_address, uint32_t index_size, uint32_t groups,
                  uint32_t segments) noexcept;

// After the bindings of a draw/dispatch are resolved (buffer_sources, shader_data final).
inline void Note(BufferCache& cache, std::span<PreparedBindings* const> stages) noexcept {
	if (InputProfile::On()) [[unlikely]] {
		NoteImpl(cache, stages);
	}
}
// After a draw's vertex buffers are acquired.
inline void NoteVertex(BufferCache& cache, const ShaderVertexInputInfo& info) noexcept {
	if (InputProfile::On()) [[unlikely]] {
		NoteVertexImpl(cache, info);
	}
}

// A direct mesh draw (geom2/mval): kind 'M' = the draw contract (index count, instances, index
// size, workgroups, restart segments, threads, LDS dwords, vertices|primitives per group; up to 4
// distinct) with the index buffer bytes as data; plus the draw's vertex buffers ('V').
inline void NoteMesh(BufferCache& cache, const ShaderVertexInputInfo& info, uint32_t index_count,
                     uint32_t instances, uint64_t index_address, uint32_t index_size,
                     uint32_t groups, uint32_t segments) noexcept {
	if (InputProfile::On()) [[unlikely]] {
		NoteMeshImpl(cache, info, index_count, instances, index_address, index_size, groups,
		             segments);
		NoteVertexImpl(cache, info);
	}
}

} // namespace Libs::Graphics::InputProfileDraw

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_INPUTPROFILEDRAW_H_ */
