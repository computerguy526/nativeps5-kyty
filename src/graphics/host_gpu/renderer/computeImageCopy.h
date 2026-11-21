#pragma once
// rtimage KYTY_COMPUTE_IMAGE_COPY (default off): a guest compute pass that copies one surface's
// memory to another's through two formatted buffers (a dword copy kernel) forces the source render
// target through the tiler into the buffer cache, and the destination back through the detiler on
// its next use. When the kernel is proven a whole-range element copy (IR::AnalyzeUniformCopy), the
// descriptors make the element copy a bit copy (ResolveComputeBufferCopy), the source range is
// exactly one cached image whose native texels are the range's contents, and the destination
// range is exactly one cached image of the same layout, the dispatch is replaced by a native image
// copy (TextureCache::CopyImageFromBuffers). Anything else runs the guest shader as before.
//   1              on
//   verify-kernel  the guest shader runs as before; afterwards the two buffer ranges are compared
//                  on the GPU (KYTY_TILER_TRACE: "count verify.imgcopy-kernel", bytes = mismatches)
//   verify         on, and the source and the destination image are both tiled out afterwards and
//                  compared ("count verify.imgcopy-image")
// Neither the shader translation nor the codegen fingerprint depends on it.
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"

#include <cstdint>
#include <optional>
#include <span>

namespace Libs::Graphics::ShaderRecompiler {
struct CompileOptions;
}

namespace Libs::Graphics::ComputeImageCopy {

enum class Mode : uint8_t { Off, On, VerifyKernel, VerifyImage };
[[nodiscard]] Mode GetMode() noexcept;

// Analyzes a small compute program once per shader hash (pipelineCache.cpp, where the guest code
// is at hand). A hash seen with two different codes is never a copy.
void Consider(uint64_t shader_hash, std::span<const uint32_t> code,
              const ShaderRecompiler::CompileOptions& options);
[[nodiscard]] std::optional<ShaderRecompiler::IR::UniformCopy> Find(uint64_t shader_hash);

} // namespace Libs::Graphics::ComputeImageCopy
