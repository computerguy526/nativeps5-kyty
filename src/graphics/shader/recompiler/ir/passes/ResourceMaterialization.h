#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCEMATERIALIZATION_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCEMATERIALIZATION_H_

#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"

namespace Libs::Graphics::ShaderRecompiler::IR {

// Canonical module-affecting resource state. Runtime addresses and descriptor payloads remain in
// ResourceSnapshot and therefore do not create shader permutations.
struct ResourceSpecialization {
	struct Buffer {
		uint32_t               packed_stride                   = 0;
		Prospero::BufferFormat descriptor_format               = Prospero::BufferFormat::kInvalid;
		uint32_t               descriptor_swizzle              = DstSel(4, 5, 6, 7);
		bool                   zero_stride_oob                 = false;
		bool                   operator==(const Buffer&) const = default;
	};

	struct Image {
		Prospero::TextureNumericClass numeric_class = Prospero::TextureNumericClass::Unsupported;
		Decoder::ImageDimension       dimension     = Decoder::ImageDimension::Unknown;
		uint32_t                      mip_count     = 1;
		Prospero::BufferFormat        conversion_format          = Prospero::BufferFormat::kInvalid;
		uint32_t                      shader_swizzle             = ShaderImageIdentitySwizzle;
		uint32_t                      indirect_root              = ImageResource::NoIndirectImage;
		uint32_t                      indirect_mapping_offset    = 0;
		uint32_t                      indirect_search_iterations = 0;
		uint32_t                      indirect_group             = ImageResource::NoIndirectImage;
		bool                          cube                       = false;
		bool                          fmask                      = false;
		bool                          operator==(const Image&) const = default;
	};

	std::vector<Buffer> buffers;
	std::vector<Image>  images;

	bool operator==(const ResourceSpecialization&) const = default;
};

// Extracts the descriptor/SRT value graph before resource specialization. The returned plan owns
// its values and is independent of the translated shader CFG.
ResourcePlan ExtractResourcePlan(const Program& program);

// rtimage (KYTY_COMPUTE_IMAGE_COPY): a compute program that is a whole-range element copy between
// two formatted buffers. Proven on the translated IR: one straight path, the only side effect is
// one formatted store of `words` components to buffer `destination` at element
// local + group * group_stride (no offsets, executed by every invocation), and its components are,
// in order, the components of one formatted load of buffer `source` at the same element under an
// always-true condition. What a formatted load and store of the element do to the bits depends on
// the bound descriptors; the dispatch checks those (ResolveComputeBufferCopy).
struct UniformCopy {
	bool     valid        = false;
	uint32_t source       = 0;
	uint32_t destination  = 0;
	uint32_t group_stride = 0;
	uint32_t words        = 0;

	bool operator==(const UniformCopy&) const = default;
};
[[nodiscard]] UniformCopy AnalyzeUniformCopy(const Program& program);

// Refreshes cached resources and specialization in place. A failed refresh must not be used.
// Threads may refresh one sealed plan concurrently when each supplies its own scratch.
bool MaterializeResources(const ResourcePlan& program, const SrtRuntime& runtime,
                          EvaluationScratch& scratch, ResourceSnapshot& snapshot,
                          ResourceSpecialization& specialization);
// Uses this thread's scratch.
bool MaterializeResources(const ResourcePlan& program, const SrtRuntime& runtime,
                          ResourceSnapshot& snapshot, ResourceSpecialization& specialization);

// KYTY_SPEC_UNMAPPED_NULL: whether a V# base is inside a guest mapping (the binding's test; the V#
// is nulled when not). Null: off. Set once by the renderer before draws.
void SetUnmappedBufferProbe(SrtMappedRange probe);
// V#s specialized as null by that rule so far.
[[nodiscard]] uint64_t UnmappedBufferNulls();

// Unit-test entry of the KYTY_SAROS_HS_DYNIDX canonical-group layout (tests/ResourceMaterializationTests.cpp).
bool CanonicalizeIndirectGroupsForTest(const ResourcePlan& program, ResourceSnapshot& snapshot,
                                       ResourceSpecialization& specialization);

// Applies an already-derived specialization to native IR before layout and emission.
void ApplyResourceSpecialization(Program& program, const ResourceSpecialization& specialization);

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCEMATERIALIZATION_H_ */
