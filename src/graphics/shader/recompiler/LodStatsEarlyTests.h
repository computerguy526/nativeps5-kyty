#ifndef KYTY_SHADER_RECOMPILER_LODSTATSEARLYTESTS_H_
#define KYTY_SHADER_RECOMPILER_LODSTATSEARLYTESTS_H_

// KYTY_LOD_STATS_EARLY_TESTS (CodegenOptions::lod_stats_early_tests, default on).
//
// The GET_LOD_STATS feedback of an instrumented pixel shader is a storage-buffer atomic. A
// fragment shader that writes memory and does not declare EarlyFragmentTests gets its depth and
// stencil tests after the shader (Vulkan's rule; Metal follows it for a function that writes
// device memory), and its feedback-free twin gets them before it. On Apple GPUs the late test does
// not see the same depth as the early one: for a fraction of a triangle's pixels it differs by
// one ulp from the depth an earlier pass stored (measured on an M4 Pro), so a colour pass with an
// EQUAL test after a depth pre-pass drops those fragments and the frame shows dark specks on every
// present whose draws use the instrumented program (Astro Bot enables the
// counters of one bucket on 3 presents in 16).
//
// So the instrumented module declares EarlyFragmentTests whenever the feedback is the program's
// only write (LodStatsEarlyTests). That is the same result as late tests unless the shader changes
// its own coverage (discard, alpha to mask) while the draw can write depth or stencil: a discarded
// fragment would then still write them. Such a draw runs the feedback-free variant instead
// (LodStatsEarlyTestsNeedPlain in host_gpu/renderer/lodStatsEarlyDraw.h; KYTY_LOD_STATS_PLAIN_VARIANT
// must be on): it records nothing, and
// the colour pass that follows records the same textures.

#include "graphics/shader/recompiler/CodegenOptions.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/shader.h"

#include <cstdint>

namespace Libs::Graphics::ShaderRecompiler {

// The mip-statistics counters are the only memory the pixel program writes. ProgramT: IR::Program
// or IR::CompiledShaderInfo.
template <typename ProgramT>
[[nodiscard]] bool MipStatsIsOnlyWrite(const ProgramT& program) {
	using Kind = IR::DescriptorBindingKind;
	if (program.stage != ShaderType::Pixel || program.has_address_writes ||
	    program.bindings.mip_stats_count == 0) {
		return false;
	}
	for (const auto& resource: program.info.buffers) {
		if (resource.written || resource.atomic) {
			return false;
		}
	}
	for (const auto& resource: program.info.images) {
		if (resource.written || resource.atomic) {
			return false;
		}
	}
	bool mip_stats = false;
	for (const auto& binding: program.bindings.descriptors) {
		// The fault buffer (direct-memory-access programs) is only written by a faulting read;
		// such a report from a fragment that fails the depth test is not needed.
		if (binding.kind == Kind::Gds || binding.kind == Kind::SharedMemory) {
			return false;
		}
		mip_stats |= binding.kind == Kind::MipStats;
	}
	return mip_stats;
}

// The module's own early tests (DB_SHADER_CONTROL), declared with or without the feedback.
[[nodiscard]] inline bool GuestEarlyTests(const ShaderPixelInputInfo& ps) {
	return ps.ps_early_z && !ps.ps_pixel_kill_enable && !ps.ps_depth_export_enable &&
	       !ps.ps_sample_mask_export_enable;
}

// The instrumented module of this pixel program declares EarlyFragmentTests because of the
// option (the feedback-free module is not changed).
template <typename ProgramT>
[[nodiscard]] bool LodStatsEarlyTests(const ProgramT& program, const ShaderPixelInputInfo& ps) {
	return GetCodegenOptions().lod_stats_early_tests && !GuestEarlyTests(ps) &&
	       !ps.ps_depth_export_enable && !ps.ps_sample_mask_export_enable &&
	       MipStatsIsOnlyWrite(program);
}

} // namespace Libs::Graphics::ShaderRecompiler

#endif
