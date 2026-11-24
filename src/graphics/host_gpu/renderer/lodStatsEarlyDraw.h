#ifndef KYTY_RENDERER_LODSTATSEARLYDRAW_H_
#define KYTY_RENDERER_LODSTATSEARLYDRAW_H_

// KYTY_LOD_STATS_EARLY_TESTS, the draw side (see shader/recompiler/LodStatsEarlyTests.h).

#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/shader/recompiler/LodStatsEarlyTests.h"

#include <atomic>
#include <cstdint>

namespace Libs::Graphics::ShaderRecompiler {

// The draw can change the depth or the stencil attachment, judged from the registers alone
// (conservative: write-disabled views, a missing attachment and clears are not looked at). A
// depth write under an EQUAL test stores the value that is already there.
[[nodiscard]] inline bool DrawWritesDepthOrStencil(const HW::Context& hw) {
	const auto& dc = hw.GetDepthControl();
	if (dc.z_enable && dc.z_write_enable && dc.zfunc != 2u /* EQUAL */) {
		return true;
	}
	if (!dc.stencil_enable) {
		return false;
	}
	const auto& sc    = hw.GetStencilControl();
	const auto& sm    = hw.GetStencilMask();
	const bool  front = sm.stencil_writemask != 0 &&
	                    (sc.stencil_fail != 0 || sc.stencil_zpass != 0 || sc.stencil_zfail != 0);
	const bool  back =
	    dc.backface_enable && sm.stencil_writemask_bf != 0 &&
	    (sc.stencil_fail_bf != 0 || sc.stencil_zpass_bf != 0 || sc.stencil_zfail_bf != 0);
	return front || back;
}

// A draw of an instrumented program with forced early tests whose shader changes its own
// coverage while depth or stencil can be written: early tests would not be equivalent, the draw
// takes the feedback-free variant.
template <typename ProgramT>
[[nodiscard]] bool LodStatsEarlyTestsNeedPlain(const ProgramT&             program,
                                               const ShaderPixelInputInfo& ps,
                                               const HW::Context&          hw) {
	return (ps.ps_pixel_kill_enable || hw.GetShaderRegisters().AlphaToMask()) &&
	       LodStatsEarlyTests(program, ps) && DrawWritesDepthOrStencil(hw);
}

// Draw counts for the KYTY_LOD_STATS_EARLY_LOG line (relaxed; written on the GPU thread).
struct LodStatsEarlyTotals {
	std::atomic<uint64_t> instrumented_early {0}; // instrumented, early tests forced by the option
	std::atomic<uint64_t> instrumented_guest {0}; // instrumented, early tests by the guest's flags
	std::atomic<uint64_t> instrumented_late {0};  // instrumented, tests after the shader
	std::atomic<uint64_t> plain_for_writes {0};   // had a counter, took the plain variant
	std::atomic<uint64_t> plain_no_counter {0};   // no counter on any image
};
inline LodStatsEarlyTotals g_lod_stats_early_totals;

} // namespace Libs::Graphics::ShaderRecompiler

#endif
