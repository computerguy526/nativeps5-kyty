#pragma once
// perfauto: named, versioned, individually switchable performance transforms of the MSL backend.
//
// A transform is one emitter rule that changes the printed MSL without changing what the guest
// program computes (class), has a stated precondition, and prints byte-identical text when off.
// The set that is active for a program is resolved once per EmitProgram:
//   1. KYTY_MSL_XF=<list>           base set for every program: "all", "none", or ids separated
//                                   by ',' ("-id" removes one, e.g. "all,-vs_no_barrier").
//   2. KYTY_MSL_VS_PERF=1           legacy alias (vsperf): adds the three vs_* fetch transforms.
//   3. KYTY_MSL_XF_TABLE=<file>     per-shader choice, one row per line:
//                                       <vs|ps|cs> <guest shader hash, 16 hex> <list or "-">
//                                   A matching row REPLACES the base set, so the text of a listed
//                                   shader does not depend on the environment defaults: this is
//                                   how the measured/verified choice (offline optimisation loop,
//                                   work/mslnative/perfauto) reaches translation deterministically.
// Nothing about the active set is printed into the text: the full vs_* set is byte-identical to
// vsperf's KYTY_MSL_VS_PERF=1 text and the empty set to the base emitter's.
#include "graphics/shader/shader.h"

#include <cstdint>
#include <string>

namespace Libs::Graphics::ShaderRecompiler::Msl::Xf {

enum Id : uint32_t {
	// Vertex fetch reads the KK root buffer through an integer address
	// (`(ulong)&kkroot.contents[0] + ulong(N)`), which Apple's compiler hoists into the per-draw
	// constant program; through `constant uchar*` derived from the argument it does not.
	VsRootInt,
	// Vertex attribute is loaded unconditionally, the out-of-range select follows (KK's form;
	// the vertex buffer is bound with KK's robustness padding). Requires nothing else.
	VsFetchSelect,
	// No device-flush barrier for S_BARRIER in a vertex function (one lane, no LDS, no
	// threadgroup): the barrier pins every later uniform load below it.
	VsNoBarrier,
	// Register pressure (megakernels, large VS): the per-access buffer bounds check in 32 bits,
	// `size >= N && addr <= size - N`, instead of `ulong(addr) + N <= ulong(size)`. Same truth
	// table for 32-bit addr and size; no 64-bit temporary (a register pair on AGX) per access, and
	// `size - N` is uniform, so it can move to the constant program. Any stage.
	BoundsU32,
	// kyty_fmin/fmax/fmed3 prelude helpers written with select() instead of ?: (no branch per
	// call). Applies wherever the KK-exact min/max helpers are printed (graphics stages today).
	MinMaxSelect,
	// GET_LOD_STATS record: plain load instead of a relaxed atomic load in the pre-check before
	// atomic_fetch_min (pixel stage; shaders with the MipStats record only).
	MipStatsPlainLoad,
	Count
};

struct Info {
	const char* id;      // stable name used in KYTY_MSL_XF, the table and the result database
	uint32_t    version; // bump when the printed form changes (invalidates stored verdicts)
	ShaderType  stage;   // stage the transform applies to (Unknown = any)
	const char* cls;     // exactness class: structural | semantic-exact | bounded | preconditioned
	const char* doc;
};

[[nodiscard]] const Info& InfoOf(Id id);

// Resolves the active set for the program about to be emitted (thread-local).
void Begin(ShaderType stage, uint64_t shader_hash);
// True when the transform is active for the current program (always false for a transform whose
// stage does not match the program's).
[[nodiscard]] bool On(Id id);
// "vs_root_int@1,vs_fetch_select@1" (registry order) or "" for the current program.
[[nodiscard]] std::string ActiveList();
// "id@version stage class" per line: the catalogue (kyty_msl_dump --xf-list).
[[nodiscard]] std::string Catalogue();

} // namespace Libs::Graphics::ShaderRecompiler::Msl::Xf
