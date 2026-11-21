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
//                                   how the measured/verified choice (offline optimisation loop)
//                                   reaches translation deterministically.
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
	// bigshader loadgroup (skyall finding 5): raw dword reads of one binding with the same dynamic
	// address
	// operands in one block (span <= 64 B) share ONE range check and packed vector loads; when the
	// check fails
	// each member runs its original guarded text. Bit-exact by construction (mslEmitterMem.cpp).
	// Any stage.
	LoadGroup,
	// V_MOVRELS select chains (>= 4 links on one M0 value, inner links used once) printed as one
	// switch on M0
	// at the chain head (skyall finding 6, foliage VS). Bit-exact (integer moves). Any stage.
	MovrelSwitch,
	// The same chains as a balanced compare tree over the sorted constants (leaf: M == k ? r_k :
	// base).
	// movrel_switch wins when both are on. Bit-exact. Any stage.
	MovrelTree,
	// foliage2 / xfmine N1: a bounds-checked buffer load `in_range ? *(buf + addr) : 0` is printed as an
	// UNCONDITIONAL load from `in_range ? buf + addr : <root>` followed by the same select. In range the
	// same bytes are read; out of range (also a zero-size or unbound buffer: its size is 0, so the check
	// fails) the read goes to the kk-bda root block, which is bound for every draw and dispatch and is
	// larger than any loaded type (the entry function itself reads the set pointer from it), and the
	// value is dropped for 0 exactly as before. Loads only: never a store or an atomic, never the
	// per-dword partial form, never a formatted (all-or-nothing) element. No-op outside kk-bda. Any stage.
	LoadSelect,
	// foltrans (botvs `vgroup`): load_group in the vertex stage only (the pixel text of the pair keeps
	// its per-load form). Same grouping, same text as load_group prints for a vertex program:
	// reads of one binding with a common dynamic base in one block share ONE range check and
	// packed vector loads; when the check fails each member runs its original guarded text.
	// Not in `all`.
	VsLoadGroup,
	// foltrans (foliage `cload`): loads of buffers the vertex program never writes come from the
	// `constant` address space (kk-bda only), so the compiler may move loads with uniform
	// addresses to the per-draw constant program even when the program holds a buffer atomic or
	// store (a device load cannot be hoisted across one: possible aliasing). Rule, as the
	// foliagevs rewrite of KosmicKrisp's text: (a) a load printed BEFORE the first buffer
	// store/atomic of the program is `constant` unconditionally (it can only observe other
	// invocations' writes, which are unordered with this invocation); (b) a load printed AFTER it
	// is `kk_calias ? device load : constant load`, kk_calias = some read-only buffer's bound
	// range overlaps a written buffer's bound range in this draw (uniform; every access is
	// bounds-checked to its range); (c) a store/atomic inside a loop, or the dispatcher fallback:
	// every load takes form (b). Programs with GDS counters are left alone. Not in `all`.
	VsConstLoad,
	// fsslow (fullscreen lighting FS, light-list loops): a loop whose continue block only branches
	// back to the header and holds nothing but phis and instructions without side effects is
	// printed WITHOUT the `do { } while (false)` wrapper and its exit flag: every jump to the
	// continue block prints the phi copies into it, its instructions, the phi copies into the
	// header and `continue;`; a jump to the merge block is a plain `break;`. Every path runs the
	// same blocks, edges and copies in the same order: only the statement nesting changes (the
	// continue block's text is repeated once per edge into it). With the wrapper every exit
	// passes the join after the `do`, which keeps the header phis live across the continue
	// block: Apple's compiler then cannot share their registers and copies every loop-carried
	// value on each iteration (the early-out iterations of a light walk are little else).
	// Any stage. Not in `all`.
	LoopCont,
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

// xfclass: emitter MODES. A shipped Metal list row (nativeps5 games/<id>/metal-*.tsv, keys format 2
// / pairs format 3) names the mode its text was verified and timed with; the public generator
// (kyty_msl_dump --mode <mode>) sets exactly these switches for that one translation, so every
// listed text is a function of (emitter commit, mode) and is reproducible. Tokens joined by '+':
//   base          nothing (the v1.1.2 / v1.2 text)
//   fix6          KYTY_MSL_FIX6=1: the KK-mimicry chain (mslfix5/6, fmafix, bcover)
//   perfr         KYTY_MSL_GFX_SPLIT=0 GFX_REASSOC=0 GFX_MINMAX=0 (relaxed pixel math without KK
//   mimicry) vsperf        KYTY_MSL_VS_PERF=1 (vertex fetch transforms + varying narrowing)
//   gfxmath=M     KYTY_MSL_GFX_MATH=M (safe | relaxed | fast)
//   lsel          KYTY_MSL_XF_LSEL=1: adds the load_select transform to the base set (combines with xf=HEX)
//   vgroup        KYTY_MSL_XF_VGROUP=1: adds vs_load_group to the base set (vertex stage only)
//   cload         KYTY_MSL_XF_CLOAD=1: adds vs_const_load to the base set (vertex stage only)
//   lcont         KYTY_MSL_XF_LCONT=1: adds loop_cont to the base set (any stage)
//   xf=HEX        KYTY_MSL_XF = the transforms of the registry-order bit mask (perfauto paxHEX)
// false + error for an unknown or repeated token, a bad value, perfr without fix6 (it changes
// nothing there), or ANY text switch of the emitter already set in the environment (KYTY_MSL_FIX6,
// VS_PERF, GFX_*, XF, XF_*, XF_TABLE, LOADGROUP*): a mode never mixes with hand-set switches.
[[nodiscard]] bool ApplyMode(std::string_view mode, std::string& error);
// The one truthiness rule of the emitter switches (review L3): set, non-empty and not "0".
[[nodiscard]] bool EnvOn(const char* name);
// "token  meaning" per line (kyty_msl_dump --mode-list).
[[nodiscard]] std::string ModeCatalogue();

} // namespace Libs::Graphics::ShaderRecompiler::Msl::Xf
