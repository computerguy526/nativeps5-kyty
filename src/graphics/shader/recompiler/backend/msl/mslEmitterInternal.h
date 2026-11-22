#pragma once

#include <unordered_set>
// Internal interface of the MSL emitter. T1 (core: ALU, flow, entry) owns MslEmitter.cpp /
// mslEmitterAlu.cpp; T2 (memory, images, LDS, atomics, wave ops, binding model) owns
// mslEmitterMem*.cpp and implements the three hooks below (T1 ships weak defaults in
// mslEmitterMemStub.cpp; a strong definition in T2's file replaces them at link time).
#include "graphics/shader/recompiler/backend/msl/MslBuilder.h"
#include "graphics/shader/recompiler/backend/msl/MslEmitter.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::Msl {

struct GfxIo; // vertex/pixel stage IO (MslStageIo.cpp)
namespace Mem {
struct BindingModel;
}

struct MslCtx {
	const IR::Program&   program;
	ShaderStageInputInfo input_info;
	MslBuilder           b;             // the kernel body is written here
	uint32_t             lanes = 1;     // guest lanes per Metal thread (1 or 2); lane h = half h
	bool vote_merge            = false; // mslbadopt KYTY_MSL_VOTE_MERGE: wave64 halves vote once
	                                    // (any(a)||any(b) = any(a||b))
	bool wave64_single =
	    false;         // wave64 program emitted one guest lane per Metal thread (no cross-lane ops)
	uint32_t half = 0; // the half being emitted (set per instruction by the core)
	std::unordered_map<const IR::Inst*, uint32_t> ids; // SSA numbering (v<id>[_<half>])
	std::map<std::string, uint32_t>               unsupported;
	bool                                          options_comments = true;
	std::string            ret = "return;";      // statement that leaves the entry point
	std::shared_ptr<GfxIo> gfx;                  // set for vertex/pixel programs
	bool                   mesh = false;         // meshpath: [[mesh]] program (MslMesh.cpp)
	// meshpath: KK's per-op float model (KYTY_MSL_MESH_MODES, default on for mesh): float results
	// outside `precise` (= Kyty's NoContraction position slice) are assigned in math_mode(relaxed)
	// blocks, slice results in safe ones, a MAD is two statements, no global contract(off).
	bool mesh_modes = false;
	// meshpath spill (KYTY_MSL_MESH_SPILL=1, idea from USER's private nxnative): the guest LDS lives in a claimed
	// slice of KK's mesh scratch buffer (buffer 3) instead of threadgroup memory (MslKkBda.cpp, MslMesh.cpp)
	bool mesh_spill = false;
	uint32_t               loop_cap     = 0;     // MslOptions::loop_cap (0 = no loop guard)
	bool                   loop_private = false; // per-loop counters (program has barriers)
	uint32_t               loop_guards  = 0;     // private counters handed out
	// nps5 gpuguard: guard statement for the loop body just opened (empty when off); declares a
	// private counter before the loop via `decl` when needed (emit decl BEFORE opening the loop).
	[[nodiscard]] std::string LoopGuard(std::string* decl);
	bool                      kk_bda = false; // mslinject: KK binding convention (MslKkBda.cpp)
	// mslaudit: LDS loads that a path reaches from an LDS store/atomic with no barrier/fence in
	// between (lockstep code: GCN omits S_BARRIER for one-wave groups) get KYTY_LDS_FENCE()
	// (simdgroup-scope memory fence) before them.
	std::unordered_set<const IR::Inst*> lds_fence;
	// The binding model of this program + input_info, built on first use (mslEmitterMem.cpp
	// ModelOf). Per emission: a cache keyed by the program's address and shader hash served a
	// previous specialization's model (same hash, same stack address) to the next one.
	mutable std::shared_ptr<const Mem::BindingModel> binding_model;
	// xfclass review M1: the perfauto load_group grouping (mslEmitterMem.cpp LgOf) and the
	// V_MOVRELS chains (MslEmitter.cpp MovrelOf) of THIS emission, built on first use. Same reason
	// as binding_model: a cache keyed by &program + shader_hash served the previous
	// specialization's analysis (stale instruction pointers, another binding model) to the next
	// one. Opaque here: the types are local to those files.
	mutable std::shared_ptr<void> lg_info;
	mutable std::shared_ptr<void> movrel_chains;
	// fmsel6 KYTY_GPU_ACCURACY=normal (pixel/compute): float results of instructions outside
	// `precise` (IR::CollectPreciseSet) are assigned inside `{ #pragma METAL fp math_mode(fast) }`
	// blocks; the rest of the program (and every prelude helper) keeps the library's math mode
	// (safe).
	bool                                selective_precision = false;
	std::unordered_set<const IR::Inst*> precise;

	explicit MslCtx(const IR::Program& p, ShaderStageInputInfo i): program(p), input_info(i) {}

	// MSL expression of an IR value for `half` (immediates as literals, instructions as v<id>).
	[[nodiscard]] std::string Val(const IR::Value& value, uint32_t h) const;
	[[nodiscard]] std::string Val(const IR::Value& value) const { return Val(value, half); }
	// Name of the variable an instruction's result lives in for `half`.
	[[nodiscard]] std::string Name(const IR::Inst& inst, uint32_t h) const;
	[[nodiscard]] std::string Name(const IR::Inst& inst) const { return Name(inst, half); }
	// Guest lane index (0..lanes*32-1 within the wave) of this thread for the current half.
	[[nodiscard]] std::string LaneExpr(uint32_t h) const;
	// Records an unsupported opcode (coverage) and returns false.
	bool Unsupported(const IR::Inst& inst);
};

// --- T2 hooks (weak defaults: "unsupported") ---
// Emits one memory/image/LDS/atomic/wave instruction for ctx.half. Return false if not handled; if
// handled and the instruction has a result, assign it to ctx.Name(inst).
bool EmitMemoryInst(MslCtx& ctx, const IR::Inst& inst);
// Extra kernel parameters ("constant Resources& r [[buffer(0)]]", "threadgroup uint* lds
// [[threadgroup(0)]]", ...), appended after the core's thread/group id and push parameters.
void EmitResourceParams(MslCtx& ctx, std::vector<std::string>& params);
// Declarations placed before the kernel (structs, helpers).
void EmitResourcePrelude(MslCtx& ctx);
// Statements at the start of the kernel body (kk-bda: builds r/push/tgpig/lds from KK's root
// table).
void EmitResourceLocals(MslCtx& ctx);

} // namespace Libs::Graphics::ShaderRecompiler::Msl

namespace Libs::Graphics::ShaderRecompiler::Msl {
// --- mslinject kk-bda convention (MslKkBda.cpp) ---
namespace Mem {
struct BindingModel;
}
bool KkBdaRequested();         // env KYTY_MSL_CONVENTION=kk-bda
// xfclass: emitter mode `fix6` (KYTY_MSL_FIX6=1; kyty_msl_dump --mode fix6): the KK-mimicry text of
// the mslfix5/6, fmafix and bcover chain (KK-exact min/max helpers, NIR re-association and
// statement split, union live components of exported vectors). Off = the v1.1.2 text, byte for
// byte.
bool ModeFix6();
// Number of V_MOVRELS select chains (>= 4 links) MovrelOf found for this emission (tests).
size_t MovrelChainHeads(const MslCtx& ctx);
bool MipStatsPlainRequested(); // env KYTY_MSL_MIPSTATS_PLAIN=1: feedback-free twin of a MipStats PS
std::string KkBdaPrelude();
std::string KkBdaLayoutComment(const IR::Program& program);
void        KkBdaParams(MslCtx& ctx, const Mem::BindingModel& m, std::vector<std::string>& params);
void        KkBdaLocals(MslCtx& ctx, const Mem::BindingModel& m);
// --- M3 graphics stage IO (MslStageIo.cpp) ---
bool GfxIsStage(ShaderType stage);
std::shared_ptr<GfxIo>
GfxSetup(MslCtx& ctx); // writes the VIn/VOut/FIn/FOut structs into ctx.b, sets ctx.ret
std::string GfxEntryPrefix(const GfxIo& g); // "[[early_fragment_tests]] fragment FOut "
const std::vector<std::string>& GfxParams(const GfxIo& g);
void                            GfxLocals(MslCtx& ctx, const GfxIo& g); // "VOut out = {};" ...
std::string GfxBuiltin(MslCtx& ctx, const GfxIo& g, uint32_t kind, uint32_t comp);
bool        GfxEmitInst(MslCtx& ctx, GfxIo& g, const IR::Inst& inst);
// --- meshpath: mesh programs as [[mesh]] functions (MslMesh.cpp) ---
std::string MeshTypes(MslCtx& ctx);  // MeshVertexOut / MeshPrimitiveOut / NpsMesh before the entry
void        MeshLocals(MslCtx& ctx); // per-lane outputs + counts (after the resource locals)
bool        MeshEmitInst(MslCtx& ctx, const IR::Inst& inst);
void        MeshEpilogue(MslCtx& ctx); // barrier + direct mesh_out writes
// --- mslfix5 KK NIR float re-association (MslNirReassoc.cpp) ---
void NirReassocBuild(const IR::Program& program, bool on, bool split);
// fmafix: statements in KK's one-NIR-op-per-statement shape for a split/re-associated float op (""
// = usual text)
std::string NirSplitLines(MslCtx& ctx, const IR::Inst& inst);
std::string NirReassocText(MslCtx& ctx, const IR::Inst& inst);
} // namespace Libs::Graphics::ShaderRecompiler::Msl
