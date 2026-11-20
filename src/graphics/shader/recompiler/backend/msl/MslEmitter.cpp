// MSL emitter core: SSA values -> typed locals, CFG -> dispatcher loop, EXEC/VCC mask emulation,
// compute entry. Memory/image/LDS/atomics are T2's (EmitMemoryInst hook).
#include "graphics/shader/recompiler/backend/msl/MslEmitter.h"

#include "graphics/shader/recompiler/CodegenOptions.h"
#include "graphics/shader/recompiler/backend/msl/MslImageOps.h"
#include "graphics/shader/recompiler/backend/msl/MslSharedOps.h"
#include "graphics/shader/recompiler/backend/msl/MslXform.h"
#include "graphics/shader/recompiler/backend/msl/mslEmitterInternal.h"
#include "graphics/shader/recompiler/ir/passes/PrecisionAnalysis.h"

#include <algorithm>
#include <bit>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fmt/format.h>
#include <map>
#include <functional>
#include <set>
#include <unordered_set>

namespace Libs::Graphics::ShaderRecompiler::Msl {

using IR::ValueOpcode;

namespace {

constexpr const char* Prelude = R"(#include <metal_stdlib>
#include <simd/simd.h>
using namespace metal;
#pragma METAL fp contract(off)

// GCN semantics helpers (kyty MSL backend).
static inline uint kyty_f2i(float x) { return as_type<uint>(isnan(x) ? 0 : (x >= 2147483648.0f ? 2147483647 : (x <= -2147483648.0f ? int(0x80000000u) : int(x)))); }
static inline uint kyty_f2u(float x) { return (isnan(x) || x <= 0.0f) ? 0u : (x >= 4294967296.0f ? 0xffffffffu : uint(x)); }
static inline uint2 kyty_add_carry(uint a, uint b) { uint s = a + b; return uint2(s, s < a ? 1u : 0u); }
static inline ushort kyty_f16_rtz(float x) {
  ushort b = as_type<ushort>(half(x));
  if (isfinite(x) && fabs(float(as_type<half>(b))) > fabs(x)) { b -= 1; }
  return b;
}
static inline uint kyty_pack_f16_rtz(float a, float b) { return uint(kyty_f16_rtz(a)) | (uint(kyty_f16_rtz(b)) << 16); }
static inline bool kyty_class_f32(float x, uint m) {
  uint b = as_type<uint>(x); bool neg = (b >> 31) != 0; uint e = (b >> 23) & 255u; uint f = b & 0x7fffffu; uint c;
  if (e == 255u) { c = f == 0u ? (neg ? 4u : 512u) : ((f & 0x400000u) != 0u ? 2u : 1u); }
  else if (e == 0u) { c = f == 0u ? (neg ? 32u : 64u) : (neg ? 16u : 128u); }
  else { c = neg ? 8u : 256u; }
  return (m & c) != 0u;
}
static inline bool kyty_class_f16(uint bits, uint m) {
  uint b = bits & 0xffffu; bool neg = (b >> 15) != 0; uint e = (b >> 10) & 31u; uint f = b & 0x3ffu; uint c;
  if (e == 31u) { c = f == 0u ? (neg ? 4u : 512u) : ((f & 0x200u) != 0u ? 2u : 1u); }
  else if (e == 0u) { c = f == 0u ? (neg ? 32u : 64u) : (neg ? 16u : 128u); }
  else { c = neg ? 8u : 256u; }
  return (m & c) != 0u;
}
// KK_WORKAROUND_3 (FB20113490): macOS 26 if-converts a branch around simd_ballot/simd_any/simd_all and evaluates them with
// every lane active; KosmicKrisp builds ballots and votes from simd_or of the lane bit instead (right on any OS).
static inline uint kyty_ballot(bool p, uint l) { return simd_or(p ? (1u << (l & 31u)) : 0u); }
static inline bool kyty_any(bool p, uint l) { return kyty_ballot(p, l) != 0u; }
static inline bool kyty_all(bool p, uint l) { return kyty_ballot(!p, l) == 0u; }
// mslaudit: simdgroup-scope memory fence between an LDS store and a later LDS load of another lane in lockstep code (no S_BARRIER).
// Not an execution barrier. MSL < 3.2 has no scoped atomic_thread_fence: simdgroup_barrier (B's control flow is wave-uniform).
#if __METAL_VERSION__ >= 320
#define KYTY_LDS_FENCE(f) atomic_thread_fence(f, memory_order_seq_cst, thread_scope_simdgroup)
#else
#define KYTY_LDS_FENCE(f) simdgroup_barrier(f)
#endif
// V_RCP_F32: KK (NIR->MSL) emits `1/x` under math_mode(relaxed) (Apple fast reciprocal); B matches it bit-exactly.
static inline float kyty_rcp(float x) {
#pragma METAL fp math_mode(relaxed)
  return 1/x;
}
// V_RSQ_F32 / V_SQRT_F32: KK always prints precise::rsqrt / precise::sqrt (mostly under math_mode(relaxed)); plain
// rsqrt/sqrt become fast:: approximations under KK's FAST function mode (mslrest2: perm 395 differed ~3000 ulp).
static inline float kyty_rsqrt(float x) {
#pragma METAL fp math_mode(relaxed)
  return precise::rsqrt(x);
}
static inline float kyty_sqrt(float x) {
#pragma METAL fp math_mode(relaxed)
  return precise::sqrt(x);
}
)";

uint32_t FloatBits(float value) {
	uint32_t bits;
	std::memcpy(&bits, &value, sizeof(bits));
	return bits;
}

const IR::BlockInfo* InfoFor(const IR::Program& program, size_t block_index) {
	return block_index < program.block_info.size() ? &program.block_info[block_index] : nullptr;
}

size_t BlockIndexById(const IR::Program& program, uint32_t id) {
	for (size_t i = 0; i < program.block_info.size(); i++) {
		if (program.block_info[i].id == id) {
			return i;
		}
	}
	return SIZE_MAX;
}

// mslrest4: integer f32->f16 round-toward-zero exactly as Kyty's SPIR-V (KK MSL: explicit half
// denormals, 0x7bff clamp, quiet NaN payload) instead of half(x) + step back (Metal's half
// conversion may flush half denormals). Graphics stages by default (pipe 3 capture: 1-ulp f16
// alpha); compute kernels keep the verified text unless KYTY_MSL_F16_RTZ_INT=1.
const char* const kF16RtzInt = R"(static inline ushort kyty_f16_rtz(float x) {
  uint u = as_type<uint>(x); uint a = u & 0x7fffffffu;
  uint e = min(31743u, (a >> 13) + 4294852608u);
  uint s = min(31u, as_type<uint>(126 - int(a >> 23)));
  uint d = ((u & 8388607u) | 8388608u) >> s;
  uint r = a < 947912704u ? d : e;
  if (a >= 2139095040u) r = ((a >> 13) & 32767u) | (a > 2139095040u ? 512u : 0u);
  return ushort(r | ((u >> 16) & 32768u));
}
)";

// mslfix4: vertex stage = ONE lane. KosmicKrisp has no vertex-stage subgroups, so Kyty's SPIR-V
// (EnableSingleLane) turns every subgroup op of a VS into its one-lane result and KK's VS MSL has
// no simd_* at all. B used real simdgroup votes/shuffles with lid = 0: a waterfall loop
// (ReadFirstLane + IEqual on a per-vertex key) kept reading HARDWARE lane 0's key, lanes holding
// another key never matched and the loop ran to the gpuguard cap (262144 iterations of a big body =
// GPU hang, gfxfuzz pipes 30/35 mipstats).
const char* const kVsSingleLane =
    R"(// mslfix4: vertex stage = one lane (KosmicKrisp has no vertex subgroups; Kyty SPIR-V EnableSingleLane)
static inline uint kyty_ballot(bool p, uint l) { return p ? 1u : 0u; }
static inline bool kyty_any(bool p, uint l) { return p; }
static inline bool kyty_all(bool p, uint l) { return p; }
#define simd_shuffle(v, l) (v)
#define simd_broadcast_first(v) (v)
#define simd_or(v) (v)
#define simd_min(v) (v)
#define simd_max(v) (v)
#define simd_is_first() true
)";

// mslfix6: V_MIN/V_MAX_F32 exactly as KosmicKrisp prints Kyty's SPIR-V min/max (NIR fmin/fmax with
// signed zeros: `!isnan(a) && !isnan(b) && a == b ? bits(a | b) : precise::fmin(a, b)` under
// math_mode(safe)): a NaN operand gives the other one, min(-0, +0) = -0, max(-0, +0) = +0.
// V_MED3_F32: any NaN operand gives min3, otherwise max(min(a, b), min(max(a, b), c)) (GCN, as
// EmitFastMedTri32). Metal's own fmin/fmax/fmedian3 under relaxed math differ on NaN/signed zeros.
constexpr const char* kKkMinMax = R"(static inline float kyty_fmin(float a, float b) {
#pragma METAL fp math_mode(safe)
  return !isnan(a) && !isnan(b) && a == b ? as_type<float>(as_type<uint>(a) | as_type<uint>(b)) : precise::fmin(a, b);
}
static inline float kyty_fmax(float a, float b) {
#pragma METAL fp math_mode(safe)
  return !isnan(a) && !isnan(b) && a == b ? as_type<float>(as_type<uint>(a) & as_type<uint>(b)) : precise::fmax(a, b);
}
static inline float kyty_fmed3(float a, float b, float c) {
#pragma METAL fp math_mode(safe)
  float lo = kyty_fmin(a, b);
  return (isnan(a) || isnan(b) || isnan(c)) ? kyty_fmin(lo, c) : kyty_fmax(lo, kyty_fmin(kyty_fmax(a, b), c));
}
)";

// perfauto transform minmax_select (MslXform.h; skyall finding 1): the same helpers with select()
// instead of ?: . Same operands and operations (both arms are side-effect free), but Metal keeps
// the ternary as control flow (one `if` per call) and the select as a conditional move.
constexpr const char* kKkMinMaxSelect = R"(static inline float kyty_fmin(float a, float b) {
#pragma METAL fp math_mode(safe)
  return select(precise::fmin(a, b), as_type<float>(as_type<uint>(a) | as_type<uint>(b)), !isnan(a) && !isnan(b) && a == b);
}
static inline float kyty_fmax(float a, float b) {
#pragma METAL fp math_mode(safe)
  return select(precise::fmax(a, b), as_type<float>(as_type<uint>(a) & as_type<uint>(b)), !isnan(a) && !isnan(b) && a == b);
}
static inline float kyty_fmed3(float a, float b, float c) {
#pragma METAL fp math_mode(safe)
  float lo = kyty_fmin(a, b);
  return select(kyty_fmax(lo, kyty_fmin(kyty_fmax(a, b), c)), kyty_fmin(lo, c), isnan(a) || isnan(b) || isnan(c));
}
)";

bool KkMinMaxEnabled() {
	const char* e = std::getenv("KYTY_MSL_GFX_MINMAX");
	return e == nullptr || std::strcmp(e, "0") != 0;
}

std::string PreludeText(bool fuse_mad, bool gfx = false, bool contract = false,
                        bool single_lane = false, bool kk_minmax = false) {
	std::string t(Prelude);
	if (kk_minmax) {
		const std::string anchor = "// nps5 gpuguard";
		const auto        at     = t.find(anchor);
		t.insert(at == std::string::npos ? t.size() : at,
		         Xf::On(Xf::MinMaxSelect) ? kKkMinMaxSelect : kKkMinMax);
	}
	if (single_lane) {
		const auto b0 = t.find("static inline uint kyty_ballot(bool p, uint l)");
		const auto b1 = t.find("static inline bool kyty_all(bool p, uint l)", b0);
		const auto e1 = b1 == std::string::npos ? b1 : t.find('\n', b1);
		if (b0 != std::string::npos && e1 != std::string::npos)
			t.replace(b0, e1 + 1 - b0, kVsSingleLane);
	}
	if (contract) { // mslfix: KK-like pixel math (GfxPixelRelaxed): the host contracts like KK's
		            // relaxed ops
		const std::string off = "#pragma METAL fp contract(off)\n";
		if (const auto at = t.find(off); at != std::string::npos) t.erase(at, off.size());
	}
	const char* fe = std::getenv("KYTY_MSL_F16_RTZ_INT");
	if (fe != nullptr ? std::strcmp(fe, "1") == 0 : gfx) {
		const auto b0 = t.find("static inline ushort kyty_f16_rtz(float x) {");
		const auto b1 = t.find("\n}\n", b0);
		if (b0 != std::string::npos && b1 != std::string::npos)
			t.replace(b0, b1 + 3 - b0, kF16RtzInt);
	}
	if (fuse_mad) { // host may contract a*b+c (what KK/SPIRV-Cross output gets by default): 2x
		            // fewer ALU ops on mad-heavy kernels
		// The relaxed pixel text (contract) has no contract(off) line left: it contracts already.
		const std::string off = "#pragma METAL fp contract(off)";
		if (const auto at = t.find(off); at != std::string::npos)
			t.replace(at, off.size(), "#pragma METAL fp contract(fast)");
	}
	return t;
}

// mslinject: first source line read by KK (kk-mslinject) when it compiles this text in place of its
// own MSL. mslfix: pixel shaders in KK's float model. KosmicKrisp prints Kyty's SPIR-V float ALU
// (SignedZeroInfNanPreserve) under
// `#pragma METAL fp math_mode(relaxed)` (comparisons safe) in a fast-math library, so Metal
// contracts mul+add chains and uses relaxed reciprocals; B's safe + contract(off) text rounded
// differently (mslcap7 pipe 13 ~1 % of 10-bit words +-1). B's comparisons are NaN-explicit
// (isordered/isunordered) and its transcendental calls are precise::, so the whole pixel function
// can be relaxed. KYTY_MSL_GFX_MATH=safe restores the old text.
bool GfxPixelRelaxed() {
	static const bool on = [] {
		const char* e = std::getenv("KYTY_MSL_GFX_MATH");
		return e == nullptr || std::strcmp(e, "safe") != 0;
	}();
	return on;
}

bool GfxPixelFast() {
	static const bool on = [] {
		const char* e = std::getenv("KYTY_MSL_GFX_MATH");
		return e != nullptr && std::strcmp(e, "fast") == 0;
	}();
	return on;
}

std::string KkInjectHeader(const MslCtx& ctx, const MslResult& r) {
	if (!ctx.kk_bda) return {};
	// mslgfx: graphics stages are swapped as a VS+PS pair inside KK's render pipeline descriptor.
	if (ctx.program.stage == ShaderType::Vertex)
		return "// kk-mslinject v1 stage vertex entry main0 math safe\n";
	if (ctx.program.stage == ShaderType::Pixel) {
		// perfmode (heavyshade, opt-in KYTY_MSL_GFX_MATH=fast): KK compiles the injected FS with Metal fast math. Validated
		// under autoval outprec (stored-output precision), never bit-exact; pair with KYTY_MSL_GFX_SPLIT=0 REASSOC=0 MINMAX=0.
		if (GfxPixelFast()) return "// kk-mslinject v1 stage fragment entry main0 math fast\n";
		return GfxPixelRelaxed() ? "// kk-mslinject v1 stage fragment entry main0 math relaxed\n"
		                         : "// kk-mslinject v1 stage fragment entry main0 math safe\n";
	}
	// Compute (litcompute 2026-11-19): 7f9ba790 lost the braces above, so every compute kernel got the fragment header.
	return fmt::format("// kk-mslinject v1 local_size {} {} {} entry main_entrypoint math safe\n",
	                   r.threadgroup[0], r.threadgroup[1], r.threadgroup[2]);
}

} // namespace

std::string MslCtx::Name(const IR::Inst& inst, uint32_t h) const {
	const auto     found = ids.find(&inst);
	const uint32_t id    = found == ids.end() ? 0xffffffffu : found->second;
	return lanes == 1 ? fmt::format("v{}", id) : fmt::format("v{}_{}", id, h);
}

std::string MslCtx::Val(const IR::Value& value, uint32_t h) const {
	using IR::Type;
	if (value.IsEmpty()) {
		return "0u";
	}
	if (value.IsImmediate()) {
		switch (value.GetType()) {
			case Type::U1: return value.U1() ? "true" : "false";
			case Type::U8: return fmt::format("uchar({})", value.U8());
			case Type::U16: return fmt::format("ushort({})", value.U16());
			case Type::U32: return MslU32(value.U32());
			case Type::U64: return MslU64(value.U64());
			case Type::F16: return fmt::format("as_type<half>(ushort({}))", value.F16Bits());
			case Type::F32: return MslF32Bits(FloatBits(value.F32Value()));
			default: return "0u /*imm*/";
		}
	}
	if (auto* inst = value.TryInstruction(); inst != nullptr) {
		return Name(*inst, h);
	}
	return "0u /*reg*/";
}

std::string MslCtx::LaneExpr(uint32_t h) const {
	if (wave64_single) return "(tid & 63u)";
	return lanes == 1 ? std::string("lid") : fmt::format("(lid + {}u)", h * 32u);
}

bool MslCtx::Unsupported(const IR::Inst& inst) {
	unsupported[std::string(IR::ValueOpcodeName(inst.GetOpcode()))]++;
	return false;
}

namespace {

struct ComputeShape {
	uint32_t threads[3] = {1, 1, 1};
};

ComputeShape ShapeOf(const IR::Program& program, ShaderStageInputInfo input) {
	ComputeShape shape;
	if (const auto* cs = ShaderWorkgroupInput(program.stage, input); cs != nullptr) {
		for (int i = 0; i < 3; i++) {
			shape.threads[i] = std::max(cs->threads_num[i], 1u);
		}
	}
	return shape;
}

std::string BuiltinExpr(MslCtx& ctx, const ComputeShape& shape, uint32_t kind, uint32_t c) {
	using K       = IR::StageInputKind;
	c             = std::min(c, 2u);
	const auto li = [&](uint32_t h) -> std::string {
		return ctx.lanes == 2 ? fmt::format("(tid + (tid & 4294967264u) + {}u)", h * 32u)
		                      : std::string("tid");
	};
	const auto local = [&](uint32_t h) {
		uint32_t divisor = 1;
		for (uint32_t a = 0; a < c; a++)
			divisor *= shape.threads[a];
		return fmt::format("(({} / {}u) % {}u)", li(h), divisor, shape.threads[c]);
	};
	switch (static_cast<K>(kind)) {
		case K::WorkgroupId: return fmt::format("tgpig[{}]", c);
		case K::NumWorkgroups: return fmt::format("nwg[{}]", c);
		case K::LocalInvocationIndex: return li(ctx.half);
		case K::LocalInvocationId: return local(ctx.half);
		case K::GlobalInvocationId:
			return fmt::format("({} + tgpig[{}] * {}u)", local(ctx.half), c, shape.threads[c]);
		default: return {};
	}
}

// Phi copies for the CFG edge from -> to (parallel assignment through temporaries).
void EmitEdge(MslCtx& ctx, const IR::Block* from, const IR::Block* to) {
	if (to == nullptr) {
		return;
	}
	std::vector<std::pair<std::string, std::string>> copies; // dest, source
	for (const auto& phi: *to) {
		if (phi.GetOpcode() != ValueOpcode::Phi) {
			break;
		}
		for (size_t i = 0; i < phi.NumArgs(); i++) {
			if (phi.PhiBlock(i) != from) {
				continue;
			}
			for (uint32_t h = 0; h < ctx.lanes; h++) {
				copies.emplace_back(ctx.Name(phi, h), ctx.Val(phi.Arg(i), h));
			}
			break;
		}
	}
	if (copies.empty()) return;
	if (copies.size() == 1) {
		ctx.b.Line(fmt::format("{} = {};", copies[0].first, copies[0].second));
		return;
	}
	ctx.b.Open(); // own scope: several edges can be emitted in one structured scope
	for (size_t i = 0; i < copies.size(); i++) {
		// Only sources that are themselves destinations need a temporary; keep it simple and safe.
		ctx.b.Line(fmt::format("auto e{} = {};", i, copies[i].second));
	}
	for (size_t i = 0; i < copies.size(); i++) {
		ctx.b.Line(fmt::format("{} = e{};", copies[i].first, i));
	}
	ctx.b.Close();
}

// Wave-uniform branch condition (EXEC/VCC emulation). The IR carries a per-lane U1 for
// EXEC/VCC-based branches (the SPIR-V emitter's BranchCondition): "...NonZero" = any lane set,
// "...Zero" = all lanes of the (inverted) value set. Scalar-instruction and goto-variable
// conditions are already wave-uniform scalars.
std::string BranchExpr(MslCtx& ctx, const IR::BlockInfo& info) {
	const auto kind = info.terminator.condition;
	if (kind == CFG::BranchCondition::ScalarInstruction ||
	    kind == CFG::BranchCondition::GotoVariable || kind == CFG::BranchCondition::Always ||
	    kind == CFG::BranchCondition::Unknown || kind == CFG::BranchCondition::SccZero ||
	    kind == CFG::BranchCondition::SccNonZero) {
		return ctx.Val(info.condition, 0);
	}
	const bool zero =
	    kind == CFG::BranchCondition::ExecZero || kind == CFG::BranchCondition::VccZero;
	const char* op = zero ? "kyty_all" : "kyty_any";
	if (ctx.lanes == 1) {
		return fmt::format("{}({}, lid)", op, ctx.Val(info.condition, 0));
	}
	if (ctx.vote_merge) { // one simdgroup vote over both halves: all(a)&&all(b) == all(a&&b),
		                  // any(a)||any(b) == any(a||b)
		return fmt::format("{}(({}) {} ({}), lid)", op, ctx.Val(info.condition, 0),
		                   zero ? "&&" : "||", ctx.Val(info.condition, 1));
	}
	return fmt::format("({}({}, lid) {} {}({}, lid))", op, ctx.Val(info.condition, 0),
	                   zero ? "&&" : "||", op, ctx.Val(info.condition, 1));
}

void EmitTerminator(MslCtx& ctx, size_t index) {
	const auto& program    = ctx.program;
	const auto& info       = program.block_info[index];
	const auto* block      = program.blocks[index];
	const auto& term       = info.terminator;
	const auto  goto_block = [&](uint32_t id) {
		const auto target = BlockIndexById(program, id);
		if (target == SIZE_MAX) {
			ctx.b.Line(ctx.ret);
			return;
		}
		EmitEdge(ctx, block, program.blocks[target]);
		ctx.b.Line(fmt::format("blk = {}u; continue;", id));
	};
	switch (term.kind) {
		case CFG::TerminatorKind::Branch: goto_block(term.true_block); return;
		case CFG::TerminatorKind::ConditionalBranch: {
			if (info.condition.IsEmpty() || BlockIndexById(program, term.true_block) == SIZE_MAX ||
			    BlockIndexById(program, term.false_block) == SIZE_MAX) {
				ctx.b.Line(ctx.ret);
				return;
			}
			if (term.true_block == term.false_block) {
				goto_block(term.true_block);
				return;
			}
			ctx.b.Open(fmt::format("if ({})", BranchExpr(ctx, info)));
			goto_block(term.true_block);
			ctx.b.Close();
			goto_block(term.false_block);
			return;
		}
		case CFG::TerminatorKind::IndirectBranch: {
			if (info.indirect_target.IsEmpty()) {
				ctx.b.Line(ctx.ret);
				return;
			}
			const bool  by_code = term.indirect_selector_code != UINT32_MAX;
			const auto& values = by_code ? term.indirect_selector_values : term.indirect_target_pcs;
			const auto& targets = by_code ? term.indirect_selector_targets : term.indirect_targets;
			const auto  sel     = ctx.Val(info.indirect_target, 0);
			for (size_t i = 0; i < std::min(values.size(), targets.size()); i++) {
				ctx.b.Open(fmt::format("if ({} == {})", sel, MslU32(values[i])));
				goto_block(targets[i]);
				ctx.b.Close();
			}
			ctx.b.Line(ctx.ret);
			return;
		}
		default: ctx.b.Line(ctx.ret); return;
	}
}

// perfauto transform movrel_switch (MslXform.h; skyall finding 6): V_MOVRELS is lowered (frontend) to
// sel_k = (M == k) ? r_k : sel_{k-1}. A chain of >= 4 such links on the same M, whose inner links are used only
// by the next link, is printed at its head as `switch (M) { case k: x = r_k; ... default: x = base; }`. Exact:
// the compares are of one value against distinct constants, so at most one arm of the chain is taken, and the
// head-most link wins on a repeated constant (later duplicates are dropped). Inner links print nothing.
struct MovrelChains {
	const IR::Program*                  program = nullptr;
	uint64_t                            hash    = 0;
	std::unordered_set<const IR::Inst*> interior;
	std::unordered_set<const IR::Inst*> heads;
};
bool MovrelLink(const IR::Inst& s, IR::Value& m, uint32_t& k) {
	if (s.GetOpcode() != ValueOpcode::SelectU32 || s.NumArgs() != 3) return false;
	const IR::Inst* c = s.Arg(0).Resolve().TryInstruction();
	if (c == nullptr || c->GetOpcode() != ValueOpcode::IEqual32) return false;
	for (uint32_t j = 0; j < 2; j++) {
		const IR::Value a = c->Arg(j).Resolve(), o = c->Arg(1u - j).Resolve();
		if (a.IsImmediate() && a.GetType() == IR::Type::U32 && !o.IsImmediate()) {
			m = o;
			k = a.U32();
			return true;
		}
	}
	return false;
}
// Inner link of `s` (its false arm) when it continues the chain on the same M and has no other use.
const IR::Inst* MovrelPrev(const IR::Inst& s, const IR::Value& m) {
	const IR::Inst* p = s.Arg(2).TryInstruction(); // direct use only: an Identity in between keeps p printed
	IR::Value       pm;
	uint32_t        pk = 0;
	if (p == nullptr || p->UseCount() != 1 || !MovrelLink(*p, pm, pk) || !(pm == m)) return nullptr;
	return p;
}
const MovrelChains& MovrelOf(const IR::Program& program) {
	static thread_local MovrelChains g;
	if (g.program == &program && g.hash == program.shader_hash) return g;
	g          = MovrelChains {};
	g.program  = &program;
	g.hash     = program.shader_hash;
	std::vector<const IR::Inst*> links;
	for (const auto* blk: program.blocks) {
		for (const auto& inst: *blk) {
			IR::Value m;
			uint32_t  k = 0;
			if (MovrelLink(inst, m, k)) {
				links.push_back(&inst);
				if (const auto* p = MovrelPrev(inst, m)) g.interior.insert(p);
			}
		}
	}
	for (const auto* l: links) {
		if (g.interior.count(l) != 0) continue;
		IR::Value m;
		uint32_t  k = 0;
		MovrelLink(*l, m, k);
		std::vector<const IR::Inst*> chain;
		for (const IR::Inst* p = MovrelPrev(*l, m); p != nullptr; p = MovrelPrev(*p, m)) chain.push_back(p);
		if (chain.size() + 1 >= 4) {
			g.heads.insert(l);
		} else {
			for (const auto* p: chain) g.interior.erase(p);
		}
	}
	return g;
}

void EmitInst(MslCtx& ctx, const IR::Inst& inst, const ComputeShape& shape) {
	using O              = ValueOpcode;
	const auto op        = inst.GetOpcode();
	const auto type_name = MslTypeName(inst.GetType());
	const auto assign    = [&](uint32_t h, const std::string& expr) {
		const auto t = inst.GetType();
		if (ctx.selective_precision && (t == IR::Type::F32 || t == IR::Type::F32x2) &&
		    !ctx.precise.contains(&inst)) {
			ctx.b.Line("{");
			ctx.b.Line("#pragma METAL fp math_mode(fast)");
			ctx.b.Line(fmt::format("{} = {};", ctx.Name(inst, h), expr));
			ctx.b.Line("}");
			return;
		}
		ctx.b.Line(fmt::format("{} = {};", ctx.Name(inst, h), expr));
	};
	// Opcodes that are not per-half.
	switch (op) {
		case O::Phi:
		case O::Void:
		case O::Reference:
		case O::ReferenceU32:
		case O::ControlNop:
		case O::DataCacheInvalidate:
		case O::Sendmsg:
		case O::TtraceData:
		case O::InstPrefetch: return;
		case O::Barrier: {
			// mslrest2 perf: S_BARRIER in a workgroup that is ONE Metal simdgroup (<= 32 threads;
			// wave64 at 2 lanes/thread) orders nothing beyond the simdgroup: simdgroup_barrier with
			// a threadgroup memory fence (KK: Kyty SPIR-V's barrier on a one-wave workgroup ->
			// simdgroup-scope fence) instead of threadgroup_barrier + device flush.
			// vsperf / perfauto (transform vs_no_barrier, MslXform.h; KYTY_MSL_VS_PERF=1 still
			// selects it): a vertex function has no threadgroup and the B vertex
			// stage is one lane with no LDS; KK (Kyty SPIR-V -> NIR) prints no barrier there. The
			// device-flush barrier is not free: it pins every later load below it, so Apple's compiler
			// cannot hoist the uniform constant-buffer reads into the per-draw constant program
			// (cloud 1548c37d VS: 64 -> 21 device loads per vertex without it).
			if (Xf::On(Xf::VsNoBarrier) && ctx.program.stage == ShaderType::Vertex) {
				ctx.b.Line("// S_BARRIER: none in a vertex function (vsperf)");
				return;
			}
			const uint32_t n     = shape.threads[0] * shape.threads[1] * shape.threads[2];
			const uint32_t metal = ctx.lanes == 2 ? (n + 63u) / 64u * 32u : n;
			if (ctx.program.stage == ShaderType::Compute && metal <= 32u)
				ctx.b.Line("simdgroup_barrier(mem_flags::mem_threadgroup);");
			else
				ctx.b.Line(
				    "threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);");
			return;
		}
		// mslaudit: S_WAITCNT lgkmcnt(0) / vmcnt(0) order only the wave's OWN memory ops (other
		// lanes of the wave read the data next). = SPIR-V EmitSharedMemoryBarrier /
		// EmitStoreCompletion (subgroup scope; KK: simdgroup atomic_thread_fence). It was a
		// threadgroup_barrier (execution barrier for ALL waves of the group) = undefined behaviour
		// when the waitcnt sits in a wave-uniform branch other waves skip (hang / wrong data).
		// KYTY_MSL_WAITCNT_FENCE=0 restores the old form.
		case O::SharedMemoryBarrier:
		case O::StoreCompletion: {
			if (ctx.gfx) return; // graphics: no workgroup memory (as the SPIR-V emitter)
			static const bool old_form = [] {
				const char* e = std::getenv("KYTY_MSL_WAITCNT_FENCE");
				return e != nullptr && std::strcmp(e, "0") == 0;
			}();
			if (old_form) {
				if (op == O::SharedMemoryBarrier)
					ctx.b.Line(
					    "threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);");
				return;
			}
			if (op == O::StoreCompletion)
				ctx.b.Line("KYTY_LDS_FENCE(mem_flags::mem_device); // s_waitcnt vmcnt(0)");
			else if (Mem::LdsThreadBytes() != 0u)
				ctx.b.Line("// s_waitcnt lgkmcnt(0): per-invocation LDS");
			else
				ctx.b.Line(
				    Mem::LdsIsDevice()
				        ? "KYTY_LDS_FENCE(mem_flags::mem_device); // s_waitcnt lgkmcnt(0)"
				        : "KYTY_LDS_FENCE(mem_flags::mem_threadgroup); // s_waitcnt lgkmcnt(0)");
			return;
		}
		default: break;
	}
	if (!ctx.lds_fence.empty() && ctx.lds_fence.count(&inst) != 0 && Mem::LdsThreadBytes() == 0u)
		ctx.b.Line(Mem::LdsIsDevice() ? "KYTY_LDS_FENCE(mem_flags::mem_device); // mslaudit: LDS "
		                                "store -> load without barrier"
		                              : "KYTY_LDS_FENCE(mem_flags::mem_threadgroup); // mslaudit: "
		                                "LDS store -> load without barrier");
	for (uint32_t h = 0; h < ctx.lanes; h++) {
		ctx.half = h;
		switch (op) {
			case O::Identity: assign(h, ctx.Val(inst.Arg(0))); continue;
			case O::UndefU1: assign(h, "false"); continue;
			case O::UndefU8:
			case O::UndefU16:
			case O::UndefU32:
			case O::UndefU64: assign(h, type_name + "(0)"); continue;
			case O::IsHelperInvocation:
				if (ctx.gfx && GfxEmitInst(ctx, *ctx.gfx, inst)) continue;
				assign(h, "false");
				continue;
			case O::GetUserData: {
				const auto& regs = ctx.program.bindings.user_data_registers;
				const auto  reg  = IR::RegIndex(inst.Arg(0).ScalarRegister());
				const auto  at   = std::lower_bound(regs.begin(), regs.end(), reg);
				assign(h, at == regs.end() || *at != reg
				              ? "0u"
				              : fmt::format("push[{}]", at - regs.begin()));
				continue;
			}
			case O::GetDispatchThreadExtent: {
				const auto start = ctx.program.bindings.dispatch_thread_dword;
				assign(h, start == IR::PushData::NoStart || !inst.Arg(0).IsImmediate()
				              ? std::string("0u")
				              : fmt::format("push[{}]", start + std::min(inst.Arg(0).U32(), 2u)));
				continue;
			}
			case O::GetBuiltin: {
				if (!inst.Arg(0).IsImmediate() || !inst.Arg(1).IsImmediate()) break;
				if (ctx.gfx) {
					const auto gexpr =
					    GfxBuiltin(ctx, *ctx.gfx, inst.Arg(0).U32(), inst.Arg(1).U32());
					if (gexpr.empty()) break;
					assign(h, gexpr);
					continue;
				}
				const auto expr = BuiltinExpr(ctx, shape, inst.Arg(0).U32(), inst.Arg(1).U32());
				if (expr.empty()) break;
				assign(h, expr);
				continue;
			}
			case O::LaneId: assign(h, ctx.LaneExpr(h)); continue;
			// S_GETPC: guest PCs stay shader-relative, exactly as the SPIR-V backend
			// (EmitGetShaderBase = 0).
			case O::GetShaderBase: assign(h, "0ul"); continue;
			case O::Ballot: {
				const auto lo = fmt::format("kyty_ballot({}, lid)", ctx.Val(inst.Arg(0), 0));
				const auto hi = ctx.lanes == 2
				                    ? fmt::format("kyty_ballot({}, lid)", ctx.Val(inst.Arg(0), 1))
				                    : std::string("0u");
				assign(h, fmt::format("uint4({}, {}, 0u, 0u)", lo, hi));
				continue;
			}
			case O::AnyLane: {
				if (ctx.vote_merge &&
				    ctx.lanes == 2) { // wave-uniform: vote once (half 0), half 1 copies it
					assign(h, h == 0 ? fmt::format("kyty_any(({}) || ({}), lid)",
					                               ctx.Val(inst.Arg(0), 0), ctx.Val(inst.Arg(0), 1))
					                 : ctx.Name(inst, 0));
					continue;
				}
				auto e = fmt::format("kyty_any({}, lid)", ctx.Val(inst.Arg(0), 0));
				if (ctx.lanes == 2)
					e = fmt::format("({} || kyty_any({}, lid))", e, ctx.Val(inst.Arg(0), 1));
				assign(h, e);
				continue;
			}
			case O::ReadFirstLane: {
				// first active (EXEC) lane of the wave; arg1 is the per-lane EXEC bit
				const auto v0 = ctx.Val(inst.Arg(0), 0), e0 = ctx.Val(inst.Arg(1), 0);
				if (ctx.lanes == 1) {
					assign(h, fmt::format("simd_shuffle({}, ctz(kyty_ballot({}, lid)))", v0, e0));
				} else if (ctx.vote_merge && h == 1) {
					assign(h, ctx.Name(inst, 0)); // wave-uniform result: computed once in half 0
				} else {
					const auto v1 = ctx.Val(inst.Arg(0), 1), e1 = ctx.Val(inst.Arg(1), 1);
					assign(h, fmt::format(
					              "(kyty_any({0}, lid) ? simd_shuffle({1}, ctz(kyty_ballot({0}, "
					              "lid))) : simd_shuffle({2}, ctz(kyty_ballot({3}, lid))))",
					              e0, v0, v1, e1));
				}
				continue;
			}
			case O::ReadLane: {
				const auto lane = ctx.Val(inst.Arg(1), 0);
				if (ctx.lanes == 1) {
					assign(h, fmt::format("simd_shuffle({}, {} & 31u)", ctx.Val(inst.Arg(0), 0),
					                      lane));
				} else {
					assign(h, fmt::format("((({} & 63u) < 32u) ? simd_shuffle({}, {} & 31u) : "
					                      "simd_shuffle({}, {} & 31u))",
					                      lane, ctx.Val(inst.Arg(0), 0), lane,
					                      ctx.Val(inst.Arg(0), 1), lane));
				}
				continue;
			}
			default: break;
		}
		if (op == O::SelectU32 && (Xf::On(Xf::MovrelSwitch) || Xf::On(Xf::MovrelTree))) {
			const auto& mc = MovrelOf(ctx.program);
			if (mc.interior.count(&inst) != 0) continue;
			if (mc.heads.count(&inst) != 0 && !Xf::On(Xf::MovrelSwitch)) {
				// movrel_tree: balanced compare tree over the sorted distinct constants; the leaf
				// compares M == k, so an M that is none of them still gives the chain's base.
				IR::Value m;
				uint32_t  k = 0;
				MovrelLink(inst, m, k);
				std::map<uint32_t, std::string> arms; // head-most link wins on a repeated constant
				const IR::Inst*                 l = &inst;
				for (;;) {
					IR::Value lm;
					uint32_t  lk = 0;
					MovrelLink(*l, lm, lk);
					arms.emplace(lk, ctx.Val(l->Arg(1)));
					const IR::Inst* p = MovrelPrev(*l, m);
					if (p == nullptr) break;
					l = p;
				}
				const std::string                                     mv = ctx.Val(m), base = ctx.Val(l->Arg(2));
				const std::vector<std::pair<uint32_t, std::string>> v(arms.begin(), arms.end());
				const std::function<std::string(size_t, size_t)> tree = [&](size_t lo, size_t hi) -> std::string {
					if (hi - lo == 1) return fmt::format("(({} == {}) ? {} : {})", mv, MslU32(v[lo].first), v[lo].second, base);
					const size_t mid = lo + (hi - lo) / 2;
					return fmt::format("(({} < {}) ? {} : {})", mv, MslU32(v[mid].first), tree(lo, mid), tree(mid, hi));
				};
				assign(h, tree(0, v.size()));
				continue;
			}
			if (mc.heads.count(&inst) != 0) {
				IR::Value m;
				uint32_t  k = 0;
				MovrelLink(inst, m, k);
				std::set<uint32_t> seen;
				const auto         name = ctx.Name(inst, h);
				ctx.b.Line(fmt::format("switch ({}) {{", ctx.Val(m)));
				const IR::Inst* l = &inst;
				for (;;) {
					IR::Value lm;
					uint32_t  lk = 0;
					MovrelLink(*l, lm, lk);
					if (seen.insert(lk).second)
						ctx.b.Line(fmt::format("    case {}: {} = {}; break;", MslU32(lk), name, ctx.Val(l->Arg(1))));
					const IR::Inst* p = MovrelPrev(*l, m);
					if (p == nullptr) break;
					l = p;
				}
				ctx.b.Line(fmt::format("    default: {} = {}; break;", name, ctx.Val(l->Arg(2))));
				ctx.b.Line("}");
				continue;
			}
		}
		if (ctx.gfx && GfxEmitInst(ctx, *ctx.gfx, inst)) continue;
		if (const auto lines = NirSplitLines(ctx, inst); !lines.empty()) {
			ctx.b.Line(lines);
			continue;
		}
		if (const auto re = NirReassocText(ctx, inst); !re.empty()) {
			assign(h, re);
			continue;
		}
		std::vector<std::string> args;
		for (size_t i = 0; i < inst.NumArgs(); i++)
			args.push_back(ctx.Val(inst.Arg(i)));
		if (!type_name.empty() && inst.NumArgs() == IR::NumArgsOf(op)) {
			if (const auto expr = AluExpression(op, args); !expr.empty()) {
				assign(h, expr);
				continue;
			}
		}
		if (EmitMemoryInst(ctx, inst)) continue;
		if (h == 0) ctx.Unsupported(inst);
		ctx.b.Comment(fmt::format("UNSUPPORTED {}", IR::ValueOpcodeName(op)));
		if (!type_name.empty()) assign(h, type_name + "(0)");
	}
}

} // namespace

const char* MslPrelude() {
	return Prelude;
}

// nps5 gpuguard loop budget (MslOptions::loop_cap): nps5_lg_step returns true while the budget
// lasts; it saturates, so once spent every later check of the invocation fails at once (nested and
// consecutive loops all unwind).
static std::string LoopGuardHelper(uint32_t cap) {
	if (cap == 0u) return {};
	return fmt::format("// nps5 gpuguard: {} loop iterations per invocation, then the loop breaks "
	                   "(wrong result instead of a GPU lockup)\n"
	                   "static inline __attribute__((always_inline)) bool nps5_lg_step(thread "
	                   "uint& n) {{ if (n >= {}u) return false; ++n; return true; }}\n",
	                   cap, cap);
}

// mslaudit (NXNative "Threadgroup stores in warp-synchronous code"): GCN compilers drop S_BARRIER
// in one-wave groups, so a lane stores its LDS word and loads its neighbour's with only s_waitcnt
// between. In MSL the load does not alias the thread's own store and Metal may hoist it above the
// store. Forward dataflow to a fixed point: state = "an LDS store/atomic may precede with no
// barrier/fence since". An LDS load reached in that state gets a fence (which clears the state; one
// fence orders all prior stores). Not a fence after every store (25,108 fences crashed the M2's
// compiler in NX).
static std::unordered_set<const IR::Inst*> LdsFenceSites(const IR::Program& program) {
	std::unordered_set<const IR::Inst*> sites;
	const size_t n         = std::min(program.blocks.size(), program.block_info.size());
	bool         any_store = false, any_load = false;
	for (size_t i = 0; i < n; i++)
		for (const auto& inst: *program.blocks[i]) {
			const auto a = IR::SharedAccessOf(inst.GetOpcode());
			any_load |= a == IR::SharedAccess::Read;
			any_store |= a != IR::SharedAccess::None && a != IR::SharedAccess::Read;
		}
	if (!any_store || !any_load) return sites;
	std::unordered_map<uint32_t, size_t> index;
	for (size_t i = 0; i < n; i++)
		index[program.block_info[i].id] = i;
	const auto succs = [&](size_t i) {
		std::vector<size_t> out;
		const auto&         t   = program.block_info[i].terminator;
		const auto          add = [&](uint32_t id) {
			if (const auto f = index.find(id); f != index.end()) out.push_back(f->second);
		};
		switch (t.kind) {
			case CFG::TerminatorKind::Branch: add(t.true_block); break;
			case CFG::TerminatorKind::ConditionalBranch:
				add(t.true_block);
				add(t.false_block);
				break;
			case CFG::TerminatorKind::IndirectBranch:
				for (auto id: t.indirect_targets)
					add(id);
				for (auto id: t.indirect_selector_targets)
					add(id);
				break;
			default: break;
		}
		return out;
	};
	const auto transfer = [&](size_t i, bool dirty, bool mark) {
		for (const auto& inst: *program.blocks[i]) {
			const auto op = inst.GetOpcode();
			if (op == ValueOpcode::Barrier || op == ValueOpcode::SharedMemoryBarrier) {
				dirty = false;
				continue;
			}
			const auto a = IR::SharedAccessOf(op);
			if (a == IR::SharedAccess::None) continue;
			if (a == IR::SharedAccess::Read) {
				if (dirty && mark) sites.insert(&inst);
				dirty = false;
			} else {
				dirty = true;
			}
		}
		return dirty;
	};
	std::vector<char>   in(n, 0), seen(n, 0);
	std::vector<size_t> work {0};
	seen[0] = 1;
	while (!work.empty()) {
		const size_t i = work.back();
		work.pop_back();
		const bool out = transfer(i, in[i] != 0, false);
		for (const size_t s: succs(i)) {
			if (!seen[s] || (out && !in[s])) {
				seen[s] = 1;
				in[s] |= out ? 1 : 0;
				work.push_back(s);
			}
		}
	}
	for (size_t i = 0; i < n; i++)
		if (seen[i]) transfer(i, in[i] != 0, true);
	return sites;
}

static bool ProgramHasBarrier(const IR::Program& program) {
	for (const auto* block: program.blocks)
		for (const auto& inst: *block)
			if (inst.GetOpcode() == ValueOpcode::Barrier ||
			    inst.GetOpcode() == ValueOpcode::SharedMemoryBarrier)
				return true;
	return false;
}

std::string MslCtx::LoopGuard(std::string* decl) {
	if (loop_cap == 0u) return {};
	if (!loop_private) return "if (!nps5_lg_step(nps5_lg_n)) break; // nps5 gpuguard";
	const std::string n = fmt::format("nps5_lg_b{}", loop_guards++);
	*decl               = fmt::format("uint {} = 0u;", n);
	return fmt::format("if (!nps5_lg_step({})) break; // nps5 gpuguard (private: barrier program)",
	                   n);
}

// ---------------------------------------------------------------------------------------------
// Structured emission: the frontend CFG already carries SPIR-V-style merge/continue info
// (terminator.merge_block / continue_block / loop_header), so if/else and loops are rebuilt as
// nested MSL statements instead of the dispatcher `switch (blk)` loop (which Apple GPUs run
// poorly). Anything not expressible (indirect branches, jumps across several constructs, a block
// reached twice) makes Plan fail and the caller falls back to the dispatcher. A dry run (no output)
// plans first so no half-emitted text or double-counted unsupported opcodes can result.
// ---------------------------------------------------------------------------------------------
class Structurer {
public:
	Structurer(MslCtx& ctx, const ComputeShape& shape, bool comments, bool dry,
	           std::set<uint32_t>& need_do)
	    : ctx_(ctx), shape_(shape), comments_(comments), dry_(dry), need_do_(need_do),
	      emitted_(ctx.program.blocks.size(), false) {}

	bool Run() {
		const auto& program = ctx_.program;
		if (std::getenv("KYTY_MSL_DEBUG") && dry_)
			for (size_t i = 0; i < program.blocks.size(); i++) {
				const auto& t = program.block_info[i].terminator;
				std::fprintf(stderr, "blk %u kind %d t=%u f=%u merge=%u cont=%u loop=%d\n",
				             program.block_info[i].id, (int)t.kind, t.true_block, t.false_block,
				             t.merge_block, t.continue_block, (int)t.loop_header);
			}
		if (program.blocks.empty()) return true;
		Region(program.block_info[0].id, UINT32_MAX);
		return !failed_;
	}

private:
	struct Loop {
		uint32_t    header, merge, cont;
		bool        trivial_cont; // continue target needs no code: `continue` directly
		bool        in_do     = false;
		bool        flag_used = false;
		std::string flag;
	};

	MslCtx&             ctx_;
	const ComputeShape& shape_;
	bool                comments_, dry_, failed_ = false;
	std::set<uint32_t>& need_do_; // loop headers whose body jumps to a non-trivial continue block
	                              // (dry run -> real run)
	std::vector<bool>     emitted_;
	std::vector<Loop>     loops_;
	std::vector<uint32_t> stops_;
	uint32_t              loop_counter_ = 0;

	void FailAt(int line) {
		failed_ = true;
		if (std::getenv("KYTY_MSL_DEBUG"))
			std::fprintf(stderr, "msl structurer failed at MslEmitter.cpp:%d\n", line);
	}
	void Line(const std::string& t) {
		if (!dry_) ctx_.b.Line(t);
	}
	void Open(const std::string& t) {
		if (!dry_) ctx_.b.Open(t);
	}
	void Close(const std::string& t = {}) {
		if (!dry_) ctx_.b.Close(t);
	}
	void Edge(const IR::Block* from, const IR::Block* to) {
		if (!dry_) EmitEdge(ctx_, from, to);
	}
	size_t           Index(uint32_t id) const { return BlockIndexById(ctx_.program, id); }
	const IR::Block* BlockOf(uint32_t id) const {
		auto i = Index(id);
		return i == SIZE_MAX ? nullptr : ctx_.program.blocks[i];
	}

	bool OnlyEmptyAndPhi(size_t idx, bool allow_phi) const {
		for (const auto& inst: *ctx_.program.blocks[idx]) {
			if (inst.GetOpcode() == ValueOpcode::Phi) {
				if (!allow_phi) return false;
				continue;
			}
			return false;
		}
		return true;
	}
	bool HasPhi(uint32_t id) const {
		const auto* b = BlockOf(id);
		return b != nullptr && !b->empty() && b->begin()->GetOpcode() == ValueOpcode::Phi;
	}

	void Insts(size_t idx) {
		if (dry_) return;
		if (comments_) ctx_.b.Comment(fmt::format("block {}", ctx_.program.block_info[idx].id));
		for (const auto& inst: *ctx_.program.blocks[idx]) {
			if (comments_ && inst.GetOpcode() != ValueOpcode::Phi)
				ctx_.b.Comment(std::string(IR::ValueOpcodeName(inst.GetOpcode())));
			EmitInst(ctx_, inst, shape_);
		}
	}

	// Control transfer from block `from` to `target`; `stop` = where the current region ends.
	void Jump(size_t from, uint32_t target, uint32_t stop) {
		const auto*  fb = ctx_.program.blocks[from];
		const size_t ti = Index(target);
		if (ti == SIZE_MAX) {
			Line(ctx_.ret);
			return;
		}
		const auto* tb = ctx_.program.blocks[ti];
		if (target == stop) {
			Edge(fb, tb);
			return;
		}
		if (!loops_.empty()) {
			auto& L = loops_.back();
			if (target == L.cont) {
				if (L.trivial_cont) {
					// trivial continue block (or the header itself): its only job is the back-edge
					// phi copy
					if (L.cont == L.header)
						Edge(fb, tb);
					else {
						Edge(fb, tb);
						Edge(tb, BlockOf(L.header));
					}
					Line("continue;");
				} else {
					Edge(fb, tb);
					if (!L.in_do) {
						FailAt(__LINE__);
						return;
					}
					need_do_.insert(L.header);
					Line("break;");
				}
				return;
			}
			if (target == L.merge) {
				Edge(fb, tb);
				if (L.in_do) {
					L.flag_used = true;
					Line(L.flag + " = true;");
				}
				Line("break;");
				return;
			}
			for (size_t i = 0; i + 1 < loops_.size(); i++) {
				if (target == loops_[i].cont || target == loops_[i].merge) {
					FailAt(__LINE__);
					return;
				}
			}
		}
		for (auto s: stops_)
			if (s == target) {
				FailAt(__LINE__);
				return;
			} // would jump over a construct
		Edge(fb, tb);
		Region(target, stop);
	}

	void EmitLoop(size_t idx, uint32_t stop) {
		const auto& info = ctx_.program.block_info[idx];
		const auto& term = info.terminator;
		const auto* hb   = ctx_.program.blocks[idx];
		(void)hb;
		const size_t ci = Index(term.continue_block);
		if (ci == SIZE_MAX) {
			FailAt(__LINE__);
			return;
		}
		Loop       L {info.id, term.merge_block, term.continue_block, false};
		const bool single = term.continue_block == info.id;
		L.trivial_cont =
		    single || (OnlyEmptyAndPhi(ci, false) &&
		               ctx_.program.block_info[ci].terminator.kind == CFG::TerminatorKind::Branch &&
		               ctx_.program.block_info[ci].terminator.true_block == info.id);
		L.flag            = fmt::format("lf{}", loop_counter_++);
		const bool use_do = !L.trivial_cont && (dry_ || need_do_.count(info.id) != 0);
		if (use_do) Line(fmt::format("bool {} = false;", L.flag));
		std::string       lg_decl;
		const std::string lg = dry_ ? std::string() : ctx_.LoopGuard(&lg_decl);
		if (!lg_decl.empty()) Line(lg_decl);
		Open("for (;;)");
		if (!lg.empty()) Line(lg);
		Insts(idx);
		loops_.push_back(L);
		stops_.push_back(L.trivial_cont ? UINT32_MAX : L.cont);
		if (use_do) {
			loops_.back().in_do = true;
			Open("do");
		}
		Terminator(idx, L.trivial_cont ? UINT32_MAX : L.cont, true);
		if (!L.trivial_cont) {
			if (use_do) {
				Close("while (false);");
				loops_.back().in_do = false;
				if (loops_.back().flag_used) Line(fmt::format("if ({}) break;", L.flag));
			}
			if (emitted_[ci]) {
				FailAt(__LINE__);
			}
			emitted_[ci] = true;
			Insts(ci);
			ContinueTerminator(ci, info.id, term.merge_block);
		}
		stops_.pop_back();
		loops_.pop_back();
		Close();
	}

	// Terminator of the continue block: back edge to the header (fall off / continue) and, for
	// do-while style loops, a conditional exit to the loop merge.
	void ContinueTerminator(size_t ci, uint32_t header, uint32_t merge) {
		const auto& cterm = ctx_.program.block_info[ci].terminator;
		const auto* cb    = ctx_.program.blocks[ci];
		const auto  arm   = [&](uint32_t t, bool last_fall) {
			if (t == header) {
				Edge(cb, BlockOf(header));
				if (!last_fall) Line("continue;");
				return true;
			}
			if (t == merge && t != UINT32_MAX) {
				Edge(cb, BlockOf(merge));
				Line("break;");
				return true;
			}
			return false;
		};
		if (cterm.kind == CFG::TerminatorKind::Branch) {
			if (!arm(cterm.true_block, true)) FailAt(__LINE__);
		} else if (cterm.kind == CFG::TerminatorKind::ConditionalBranch &&
		           !ctx_.program.block_info[ci].condition.IsEmpty() &&
		           cterm.true_block != cterm.false_block) {
			Open(fmt::format("if ({})", dry_ ? std::string("c")
			                                 : BranchExpr(ctx_, ctx_.program.block_info[ci])));
			if (!arm(cterm.true_block, false)) FailAt(__LINE__);
			Close();
			if (!arm(cterm.false_block, true)) FailAt(__LINE__);
		} else {
			FailAt(__LINE__);
		}
	}

	// Emits the block terminator; for a loop header `loop` the loop frame is already pushed.
	// Returns the block to continue with after the construct (merge) or UINT32_MAX.
	uint32_t Terminator(size_t idx, uint32_t stop, bool loop) {
		const auto& info = ctx_.program.block_info[idx];
		const auto& term = info.terminator;
		switch (term.kind) {
			case CFG::TerminatorKind::Branch: Jump(idx, term.true_block, stop); return UINT32_MAX;
			case CFG::TerminatorKind::ConditionalBranch: {
				if (info.condition.IsEmpty() || Index(term.true_block) == SIZE_MAX ||
				    Index(term.false_block) == SIZE_MAX) {
					Line(ctx_.ret);
					return UINT32_MAX;
				}
				if (term.true_block == term.false_block) {
					Jump(idx, term.true_block, stop);
					return UINT32_MAX;
				}
				uint32_t merge    = loop ? UINT32_MAX : term.merge_block;
				uint32_t arm_stop = loop ? stop : (merge != UINT32_MAX ? merge : stop);
				if (!loop && merge == UINT32_MAX) {
					// both arms must leave (break/continue/return); falling out of an arm is
					// unstructured
					arm_stop = UINT32_MAX;
				}
				if (!loop && merge != UINT32_MAX) stops_.push_back(merge);
				Open(fmt::format("if ({})", dry_ ? std::string("c") : BranchExpr(ctx_, info)));
				Jump(idx, term.true_block, arm_stop);
				Close();
				const bool skip_else = term.false_block == arm_stop && arm_stop != UINT32_MAX &&
				                       !HasPhi(term.false_block);
				if (!skip_else) {
					Open("else");
					Jump(idx, term.false_block, arm_stop);
					Close();
				}
				if (!loop && merge != UINT32_MAX) stops_.pop_back();
				return merge;
			}
			default:
				Line(ctx_.ret);
				return UINT32_MAX; // Return / Unsupported (indirect: see below)
		}
	}

	void Region(uint32_t id, uint32_t stop) {
		while (!failed_) {
			if (id == stop) return;
			if (!loops_.empty()) {
				auto& L = loops_.back();
				if (id == L.cont && L.trivial_cont) {
					if (L.cont != L.header)
						Edge(ctx_.program.blocks[Index(L.cont)], BlockOf(L.header));
					Line("continue;");
					return;
				}
				if (id == L.merge) {
					if (L.in_do) {
						L.flag_used = true;
						Line(L.flag + " = true;");
					}
					Line("break;");
					return;
				}
			}
			const size_t idx = Index(id);
			if (idx == SIZE_MAX) {
				Line(ctx_.ret);
				return;
			}
			if (emitted_[idx]) {
				FailAt(__LINE__);
				return;
			}
			emitted_[idx]    = true;
			const auto& info = ctx_.program.block_info[idx];
			const auto& term = info.terminator;
			if (term.kind == CFG::TerminatorKind::IndirectBranch) {
				FailAt(__LINE__);
				return;
			}
			if (term.loop_header) {
				if (term.kind != CFG::TerminatorKind::Branch &&
				    term.kind != CFG::TerminatorKind::ConditionalBranch) {
					FailAt(__LINE__);
					return;
				}
				EmitLoop(idx, stop);
				// after the loop: its merge block (edge copies into it were done at each break)
				if (term.merge_block == UINT32_MAX) return;
				id = term.merge_block;
				continue;
			}
			Insts(idx);
			const uint32_t next = Terminator(idx, stop, false);
			if (next == UINT32_MAX) return;
			id = next;
		}
	}
};

// True when a wave64 compute program uses nothing that crosses guest lanes (wave ops, DPP/permutes,
// LDS, barriers, GDS counters, exec/vcc masks as integers; AnyLane only feeds wave-uniform branches
// whose bodies are exec-masked, so a per-simdgroup answer is equivalent): then each guest lane can
// be its own Metal thread (threadgroup = the guest threadgroup, two simdgroups per wave64, half the
// registers per thread) instead of two halves per thread.
bool CanRunOneLanePerThread(const IR::Program& program, uint32_t guest_threads) {
	if (guest_threads == 0u || guest_threads % 64u != 0u)
		return false; // partial waves keep the full-wave (halves) model
	for (const auto* block: program.blocks) {
		for (const auto& inst: *block) {
			const std::string_view   n     = IR::ValueOpcodeName(inst.GetOpcode());
			static const char* const bad[] = {
			    "Ballot",      "ReadFirstLane", "ReadLane", "WriteLane", "Permlane", "Permute",
			    "Swizzle",     "Dpp",           "Wqm",      "Shared",    "Barrier",  "DataAppend",
			    "DataConsume", "ExecLo",        "ExecHi",   "VccLo",     "VccHi",    "ThreadBit",
			    "ScalarMask",  "Bvh",           "Scratch",  "Sendmsg",   "GetM0",    "SetM0",
			    "ReadClock",   "IsHelper"};
			for (const char* b: bad) {
				if (n.find(b) != std::string_view::npos) return false;
			}
		}
	}
	return true;
}

uint32_t ChooseLanes(const IR::Program& program, ShaderStageInputInfo input_info,
                     uint32_t lanes_override) {
	if (lanes_override != 0) return lanes_override;
	uint32_t lanes = program.wave_size == 64u ? 2u : 1u;
	// KYTY_MSL_WAVE64=halves|lane overrides; default (auto): one lane per thread when no cross-lane
	// op is used.
	const char* mode        = std::getenv("KYTY_MSL_WAVE64");
	const bool  want_single = mode != nullptr ? std::strcmp(mode, "lane") == 0 : true;
	if (lanes == 2 && program.stage == ShaderType::Compute && want_single &&
	    CanRunOneLanePerThread(program, [&] {
		    const auto sh = ShapeOf(program, input_info);
		    return sh.threads[0] * sh.threads[1] * sh.threads[2];
	    }()))
		lanes = 1;
	return lanes;
}

bool ModeFix6() { // read per call: the unit tests switch it inside one process
	const char* e = std::getenv("KYTY_MSL_FIX6");
	return e != nullptr && std::strcmp(e, "0") != 0;
}

MslResult EmitProgram(const IR::Program& program, ShaderStageInputInfo input_info,
                      const MslOptions& options) {
	MslResult result;
	MslCtx    ctx(program, input_info);
	Xf::Begin(program.stage, program.shader_hash); // perfauto: transform set of this program
	if (program.stage == ShaderType::Pixel || program.stage == ShaderType::Compute) {
		const char* stage_name = program.stage == ShaderType::Pixel ? "fs" : "cs";
		const bool  all_fast =
		    AllFastApplies(program.shader_hash, stage_name); // low: empty precise set
		ctx.selective_precision =
		    all_fast || SelectivePrecisionApplies(program.shader_hash, stage_name);
		if (ctx.selective_precision && !all_fast) {
			IR::CollectPreciseSet(program, GetCodegenOptions().gpu_accuracy_extra, ctx.precise);
		}
	}
	ctx.options_comments = options.comments;
	ctx.loop_cap         = options.loop_cap;
	if (const char* e = std::getenv("KYTY_MSL_LOOP_CAP"))
		ctx.loop_cap = static_cast<uint32_t>(std::strtoul(e, nullptr, 0));
	ctx.loop_private = ctx.loop_cap != 0u && ProgramHasBarrier(program);
	if (const char* e = std::getenv("KYTY_MSL_LDS_FENCE"); e == nullptr || std::strcmp(e, "0") != 0)
		ctx.lds_fence = LdsFenceSites(program);
	// MadMode (CodegenOptions): Exact = unfused everywhere; Position = unfused only on the position
	// data flow (graphics: kept unfused for the whole VS/PS here); Fused = fma everywhere. Compute
	// shaders have no position export => fused unless Exact.
	const auto mad_mode = GetCodegenOptions().mad_mode;
	const bool fuse_mad = mad_mode == MadMode::Fused ||
	                      (mad_mode == MadMode::Position && program.stage == ShaderType::Compute);
	SetFuseMad(fuse_mad);
	const auto shape = ShapeOf(program, input_info);

	// Lane model (see STATUS.md): wave64 on Apple's simdgroup(32) = two guest halves per thread.
	uint32_t lanes    = ChooseLanes(program, input_info, options.lanes_override);
	ctx.wave64_single = program.wave_size == 64u && lanes == 1;
	// mslbadopt: default ON (harness exact + faster on 472/473/304/753 and the 16 changed v1.1
	// kernels); KYTY_MSL_VOTE_MERGE=0 = old text.
	if (const char* e = std::getenv("KYTY_MSL_VOTE_MERGE");
	    e == nullptr || std::strcmp(e, "0") != 0)
		ctx.vote_merge = true;
	ctx.lanes      = lanes;
	result.lanes   = lanes;
	uint32_t total = shape.threads[0] * shape.threads[1] * shape.threads[2];
	if (lanes == 2) {
		total = ((total + 63u) / 64u) * 32u;
	}
	result.threadgroup[0] = total;

	const bool gfx_stage   = GfxIsStage(program.stage);
	const bool gfx_relaxed = program.stage == ShaderType::Pixel && GfxPixelRelaxed() &&
	                         (options.kk_bda || KkBdaRequested());
	// mslfix4: KYTY_MSL_VS_SINGLE_LANE=0 = old (real simdgroup ops in the VS with lid 0)
	const bool vs_single_lane = program.stage == ShaderType::Vertex &&
	                            (options.kk_bda || KkBdaRequested() || !GetHostVertexSubgroups()) &&
	                            [] {
		                            const char* e = std::getenv("KYTY_MSL_VS_SINGLE_LANE");
		                            return e == nullptr || std::strcmp(e, "0") != 0;
	                            }();
	SetPreciseTranscendentals(gfx_relaxed);
	// mslfix6: KYTY_MSL_GFX_MINMAX=0 = old text (Metal fmin/fmax/fmin3/fmax3/fmedian3)
	const bool kk_minmax =
	    gfx_stage && (options.kk_bda || KkBdaRequested()) && KkMinMaxEnabled() && ModeFix6();
	SetKkMinMax(kk_minmax);
	// mslfix5: replay KK NIR float re-association on pixel text (MslNirReassoc.cpp);
	// KYTY_MSL_GFX_REASSOC=0 = old text
	// fmafix: one statement per NIR op (V_MAD/V_FMA as fmul then fadd) like KK's text, so Metal
	// contracts B exactly where it contracts KK; KYTY_MSL_GFX_SPLIT=0 = mslfix5 text (one
	// expression per IR instruction)
	NirReassocBuild(
	    program,
	    gfx_relaxed && ModeFix6() &&
	        [] {
		        const char* e = std::getenv("KYTY_MSL_GFX_REASSOC");
		        return e == nullptr || std::strcmp(e, "0") != 0;
	        }(),
	    gfx_relaxed && ModeFix6() &&
	        [] {
		        const char* e = std::getenv("KYTY_MSL_GFX_SPLIT");
		        return e == nullptr || std::strcmp(e, "0") != 0;
	        }());
	SetFlushTranscendentalInputs(gfx_stage && (options.kk_bda || KkBdaRequested()) && [] {
		const char* e = std::getenv("KYTY_MSL_GFX_FTZ");
		return e == nullptr || std::strcmp(e, "0") != 0;
	}());
	ctx.kk_bda =
	    (program.stage == ShaderType::Compute || gfx_stage) && (options.kk_bda || KkBdaRequested());
	Mem::SetKkSamplerBias(false); // KkBdaLocals turns it on for kk-bda kernels
	Mem::SetLdsDevice(false); // KkBdaLocals: LDS in KK's SharedMemory device buffer (lds_storage)
	Mem::SetLdsThread(0u);    // KkBdaLocals: graphics per-invocation LDS (mslrest4)
	Mem::SetMipStats({});     // KkBdaLocals: GET_LOD_STATS feedback (kk-bda pixel shaders)
	if (program.stage != ShaderType::Compute && !gfx_stage) {
		ctx.unsupported[fmt::format("stage:{}", static_cast<int>(program.stage))]++;
		result.source = PreludeText(fuse_mad) + "// stage not supported by the MSL backend yet\n";
		result.unsupported = ctx.unsupported;
		return result;
	}
	if (gfx_stage) { // graphics: one guest lane per Metal thread (see MslStageIo.cpp)
		ctx.lanes = lanes     = 1;
		result.lanes          = 1;
		total                 = 1;
		result.threadgroup[0] = 1;
	}

	// SSA numbering + declarations.
	uint32_t                     next = 0;
	std::vector<const IR::Inst*> declared;
	for (const auto* block: program.blocks) {
		for (const auto& inst: *block) {
			ctx.ids[&inst] = next++;
			result.instructions++;
			if (!MslTypeName(inst.GetType()).empty()) declared.push_back(&inst);
		}
	}

	auto& b = ctx.b;
	EmitResourcePrelude(ctx);
	std::vector<std::string> params;
	if (!ctx.kk_bda) params.push_back("constant uint* push [[buffer(1)]]");
	if (gfx_stage) {
		ctx.gfx = GfxSetup(ctx);
		for (const auto& gp: GfxParams(*ctx.gfx))
			params.push_back(gp);
		if (program.stage == ShaderType::Pixel)
			params.push_back("uint lid [[thread_index_in_simdgroup]]");
	} else {
		params.push_back(ctx.kk_bda ? "uint3 kk_tgpig [[threadgroup_position_in_grid]]"
		                            : "uint3 tgpig [[threadgroup_position_in_grid]]");
		params.push_back("uint3 nwg [[threadgroups_per_grid]]");
		params.push_back("uint tid [[thread_index_in_threadgroup]]");
		params.push_back("uint lid [[thread_index_in_simdgroup]]");
	}
	EmitResourceParams(ctx, params);
	std::string header =
	    gfx_stage ? GfxEntryPrefix(*ctx.gfx) + "main0("
	              : fmt::format("[[max_total_threads_per_threadgroup({})]]\nkernel void {}(",
	                            std::max(total, 1u), ctx.kk_bda ? "main_entrypoint" : "main0");
	for (size_t i = 0; i < params.size(); i++)
		header += (i ? ",\n    " : "") + params[i];
	header += ")";
	b.Comment(fmt::format("hash {:016x}  wave{}  lanes/thread {}  threads {}x{}x{} -> tg {}",
	                      program.shader_hash, program.wave_size, lanes, shape.threads[0],
	                      shape.threads[1], shape.threads[2], total));
	if (ctx.kk_bda) b.Raw(KkBdaLayoutComment(program));
	b.Open(header);
	if (ctx.loop_cap != 0u)
		b.Line("uint nps5_lg_n = 0u; // nps5 gpuguard loop budget (MslOptions::loop_cap)");
	if (gfx_stage) GfxLocals(ctx, *ctx.gfx);
	EmitResourceLocals(ctx);
	for (const auto* inst: declared) {
		const auto type = MslTypeName(inst->GetType());
		for (uint32_t h = 0; h < lanes; h++)
			b.Line(fmt::format("{} {};", type, ctx.Name(*inst, h)));
	}
	const uint32_t     entry = program.block_info.empty() ? 0u : program.block_info[0].id;
	std::set<uint32_t> need_do;
	bool               structured = !program.dispatcher_fallback && !options.force_dispatcher &&
	                                std::getenv("KYTY_MSL_FORCE_DISPATCHER") == nullptr;
	if (structured) {
		Structurer dry(ctx, shape, options.comments, true, need_do);
		structured = dry.Run();
	}
	result.structured = structured;
	if (structured) {
		Structurer real(ctx, shape, options.comments, false, need_do);
		real.Run();
		if (gfx_stage) b.Line(ctx.ret);
		b.Close(); // kernel
		result.source      = KkInjectHeader(ctx, result) +
		                     PreludeText(fuse_mad, gfx_stage, gfx_relaxed, vs_single_lane, kk_minmax) +
		                     LoopGuardHelper(ctx.loop_cap) + ctx.b.Take();
		result.unsupported = ctx.unsupported;
		result.complete    = ctx.unsupported.empty();
		return result;
	}
	b.Line(fmt::format("uint blk = {}u;", entry));
	{
		std::string lg_decl;
		ctx.loop_private =
		    false; // the dispatcher is one loop: the shared budget counts block transitions
		const std::string lg = ctx.LoopGuard(&lg_decl);
		b.Open("while (true)");
		if (!lg.empty()) b.Line(lg);
	}
	b.Open("switch (blk)");
	for (size_t i = 0; i < program.blocks.size(); i++) {
		b.Open(fmt::format("case {}u:", program.block_info[i].id));
		for (const auto& inst: *program.blocks[i]) {
			if (options.comments && inst.GetOpcode() != ValueOpcode::Phi) {
				b.Comment(std::string(IR::ValueOpcodeName(inst.GetOpcode())));
			}
			EmitInst(ctx, inst, shape);
		}
		EmitTerminator(ctx, i);
		b.Close();
	}
	b.Line("default: " + ctx.ret);
	b.Close(); // switch
	b.Close(); // while
	if (gfx_stage) b.Line(ctx.ret);
	b.Close(); // kernel

	result.source      = KkInjectHeader(ctx, result) +
	                     PreludeText(fuse_mad, gfx_stage, gfx_relaxed, vs_single_lane, kk_minmax) +
	                     LoopGuardHelper(ctx.loop_cap) + ctx.b.Take();
	result.unsupported = ctx.unsupported;
	result.complete    = ctx.unsupported.empty();
	return result;
}

} // namespace Libs::Graphics::ShaderRecompiler::Msl
