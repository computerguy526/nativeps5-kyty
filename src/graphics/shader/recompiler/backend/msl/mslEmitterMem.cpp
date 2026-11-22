// T2 msl-mem: strong definitions of the T2 hooks (mslEmitterInternal.h): buffer / LDS / atomic /
// image / wave-permute instructions of the IR lowered through the text emitters in Msl*Ops.h.
// Resource model: see MslBinding.h (root block `r`, push constants, threadgroup LDS).
#include "graphics/shader/recompiler/BufferFormat.h"
#include "graphics/shader/recompiler/backend/msl/MslBindingFromIr.h"
#include "graphics/shader/recompiler/backend/msl/MslBufferOps.h"
#include "graphics/shader/recompiler/backend/msl/MslImageOps.h"
#include "graphics/shader/recompiler/backend/msl/MslSharedOps.h"
#include "graphics/shader/recompiler/backend/msl/MslWaveOps.h"
#include "graphics/shader/recompiler/backend/msl/MslXform.h"
#include "graphics/shader/recompiler/backend/msl/mslEmitterInternal.h"

#include <unordered_map>

namespace Libs::Graphics::ShaderRecompiler::Msl {

using IR::ValueOpcode;
using namespace Mem;

namespace {

const BindingModel& ModelOf(const MslCtx& ctx) {
	// Built once per emission (MslCtx), never shared between programs: two specializations of one
	// shader have the same shader_hash and, compiled one after the other, the same IR::Program
	// address, but different resource info (buffer strides/formats, image numeric classes).
	if (ctx.binding_model == nullptr) {
		ctx.binding_model =
		    std::make_shared<const BindingModel>(BuildBindingModel(ctx.program, ctx.input_info));
	}
	return *ctx.binding_model;
}

struct Em {
	MslCtx&             ctx;
	const IR::Inst&     inst;
	const BindingModel& model;
	MemOut              o;
	std::string         type;

	Em(MslCtx& c, const IR::Inst& i): ctx(c), inst(i), model(ModelOf(c)) {
		o.prefix = "m" + std::to_string(c.ids.at(&i)) + "h" + std::to_string(c.half) + "_";
		type     = MslTypeName(i.GetType());
	}
	std::string           A(size_t i) const { return ctx.Val(inst.Arg(i)); }
	const IR::MemoryInfo& Mem() const {
		return ctx.program.memory_info.at(inst.Flags<IR::MemoryFlags>().index);
	}
	// Assigns `expr` (of the IR result type) to the result variable.
	void Result(const std::string& expr) {
		o.Line(ctx.Name(inst) + " = " + type + "(" + expr + ");");
	}
	// Flushes to the kernel body; on error records the coverage entry and returns false.
	bool Done() {
		if (!o.error.empty()) {
			ctx.unsupported["T2:" + std::string(IR::ValueOpcodeName(inst.GetOpcode())) + ": " +
			                o.error]++;
			return false;
		}
		ctx.b.Raw(o.code);
		return true;
	}
	bool Fail(const std::string& why) {
		o.Fail(why);
		return Done();
	}
};

AtomicOp AtomicOf(ValueOpcode op, bool& wide) {
	// Names are Buffer/Shared/ImageAtomic<Op>{32,64}.
	std::string n(IR::ValueOpcodeName(op));
	wide           = n.size() > 2 && n.compare(n.size() - 2, 2, "64") == 0;
	const auto  at = n.find("Atomic");
	std::string k  = at == std::string::npos ? n : n.substr(at + 6);
	if (k.size() > 2) k.resize(k.size() - 2);
	static const std::pair<const char*, AtomicOp> table[] = {
	    {"Swap", AtomicOp::Swap}, {"CmpSwap", AtomicOp::CmpSwap}, {"IAdd", AtomicOp::Add},
	    {"ISub", AtomicOp::Sub},  {"SMin", AtomicOp::SMin},       {"UMin", AtomicOp::UMin},
	    {"SMax", AtomicOp::SMax}, {"UMax", AtomicOp::UMax},       {"And", AtomicOp::And},
	    {"Or", AtomicOp::Or},     {"Xor", AtomicOp::Xor},         {"Inc", AtomicOp::Inc},
	    {"Dec", AtomicOp::Dec},   {"FMin", AtomicOp::FMin},       {"FMax", AtomicOp::FMax}};
	for (const auto& [name, v]: table)
		if (k == name) return v;
	return AtomicOp::Add;
}

bool IsBufferAtomic(ValueOpcode op) {
	return IR::BufferAccessOf(op) == IR::BufferAccess::Atomic;
}

const BufferBinding* Buf(const Em& e, uint32_t idx) {
	return idx < e.model.buffers.size() ? &e.model.buffers[idx] : nullptr;
}

// Runs `body` inside `if (active)`; the result variable is zero when inactive.
template <typename Fn>
void Guarded(Em& e, const std::string& active, bool has_result, Fn&& body) {
	if (has_result) e.o.Line(e.ctx.Name(e.inst) + " = " + e.type + "(0);");
	e.o.Line("if (" + active + ") {");
	body();
	e.o.Line("}");
}

// ---- bigshader loadgroup (KYTY_MSL_LOADGROUP): grouped bounds check + packed vector loads
// ---------------------------- Members = raw 32-bit buffer reads (ReadConstBuffer /
// LoadBufferU32[xN], not formatted, linear non-swizzled addressing) in ONE IR block, with the same
// binding, the same address operand Values (index, offset, soffset / base) and the same exec Value,
// whose immediates differ by multiples of 4 inside a 64-byte span, with no side-effecting
// instruction between them. Member k reads dwords at A + imm_k (uint arithmetic, A = the common
// address without the immediate) and guards itself with `ulong(addr_k) + 4*comps_k <= size`. At the
// first member we evaluate
//   F = active && (hi >= lo) && ulong(hi) + 4 <= size,   lo = A + imm_min, hi = A + (imm_end - 4)
// with the very same uint expressions. If F: no uint wrap between lo and hi (if hi >= lo, A + imm
// never wraps inside [imm_min, imm_end-4], because the span < 2^32), so addr_k = lo + (imm_k -
// imm_min) exactly, and addr_k + 4*comps_k <= hi + 4 <= size: every member's own guard is true and
// its load reads exactly W[...]. Loads are pure moves of bits (no float op is touched), so rounding
// cannot change. If !F each member runs its original text verbatim (the else branch), so partial /
// out-of-bounds cases are byte-for-byte the old behaviour. KYTY_MSL_LOADGROUP: unset/0 = off (old
// text), 1 = vertex+pixel, 2 = all stages; KYTY_MSL_LOADGROUP_MUTANT=1 shifts the fast-path loads
// by +4 bytes (validation: must change results).
// The load_group transform or its vertex-only twin vs_load_group (mode token vgroup).
bool LgXf() {
	return Xf::On(Xf::LoadGroup) || Xf::On(Xf::VsLoadGroup);
}
int LoadGroupMode() {
	static const int v = [] {
		const char* e = std::getenv("KYTY_MSL_LOADGROUP");
		return e == nullptr ? 0 : std::atoi(e);
	}();
	return v;
}
bool LoadGroupMutant() {
	static const bool v = [] {
		const char* e = std::getenv("KYTY_MSL_LOADGROUP_MUTANT");
		return e != nullptr && e[0] == '1';
	}();
	return v;
}
struct LgGroup {
	const IR::Inst* first  = nullptr;
	uint32_t        lo_imm = 0, end_imm = 0; // byte immediates: [lo_imm, end_imm)
	uint32_t        id = 0;
	IR::Value       index; // load_group: common index Value (vindex) the immediates are relative to
	std::vector<std::pair<const IR::Inst*, uint32_t>> members; // inst, imm
};
struct LgInfo {
	uint64_t                                      hash = 0;
	bool                                          xf   = false;
	std::vector<LgGroup>                          groups;
	std::unordered_map<const IR::Inst*, uint32_t> at; // member -> group index
};
bool SameVal(const IR::Value& a, const IR::Value& b) {
	return a.Resolve() == b.Resolve();
}
// Candidate member: returns false if the instruction cannot join a group.
bool LgCandidate(const MslCtx& ctx, const BindingModel& model, const IR::Inst& inst, uint32_t& imm,
                 uint32_t& bytes, std::vector<IR::Value>& key, uint32_t& res) {
	const auto op  = inst.GetOpcode();
	const bool rcb = op == ValueOpcode::ReadConstBuffer;
	if (!rcb && IR::BufferAccessOf(op) != IR::BufferAccess::Read) return false;
	const auto& mem = ctx.program.memory_info.at(inst.Flags<IR::MemoryFlags>().index);
	if (mem.kind != IR::ResourceKind::Buffer && mem.kind != IR::ResourceKind::ScalarBuffer)
		return false;
	if (mem.resource >= model.buffers.size()) return false;
	const auto& b = model.buffers[mem.resource];
	res           = mem.resource;
	imm           = mem.offset;
	if (rcb) {
		// constant addresses: AGX already hoists these per-dword selects into the uniform prolog;
		// grouping them (array + branch) pulls them back into the main program (measured: dload 9
		// -> 62). Mode 3 groups them too.
		if (inst.Arg(1).Resolve().IsImmediate() && LoadGroupMode() < 3) return false;
		bytes = 4;
		key   = {IR::Value(1u), inst.Arg(1)};
		return true;
	}
	if (mem.kind != IR::ResourceKind::Buffer || mem.formatted) return false;
	if (inst.GetType() == IR::Type::U8 || inst.GetType() == IR::Type::U16) return false;
	const uint32_t stride = b.packed_stride & 0x3fffu;
	const bool     swz    = stride != 0u && (b.packed_stride & (1u << 14u)) != 0u;
	if (swz || (b.packed_stride & (1u << 20u)) != 0u) return false;
	bytes           = 4u * std::max(IR::BufferComponentCount(op), 1u);
	IR::Value index = mem.idxen ? inst.Arg(1) : IR::Value();
	// load_group@1: an index `c + X` (IAdd32 with an immediate c) is index X with c*stride added to
	// the immediate, so consecutive structured elements (the rows of a bone matrix) join one group.
	// Same uint value: the member computes (c + X) * stride + off + imm, the group X * stride + off
	// + (imm + c * stride), equal mod 2^32, and the no-wrap argument below is about that common
	// value.
	if (mem.idxen && stride != 0u && LgXf()) {
		const IR::Value r = index.Resolve();
		if (!r.IsImmediate()) {
			if (const IR::Inst* ai = r.TryInstruction();
			    ai != nullptr && ai->GetOpcode() == ValueOpcode::IAdd32) {
				for (uint32_t j = 0; j < 2; j++) {
					const IR::Value c = ai->Arg(j).Resolve();
					if (!c.IsImmediate() || c.GetType() != IR::Type::U32) continue;
					const uint64_t add = uint64_t(c.U32()) * stride;
					if (add > 64u || uint64_t(imm) + add > 0xffffff00ull) break;
					index = ai->Arg(1u - j);
					imm += static_cast<uint32_t>(add);
					break;
				}
			}
		}
	}
	key = {IR::Value(2u), index, mem.offen ? inst.Arg(2) : IR::Value(), inst.Arg(3),
	       inst.Arg(inst.NumArgs() - 1)};
	return true;
}
const LgInfo& LgOf(const MslCtx& ctx, const BindingModel& model) {
	// Per emission (MslCtx::lg_info): never shared between two programs, see mslEmitterInternal.h.
	if (ctx.lg_info != nullptr) return *std::static_pointer_cast<const LgInfo>(ctx.lg_info);
	auto gp     = std::make_shared<LgInfo>();
	ctx.lg_info = gp;
	auto& g     = *gp;
	// perfauto transform load_group (MslXform.h): the same grouping for the current program in any
	// stage, dynamic bases only (mode-1 rule), independent of KYTY_MSL_LOADGROUP. The transform set
	// is fixed per emission (Xf::Begin).
	const bool xf       = LgXf();
	g.hash              = ctx.program.shader_hash;
	g.xf                = xf;
	const int  mode     = LoadGroupMode();
	const bool stage_ok = xf || mode >= 2 ||
	                      (mode == 1 && (ctx.program.stage == ShaderType::Vertex ||
	                                     ctx.program.stage == ShaderType::Pixel));
	if (!stage_ok) return g;
	struct Open {
		uint32_t               res;
		std::vector<IR::Value> key;
		LgGroup                grp;
	};
	for (const auto* blk: ctx.program.blocks) {
		std::vector<Open> open;
		auto              close = [&](size_t i) {
			auto& gr = open[i].grp;
			if (gr.members.size() >= 2 && gr.end_imm - gr.lo_imm >= 8u) {
				gr.id = static_cast<uint32_t>(g.groups.size());
				for (auto& m: gr.members)
					g.at[m.first] = gr.id;
				g.groups.push_back(gr);
			}
			open.erase(open.begin() + static_cast<long>(i));
		};
		for (const auto& inst: *blk) {
			uint32_t               imm = 0, bytes = 0, res = 0;
			std::vector<IR::Value> key;
			if (LgCandidate(ctx, model, inst, imm, bytes, key, res)) {
				size_t i = 0;
				for (; i < open.size(); i++) {
					if (open[i].res != res || open[i].key.size() != key.size()) continue;
					bool same = true;
					for (size_t k = 0; k < key.size() && same; k++)
						same = SameVal(open[i].key[k], key[k]);
					if (same) break;
				}
				if (i < open.size()) {
					auto&          gr  = open[i].grp;
					const uint32_t lo  = std::min(gr.lo_imm, imm),
					               end = std::max(gr.end_imm, imm + bytes);
					const bool     fits =
					    imm + bytes > imm && (imm - gr.lo_imm) % 4u == 0u && end - lo <= 64u;
					if (fits) {
						gr.lo_imm  = lo;
						gr.end_imm = end;
						gr.members.emplace_back(&inst, imm);
						continue;
					}
					close(i);
				}
				if (imm + bytes > imm) {
					Open o {res, key, {}};
					o.grp.first   = &inst;
					o.grp.index   = key.size() > 1 ? key[1] : IR::Value();
					o.grp.lo_imm  = imm;
					o.grp.end_imm = imm + bytes;
					o.grp.members.emplace_back(&inst, imm);
					open.push_back(std::move(o));
				}
				continue;
			}
			if (inst.MayHaveSideEffects())
				while (!open.empty())
					close(open.size() - 1);
		}
		while (!open.empty())
			close(open.size() - 1);
	}
	return g;
}

bool BufferOpBody(Em& e, ValueOpcode op) {
	const auto& mem = e.Mem();
	if (mem.kind != IR::ResourceKind::Buffer && mem.kind != IR::ResourceKind::ScalarBuffer)
		return false;
	const auto* b = Buf(e, mem.resource);
	if (b == nullptr) return false;
	const auto access = IR::BufferAccessOf(op);
	const auto comps  = std::max(IR::BufferComponentCount(op), 1u);
	if (op == ValueOpcode::ReadConstBuffer) {
		const std::string off = "(" + e.A(1) + " + " + U(mem.offset) + ")";
		e.Result(ReadConstBuffer(e.o, *b, off));
		return e.Done();
	}
	if (mem.kind != IR::ResourceKind::Buffer) return false;
	if (mem.formatted && mem.data_bits == 16u) return e.Fail("D16 formatted buffer access");
	BufferAddrIn in;
	in.index                 = mem.idxen ? e.A(1) : std::string();
	in.offset                = mem.offen ? e.A(2) : std::string();
	in.soffset               = e.A(3);
	in.imm                   = mem.offset;
	in.lane                  = e.ctx.LaneExpr(e.ctx.half);
	const std::string active = e.A(e.inst.NumArgs() - 1);
	BufferOpts        raw_opts;
	raw_opts.per_dword_bounds = e.ctx.program.stage != ShaderType::Compute; // v1.1.1: graphics only

	if (access == IR::BufferAccess::Read) {
		const uint32_t bits = e.inst.GetType() == IR::Type::U8    ? 8u
		                      : e.inst.GetType() == IR::Type::U16 ? 16u
		                                                          : 32u;
		Guarded(e, active, true, [&] {
			const auto addr = BufferByteAddress(e.o, *b, in);
			if (mem.formatted) {
				const auto fmt =
				    mem.typed ? Format::DecodeTBufferFormat(mem.data_format, mem.number_format)
				              : static_cast<Prospero::BufferFormat>(b->format);
				const auto info = Format::GetFormatInfo(fmt);
				if (info.component_count == 0u) {
					const auto v = LoadRaw(e.o, *b, addr, 32, comps);
					e.o.Line("    " + e.ctx.Name(e.inst) + " = " + e.type + "(" + v + ");");
					return;
				}
				// typed: memory component k else 0; untyped: the V# dst_sel
				uint32_t sel = 4u | (5u << 3) | (6u << 6) | (7u << 9);
				if (mem.typed) {
					sel = 0;
					for (uint32_t k = 0; k < info.component_count && k < 4; k++)
						sel |= (4u + k) << (3u * k);
				} else {
					const auto swz = e.ctx.program.info.buffers[mem.resource].descriptor_swizzle;
					sel            = swz;
				}
				const auto  v = LoadFormatted(e.o, *b, addr, static_cast<uint32_t>(fmt), sel);
				std::string x = comps == 1   ? v + ".x"
				                : comps == 2 ? v + ".xy"
				                : comps == 3 ? v + ".xyz"
				                             : v;
				e.o.Line("    " + e.ctx.Name(e.inst) + " = " + e.type + "(" + x + ");");
				return;
			}
			const auto v = LoadRaw(e.o, *b, addr, bits, bits == 32 ? comps : 1u, raw_opts);
			e.o.Line("    " + e.ctx.Name(e.inst) + " = " + e.type + "(" + v + ");");
		});
		return e.Done();
	}
	if (access == IR::BufferAccess::Write) {
		const std::string data = e.A(e.inst.NumArgs() - 2);
		const auto        dty  = e.inst.Arg(e.inst.NumArgs() - 2).GetType();
		const uint32_t    bits = dty == IR::Type::U8 ? 8u : dty == IR::Type::U16 ? 16u : 32u;
		Guarded(e, active, false, [&] {
			const auto               addr = BufferByteAddress(e.o, *b, in);
			std::vector<std::string> vals;
			static const char*       sw[4] = {"x", "y", "z", "w"};
			for (uint32_t k = 0; k < comps; k++)
				vals.push_back(comps == 1 ? data : "(" + data + ")." + sw[k]);
			if (mem.formatted) {
				const auto fmt =
				    mem.typed ? Format::DecodeTBufferFormat(mem.data_format, mem.number_format)
				              : static_cast<Prospero::BufferFormat>(b->format);
				if (Format::GetFormatInfo(fmt).component_count != 0u) {
					StoreFormatted(e.o, *b, addr, static_cast<uint32_t>(fmt), vals);
					return;
				}
			}
			StoreRaw(e.o, *b, addr, bits, vals, raw_opts);
		});
		return e.Done();
	}
	if (access == IR::BufferAccess::Atomic) {
		bool              wide       = false;
		const auto        aop        = AtomicOf(op, wide);
		const bool        cas        = aop == AtomicOp::CmpSwap;
		const size_t      n          = e.inst.NumArgs();
		const std::string value      = cas ? e.A(n - 3) : e.A(n - 2);
		const std::string cmp        = cas ? e.A(n - 2) : (wide ? "0ul" : "0u");
		const bool        has_result = e.inst.GetType() != IR::Type::Void;
		Guarded(e, active, has_result, [&] {
			const auto addr = BufferByteAddress(e.o, *b, in);
			const auto old  = wide ? BufferAtomic64(e.o, *b, addr, aop, value, cmp)
			                       : BufferAtomic32(e.o, *b, addr, aop, value, cmp);
			if (has_result) e.o.Line("    " + e.ctx.Name(e.inst) + " = " + old + ";");
		});
		return e.Done();
	}
	return false;
}

// Wraps BufferOpBody with the loadgroup fast path (no-op unless KYTY_MSL_LOADGROUP selects this
// stage).
bool BufferOp(Em& e, ValueOpcode op) {
	const auto& lg = LgOf(e.ctx, e.model);
	const auto  it = lg.at.find(&e.inst);
	if (it == lg.at.end()) return BufferOpBody(e, op);
	const auto&       g   = lg.groups[it->second];
	const auto&       mem = e.Mem();
	const auto&       b   = e.model.buffers[mem.resource];
	const std::string pre = "lg" + std::to_string(g.id) + "h" + std::to_string(e.ctx.half);
	const std::string W = pre + "_w", F = pre + "_f";
	const uint32_t    ndw = (g.end_imm - g.lo_imm) / 4u;
	if (&e.inst == g.first) {
		std::string lo, hi, active;
		if (op == ValueOpcode::ReadConstBuffer) {
			lo = e.o.Def("uint", "(" + e.A(1) + " + " + U(g.lo_imm) + ")");
			hi = e.o.Def("uint", "(" + e.A(1) + " + " + U(g.end_imm - 4u) + ")");
		} else {
			BufferAddrIn in;
			in.index   = mem.idxen ? (LgXf() && !g.index.IsEmpty() ? e.ctx.Val(g.index) : e.A(1))
			                       : std::string();
			in.offset  = mem.offen ? e.A(2) : std::string();
			in.soffset = e.A(3);
			in.lane    = e.ctx.LaneExpr(e.ctx.half);
			in.imm     = g.lo_imm;
			lo         = BufferByteAddress(e.o, b, in);
			in.imm     = g.end_imm - 4u;
			hi         = BufferByteAddress(e.o, b, in);
			active     = e.A(e.inst.NumArgs() - 1);
		}
		GroupLoadWords(e.o, b, lo, hi, active, ndw, W, F, LoadGroupMutant());
	}
	uint32_t imm = 0;
	for (const auto& m: g.members)
		if (m.first == &e.inst) imm = m.second;
	const uint32_t k = (imm - g.lo_imm) / 4u;
	const uint32_t comps =
	    op == ValueOpcode::ReadConstBuffer ? 1u : std::max(IR::BufferComponentCount(op), 1u);
	std::string v;
	if (comps == 1)
		v = W + "[" + std::to_string(k) + "]";
	else {
		v = "uint" + std::to_string(comps) + "(";
		for (uint32_t q = 0; q < comps; q++)
			v += (q ? ", " : "") + W + "[" + std::to_string(k + q) + "]";
		v += ")";
	}
	e.o.Line("if (" + F + ") {");
	e.o.Line("    " + e.ctx.Name(e.inst) + " = " + e.type + "(" + v + ");");
	e.o.Line("} else {");
	if (!BufferOpBody(e, op)) return false;
	e.ctx.b.Raw("    }\n");
	return true;
}

bool SharedOp(Em& e, ValueOpcode op) {
	const auto& mem = e.Mem();
	if (mem.kind != IR::ResourceKind::Lds) return false;
	const LdsInfo     l      = MakeLdsInfo(e.model.lds_bytes);
	const std::string active = e.A(e.inst.NumArgs() - 1);
	const auto        base   = [&](bool truncate) {
		std::string a = mem.offset ? "(" + e.A(0) + " + " + U(mem.offset) + ")" : e.A(0);
		return truncate ? "((" + a + ") & 0xffffu)" : a;
	};
	const auto access = IR::SharedAccessOf(op);
	const auto comps  = std::max(IR::SharedComponentCount(op), 1u);
	if (access == IR::SharedAccess::Read) {
		const uint32_t bits = e.inst.GetType() == IR::Type::U8    ? 8u
		                      : e.inst.GetType() == IR::Type::U16 ? 16u
		                                                          : 32u;
		Guarded(e, active, true, [&] {
			const auto v = LdsLoad(e.o, l, base(false), bits, bits == 32 ? comps : 1u);
			e.o.Line("    " + e.ctx.Name(e.inst) + " = " + e.type + "(" + v + ");");
		});
		return e.Done();
	}
	if (access == IR::SharedAccess::Write) {
		const auto               lt   = e.inst.Arg(1).GetType();
		const uint32_t           bits = lt == IR::Type::U8 ? 8u : lt == IR::Type::U16 ? 16u : 32u;
		std::vector<std::string> vals;
		for (uint32_t k = 0; k < (bits == 32 ? comps : 1u); k++)
			vals.push_back(e.A(1 + k));
		Guarded(e, active, false, [&] { LdsStore(e.o, l, base(false), bits, vals); });
		return e.Done();
	}
	if (access == IR::SharedAccess::Atomic) {
		if (op == ValueOpcode::SharedAtomicMaskedOr32) {
			Guarded(e, active, false, [&] { LdsMaskedOr32(e.o, l, base(true), e.A(1), e.A(2)); });
			return e.Done();
		}
		bool       wide = false;
		const auto aop  = AtomicOf(op, wide);
		if (op == ValueOpcode::SharedAtomicFMin32 || op == ValueOpcode::SharedAtomicFMax32) {
			Guarded(e, active, false, [&] {
				LdsAtomic32(e.o, l, base(true),
				            aop == AtomicOp::FMin || op == ValueOpcode::SharedAtomicFMin32
				                ? AtomicOp::FMin
				                : AtomicOp::FMax,
				            e.A(1));
			});
			return e.Done();
		}
		const bool   has_result = e.inst.GetType() != IR::Type::Void;
		const size_t n          = e.inst.NumArgs();
		Guarded(e, active, has_result, [&] {
			std::string old;
			if (wide) {
				old = LdsAtomic64(e.o, l, base(true), aop, e.A(n - 2));
			} else if (aop == AtomicOp::CmpSwap) {
				// DS compares DATA (n-3) and stores DATA2 (n-2): opposite of BUFFER atomics.
				old = LdsAtomic32(e.o, l, base(true), aop, e.A(n - 2), e.A(n - 3));
			} else {
				old = LdsAtomic32(e.o, l, base(true), aop, e.A(n - 2));
			}
			if (has_result)
				e.o.Line("    " + e.ctx.Name(e.inst) + " = " + e.type + "(" + old + ");");
		});
		return e.Done();
	}
	return false;
}

bool ImageOpEmit(Em& e, ValueOpcode op, const IR::ImageOpcodeInfo& info) {
	const auto& mem = e.Mem();
	if (mem.resource >= e.model.images.size()) return false;
	const auto& ir_img = e.ctx.program.info.images[mem.resource];
	if (ir_img.indirect_root == mem.resource ||
	    ir_img.indirect_root != IR::ImageResource::NoIndirectImage)
		return e.Fail("indirect image descriptor");
	ImageOp iop;
	iop.img = &e.model.images[mem.resource];
	if (info.needs_sampler) {
		if (mem.sampler >= e.model.samplers.size()) return false;
		iop.samp = &e.model.samplers[mem.sampler];
	}
	iop.sample_flags       = mem.image_sample_flags;
	iop.dmask              = mem.dmask ? mem.dmask : 0xf;
	iop.data_dwords        = mem.data_dwords;
	iop.data_bits          = mem.data_bits;
	iop.address_components = mem.image_address_components;
	iop.has_mip            = mem.image_has_mip;
	iop.dynamic_mip        = ir_img.mip_mode == IR::ImageMipMode::Dynamic;
	iop.pixel_stage        = e.ctx.program.stage == ShaderType::Pixel;
	iop.graphics_stage     = e.ctx.program.stage != ShaderType::Compute;
	iop.converted          = ir_img.conversion_format != Prospero::BufferFormat::kInvalid;
	const size_t addr_arg  = info.needs_sampler ? 2 : 1;
	const auto*  addr_inst = e.inst.Arg(addr_arg).ResolveInstruction();
	if (addr_inst == nullptr || addr_inst->GetOpcode() != ValueOpcode::MakeImageAddress)
		return e.Fail("image address not from MakeImageAddress");
	std::vector<std::string> addr;
	for (size_t i = 0; i < addr_inst->NumArgs(); i++)
		addr.push_back(e.ctx.Val(addr_inst->Arg(i)));
	if ((iop.sample_flags & SampleFlag::Offset) != 0u &&
	    !addr_inst->Arg(0).IsImmediate() == false) {
		iop.const_offset = addr_inst->Arg(0).U32();
	}
	const std::string active = e.A(e.inst.NumArgs() - 1);
	switch (op) {
		case ValueOpcode::ImageQueryDimensions: e.Result(ImageQueryDims(e.o, iop, addr)); break;
		case ValueOpcode::ImageQueryLod: e.Result(ImageQueryLod(e.o, iop, addr)); break;
		case ValueOpcode::ImageRead: e.Result(ImageLoad(e.o, iop, addr, active)); break;
		case ValueOpcode::ImageWrite: ImageStore(e.o, iop, addr, e.A(2), active); break;
		case ValueOpcode::ImageSampleRaw: e.Result(ImageSample(e.o, iop, addr, false)); break;
		case ValueOpcode::ImageGatherRaw: e.Result(ImageSample(e.o, iop, addr, true)); break;
		default: {
			bool       wide = false;
			const auto aop  = AtomicOf(op, wide);
			if (wide) return e.Fail("64-bit image atomics");
			e.Result(ImageAtomic32(e.o, iop, addr, static_cast<int>(aop), e.A(2), active));
		}
	}
	return e.Done();
}

// DS_APPEND / DS_CONSUME on an LDS counter: one wave-wide atomic by the first active lane, the old
// value is returned to every lane (both guest halves share it).
bool AppendConsume(Em& e, ValueOpcode op) {
	const auto& mem = e.Mem();
	const bool  gds = mem.kind == IR::ResourceKind::Gds;
	if (mem.kind != IR::ResourceKind::Lds && !(gds && e.model.gds_bytes != 0u)) return false;
	if (e.ctx.half == 1u) {
		e.o.Line(e.ctx.Name(e.inst, 1) + " = " + e.ctx.Name(e.inst, 0) + ";");
		return e.Done();
	}
	const LdsInfo     l  = MakeLdsInfo(e.model.lds_bytes);
	const std::string m0 = e.A(0);
	const std::string e0 = e.o.Def("uint", "kyty_ballot(" + e.ctx.Val(e.inst.Arg(1), 0) + ", lid)");
	const std::string e1 =
	    e.ctx.lanes == 2 ? e.o.Def("uint", "kyty_ballot(" + e.ctx.Val(e.inst.Arg(1), 1) + ", lid)")
	                     : std::string("0u");
	const std::string cnt   = e.o.Def("uint", "popcount(" + e0 + ") + popcount(" + e1 + ")");
	const std::string first = e.o.Def("uint", "ctz(" + e0 + " | " + e1 + ")");
	const std::string addr  = e.o.Def(
	    "uint",
	    "((" + m0 + " >> 16u) & ~3u) + " +
	        U(mem.offset)); // DS_APPEND/CONSUME: dword address (SPIR-V/KK align M0[31:16] down)
	const std::string old = e.o.Tmp();
	e.o.Line("uint " + old + " = 0u;");
	const std::string size_ok =
	    gds ? "(" + m0 + " & 0xffffu) != 0u" : U(mem.offset + 3u) + " < (" + m0 + " & 0xffffu)";
	e.o.Line("if (lid == " + first + " && " + cnt + " != 0u && " + size_ok + ") {");
	const auto aop = op == ValueOpcode::DataAppend ? AtomicOp::Add : AtomicOp::Sub;
	if (gds) {
		// device-scope counter in the root block's gds buffer
		e.o.Line("    if (ulong(" + addr + ") + 4ul <= " + std::to_string(e.model.gds_bytes) +
		         "ul) {");
		const auto p =
		    e.o.Def("device atomic_uint*", "(device atomic_uint*)(r.gds + " + addr + ")");
		const auto v = AtomicRmw32(e.o, p, "device", aop, cnt, "0u", "        ");
		e.o.Line("        " + old + " = " + v + ";");
		e.o.Line("    }");
	} else {
		const auto v = LdsAtomic32(e.o, l, addr, aop, cnt);
		e.o.Line("    " + old + " = " + v + ";");
	}
	e.o.Line("}");
	e.Result("simd_shuffle(" + old + ", ushort(" + first + "))");
	return e.Done();
}

// Physical-address memory through a 64-bit guest address (GetAddressResource base + offset).
bool AddressOp(Em& e, ValueOpcode op) {
	const auto& mem = e.Mem();
	if (mem.kind == IR::ResourceKind::Scratch || mem.kind == IR::ResourceKind::IndirectBuffer)
		return false;
	const bool is_load = op == ValueOpcode::LoadAddressU8 || op == ValueOpcode::LoadAddressU16 ||
	                     op == ValueOpcode::LoadAddressU32;
	if (is_load && mem.planning_only) {
		e.Result("0u");
		return e.Done();
	}
	if (e.ctx.kk_bda && !(is_load && KkBdaPagetable()))
		return e.Fail("kk-bda: guest address access needs the BDA page table");
	const bool  scalar = mem.kind == IR::ResourceKind::ScalarAddress;
	std::string low    = e.A(1);
	if (scalar) low = "(" + low + " & ~3u)";
	std::string addr;
	if (mem.address_is_full) {
		addr = "(ulong(" + low + ") | (ulong(" + e.A(2) + ") << 32))";
	} else {
		const auto* h = e.inst.Arg(0).ResolveInstruction();
		if (h == nullptr || h->GetOpcode() != ValueOpcode::GetAddressResource || h->NumArgs() != 2)
			return e.Fail("no address base pair");
		std::string bl = e.ctx.Val(h->Arg(0));
		if (scalar) bl = "(" + bl + " & ~3u)";
		addr = "((ulong(" + bl + ") | (ulong(" + e.ctx.Val(h->Arg(1)) + ") << 32)) + ulong(" + low +
		       "))";
	}
	int32_t imm = static_cast<int32_t>(mem.offset);
	if (scalar) imm = static_cast<int32_t>(static_cast<uint32_t>(imm) & ~3u);
	if (imm != 0) addr = "(" + addr + " + ulong(long(" + std::to_string(imm) + ")))";
	const std::string active  = e.A(e.inst.NumArgs() - 1);
	const auto        bits_of = [](IR::Type t) {
		return t == IR::Type::U8 ? 8u : t == IR::Type::U16 ? 16u : 32u;
	};
	if (is_load && e.ctx.kk_bda) {
		// meshpath: Kyty's SPIR-V LoadBda / LoadBdaDword / GetBdaPointer: page = a >> 14 (< 2^26, else
		// unmapped), entry = page table[page] (0 past the table), 0 -> fault bit (page) in the
		// FaultBuffer bitmap and the dword reads 0; a load that crosses a dword reads the next one too.
		const uint32_t bits = scalar ? 32u : bits_of(e.inst.GetType());
		const char*    ld   = mem.coherent ? "coherent device const uint*" : "device const uint*";
		const auto     dw   = [&](const std::string& a, const std::string& out) {
            e.o.Line("    { const ulong pg = (" + a + ") >> 14ul; const bool in = pg < 67108864ul; const uint p = in ? uint(pg) : 0u;");
            e.o.Line("      const ulong en = (in && (p << 3u) + 7u < kk_bda_ptn) ? *(device const ulong*)(kk_bda_pt + ulong(p << 3u)) : 0ul;");
            e.o.Line("      if (en == 0ul) { if (in && ((p >> 5u) << 2u) + 3u < kk_bda_fbn) atomic_fetch_or_explicit((device atomic_uint*)(kk_bda_fb + ulong((p >> 5u) << 2u)), 1u << (p & 31u), memory_order_relaxed); }");
            e.o.Line("      else " + out + " = *(" + std::string(ld) + ")(en + ((" + a + ") & 16383ul)); }");
		};
		Guarded(e, active, true, [&] {
			e.o.Line("    const ulong ba = " + addr + "; const ulong al = ba & ~3ul; const uint by = uint(ba) & 3u; uint w0 = 0u, w1 = 0u;");
			dw("al", "w0");
			if (bits != 8u) {
				e.o.Line(std::string("    if (by ") + (bits == 16u ? "> 2u" : "!= 0u") + ")");
				dw("al + 4ul", "w1");
			}
			std::string v = "((w0 >> (by << 3u)) | (w1 << (((4u - by) & 3u) << 3u)))";
			if (bits != 32u) v = "(" + v + (bits == 8u ? " & 0xffu)" : " & 0xffffu)");
			e.o.Line("    " + e.ctx.Name(e.inst) + " = " + e.type + "(" + v + ");");
		});
	} else if (is_load) {
		const uint32_t bits = bits_of(e.inst.GetType());
		Guarded(e, active, true, [&] {
			const auto v = AddressLoad(e.o, addr, bits);
			e.o.Line("    " + e.ctx.Name(e.inst) + " = " + e.type + "(" + v + ");");
		});
	} else {
		const size_t   n    = e.inst.NumArgs();
		const uint32_t bits = bits_of(e.inst.Arg(n - 2).GetType());
		Guarded(e, active, false, [&] { AddressStore(e.o, addr, bits, e.A(n - 2)); });
	}
	return e.Done();
}

bool WaveOp(Em& e, ValueOpcode op) {
	WaveCtx w;
	w.lane = "lid";
	switch (op) {
		case ValueOpcode::WriteLane:
			e.Result(WaveWriteLane(e.o, e.ctx.LaneExpr(e.ctx.half), e.A(1), e.A(0), e.A(2)));
			break;
		case ValueOpcode::DppMoveU32:
		case ValueOpcode::DppUpdateU32: {
			const auto f = e.inst.Flags<IR::DppMoveFlags>();
			DppFlags   df;
			df.control        = f.control;
			df.row_mask       = f.row_mask;
			df.bank_mask      = f.bank_mask;
			df.bound_control  = f.bound_control;
			df.fetch_inactive = f.fetch_inactive;
			df.dpp8           = f.dpp8;
			if (op == ValueOpcode::DppMoveU32)
				e.Result(WaveDppMove(e.o, w, df, e.A(0), e.A(1)));
			else
				e.Result(WaveDppUpdate(e.o, w, df, e.A(0), e.A(1), e.A(2)));
			break;
		}
		case ValueOpcode::SwizzleU32:
			if (!e.inst.Arg(1).IsImmediate())
				return e.Fail("ds_swizzle with a non-immediate control");
			e.Result(WaveSwizzle(e.o, w, e.inst.Arg(1).U32(), e.A(0), e.A(2)));
			break;
		case ValueOpcode::Permlane16U32: {
			// V_PERMLANE16_B32 / PERMLANEX16: gather within a 16-lane row (x16: the other row of
			// the 32-lane half), selects packed 4 bits/lane in two scalars.
			const auto f = e.inst.Flags<IR::PermlaneFlags>();
			const auto rv =
			    f.x16 ? std::string("((lid & ~15u) ^ 16u)") : std::string("(lid & ~15u)");
			const auto sel = "(((lid & 15u) >= 8u) ? " + e.A(2) + " : " + e.A(1) + ")";
			const auto tgt =
			    e.o.Def("uint", "(" + rv + " | ((" + sel + " >> ((lid & 7u) * 4u)) & 15u))");
			const auto val = e.o.Def("uint", "simd_shuffle(" + e.A(0) + ", ushort(" + tgt + "))");
			if (f.fetch_inactive)
				e.Result(val);
			else
				e.Result("(simd_shuffle(uint(" + e.A(3) + "), ushort(" + tgt + ")) != 0u ? " + val +
				         " : 0u)");
			break;
		}
		case ValueOpcode::BpermuteU32:
			e.Result(WaveBpermute(e.o, w, e.A(0), e.A(1), e.A(2)));
			break;
		case ValueOpcode::PermuteU32: e.Result(WavePermute(e.o, w, e.A(0), e.A(1), e.A(2))); break;
		default: return false;
	}
	return e.Done();
}

} // namespace

bool EmitMemoryInst(MslCtx& ctx, const IR::Inst& inst) {
	const auto op = inst.GetOpcode();
	// Resource handles carry only a descriptor index (their flags); they have no runtime value.
	switch (op) {
		case ValueOpcode::GetBufferResource:
		case ValueOpcode::GetImageResource:
		case ValueOpcode::GetSamplerResource:
		case ValueOpcode::MakeImageAddress:
		case ValueOpcode::GetSrtResource:
		case ValueOpcode::GetAddressResource:
		case ValueOpcode::GetScratchResource: return true;
		default: break;
	}
	Em e(ctx, inst);
	switch (op) {
		case ValueOpcode::LoadAddressU8:
		case ValueOpcode::LoadAddressU16:
		case ValueOpcode::LoadAddressU32:
		case ValueOpcode::StoreAddressU8:
		case ValueOpcode::StoreAddressU16:
		case ValueOpcode::StoreAddressU32: return AddressOp(e, op);
		case ValueOpcode::DataAppend:
		case ValueOpcode::DataConsume: return AppendConsume(e, op);
		case ValueOpcode::ReadConst:
			if (e.model.srt_bytes == 0u) return false;
			e.Result(SrtLoad(e.A(1)));
			return e.Done();
		case ValueOpcode::WriteLane:
		case ValueOpcode::DppMoveU32:
		case ValueOpcode::DppUpdateU32:
		case ValueOpcode::SwizzleU32:
		case ValueOpcode::BpermuteU32:
		case ValueOpcode::PermuteU32:
		case ValueOpcode::Permlane16U32: return WaveOp(e, op);
		default: break;
	}
	if (IR::SharedAccessOf(op) != IR::SharedAccess::None &&
	    IR::SharedAccessOf(op) != IR::SharedAccess::Append &&
	    IR::SharedAccessOf(op) != IR::SharedAccess::Consume)
		return SharedOp(e, op);
	if (IR::BufferAccessOf(op) != IR::BufferAccess::None || op == ValueOpcode::ReadConstBuffer)
		return BufferOp(e, op);
	const auto info = IR::ImageOpcodeInfoOf(op);
	if (info.access != IR::ImageAccess::None || op == ValueOpcode::ImageQueryDimensions ||
	    op == ValueOpcode::ImageQueryLod || op == ValueOpcode::ImageSampleRaw ||
	    op == ValueOpcode::ImageGatherRaw)
		return ImageOpEmit(e, op, info);
	return false;
}

void EmitResourceParams(MslCtx& ctx, std::vector<std::string>& params) {
	const auto& m = ModelOf(ctx);
	if (ctx.kk_bda) {
		KkBdaParams(ctx, m, params);
		return;
	}
	params.push_back("constant Resources& r [[buffer(" + std::to_string(RootBufferIndex) + ")]]");
	if (m.lds_bytes != 0u) params.push_back("threadgroup uint* lds [[threadgroup(0)]]");
}

void EmitResourcePrelude(MslCtx& ctx) {
	const auto& m = ModelOf(ctx);
	ctx.b.Raw(BufferHelpersPrelude());
	ctx.b.Raw(AddressHelpersPrelude());
	if (ctx.kk_bda) ctx.b.Raw(KkBdaPrelude());
	ctx.b.Raw(ArgumentBufferStruct(m));
}

void EmitResourceLocals(MslCtx& ctx) {
	if (ctx.kk_bda) KkBdaLocals(ctx, ModelOf(ctx));
}

} // namespace Libs::Graphics::ShaderRecompiler::Msl
