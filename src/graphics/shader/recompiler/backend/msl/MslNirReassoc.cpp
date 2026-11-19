// mslfix5: KK's float re-association, replayed on the MSL text of graphics pixel shaders.
//
// In game, Kyty's SPIR-V float ALU of a pixel shader is not NoContraction (FAST_MATH), so every
// fadd/fmul KK's NIR sees is inexact and nir_opt_algebraic is free to re-associate it (V_MAD
// reaches NIR as ffma_weak(contract) and is split into fadd(fmul(a, b), c)). KK prints the result
// per op under math_mode(relaxed). Our B text kept the ISA order, so f32 results rounded
// differently (gfxfuzz 7p: pipes 100/101 rt1, G-buffer RGBA16F words +-1 half ulp on ~14 pipes).
// This replays the subset of nir_opt_algebraic rules that change f32 rounding of add/mul chains, in
// the rule-list order (first match wins), operands before users, to a fixed point:
//   ~fadd(fmul(a, b), fmul(a, c))                              -> fmul(a, fadd(b, c))
//   fmul(fneg(a), b)                                           -> fneg(fmul(a, b))
//   ~fmul(fmul(is_used_once)(a!const, b!const), #c)            -> fmul(fmul(a, c), b)
//   ~fadd(fadd(is_used_once)(a!const, b(is_fmul)), #c)         -> fadd(fadd(a, c), b)
//   ~fadd(fadd(is_used_once)(a!const, b!const), #c)            -> fadd(fadd(a, c), b)
//   ~fmul(#a, fmul(b, #c))                                     -> fmul(a*c, b)
//   ~fadd(#a, fadd(b, #c))                                     -> fadd(a+c, b)
//   ~fadd(fadd(is_used_once)(a(is_fmul), b(is_fmul)), c(is_not_fmul)) -> fadd(fadd(a, c), b)
// Add and mul are commutative bit for bit, so only the association is replayed; untouched
// instructions keep their text. Pixel shaders in KK's float model only (GfxPixelRelaxed + kk-bda);
// KYTY_MSL_GFX_REASSOC=0 = old text.
#include "graphics/shader/recompiler/backend/msl/MslEmitter.h"
#include "graphics/shader/recompiler/backend/msl/mslEmitterInternal.h"

#include <cstring>
#include <deque>
#include <fmt/format.h>
#include <memory>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::Msl {

namespace {

using O = IR::ValueOpcode;

struct Node {
	enum Kind { Leaf, Const, Add, Mul, Neg, Flrp } kind = Leaf;
	float           c                             = 0.0f; // Const
	IR::Value       leaf;           // Leaf (an IR value, emitted with ctx.Val)
	Node*           a    = nullptr; // Add/Mul/Neg operands
	Node*           b    = nullptr;
	Node*           t    = nullptr; // Flrp: a + (b - a) * t
	const IR::Inst* inst = nullptr; // the IR instruction this node is the value of (null: virtual)
	int             uses = 0;
	bool            changed = false; // inst node whose expression was rewritten
};

struct State {
	std::vector<std::unique_ptr<Node>>         pool;
	std::unordered_map<const IR::Inst*, Node*> by_inst;
	std::vector<Node*>                         order; // inst nodes, program order
	std::unordered_map<const IR::Inst*, std::string>
	     text; // per rewritten inst (filled lazily at emit time)
	bool on    = false;
	bool split = false; // fmafix: one statement per NIR op
};

thread_local State g;

Node* New() {
	g.pool.push_back(std::make_unique<Node>());
	return g.pool.back().get();
}

const IR::Inst* SkipCast(IR::Value v) {
	if (v.IsEmpty() || v.IsImmediate()) return nullptr;
	const IR::Inst* i = v.TryInstruction();
	while (i != nullptr &&
	       (i->GetOpcode() == O::BitCastF32U32 || i->GetOpcode() == O::BitCastU32F32)) {
		const auto x = i->Arg(0);
		if (x.IsEmpty() || x.IsImmediate()) return i;
		i = x.TryInstruction();
	}
	return i;
}

int NirUses(const IR::Inst* i) {
	int n = 0;
	for (const auto& u: i->Uses()) {
		if (u.user == nullptr) continue;
		const auto o = u.user->GetOpcode();
		if (o == O::BitCastF32U32 || o == O::BitCastU32F32)
			n += NirUses(u.user);
		else
			n++;
	}
	return n;
}

uint32_t FloatBitsOf(float v) {
	uint32_t u;
	std::memcpy(&u, &v, 4);
	return u;
}

bool IsFloatArith(O o) {
	// fmafix: V_FMA_F32 reaches KK's NIR as ffma_weak(contract) too (Kyty's SPIR-V GLSL Fma carries
	// no NoContraction off the position slice), so it is split into fadd(fmul) and re-associated
	// exactly like V_MAD_F32.
	return o == O::FPAdd32 || o == O::FPSub32 || o == O::FPMul32 || o == O::FPMad32 ||
	       o == O::FPNeg32 || (g.split && o == O::FPFma32);
}

Node* Operand(IR::Value v) {
	if (v.IsImmediate() && (v.GetType() == IR::Type::F32)) {
		auto* n = New();
		n->kind = Node::Const;
		n->c    = v.F32Value();
		n->uses = 1;
		return n;
	}
	if (const auto* i = SkipCast(v); i != nullptr) {
		if (auto f = g.by_inst.find(i); f != g.by_inst.end()) return f->second;
	}
	auto* n = New();
	n->kind = Node::Leaf;
	n->leaf = v;
	n->uses = 1;
	return n;
}

Node* Bin(Node::Kind k, Node* a, Node* b) {
	auto* n = New();
	n->kind = k;
	n->a    = a;
	n->b    = b;
	n->uses = 1;
	a->uses++;
	if (b != nullptr) b->uses++;
	return n;
}

void Release(Node* n) {
	if (n == nullptr || --n->uses > 0) return;
	// dead in NIR terms (an IR instruction still prints its now unused text): its operands lose a
	// use
	Release(n->a);
	Release(n->b);
	Release(n->t);
}

bool IsConst(const Node* n) {
	return n->kind == Node::Const;
}
bool IsFmul(const Node* n) {
	return n->kind == Node::Mul || (n->kind == Node::Neg && n->a->kind == Node::Mul);
}

// Replace n's expression with (k, a, b); a/b already carry the use added by the caller via Bin/++.
void Rewrite(Node* n, Node::Kind k, Node* a, Node* b) {
	Node* oa = n->a;
	Node* ob = n->b;
	Node* ot = n->t;
	n->kind  = k;
	n->a     = a;
	n->b     = b;
	n->t     = nullptr;
	Release(ot);
	if (n->inst != nullptr) n->changed = true;
	Release(oa);
	Release(ob);
}

bool Same(const Node* x, const Node* y) {
	if (x == y) return true;
	if (x->kind == Node::Const && y->kind == Node::Const)
		return std::memcmp(&x->c, &y->c, sizeof(float)) == 0;
	if (x->kind == Node::Leaf && y->kind == Node::Leaf) {
		if (x->leaf == y->leaf) return true;
		// heavyshade: NIR has no u32<->f32 bitcasts, so two BitCast instructions of one IR value are
		// ONE NIR SSA def. Kyty's IR keeps a separate BitCastF32U32 per use; without this the
		// factoring rule a*b + a*c -> a*(b+c) missed every operand read through two casts
		// (isatest nir_rule_assoc lane 0/1: 2614 words 1 ulp A != B).
		const IR::Inst* rx = SkipCast(x->leaf);
		const IR::Inst* ry = SkipCast(y->leaf);
		if (rx == nullptr || ry == nullptr) return false;
		const auto o = rx->GetOpcode();
		if (o == O::BitCastF32U32 || o == O::BitCastU32F32) return false; // cast of an immediate
		return rx == ry;
	}
	return false;
}

bool TryRules(Node* n) {
	if (n->kind == Node::Add) {
		// ~fadd(fmul(a, b), fmul(a, c)) -> fmul(a, fadd(b, c))
		if (n->a->kind == Node::Mul && n->b->kind == Node::Mul) {
			Node* x = n->a;
			Node* y = n->b;
			for (int i = 0; i < 2; i++)
				for (int j = 0; j < 2; j++) {
					Node* xa = i ? x->b : x->a;
					Node* xb = i ? x->a : x->b;
					Node* ya = j ? y->b : y->a;
					Node* yb = j ? y->a : y->b;
					if (!Same(xa, ya)) continue;
					xa->uses++;
					Rewrite(n, Node::Mul, xa, Bin(Node::Add, xb, yb));
					return true;
				}
		}
	}
	if (n->kind == Node::Mul) {
		// fmul(fneg(a), b) -> fneg(fmul(a, b)) (exact; it hides the product from the fmul patterns,
		// as in KK)
		for (int k = 0; k < 2; k++) {
			Node* x = k ? n->b : n->a;
			Node* y = k ? n->a : n->b;
			if (x->kind != Node::Neg) continue;
			Node* a = x->a; // Bin adds the use of a (mslfix6: no second ++, which kept a product
			                // "used twice" after fneg(fmul) chains and blocked (x*y)*c -> (x*c)*y)
			Rewrite(n, Node::Neg, Bin(Node::Mul, a, y), nullptr);
			return true;
		}
	}
	if (n->kind == Node::Mul) {
		// ~fmul(fmul(is_used_once)(a!const, b!const), #c) -> fmul(fmul(a, c), b)
		for (int k = 0; k < 2; k++) {
			Node* x = k ? n->b : n->a;
			Node* c = k ? n->a : n->b;
			if (x->kind != Node::Mul || x->uses != 1 || !IsConst(c) || IsConst(x->a) ||
			    IsConst(x->b))
				continue;
			Node* a = x->a;
			Node* b = x->b;
			b->uses++;
			Rewrite(n, Node::Mul, Bin(Node::Mul, a, c), b);
			return true;
		}
	}
	if (n->kind == Node::Add) {
		// ~fadd(fadd(is_used_once)(a!const, b(is_fmul)), #c) / (a!const, b!const) -> fadd(fadd(a,
		// c), b)
		for (int k = 0; k < 2; k++) {
			Node* x = k ? n->b : n->a;
			Node* c = k ? n->a : n->b;
			if (x->kind != Node::Add || x->uses != 1 || !IsConst(c) || IsConst(x->a) ||
			    IsConst(x->b))
				continue;
			Node* a = x->a;
			Node* b = x->b;
			if (!IsFmul(b) && IsFmul(a)) std::swap(a, b);
			b->uses++;
			Rewrite(n, Node::Add, Bin(Node::Add, a, c), b);
			return true;
		}
	}
	if (n->kind == Node::Mul || n->kind == Node::Add) {
		// ~fmul(#a, fmul(b, #c)) -> fmul(a*c, b) ; ~fadd(#a, fadd(b, #c)) -> fadd(a+c, b)
		for (int k = 0; k < 2; k++) {
			Node* ca = k ? n->b : n->a;
			Node* x  = k ? n->a : n->b;
			if (!IsConst(ca) || x->kind != n->kind) continue;
			for (int m = 0; m < 2; m++) {
				Node* b  = m ? x->a : x->b;
				Node* cc = m ? x->b : x->a;
				if (!IsConst(cc)) continue;
				auto* f = New();
				f->kind = Node::Const;
				f->uses = 1;
				f->c    = n->kind == Node::Mul ? ca->c * cc->c : ca->c + cc->c;
				b->uses++;
				Rewrite(n, n->kind, f, b);
				return true;
			}
		}
	}
	if (n->kind == Node::Add) {
		// ~fadd(fadd(is_used_once)(a(is_fmul), b(is_fmul)), c(is_not_fmul)) -> fadd(fadd(a, c), b)
		for (int k = 0; k < 2; k++) {
			Node* x = k ? n->b : n->a;
			Node* c = k ? n->a : n->b;
			if (x->kind != Node::Add || x->uses != 1 || !IsFmul(x->a) || !IsFmul(x->b) || IsFmul(c))
				continue;
			Node* a = x->a;
			Node* b = x->b;
			b->uses++;
			Rewrite(n, Node::Add, Bin(Node::Add, a, c), b);
			return true;
		}
	}
	if (n->kind == Node::Add) {
		// heavyshade: ~fadd(a, fmul(c, fadd(b, fneg(a)))) -> flrp(a, b, c) (nir_opt_algebraic, !lower_flrp32).
		// V_INTERP_P1_F32 of a manually interpolated attribute (P0 + I * (P1 - P0)) is this shape; KK
		// prints flrp as `a + (b - a) * c` in one statement, which Metal contracts.
		for (int k = 0; k < 2; k++) {
			Node* a = k ? n->b : n->a;
			Node* m = k ? n->a : n->b;
			if (m->kind != Node::Mul) continue;
			for (int j = 0; j < 2; j++) {
				Node* c = j ? m->b : m->a;
				Node* s = j ? m->a : m->b;
				if (s->kind != Node::Add) continue;
				for (int q = 0; q < 2; q++) {
					Node* ng = q ? s->a : s->b;
					Node* b  = q ? s->b : s->a;
					if (ng->kind != Node::Neg || !Same(ng->a, a)) continue;
					a->uses++;
					b->uses++;
					c->uses++;
					Node* oa = n->a;
					Node* ob = n->b;
					n->kind  = Node::Flrp;
					n->a     = a;
					n->b     = b;
					n->t     = c;
					if (n->inst != nullptr) n->changed = true;
					Release(oa);
					Release(ob);
					return true;
				}
			}
		}
	}
	return false;
}

void Collect(Node* n, std::vector<Node*>& out) { // virtual nodes under an inst node, children first
	for (Node* c: {n->a, n->b, n->t}) {
		if (c == nullptr || c->inst != nullptr || c->kind == Node::Leaf || c->kind == Node::Const)
			continue;
		Collect(c, out);
		out.push_back(c);
	}
}

std::string Text(MslCtx& ctx, const Node* n, bool root) {
	if (!root && n->inst != nullptr) return ctx.Name(*n->inst);
	switch (n->kind) {
		case Node::Const: return MslF32Bits(FloatBitsOf(n->c));
		case Node::Leaf: return ctx.Val(n->leaf);
		case Node::Neg: return fmt::format("(-{})", Text(ctx, n->a, false));
		case Node::Flrp:
			return fmt::format("({0} + (({1} - {0}) * {2}))", Text(ctx, n->a, false), Text(ctx, n->b, false),
			                   Text(ctx, n->t, false));
		case Node::Add:
			return fmt::format("({} + {})", Text(ctx, n->a, false), Text(ctx, n->b, false));
		case Node::Mul:
			return fmt::format("({} * {})", Text(ctx, n->a, false), Text(ctx, n->b, false));
	}
	return {};
}

// fmafix: KK prints every NIR ALU op as its own statement (`tN = a * b;` then `tM = tN + c;`, each
// in its own math_mode(relaxed) block), so Metal's front end never sees a multiply as a direct
// operand of an add and never forms an llvm.fmuladd (-ffp-contract=on contracts only WITHIN one
// expression). B printed V_MAD and re-associated chains as one expression ((a * b) + c), which
// Metal fused into one fma (pipe 100 v617: 1 ulp on 2114/8192 words). Here every virtual node (an
// op that has no IR instruction of its own) becomes a block-local temporary, so B's text has KK's
// statement structure and whatever Metal's middle end does to KK's fmul/fadd it does to B's too.
std::string Stmt(MslCtx& ctx, const Node* n, bool root, const std::string& base, int& k,
                 std::string& out) {
	if (!root && n->inst != nullptr) return ctx.Name(*n->inst);
	std::string s;
	switch (n->kind) {
		case Node::Const: return MslF32Bits(FloatBitsOf(n->c));
		case Node::Leaf: return ctx.Val(n->leaf);
		case Node::Neg: s = fmt::format("(-{})", Stmt(ctx, n->a, false, base, k, out)); break;
		case Node::Flrp: {
			// KK prints nir flrp inline in ONE statement, so Metal may contract it (fma((b-a), t, a))
			const auto x = Stmt(ctx, n->a, false, base, k, out);
			const auto y = Stmt(ctx, n->b, false, base, k, out);
			const auto z = Stmt(ctx, n->t, false, base, k, out);
			s            = fmt::format("({0} + (({1} - {0}) * {2}))", x, y, z);
			break;
		}
		case Node::Add:
		case Node::Mul: {
			const auto x = Stmt(ctx, n->a, false, base, k, out);
			const auto y = Stmt(ctx, n->b, false, base, k, out);
			s            = fmt::format("({} {} {})", x, n->kind == Node::Add ? '+' : '*', y);
			break;
		}
	}
	if (root) return s;
	const auto name = fmt::format("{}_{}", base, k++);
	out += fmt::format("const float {} = {}; ", name, s);
	return name;
}

} // namespace

void NirReassocBuild(const IR::Program& program, bool on, bool split) {
	g       = State {};
	g.on    = on;
	g.split = split;
	if (!on) return;
	for (const auto* block: program.blocks)
		for (const auto& inst: *block) {
			const auto o = inst.GetOpcode();
			if (!IsFloatArith(o) || inst.GetType() != IR::Type::F32) continue;
			auto* n = New();
			n->inst = &inst;
			n->uses = NirUses(&inst);
			switch (o) {
				case O::FPAdd32:
					n->kind = Node::Add;
					n->a    = Operand(inst.Arg(0));
					n->b    = Operand(inst.Arg(1));
					break;
				case O::FPSub32: {
					n->kind = Node::Add;
					n->a    = Operand(inst.Arg(0));
					Node* y = Operand(inst.Arg(1));
					if (y->kind == Node::Const) {
						y->c = -y->c;
						n->b = y;
						break;
					} // nir folds fneg of a constant
					auto* neg = New();
					neg->kind = Node::Neg;
					neg->a    = y;
					neg->uses = 1;
					n->b      = neg;
					break;
				}
				case O::FPMul32:
					n->kind = Node::Mul;
					n->a    = Operand(inst.Arg(0));
					n->b    = Operand(inst.Arg(1));
					break;
				case O::FPFma32:
				case O::FPMad32: {
					n->kind = Node::Add;
					auto* m = New();
					m->kind = Node::Mul;
					m->a    = Operand(inst.Arg(0));
					m->b    = Operand(inst.Arg(1));
					m->uses = 1;
					n->a    = m;
					n->b    = Operand(inst.Arg(2));
					break;
				}
				case O::FPNeg32:
					n->kind = Node::Neg;
					n->a    = Operand(inst.Arg(0));
					break;
				default: break;
			}
			g.by_inst[&inst] = n;
			g.order.push_back(n);
		}
	// Program order (operands before their users: KK's text shows the operand rewrites winning,
	// e.g. fneg propagated out of an fmul before ~fadd(fmul(a,b), fmul(a,c)) could factor it),
	// repeated to a fixed point like the opt loop.
	for (int pass = 0; pass < 16; pass++) {
		bool progress = false;
		for (auto it = g.order.begin(); it != g.order.end(); ++it) {
			Node* n = *it;
			if (n->uses <= 0) continue;
			std::vector<Node*> work {n};
			Collect(n, work);
			for (size_t w = 0; w < work.size() && w < 64; w++) {
				if (TryRules(work[w])) {
					progress   = true;
					n->changed = true;
					work.push_back(work[w]); // the rewritten node may match again
					Collect(work[w], work);
				}
			}
		}
		if (!progress) break;
	}
}

// fmafix: the statements for inst in KK's one-op-per-statement shape ("" = emit the usual text): a
// re-associated expression, or V_MAD_F32/V_FMA_F32 as `{ const float vN_0 = (a * b); vN = (vN_0 +
// c); }`.
std::string NirSplitLines(MslCtx& ctx, const IR::Inst& inst) {
	if (!g.split) return {};
	const auto dst = ctx.Name(inst);
	const auto o   = inst.GetOpcode();
	if (g.on) {
		const auto f = g.by_inst.find(&inst);
		if (f != g.by_inst.end() && (f->second->changed || o == O::FPMad32 || o == O::FPFma32)) {
			std::string pre;
			int         k    = 0;
			const auto  expr = Stmt(ctx, f->second, true, dst, k, pre);
			return pre.empty() ? fmt::format("{} = {};", dst, expr)
			                   : fmt::format("{{ {}{} = {}; }}", pre, dst, expr);
		}
		if (f != g.by_inst.end()) return {};
	}
	if ((o != O::FPMad32 && o != O::FPFma32) || inst.GetType() != IR::Type::F32) return {};
	return fmt::format("{{ const float {0}_0 = ({1} * {2}); {0} = ({0}_0 + {3}); }}", dst,
	                   ctx.Val(inst.Arg(0)), ctx.Val(inst.Arg(1)), ctx.Val(inst.Arg(2)));
}

// Rewritten expression for inst ("" = emit the usual text).
std::string NirReassocText(MslCtx& ctx, const IR::Inst& inst) {
	if (!g.on || g.split) return {};
	const auto f = g.by_inst.find(&inst);
	if (f == g.by_inst.end() || !f->second->changed) return {};
	return Text(ctx, f->second, true);
}

} // namespace Libs::Graphics::ShaderRecompiler::Msl
