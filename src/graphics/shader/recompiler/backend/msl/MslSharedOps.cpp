#include "graphics/shader/recompiler/backend/msl/MslSharedOps.h"

namespace Libs::Graphics::ShaderRecompiler::Msl::Mem {

namespace {
thread_local bool g_lds_device = false;
std::string Sp(const LdsInfo& l) { return l.space; }
std::string Base(const LdsInfo& l, const std::string& addr) { return "((" + l.space + " uchar*)lds + " + addr + ")"; }
std::string Fits(const LdsInfo& l, const std::string& addr, uint32_t n) {
	return "(ulong(" + addr + ") + " + std::to_string(n) + "ul <= " + (l.bound.empty() ? std::to_string(l.bytes) + "ul" : "ulong(" + l.bound + ")") + ")";
}
// atomics: Metal has no `coherent` atomic pointers; device atomics are coherent anyway.
std::string ASp(const LdsInfo& l) { return l.bound.empty() ? std::string("threadgroup") : std::string("device"); }
} // namespace

void SetLdsDevice(bool on) { g_lds_device = on; }
bool LdsIsDevice() { return g_lds_device; }
LdsInfo MakeLdsInfo(uint32_t bytes) {
	LdsInfo l;
	l.bytes = bytes;
	if (g_lds_device) { l.space = "KK_COHERENT device"; l.bound = "kk_lds_bytes"; }
	return l;
}

std::string LdsLoad(MemOut& o, const LdsInfo& l, const std::string& addr, uint32_t bits, uint32_t comps) {
	const uint32_t bytes = bits == 32 ? 4u * comps : bits / 8u;
	const std::string rt = bits == 32 && comps > 1 ? "uint" + std::to_string(comps) : "uint";
	std::string load;
	if (bits == 8) load = "uint(*(" + Sp(l) + " uchar*)" + Base(l, addr) + ")";
	else if (bits == 16) load = "uint(*(" + Sp(l) + " ushort*)" + Base(l, addr) + ")";
	else if (comps == 1) load = "*(" + Sp(l) + " uint*)" + Base(l, addr);
	else load = rt + "(*(" + Sp(l) + " packed_uint" + std::to_string(comps) + "*)" + Base(l, addr) + ")";
	return o.Def(rt.c_str(), Fits(l, addr, bytes) + " ? " + load + " : " + rt + "(0u)");
}

void LdsStore(MemOut& o, const LdsInfo& l, const std::string& addr, uint32_t bits, const std::vector<std::string>& v) {
	const uint32_t comps = static_cast<uint32_t>(v.size());
	const uint32_t bytes = bits == 32 ? 4u * comps : bits / 8u;
	std::string st;
	if (bits == 8) st = "*(" + Sp(l) + " uchar*)" + Base(l, addr) + " = uchar(" + v[0] + ");";
	else if (bits == 16) st = "*(" + Sp(l) + " ushort*)" + Base(l, addr) + " = ushort(" + v[0] + ");";
	else if (comps == 1) st = "*(" + Sp(l) + " uint*)" + Base(l, addr) + " = " + v[0] + ";";
	else {
		st = "*(" + Sp(l) + " packed_uint" + std::to_string(comps) + "*)" + Base(l, addr) + " = packed_uint" + std::to_string(comps) + "(";
		for (uint32_t i = 0; i < comps; i++) st += (i ? ", " : "") + v[i];
		st += ");";
	}
	o.Line("if (" + Fits(l, addr, bytes) + ") { " + st + " }");
}

std::string LdsAtomic32(MemOut& o, const LdsInfo& l, const std::string& addr, AtomicOp op, const std::string& value, const std::string& cmp) {
	const std::string res = o.Tmp();
	o.Line("uint " + res + " = 0u;");
	o.Line("if (" + Fits(l, addr, 4) + ") {");
	const auto p = o.Def((ASp(l) + " atomic_uint*").c_str(), "(" + ASp(l) + " atomic_uint*)" + Base(l, addr));
	const auto old = AtomicRmw32(o, p, ASp(l).c_str(), op, value, cmp, "    ");
	o.Line("    " + res + " = " + old + ";");
	o.Line("}");
	return res;
}

std::string LdsMaskedOr32(MemOut& o, const LdsInfo& l, const std::string& addr, const std::string& mask, const std::string& value) {
	const std::string res = o.Tmp();
	o.Line("uint " + res + " = 0u;");
	o.Line("if (" + Fits(l, addr, 4) + ") {");
	const auto p = o.Def((ASp(l) + " atomic_uint*").c_str(), "(" + ASp(l) + " atomic_uint*)" + Base(l, addr));
	const auto cur = o.Tmp();
	o.Line("    uint " + cur + " = atomic_load_explicit(" + p + ", memory_order_relaxed);");
	o.Line("    while (!atomic_compare_exchange_weak_explicit(" + p + ", &" + cur + ", ((" + cur + " & ~(" + mask + ")) | (" + value + ")), memory_order_relaxed, memory_order_relaxed)) {}");
	o.Line("    " + res + " = " + cur + ";");
	o.Line("}");
	return res;
}

std::string LdsAtomic64(MemOut& o, const LdsInfo& l, const std::string& addr, AtomicOp op, const std::string& value) {
	const std::string res = o.Tmp();
	o.Line("ulong " + res + " = 0ul;");
	if (op != AtomicOp::Add && op != AtomicOp::Sub && op != AtomicOp::Or && op != AtomicOp::And && op != AtomicOp::Xor) {
		o.Fail("LdsAtomic64: only add/sub/and/or/xor are supported (no 64-bit threadgroup atomics)");
		return res;
	}
	o.Line("if (" + Fits(l, addr, 8) + ") {");
	const auto lo = o.Def((ASp(l) + " atomic_uint*").c_str(), "(" + ASp(l) + " atomic_uint*)" + Base(l, addr));
	const auto hi = o.Def((ASp(l) + " atomic_uint*").c_str(), "(" + ASp(l) + " atomic_uint*)" + Base(l, "(" + addr + " + 4u)"));
	const std::string vlo = "uint(" + value + ")", vhi = "uint((" + value + ") >> 32)";
	const std::string L = o.Tmp(), H = o.Tmp();
	const char* rel = "memory_order_relaxed";
	if (op == AtomicOp::Add || op == AtomicOp::Sub) {
		const bool add = op == AtomicOp::Add;
		o.Line("    uint " + L + " = atomic_fetch_" + (add ? "add" : "sub") + "_explicit(" + lo + ", " + vlo + ", " + rel + ");");
		o.Line("    uint " + H + " = atomic_fetch_" + (add ? "add" : "sub") + "_explicit(" + hi + ", " + vhi + " + ((" + (add ? L + " + " + vlo + " < " + L : L + " < " + vlo) + ") ? 1u : 0u), " + rel + ");");
	} else {
		const char* fn = op == AtomicOp::And ? "and" : op == AtomicOp::Or ? "or" : "xor";
		o.Line("    uint " + L + " = atomic_fetch_" + fn + "_explicit(" + lo + ", " + vlo + ", " + rel + ");");
		o.Line("    uint " + H + " = atomic_fetch_" + fn + "_explicit(" + hi + ", " + vhi + ", " + rel + ");");
	}
	o.Line("    " + res + " = ulong(" + L + ") | (ulong(" + H + ") << 32);");
	o.Line("}");
	return res;
}

void LdsBarrier(MemOut& o) { o.Line("threadgroup_barrier(mem_flags::mem_threadgroup);"); }

} // namespace Libs::Graphics::ShaderRecompiler::Msl::Mem
