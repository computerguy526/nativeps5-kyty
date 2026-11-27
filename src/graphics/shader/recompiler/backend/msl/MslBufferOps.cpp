#include "graphics/shader/recompiler/backend/msl/MslBufferOps.h"

#include "graphics/shader/recompiler/BufferFormat.h"
#include "graphics/shader/recompiler/backend/msl/MslXform.h"

#include <algorithm>

namespace Libs::Graphics::ShaderRecompiler::Msl::Mem {

namespace {

bool Checked(const BufferBinding& b, const BufferOpts& opts) {
	return b.bounds_check && !opts.elide_bounds;
}

std::string InBounds(const BufferBinding& b, const std::string& addr, uint32_t bytes) {
	// perfauto transform bounds_u32 (MslXform.h): the same predicate without a 64-bit sum. For a
	// 64-bit `addr` the comparison still promotes, so the result is unchanged there too.
	// fsslow transform bounds_lim (MslXform.h): one compare against the uniform limit
	if (Xf::On(Xf::BoundsLim) && bytes != 0u) {
		const std::string size = BufSizeName(b.id);
		return "((" + addr + ") < (" + size + " >= " + std::to_string(bytes) + "u ? " + size +
		       " - " + std::to_string(bytes - 1u) + "u : 0u))";
	}
	if (Xf::On(Xf::BoundsU32)) {
		const std::string n = std::to_string(bytes) + "u";
		return "(" + BufSizeName(b.id) + " >= " + n + " && (" + addr + ") <= " + BufSizeName(b.id) +
		       " - " + n + ")";
	}
	return "(ulong(" + addr + ") + " + std::to_string(bytes) + "ul <= ulong(" + BufSizeName(b.id) +
	       "))";
}

const char* PtrQual(const BufferBinding& b) {
	return (b.written || b.atomic) ? "device" : "const device";
}

thread_local int         g_const_load = 0;
thread_local std::string g_const_alias;

// vs_const_load applies to this load: a buffer the program never writes, not a coherent load.
bool ConstB(const BufferBinding& b, bool volatile_) {
	return g_const_load != 0 && !volatile_ && !b.written && !b.atomic;
}

// The load of `type` at the integer address expression `at` (see MslXform.h vs_const_load).
std::string ConstDeref(const std::string& type, const std::string& at) {
	const std::string c = "*(constant " + type + "*)(" + at + ")";
	if (g_const_load == 1) return c;
	return "(" + g_const_alias + " ? *(const device " + type + "*)(" + at + ") : " + c + ")";
}

// `*(T*)(base + addr)` with T an alignment-1 friendly type.
std::string Mem(const BufferBinding& b, const std::string& addr, const std::string& type,
                bool volatile_) {
	if (ConstB(b, volatile_))
		return ConstDeref(type, "(ulong)(" + BufName(b.id) + " + " + addr + ")");
	return "*(" + std::string(volatile_ ? "volatile " : "") + PtrQual(b) + " " + type + "*)(" +
	       BufName(b.id) + " + " + addr + ")";
}

// load_select (MslXform.h): `type name = load from (ok ? buf + addr : safe)`; `conv` wraps the load
// ("uint", "uint4", or empty). The select on `ok` is the caller's.
std::string LoadSelected(MemOut& o, const BufferBinding& b, const std::string& ok,
                         const std::string& addr, const std::string& type, bool volatile_,
                         const char* decl, const std::string& conv, const std::string& qual) {
	const std::string at = o.Def("ulong", ok + " ? (ulong)(" + BufName(b.id) + " + " + addr +
	                                          ") : " + LoadSelectSafe());
	std::string       load =
	    "*(" + std::string(volatile_ ? "volatile " : "") + qual + " " + type + "*)(" + at + ")";
	if (ConstB(b, volatile_)) load = ConstDeref(type, at);
	if (!conv.empty()) load = conv + "(" + load + ")";
	return o.Def(decl, load);
}

} // namespace

static thread_local std::string g_load_select_safe;
void                            SetLoadSelectSafe(const std::string& expr) {
	g_load_select_safe = expr;
}
const std::string& LoadSelectSafe() {
	return g_load_select_safe;
}

void SetConstLoad(int state, const std::string& alias) {
	g_const_load  = state;
	g_const_alias = alias;
}
int ConstLoadState() {
	return g_const_load;
}
void ConstLoadWriter() {
	if (g_const_load == 1) g_const_load = 2;
}
std::string SrtLoad(const std::string& index) {
	if (g_const_load == 0) return "r.srt[" + index + "]";
	return "(*(constant uint*)((ulong)(r.srt + " + index + ")))";
}
bool LoadSelectOn() {
	return Xf::On(Xf::LoadSelect) && !g_load_select_safe.empty();
}

std::string BufferByteAddress(MemOut& o, const BufferBinding& b, const BufferAddrIn& in) {
	const uint32_t stride     = b.packed_stride & 0x3fffu;
	const bool     swz        = stride != 0u && (b.packed_stride & (1u << 14u)) != 0u;
	const bool     add_tid    = (b.packed_stride & (1u << 20u)) != 0u;
	const uint32_t idx_stride = (b.packed_stride >> 16u) & 3u;
	std::string    index      = in.index.empty() ? "0u" : in.index;
	if (add_tid)
		index = "(" + index + " + (" + (in.lane.empty() ? std::string("0u") : in.lane) + " & 63u))";
	std::string offset = in.offset.empty() ? "0u" : in.offset;
	if (in.imm != 0u) offset = "(" + offset + " + " + U(in.imm) + ")";
	std::string linear = "(" + index + " * " + U(stride) + " + " + offset + ")";
	std::string addr;
	if (swz) {
		const std::string idx_shift = U(idx_stride + 3u);
		const std::string indices   = "(1u << " + idx_shift + ")";
		const std::string msb = "(((" + index + " >> " + idx_shift + ") * " + U(stride) + " + (" +
		                        offset + " & ~3u)) * " + indices + ")";
		const std::string lsb =
		    "(((" + index + " & (" + indices + " - 1u)) << 2u) + (" + offset + " & 3u))";
		addr = "(" + msb + " + " + lsb + ")";
	} else {
		addr = linear;
	}
	if (!in.soffset.empty()) addr = "(" + addr + " + " + in.soffset + ")";
	return o.Def("uint", addr);
}

std::string LoadRaw(MemOut& o, const BufferBinding& b, const std::string& addr, uint32_t bits,
                    uint32_t comps, const BufferOpts& opts) {
	const bool     vol         = opts.coherent_loads;
	std::string    type        = bits == 8    ? "uchar"
	                             : bits == 16 ? "ushort"
	                             : comps == 1 ? "uint"
	                                          : "packed_uint" + std::to_string(comps);
	const uint32_t bytes       = bits == 32 ? 4u * comps : bits / 8u;
	std::string    result_type = bits == 32 && comps > 1 ? "uint" + std::to_string(comps) : "uint";
	std::string    load        = Mem(b, addr, type, vol);
	if (bits == 32 && comps > 1)
		load = result_type + "(" + load + ")";
	else if (bits != 32)
		load = "uint(" + load + ")";
	if (!Checked(b, opts)) return o.Def(result_type.c_str(), load);
	if (opts.per_dword_bounds && bits == 32 && comps > 1) {
		std::string part = result_type + "(";
		for (uint32_t i = 0; i < comps; i++) {
			// Checked in 64 bits from the base (addr + 4i + 4 <= size): a uint `addr + 4i` wraps
			// for an address just below 2^32 (a negative offset) and would pass as dword 0.
			const std::string a = "(" + addr + " + " + std::to_string(4u * i) + "u)";
			part += (i ? ", " : "") + InBounds(b, addr, 4u * (i + 1u)) + " ? " +
			        Mem(b, a, "uint", vol) + " : 0u";
		}
		part += ")";
		return o.Def(result_type.c_str(), InBounds(b, addr, bytes) + " ? " + load + " : " + part);
	}
	if (LoadSelectOn()) {
		const std::string ok   = o.Def("bool", InBounds(b, addr, bytes));
		const std::string conv = (bits == 32 && comps > 1)
		                             ? result_type
		                             : (bits != 32 ? std::string("uint") : std::string());
		const std::string raw =
		    LoadSelected(o, b, ok, addr, type, vol, result_type.c_str(), conv, PtrQual(b));
		return o.Def(result_type.c_str(), ok + " ? " + raw + " : " + result_type + "(0u)");
	}
	return o.Def(result_type.c_str(),
	             InBounds(b, addr, bytes) + " ? " + load + " : " + result_type + "(0u)");
}

void StoreRaw(MemOut& o, const BufferBinding& b, const std::string& addr, uint32_t bits,
              const std::vector<std::string>& v, const BufferOpts& opts) {
	ConstLoadWriter();
	const uint32_t comps = static_cast<uint32_t>(v.size());
	const uint32_t bytes = bits == 32 ? 4u * comps : bits / 8u;
	std::string    stmt;
	if (bits == 8)
		stmt = Mem(b, addr, "uchar", false) + " = uchar(" + v[0] + ");";
	else if (bits == 16)
		stmt = Mem(b, addr, "ushort", false) + " = ushort(" + v[0] + ");";
	else if (comps == 1)
		stmt = Mem(b, addr, "uint", false) + " = " + v[0] + ";";
	else {
		stmt = Mem(b, addr, "packed_uint" + std::to_string(comps), false) + " = packed_uint" +
		       std::to_string(comps) + "(";
		for (uint32_t i = 0; i < comps; i++)
			stmt += (i ? ", " : "") + v[i];
		stmt += ");";
	}
	if (Checked(b, opts) && opts.per_dword_bounds && bits == 32 && comps > 1) {
		std::string part;
		for (uint32_t i = 0; i < comps; i++) {
			const std::string a = "(" + addr + " + " + std::to_string(4u * i) + "u)";
			part += std::string(i ? " " : "") + "if (" + InBounds(b, addr, 4u * (i + 1u)) + ") { " +
			        Mem(b, a, "uint", false) + " = " + v[i] + "; }"; // 64-bit check (LoadRaw)
		}
		o.Line("if (" + InBounds(b, addr, bytes) + ") { " + stmt + " } else { " + part + " }");
	} else if (Checked(b, opts))
		o.Line("if (" + InBounds(b, addr, bytes) + ") { " + stmt + " }");
	else
		o.Line(stmt);
}

std::string BufferHelpersPrelude() {
	return R"(// ---- T2 buffer format helpers ----
inline float t2_ufloat_to_f32(uint raw, uint mb) {
    uint e = (raw >> mb) & 31u; uint m = raw & ((1u << mb) - 1u);
    if (e == 0u) return m == 0u ? 0.0f : ldexp(float(m), -14 - int(mb));
    if (e == 31u) return m == 0u ? INFINITY : NAN;
    return ldexp(float((1u << mb) | m), int(e) - 15 - int(mb));
}
inline uint t2_f32_to_ufloat(float f, uint mb) {
    uint bits = as_type<uint>(f);
    if ((bits >> 31) != 0u) return isnan(f) ? ((31u << mb) | 1u) : 0u;
    uint e = (bits >> 23) & 255u; uint m = bits & 0x7fffffu;
    uint maxfin = (30u << mb) | ((1u << mb) - 1u);
    if (e == 255u) return m != 0u ? ((31u << mb) | 1u) : (31u << mb);
    int ne = int(e) - 112;
    if (ne >= 31) return maxfin;
    if (ne <= 0) return min(uint(rint(f * exp2(float(14u + mb)))), maxfin);
    uint sh = 23u - mb; uint x = (uint(ne) << 23) | m;
    uint r = (x + ((1u << (sh - 1u)) - 1u) + ((x >> sh) & 1u)) >> sh;
    return min(r, maxfin);
}
inline int t2_sx(uint v, uint bits) { return as_type<int>(v << (32u - bits)) >> (32u - bits); }
// ---- end helpers ----
)";
}

namespace {

using Format::ComponentType;

// Raw extraction of component c (zero-extended) from the element base `addr`.
std::string ExtractRaw(MemOut& o, const BufferBinding& b, const std::string& addr,
                       const Format::BufferFormatInfo& f, uint32_t c,
                       const std::string& packed_word) {
	if (f.packed_bitfield) {
		return o.Def("uint",
		             "(" + packed_word + " >> " + U(f.component_bit_offset[c]) + ") & " +
		                 Hex((f.component_bits[c] >= 32 ? 0u : (1u << f.component_bits[c])) - 1u));
	}
	const uint32_t    byte = f.component_bit_offset[c] / 8u;
	const std::string a    = byte ? "(" + addr + " + " + U(byte) + ")" : addr;
	const char*       t    = f.component_bits[c] == 8    ? "uchar"
	                         : f.component_bits[c] == 16 ? "ushort"
	                                                     : "uint";
	return o.Def("uint", "uint(" + Mem(b, a, t, false) + ")");
}

std::string DecodeComponent(MemOut& o, const Format::BufferFormatInfo& f, uint32_t c,
                            const std::string& raw) {
	const uint32_t    bits = f.component_bits[c];
	const uint32_t    maxv = bits >= 32 ? 0xffffffffu : (1u << bits) - 1u;
	const std::string sx   = "t2_sx(" + raw + ", " + U(bits) + ")";
	switch (f.type) {
		case ComponentType::Uint: return raw;
		case ComponentType::Sint:
			return bits >= 32 ? raw : o.Def("uint", "as_type<uint>(" + sx + ")");
		case ComponentType::Unorm:
			return o.Def("uint",
			             "as_type<uint>(float(" + raw + ") * " + F32Lit(1.0f / float(maxv)) + ")");
		case ComponentType::Uscaled: return o.Def("uint", "as_type<uint>(float(" + raw + "))");
		case ComponentType::Sscaled: return o.Def("uint", "as_type<uint>(float(" + sx + "))");
		case ComponentType::Snorm: {
			const uint32_t smax = (1u << (bits - 1u)) - 1u;
			return o.Def("uint", "as_type<uint>(max(float(" + sx + ") * " +
			                         F32Lit(1.0f / float(smax)) + ", -1.0f))");
		}
		case ComponentType::Float:
			if (bits == 32) return raw;
			if (bits == 16)
				return o.Def("uint", "as_type<uint>(float(as_type<half>(ushort(" + raw + "))))");
			return o.Def("uint",
			             "as_type<uint>(t2_ufloat_to_f32(" + raw + ", " + U(bits - 5u) + "))");
		default: return "0u";
	}
}

} // namespace

std::string LoadFormatted(MemOut& o, const BufferBinding& b, const std::string& addr,
                          uint32_t format, uint32_t dst_sel, const BufferOpts& opts) {
	const auto info = Format::GetFormatInfo(static_cast<Prospero::BufferFormat>(format));
	if (info.component_count == 0u) {
		o.Fail("LoadFormatted: unsupported buffer format " + std::to_string(format));
		return "uint4(0u)";
	}
	// Outputs are declared first and assigned inside one guard so the whole element is
	// all-or-nothing.
	const std::string ok =
	    Checked(b, opts) ? InBounds(b, addr, info.byte_size) : std::string("true");
	std::string comp[4];
	std::string out = o.Tmp();
	o.Line("uint4 " + out + " = uint4(0u);");
	o.Line("if (" + ok + ") {");
	std::string packed_word;
	if (info.packed_bitfield) packed_word = o.Def("uint", Mem(b, addr, "uint", false));
	for (uint32_t c = 0; c < info.component_count; c++) {
		const auto raw = ExtractRaw(o, b, addr, info, c, packed_word);
		comp[c]        = DecodeComponent(o, info, c, raw);
	}
	const uint32_t one =
	    (info.type == ComponentType::Uint || info.type == ComponentType::Sint) ? 1u : 0x3f800000u;
	for (uint32_t i = 0; i < 4; i++) {
		const uint32_t sel = (dst_sel >> (i * 3u)) & 7u;
		std::string    v   = "0u";
		if (sel == 1u)
			v = Hex(one);
		else if (sel >= 4u)
			v = comp[(sel - 4u) % info.component_count];
		o.Line("    " + out + "[" + std::to_string(i) + "] = " + v + ";");
	}
	o.Line("}");
	return out;
}

void StoreFormatted(MemOut& o, const BufferBinding& b, const std::string& addr, uint32_t format,
                    const std::vector<std::string>& values, const BufferOpts& opts) {
	const auto info = Format::GetFormatInfo(static_cast<Prospero::BufferFormat>(format));
	if (info.component_count == 0u) {
		o.Fail("StoreFormatted: unsupported buffer format " + std::to_string(format));
		return;
	}
	ConstLoadWriter();
	const std::string ok =
	    Checked(b, opts) ? InBounds(b, addr, info.byte_size) : std::string("true");
	o.Line("if (" + ok + ") {");
	std::string word = "0u";
	for (uint32_t c = 0; c < info.component_count; c++) {
		const std::string v    = c < values.size() ? values[c] : "0u";
		const uint32_t    bits = info.component_bits[c];
		const uint32_t    maxv = bits >= 32 ? 0xffffffffu : (1u << bits) - 1u;
		const uint32_t    smax = bits >= 2 ? (1u << (bits - 1u)) - 1u : 0u;
		std::string       enc;
		switch (info.type) {
			case ComponentType::Uint:
				enc = bits >= 32 ? v : "min(" + v + ", " + Hex(maxv) + ")";
				break;
			case ComponentType::Sint:
				enc = bits >= 32 ? v
				                 : "(uint(clamp(as_type<int>(" + v + "), " +
				                       std::to_string(-int64_t(smax) - 1) + ", " +
				                       std::to_string(smax) + ")) & " + Hex(maxv) + ")";
				break;
			case ComponentType::Unorm:
				enc = "uint(rint(clamp(as_type<float>(" + v + "), 0.0f, 1.0f) * " +
				      F32Lit(float(maxv)) + "))";
				break;
			case ComponentType::Snorm:
				enc = "(uint(int(rint(clamp(as_type<float>(" + v + "), -1.0f, 1.0f) * " +
				      F32Lit(float(smax)) + "))) & " + Hex(maxv) + ")";
				break;
			case ComponentType::Uscaled:
				enc = "min(uint(rint(max(as_type<float>(" + v + "), 0.0f))), " + Hex(maxv) + ")";
				break;
			case ComponentType::Sscaled:
				enc = "(uint(int(clamp(rint(as_type<float>(" + v + ")), " +
				      F32Lit(-float(smax) - 1.0f) + ", " + F32Lit(float(smax)) + "))) & " +
				      Hex(maxv) + ")";
				break;
			case ComponentType::Float:
				enc = bits == 32 ? v
				      : bits == 16
				          ? "uint(as_type<ushort>(half(as_type<float>(" + v + "))))"
				          : "t2_f32_to_ufloat(as_type<float>(" + v + "), " + U(bits - 5u) + ")";
				break;
			default: enc = "0u";
		}
		auto e = o.Def("uint", enc);
		if (info.packed_bitfield) {
			word = "(" + word + " | (" + e + " << " + U(info.component_bit_offset[c]) + "))";
		} else {
			const uint32_t    byte = info.component_bit_offset[c] / 8u;
			const std::string a    = byte ? "(" + addr + " + " + U(byte) + ")" : addr;
			const char*       t    = bits == 8 ? "uchar" : bits == 16 ? "ushort" : "uint";
			o.Line(Mem(b, a, t, false) + " = " + std::string(t) + "(" + e + ");");
		}
	}
	if (info.packed_bitfield) o.Line(Mem(b, addr, "uint", false) + " = " + word + ";");
	o.Line("}");
}

namespace {

std::string AtomPtr(const BufferBinding& b, const std::string& addr) {
	return "(device atomic_uint*)(" + BufName(b.id) + " + " + addr + ")";
}

} // namespace

std::string AtomicRmw32(MemOut& o, const std::string& p, const char* space, AtomicOp op,
                        const std::string& value, const std::string& cmp, std::string indent) {
	const std::string old = o.Tmp();
	o.Line(indent + "uint " + old + " = 0u;");
	const std::string rel   = "memory_order_relaxed";
	const auto        fetch = [&](const char* fn, const std::string& val) {
		o.Line(indent + old + " = " + fn + "_explicit(" + p + ", " + val + ", " + rel + ");");
	};
	switch (op) {
		case AtomicOp::Swap: fetch("atomic_exchange", value); break;
		case AtomicOp::Add: fetch("atomic_fetch_add", value); break;
		case AtomicOp::Sub: fetch("atomic_fetch_sub", value); break;
		case AtomicOp::UMin: fetch("atomic_fetch_min", value); break;
		case AtomicOp::UMax: fetch("atomic_fetch_max", value); break;
		case AtomicOp::And: fetch("atomic_fetch_and", value); break;
		case AtomicOp::Or: fetch("atomic_fetch_or", value); break;
		case AtomicOp::Xor: fetch("atomic_fetch_xor", value); break;
		case AtomicOp::SMin:
		case AtomicOp::SMax:
			o.Line(indent + old + " = as_type<uint>(atomic_fetch_" +
			       std::string(op == AtomicOp::SMin ? "min" : "max") + "_explicit((" + space +
			       " atomic_int*)" + p + ", as_type<int>(" + value + "), " + rel + "));");
			break;
		case AtomicOp::CmpSwap: {
			const std::string e = o.Tmp();
			o.Line(indent + "uint " + e + " = " + cmp + "; bool ok_" + e + " = false;");
			o.Line(indent + "do { " + e + " = " + cmp + "; ok_" + e +
			       " = atomic_compare_exchange_weak_explicit(" + p + ", &" + e + ", " + value +
			       ", " + rel + ", " + rel + "); } while (!ok_" + e + " && " + e + " == " + cmp +
			       ");");
			o.Line(indent + old + " = " + e + ";");
			break;
		}
		case AtomicOp::Inc:
		case AtomicOp::Dec:
		case AtomicOp::FMin:
		case AtomicOp::FMax: {
			const std::string cur = o.Tmp();
			std::string       nv;
			if (op == AtomicOp::Inc)
				nv = cur + " >= " + value + " ? 0u : " + cur + " + 1u";
			else if (op == AtomicOp::Dec)
				nv = "(" + cur + " == 0u || " + cur + " > " + value + ") ? " + value + " : " + cur +
				     " - 1u";
			else
				nv = "as_type<uint>(" + std::string(op == AtomicOp::FMin ? "fmin" : "fmax") +
				     "(as_type<float>(" + cur + "), as_type<float>(" + value + ")))";
			o.Line(indent + "uint " + cur + " = atomic_load_explicit(" + p + ", " + rel + ");");
			o.Line(indent + "while (!atomic_compare_exchange_weak_explicit(" + p + ", &" + cur +
			       ", (" + nv + "), " + rel + ", " + rel + ")) {}");
			o.Line(indent + old + " = " + cur + ";");
			break;
		}
	}
	return old;
}

std::string BufferAtomic32(MemOut& o, const BufferBinding& b, const std::string& addr, AtomicOp op,
                           const std::string& value, const std::string& cmp,
                           const BufferOpts& opts) {
	ConstLoadWriter();
	const std::string guard = Checked(b, opts) ? InBounds(b, addr, 4) : std::string("true");
	const std::string res   = o.Tmp();
	o.Line("uint " + res + " = 0u;");
	o.Line("if (" + guard + ") {");
	const std::string p   = o.Def("device atomic_uint*", AtomPtr(b, addr));
	const auto        old = AtomicRmw32(o, p, "device", op, value, cmp, "    ");
	o.Line("    " + res + " = " + old + ";");
	o.Line("}");
	return res;
}

std::string BufferAtomic64(MemOut& o, const BufferBinding& b, const std::string& addr, AtomicOp op,
                           const std::string& value, const std::string& cmp,
                           const BufferOpts& opts) {
	(void)cmp;
	ConstLoadWriter();
	const std::string old = o.Tmp();
	o.Line("ulong " + old + " = 0ul;");
	const std::string guard = Checked(b, opts) ? InBounds(b, addr, 8) : std::string("true");
	o.Line("if (" + guard + ") {");
	const std::string lo  = o.Def("device atomic_uint*", AtomPtr(b, addr));
	const std::string hi  = o.Def("device atomic_uint*", AtomPtr(b, "(" + addr + " + 4u)"));
	const std::string rel = "memory_order_relaxed";
	const std::string vlo = "uint(" + value + ")", vhi = "uint((" + value + ") >> 32)";
	const auto        join = [&](const std::string& l, const std::string& h) {
		o.Line("    " + old + " = ulong(" + l + ") | (ulong(" + h + ") << 32);");
	};
	switch (op) {
		case AtomicOp::Add: {
			const std::string l = o.Tmp(), h = o.Tmp();
			o.Line("    uint " + l + " = atomic_fetch_add_explicit(" + lo + ", " + vlo + ", " +
			       rel + ");");
			o.Line("    uint " + h + " = atomic_fetch_add_explicit(" + hi + ", " + vhi + " + ((" +
			       l + " + " + vlo + " < " + l + ") ? 1u : 0u), " + rel + ");");
			join(l, h);
			break;
		}
		case AtomicOp::Sub: {
			const std::string l = o.Tmp(), h = o.Tmp();
			o.Line("    uint " + l + " = atomic_fetch_sub_explicit(" + lo + ", " + vlo + ", " +
			       rel + ");");
			o.Line("    uint " + h + " = atomic_fetch_sub_explicit(" + hi + ", " + vhi + " + ((" +
			       l + " < " + vlo + ") ? 1u : 0u), " + rel + ");");
			join(l, h);
			break;
		}
		case AtomicOp::And:
		case AtomicOp::Or:
		case AtomicOp::Xor: {
			const char*       fn = op == AtomicOp::And  ? "atomic_fetch_and_explicit"
			                       : op == AtomicOp::Or ? "atomic_fetch_or_explicit"
			                                            : "atomic_fetch_xor_explicit";
			const std::string l = o.Tmp(), h = o.Tmp();
			o.Line("    uint " + l + " = " + fn + "(" + lo + ", " + vlo + ", " + rel + ");");
			o.Line("    uint " + h + " = " + fn + "(" + hi + ", " + vhi + ", " + rel + ");");
			join(l, h);
			break;
		}
		case AtomicOp::Swap: {
			// Result-less swap == atomic 64-bit store (see atomic64Lowering.h); with a result: two
			// halves.
			const std::string l = o.Tmp(), h = o.Tmp();
			o.Line("    uint " + l + " = atomic_exchange_explicit(" + lo + ", " + vlo + ", " + rel +
			       ");");
			o.Line("    uint " + h + " = atomic_exchange_explicit(" + hi + ", " + vhi + ", " + rel +
			       ");");
			join(l, h);
			break;
		}
		case AtomicOp::UMin:
		case AtomicOp::UMax:
		case AtomicOp::SMin:
		case AtomicOp::SMax:
		case AtomicOp::CmpSwap:
		default:
			// Needs a true 64-bit RMW. Cheap CAS fallback on the 64-bit word via a lock-free
			// seqlock is not possible; report so the caller can flag the shader.
			o.Fail("BufferAtomic64: min/max/cmpswap on 64-bit needs hardware 64-bit atomics (not "
			       "in this toolchain)");
			break;
	}
	o.Line("}");
	return old;
}

std::string ReadConstBuffer(MemOut& o, const BufferBinding& b, const std::string& byte_offset,
                            const BufferOpts& opts) {
	// S_BUFFER_LOAD reads go through `constant` when the buffer is never written, which engages
	// Apple's constant/uniform cache path (KK emitted device+coherent here).
	const std::string load =
	    ConstB(b, false) ? Mem(b, byte_offset, "uint", false)
	                     : "*(" + std::string(b.written || b.atomic ? "device" : "const device") +
	                           " uint*)(" + BufName(b.id) + " + " + byte_offset + ")";
	if (!Checked(b, opts)) return o.Def("uint", load);
	if (LoadSelectOn()) {
		const std::string ok  = o.Def("bool", InBounds(b, byte_offset, 4));
		const std::string raw = LoadSelected(o, b, ok, byte_offset, "uint", false, "uint", "",
		                                     b.written || b.atomic ? "device" : "const device");
		return o.Def("uint", ok + " ? " + raw + " : 0u");
	}
	return o.Def("uint", InBounds(b, byte_offset, 4) + " ? " + load + " : 0u");
}

bool RegionChecked(const BufferBinding& b) {
	return Checked(b, BufferOpts {});
}
std::string RegionLimit(const BufferBinding& b, uint32_t bytes) {
	const std::string size = BufSizeName(b.id);
	return "(" + size + " >= " + std::to_string(bytes) + "u ? " + size + " - " +
	       std::to_string(bytes - 1u) + "u : 0u)";
}
std::string RegionFastLoad(const BufferBinding& b, const std::string& addr, uint32_t comps,
                           bool mutant) {
	const std::string at = mutant ? "(" + addr + " + 4u)" : addr;
	return "uint" + std::to_string(comps) + "(" +
	       Mem(b, at, "packed_uint" + std::to_string(comps), false) + ")";
}

void GroupLoadWords(MemOut& o, const BufferBinding& b, const std::string& lo, const std::string& hi,
                    const std::string& pre, uint32_t ndw, const std::string& W,
                    const std::string& F, bool mutant) {
	const BufferOpts opts;
	o.Line("uint " + W + "[" + std::to_string(ndw) + "];");
	std::string cond = "(" + hi + " >= " + lo + ")";
	if (Checked(b, opts))
		cond += " && " + InBounds(b, hi, 4); // bounds_u32 form when that transform is on
	if (!pre.empty()) cond = "(" + pre + ") && " + cond;
	o.Line("bool " + F + " = " + cond + ";");
	o.Line("if (" + F + ") {");
	for (uint32_t k = 0; k < ndw;) {
		const uint32_t    n = std::min(4u, ndw - k);
		const std::string at =
		    "(" + lo + " + " + std::to_string(4u * k + (mutant ? 4u : 0u)) + "u)";
		if (n == 1)
			o.Line("    " + W + "[" + std::to_string(k) + "] = " + Mem(b, at, "uint", false) + ";");
		else {
			const std::string v = W + "_v" + std::to_string(k);
			o.Line("    uint" + std::to_string(n) + " " + v + " = uint" + std::to_string(n) + "(" +
			       Mem(b, at, "packed_uint" + std::to_string(n), false) + ");");
			for (uint32_t q = 0; q < n; q++)
				o.Line("    " + W + "[" + std::to_string(k + q) + "] = " + v + "." + "xyzw"[q] +
				       ";");
		}
		k += n;
	}
	o.Line("}");
}

std::string AddressHelpersPrelude() {
	return R"(// ---- T2 guest address helpers ----
#ifndef T2_GUEST_PTR
#define T2_GUEST_PTR(a) ((device uchar*)(a))
#endif
inline uint t2_ld32(ulong a) {
    uint o = uint(a & 3ul); ulong b = a & ~3ul;
    uint w0 = *(device const uint*)T2_GUEST_PTR(b);
    if (o == 0u) return w0;
    uint w1 = *(device const uint*)T2_GUEST_PTR(b + 4ul);
    return (w0 >> (o * 8u)) | (w1 << ((4u - o) * 8u));
}
inline uint t2_ld16(ulong a) {
    uint o = uint(a & 3ul); ulong b = a & ~3ul;
    uint w0 = *(device const uint*)T2_GUEST_PTR(b);
    uint v = w0 >> (o * 8u);
    if (o == 3u) v |= (*(device const uint*)T2_GUEST_PTR(b + 4ul)) << 8u;
    return v & 0xffffu;
}
inline uint t2_ld8(ulong a) { return uint(*(device const uchar*)T2_GUEST_PTR(a)); }
inline void t2_st32(ulong a, uint v) {
    if ((a & 3ul) == 0ul) { *(device uint*)T2_GUEST_PTR(a) = v; return; }
    for (uint i = 0u; i < 4u; i++) { *(device uchar*)T2_GUEST_PTR(a + ulong(i)) = uchar(v >> (i * 8u)); }
}
inline void t2_st16(ulong a, uint v) {
    if ((a & 1ul) == 0ul) { *(device ushort*)T2_GUEST_PTR(a) = ushort(v); return; }
    *(device uchar*)T2_GUEST_PTR(a) = uchar(v); *(device uchar*)T2_GUEST_PTR(a + 1ul) = uchar(v >> 8u);
}
// ---- end guest address helpers ----
)";
}

std::string AddressLoad(MemOut& o, const std::string& a, uint32_t bits) {
	return o.Def("uint", std::string(bits == 8    ? "t2_ld8("
	                                 : bits == 16 ? "t2_ld16("
	                                              : "t2_ld32(") +
	                         a + ")");
}

void AddressStore(MemOut& o, const std::string& a, uint32_t bits, const std::string& value) {
	if (bits == 8)
		o.Line("*(device uchar*)T2_GUEST_PTR(" + a + ") = uchar(" + value + ");");
	else
		o.Line(std::string(bits == 16 ? "t2_st16(" : "t2_st32(") + a + ", " + value + ");");
}

} // namespace Libs::Graphics::ShaderRecompiler::Msl::Mem
