// Scalar & vector ALU, conversions, compares and selects as MSL expressions.
#include "graphics/shader/recompiler/backend/msl/MslEmitter.h"

#include <fmt/format.h>

namespace Libs::Graphics::ShaderRecompiler::Msl {

using IR::ValueOpcode;

std::string MslTypeName(IR::Type type) {
	switch (type) {
		case IR::Type::U1: return "bool";
		case IR::Type::U8: return "uchar";
		case IR::Type::U16: return "ushort";
		case IR::Type::U32: return "uint";
		case IR::Type::U64: return "ulong";
		case IR::Type::F16: return "half";
		case IR::Type::F32: return "float";
		case IR::Type::U32x2: return "uint2";
		case IR::Type::U32x3: return "uint3";
		case IR::Type::U32x4: return "uint4";
		case IR::Type::F32x2: return "float2";
		default: return {};
	}
}

namespace {

std::string Si(const std::string& a) {
	return fmt::format("as_type<int>({})", a);
}
std::string Sl(const std::string& a) {
	return fmt::format("as_type<long>({})", a);
}

} // namespace

// Notes on GCN semantics baked into the text:
//  - shift counts are masked (& 31 / & 63) because Metal shifts by >= width are undefined;
//  - integer<->float conversions go through kyty_* prelude helpers (saturating, NaN -> 0);
//  - FPMad32 relies on `#pragma METAL fp contract(off)` in the prelude (not fused);
//  - FP compares use isordered/isunordered so NaN handling does not depend on the fast-math mode.
static thread_local bool g_fuse_mad = false;
void                     SetFuseMad(bool fuse) {
	g_fuse_mad = fuse;
}
// mslfix: relaxed pixel text spells exp2/log2 precise:: as KK does (fast:: approximations
// otherwise).
static thread_local bool g_precise_tr = false;
void                     SetPreciseTranscendentals(bool on) {
	g_precise_tr = on;
}
// mslfix2: graphics text flushes f32 denormal inputs of V_RCP/RSQ/SQRT/EXP/LOG to zeros of the same
// sign, as Kyty's SPIR-V (FlushTranscendentalInput, FLOAT_MODE 0xC0) that KK runs in game
// (rcp(-denorm) = -inf). Compute text is unchanged.
static thread_local bool g_ftz_tr = false;
void                     SetFlushTranscendentalInputs(bool on) {
	g_ftz_tr = on;
}
static std::string FtzIn(const std::string& x) {
	if (!g_ftz_tr) return x;
	return fmt::format("as_type<float>((as_type<uint>({0}) & 0x7fffffffu) < 8388608u ? "
	                   "(as_type<uint>({0}) & 0x80000000u) : as_type<uint>({0}))",
	                   x);
}

std::string AluExpression(ValueOpcode op, std::span<const std::string> a) {
	using O = ValueOpcode;
	switch (op) {
		// --- bit casts & integer width conversions
		case O::BitCastU16F16: return fmt::format("as_type<ushort>({})", a[0]);
		case O::BitCastF16U16: return fmt::format("as_type<half>({})", a[0]);
		case O::BitCastU32F32: return fmt::format("as_type<uint>({})", a[0]);
		case O::BitCastF32U32: return fmt::format("as_type<float>({})", a[0]);
		case O::BitCastU64F64:
		case O::BitCastF64U64: return {}; // F64: not representable in MSL (unsupported)
		case O::ConvertU16U32: return fmt::format("ushort({})", a[0]);
		case O::ConvertU32U16: return fmt::format("uint({})", a[0]);
		case O::ConvertU8U32: return fmt::format("uchar({})", a[0]);
		case O::ConvertU32U8: return fmt::format("uint({})", a[0]);
		case O::ConvertF32F16: return fmt::format("float({})", a[0]);
		case O::ConvertF16F32: return fmt::format("half({})", a[0]);
		case O::ConvertS32F32: return fmt::format("kyty_f2i({})", a[0]);
		case O::ConvertU32F32: return fmt::format("kyty_f2u({})", a[0]);
		case O::ConvertF32S32: return fmt::format("float({})", Si(a[0]));
		case O::ConvertF32U32: return fmt::format("float({})", a[0]);
		// --- composites
		case O::CompositeConstructU64:
			return fmt::format("(ulong({}) | (ulong({}) << 32))", a[0], a[1]);
		case O::CompositeConstructU32x2: return fmt::format("uint2({}, {})", a[0], a[1]);
		case O::CompositeConstructU32x3: return fmt::format("uint3({}, {}, {})", a[0], a[1], a[2]);
		case O::CompositeConstructU32x4:
			return fmt::format("uint4({}, {}, {}, {})", a[0], a[1], a[2], a[3]);
		case O::CompositeConstructF32x2: return fmt::format("float2({}, {})", a[0], a[1]);
		case O::CompositeExtractU64:
			return fmt::format("uint(({} >> (({} & 1u) * 32u)))", a[0], a[1]);
		case O::CompositeExtractU32x2:
		case O::CompositeExtractU32x3:
		case O::CompositeExtractU32x4: return fmt::format("{}[{}]", a[0], a[1]);
		case O::PackHalf2x16: return fmt::format("as_type<uint>(half2({}))", a[0]);
		case O::PackSnorm2x16: return fmt::format("pack_float_to_snorm2x16({})", a[0]);
		case O::PackUnorm2x16: return fmt::format("pack_float_to_unorm2x16({})", a[0]);
		case O::PackFloat2x16Rtz: return fmt::format("kyty_pack_f16_rtz({}, {})", a[0], a[1]);
		// --- source/destination modifiers
		case O::FPAbs32: return fmt::format("fabs({})", a[0]);
		case O::FPNeg32: return fmt::format("(-{})", a[0]);
		case O::FPSaturate32: return fmt::format("saturate({})", a[0]);
		case O::BitFieldInsert:
			return fmt::format("insert_bits({}, {}, {}, {})", a[0], a[1], a[2], a[3]);
		case O::BitFieldUExtract: return fmt::format("extract_bits({}, {}, {})", a[0], a[1], a[2]);
		case O::BitFieldSExtract:
			return fmt::format("as_type<uint>(extract_bits({}, {}, {}))", Si(a[0]), a[1], a[2]);
		// --- selects
		case O::SelectU1:
		case O::SelectF32:
		case O::SelectU32: return fmt::format("({} ? {} : {})", a[0], a[1], a[2]);
		// --- integer arithmetic
		case O::IAdd32:
		case O::IAdd64: return fmt::format("({} + {})", a[0], a[1]);
		case O::IAddCarry32: return fmt::format("kyty_add_carry({}, {})", a[0], a[1]);
		case O::ISub32:
		case O::ISub64: return fmt::format("({} - {})", a[0], a[1]);
		case O::IMul32:
		case O::IMul64: return fmt::format("({} * {})", a[0], a[1]);
		case O::UDiv32: return fmt::format("({} / {})", a[0], a[1]);
		case O::SMulHi: return fmt::format("as_type<uint>(mulhi({}, {}))", Si(a[0]), Si(a[1]));
		case O::UMulHi: return fmt::format("mulhi({}, {})", a[0], a[1]);
		case O::IAbs32: return fmt::format("uint(abs({}))", Si(a[0]));
		case O::ShiftLeftLogical32: return fmt::format("({} << ({} & 31u))", a[0], a[1]);
		case O::ShiftLeftLogical64: return fmt::format("({} << ({} & 63u))", a[0], a[1]);
		case O::ShiftRightLogical32: return fmt::format("({} >> ({} & 31u))", a[0], a[1]);
		case O::ShiftRightLogical64: return fmt::format("({} >> ({} & 63u))", a[0], a[1]);
		case O::WqmU64:
			return fmt::format("(((({0}) | (({0}) >> 1ul) | (({0}) >> 2ul) | (({0}) >> 3ul)) & "
			                   "0x1111111111111111ul) * 15ul)",
			                   a[0]);
		case O::ShiftRightArithmetic32:
			return fmt::format("as_type<uint>({} >> ({} & 31u))", Si(a[0]), a[1]);
		case O::ShiftRightArithmetic64:
			return fmt::format("as_type<ulong>({} >> ({} & 63u))", Sl(a[0]), a[1]);
		case O::BitwiseAnd32:
		case O::BitwiseAnd64: return fmt::format("({} & {})", a[0], a[1]);
		case O::BitwiseOr32: return fmt::format("({} | {})", a[0], a[1]);
		case O::BitwiseXor32: return fmt::format("({} ^ {})", a[0], a[1]);
		case O::BitwiseNot32: return fmt::format("(~{})", a[0]);
		case O::BitReverse32: return fmt::format("reverse_bits({})", a[0]);
		case O::BitCount32: return fmt::format("popcount({})", a[0]);
		case O::BitCount64: return fmt::format("uint(popcount({}))", a[0]);
		case O::FindUMsb32: return fmt::format("({0} == 0u ? 0xffffffffu : 31u - clz({0}))", a[0]);
		case O::FindUMsb64:
			return fmt::format("({0} == 0ul ? 0xffffffffu : 63u - uint(clz({0})))", a[0]);
		case O::FindILsb32: return fmt::format("({0} == 0u ? 0xffffffffu : ctz({0}))", a[0]);
		case O::SMin32: return fmt::format("as_type<uint>(min({}, {}))", Si(a[0]), Si(a[1]));
		case O::UMin32: return fmt::format("min({}, {})", a[0], a[1]);
		case O::SMax32: return fmt::format("as_type<uint>(max({}, {}))", Si(a[0]), Si(a[1]));
		case O::UMax32: return fmt::format("max({}, {})", a[0], a[1]);
		case O::SMinTri32:
			return fmt::format("as_type<uint>(min3({}, {}, {}))", Si(a[0]), Si(a[1]), Si(a[2]));
		case O::UMinTri32: return fmt::format("min3({}, {}, {})", a[0], a[1], a[2]);
		case O::SMaxTri32:
			return fmt::format("as_type<uint>(max3({}, {}, {}))", Si(a[0]), Si(a[1]), Si(a[2]));
		case O::UMaxTri32: return fmt::format("max3({}, {}, {})", a[0], a[1], a[2]);
		case O::SMedTri32:
			return fmt::format("as_type<uint>(median3({}, {}, {}))", Si(a[0]), Si(a[1]), Si(a[2]));
		case O::UMedTri32: return fmt::format("median3({}, {}, {})", a[0], a[1], a[2]);
		// --- integer compares
		case O::SLessThan32: return fmt::format("({} < {})", Si(a[0]), Si(a[1]));
		case O::SLessThan64: return fmt::format("({} < {})", Sl(a[0]), Sl(a[1]));
		case O::SLessThanEqual64: return fmt::format("({} <= {})", Sl(a[0]), Sl(a[1]));
		case O::ULessThanEqual64:
		case O::ULessThanEqual32: return fmt::format("({} <= {})", a[0], a[1]);
		case O::UGreaterThanEqual64:
		case O::UGreaterThanEqual32: return fmt::format("({} >= {})", a[0], a[1]);
		case O::ULessThan32:
		case O::ULessThan64: return fmt::format("({} < {})", a[0], a[1]);
		case O::IEqual32:
		case O::IEqual64: return fmt::format("({} == {})", a[0], a[1]);
		case O::SLessThanEqual32: return fmt::format("({} <= {})", Si(a[0]), Si(a[1]));
		case O::SGreaterThan32: return fmt::format("({} > {})", Si(a[0]), Si(a[1]));
		case O::UGreaterThan32:
		case O::UGreaterThan64: return fmt::format("({} > {})", a[0], a[1]);
		case O::INotEqual32:
		case O::INotEqual64: return fmt::format("({} != {})", a[0], a[1]);
		case O::SGreaterThanEqual32: return fmt::format("({} >= {})", Si(a[0]), Si(a[1]));
		case O::LogicalOr: return fmt::format("({} || {})", a[0], a[1]);
		case O::LogicalAnd: return fmt::format("({} && {})", a[0], a[1]);
		case O::LogicalXor: return fmt::format("({} != {})", a[0], a[1]);
		case O::LogicalNot: return fmt::format("(!{})", a[0]);
		// --- float compares (ordered: false on NaN; unordered: true on NaN)
		case O::FPOrdEqual32: return fmt::format("({} == {})", a[0], a[1]);
		case O::FPOrdNotEqual32:
			return fmt::format("(isordered({0}, {1}) && {0} != {1})", a[0], a[1]);
		case O::FPOrdLessThan32: return fmt::format("({} < {})", a[0], a[1]);
		case O::FPOrdGreaterThan32: return fmt::format("({} > {})", a[0], a[1]);
		case O::FPOrdLessThanEqual32: return fmt::format("({} <= {})", a[0], a[1]);
		case O::FPOrdGreaterThanEqual32: return fmt::format("({} >= {})", a[0], a[1]);
		case O::FPUnordEqual32:
			return fmt::format("(isunordered({0}, {1}) || {0} == {1})", a[0], a[1]);
		case O::FPUnordNotEqual32: return fmt::format("({} != {})", a[0], a[1]);
		case O::FPUnordLessThan32:
			return fmt::format("(isunordered({0}, {1}) || {0} < {1})", a[0], a[1]);
		case O::FPUnordGreaterThan32:
			return fmt::format("(isunordered({0}, {1}) || {0} > {1})", a[0], a[1]);
		case O::FPUnordLessThanEqual32:
			return fmt::format("(isunordered({0}, {1}) || {0} <= {1})", a[0], a[1]);
		case O::FPUnordGreaterThanEqual32:
			return fmt::format("(isunordered({0}, {1}) || {0} >= {1})", a[0], a[1]);
		case O::FPIsNan32: return fmt::format("isnan({})", a[0]);
		case O::FPCmpClass32: return fmt::format("kyty_class_f32({}, {})", a[0], a[1]);
		case O::FPCmpClass16: return fmt::format("kyty_class_f16({}, {})", a[0], a[1]);
		// --- float arithmetic
		case O::FPAdd32: return fmt::format("({} + {})", a[0], a[1]);
		case O::FPSub32: return fmt::format("({} - {})", a[0], a[1]);
		case O::FPMul32: return fmt::format("({} * {})", a[0], a[1]);
		case O::FPFma32: return fmt::format("fma({}, {}, {})", a[0], a[1], a[2]);
		case O::FPMad32:
			if (g_fuse_mad)
				return fmt::format(
				    "fma({}, {}, {})", a[0], a[1],
				    a[2]); // MadMode Position/Fused outside position data: same as the SPIR-V path
			return fmt::format("(({} * {}) + {})", a[0], a[1], a[2]);
		case O::FPMin32: return fmt::format("fmin({}, {})", a[0], a[1]);
		case O::FPMax32: return fmt::format("fmax({}, {})", a[0], a[1]);
		case O::FPMinTri32: return fmt::format("fmin3({}, {}, {})", a[0], a[1], a[2]);
		case O::FPMaxTri32: return fmt::format("fmax3({}, {}, {})", a[0], a[1], a[2]);
		case O::FPMedTri32: return fmt::format("fmedian3({}, {}, {})", a[0], a[1], a[2]);
		case O::FPRecip32:
			return fmt::format("kyty_rcp({})",
			                   FtzIn(a[0])); // relaxed 1/x = what KK's NIR->MSL emits (mslrest:
			                                 // perms 18/211 bit-exact)
		case O::FPRecipIFlag32:
			return fmt::format(
			    "kyty_rcp({})",
			    a[0]); // integer-converted input: never denormal (no flush, as SPIR-V)
		case O::FPRecipSqrt32:
			return fmt::format("kyty_rsqrt({})",
			                   FtzIn(a[0])); // precise::rsqrt as KK (mslrest2: perm 395)
		case O::FPSqrt: return fmt::format("kyty_sqrt({})", FtzIn(a[0]));
		case O::FPSin: return fmt::format("sin(fract({}) * 6.28318548f)", a[0]);
		case O::FPCos: return fmt::format("cos(fract({}) * 6.28318548f)", a[0]);
		case O::FPExp2: return (g_precise_tr ? "precise::exp2(" : "exp2(") + FtzIn(a[0]) + ")";
		case O::FPLog2: return (g_precise_tr ? "precise::log2(" : "log2(") + FtzIn(a[0]) + ")";
		case O::FPLdexp: return fmt::format("ldexp({}, {})", a[0], Si(a[1]));
		case O::FPRoundEven32: return fmt::format("rint({})", a[0]);
		case O::FPFloor32: return fmt::format("floor({})", a[0]);
		case O::FPCeil32: return fmt::format("ceil({})", a[0]);
		case O::FPTrunc32: return fmt::format("trunc({})", a[0]);
		case O::FPFract32: return fmt::format("fract({})", a[0]);
		default: return {};
	}
}

} // namespace Libs::Graphics::ShaderRecompiler::Msl
