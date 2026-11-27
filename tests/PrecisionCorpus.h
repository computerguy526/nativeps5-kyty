// precise (2026-11-27): the synthetic guest programs of the precision tests (tests/PrecisionSpirvTests.cpp, tests/MslCoreTests.cpp).
// Hand-encoded RDNA2 instructions, no game data. Each program computes one value that reaches the position export and one that
// reaches only a parameter (or, for the pixel / compute programs, a value that reaches no position at all), so a test can tell
// "exact on the position data flow" from "exact everywhere" and from "exact nowhere".
#ifndef KYTY_TESTS_PRECISIONCORPUS_H_
#define KYTY_TESTS_PRECISIONCORPUS_H_

#include <cstdint>
#include <vector>

namespace PrecisionCorpus {

constexpr uint32_t Vgpr(uint32_t reg) { return 256u + reg; }
constexpr uint32_t Vop1(uint32_t opcode, uint32_t dst, uint32_t src0) {
	return (0x3fu << 25u) | ((dst & 0xffu) << 17u) | ((opcode & 0xffu) << 9u) | (src0 & 0x1ffu);
}
constexpr uint32_t Vop2(uint32_t opcode, uint32_t dst, uint32_t src0, uint32_t src1) {
	return ((opcode & 0x3fu) << 25u) | ((dst & 0xffu) << 17u) | ((src1 & 0xffu) << 9u) | (src0 & 0x1ffu);
}
constexpr uint32_t Exp0(uint32_t target, uint32_t en, bool done, bool vm = false) {
	return (0x3eu << 26u) | ((target & 0x3fu) << 4u) | (en & 0xfu) | (done ? (1u << 11u) : 0u) | (vm ? (1u << 12u) : 0u);
}
constexpr uint32_t Exp1(uint32_t s0, uint32_t s1, uint32_t s2, uint32_t s3) {
	return (s0 & 0xffu) | ((s1 & 0xffu) << 8u) | ((s2 & 0xffu) << 16u) | ((s3 & 0xffu) << 24u);
}
constexpr uint32_t Ds0(uint32_t opcode, uint32_t offset = 0) { return (0x36u << 26u) | ((opcode & 0xffu) << 18u) | (offset & 0xffffu); }
constexpr uint32_t Ds1(uint32_t vdst, uint32_t data0, uint32_t addr) { return ((vdst & 0xffu) << 24u) | ((data0 & 0xffu) << 8u) | (addr & 0xffu); }

using Code = std::vector<uint32_t>;

// v_mad_f32 dst = a * b + c (VOP3 0x141): unfused on the guest.
inline void Mad(Code& c, uint32_t dst, uint32_t a, uint32_t b, uint32_t add) {
	c.push_back((0x35u << 26u) | (0x141u << 16u) | (dst & 0xffu));
	c.push_back((Vgpr(a) & 0x1ffu) | ((Vgpr(b) & 0x1ffu) << 9u) | ((Vgpr(add) & 0x1ffu) << 18u));
}
// dst = a + (b - a) * t as the guest writes a lerp: v_sub_f32, v_mul_f32, v_add_f32 (tmp is clobbered).
inline void Lerp(Code& c, uint32_t dst, uint32_t tmp, uint32_t a, uint32_t b, uint32_t t) {
	c.push_back(Vop2(0x04, tmp, Vgpr(b), a));   // v_sub_f32 tmp, b, a
	c.push_back(Vop2(0x08, tmp, Vgpr(tmp), t)); // v_mul_f32 tmp, tmp, t
	c.push_back(Vop2(0x03, dst, Vgpr(a), tmp)); // v_add_f32 dst, a, tmp
}
inline void MovLiteral(Code& c, uint32_t dst, uint32_t literal) {
	c.push_back(Vop1(0x01, dst, 255u));
	c.push_back(literal);
}
inline void MovZero(Code& c, uint32_t dst) { c.push_back(Vop1(0x01, dst, 128u)); }
inline void Export(Code& c, uint32_t target, uint32_t xyz, uint32_t w, bool done, bool vm = false) {
	c.push_back(Exp0(target, 0xf, done, vm));
	c.push_back(Exp1(xyz, xyz, xyz, w));
}
inline void LdsWrite(Code& c, uint32_t addr, uint32_t data) {
	c.push_back(Ds0(0x0d)); // ds_write_b32
	c.push_back(Ds1(0, data, addr));
}
inline void LdsRead(Code& c, uint32_t dst, uint32_t addr) {
	c.push_back(Ds0(0x36)); // ds_read_b32
	c.push_back(Ds1(dst, 0, addr));
}
inline void End(Code& c) { c.push_back(0xbf810000u); } // s_endpgm

constexpr uint32_t kPos0 = 12, kParam0 = 32, kMrt0 = 0;

// Position = MAD, parameter = another MAD.
inline Code DirectMad() {
	Code c;
	Mad(c, 4, 0, 1, 2);
	Mad(c, 5, 0, 2, 1);
	MovLiteral(c, 6, 0x3f800000u);
	Export(c, kPos0, 4, 6, true);
	Export(c, kParam0, 5, 6, false);
	End(c);
	return c;
}
// Position = lerp, parameter = another lerp.
inline Code DirectLerp() {
	Code c;
	Lerp(c, 4, 3, 0, 1, 2);
	Lerp(c, 5, 7, 2, 1, 0);
	MovLiteral(c, 6, 0x3f800000u);
	Export(c, kPos0, 4, 6, true);
	Export(c, kParam0, 5, 6, false);
	End(c);
	return c;
}
// The mesh-program shape (NGG): the MAD is stored to LDS, the position export takes what is loaded back; the parameter MAD
// never touches LDS.
inline Code LdsMad() {
	Code c;
	Mad(c, 4, 0, 1, 2);
	Mad(c, 7, 0, 2, 1);
	MovZero(c, 3);
	LdsWrite(c, 3, 4);
	LdsRead(c, 5, 3);
	MovLiteral(c, 6, 0x3f800000u);
	Export(c, kPos0, 5, 6, true);
	Export(c, kParam0, 7, 6, false);
	End(c);
	return c;
}
// A pixel program: MAD to the colour export (no position in this stage).
inline Code PixelMad() {
	Code c;
	Mad(c, 4, 0, 1, 2);
	MovLiteral(c, 6, 0x3f800000u);
	Export(c, kMrt0, 4, 6, true, true);
	End(c);
	return c;
}
// A compute program: MAD stored to LDS (no position in this stage; the LDS hop must not apply).
inline Code ComputeLdsMad() {
	Code c;
	Mad(c, 4, 0, 1, 2);
	MovZero(c, 3);
	LdsWrite(c, 3, 4);
	End(c);
	return c;
}

} // namespace PrecisionCorpus

#endif
