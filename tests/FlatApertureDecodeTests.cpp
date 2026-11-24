// nps5 one: the flat-pointer aperture scalar sources (SRC_SHARED_BASE 0xEB,
// SRC_SHARED_LIMIT 0xEC, SRC_PRIVATE_BASE 0xED, SRC_PRIVATE_LIMIT 0xEE) decode
// to their own operand kinds. Unreal Engine 5.5 compute shaders (PPSA07631)
// read them; a decoder without them ends the session with "unsupported scalar
// source operand". Decoder only, no GPU.
#include "graphics/shader/recompiler/frontend/decode/ShaderDecoder.h"

#include <cstdio>

using namespace Libs::Graphics::ShaderRecompiler::Decoder;

namespace {
int g_failures = 0;
void Expect(bool condition, const char *what) {
  if (!condition) {
    std::printf("FlatApertureDecodeTests: failed: %s\n", what);
    g_failures++;
  }
}
// SOP1 s_mov_b32 sdst, ssrc0.
constexpr uint32_t EncodeSMovB32(uint32_t dst, uint32_t src) {
  return 0x80000000u | (0x7du << 23u) | ((dst & 0x7fu) << 16u) | (0x03u << 8u) |
         (src & 0xffu);
}
OperandKind DecodeSource(uint32_t src) {
  const uint32_t code[] = {EncodeSMovB32(0, src), 0xbf810000u};
  Instruction inst;
  DecodeInstruction(code, 0, inst);
  return inst.src0.kind;
}
} // namespace

int main() {
  Expect(DecodeSource(0xEB) == OperandKind::SharedBase,
         "0xEB is SRC_SHARED_BASE");
  Expect(DecodeSource(0xEC) == OperandKind::SharedLimit,
         "0xEC is SRC_SHARED_LIMIT");
  Expect(DecodeSource(0xED) == OperandKind::PrivateBase,
         "0xED is SRC_PRIVATE_BASE");
  Expect(DecodeSource(0xEE) == OperandKind::PrivateLimit,
         "0xEE is SRC_PRIVATE_LIMIT");
  // The neighbours keep their meaning: 125 is the null register, 126/127 are
  // EXEC.
  Expect(DecodeSource(125) == OperandKind::Null, "125 stays null");
  Expect(DecodeSource(126) == OperandKind::ExecLo, "126 stays EXEC_LO");

  // A whole program with an aperture read decodes to the end and keeps its
  // instruction count.
  const uint32_t program_code[] = {EncodeSMovB32(0, 0xEB),
                                   EncodeSMovB32(1, 0xED), 0xbf810000u};
  Program program;
  DecodeProgram(program_code, program);
  Expect(program.instructions.size() == 3,
         "aperture program decodes all three instructions");
  Expect(!program.instructions.empty() &&
             program.instructions[0].src0.kind == OperandKind::SharedBase &&
             program.instructions[1].src0.kind == OperandKind::PrivateBase,
         "aperture operands survive DecodeProgram");
  uint32_t pc = 0, word = 0;
  Expect(!TakeUndecodable(pc, word), "no operand was recorded as undecodable");

  if (g_failures != 0) {
    std::printf("FlatApertureDecodeTests: %d failure(s)\n", g_failures);
    return 1;
  }
  std::puts("FlatApertureDecodeTests: all cases passed");
  return 0;
}
