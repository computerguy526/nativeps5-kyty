// kytyApertureScan: decodes raw ISA (.bin) files and prints "<file> ap=<n>
// flat=<n> bvh=<0|1>": ap = instructions with a SRC_SHARED/PRIVATE_BASE/LIMIT
// operand, flat = FLAT-family instructions. Links the kyty_gpu dylib.
#include "graphics/shader/recompiler/frontend/decode/ShaderDecoder.h"
#include <cstdio>
#include <vector>
using namespace Libs::Graphics::ShaderRecompiler::Decoder;
static bool Ap(const Operand &o) {
  return o.kind == OperandKind::SharedBase ||
         o.kind == OperandKind::SharedLimit ||
         o.kind == OperandKind::PrivateBase ||
         o.kind == OperandKind::PrivateLimit;
}
int main(int argc, char **argv) {
  for (int a = 1; a < argc; a++) {
    std::FILE *f = std::fopen(argv[a], "rb");
    if (!f)
      continue;
    std::vector<uint32_t> code;
    uint32_t w;
    while (std::fread(&w, 4, 1, f) == 1)
      code.push_back(w);
    std::fclose(f);
    Program p;
    DecodeProgram(code, p, true);
    unsigned ap = 0, flat = 0;
    for (const auto &i : p.instructions) {
      if (Ap(i.src0) || Ap(i.src1) || Ap(i.src2) || Ap(i.src3))
        ap++;
      if (i.family == Family::FLAT)
        flat++;
    }
    std::printf("%s ap=%u flat=%u bvh=%d\n", argv[a], ap, flat,
                p.has_bvh ? 1 : 0);
  }
}
