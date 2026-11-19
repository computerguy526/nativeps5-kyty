// MSL backend core tests (T1): ALU snippets, MslBuilder, whole-program emission
// from hand-assembled RDNA2 code. If `xcrun metal` exists every emitted snippet
// is also compiled (-fsyntax-only). Without arguments runs everything; exit
// code 0 = pass.
#include "graphics/shader/recompiler/CodegenOptions.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/backend/msl/MslBufferOps.h"
#include "graphics/shader/recompiler/backend/msl/MslBuilder.h"
#include "graphics/shader/recompiler/backend/msl/MslEmitter.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

namespace Common {
int DbgExitIfHandler(const char *, const char *, int) { return 1; }
int DbgNotImplementedHandler(const char *, const char *, int) { return 1; }
int DbgExitHandler(const char *, int, std::string_view text) {
  std::printf("EXIT %.*s\n", (int)text.size(), text.data());
  std::_Exit(3);
}
int DbgExitHandler(const char *, int, fmt::text_style, std::string_view text) {
  std::printf("EXIT %.*s\n", (int)text.size(), text.data());
  std::_Exit(3);
}
void DbgExit(int) { std::abort(); }
} // namespace Common

namespace Libs::Graphics::ShaderRecompiler::Spirv {
// Standalone build: no SPIR-V backend (see tools/msl-dump/spirvStubs.cpp for
// the rest).
}

using namespace Libs::Graphics;
namespace SR = Libs::Graphics::ShaderRecompiler;
namespace Msl = Libs::Graphics::ShaderRecompiler::Msl;
using SR::IR::ValueOpcode;

namespace {

int g_failed = 0, g_checks = 0;

void Check(bool ok, const std::string &what) {
  g_checks++;
  if (!ok) {
    g_failed++;
    std::printf("FAIL %s\n", what.c_str());
  }
}

std::string Alu(ValueOpcode op, std::vector<std::string> args) {
  return Msl::AluExpression(op, args);
}

void TestBuilder() {
  Msl::MslBuilder b;
  b.Open("void f()");
  b.Line("int x = 1;");
  b.Open("if (x)");
  b.Comment("hi");
  b.Close();
  b.Close();
  Check(b.Str() == "void f() {\n  int x = 1;\n  if (x) {\n    // hi\n  }\n}\n",
        "builder indentation: " + b.Str());
  Check(Msl::MslU32(7) == "7u" && Msl::MslU64(7) == "7ul", "literals");
  Check(Msl::MslF32Bits(0x3f800000u) == "1.0f",
        "f32 1.0: " + Msl::MslF32Bits(0x3f800000u));
  Check(Msl::MslF32Bits(0x80000000u) == "-0.0f", "f32 -0");
  Check(Msl::MslF32Bits(0x7fc00000u).starts_with("as_type<float>("), "f32 nan");
}

void TestAluGolden() {
  Check(Alu(ValueOpcode::IAdd32, {"a", "b"}) == "(a + b)", "iadd");
  Check(Alu(ValueOpcode::ShiftLeftLogical32, {"a", "b"}) == "(a << (b & 31u))",
        "shl masks the count");
  Check(Alu(ValueOpcode::ShiftRightArithmetic32, {"a", "b"}) ==
            "as_type<uint>(as_type<int>(a) >> (b & 31u))",
        "sra");
  Check(Alu(ValueOpcode::SLessThan32, {"a", "b"}) ==
            "(as_type<int>(a) < as_type<int>(b))",
        "slt");
  Check(Alu(ValueOpcode::ULessThan32, {"a", "b"}) == "(a < b)", "ult");
  Check(Alu(ValueOpcode::SelectU32, {"c", "a", "b"}) == "(c ? a : b)",
        "select");
  Check(Alu(ValueOpcode::FPMad32, {"a", "b", "c"}) == "((a * b) + c)",
        "mad is not fma");
  Check(Alu(ValueOpcode::FPFma32, {"a", "b", "c"}) == "fma(a, b, c)", "fma");
  Check(Alu(ValueOpcode::ConvertS32F32, {"a"}) == "kyty_f2i(a)",
        "f2i saturating helper");
  Check(Alu(ValueOpcode::FPUnordLessThan32, {"a", "b"}) ==
            "(isunordered(a, b) || a < b)",
        "unordered lt");
  Check(Alu(ValueOpcode::FPOrdNotEqual32, {"a", "b"}) ==
            "(isordered(a, b) && a != b)",
        "ordered ne");
  Check(Alu(ValueOpcode::BitFieldUExtract, {"a", "o", "c"}) ==
            "extract_bits(a, o, c)",
        "bfe");
  Check(Alu(ValueOpcode::FindUMsb32, {"a"}).find("clz(a)") != std::string::npos,
        "findumsb");
  Check(Alu(ValueOpcode::UMulHi, {"a", "b"}) == "mulhi(a, b)", "umulhi");
  Check(Alu(ValueOpcode::CompositeConstructU32x2, {"a", "b"}) == "uint2(a, b)",
        "construct");
  Check(Alu(ValueOpcode::CompositeExtractU32x4, {"v", "2u"}) == "v[2u]",
        "extract");
  Check(Alu(ValueOpcode::BitCastF32U32, {"a"}) == "as_type<float>(a)",
        "bitcast");
  // not ALU: memory ops and F64 return empty (handled by the T2 hook /
  // unsupported)
  Check(Alu(ValueOpcode::LoadBufferU32, {"a", "b", "c", "d", "e"}).empty(),
        "buffer load is not ALU");
  Check(Alu(ValueOpcode::FPAdd64, {"a", "b"}).empty(), "f64 unsupported");
}

bool HaveMetal() {
  return std::system("xcrun -sdk macosx -f metal >/dev/null 2>&1") == 0;
}

// One kernel that evaluates every ALU opcode over typed parameters, compiled
// with the Metal compiler.
void TestAluCompiles() {
  if (!HaveMetal()) {
    std::printf("(no xcrun metal: skipping compile checks)\n");
    return;
  }
  std::string src = Msl::MslPrelude();
  std::string body;
  int n = 0;
  for (int i = 0; i < static_cast<int>(ValueOpcode::Count); i++) {
    const auto op = static_cast<ValueOpcode>(i);
    const auto count = SR::IR::NumArgsOf(op);
    std::vector<std::string> args;
    bool ok = true;
    for (size_t a = 0; a < count; a++) {
      const auto t = Msl::MslTypeName(SR::IR::ArgTypeOf(op, a));
      if (t.empty()) {
        ok = false;
        break;
      }
      args.push_back("p" + std::to_string(a) + "_" + t);
    }
    const auto result = Msl::MslTypeName(SR::IR::TypeOf(op));
    if (!ok || result.empty())
      continue;
    const auto expr = Msl::AluExpression(op, args);
    if (expr.empty())
      continue;
    std::string params;
    for (size_t a = 0; a < args.size(); a++)
      params += (a ? ", " : "") + Msl::MslTypeName(SR::IR::ArgTypeOf(op, a)) +
                " " + args[a];
    body += "static inline " + result + " alu" + std::to_string(n++) + "(" +
            params + ") { return " + expr + "; } // " +
            std::string(SR::IR::ValueOpcodeName(op)) + "\n";
  }
  src += body + "kernel void k() {}\n";
  std::ofstream("/tmp/msl_core_alu.metal") << src;
  const int rc =
      std::system("xcrun -sdk macosx metal -std=metal3.1 -fsyntax-only "
                  "/tmp/msl_core_alu.metal 2>/tmp/msl_core_alu.err");
  Check(rc == 0, "all ALU expressions compile with the Metal compiler (" +
                     std::to_string(n) + " ops, see /tmp/msl_core_alu.err)");
}

// RDNA2 words (from tests/LaneLocalTests.cpp): s_mov_b64 s[10:11], 0; L:
// v_cmp_ne_u32 vcc, 0, v0; s_or_b64 s[10:11], s[10:11], vcc; s_andn2_b64 exec,
// exec, vcc; s_cbranch_execnz L; ...
void TestProgram() {
  const std::vector<uint32_t> code = {0xbe8a0480, 0x7d8a0080, 0x888a6a0a,
                                      0x8afe6a7e, 0xbf89fffc, 0x88fe0a7e,
                                      0x7e020281, 0xbf810000};
  ShaderComputeInputInfo info;
  info.threads_num[0] = 64;
  info.threads_num[1] = 1;
  info.threads_num[2] = 1;
  info.host_subgroup_size = 32;
  SR::CompileOptions options;
  options.stage = ShaderType::Compute;
  options.wave_size = 64;
  options.dump_ir = false;
  options.emit_msl = true;
  options.emit_spirv = false;
  std::vector<uint32_t> ud(16, 0);
  options.user_data = ud;
  options.input_info.compute = &info;
  for (const bool force_dispatcher : {false, true}) {
    options.msl_force_dispatcher = force_dispatcher;
    const std::string tag = force_dispatcher ? "dispatcher" : "structured";
    auto translated = SR::TranslateProgram(code, options);
    SR::IR::ResourceSpecialization spec;
    auto result = SR::CompileProgram(std::move(translated), options, spec, 0);
    Check(result.msl.find("kernel void main0") != std::string::npos,
          tag + ": kernel emitted");
    Check(force_dispatcher ==
              (result.msl.find("switch (blk)") != std::string::npos),
          tag + ": control flow form");
    Check(force_dispatcher ||
              (result.msl_structured &&
               result.msl.find("for (;;)") != std::string::npos),
          tag + ": loop rebuilt as for(;;)");
    Check(result.msl.find("[[max_total_threads_per_threadgroup(32)]]") !=
              std::string::npos,
          tag + ": wave64 -> 32 threads (two halves)");
    Check(result.msl.find("_1;") != std::string::npos,
          tag + ": second guest half emitted");
    if (HaveMetal()) {
      const std::string f = "/tmp/msl_core_prog_" + tag + ".metal";
      std::ofstream(f) << result.msl;
      Check(
          std::system(("xcrun -sdk macosx metal -std=metal3.1 -fsyntax-only " +
                       f + " 2>" + f + ".err")
                          .c_str()) == 0,
          tag + ": emitted MSL compiles (see " + f + ".err)");
    }
  }
}

// nps5 gpuguard: MslOptions::loop_cap puts a budget check at the top of every
// loop (structured for(;;) and the dispatcher while(true)); off = no trace of
// it. The GPU-side proof (an infinite loop ends) is
// work/mslnative/harness/tests/loopguard.
void TestLoopCap() {
  const std::vector<uint32_t> code = {0xbe8a0480, 0x7d8a0080, 0x888a6a0a,
                                      0x8afe6a7e, 0xbf89fffc, 0x88fe0a7e,
                                      0x7e020281, 0xbf810000};
  ShaderComputeInputInfo info;
  info.threads_num[0] = 64;
  info.threads_num[1] = 1;
  info.threads_num[2] = 1;
  info.host_subgroup_size = 32;
  SR::CompileOptions options;
  options.stage = ShaderType::Compute;
  options.wave_size = 64;
  options.dump_ir = false;
  options.emit_msl = true;
  options.emit_spirv = false;
  std::vector<uint32_t> ud(16, 0);
  options.user_data = ud;
  options.input_info.compute = &info;
  for (const uint32_t cap : {0u, 1u << 18u, 7u}) {
    for (const bool force_dispatcher : {false, true}) {
      options.msl_force_dispatcher = force_dispatcher;
      options.msl_loop_cap = cap;
      const std::string tag =
          std::string(force_dispatcher ? "dispatcher" : "structured") +
          " cap " + std::to_string(cap);
      auto translated = SR::TranslateProgram(code, options);
      SR::IR::ResourceSpecialization spec;
      auto result = SR::CompileProgram(std::move(translated), options, spec, 0);
      const std::string &m = result.msl;
      if (cap == 0u) {
        Check(m.find("nps5_lg") == std::string::npos,
              tag + ": no loop guard when off");
        continue;
      }
      Check(m.find("bool nps5_lg_step(thread uint& n) { if (n >= " +
                   std::to_string(cap) + "u) return false;") !=
                std::string::npos,
            tag + ": helper with the budget");
      Check(m.find("uint nps5_lg_n = 0u;") != std::string::npos,
            tag + ": per-invocation counter");
      const std::string head = force_dispatcher ? "while (true)" : "for (;;)";
      const size_t h = m.find(head);
      Check(h != std::string::npos &&
                m.find("if (!nps5_lg_step(nps5_lg_n)) break;", h) !=
                    std::string::npos &&
                m.find("if (!nps5_lg_step(nps5_lg_n)) break;", h) <
                    m.find('\n', m.find('{', h) + 1) + 80,
            tag + ": guard is the first statement of the loop body");
      if (HaveMetal()) {
        const std::string f = "/tmp/msl_core_loopcap_" + std::to_string(cap) +
                              (force_dispatcher ? "_d" : "_s") + ".metal";
        std::ofstream(f) << m;
        Check(std::system(
                  ("xcrun -sdk macosx metal -std=metal3.1 -fsyntax-only " + f +
                   " 2>" + f + ".err")
                      .c_str()) == 0,
              tag + ": guarded MSL compiles (see " + f + ".err)");
      }
    }
  }
  options.msl_loop_cap = 0;
  options.msl_force_dispatcher = false;
}

// mslbadopt: math semantics guard (rv103 N1 / skyflick: fast math folds
// isnan/isinf guards -> sky flicker). Compute B must be compiled with 'math
// safe' (kk-mslinject header), never contain math_mode(fast) or fast:: calls;
// only relaxed scopes (inf/NaN kept). Also KYTY_MSL_VOTE_MERGE (default 1; 0 =
// old): wave64 halves vote once (any(a)||any(b) == any(a||b)), the merged text
// compiles.
size_t CountOutsideComments(const std::string &s, const std::string &pat) {
  size_t n = 0, pos = 0;
  while ((pos = s.find(pat, pos)) != std::string::npos) {
    const size_t ls = s.rfind('\n', pos);
    const size_t lb = ls == std::string::npos ? 0 : ls + 1;
    if (s.find("//", lb) == std::string::npos || s.find("//", lb) > pos)
      n++;
    pos += pat.size();
  }
  return n;
}

void TestMathSafeAndVoteMerge() {
  const std::string prelude = Msl::MslPrelude();
  Check(CountOutsideComments(prelude, "math_mode(fast)") == 0,
        "prelude: no math_mode(fast)");
  Check(CountOutsideComments(prelude, "fast::") == 0,
        "prelude: no fast:: functions");
  const std::vector<uint32_t> code = {0xbe8a0480, 0x7d8a0080, 0x888a6a0a,
                                      0x8afe6a7e, 0xbf89fffc, 0x88fe0a7e,
                                      0x7e020281, 0xbf810000};
  ShaderComputeInputInfo info;
  info.threads_num[0] = 64;
  info.threads_num[1] = 1;
  info.threads_num[2] = 1;
  info.host_subgroup_size = 32;
  SR::CompileOptions options;
  options.stage = ShaderType::Compute;
  options.wave_size = 64;
  options.dump_ir = false;
  options.emit_msl = true;
  options.emit_spirv = false;
  std::vector<uint32_t> ud(16, 0);
  options.user_data = ud;
  options.input_info.compute = &info;
  std::string text[2];
  for (const int merge : {0, 1}) {
    setenv("KYTY_MSL_CONVENTION", "kk-bda", 1);
    setenv("KYTY_MSL_VOTE_MERGE", merge ? "1" : "0", 1);
    auto translated = SR::TranslateProgram(code, options);
    SR::IR::ResourceSpecialization spec;
    auto result = SR::CompileProgram(std::move(translated), options, spec, 0);
    unsetenv("KYTY_MSL_CONVENTION");
    unsetenv("KYTY_MSL_VOTE_MERGE");
    const std::string tag = merge ? "vote merge" : "default";
    text[merge] = result.msl;
    Check(result.msl.rfind("// kk-mslinject v1 local_size", 0) == 0 &&
              result.msl.find(" math safe\n") != std::string::npos,
          tag + ": kk-bda header asks for math safe");
    Check(CountOutsideComments(result.msl, "math_mode(fast)") == 0 &&
              CountOutsideComments(result.msl, "fast::") == 0,
          tag + ": no fast-math scope in compute");
    if (HaveMetal()) {
      const std::string f =
          "/tmp/msl_core_vote_" + std::to_string(merge) + ".metal";
      std::ofstream(f) << result.msl;
      Check(std::system(("xcrun -sdk macosx metal -std=metal3.1 "
                         "-fmetal-math-mode=safe -fsyntax-only " +
                         f + " 2>" + f + ".err")
                            .c_str()) == 0,
            tag + ": compiles (see " + f + ".err)");
    }
  }
  const auto votes = [](const std::string &s) {
    return CountOutsideComments(s, "kyty_any(") +
           CountOutsideComments(s, "kyty_all(");
  };
  Check(votes(text[1]) < votes(text[0]),
        "vote merge: fewer simdgroup votes (" + std::to_string(votes(text[0])) +
            " -> " + std::to_string(votes(text[1])) + ")");
  Check(text[1].find(") || (") != std::string::npos ||
            text[1].find(") && (") != std::string::npos,
        "vote merge: halves combined before the vote");
}

// review-fixes v1.1.1: the binding model belongs to one emission. Two
// specializations of one shader have the same shader_hash and, compiled one
// after the other, the same IR::Program address; a cache keyed by both served
// the first one's resources to the second.
void TestBindingModelPerEmission() {
  const std::vector<uint32_t> code = {0xbe8a0480, 0x7d8a0080, 0x888a6a0a,
                                      0x8afe6a7e, 0xbf89fffc, 0x88fe0a7e,
                                      0x7e020281, 0xbf810000};
  ShaderComputeInputInfo info;
  info.threads_num[0] = 64;
  info.threads_num[1] = 1;
  info.threads_num[2] = 1;
  info.host_subgroup_size = 32;
  SR::CompileOptions options;
  options.stage = ShaderType::Compute;
  options.wave_size = 64;
  options.dump_ir = false;
  options.emit_msl = true;
  options.emit_spirv = false;
  std::vector<uint32_t> ud(16, 0);
  options.user_data = ud;
  options.input_info.compute = &info;
  auto translated = SR::TranslateProgram(code, options);
  SR::IR::ResourceSpecialization spec;
  auto result = SR::CompileProgram(std::move(translated), options, spec, 0);
  auto &program = result.program;
  ShaderStageInputInfo input;
  input.compute = &info;
  const auto first = Msl::EmitProgram(program, input).source;
  Check(first.find("uchar* b0;") == std::string::npos,
        "binding model: program without buffers declares none");
  // Same object, same hash: another specialization's resources.
  program.info.buffers.emplace_back();
  program.info.buffers.back().written = true;
  const auto second = Msl::EmitProgram(program, input).source;
  Check(second.find("device uchar* b0;") != std::string::npos,
        "binding model: rebuilt for the next emission at the same address "
        "with the same shader hash");
  program.info.buffers.clear();
  Check(Msl::EmitProgram(program, input).source == first,
        "binding model: emission is a function of its inputs only");
}

// review-fixes v1.1.1: the per-dword fallback of a raw x2..x4 access checks
// each dword in 64 bits from the base address (a uint addr + 4i wraps to a
// small in-range offset for an address just below 2^32).
void TestPerDwordBoundsNoWrap() {
  namespace Mem = Msl::Mem;
  Mem::BufferBinding b;
  b.id = 0;
  b.bounds_check = true;
  Mem::BufferOpts opts;
  opts.elide_bounds = false;
  opts.per_dword_bounds = true;
  Mem::MemOut lo;
  const auto v = Mem::LoadRaw(lo, b, "a", 32, 4, opts);
  const std::string load = lo.code;
  Mem::MemOut so;
  Mem::StoreRaw(so, b, "a", 32, {"x0", "x1", "x2", "x3"}, opts);
  const std::string store = so.code;
  for (const auto &[what, text] : {std::pair<std::string, std::string>{"load", load},
                                   {"store", store}}) {
    Check(text.find("(ulong(a) + 16ul <= ulong(") != std::string::npos,
          "per-dword " + what + ": whole-vector check kept");
    for (const char *k : {"4ul", "8ul", "12ul"})
      Check(text.find(std::string("(ulong(a) + ") + k + " <= ulong(") !=
                std::string::npos,
            "per-dword " + what + ": dword check from the base (" + k + ")");
    Check(text.find("ulong((a + ") == std::string::npos,
          "per-dword " + what + ": no 32-bit wrapped address in a check");
  }
  (void)v;
}

// review-fixes v1.1.1: KYTY_MAD_MODE=fused + a relaxed (kk-bda) pixel shader:
// the relaxed text has no contract(off) pragma left for the fused mode to
// replace (std::string::replace(npos) threw std::out_of_range).
void TestFusedRelaxedPixelPrelude() {
  auto saved = SR::GetCodegenOptions();
  auto fused = saved;
  fused.mad_mode = SR::MadMode::Fused;
  SR::SetCodegenOptions(fused);
  ShaderPixelInputInfo pinfo;
  SR::CompileOptions options;
  options.stage = ShaderType::Pixel;
  options.wave_size = 64;
  options.dump_ir = false;
  options.emit_msl = true;
  options.emit_spirv = false;
  std::vector<uint32_t> ud(16, 0);
  options.user_data = ud;
  options.input_info.pixel = &pinfo;
  setenv("KYTY_MSL_CONVENTION", "kk-bda", 1);
  // Built without exceptions: the old text aborted the process. Emit in a
  // child first, then here.
  const auto emit = [&] {
    const std::vector<uint32_t> code = {0xbf810000}; // s_endpgm
    auto translated = SR::TranslateProgram(code, options);
    SR::IR::ResourceSpecialization spec;
    return SR::CompileProgram(std::move(translated), options, spec, 0).msl;
  };
  int status = -1;
  std::fflush(stdout);
  if (const pid_t pid = fork(); pid == 0) {
    (void)emit();
    std::_Exit(0);
  } else if (pid > 0) {
    waitpid(pid, &status, 0);
  }
  const bool clean = WIFEXITED(status) && WEXITSTATUS(status) == 0;
  const std::string msl = clean ? emit() : std::string();
  unsetenv("KYTY_MSL_CONVENTION");
  SR::SetCodegenOptions(saved);
  Check(clean, "fused + relaxed pixel: emission does not abort");
  Check(msl.find("math relaxed") != std::string::npos &&
            msl.find("contract(off)") == std::string::npos,
        "fused + relaxed pixel: relaxed header, no contract(off)");
}

} // namespace

int main() {
  TestBuilder();
  TestAluGolden();
  TestAluCompiles();
  TestProgram();
  TestLoopCap();
  TestMathSafeAndVoteMerge();
  TestBindingModelPerEmission();
  TestPerDwordBoundsNoWrap();
  TestFusedRelaxedPixelPrelude();
  std::printf("%d checks, %d failed\n", g_checks, g_failed);
  return g_failed == 0 ? 0 : 1;
}
