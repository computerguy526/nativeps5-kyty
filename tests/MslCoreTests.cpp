// MSL backend core tests (T1): ALU snippets, MslBuilder, whole-program emission
// from hand-assembled RDNA2 code. If `xcrun metal` exists every emitted snippet
// is also compiled (-fsyntax-only). Without arguments runs everything; exit
// code 0 = pass.
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/backend/msl/MslBuilder.h"
#include "graphics/shader/recompiler/backend/msl/MslEmitter.h"
#include "graphics/shader/recompiler/backend/msl/mslEmitterInternal.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

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

// bcover: kk-bda VS liveness of an exported vector shared by two parameter
// exports (e.g. the same uint4 goes to loc 1 .xyz and loc 2 .w). A
// component is dead only when no user reads it; the old rule unioned the
// DROPPED components of both exports and zeroed a live UV (and dropped its
// vertex attribute load, so B's slots no longer matched KosmicKrisp's).
void TestKkLiveComponents() {
  SR::IR::Inst v(ValueOpcode::CompositeConstructU32x4);
  SR::IR::Inst e1(ValueOpcode::SetAttribute), e2(ValueOpcode::SetAttribute),
      other(ValueOpcode::IAdd32);
  e1.SetArg(0, SR::IR::Value(&v));
  e2.SetArg(0, SR::IR::Value(&v));
  auto mask = [&](uint32_t m1, uint32_t m2) {
    return [&, m1, m2](const SR::IR::Inst &u) -> uint32_t {
      return &u == &e1 ? m1 : &u == &e2 ? m2 : 0xfu;
    };
  };
  Check(Msl::KkLiveComponents(v, mask(0x7u, 0x8u)) == 0xfu,
        "live components: .xyz + .w of one vector = all four");
  Check(Msl::KkLiveComponents(v, mask(0x3u, 0x1u)) == 0x3u,
        "live components: .xy + .x = .xy (z, w dead)");
  Check(Msl::KkLiveComponents(v, mask(0x0u, 0x4u)) == 0x4u,
        "live components: a dropped export keeps nothing alive");
  other.SetArg(0, SR::IR::Value(&v));
  Check(Msl::KkLiveComponents(v, mask(0x1u, 0x1u)) == 0xfu,
        "live components: a non-export user reads the whole vector");
}

// fmafix: KosmicKrisp prints every NIR ALU op as its own statement (each in its
// own math_mode(relaxed) block), so Metal's front end never forms an fmuladd
// from Kyty's split V_MAD/V_FMA (ffma_weak(contract) -> fadd(fmul)). Relaxed
// pixel text gives every multiply its own statement too; with the split off the
// mslfix5 one-expression text is kept.
void TestKkStatementSplit() {
  SR::IR::Program p;
  p.block_storage.push_back(std::make_unique<SR::IR::Block>());
  auto *blk = p.block_storage.back().get();
  p.blocks.push_back(blk);
  using SR::IR::Value;
  auto &x = blk->AppendNewInst(ValueOpcode::BitCastU32F32, {Value(1u)});
  auto &y = blk->AppendNewInst(ValueOpcode::BitCastU32F32, {Value(2u)});
  auto &z = blk->AppendNewInst(ValueOpcode::BitCastU32F32, {Value(3u)});
  auto &mad = blk->AppendNewInst(ValueOpcode::FPMad32,
                                 {Value(&x), Value(&y), Value(&z)});
  auto &fma = blk->AppendNewInst(ValueOpcode::FPFma32,
                                 {Value(&x), Value(&y), Value(&z)});
  auto &mul = blk->AppendNewInst(ValueOpcode::FPMul32, {Value(&x), Value(&y)});
  auto &mulc = blk->AppendNewInst(
      ValueOpcode::FPMul32, {Value(&mul), Value::F32(0.5f)});
  auto &add = blk->AppendNewInst(ValueOpcode::FPAdd32, {Value(&x), Value(&z)});
  for (auto *i : {&mad, &fma, &mulc, &add}) {
    auto &u = blk->AppendNewInst(ValueOpcode::FPMax32, {Value(i), Value(i)}); // a use
    auto &bits = blk->AppendNewInst(ValueOpcode::BitCastF32U32, {Value(&u)});
    blk->AppendNewInst(ValueOpcode::ReferenceU32, {Value(&bits)}); // keeps it live
  }
  Msl::MslCtx ctx(p, {});
  uint32_t id = 0;
  for (auto &i : *blk) ctx.ids[&i] = id++;
  Msl::NirReassocBuild(p, true, true);
  Check(Msl::NirSplitLines(ctx, mad) ==
            "{ const float v3_0 = (v0 * v1); v3 = (v3_0 + v2); }",
        "split: V_MAD = fmul statement then fadd: " +
            Msl::NirSplitLines(ctx, mad));
  Check(Msl::NirSplitLines(ctx, fma) ==
            "{ const float v4_0 = (v0 * v1); v4 = (v4_0 + v2); }",
        "split: V_FMA is ffma_weak(contract) in KK too: " +
            Msl::NirSplitLines(ctx, fma));
  Check(Msl::NirSplitLines(ctx, mulc) ==
            "{ const float v6_0 = (v0 * 0.5f); v6 = (v6_0 * v1); }",
        "split: re-associated (x*y)*c -> (x*c)*y one op per statement: " +
            Msl::NirSplitLines(ctx, mulc));
  Check(Msl::NirSplitLines(ctx, add).empty() &&
            Msl::NirReassocText(ctx, add).empty(),
        "split: a plain fadd keeps its usual text");
  Msl::NirReassocBuild(p, true, false);
  Check(Msl::NirSplitLines(ctx, mad).empty() &&
            Msl::NirReassocText(ctx, mulc) == "((v0 * 0.5f) * v1)",
        "split off: mslfix5 one-expression text: " + Msl::NirReassocText(ctx, mulc));
  Msl::NirReassocBuild(p, false, true);
  Check(Msl::NirSplitLines(ctx, mad) ==
                "{ const float v3_0 = (v0 * v1); v3 = (v3_0 + v2); }" &&
            Msl::NirSplitLines(ctx, mulc).empty(),
        "split without re-association: only V_MAD/V_FMA split");
  Msl::NirReassocBuild(p, false, false);
}

// mslfix6: graphics min/max/med3 use the KK-exact helpers (NaN -> other operand,
// -0 < +0, med3 with a NaN operand = min3); compute keeps Metal's builtins.
void TestKkMinMax() {
  const std::vector<std::string> a2 {"a", "b"}, a3 {"a", "b", "c"};
  Msl::SetKkMinMax(true);
  Check(Msl::AluExpression(ValueOpcode::FPMin32, a2) == "kyty_fmin(a, b)" &&
            Msl::AluExpression(ValueOpcode::FPMax32, a2) == "kyty_fmax(a, b)" &&
            Msl::AluExpression(ValueOpcode::FPMinTri32, a3) == "kyty_fmin(kyty_fmin(a, b), c)" &&
            Msl::AluExpression(ValueOpcode::FPMaxTri32, a3) == "kyty_fmax(kyty_fmax(a, b), c)" &&
            Msl::AluExpression(ValueOpcode::FPMedTri32, a3) == "kyty_fmed3(a, b, c)",
        "kk min/max: graphics spellings");
  Msl::SetKkMinMax(false);
  Check(Msl::AluExpression(ValueOpcode::FPMin32, a2) == "fmin(a, b)" &&
            Msl::AluExpression(ValueOpcode::FPMedTri32, a3) == "fmedian3(a, b, c)",
        "kk min/max off: Metal builtins (compute text unchanged)");
}

// mslfix6 (pipe 100/101 rt1): fneg is pulled out of fmul chains before the
// constant moves: m = fmul(fneg(x), y); r = fmul(m, 0.5) is fneg((x*0.5)*y) in
// KK's NIR (the inner x*y is used once after the fneg moves), not -((x*y)*0.5).
void TestReassocFnegChain() {
  SR::IR::Program p;
  p.block_storage.push_back(std::make_unique<SR::IR::Block>());
  auto *blk = p.block_storage.back().get();
  p.blocks.push_back(blk);
  using SR::IR::Value;
  auto &x = blk->AppendNewInst(ValueOpcode::BitCastU32F32, {Value(1u)});
  auto &y = blk->AppendNewInst(ValueOpcode::BitCastU32F32, {Value(2u)});
  auto &nx = blk->AppendNewInst(ValueOpcode::FPNeg32, {Value(&x)});
  auto &m = blk->AppendNewInst(ValueOpcode::FPMul32, {Value(&nx), Value(&y)});
  auto &r = blk->AppendNewInst(ValueOpcode::FPMul32, {Value(&m), Value::F32(0.5f)});
  auto &bits = blk->AppendNewInst(ValueOpcode::BitCastF32U32, {Value(&r)});
  blk->AppendNewInst(ValueOpcode::ReferenceU32, {Value(&bits)});
  Msl::MslCtx ctx(p, {});
  uint32_t id = 0;
  for (auto &i : *blk) ctx.ids[&i] = id++;
  Msl::NirReassocBuild(p, true, true);
  const auto t = Msl::NirSplitLines(ctx, r);
  Check(t == "{ const float v4_0 = (v0 * 0.5f); const float v4_1 = (v4_0 * v1); v4 = (-v4_1); }",
        "reassoc: fneg(fmul) chain then (x*y)*c -> (x*c)*y: " + t);
  Msl::NirReassocBuild(p, false, false);
}

} // namespace

int main() {
  TestBuilder();
  TestAluGolden();
  TestAluCompiles();
  TestProgram();
  TestLoopCap();
  TestMathSafeAndVoteMerge();
  TestKkLiveComponents();
  TestKkStatementSplit();
  TestReassocFnegChain();
  TestKkMinMax();
  std::printf("%d checks, %d failed\n", g_checks, g_failed);
  return g_failed == 0 ? 0 : 1;
}
