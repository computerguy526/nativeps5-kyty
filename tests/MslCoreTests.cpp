// MSL backend core tests (T1): ALU snippets, MslBuilder, whole-program emission
// from hand-assembled RDNA2 code. If `xcrun metal` exists every emitted snippet
// is also compiled (-fsyntax-only). Without arguments runs everything; exit
// code 0 = pass.
#include "graphics/shader/recompiler/CodegenOptions.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/backend/msl/MslBufferOps.h"
#include "graphics/shader/recompiler/backend/msl/MslBuilder.h"
#include "graphics/shader/recompiler/backend/msl/MslEmitter.h"
#include "graphics/shader/recompiler/backend/msl/MslXform.h"
#include "graphics/shader/recompiler/backend/msl/mslEmitterInternal.h"
#include "graphics/shader/recompiler/ir/ProgramCodec.h"
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
    Check(result.msl.find("stage fragment") == std::string::npos &&
              result.msl.find("math relaxed") == std::string::npos,
          tag + ": compute header is not the fragment header (dangling else, "
                "7f9ba790)");
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
  auto &mulc =
      blk->AppendNewInst(ValueOpcode::FPMul32, {Value(&mul), Value::F32(0.5f)});
  auto &add = blk->AppendNewInst(ValueOpcode::FPAdd32, {Value(&x), Value(&z)});
  for (auto *i : {&mad, &fma, &mulc, &add}) {
    auto &u =
        blk->AppendNewInst(ValueOpcode::FPMax32, {Value(i), Value(i)}); // a use
    auto &bits = blk->AppendNewInst(ValueOpcode::BitCastF32U32, {Value(&u)});
    blk->AppendNewInst(ValueOpcode::ReferenceU32,
                       {Value(&bits)}); // keeps it live
  }
  Msl::MslCtx ctx(p, {});
  uint32_t id = 0;
  for (auto &i : *blk)
    ctx.ids[&i] = id++;
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
        "split off: mslfix5 one-expression text: " +
            Msl::NirReassocText(ctx, mulc));
  Msl::NirReassocBuild(p, false, true);
  Check(Msl::NirSplitLines(ctx, mad) ==
                "{ const float v3_0 = (v0 * v1); v3 = (v3_0 + v2); }" &&
            Msl::NirSplitLines(ctx, mulc).empty(),
        "split without re-association: only V_MAD/V_FMA split");
  Msl::NirReassocBuild(p, false, false);
}

// mslfix6: graphics min/max/med3 use the KK-exact helpers (NaN -> other
// operand, -0 < +0, med3 with a NaN operand = min3); compute keeps Metal's
// builtins.
void TestKkMinMax() {
  const std::vector<std::string> a2{"a", "b"}, a3{"a", "b", "c"};
  Msl::SetKkMinMax(true);
  Check(Msl::AluExpression(ValueOpcode::FPMin32, a2) == "kyty_fmin(a, b)" &&
            Msl::AluExpression(ValueOpcode::FPMax32, a2) == "kyty_fmax(a, b)" &&
            Msl::AluExpression(ValueOpcode::FPMinTri32, a3) ==
                "kyty_fmin(kyty_fmin(a, b), c)" &&
            Msl::AluExpression(ValueOpcode::FPMaxTri32, a3) ==
                "kyty_fmax(kyty_fmax(a, b), c)" &&
            Msl::AluExpression(ValueOpcode::FPMedTri32, a3) ==
                "kyty_fmed3(a, b, c)",
        "kk min/max: graphics spellings");
  Check(Msl::AluExpression(ValueOpcode::FPMax32,
                           std::vector<std::string>{"-32767.0f", "x"}) ==
                "fmax(-32767.0f, x)" &&
            Msl::AluExpression(ValueOpcode::FPMin32,
                               std::vector<std::string>{"x", "1.13097346f"}) ==
                "fmin(x, 1.13097346f)" &&
            Msl::AluExpression(ValueOpcode::FPMax32,
                               std::vector<std::string>{"0.0f", "x"}) ==
                "kyty_fmax(0.0f, x)" &&
            Msl::AluExpression(ValueOpcode::FPMax32,
                               std::vector<std::string>{"-0.0f", "x"}) ==
                "kyty_fmax(-0.0f, x)",
        "kk min/max: a non-zero finite constant operand keeps Metal fmin/fmax "
        "(pair 70e345eb), zero uses the helper");
  Msl::SetKkMinMax(false);
  Check(Msl::AluExpression(ValueOpcode::FPMin32, a2) == "fmin(a, b)" &&
            Msl::AluExpression(ValueOpcode::FPMedTri32, a3) ==
                "fmedian3(a, b, c)",
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
  auto &r =
      blk->AppendNewInst(ValueOpcode::FPMul32, {Value(&m), Value::F32(0.5f)});
  auto &bits = blk->AppendNewInst(ValueOpcode::BitCastF32U32, {Value(&r)});
  blk->AppendNewInst(ValueOpcode::ReferenceU32, {Value(&bits)});
  Msl::MslCtx ctx(p, {});
  uint32_t id = 0;
  for (auto &i : *blk)
    ctx.ids[&i] = id++;
  Msl::NirReassocBuild(p, true, true);
  const auto t = Msl::NirSplitLines(ctx, r);
  Check(t == "{ const float v4_0 = (v0 * 0.5f); const float v4_1 = (v4_0 * "
             "v1); v4 = (-v4_1); }",
        "reassoc: fneg(fmul) chain then (x*y)*c -> (x*c)*y: " + t);
  Msl::NirReassocBuild(p, false, false);
}

} // namespace

// perfauto: every transform is listed in the catalogue with a version, and
// nothing is active without KYTY_MSL_XF / KYTY_MSL_XF_TABLE (the default text
// is the base emitter's).
void TestXfCatalogue() {
  namespace Xf = Msl::Xf;
  const std::string cat = Xf::Catalogue();
  for (uint32_t i = 0; i < Xf::Count; i++) {
    const auto &info = Xf::InfoOf(static_cast<Xf::Id>(i));
    Check(cat.find(std::string(info.id) + "@" + std::to_string(info.version)) !=
              std::string::npos,
          std::string("xf catalogue lists ") + info.id);
  }
  Check(cat.find("load_group@1 any semantic-exact") != std::string::npos,
        "xf load_group registered");
  if (std::getenv("KYTY_MSL_XF") == nullptr &&
      std::getenv("KYTY_MSL_XF_TABLE") == nullptr &&
      std::getenv("KYTY_MSL_VS_PERF") == nullptr) {
    for (const ShaderType st :
         {ShaderType::Vertex, ShaderType::Pixel, ShaderType::Compute}) {
      Xf::Begin(st, 0x1234u);
      Check(Xf::ActiveList().empty(), "xf: nothing active by default");
    }
  }
}

// xfclass review M1: the movrel_switch/tree analysis belongs to one emission
// (MslCtx), like the binding model (TestBindingModelPerEmission): the same
// IR::Program object with the same shader hash but other instructions (the next
// specialization at the same address) gets its own chains.
void TestMovrelChainsPerEmission() {
  SR::IR::Program p;
  p.shader_hash = 0x5151u;
  p.block_storage.push_back(std::make_unique<SR::IR::Block>());
  auto *blk = p.block_storage.back().get();
  p.blocks.push_back(blk);
  using SR::IR::Value;
  auto chain = [&](uint32_t seed) {
    auto &m = blk->AppendNewInst(ValueOpcode::BitCastF32U32,
                                 {Value::F32(float(seed))});
    const SR::IR::Inst *prev =
        &blk->AppendNewInst(ValueOpcode::BitCastF32U32, {Value::F32(1.0f)});
    for (uint32_t k = 0; k < 4; k++) {
      auto &c =
          blk->AppendNewInst(ValueOpcode::IEqual32, {Value(&m), Value(k)});
      prev = &blk->AppendNewInst(ValueOpcode::SelectU32,
                                 {Value(&c), Value(100u + k),
                                  Value(const_cast<SR::IR::Inst *>(prev))});
    }
    blk->AppendNewInst(ValueOpcode::ReferenceU32,
                       {Value(const_cast<SR::IR::Inst *>(prev))});
  };
  chain(3u);
  {
    Msl::MslCtx ctx(p, {});
    Check(Msl::MovrelChainHeads(ctx) == 1u, "movrel per emission: one chain");
  }
  chain(5u); // same object, same hash, one more chain
  {
    Msl::MslCtx ctx(p, {});
    Check(Msl::MovrelChainHeads(ctx) == 2u,
          "movrel per emission: a new emission of the same program object sees "
          "its own chains");
  }
}

// foliage2: transform load_select (mode token lsel). Off: the checked load is the old conditional text, byte for byte.
// On (kk-bda safe address set): the load itself is unconditional and its address falls back to the safe expression when
// the range check fails, so a failing check never dereferences buf + addr; stores and the per-dword form are untouched.
void TestLoadSelect() {
  namespace Mem = Msl::Mem;
  namespace Xf = Msl::Xf;
  Mem::BufferBinding b;
  b.id = 3;
  b.bounds_check = true;
  Mem::BufferOpts opts;
  opts.elide_bounds = false;
  auto gen = [&](uint32_t bits, uint32_t comps) { Mem::MemOut o; const auto v = Mem::LoadRaw(o, b, "a", bits, comps, opts); return o.code + "=> " + v; };
  auto genc = [&] { Mem::MemOut o; const auto v = Mem::ReadConstBuffer(o, b, "a", opts); return o.code + "=> " + v; };
  auto gens = [&] { Mem::MemOut o; Mem::StoreRaw(o, b, "a", 32, {"x0"}, opts); return o.code; };
  for (const char *v : {"KYTY_MSL_XF", "KYTY_MSL_XF_AUTO", "KYTY_MSL_XF_LSEL", "KYTY_MSL_VS_PERF"}) unsetenv(v);
  Xf::Begin(ShaderType::Vertex, 0);
  Mem::SetLoadSelectSafe({});
  const std::string off1 = gen(32, 1), off4 = gen(32, 4), off16 = gen(16, 1), offc = genc(), offs = gens();
  Check(!Xf::On(Xf::LoadSelect) && off1.find(" ? *(") != std::string::npos && off1.find("kk_lsafe") == std::string::npos,
        "load_select off: conditional load text");
  setenv("KYTY_MSL_XF_LSEL", "1", 1);
  Xf::Begin(ShaderType::Vertex, 0);
  Check(Xf::On(Xf::LoadSelect), "load_select: on with KYTY_MSL_XF_LSEL (vertex)");
  Check(gen(32, 1) == off1 && genc() == offc, "load_select: no safe address (not kk-bda) -> text unchanged");
  Mem::SetLoadSelectSafe("kk_lsafe");
  const std::string on1 = gen(32, 1), on4 = gen(32, 4), on16 = gen(16, 1), onc = genc();
  for (const auto &[what, t] : {std::pair<std::string, std::string>{"dword", on1}, {"vec4", on4}, {"ushort", on16}, {"const", onc}}) {
    Check(t.find(" ? (ulong)(r.b3 + a) : kk_lsafe;") != std::string::npos, "load_select " + what + ": address falls back to the safe address when the check is false");
    Check(t.find("*)(r.b3 + a)") == std::string::npos, "load_select " + what + ": no dereference of buf + addr outside the selected address");
    Check(t.find(" ? *(") == std::string::npos, "load_select " + what + ": the load is not inside the conditional");
    Check(t.find("(0u);\n=> ") != std::string::npos || t.find(" : 0u;\n=> ") != std::string::npos, "load_select " + what + ": out of range still yields 0");
  }
  Check(on4.find("uint4(*(") != std::string::npos && on4.find("packed_uint4*)(") != std::string::npos, "load_select vec4: packed vector load kept");
  Check(gens() == offs, "load_select: stores unchanged");
  { Mem::BufferOpts pd = opts; pd.per_dword_bounds = true; Mem::MemOut o; (void)Mem::LoadRaw(o, b, "a", 32, 4, pd);
    Check(o.code.find("kk_lsafe") == std::string::npos, "load_select: per-dword partial form unchanged"); }
  { Mem::BufferOpts el = opts; el.elide_bounds = true; Mem::MemOut o; (void)Mem::LoadRaw(o, b, "a", 32, 1, el);
    Check(o.code.find("kk_lsafe") == std::string::npos, "load_select: unchecked load unchanged"); }
  Xf::Begin(ShaderType::Pixel, 0);
  Check(Xf::On(Xf::LoadSelect), "load_select: any stage (pixel)");
  unsetenv("KYTY_MSL_XF_LSEL");
  Xf::Begin(ShaderType::Vertex, 0);
  Check(gen(32, 1) == off1 && gen(32, 4) == off4 && gen(16, 1) == off16 && genc() == offc, "load_select off again: byte-identical text");
  Mem::SetLoadSelectSafe({});
  std::string err;
  Check(Xf::ApplyMode("fix6+lsel", err) && std::string(std::getenv("KYTY_MSL_XF_LSEL")) == "1", "mode: lsel sets KYTY_MSL_XF_LSEL");
  for (const char *v : {"KYTY_MSL_FIX6", "KYTY_MSL_XF_LSEL"}) unsetenv(v);
  Check(Xf::ApplyMode("fix6+xf=7+lsel", err) && std::string(std::getenv("KYTY_MSL_XF")) == "vs_root_int,vs_fetch_select,vs_no_barrier", "mode: lsel combines with xf=HEX");
  for (const char *v : {"KYTY_MSL_FIX6", "KYTY_MSL_XF", "KYTY_MSL_XF_LSEL"}) unsetenv(v);
  setenv("KYTY_MSL_XF_LSEL", "1", 1);
  Check(!Xf::ApplyMode("fix6", err), "mode: refuses a leftover KYTY_MSL_XF_LSEL");
  unsetenv("KYTY_MSL_XF_LSEL");
}

// foltrans: transforms vs_load_group (mode token vgroup) and vs_const_load (mode token cload). Vertex stage only, not in `all`.
// cload off: old text byte for byte. On: a read-only buffer's load is `constant` before the first buffer store/atomic of the
// program, `alias ? device : constant` after it; written buffers, stores and coherent loads are untouched.
void TestFoltrans() {
  namespace Mem = Msl::Mem;
  namespace Xf = Msl::Xf;
  Mem::BufferBinding b, w;
  b.id = 3; b.bounds_check = true;
  w.id = 5; w.bounds_check = true; w.atomic = true;
  Mem::BufferOpts opts;
  auto gen = [&](const Mem::BufferBinding &x, uint32_t bits, uint32_t comps) { Mem::MemOut o; const auto v = Mem::LoadRaw(o, x, "a", bits, comps, opts); return o.code + "=> " + v; };
  auto genc = [&] { Mem::MemOut o; const auto v = Mem::ReadConstBuffer(o, b, "a", opts); return o.code + "=> " + v; };
  auto gens = [&] { Mem::MemOut o; Mem::StoreRaw(o, w, "a", 32, {"x0"}, opts); return o.code; };
  auto gena = [&] { Mem::MemOut o; (void)Mem::BufferAtomic32(o, w, "a", Mem::AtomicOp::UMax, "x0", "", opts); return o.code; };
  for (const char *v : {"KYTY_MSL_XF", "KYTY_MSL_XF_LSEL", "KYTY_MSL_XF_VGROUP", "KYTY_MSL_XF_CLOAD", "KYTY_MSL_VS_PERF"}) unsetenv(v);
  Xf::Begin(ShaderType::Vertex, 0);
  Mem::SetLoadSelectSafe({});
  Mem::SetConstLoad(0, {});
  const std::string off1 = gen(b, 32, 1), off4 = gen(b, 32, 4), offc = genc(), offw = gen(w, 32, 1), offs = gens(), offa = gena();
  Check(!Xf::On(Xf::VsConstLoad) && !Xf::On(Xf::VsLoadGroup), "foltrans: both transforms off by default");
  Check(off1.find("constant") == std::string::npos && Mem::SrtLoad("4u") == "r.srt[4u]", "cload off: no constant load, SRT indexed");
  const std::string cat = Xf::Catalogue();
  Check(cat.find("vs_load_group@1 vs semantic-exact") != std::string::npos && cat.find("vs_const_load@1 vs preconditioned") != std::string::npos, "foltrans: both transforms registered (vertex)");
  setenv("KYTY_MSL_XF_CLOAD", "1", 1);
  setenv("KYTY_MSL_XF_VGROUP", "1", 1);
  Xf::Begin(ShaderType::Vertex, 0);
  Check(Xf::On(Xf::VsConstLoad) && Xf::On(Xf::VsLoadGroup) && !Xf::On(Xf::LoadGroup), "foltrans: tokens switch the vertex transforms on (not load_group)");
  Check(gen(b, 32, 1) == off1 && genc() == offc, "cload: state 0 (not kk-bda / GDS) -> text unchanged");
  Xf::Begin(ShaderType::Pixel, 0);
  Check(!Xf::On(Xf::VsConstLoad) && !Xf::On(Xf::VsLoadGroup), "foltrans: vertex stage only (pixel off)");
  Xf::Begin(ShaderType::Compute, 0);
  Check(!Xf::On(Xf::VsConstLoad) && !Xf::On(Xf::VsLoadGroup), "foltrans: vertex stage only (compute off)");
  Xf::Begin(ShaderType::Vertex, 0);
  Mem::SetConstLoad(1, "kk_calias");
  const std::string c1 = gen(b, 32, 1), c4 = gen(b, 32, 4), cc = genc();
  for (const auto &[what, t] : {std::pair<std::string, std::string>{"dword", c1}, {"vec4", c4}, {"const", cc}}) {
    Check(t.find("*(constant ") != std::string::npos && t.find("*)((ulong)(r.b3 + a))") != std::string::npos, "cload " + what + ": constant load through the integer address");
    Check(t.find("device") == std::string::npos && t.find("kk_calias") == std::string::npos, "cload " + what + ": before a store/atomic no device load and no alias select");
    Check(t.find("(ulong(a) + ") != std::string::npos, "cload " + what + ": range check kept");
  }
  Check(gen(w, 32, 1) == offw, "cload: a written/atomic buffer's load unchanged");
  Check(Mem::SrtLoad("4u") == "(*(constant uint*)((ulong)(r.srt + 4u)))", "cload: SRT read constant");
  { Mem::BufferOpts co = opts; co.coherent_loads = true; Mem::MemOut o; (void)Mem::LoadRaw(o, b, "a", 32, 1, co);
    Check(o.code.find("constant") == std::string::npos, "cload: coherent load unchanged"); }
  Check(Mem::ConstLoadState() == 1, "cload: state 1 before a writer");
  Check(gena() == offa && Mem::ConstLoadState() == 2, "cload: atomic text unchanged, state 1 -> 2");
  const std::string g1 = gen(b, 32, 1);
  Check(g1.find("(kk_calias ? *(const device uint*)((ulong)(r.b3 + a)) : *(constant uint*)((ulong)(r.b3 + a)))") != std::string::npos, "cload: after the atomic the load is alias ? device : constant");
  Mem::SetConstLoad(1, "kk_calias");
  Check(gens() == offs && Mem::ConstLoadState() == 2, "cload: store text unchanged, state 1 -> 2");
  Mem::SetConstLoad(1, "kk_calias");
  Mem::SetLoadSelectSafe("kk_lsafe");
  setenv("KYTY_MSL_XF_LSEL", "1", 1);
  Xf::Begin(ShaderType::Vertex, 0);
  const std::string l1 = gen(b, 32, 1);
  Check(l1.find(" ? (ulong)(r.b3 + a) : kk_lsafe;") != std::string::npos && l1.find("= *(constant uint*)(") != std::string::npos && l1.find("device") == std::string::npos, "cload + lsel: unconditional constant load from the selected address");
  for (const char *v : {"KYTY_MSL_XF_LSEL", "KYTY_MSL_XF_VGROUP", "KYTY_MSL_XF_CLOAD"}) unsetenv(v);
  Mem::SetLoadSelectSafe({});
  Mem::SetConstLoad(0, {});
  Xf::Begin(ShaderType::Vertex, 0);
  Check(gen(b, 32, 1) == off1 && gen(b, 32, 4) == off4 && genc() == offc && gens() == offs && gena() == offa, "foltrans off again: byte-identical text");
  std::string err;
  Check(Xf::ApplyMode("fix6+vsperf+vgroup+cload+lsel", err) && std::string(std::getenv("KYTY_MSL_XF_VGROUP")) == "1" && std::string(std::getenv("KYTY_MSL_XF_CLOAD")) == "1", "mode: vgroup / cload set their switches");
  for (const char *v : {"KYTY_MSL_FIX6", "KYTY_MSL_VS_PERF", "KYTY_MSL_XF_LSEL", "KYTY_MSL_XF_VGROUP", "KYTY_MSL_XF_CLOAD"}) unsetenv(v);
  Check(Xf::ApplyMode("fix6+xf=c00", err) && std::string(std::getenv("KYTY_MSL_XF")) == "vs_load_group,vs_const_load", "mode: xf=c00 names the two transforms");
  for (const char *v : {"KYTY_MSL_FIX6", "KYTY_MSL_XF"}) unsetenv(v);
  Check(!Xf::ApplyMode("fix6+xf=8000", err), "mode: xf bit beyond the registry refused");
  for (const char *v : {"KYTY_MSL_FIX6", "KYTY_MSL_XF"}) unsetenv(v);
  setenv("KYTY_MSL_XF_CLOAD", "1", 1);
  Check(!Xf::ApplyMode("fix6", err), "mode: refuses a leftover KYTY_MSL_XF_CLOAD");
  unsetenv("KYTY_MSL_XF_CLOAD");
  setenv("KYTY_MSL_XF_VGROUP", "1", 1);
  Check(!Xf::ApplyMode("fix6", err), "mode: refuses a leftover KYTY_MSL_XF_VGROUP");
  unsetenv("KYTY_MSL_XF_VGROUP");
}

// fsslow: transform loop_cont (mode token lcont). A while-style loop (exit at the header) whose
// continue block holds phis and one side-effect-free instruction: off = the do { } while (false)
// wrapper with its exit flag (old text byte for byte); on = no wrapper and no flag, the exit is a
// plain break and the continue block is printed once per edge into it, followed by `continue;`.
// Any stage. (KYTY_MSL_XF is read once per process, so `all` cannot be exercised here.)
void TestLoopCont() {
  namespace Xf = Msl::Xf;
  // s_mov s10,0 | L: v_cmp; s_and s10,s10,vcc; s_cbranch_scc0 exit; s_cbranch_scc1 latch;
  // s_mov s10,0 | latch: s_add_u32 s10,s10,s10; s_branch L | exit: v_mov; s_endpgm
  const std::vector<uint32_t> code = {0xbe8a0480, 0x7d8a0080, 0x888a6a0a, 0xbf840004, 0xbf850001,
                                      0xbe8a0480, 0x800a0a0a, 0xbf82fff9, 0x7e020281, 0xbf810000};
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
  auto gen = [&] {
    auto translated = SR::TranslateProgram(code, options);
    SR::IR::ResourceSpecialization spec;
    auto result = SR::CompileProgram(std::move(translated), options, spec, 0);
    return result.msl_structured ? result.msl : std::string("<dispatcher>");
  };
  auto count = [](const std::string &t, const std::string &w) {
    int n = 0;
    for (size_t p = t.find(w); p != std::string::npos; p = t.find(w, p + 1)) n++;
    return n;
  };
  for (const char *v : {"KYTY_MSL_XF", "KYTY_MSL_XF_LCONT", "KYTY_MSL_VS_PERF"}) unsetenv(v);
  const std::string off = gen();
  if (std::getenv("FSSLOW_DUMP")) std::printf("%s\n", off.c_str());
  Check(off.find("do {") != std::string::npos && off.find("while (false);") != std::string::npos &&
            off.find("bool lf0 = false;") != std::string::npos && off.find("if (lf0) break;") != std::string::npos,
        "loop_cont off: do/while(false) wrapper with the exit flag");
  setenv("KYTY_MSL_XF_LCONT", "1", 1);
  const std::string on = gen();
  if (std::getenv("FSSLOW_DUMP")) std::printf("%s\n", on.c_str());
  Xf::Begin(ShaderType::Pixel, 0);
  Check(Xf::On(Xf::LoopCont), "loop_cont: on with KYTY_MSL_XF_LCONT, any stage (pixel)");
  Check(on.find("do {") == std::string::npos && on.find("while (false)") == std::string::npos &&
            on.find("lf0") == std::string::npos,
        "loop_cont on: no wrapper, no flag");
  Check(on.find("for (;;)") != std::string::npos && count(on, "break;") == count(off, "break;") - 1 - count(on, "continue;") + count(off, "continue;"),
        "loop_cont on: each continue edge is `continue;`, the exit a plain break");
  // the continue block's instructions (s_add_u32 = the IAddCarry32 pair) once per edge into it
  Check(count(off, "// IAddCarry32") == 2 && count(on, "// IAddCarry32") == 2 * count(on, "continue;") && count(on, "continue;") == 2,
        "loop_cont on: the continue block is printed at every edge into it");
  if (HaveMetal()) {
    int i = 0;
    for (const std::string *t : {&off, &on}) {
      const std::string f = "/tmp/msl_core_loopcont_" + std::to_string(i++) + ".metal";
      std::ofstream(f) << *t;
      Check(std::system(("xcrun -sdk macosx metal -std=metal3.1 -fsyntax-only " + f + " 2>" + f + ".err").c_str()) == 0,
            "loop_cont: text compiles (see " + f + ".err)");
    }
  }
  unsetenv("KYTY_MSL_XF_LCONT");
  Check(gen() == off, "loop_cont off again: byte-identical text");
  std::string err;
  Check(Xf::ApplyMode("fix6+lcont", err) && std::string(std::getenv("KYTY_MSL_XF_LCONT")) == "1", "mode: lcont sets KYTY_MSL_XF_LCONT");
  for (const char *v : {"KYTY_MSL_FIX6", "KYTY_MSL_XF_LCONT"}) unsetenv(v);
  Check(Xf::ApplyMode("fix6+xf=1000", err) && std::string(std::getenv("KYTY_MSL_XF")) == "loop_cont", "mode: xf=1000 = loop_cont");
  for (const char *v : {"KYTY_MSL_FIX6", "KYTY_MSL_XF"}) unsetenv(v);
  setenv("KYTY_MSL_XF_LCONT", "1", 1);
  Check(!Xf::ApplyMode("fix6", err), "mode: refuses a leftover KYTY_MSL_XF_LCONT");
  unsetenv("KYTY_MSL_XF_LCONT");
}

// fsslow: transform bounds_lim (mode token blim): the checked-load predicate as one compare against a per-buffer limit.
// Off = old text; on = `(addr) < (size >= N ? size - (N-1) : 0)`; same truth table as the base form on the edge grid.
void TestBoundsLim() {
  namespace Mem = Msl::Mem;
  namespace Xf = Msl::Xf;
  Mem::BufferBinding b;
  b.id = 3;
  b.bounds_check = true;
  Mem::BufferOpts opts;
  opts.elide_bounds = false;
  auto gen = [&](uint32_t bits, uint32_t comps) { Mem::MemOut o; const auto v = Mem::LoadRaw(o, b, "a", bits, comps, opts); return o.code + "=> " + v; };
  for (const char *v : {"KYTY_MSL_XF", "KYTY_MSL_XF_BLIM", "KYTY_MSL_XF_LSEL", "KYTY_MSL_VS_PERF"}) unsetenv(v);
  Xf::Begin(ShaderType::Pixel, 0);
  Mem::SetLoadSelectSafe({});
  const std::string off1 = gen(32, 1), off4 = gen(32, 4);
  Check(!Xf::On(Xf::BoundsLim) && off1.find("(ulong(a) + 4ul <= ulong(r.b3_size))") != std::string::npos, "bounds_lim off: base 64-bit predicate");
  setenv("KYTY_MSL_XF_BLIM", "1", 1);
  for (const ShaderType st : {ShaderType::Vertex, ShaderType::Pixel, ShaderType::Compute}) {
    Xf::Begin(st, 0);
    Check(Xf::On(Xf::BoundsLim), "bounds_lim: on with KYTY_MSL_XF_BLIM, any stage");
  }
  const std::string on1 = gen(32, 1), on4 = gen(32, 4), on16 = gen(16, 1);
  Check(on1.find("((a) < (r.b3_size >= 4u ? r.b3_size - 3u : 0u))") != std::string::npos && on1.find("ulong(a)") == std::string::npos, "bounds_lim dword: one compare, no 64-bit sum");
  Check(on4.find("((a) < (r.b3_size >= 16u ? r.b3_size - 15u : 0u))") != std::string::npos, "bounds_lim vec4: 16-byte limit");
  Check(on16.find("((a) < (r.b3_size >= 2u ? r.b3_size - 1u : 0u))") != std::string::npos, "bounds_lim ushort: 2-byte limit");
  unsetenv("KYTY_MSL_XF_BLIM");
  Xf::Begin(ShaderType::Pixel, 0);
  Check(gen(32, 1) == off1 && gen(32, 4) == off4, "bounds_lim off again: byte-identical text");
  bool same = true;
  const uint32_t edge[] = {0u, 1u, 2u, 3u, 4u, 5u, 15u, 16u, 17u, 0x7fffffffu, 0x80000000u, 0xfffffff0u, 0xfffffffbu, 0xfffffffcu, 0xfffffffdu, 0xfffffffeu, 0xffffffffu};
  for (const uint32_t n : {1u, 2u, 4u, 8u, 12u, 16u})
    for (const uint32_t size : edge)
      for (const uint32_t a : edge) {
        const bool base = uint64_t(a) + n <= uint64_t(size);
        const bool lim = a < (size >= n ? size - (n - 1u) : 0u);
        same = same && base == lim;
      }
  Check(same, "bounds_lim: same truth table as the base predicate on the 32-bit edge grid");
  std::string err;
  Check(Xf::ApplyMode("fix6+lcont+blim", err) && std::string(std::getenv("KYTY_MSL_XF_BLIM")) == "1" && std::string(std::getenv("KYTY_MSL_XF_LCONT")) == "1", "mode: blim sets KYTY_MSL_XF_BLIM, combines with lcont");
  for (const char *v : {"KYTY_MSL_FIX6", "KYTY_MSL_XF_BLIM", "KYTY_MSL_XF_LCONT"}) unsetenv(v);
  Check(Xf::ApplyMode("fix6+xf=2000", err) && std::string(std::getenv("KYTY_MSL_XF")) == "bounds_lim", "mode: xf=2000 = bounds_lim");
  for (const char *v : {"KYTY_MSL_FIX6", "KYTY_MSL_XF"}) unsetenv(v);
  setenv("KYTY_MSL_XF_BLIM", "1", 1);
  Check(!Xf::ApplyMode("fix6", err), "mode: refuses a leftover KYTY_MSL_XF_BLIM");
  unsetenv("KYTY_MSL_XF_BLIM");
}

// fsslow: transform guard_exit (mode token gexit): the gpuguard budget check inside the loop's own exit test.
// Off = the old `if (!nps5_lg_step(..)) break;` first statement; on = no such statement, the step call is part of the
// header's if; without a loop cap the transform prints nothing.
void TestGuardExit() {
  namespace Xf = Msl::Xf;
  const std::vector<uint32_t> code = {0xbe8a0480, 0x7d8a0080, 0x888a6a0a, 0xbf840004, 0xbf850001,
                                      0xbe8a0480, 0x800a0a0a, 0xbf82fff9, 0x7e020281, 0xbf810000};
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
  auto gen = [&](uint32_t cap) {
    options.msl_loop_cap = cap;
    auto translated = SR::TranslateProgram(code, options);
    SR::IR::ResourceSpecialization spec;
    auto result = SR::CompileProgram(std::move(translated), options, spec, 0);
    return result.msl_structured ? result.msl : std::string("<dispatcher>");
  };
  auto count = [](const std::string &t, const std::string &w) {
    int n = 0;
    for (size_t p = t.find(w); p != std::string::npos; p = t.find(w, p + 1)) n++;
    return n;
  };
  for (const char *v : {"KYTY_MSL_XF", "KYTY_MSL_XF_GEXIT", "KYTY_MSL_XF_LCONT", "KYTY_MSL_VS_PERF"}) unsetenv(v);
  const std::string off = gen(1u << 18u), off0 = gen(0);
  Check(count(off, "if (!nps5_lg_step(nps5_lg_n)) break; // nps5 gpuguard") == 1, "guard_exit off: guard is the first statement of the loop body");
  for (const bool lcont : {false, true}) {
    setenv("KYTY_MSL_XF_GEXIT", "1", 1);
    if (lcont) setenv("KYTY_MSL_XF_LCONT", "1", 1);
    const std::string tag = lcont ? "guard_exit+loop_cont" : "guard_exit";
    const std::string on = gen(1u << 18u);
    Xf::Begin(ShaderType::Pixel, 0);
    Check(Xf::On(Xf::GuardExit), tag + ": on with KYTY_MSL_XF_GEXIT, any stage (pixel)");
    Check(count(on, ")) break;") == 0 && count(on, "nps5_lg_step(nps5_lg_n)") == 1 &&
              (count(on, ") || !nps5_lg_step(nps5_lg_n)) {") == 1 || count(on, ") && nps5_lg_step(nps5_lg_n)) {") == 1),
          tag + ": one budget check, inside the loop's exit test");
    Check(on.find("bool nps5_lg_step(thread uint& n)") != std::string::npos && on.find("uint nps5_lg_n = 0u;") != std::string::npos, tag + ": helper and counter kept");
    if (!lcont) Check(gen(0) == off0, "guard_exit: nothing printed without a loop cap");
    if (HaveMetal()) {
      const std::string f = std::string("/tmp/msl_core_guardexit_") + (lcont ? "1" : "0") + ".metal";
      std::ofstream(f) << on;
      Check(std::system(("xcrun -sdk macosx metal -std=metal3.1 -fsyntax-only " + f + " 2>" + f + ".err").c_str()) == 0, tag + ": text compiles (see " + f + ".err)");
    }
    unsetenv("KYTY_MSL_XF_GEXIT");
    unsetenv("KYTY_MSL_XF_LCONT");
  }
  Check(gen(1u << 18u) == off, "guard_exit off again: byte-identical text");
  options.msl_loop_cap = 0;
  std::string err;
  Check(Xf::ApplyMode("fix6+lcont+blim+gexit", err) && std::string(std::getenv("KYTY_MSL_XF_GEXIT")) == "1", "mode: gexit sets KYTY_MSL_XF_GEXIT, combines with lcont+blim");
  for (const char *v : {"KYTY_MSL_FIX6", "KYTY_MSL_XF_BLIM", "KYTY_MSL_XF_LCONT", "KYTY_MSL_XF_GEXIT"}) unsetenv(v);
  Check(Xf::ApplyMode("fix6+xf=4000", err) && std::string(std::getenv("KYTY_MSL_XF")) == "guard_exit", "mode: xf=4000 = guard_exit");
  for (const char *v : {"KYTY_MSL_FIX6", "KYTY_MSL_XF"}) unsetenv(v);
  Check(!Xf::ApplyMode("fix6+xf=8000", err), "mode: xf bit beyond the registry refused (15 transforms)");
  for (const char *v : {"KYTY_MSL_FIX6", "KYTY_MSL_XF"}) unsetenv(v);
  setenv("KYTY_MSL_XF_GEXIT", "1", 1);
  Check(!Xf::ApplyMode("fix6", err), "mode: refuses a leftover KYTY_MSL_XF_GEXIT");
  unsetenv("KYTY_MSL_XF_GEXIT");
}

// xfclass: emitter modes of the Metal list rows (kyty_msl_dump --mode). Runs
// last: it sets env vars.
void TestEmitterModes() {
  namespace Xf = Msl::Xf;
  const char *vars[] = {"KYTY_MSL_FIX6",       "KYTY_MSL_VS_PERF",
                        "KYTY_MSL_GFX_SPLIT",  "KYTY_MSL_GFX_REASSOC",
                        "KYTY_MSL_GFX_MINMAX", "KYTY_MSL_GFX_MATH",
                        "KYTY_MSL_XF"};
  auto clear = [&] {
    for (const char *v : vars)
      unsetenv(v);
  };
  auto env = [](const char *v) {
    const char *e = std::getenv(v);
    return std::string(e ? e : "<unset>");
  };
  std::string err;
  clear();
  Check(!Msl::ModeFix6(), "mode: fix6 off by default (v1.1.2 text)");
  Check(Xf::ApplyMode("base", err) && Xf::ApplyMode("", err) &&
            env("KYTY_MSL_FIX6") == "<unset>",
        "mode: base sets nothing");
  Check(Xf::ApplyMode("fix6+perfr+vsperf", err) && Msl::ModeFix6() &&
            env("KYTY_MSL_VS_PERF") == "1" &&
            env("KYTY_MSL_GFX_SPLIT") == "0" &&
            env("KYTY_MSL_GFX_REASSOC") == "0" &&
            env("KYTY_MSL_GFX_MINMAX") == "0",
        "mode: vspr = fix6+perfr+vsperf");
  clear();
  Check(Xf::ApplyMode("fix6+xf=117", err) &&
            env("KYTY_MSL_XF") == "vs_root_int,vs_fetch_select,vs_no_barrier,"
                                  "minmax_select,movrel_tree",
        "mode: xf=117 = perfauto pax117");
  clear();
  Check(Xf::ApplyMode("xf=0", err) && env("KYTY_MSL_XF") == "none",
        "mode: xf=0 = pax0 (none)");
  clear();
  Check(Xf::ApplyMode("gfxmath=fast", err) &&
            env("KYTY_MSL_GFX_MATH") == "fast",
        "mode: gfxmath");
  clear();
  for (const char *bad :
       {"fix7", "fix6+fix6", "xf=zz", "xf=fffff", "gfxmath=turbo",
        "fix6++vsperf", "vsperf=1", "FIX6", "perfr", "perfr+vsperf"}) {
    err.clear();
    Check(!Xf::ApplyMode(bad, err) && !err.empty(),
          std::string("mode: refused ") + bad);
    clear();
  }
  setenv("KYTY_MSL_VS_PERF", "0", 1);
  Check(!Xf::ApplyMode("vsperf", err),
        "mode: never mixes with a hand-set switch");
  clear();
  for (const char *left :
       {"KYTY_MSL_XF_TABLE", "KYTY_MSL_LOADGROUP", "KYTY_MSL_GFX_FTZ",
        "KYTY_MSL_GFX_MATH"}) { // review L1
    setenv(left, "1", 1);
    err.clear();
    Check(!Xf::ApplyMode("fix6", err) && err.find(left) != std::string::npos &&
              !Xf::ApplyMode("base", err),
          std::string("mode: refuses a leftover ") + left);
    unsetenv(left);
  }
  setenv("KYTY_MSL_FIX6", "", 1); // review L3: one truthiness rule
  Check(!Msl::ModeFix6() && !Xf::EnvOn("KYTY_MSL_FIX6"),
        "mode: an empty value is off");
  clear();
  Check(Xf::ModeCatalogue().find("fix6") != std::string::npos &&
            Xf::ModeCatalogue().find("movrel_tree@1") != std::string::npos,
        "mode: catalogue");
}

// emitfix: kyty_msl_dump reads the specialization blobs of a program cache
// written before ResourceSpecialization::Image::indirect_group existed (image
// record four bytes shorter). The strict decoder refuses them; the tool then
// used the translation's own resource list, whose buffer strides are 0, and
// printed `index * 0u` for every indexed buffer read.
void TestSpecializationCompat() {
  namespace IR = SR::IR;
  IR::ResourceSpecialization spec;
  IR::ResourceSpecialization::Buffer b0, b1;
  b0.packed_stride = 16;
  b1.packed_stride = 704;
  b1.zero_stride_oob = true;
  spec.buffers = {b0, b1};
  IR::ResourceSpecialization::Image i0, i1;
  i0.mip_count = 7;
  i0.cube = true;
  i1.shader_swizzle = 0x1234u;
  i1.indirect_root = 3;
  i1.indirect_mapping_offset = 40;
  i1.indirect_search_iterations = 5;
  i1.fmask = true;
  spec.images = {i0, i1};
  std::vector<uint8_t> cur;
  IR::EncodeSpecialization(spec, cur);
  // the same value in the old layout, written field by field
  std::vector<uint8_t> old;
  {
    IR::CodecWriter w(old);
    w.U32(static_cast<uint32_t>(spec.buffers.size()));
    for (const auto &b : spec.buffers) {
      w.U32(b.packed_stride);
      w.U32(static_cast<uint32_t>(b.descriptor_format));
      w.U32(b.descriptor_swizzle);
      w.Bool(b.zero_stride_oob);
    }
    w.U32(static_cast<uint32_t>(spec.images.size()));
    for (const auto &i : spec.images) {
      w.U32(static_cast<uint32_t>(i.numeric_class));
      w.U32(static_cast<uint32_t>(i.dimension));
      w.U32(i.mip_count);
      w.U32(static_cast<uint32_t>(i.conversion_format));
      w.U32(i.shader_swizzle);
      w.U32(i.indirect_root);
      w.U32(i.indirect_mapping_offset);
      w.U32(i.indirect_search_iterations);
      w.Bool(i.cube);
      w.Bool(i.fmask);
    }
  }
  Check(old.size() + 4 * spec.images.size() == cur.size(),
        "spec compat: an old image record is four bytes shorter");
  IR::ResourceSpecialization back;
  bool legacy = true;
  Check(IR::DecodeSpecializationCompat(cur, back, &legacy) && !legacy &&
            back == spec,
        "spec compat: the current layout decodes as before");
  IR::ResourceSpecialization strict;
  Check(!IR::DecodeSpecialization(old, strict),
        "spec compat: the strict decoder still refuses the old layout");
  IR::ResourceSpecialization oldback;
  legacy = false;
  Check(IR::DecodeSpecializationCompat(old, oldback, &legacy) && legacy &&
            oldback == spec && oldback.buffers[1].packed_stride == 704 &&
            oldback.images[1].indirect_group ==
                IR::ImageResource::NoIndirectImage,
        "spec compat: the old layout decodes, buffer strides kept, no image "
        "group");
  // skipfix: the replay re-encodes a legacy blob in the current layout (what a draw publishes), so
  // permutation identities match modules built today.
  {
    std::vector<uint8_t> reenc;
    IR::EncodeSpecialization(oldback, reenc);
    Check(reenc == cur, "spec compat: a legacy blob re-encodes to the current bytes");
  }
  IR::ResourceSpecialization keep = spec;
  for (size_t cut : {size_t{1}, size_t{3}, old.size() / 2}) {
    std::vector<uint8_t> t(old.begin(), old.end() - static_cast<long>(cut));
    Check(!IR::DecodeSpecializationCompat(t, keep) && keep == spec,
          "spec compat: a truncated blob fails and leaves the value alone");
  }
  std::vector<uint8_t> more = old;
  more.push_back(0);
  Check(!IR::DecodeSpecializationCompat(more, keep),
        "spec compat: trailing bytes fail");
  IR::ResourceSpecialization noimg;
  noimg.buffers = {b0};
  std::vector<uint8_t> nb;
  IR::EncodeSpecialization(noimg, nb);
  legacy = true;
  Check(
      IR::DecodeSpecializationCompat(nb, back, &legacy) && !legacy &&
          back == noimg,
      "spec compat: without images both layouts are the same bytes (current)");
}

// emitfix: vertex attribute liveness as NIR computes it (mode token vsadce). A
// guest vertex shader keeps a masked-off value alive in the IR (`cond ? new :
// old` inside the branch taken only when cond holds); NIR's opt_if folds such a
// select, the old value dies and with it the attribute load that only fed it,
// so KosmicKrisp numbers its attribute slots without it.
void TestVsLiveAdce() {
  namespace IR = SR::IR;
  namespace CFG = SR::CFG;
  using O = ValueOpcode;
  const auto U = [](uint32_t v) { return IR::Value(v); };
  struct Built {
    IR::Program p;
    const IR::Inst *a0 = nullptr, *a1 = nullptr, *a2 = nullptr, *sel = nullptr,
                   *exp = nullptr, *exp2 = nullptr, *late = nullptr;
  };
  // b0: a0, a1, a2 = attributes 0..2; c = a0 != 0; d = a0 != 1; cond =
  // `make_cond`
  //     branch cond ? b1 : b2
  // b1: sel = select(`sel_cond`, a0, a1)      (true arm of the branch)
  // b2: (false arm)
  // b3: phi(b1: sel, b2: a0); late = select(c, phi, a2); export 0 = (late, 0,
  // 0, 0);
  //     export 1 = (a2, 0, 0, 0)
  enum Shape { NotC, AndCD, OrCD, PlainC };
  auto build = [&](Shape shape, bool sel_on_d, bool sel_in_false_arm,
                   Built &o) {
    auto &p = o.p;
    for (int i = 0; i < 4; i++) {
      p.block_storage.push_back(std::make_unique<IR::Block>());
      p.blocks.push_back(p.block_storage.back().get());
    }
    auto *b0 = p.blocks[0], *b1 = p.blocks[1], *b2 = p.blocks[2],
         *b3 = p.blocks[3];
    auto &a0 = b0->AppendNewInst(O::GetAttribute, {U(0), U(0)});
    auto &a1 = b0->AppendNewInst(O::GetAttribute, {U(1), U(0)});
    auto &a2 = b0->AppendNewInst(O::GetAttribute, {U(2), U(0)});
    auto &c = b0->AppendNewInst(O::INotEqual32, {IR::Value(&a0), U(0)});
    auto &d = b0->AppendNewInst(O::INotEqual32, {IR::Value(&a0), U(1)});
    IR::Inst *cond = &c;
    if (shape == NotC) {
      auto &any = b0->AppendNewInst(O::AnyLane, {IR::Value(&c)});
      cond = &b0->AppendNewInst(O::LogicalNot, {IR::Value(&any)});
    } else if (shape == AndCD) {
      cond = &b0->AppendNewInst(O::LogicalAnd, {IR::Value(&c), IR::Value(&d)});
    } else if (shape == OrCD) {
      cond = &b0->AppendNewInst(O::LogicalOr, {IR::Value(&c), IR::Value(&d)});
    }
    auto *arm = sel_in_false_arm ? b2 : b1;
    auto &sel =
        arm->AppendNewInst(O::SelectU32, {IR::Value(sel_on_d ? &d : &c),
                                          IR::Value(&a0), IR::Value(&a1)});
    auto &phi = b3->AppendNewInst(O::Phi);
    phi.SetFlags(IR::Type::U32); // a phi's type lives in its flags
    phi.AddPhiOperand(b1, sel_in_false_arm ? IR::Value(&a0) : IR::Value(&sel));
    phi.AddPhiOperand(b2, sel_in_false_arm ? IR::Value(&sel) : IR::Value(&a0));
    auto &late = b3->AppendNewInst(
        O::SelectU32, {IR::Value(&c), IR::Value(&phi), IR::Value(&a2)});
    auto &v = b3->AppendNewInst(O::CompositeConstructU32x4,
                                {IR::Value(&late), U(0), U(0), U(0)});
    auto &e =
        b3->AppendNewInst(O::SetAttribute, {IR::Value(&v), IR::Value(true)});
    auto &v2 = b3->AppendNewInst(O::CompositeConstructU32x4,
                                 {IR::Value(&a2), U(0), U(0), U(0)});
    auto &e2 =
        b3->AppendNewInst(O::SetAttribute, {IR::Value(&v2), IR::Value(true)});
    b0->AddBranch(b1);
    b0->AddBranch(b2);
    b1->AddBranch(b3);
    b2->AddBranch(b3);
    p.block_info.resize(4);
    for (uint32_t i = 0; i < 4; i++)
      p.block_info[i].id = 10 + i;
    p.block_info[0].terminator.kind = CFG::TerminatorKind::ConditionalBranch;
    p.block_info[0].terminator.true_block = 11;
    p.block_info[0].terminator.false_block = 12;
    p.block_info[0].condition = IR::Value(cond);
    p.block_info[1].terminator.kind = CFG::TerminatorKind::Branch;
    p.block_info[1].terminator.true_block = 13;
    p.block_info[2].terminator.kind = CFG::TerminatorKind::Branch;
    p.block_info[2].terminator.true_block = 13;
    o.a0 = &a0, o.a1 = &a1, o.a2 = &a2, o.sel = &sel, o.exp = &e, o.exp2 = &e2,
    o.late = &late;
  };
  const std::set<const IR::Inst *> none;
  const std::set<std::pair<const IR::Inst *, size_t>> nouse;
  {
    Built t; // if (!any(c)) { select(c, a0, a1) }: c is false there, a0 is the
             // dead arm
    build(NotC, false, false, t);
    const auto live = Msl::VsLiveAdce(t.p, none, nouse);
    Check(live.contains(t.a1) && live.contains(t.sel) && live.contains(t.a2) &&
              live.contains(t.exp),
          "vsadce: !c arm keeps the false operand of select(c, ..), exports "
          "are roots");
    // a0 stays live: the branch condition and the merge phi read it
    Check(live.contains(t.a0),
          "vsadce: a value the live branch condition reads stays live");
  }
  {
    Built t; // if (!any(c)) {} else { select(c, a0, a1) }: c is true there, a1
             // dies
    build(NotC, false, true, t);
    const auto live = Msl::VsLiveAdce(t.p, none, nouse);
    Check(!live.contains(t.a1) && live.contains(t.a0) && live.contains(t.sel),
          "vsadce: in the else arm of if (!c) the select on c drops its false "
          "operand");
  }
  {
    Built t; // if (c && d) { select(d, a0, a1) }: d is true there, a1 dies
    build(AndCD, true, false, t);
    Check(!Msl::VsLiveAdce(t.p, none, nouse).contains(t.a1),
          "vsadce: both operands of a true && are true in the then arm");
    Built f; // else arm of (c && d): nothing known about d
    build(AndCD, true, true, f);
    Check(Msl::VsLiveAdce(f.p, none, nouse).contains(f.a1),
          "vsadce: the else arm of && knows neither operand");
  }
  {
    Built t; // if (c || d) {} else { select(d, a0, a1) }: d is false there, a0
             // is the dead arm
    build(OrCD, true, true, t);
    const auto live = Msl::VsLiveAdce(t.p, none, nouse);
    Check(live.contains(t.a1),
          "vsadce: both operands of a false || are false in the else arm");
    Built f; // then arm of (c || d): nothing known about d
    build(OrCD, true, false, f);
    Check(Msl::VsLiveAdce(f.p, none, nouse).contains(f.a1),
          "vsadce: the then arm of || knows neither operand");
  }
  {
    Built
        t; // the merge block is outside both arms: select(c, phi, a2) keeps a2
    build(PlainC, false, false, t);
    const auto live = Msl::VsLiveAdce(t.p, none, nouse);
    Check(live.contains(t.a2) && live.contains(t.late) && !live.contains(t.a1),
          "vsadce: no fact outside the arms; if (c) { select(c, a0, a1) } "
          "drops a1");
  }
  {
    Built t; // dropped exports are no roots; an operand in dead_use is not
             // followed
    build(PlainC, false, false, t);
    const std::set<const IR::Inst *> dropped{t.exp2};
    const std::set<std::pair<const IR::Inst *, size_t>> du{{t.late, 2}};
    const auto live = Msl::VsLiveAdce(t.p, dropped, du);
    Check(!live.contains(t.a2) && !live.contains(t.exp2) &&
              live.contains(t.exp),
          "vsadce: a dropped export and a dead operand keep nothing alive");
    const std::set<const IR::Inst *> all{t.exp, t.exp2};
    Check(Msl::VsLiveAdce(t.p, all, nouse).empty(),
          "vsadce: no kept export, nothing live");
  }
  // mode token and off switch
  namespace Xf = Msl::Xf;
  std::string err;
  unsetenv("KYTY_MSL_KK_VS_ADCE");
  unsetenv("KYTY_MSL_FIX6");
  Check(Xf::ApplyMode("fix6+vsadce", err) &&
            std::string(std::getenv("KYTY_MSL_KK_VS_ADCE")) == "1",
        "mode: vsadce sets KYTY_MSL_KK_VS_ADCE");
  unsetenv("KYTY_MSL_FIX6");
  Check(!Xf::ApplyMode("fix6", err) &&
            err.find("KYTY_MSL_KK_VS_ADCE") != std::string::npos,
        "mode: refuses a leftover KYTY_MSL_KK_VS_ADCE");
  unsetenv("KYTY_MSL_KK_VS_ADCE");
  Check(Xf::ModeCatalogue().find("vsadce") != std::string::npos,
        "mode: catalogue lists vsadce");
}

// onereg T2: PS_UINT32 / SINT export modes. A raw 32-bit MRT export of a pixel program with
// target_output_mode 7 (UINT) / 10 (SINT) declares a uint4 / int4 colour output and stores bit
// casts of the exported registers; mode 0 stays float4 (text unchanged).
void TestPixelOutputModes() {
  const std::vector<uint32_t> code = {0xf800080fu, 0x03020100u, // exp mrt0 v0, v1, v2, v3 done
                                      0xbf810000u};             // s_endpgm
  const auto emit = [&](uint32_t mode) {
    ShaderPixelInputInfo pinfo;
    pinfo.target_output_mode[0] = mode;
    SR::CompileOptions options;
    options.stage = ShaderType::Pixel;
    options.wave_size = 64;
    options.dump_ir = false;
    options.emit_msl = true;
    options.emit_spirv = false;
    std::vector<uint32_t> ud(16, 0);
    options.user_data = ud;
    options.input_info.pixel = &pinfo;
    auto translated = SR::TranslateProgram(code, options);
    SR::IR::ResourceSpecialization spec;
    return SR::CompileProgram(std::move(translated), options, spec, 0).msl;
  };
  const auto m0 = emit(0), m7 = emit(7), m10 = emit(10);
  Check(!m0.empty() && !m7.empty() && !m10.empty(), "pixel output modes: MSL emitted");
  Check(m0.find("float4 c0 [[color(0)]]") != std::string::npos &&
            m0.find("uint4 c0") == std::string::npos && m0.find("int4 c0") == std::string::npos,
        "pixel output modes: mode 0 stays float4");
  Check(m7.find("uint4 c0 [[color(0)]]") != std::string::npos &&
            m7.find("float4 c0") == std::string::npos,
        "pixel output modes: mode 7 (UINT) declares uint4");
  Check(m10.find("int4 c0 [[color(0)]]") != std::string::npos &&
            m10.find("uint4 c0") == std::string::npos && m10.find("float4 c0") == std::string::npos,
        "pixel output modes: mode 10 (SINT) declares int4");
  Check(m7.find("as_type<") != std::string::npos || m7.find("uint4(") != std::string::npos,
        "pixel output modes: mode 7 stores a bit cast / uint4 value");
  Check(m10.find("as_type<int") != std::string::npos || m10.find("int4(") != std::string::npos,
        "pixel output modes: mode 10 stores a bit cast / int4 value");
}

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
  TestKkLiveComponents();
  TestKkStatementSplit();
  TestReassocFnegChain();
  TestKkMinMax();
  TestXfCatalogue();
  TestMovrelChainsPerEmission();
  TestLoadSelect();
  TestFoltrans();
  TestLoopCont();
  TestBoundsLim();
  TestGuardExit();
  TestEmitterModes();
  TestPixelOutputModes();
  TestSpecializationCompat();
  TestVsLiveAdce();
  std::printf("%d checks, %d failed\n", g_checks, g_failed);
  return g_failed == 0 ? 0 : 1;
}
