// T2 msl-mem unit tests (standalone, no gtest): string-level checks per op
// family, plus every generated kernel is written to <outdir>/*.metal for `xcrun
// metal` compilation (run_msl_mem_tests.sh).
#include "graphics/shader/recompiler/BufferFormat.h"
#include "graphics/shader/recompiler/backend/msl/MslBinding.h"
#include "graphics/shader/recompiler/backend/msl/MslBufferOps.h"
#include "graphics/shader/recompiler/backend/msl/MslImageOps.h"
#include "graphics/shader/recompiler/backend/msl/MslSharedOps.h"
#include "graphics/shader/recompiler/backend/msl/MslWaveOps.h"

#include <cstdio>
#include <fstream>
#include <string>

using namespace Libs::Graphics::ShaderRecompiler::Msl::Mem;
using Libs::Graphics::ShaderRecompiler::Format::GetFormatInfo;
using Prospero_BF = Libs::Graphics::Prospero::BufferFormat;

static int g_fail = 0;
#define CHECK(c)                                                               \
  do {                                                                         \
    if (!(c)) {                                                                \
      std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c);                  \
      g_fail++;                                                                \
    }                                                                          \
  } while (0)

static std::string g_outdir;

static void WriteKernel(const std::string &name, const BindingModel &m,
                        const std::string &body, bool lane = false) {
  std::string s = "#include <metal_stdlib>\nusing namespace metal;\n" +
                  BufferHelpersPrelude() + ArgumentBufferStruct(m);
  s += "kernel void main_entrypoint(\n" + EntryParams(m) +
       ",\n    uint tid [[thread_position_in_grid]],\n    uint lane "
       "[[thread_index_in_simdgroup]]\n)\n{\n" +
       body + "}\n";
  (void)lane;
  std::ofstream(g_outdir + "/" + name + ".metal") << s;
  std::ofstream(g_outdir + "/" + name + ".json") << ManifestJson(m);
}

static BindingModel BaseModel() {
  BindingModel m;
  BufferBinding b0;
  b0.id = 0;
  b0.read = true;
  b0.written = true;
  m.buffers.push_back(b0);
  BufferBinding b1;
  b1.id = 1;
  b1.read = true;
  b1.written = false;
  m.buffers.push_back(b1);
  m.push_bytes = 128;
  m.lds_bytes = 4096;
  return m;
}

static void TestRaw() {
  auto m = BaseModel();
  MemOut o;
  BufferAddrIn in;
  in.index = "tid";
  in.offset = "0u";
  in.imm = 16;
  m.buffers[0].packed_stride = 16;
  auto a = BufferByteAddress(o, m.buffers[0], in);
  CHECK(o.code.find("* 16u") != std::string::npos);
  auto v4 = LoadRaw(o, m.buffers[1], a, 32, 4);
  auto v1 = LoadRaw(o, m.buffers[1], a, 32, 1);
  auto v8 = LoadRaw(o, m.buffers[1], a, 8, 1);
  auto v16 = LoadRaw(o, m.buffers[1], a, 16, 1);
  CHECK(o.code.find("packed_uint4") != std::string::npos);
  CHECK(o.code.find("coherent") == std::string::npos);
  StoreRaw(o, m.buffers[0], a, 32, {v4 + ".x", v4 + ".y"});
  StoreRaw(o, m.buffers[0], a, 8, {v8});
  StoreRaw(o, m.buffers[0], a, 16, {v16});
  StoreRaw(o, m.buffers[0], a, 32, {v1});
  auto rc = ReadConstBuffer(o, m.buffers[1], "64u");
  // unchecked variant has no guard
  BufferOpts nb;
  nb.elide_bounds = true;
  MemOut o2;
  LoadRaw(o2, m.buffers[1], "4u", 32, 1, nb);
  CHECK(o2.code.find("_size") == std::string::npos);
  CHECK(o.code.find("_size") != std::string::npos);
  (void)rc;
  // swizzled addressing
  auto sb = m.buffers[0];
  sb.packed_stride = 16u | (1u << 14u) | (1u << 20u);
  MemOut o3;
  BufferAddrIn sin;
  sin.index = "tid";
  sin.offset = "8u";
  sin.lane = "lane";
  BufferByteAddress(o3, sb, sin);
  CHECK(o3.code.find("& 63u") != std::string::npos);
  CHECK(o.error.empty());
  WriteKernel("t_buffer_raw", m, o.code);
}

static void TestFormats() {
  auto m = BaseModel();
  MemOut o;
  BufferAddrIn in;
  in.index = "tid";
  int n = 0;
  // Every format with a valid info row, from BufferFormat.h.
  for (uint32_t f = 0; f < 128; f++) {
    const auto info = GetFormatInfo(static_cast<Prospero_BF>(f));
    if (info.component_count == 0u)
      continue;
    o.Line("{");
    auto a = o.Def("uint", "tid * 16u");
    auto v = LoadFormatted(o, m.buffers[1], a, f, 0xfac);
    StoreFormatted(o, m.buffers[0], a, f,
                   {v + ".x", v + ".y", v + ".z", v + ".w"});
    o.Line("}");
    n++;
  }
  CHECK(n >= 70);
  CHECK(o.error.empty());
  std::printf("  formats covered: %d\n", n);
  // dst_sel constants: sel 1 -> one bits for float
  MemOut o2;
  LoadFormatted(o2, m.buffers[1], "0u", uint32_t(Prospero_BF::k8_8UNorm),
                (4u) | (5u << 3) | (0u << 6) | (1u << 9));
  CHECK(o2.code.find("0x3f800000u") != std::string::npos);
  MemOut bad;
  LoadFormatted(bad, m.buffers[1], "0u", 0, 0xfac);
  CHECK(!bad.error.empty());
  WriteKernel("t_buffer_formats", m, o.code);
}

static void TestAtomics() {
  auto m = BaseModel();
  MemOut o;
  const AtomicOp ops[] = {AtomicOp::Swap, AtomicOp::Add,  AtomicOp::Sub,
                          AtomicOp::SMin, AtomicOp::UMin, AtomicOp::SMax,
                          AtomicOp::UMax, AtomicOp::And,  AtomicOp::Or,
                          AtomicOp::Xor,  AtomicOp::Inc,  AtomicOp::Dec,
                          AtomicOp::FMin, AtomicOp::FMax, AtomicOp::CmpSwap};
  for (auto op : ops) {
    o.Line("{");
    BufferAtomic32(o, m.buffers[0], "(tid * 4u)", op, "push[0]", "push[1]");
    LdsAtomic32(o, {m.lds_bytes}, "((tid & 63u) * 4u)", op, "push[0]",
                "push[1]");
    o.Line("}");
  }
  for (auto op : {AtomicOp::Add, AtomicOp::Sub, AtomicOp::And, AtomicOp::Or,
                  AtomicOp::Xor, AtomicOp::Swap}) {
    o.Line("{");
    BufferAtomic64(o, m.buffers[0], "(tid * 8u)", op, "ulong(push[0])");
    if (op != AtomicOp::Swap)
      LdsAtomic64(o, {m.lds_bytes}, "((tid & 31u) * 8u)", op, "ulong(push[0])");
    o.Line("}");
  }
  LdsMaskedOr32(o, {m.lds_bytes}, "0u", "push[2]", "push[3]");
  CHECK(o.error.empty());
  CHECK(o.code.find("atomic_compare_exchange_weak_explicit") !=
        std::string::npos);
  MemOut bad;
  BufferAtomic64(bad, m.buffers[0], "0u", AtomicOp::UMax, "1ul");
  CHECK(!bad.error.empty());
  WriteKernel("t_atomics", m, o.code);
}

static void TestLds() {
  auto m = BaseModel();
  MemOut o;
  LdsInfo l{m.lds_bytes};
  auto a = o.Def("uint", "(tid & 255u) * 4u");
  auto v = LdsLoad(o, l, a, 32, 4);
  auto b = LdsLoad(o, l, a, 8, 1);
  auto h = LdsLoad(o, l, a, 16, 1);
  LdsStore(o, l, a, 32, {v + ".x", v + ".y", v + ".z"});
  LdsStore(o, l, a, 8, {b});
  LdsStore(o, l, a, 16, {h});
  LdsBarrier(o);
  CHECK(o.code.find("threadgroup_barrier") != std::string::npos);
  CHECK(o.code.find("packed_uint3") != std::string::npos);
  WriteKernel("t_lds", m, o.code);
}

static void TestWave() {
  auto m = BaseModel();
  MemOut o;
  WaveCtx w;
  auto b = WaveBallot(o, w, "(tid & 1u) != 0u");
  WaveAny(o, w, "(tid & 1u) != 0u");
  auto rf = WaveReadFirstLane(o, w, "tid");
  auto rl = WaveReadLane(o, w, "tid", "5u");
  WaveWriteLane(o, "lane", "tid", rl, "3u");
  DppFlags f;
  for (uint32_t c : {0x1bu, 0xe4u, 0x101u, 0x105u, 0x111u, 0x11fu, 0x121u,
                     0x12fu, 0x140u, 0x141u, 0x150u, 0x15fu, 0x160u, 0x16fu}) {
    f.control = c;
    auto mv = WaveDppMove(o, w, f, "tid", "true");
    WaveDppUpdate(o, w, f, mv, "0u", "true");
    f.bound_control = true;
    f.row_mask = 0x5;
    f.bank_mask = 0xa;
    auto mv2 = WaveDppMove(o, w, f, "tid", "true");
    WaveDppUpdate(o, w, f, mv2, rf, "true");
    f.bound_control = false;
    f.row_mask = 0xf;
    f.bank_mask = 0xf;
  }
  f.dpp8 = true;
  f.control = 0x1b5;
  WaveDppMove(o, w, f, "tid", "true");
  for (uint32_t c : {0x8000u | 0x1bu, 0xc000u | 0x3u, 0xc400u | 0x23u,
                     0x001fu | (0x0u << 5) | (0x1u << 10)})
    WaveSwizzle(o, w, c, "tid", "true");
  WaveBpermute(o, w, "tid", "(tid * 4u)", "true");
  WavePermute(o, w, "tid", "(tid * 4u)", "true");
  CHECK(o.error.empty());
  CHECK(b.size() > 0);
  MemOut bad;
  f.dpp8 = false;
  f.control = 0x130;
  WaveDppMove(bad, w, f, "tid", "true");
  CHECK(!bad.error.empty());
  o.Line("((device uint*)r.b0)[tid] = " + WaveBallot(o, w, "true") + ".x;");
  WriteKernel("t_wave", m, o.code);
}

static void TestImages() {
  BindingModel m = BaseModel();
  SamplerBinding s0;
  s0.id = 0;
  m.samplers.push_back(s0);
  SamplerBinding s1;
  s1.id = 1;
  s1.compare = true;
  m.samplers.push_back(s1);
  struct D {
    ImgDim d;
    bool depth;
  };
  const D dims[] = {{ImgDim::D1, false}, {ImgDim::D1Array, false},
                    {ImgDim::D2, false}, {ImgDim::D2Array, false},
                    {ImgDim::D3, false}, {ImgDim::Cube, false},
                    {ImgDim::D2, true},  {ImgDim::D2Array, true}};
  uint32_t id = 0;
  for (auto &x : dims) {
    ImageBinding t;
    t.id = id++;
    t.dim = x.d;
    t.depth = x.depth;
    t.access = Access::Sample;
    m.images.push_back(t);
  }
  // storage images
  ImageBinding st;
  st.id = id++;
  st.dim = ImgDim::D2;
  st.numeric = Numeric::Float;
  st.access = Access::ReadWrite;
  m.images.push_back(st);
  ImageBinding su;
  su.id = id++;
  su.dim = ImgDim::D2;
  su.numeric = Numeric::Uint;
  su.access = Access::Atomic;
  m.images.push_back(su);
  ImageBinding sa;
  sa.id = id++;
  sa.dim = ImgDim::D2Array;
  sa.numeric = Numeric::Float;
  sa.access = Access::Read;
  m.images.push_back(sa);
  ImageBinding sm;
  sm.id = id++;
  sm.dim = ImgDim::D2Ms;
  sm.numeric = Numeric::Float;
  sm.access = Access::Read;
  m.images.push_back(sm);
  ImageBinding sn;
  sn.id = id++;
  sn.dim = ImgDim::D2;
  sn.numeric = Numeric::Sint;
  sn.access = Access::Sample;
  m.images.push_back(sn);

  MemOut o;
  o.Line("float fx = float(tid) * 0.01f;");
  std::vector<std::string> A = {
      "as_type<uint>(fx)",   "as_type<uint>(fx + 0.1f)", "as_type<uint>(2.0f)",
      "as_type<uint>(0.0f)", "as_type<uint>(1.5f)",      "as_type<uint>(0.25f)",
      "as_type<uint>(0.5f)", "as_type<uint>(0.5f)",      "as_type<uint>(0.5f)",
      "as_type<uint>(0.5f)"};
  const auto run = [&](const char *label, uint32_t imgi, uint32_t flags,
                       bool gather, uint32_t dmask = 0xf, bool pixel = false) {
    ImageOp op;
    op.img = &m.images[imgi];
    op.samp = m.images[imgi].depth && (flags & SampleFlag::Compare)
                  ? &m.samplers[1]
                  : &m.samplers[0];
    op.sample_flags = flags;
    op.dmask = dmask;
    op.pixel_stage = pixel;
    op.address_components = 10;
    o.Line(std::string("// ") + label);
    o.Line("{");
    auto r = ImageSample(o, op, A, gather);
    o.Line("((device uint*)r.b0)[tid] += " + r + ".x;");
    o.Line("}");
  };
  // layout: offset? bias? dref? grads? coord lod clamp. address dwords above
  // are plain floats.
  run("2d implicit", 2, 0, false, 0xf, true);
  run("2d bias", 2, SampleFlag::Bias, false, 0xf, true);
  run("2d lod", 2, SampleFlag::Lod, false);
  run("2d lz", 2, SampleFlag::LevelZero, false);
  run("2d grad", 2, SampleFlag::Derivative, false);
  run("2d clamp", 2, SampleFlag::LodClamp, false, 0xf, true);
  run("2d lod+clamp", 2, SampleFlag::Lod | SampleFlag::LodClamp, false);
  run("2d grad+clamp", 2, SampleFlag::Derivative | SampleFlag::LodClamp, false);
  run("2d dyn offset", 2, SampleFlag::Offset, false, 0xf, true);
  run("1d lod", 0, SampleFlag::Lod, false);
  run("1darray lod", 1, SampleFlag::Lod, false);
  run("2darray bias", 3, SampleFlag::Bias, false, 0xf, true);
  run("3d lod", 4, SampleFlag::Lod, false);
  run("3d grad", 4, SampleFlag::Derivative, false);
  run("cube implicit", 5, 0, false, 0xf, true);
  run("depth cmp", 6, SampleFlag::Compare, false, 0xf, true);
  run("depth cmp lz", 6, SampleFlag::Compare | SampleFlag::LevelZero, false);
  run("depth array cmp", 7, SampleFlag::Compare | SampleFlag::LevelZero, false);
  run("gather r", 2, 0, true, 0x1);
  run("gather g", 2, 0, true, 0x2);
  run("gather b offset", 2, SampleFlag::Offset, true, 0x4);
  run("gather array", 3, 0, true, 0x8);
  run("gather cube", 5, 0, true, 0x1);
  run("gather depth cmp", 6, SampleFlag::Compare, true, 0x1);
  run("int gather", 12, 0, true, 0x1);
  run("int sample", 12, SampleFlag::LevelZero, false);
  {
    ImageOp op;
    op.img = &m.images[2];
    op.samp = &m.samplers[0];
    op.address_components = 10;
    op.data_bits = 16;
    op.dmask = 0x7;
    auto r = ImageSample(o, op, A, false);
    o.Line("((device uint*)r.b0)[tid] += " + r + ".x;");
    ImageOp q = op;
    q.data_bits = 32;
    auto d = ImageQueryDims(o, q, A);
    auto l = ImageQueryLod(o, q, A);
    o.Line("((device uint*)r.b0)[tid] += " + d + ".x + " + l + ".y;");
  }
  // load/store/atomic
  {
    std::vector<std::string> I = {"tid", "tid + 1u", "0u", "0u"};
    ImageOp rw;
    rw.img = &m.images[8];
    rw.address_components = 3;
    rw.has_mip = true;
    rw.dynamic_mip = true;
    auto v = ImageLoad(o, rw, I);
    ImageStore(o, rw, I, v);
    ImageOp at;
    at.img = &m.images[9];
    at.address_components = 2;
    for (auto op :
         {AtomicOp::Add, AtomicOp::Sub, AtomicOp::And, AtomicOp::Or,
          AtomicOp::Xor, AtomicOp::UMin, AtomicOp::UMax, AtomicOp::Swap})
      ImageAtomic32(o, at, I, static_cast<int>(op), "push[0]");
    ImageOp la;
    la.img = &m.images[10];
    la.address_components = 3;
    ImageLoad(o, la, I);
    ImageOp ms;
    ms.img = &m.images[11];
    ms.address_components = 3;
    ImageLoad(o, ms, I);
    MemOut bad;
    ImageAtomic32(bad, at, I, static_cast<int>(AtomicOp::CmpSwap), "1u");
    CHECK(!bad.error.empty());
  }
  CHECK(o.error.empty());
  if (!o.error.empty())
    std::printf("  image error: %s\n", o.error.c_str());
  auto l = ImageAddressComponentLayout(SampleFlag::A16 | SampleFlag::Bias, 1);
  CHECK(l.bit_width == 16 && l.bit_offset == 16);
  WriteKernel("t_image", m, o.code);
}

static void TestManifest() {
  auto m = BaseModel();
  ImageBinding t;
  t.id = 0;
  t.dim = ImgDim::Cube;
  m.images.push_back(t);
  SamplerBinding s;
  s.id = 0;
  m.samplers.push_back(s);
  auto rl = ComputeRootLayout(m);
  // b0 ptr@0 size@8, b1 ptr@16 size@24, t0@32, s0@40 -> 48
  CHECK(rl.buffer_ptr.at(0) == 0 && rl.buffer_size.at(0) == 8 &&
        rl.buffer_ptr.at(1) == 16 && rl.texture.at(0) == 32 &&
        rl.sampler.at(0) == 40 && rl.bytes == 48);
  auto j = ManifestJson(m);
  CHECK(j.find("\"schema\": 1") != std::string::npos);
  CHECK(j.find("\"dim\":\"2darray\"") != std::string::npos &&
        j.find("\"cube\":true") != std::string::npos);
  CHECK(j.find("\"init\":\"hex\"") != std::string::npos);
  CHECK(j.find("{\"kind\":\"buffer\",\"index\":1,\"res\":\"push\"}") !=
        std::string::npos);
  CHECK(j.find("\"tg_memory\":4096") != std::string::npos);
  auto st = ArgumentBufferStruct(m);
  CHECK(st.find("const device uchar* b1;") != std::string::npos);
  CHECK(st.find("texture2d_array<float, access::sample> t0;") !=
        std::string::npos);
}

// Runtime kernels: run through mslh (tests/msl/verify_runtime.py checks outputs
// against numpy).
static void WriteRuntime() {
  // 1) formatted loads: b1 = seeded bytes, b0[tid*4..] = decoded uint4
  const struct {
    const char *n;
    Prospero_BF f;
  } fm[] = {{"8888unorm", Prospero_BF::k8_8_8_8UNorm},
            {"8888snorm", Prospero_BF::k8_8_8_8SNorm},
            {"1616float", Prospero_BF::k16_16Float},
            {"16uint", Prospero_BF::k16UInt},
            {"88sint", Prospero_BF::k8_8SInt},
            {"3232float", Prospero_BF::k32_32Float},
            {"11_11_10float", Prospero_BF::k11_11_10Float},
            {"2_10_10_10unorm", Prospero_BF::k2_10_10_10UNorm},
            {"16161616snorm", Prospero_BF::k16_16_16_16SNorm}};
  for (auto &f : fm) {
    BindingModel m;
    m.threadgroup = {32, 1, 1};
    BufferBinding b0;
    b0.id = 0;
    b0.read = false;
    b0.written = true;
    b0.bytes = 4096;
    m.buffers.push_back(b0);
    BufferBinding b1;
    b1.id = 1;
    b1.bytes = 4096;
    m.buffers.push_back(b1);
    MemOut o;
    auto a = o.Def("uint", "tid * 16u");
    auto v = LoadFormatted(o, m.buffers[1], a, uint32_t(f.f), 0xfac);
    o.Line("*(device uint4*)(r.b0 + tid * 16u) = " + v + ";");
    WriteKernel(std::string("r_fmt_") + f.n, m, o.code);
    std::printf("  fmt %s = %u\n", f.n, uint32_t(f.f));
  }
  // 2) wave ops: out[tid*8+k]
  {
    BindingModel m;
    m.threadgroup = {32, 1, 1};
    BufferBinding b0;
    b0.id = 0;
    b0.read = false;
    b0.written = true;
    b0.bytes = 4096;
    m.buffers.push_back(b0);
    MemOut o;
    WaveCtx w;
    DppFlags f;
    o.Line("uint v = tid * 3u + 1u;");
    auto rl = WaveReadLane(o, w, "v", "5u");
    auto rf = WaveReadFirstLane(o, w, "v");
    auto bl = WaveBallot(o, w, "(tid & 1u) != 0u");
    f.control = 0x111;
    auto d1 = WaveDppUpdate(o, w, f, WaveDppMove(o, w, f, "v", "true"), "999u",
                            "true"); // row_shr:1
    f.control = 0x1b;
    auto d2 = WaveDppMove(o, w, f, "v", "true"); // quad_perm 0,2,1,0
    f.control = 0x141;
    auto d3 = WaveDppMove(o, w, f, "v", "true"); // half mirror
    auto sw = WaveSwizzle(o, w, 0x001f | (0x00 << 5) | (0x01 << 10), "v",
                          "true"); // xor 1
    auto bp = WaveBpermute(o, w, "v", "((31u - lane) * 4u)", "true");
    auto pm = WavePermute(o, w, "v", "(((lane + 1u) & 31u) * 4u)", "true");
    o.Line("device uint* out = (device uint*)(r.b0 + tid * 40u);");
    o.Line("out[0] = " + rl + "; out[1] = " + rf + "; out[2] = " + bl +
           ".x; out[3] = " + d1 + "; out[4] = " + d2 + "; out[5] = " + d3 +
           "; out[6] = " + sw + "; out[7] = " + bp + "; out[8] = " + pm + ";");
    WriteKernel("r_wave", m, o.code);
  }
  // 3) LDS + atomics: out[tid*4+k]
  {
    BindingModel m;
    m.threadgroup = {32, 1, 1};
    m.lds_bytes = 1024;
    BufferBinding b0;
    b0.id = 0;
    b0.read = false;
    b0.written = true;
    b0.atomic = true;
    b0.bytes = 4096;
    b0.init = "zero";
    m.buffers.push_back(b0);
    MemOut o;
    LdsInfo l{m.lds_bytes};
    LdsStore(o, l, "(tid * 4u)", 32, {"tid * 7u"});
    o.Line("if (tid < 12u) { *(threadgroup uint*)((threadgroup uchar*)lds + "
           "512u + tid * 4u) = 0u; }");
    LdsBarrier(o);
    LdsAtomic32(o, l, "512u", AtomicOp::Add, "tid + 1u");
    LdsAtomic32(o, l, "516u", AtomicOp::UMax, "tid * 3u");
    LdsAtomic32(o, l, "520u", AtomicOp::SMin, "uint(-int(tid))");
    LdsAtomic32(o, l, "524u", AtomicOp::Inc, "9u");
    LdsAtomic64(o, l, "528u", AtomicOp::Add, "ulong(tid) << 31");
    LdsBarrier(o);
    auto nb = LdsLoad(o, l, "(((tid + 1u) & 31u) * 4u)", 32, 1);
    auto a = LdsLoad(o, l, "512u", 32, 4);
    auto big = LdsLoad(o, l, "528u", 32, 2);
    auto oob = LdsLoad(o, l, "1022u", 32, 1);
    BufferAtomic32(o, m.buffers[0], "1024u", AtomicOp::Add, "1u");
    BufferAtomic32(o, m.buffers[0], "1028u", AtomicOp::Or, "1u << (tid & 31u)");
    BufferAtomic32(o, m.buffers[0], "1032u", AtomicOp::CmpSwap, "tid + 100u",
                   "0u");
    BufferAtomic64(o, m.buffers[0], "1040u", AtomicOp::Add,
                   "ulong(0xffffffffu)");
    o.Line("device uint* out = (device uint*)(r.b0 + tid * 32u);");
    o.Line("out[0] = " + nb + "; out[1] = " + a + ".x; out[2] = " + a +
           ".y; out[3] = " + a + ".z; out[4] = " + a + ".w; out[5] = " + big +
           ".x; out[6] = " + big + ".y; out[7] = " + oob + ";");
    WriteKernel("r_lds", m, o.code);
  }
}

int main(int argc, char **argv) {
  g_outdir = argc > 1 ? argv[1] : ".";
  if (argc > 2) {
    WriteRuntime();
    return 0;
  }
  TestRaw();
  TestFormats();
  TestAtomics();
  TestLds();
  TestWave();
  TestImages();
  TestManifest();
  std::printf(g_fail ? "FAILED (%d)\n" : "ALL PASSED\n", g_fail);
  return g_fail ? 1 : 0;
}
