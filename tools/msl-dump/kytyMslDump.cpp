// kyty_msl_dump: prints the MSL the direct backend (recompiler/backend/msl)
// emits for one shader.
//   kyty_msl_dump --programs <programs.bin> (--index N | --hash HEX) [--list]
//   [--lanes 1|2] [--no-comments] [--loop-cap N (default 262144, 0 = off)]
//   [--out F.metal] [--manifest F.json (T2 binding manifest)]
//   [--mode M (emitter mode of a Metal list row: fix6+vsperf, xf=7 ...;
//   --mode-list)] kyty_msl_dump
//   --bin <code.bin> [--stage cs] [--wave 64] [--threads X,Y,Z] [--ud N]
// programs.bin is Kyty's persistent program cache (source records carry the
// guest code). The translation runs in a forked child (the frontend EXITs on
// unsupported encodings). Game-derived output stays local; never commit it.
#include "graphics/shader/recompiler/CodegenOptions.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/frontend/translate/Translate.h"
#include "graphics/shader/recompiler/backend/msl/MslXform.h"
#include "graphics/shader/recompiler/backend/msl/mslEmitterInternal.h"
#include "graphics/shader/recompiler/ir/ProgramCodec.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shader.h"
#include "kprcLoader.h"

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace Common {
void DbgExitIfHandler(const char *e, const char *, int) {
  std::fprintf(stderr, "EXIT_IF %s\n", e);
  _exit(3);
}
void DbgNotImplementedHandler(const char *e, const char *, int) {
  std::fprintf(stderr, "NOTIMPL %s\n", e);
  _exit(3);
}
void DbgExitHandler(const char *, int, std::string_view t) {
  std::fprintf(stderr, "EXIT %.*s\n", (int)t.size(), t.data());
  _exit(3);
}
void DbgExit(int) { _exit(3); }
} // namespace Common

using namespace Libs::Graphics;
namespace SR = Libs::Graphics::ShaderRecompiler;

namespace {

using Kprc::Perm;
using Kprc::Src;

bool ReadAll(const char *path, std::vector<uint8_t> &d) {
  return Kprc::ReadFile(path, d);
}

// KPRC source records: tools/msl-dump/kprcLoader.h (bounds-checked; parses the SAROS variant word of hand-shader keys).
// A damaged file reports the first bad record on stderr and keeps the records before it.
std::vector<Src> LoadPrograms(const char *path,
                              std::vector<Perm> *perms = nullptr,
                              std::vector<uint8_t> *image = nullptr,
                              size_t *head_len = nullptr) {
  std::vector<uint8_t> local;
  std::vector<uint8_t> &d = image ? *image : local;
  if (!ReadAll(path, d))
    return {};
  std::string err;
  auto out = Kprc::Parse(d, perms, &err, head_len);
  if (!err.empty())
    std::fprintf(stderr, "programs: %s (%zu sources loaded)\n", err.c_str(),
                 out.size());
  return out;
}

// Inverse of BuildStageStaticKey(ShaderComputeInputInfo)
// (graphics/shader/shaderStaticKey.cpp).
void ComputeInfoFromKey(const std::vector<uint32_t> &k,
                        ShaderComputeInputInfo &info) {
  if (k.size() < 14)
    return;
  info.workgroup_register = static_cast<int>(k[0]);
  info.wave_size = k[1] & 0xffu;
  info.float_mode = static_cast<uint8_t>(k[1] >> 8u);
  info.host_subgroup_size =
      32; // Apple simdgroup: not the recorded host's value
  info.thread_ids_num = static_cast<int>(k[3]);
  info.lds_size_dwords = k[4] & 0x7fffffffu;
  info.lds_storage = (k[4] & 0x80000000u) != 0;
  info.scratch_size_dwords = k[5];
  info.dispatch_thread_dimensions = (k[6] & 1u) != 0;
  info.async_compute = (k[6] & 2u) != 0;
  for (int i = 0; i < 3; i++) {
    info.threads_num[i] = k[7 + 2 * i];
    info.group_id[i] = k[8 + 2 * i] != 0;
  }
  info.tg_size_en = k[13] != 0;
}

// Inverse of BuildStageStaticKey(ShaderVertexInputInfo) /
// (ShaderPixelInputInfo). Mesh/tessellation vertex shaders are not decoded.
bool VertexInfoFromKey(const std::vector<uint32_t> &k, ShaderVertexInputInfo &v,
                       bool mesh_ok = false) {
  if (k.size() < 10)
    return false;
  size_t i = 0;
  v.fetch_embedded = k[i++] != 0;
  v.fetch_attrib_reg = static_cast<int>(k[i++]);
  v.fetch_buffer_reg = static_cast<int>(k[i++]);
  v.resources_num = static_cast<int>(k[i++]);
  v.wave_size = k[i++];
  v.scratch_size_dwords = k[i++];
  v.pa_cl_vs_out_cntl = k[i++];
  v.start_instance_sgpr = static_cast<int32_t>(k[i++]);
  if ((k[i] & 0xfff00000u) == 0x1d500000u)
    v.lds_size_dwords = k[i++] & 0xfffffu;
  if ((k[i] & 0xffff0000u) == 0x55c90000u)
    v.user_clip_planes = k[i++] & 0xffffu;
  if (k[i] == 0x7A11A5u) {
    i++;
    v.param_alias_mask = k[i++];
    for (uint32_t d = 0; d < 32; d++)
      if (v.param_alias_mask & (1u << d))
        v.param_alias_source[d] = k[i++];
  }
  v.clip_space.enabled = k[i++] != 0;
  if (v.clip_space.enabled) {
    for (int j = 0; j < 2; j++)
      v.clip_space.scale[j] = std::bit_cast<float>(k[i++]);
    for (int j = 0; j < 2; j++)
      v.clip_space.offset[j] = std::bit_cast<float>(k[i++]);
    for (int j = 0; j < 2; j++)
      v.clip_space.half_extent[j] = std::bit_cast<float>(k[i++]);
  }
  v.mesh.threads_num[0] = k[i++];
  if (v.mesh.threads_num[0] != 0) {
    // meshpath: Kyty's NGG/merged-GS mesh program (ShaderType::Mesh, 5)
    if (!mesh_ok || k.size() < i + 13)
      return false;
    auto &m = v.mesh;
    m.threads_num[1] = k[i++];
    m.threads_num[2] = k[i++];
    m.wave_size = k[i++];
    m.host_subgroup_size = k[i++];
    m.lds_size_dwords = k[i++];
    m.scratch_size_dwords = k[i++];
    m.input_primitive = k[i++];
    m.primitives_per_group = k[i++];
    m.vertices_per_group = k[i++];
    m.max_vertices = k[i++];
    m.max_primitives = k[i++];
    m.provoking_vertex = k[i++];
    m.split_groups = k[i++];
    v.logical_stage = ShaderType::Mesh;
  } else if (mesh_ok) {
    return false; // a mesh source with a plain vertex key
  }
  v.tess.input_control_points = k[i++];
  if (v.tess.input_control_points != 0)
    return false;
  if (v.resources_num < 0 || v.resources_num > ShaderVertexInputInfo::RES_MAX ||
      k.size() < i + 3ull * v.resources_num)
    return false;
  for (int r = 0; r < v.resources_num; r++) {
    v.resources_dst[r].registers_num = static_cast<int>(k[i++]);
    v.resources_dst[r].attr_id = static_cast<int>(k[i++]);
    v.resources[r].fields[3] = k[i++];
  }
  return true;
}

bool PixelInfoFromKey(const std::vector<uint32_t> &k, ShaderPixelInputInfo &p) {
  if (k.size() < 30)
    return false;
  size_t i = 0;
  p.scratch_size_dwords = k[i++];
  p.input_num = k[i++];
  p.wave_size = k[i++];
  p.ps_system_input_base = k[i++];
  if ((k[i] & 0xfff00000u) == 0x1d500000u)
    p.lds_size_dwords = k[i++] & 0xfffffu;
  p.custom_interpolation_mask = k[i++];
  p.ps_perspective_center_vgpr = k[i++];
  p.ps_perspective_centroid_vgpr = k[i++];
  p.ps_perspective_sample_vgpr = k[i++];
  p.ps_linear_sample_vgpr = k[i++];
  p.ps_linear_center_vgpr = k[i++];
  p.ps_linear_centroid_vgpr = k[i++];
  p.ps_pos_x = k[i++];
  p.ps_pos_y = k[i++];
  p.ps_pos_z = k[i++];
  p.ps_pos_w = k[i++];
  {
    const uint32_t f = k[i++];
    p.ps_front_face = f & 1u;
    p.ps_front_face_all_bits = (f >> 1) & 1u;
  }
  p.ps_ancillary = k[i++] != 0;
  {
    const uint32_t f = k[i++];
    p.ps_no_perspective = f & 1u;
    p.parameter_mode = static_cast<ShaderPixelParameterMode>((f >> 8) & 0xffu);
  }
  p.ps_pixel_kill_enable = k[i++];
  p.ps_depth_export_enable = k[i++];
  p.ps_sample_mask_export_enable = k[i++];
  p.ps_early_z = k[i++];
  p.dual_source_blending = k[i++];
  p.alpha_blend_source = static_cast<ShaderAlphaBlendSource>(k[i++]);
  for (int j = 0; j < 8; j++)
    p.target_output_mode[j] = static_cast<uint8_t>(k[i++]);
  if (k.size() < i + 2 + p.input_num)
    return false;
  if (k.size() == i + 3 + p.input_num) {
    p.target_shader_mask = k[i++];
  }
  for (uint32_t base = 0; base < 8; base += 4) {
    const uint32_t w = k[i++];
    for (uint32_t j = 0; j < 4; j++)
      p.target_export_mapping[base + j].packed =
          static_cast<uint8_t>((w >> (j * 8)) & 0xffu);
  }
  for (uint32_t j = 0; j < p.input_num && j < 32; j++)
    p.interpolator_settings[j] = k[i++];
  return true;
}

// nps5 gpuguard: loop budget of the emitted MSL (MslOptions::loop_cap); ON by
// default in this tool (the MSL goes to the GPU test harness, where an
// unbounded loop on synthetic input locked up the GPU, 2026-11-18). --loop-cap
// 0 = off.
uint32_t g_loop_cap = 1u << 18u;

int Translate(const Src &s, const char *out_path, bool comments, uint32_t lanes,
              const char *manifest_path = nullptr, const Perm *perm = nullptr,
              const Src *link_ps = nullptr, const Perm *link_perm = nullptr,
              const char *kk_layout_path = nullptr) {
  ShaderComputeInputInfo info;
  ShaderVertexInputInfo vinfo;
  ShaderPixelInputInfo pinfo;
  ShaderPixelInputInfo link_pinfo;
  SR::IR::CompiledShaderInfo link_program;
  if (s.stage == static_cast<uint32_t>(ShaderType::Compute)) {
    ComputeInfoFromKey(s.statics, info);
  } else if (s.stage == static_cast<uint32_t>(ShaderType::Vertex)) {
    if (!VertexInfoFromKey(s.statics, vinfo)) {
      std::fprintf(stderr, "cannot decode vertex static key (mesh/tess?)\n");
      return 5;
    }
    if (link_ps) {
      if (!PixelInfoFromKey(link_ps->statics, link_pinfo)) {
        std::fprintf(stderr, "cannot decode linked pixel key\n");
        return 5;
      }
      // mslfix: link against the compiled pixel program as the runtime does
      // (pixel.stage.program): its inputs give the flat/smooth aliases and the
      // KYTY_PS_PER_VERTEX=unpack field slots the vertex shader must export.
      ShaderPixelInputInfo ps_info;
      if (!PixelInfoFromKey(link_ps->statics, ps_info)) {
        std::fprintf(stderr, "cannot decode linked pixel key\n");
        return 5;
      }
      SR::CompileOptions po;
      po.stage = ShaderType::Pixel;
      po.wave_size = link_ps->wave;
      po.user_data_base = link_ps->user_data_base;
      po.shader_hash = link_ps->hash;
      po.emit_msl = false;
      po.emit_spirv = false;
      std::vector<uint32_t> ps_ud(std::max(link_ps->user_data_count, 16u), 0u);
      po.user_data = ps_ud;
      po.input_info.pixel = &ps_info;
      auto pt = SR::TranslateProgram(link_ps->code, po);
      if (pt.skip_dispatch) {
        std::fprintf(stderr, "linked pixel shader: skip_dispatch\n");
        return 4;
      }
      SR::IR::ResourceSpecialization
          pspec; // as Translate below: from the program's own resource list
      for (const auto &b : pt.program.info.buffers) {
        SR::IR::ResourceSpecialization::Buffer sb;
        sb.packed_stride = b.packed_stride;
        sb.descriptor_format = b.descriptor_format;
        sb.descriptor_swizzle = b.descriptor_swizzle;
        pspec.buffers.push_back(sb);
      }
      for (const auto &i : pt.program.info.images) {
        SR::IR::ResourceSpecialization::Image si;
        si.numeric_class = i.numeric_class;
        si.dimension = i.dimension;
        si.mip_count = i.mip_count;
        si.conversion_format = i.conversion_format;
        si.shader_swizzle = i.shader_swizzle;
        si.cube = i.cube;
        if (si.numeric_class == Prospero::TextureNumericClass::Unsupported)
          si.numeric_class = Prospero::TextureNumericClass::Float;
        if (si.dimension == SR::Decoder::ImageDimension::Unknown)
          si.dimension = SR::Decoder::ImageDimension::Dim2D;
        pspec.images.push_back(si);
      }
      // bcover: the linked pixel permutation's own recorded specialization, as
      // for the translated program below (the defaults gave a MipStats atomic
      // storage image the float class: "invalid binding class", so every VS
      // linked to such a PS failed).
      // xfclass: part of mode fix6 (bcover); without it the v1.1.2 defaults,
      // byte for byte.
      if (link_perm && !link_perm->spec.empty() &&
          SR::Msl::Xf::EnvOn("KYTY_MSL_FIX6")) {
        SR::IR::ResourceSpecialization ds;
        if (SR::IR::DecodeSpecializationCompat(link_perm->spec, ds))
          pspec = ds;
        else
          std::fprintf(stderr, "warning: spec decode failed for link perm %d\n",
                       link_perm->perm);
      }
      auto pr = SR::CompileProgram(std::move(pt), po, pspec, 0);
      link_program.stage = ShaderType::Pixel;
      link_program.info = pr.program.info;
      link_pinfo.stage.program = &link_program;
      ShaderLinkVertexPixelParameters(vinfo, link_pinfo);
    }
  } else if (s.stage == static_cast<uint32_t>(ShaderType::Mesh)) {
    // meshpath: the mesh program keeps the vertex input info (mesh part).
    if (!VertexInfoFromKey(s.statics, vinfo, true)) {
      std::fprintf(stderr, "cannot decode mesh static key\n");
      return 5;
    }
    // Apple simdgroup, not the recorded host's value (as compute).
    vinfo.mesh.host_subgroup_size = 32;
  } else if (s.stage == static_cast<uint32_t>(ShaderType::Pixel)) {
    if (!PixelInfoFromKey(s.statics, pinfo)) {
      std::fprintf(stderr, "cannot decode pixel static key\n");
      return 5;
    }
  }
  SR::CompileOptions options;
  options.stage = static_cast<ShaderType>(s.stage);
  options.wave_size = s.wave;
  options.user_data_base = s.user_data_base;
  options.shader_hash = s.hash;
  options.dump_ir = std::getenv("KYTY_DUMP_STDOUT") != nullptr;
  options.early_dump = options.dump_ir;
  options.emit_msl = true;
  options.emit_spirv = false;
  options.msl_loop_cap = g_loop_cap;
  std::vector<uint32_t> user_data(std::max(s.user_data_count, 16u), 0u);
  options.user_data = user_data;
  options.back_code = s.back_code; // meshpath: merged GS back half
  if (s.stage == static_cast<uint32_t>(ShaderType::Vertex) ||
      s.stage == static_cast<uint32_t>(ShaderType::Mesh))
    options.input_info.vertex = &vinfo;
  else if (s.stage == static_cast<uint32_t>(ShaderType::Pixel))
    options.input_info.pixel = &pinfo;
  else
    options.input_info.compute = &info;
  auto translated = SR::TranslateProgram(s.code, options);
  if (translated.skip_dispatch) {
    std::fprintf(stderr, "skip_dispatch (unsupported call)\n");
    return 4;
  }
  // Specialization from the program's own resource list (numeric class etc. as
  // translated).
  SR::IR::ResourceSpecialization spec;
  for (const auto &b : translated.program.info.buffers) {
    SR::IR::ResourceSpecialization::Buffer sb;
    sb.packed_stride = b.packed_stride;
    sb.descriptor_format = b.descriptor_format;
    sb.descriptor_swizzle = b.descriptor_swizzle;
    spec.buffers.push_back(sb);
  }
  for (const auto &i : translated.program.info.images) {
    SR::IR::ResourceSpecialization::Image si;
    si.numeric_class = i.numeric_class;
    si.dimension = i.dimension;
    si.mip_count = i.mip_count;
    si.conversion_format = i.conversion_format;
    si.shader_swizzle = i.shader_swizzle;
    si.cube = i.cube;
    // The real descriptors are unknown offline: assume a 2D float image where
    // the translation could not tell.
    if (si.numeric_class == Prospero::TextureNumericClass::Unsupported)
      si.numeric_class = Prospero::TextureNumericClass::Float;
    if (si.dimension == SR::Decoder::ImageDimension::Unknown)
      si.dimension = SR::Decoder::ImageDimension::Dim2D;
    spec.images.push_back(si);
  }
  if (perm && !perm->spec.empty()) {
    SR::IR::ResourceSpecialization ds;
    if (SR::IR::DecodeSpecializationCompat(perm->spec, ds))
      spec = ds;
    else
      std::fprintf(stderr, "warning: spec decode failed for perm %d\n",
                   perm->perm);
  }
  (void)lanes;
  (void)comments;
  // mslfix2: graphics permutations keep their in-game push-data cursor (the
  // pipeline packs the VS push data after the PS's: KK FS/VS read
  // root+672+4*(cursor+k)); compute stays at 0 (v1.1 kernels byte-identical).
  // KYTY_MSL_PERM_CURSOR=0 = old.
  uint32_t cursor = 0;
  if (perm && s.stage != static_cast<uint32_t>(ShaderType::Compute)) {
    const char *e = std::getenv("KYTY_MSL_PERM_CURSOR");
    if (e == nullptr || std::strcmp(e, "0") != 0)
      cursor = perm->cursor;
  }
  auto result =
      SR::CompileProgram(std::move(translated), options, spec, cursor);
  if (kk_layout_path) { // kk_spv_lay layout file from program.bindings (no layout_dump)
    const auto text = SR::Msl::KkLayoutText(result.program);
    std::FILE *lf = std::fopen(kk_layout_path, "wb");
    if (!lf || std::fwrite(text.data(), 1, text.size(), lf) != text.size()) {
      std::fprintf(stderr, "--kk-layout: cannot write %s\n", kk_layout_path);
      return 2;
    }
    std::fclose(lf);
  }
  std::FILE *out = out_path ? std::fopen(out_path, "wb") : stdout;
  std::fwrite(result.msl.data(), 1, result.msl.size(), out);
  if (out_path)
    std::fclose(out);
  if (manifest_path) {
    std::FILE *mf = std::fopen(manifest_path, "wb");
    if (mf) {
      std::fwrite(result.msl_manifest.data(), 1, result.msl_manifest.size(),
                  mf);
      std::fclose(mf);
    }
  }
  std::fprintf(stderr,
               "// complete=%d unsupported:", result.msl_complete ? 1 : 0);
  for (const auto &u : result.msl_unsupported)
    std::fprintf(stderr, " %s", u.c_str());
  std::fprintf(stderr, "\n");
  // A refusal (the emitter will not guess a binding, e.g. a bindless image table) is not a text
  // with holes: exit 5, the reason is the "refused:" entry printed above.
  for (const auto &u : result.msl_unsupported)
    if (u.rfind("refused:", 0) == 0)
      return 5;
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  const char *programs = nullptr, *bin = nullptr, *out = nullptr,
             *manifest = nullptr, *subset = nullptr, *kk_layout = nullptr;
  long index = -1, perm_sel = -1, link_perm = -1;
  bool xf_keys =
      false; // perfauto: print the KYTY_MSL_XF_TABLE key of every source
  bool mode_given = false;

  uint64_t hash = 0;
  bool have_hash = false, list = false, comments = true;
  uint32_t lanes = 0, wave = 64, ud = 16;
  uint32_t stage = static_cast<uint32_t>(ShaderType::Compute),
           threads[3] = {64, 1, 1};
  // reprofix: the lists' texts were made with every DPP write as DppUpdateU32 (Translate.h SetDppAlwaysUpdate);
  // --dpp-upstream emits the runtime's form instead (diagnostics only, its texts match no list row).
  SR::Frontend::SetDppAlwaysUpdate(true);
  for (int i = 1; i < argc; i++) {
    std::string_view a = argv[i];
    auto next = [&]() { return i + 1 < argc ? argv[++i] : ""; };
    if (a == "--programs")
      programs = next();
    else if (a == "--bin")
      bin = next();
    else if (a == "--index")
      index = std::atol(next());
    else if (a == "--hash") {
      hash = std::strtoull(next(), nullptr, 16);
      have_hash = true;
    } else if (a ==
               "--xf-list") { // perfauto: the transform catalogue (MslXform.h)
      std::fputs(SR::Msl::Xf::Catalogue().c_str(), stdout);
      return 0;
    } else if (a == "--xf-keys")
      xf_keys = true;
    else if (a ==
             "--mode-list") { // xfclass: emitter modes (MslXform.h ApplyMode)
      std::fputs(SR::Msl::Xf::ModeCatalogue().c_str(), stdout);
      return 0;
    } else if (a == "--mode") { // xfclass: a Metal list row's emitter mode,
                                // e.g. fix6+vsperf
      if (mode_given) {
        std::fprintf(stderr, "--mode given twice\n");
        return 2;
      }
      mode_given = true;
      std::string err;
      if (!SR::Msl::Xf::ApplyMode(next(), err)) {
        std::fprintf(stderr, "--mode: %s\n", err.c_str());
        return 2;
      }
    } else if (a == "--list")
      list = true;
    else if (a == "--kk-layout")
      kk_layout = next();
    else if (a == "--subset")
      subset = next();
    else if (a == "--perm")
      perm_sel = std::atol(next());
    else if (a == "--link-ps")
      link_perm = std::atol(next());
    else if (a == "--lanes")
      lanes = static_cast<uint32_t>(std::atoi(next()));
    else if (a == "--dpp-upstream")
      SR::Frontend::SetDppAlwaysUpdate(false);
    else if (a == "--no-comments")
      comments = false;
    else if (a == "--loop-cap")
      g_loop_cap = static_cast<uint32_t>(std::strtoul(next(), nullptr, 0));
    else if (a == "--kk-bda")
      setenv("KYTY_MSL_CONVENTION", "kk-bda",
             1); // mslinject: KosmicKrisp binding convention
    else if (a == "--out")
      out = next();
    else if (a == "--manifest")
      manifest = next();
    else if (a == "--wave")
      wave = static_cast<uint32_t>(std::atoi(next()));
    else if (a == "--ud")
      ud = static_cast<uint32_t>(std::atoi(next()));
    else if (a == "--threads")
      std::sscanf(next(), "%u,%u,%u", &threads[0], &threads[1], &threads[2]);
    else if (a == "--stage")
      stage = std::string_view(next()) == "cs"
                  ? static_cast<uint32_t>(ShaderType::Compute)
                  : 0;
    else {
      std::fprintf(stderr, "unknown option %.*s\n", (int)a.size(), a.data());
      return 2;
    }
  }
  // mslfix: raw-vertex pixel reads as on the KosmicKrisp (Metal) host:
  // KYTY_PS_PER_VERTEX=unpack (vulkanWindow.cpp sets
  // SetHostPsPerVertex(barycentric_supported); KK runs the unpack path).
  // Graphics only; compute text unchanged.
  SR::SetHostPsPerVertex(false);
  std::vector<Src> srcs;
  std::vector<Perm> perms;
  const Perm *use_perm = nullptr;
  if (programs) {
    std::vector<uint8_t> image;
    size_t head_len = 0;
    srcs = LoadPrograms(programs, &perms, subset ? &image : nullptr, &head_len);
    if (subset) { // one permutation + its source record as a small KPRC file (readable by --programs OUT --perm 0)
      std::vector<uint8_t> sub;
      std::string e;
      if (perm_sel < 0 ||
          !Kprc::BuildSubset(image, head_len, srcs, perms, static_cast<int>(perm_sel), sub, &e)) {
        std::fprintf(stderr, "--subset: %s\n", perm_sel < 0 ? "needs --perm N" : e.c_str());
        return 2;
      }
      std::FILE *fp = std::fopen(subset, "wb");
      if (!fp || std::fwrite(sub.data(), 1, sub.size(), fp) != sub.size()) {
        std::fprintf(stderr, "--subset: cannot write %s\n", subset);
        if (fp) std::fclose(fp);
        return 2;
      }
      std::fclose(fp);
      std::printf("subset perm=%ld bytes=%zu\n", perm_sel, sub.size());
      return 0;
    }
    if (list) {
      for (size_t i = 0; i < srcs.size(); i++) {
        std::printf("%zu hash=%016" PRIx64 " stage=%u wave=%u words=%zu", i,
                    srcs[i].hash, srcs[i].stage, srcs[i].wave,
                    srcs[i].code.size());
        if (srcs[i].has_variant) // SAROS hand shaders only; plain keys print as before
          std::printf(" variant=%016" PRIx64, srcs[i].variant);
        std::printf("\n");
      }
      return 0;
    }
    if (xf_keys) { // "<vs|ps|cs> <guest hash> <source digest>": the table key
                   // (MslXform.h) per source
      for (const auto &src : srcs) {
        const auto st = static_cast<ShaderType>(src.stage);
        std::printf("%s %016" PRIx64 " ",
                    st == ShaderType::Vertex    ? "vs"
                    : st == ShaderType::Pixel   ? "ps"
                    : st == ShaderType::Compute ? "cs"
                                                : "??",
                    src.hash);
        for (const unsigned char ch : src.digest) // the digest is raw bytes
          std::printf("%02x", ch);
        std::printf("\n");
      }
      return 0;
    }
    long pick = index;
    if (have_hash)
      for (size_t i = 0; i < srcs.size(); i++)
        if (srcs[i].hash == hash) {
          pick = static_cast<long>(i);
          break;
        }
    if (perm_sel >= 0) {
      for (const auto &pm : perms)
        if (pm.perm == perm_sel)
          use_perm = &pm;
      if (!use_perm) {
        std::fprintf(stderr, "no such perm\n");
        return 2;
      }
      for (size_t i = 0; i < srcs.size(); i++)
        if (srcs[i].digest == use_perm->digest) {
          pick = static_cast<long>(i);
          break;
        }
    }
    if (pick < 0 || static_cast<size_t>(pick) >= srcs.size()) {
      std::fprintf(stderr, "no such shader (%zu sources)\n", srcs.size());
      return 2;
    }
    Src s = srcs[static_cast<size_t>(pick)];
    srcs.clear();
    srcs.push_back(std::move(s));
  } else if (bin) {
    std::vector<uint8_t> d;
    if (!ReadAll(bin, d)) {
      std::fprintf(stderr, "cannot read %s\n", bin);
      return 2;
    }
    Src s;
    s.stage = stage;
    s.wave = wave;
    s.user_data_count = ud;
    s.code.resize(d.size() / 4);
    std::memcpy(s.code.data(), d.data(), s.code.size() * 4);
    // statics = BuildStageStaticKey layout for a compute shader with the given
    // threads.
    s.statics = {0, wave | (0xc0u << 8u), 64, 0,          0, 0, 0, threads[0],
                 0, threads[1],           0,  threads[2], 0, 0};
    srcs.push_back(std::move(s));
  } else {
    std::fprintf(
        stderr,
        "usage: kyty_msl_dump --programs <programs.bin> (--index N|--hash "
        "H|--list) | --bin <code.bin> [--threads X,Y,Z --wave 64]\n");
    return 2;
  }
  // The frontend EXITs (handler _exit) on unsupported input: run in a child.
  std::fflush(stdout);
  Src link_src;
  const Src *link_ptr = nullptr;
  Perm link_perm_rec;
  const Perm *link_perm_ptr = nullptr;
  if (link_perm >= 0) {
    // own perm list: reloading into `perms` would free the Perm that use_perm
    // points at (flaky SIGSEGV in DecodeSpecialization of --link-ps runs,
    // mslrest3)
    std::vector<Perm> link_perms;
    auto all = LoadPrograms(programs, &link_perms);
    for (const auto &pm : link_perms)
      if (pm.perm == link_perm)
        for (const auto &x : all)
          if (x.digest == pm.digest) {
            link_src = x;
            link_ptr = &link_src;
            link_perm_rec = pm;
            link_perm_ptr = &link_perm_rec;
          }
    if (!link_ptr) {
      std::fprintf(stderr, "no such link perm\n");
      return 2;
    }
  }
  const pid_t pid = std::getenv("KYTY_MSL_DUMP_NOFORK") ? 0 : fork();
  if (pid == 0) {
    const int rc = Translate(srcs[0], out, comments, lanes, manifest, use_perm,
                             link_ptr, link_perm_ptr, kk_layout);
    std::fflush(stdout);
    _exit(rc);
  }
  int st = 0;
  waitpid(pid, &st, 0);
  return WIFEXITED(st) ? WEXITSTATUS(st) : 1;
}
