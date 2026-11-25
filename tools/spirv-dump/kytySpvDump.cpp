// kyty_spv_dump (keyfit): the SPIR-V the SPIR-V back end emits for one permutation of a program cache under THIS process's env
//   (KYTY_* codegen words are read from the environment exactly as the title does), so a key-moving word can be applied offline.
//   kyty_spv_dump --programs <programs.bin> --perm N --out F.spv      one permutation
//   kyty_spv_dump --programs <programs.bin> --all DIR [--stage cs|all]  every permutation -> DIR/<perm>.spv (rc per perm in DIR/index.tsv)
// Older header of the tool this file started as (kyty_msl_dump, same loader):
//   kyty_msl_dump --programs <programs.bin> (--index N | --hash HEX) [--list] [--lanes 1|2] [--no-comments] [--loop-cap N (default 262144, 0 = off)] [--out F.metal] [--manifest F.json (T2 binding manifest)]
//   kyty_msl_dump --bin <code.bin> [--stage cs] [--wave 64] [--threads X,Y,Z] [--ud N]
// programs.bin is Kyty's persistent program cache (source records carry the guest code). The
// translation runs in a forked child (the frontend EXITs on unsupported encodings). Game-derived
// output stays local; never commit it.
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/recompiler/ir/ProgramCodec.h"
#include "graphics/shader/shader.h"

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
int DbgExitIfHandler(const char* e, const char*, int) { std::fprintf(stderr, "EXIT_IF %s\n", e); _exit(3); }
int DbgNotImplementedHandler(const char* e, const char*, int) { std::fprintf(stderr, "NOTIMPL %s\n", e); _exit(3); }
int DbgExitHandler(const char*, int, std::string_view t) { std::fprintf(stderr, "EXIT %.*s\n", (int)t.size(), t.data()); _exit(3); }
int DbgExitHandler(const char*, int, fmt::text_style, std::string_view t) { std::fprintf(stderr, "EXIT %.*s\n", (int)t.size(), t.data()); _exit(3); }
void DbgExit(int) { _exit(3); }
} // namespace Common

using namespace Libs::Graphics;
namespace SR = Libs::Graphics::ShaderRecompiler;

namespace {

struct Src {
	uint32_t stage = 0, wave = 64, user_data_count = 0, user_data_base = 0;
	uint64_t hash = 0;
	std::vector<uint32_t> statics, code;
	std::string digest;
};
struct Perm {
	int perm = 0;
	std::string digest;
	std::vector<uint8_t> spec;
};

bool ReadAll(const char* path, std::vector<uint8_t>& d) {
	std::FILE* fp = std::fopen(path, "rb");
	if (!fp) return false;
	uint8_t buf[1 << 16];
	for (size_t n; (n = std::fread(buf, 1, sizeof(buf), fp)) > 0;) d.insert(d.end(), buf, buf + n);
	std::fclose(fp);
	return true;
}

// KPRC source records (programDiskCache.cpp BuildSourceKey layout), as work/mslnative/tools/corpus_dec.cpp reads them.
std::vector<Src> LoadPrograms(const char* path, std::vector<Perm>* perms = nullptr) {
	std::vector<uint8_t> d;
	std::vector<Src>     out;
	if (!ReadAll(path, d) || d.size() < 32) return out;
	size_t   off = 16;
	uint64_t n;
	std::memcpy(&n, &d[off], 8);
	off += 8 + n + 8;
	int perm_idx = 0;
	while (off + 24 <= d.size()) {
		uint32_t magic, kind;
		uint64_t size;
		std::memcpy(&magic, &d[off], 4);
		std::memcpy(&kind, &d[off + 4], 4);
		std::memcpy(&size, &d[off + 8], 8);
		if (magic != 0x4352504bu) break;
		const size_t p = off + 24, end = p + size;
		if (kind == 1) {
			size_t   q = p + 16;
			uint32_t kl;
			std::memcpy(&kl, &d[q], 4);
			q += 4;
			const uint8_t* key = &d[q];
			Src            s;
			s.digest.assign(reinterpret_cast<const char*>(&d[p]), 16);
			uint32_t       lay, ns, nc;
			std::memcpy(&lay, key, 4);
			std::memcpy(&s.stage, key + 4, 4);
			std::memcpy(&s.hash, key + 8, 8);
			std::memcpy(&s.user_data_count, key + 16, 4);
			size_t k = 24;
			std::memcpy(&ns, key + k, 4);
			s.statics.resize(ns);
			std::memcpy(s.statics.data(), key + k + 4, 4ull * ns);
			k += 4 + 4ull * ns;
			std::memcpy(&s.wave, key + k, 4);
			std::memcpy(&s.user_data_base, key + k + 4, 4);
			k += 8 + 1;
			std::memcpy(&nc, key + k, 4);
			k += 4;
			s.code.resize(nc);
			std::memcpy(s.code.data(), key + k, 4ull * nc);
			out.push_back(std::move(s));
		}
		if (kind != 1) {
			if (perms) {
				Perm pm;
				pm.perm = perm_idx;
				pm.digest.assign(reinterpret_cast<const char*>(&d[p]), 16);
				uint32_t sl;
				std::memcpy(&sl, &d[p + 20], 4);
				pm.spec.assign(&d[p + 24], &d[p + 24] + sl);
				perms->push_back(std::move(pm));
			}
			perm_idx++;
		}
		off = p + size;
	}
	return out;
}

// Inverse of BuildStageStaticKey(ShaderComputeInputInfo) (graphics/shader/shaderStaticKey.cpp).
void ComputeInfoFromKey(const std::vector<uint32_t>& k, ShaderComputeInputInfo& info) {
	if (k.size() < 14) return;
	info.workgroup_register     = static_cast<int>(k[0]);
	info.wave_size              = k[1] & 0xffu;
	info.float_mode             = static_cast<uint8_t>(k[1] >> 8u);
	info.host_subgroup_size     = 32; // Apple simdgroup: not the recorded host's value
	info.thread_ids_num         = static_cast<int>(k[3]);
	info.lds_size_dwords        = k[4] & 0x7fffffffu;
	info.lds_storage            = (k[4] & 0x80000000u) != 0;
	info.scratch_size_dwords    = k[5];
	info.dispatch_thread_dimensions = (k[6] & 1u) != 0;
	info.async_compute          = (k[6] & 2u) != 0;
	for (int i = 0; i < 3; i++) {
		info.threads_num[i] = k[7 + 2 * i];
		info.group_id[i]    = k[8 + 2 * i] != 0;
	}
	info.tg_size_en = k[13] != 0;
}

// Inverse of BuildStageStaticKey(ShaderVertexInputInfo) / (ShaderPixelInputInfo). Mesh/tessellation vertex shaders are not decoded.
bool VertexInfoFromKey(const std::vector<uint32_t>& k, ShaderVertexInputInfo& v) {
	if (k.size() < 10) return false;
	size_t i = 0;
	v.fetch_embedded = k[i++] != 0; v.fetch_attrib_reg = static_cast<int>(k[i++]); v.fetch_buffer_reg = static_cast<int>(k[i++]);
	v.resources_num = static_cast<int>(k[i++]); v.wave_size = k[i++]; v.scratch_size_dwords = k[i++]; v.pa_cl_vs_out_cntl = k[i++];
	v.start_instance_sgpr = static_cast<int32_t>(k[i++]);
	if ((k[i] & 0xfff00000u) == 0x1d500000u) v.lds_size_dwords = k[i++] & 0xfffffu;
	if ((k[i] & 0xffff0000u) == 0x55c90000u) v.user_clip_planes = k[i++] & 0xffffu;
	if (k[i] == 0x7A11A5u) {
		i++; v.param_alias_mask = k[i++];
		for (uint32_t d = 0; d < 32; d++) if (v.param_alias_mask & (1u << d)) v.param_alias_source[d] = k[i++];
	}
	v.clip_space.enabled = k[i++] != 0;
	if (v.clip_space.enabled) {
		for (int j = 0; j < 2; j++) v.clip_space.scale[j] = std::bit_cast<float>(k[i++]);
		for (int j = 0; j < 2; j++) v.clip_space.offset[j] = std::bit_cast<float>(k[i++]);
		for (int j = 0; j < 2; j++) v.clip_space.half_extent[j] = std::bit_cast<float>(k[i++]);
	}
	v.mesh.threads_num[0] = k[i++];
	if (v.mesh.threads_num[0] != 0) return false;
	v.tess.input_control_points = k[i++];
	if (v.tess.input_control_points != 0) return false;
	if (v.resources_num < 0 || v.resources_num > ShaderVertexInputInfo::RES_MAX || k.size() < i + 3ull * v.resources_num) return false;
	for (int r = 0; r < v.resources_num; r++) {
		v.resources_dst[r].registers_num = static_cast<int>(k[i++]);
		v.resources_dst[r].attr_id = static_cast<int>(k[i++]);
		v.resources[r].fields[3] = k[i++];
	}
	return true;
}

bool PixelInfoFromKey(const std::vector<uint32_t>& k, ShaderPixelInputInfo& p) {
	if (k.size() < 30) return false;
	size_t i = 0;
	p.scratch_size_dwords = k[i++]; p.input_num = k[i++]; p.wave_size = k[i++]; p.ps_system_input_base = k[i++];
	if ((k[i] & 0xfff00000u) == 0x1d500000u) p.lds_size_dwords = k[i++] & 0xfffffu;
	p.custom_interpolation_mask = k[i++];
	p.ps_perspective_center_vgpr = k[i++]; p.ps_perspective_centroid_vgpr = k[i++]; p.ps_perspective_sample_vgpr = k[i++];
	p.ps_linear_sample_vgpr = k[i++]; p.ps_linear_center_vgpr = k[i++]; p.ps_linear_centroid_vgpr = k[i++];
	p.ps_pos_x = k[i++]; p.ps_pos_y = k[i++]; p.ps_pos_z = k[i++]; p.ps_pos_w = k[i++];
	{ const uint32_t f = k[i++]; p.ps_front_face = f & 1u; p.ps_front_face_all_bits = (f >> 1) & 1u; }
	p.ps_ancillary = k[i++] != 0;
	{ const uint32_t f = k[i++]; p.ps_no_perspective = f & 1u; p.parameter_mode = static_cast<ShaderPixelParameterMode>((f >> 8) & 0xffu); }
	p.ps_pixel_kill_enable = k[i++]; p.ps_depth_export_enable = k[i++]; p.ps_sample_mask_export_enable = k[i++]; p.ps_early_z = k[i++];
	p.dual_source_blending = k[i++]; p.alpha_blend_source = static_cast<ShaderAlphaBlendSource>(k[i++]);
	for (int j = 0; j < 8; j++) p.target_output_mode[j] = static_cast<uint8_t>(k[i++]);
	if (k.size() < i + 2 + p.input_num) return false;
	if (k.size() == i + 3 + p.input_num) { p.target_shader_mask = k[i++]; }
	for (uint32_t base = 0; base < 8; base += 4) { const uint32_t w = k[i++]; for (uint32_t j = 0; j < 4; j++) p.target_export_mapping[base + j].packed = static_cast<uint8_t>((w >> (j * 8)) & 0xffu); }
	for (uint32_t j = 0; j < p.input_num && j < 32; j++) p.interpolator_settings[j] = k[i++];
	return true;
}

// nps5 gpuguard: loop budget of the emitted MSL (MslOptions::loop_cap); ON by default in this tool (the MSL goes to the
// GPU test harness, where an unbounded loop on synthetic input locked up the GPU, 2026-11-18). --loop-cap 0 = off.
uint32_t g_loop_cap = 1u << 18u;

int Translate(const Src& s, const char* out_path, bool comments, uint32_t lanes, const char* manifest_path = nullptr, const Perm* perm = nullptr, const Src* link_ps = nullptr) {
	ShaderComputeInputInfo info;
	ShaderVertexInputInfo  vinfo;
	ShaderPixelInputInfo   pinfo;
	ShaderPixelInputInfo   link_pinfo;
	if (s.stage == static_cast<uint32_t>(ShaderType::Compute)) {
		ComputeInfoFromKey(s.statics, info);
		if (const char* sd = std::getenv("NPS5_SCRATCH_DWORDS")) info.scratch_size_dwords = static_cast<uint32_t>(std::strtoul(sd, nullptr, 0)); // flatlds: static scratch for FLAT-aperture tests
	} else if (s.stage == static_cast<uint32_t>(ShaderType::Vertex)) {
		if (!VertexInfoFromKey(s.statics, vinfo)) { std::fprintf(stderr, "cannot decode vertex static key (mesh/tess?)\n"); return 5; }
		if (link_ps) {
			if (!PixelInfoFromKey(link_ps->statics, link_pinfo)) { std::fprintf(stderr, "cannot decode linked pixel key\n"); return 5; }
			ShaderLinkVertexPixelParameters(vinfo, link_pinfo);
		}
	} else if (s.stage == static_cast<uint32_t>(ShaderType::Pixel)) {
		if (!PixelInfoFromKey(s.statics, pinfo)) { std::fprintf(stderr, "cannot decode pixel static key\n"); return 5; }
	}
	SR::CompileOptions options;
	options.stage          = static_cast<ShaderType>(s.stage);
	options.wave_size      = s.wave;
	options.user_data_base = s.user_data_base;
	options.shader_hash    = s.hash;
	options.dump_ir        = false;
	options.emit_msl       = false;
	options.emit_spirv     = true;
	options.msl_loop_cap   = g_loop_cap;
	std::vector<uint32_t> user_data(std::max(s.user_data_count, 16u), 0u);
	options.user_data   = user_data;
	if (s.stage == static_cast<uint32_t>(ShaderType::Vertex)) options.input_info.vertex = &vinfo;
	else if (s.stage == static_cast<uint32_t>(ShaderType::Pixel)) options.input_info.pixel = &pinfo;
	else options.input_info.compute = &info;
	auto translated = SR::TranslateProgram(s.code, options);
	if (translated.skip_dispatch) {
		std::fprintf(stderr, "skip_dispatch (unsupported call)\n");
		return 4;
	}
	// Specialization from the program's own resource list (numeric class etc. as translated).
	SR::IR::ResourceSpecialization spec;
	for (const auto& b: translated.program.info.buffers) {
		SR::IR::ResourceSpecialization::Buffer sb;
		sb.packed_stride = b.packed_stride; sb.descriptor_format = b.descriptor_format; sb.descriptor_swizzle = b.descriptor_swizzle;
		spec.buffers.push_back(sb);
	}
	for (const auto& i: translated.program.info.images) {
		SR::IR::ResourceSpecialization::Image si;
		si.numeric_class = i.numeric_class; si.dimension = i.dimension; si.mip_count = i.mip_count;
		si.conversion_format = i.conversion_format; si.shader_swizzle = i.shader_swizzle; si.cube = i.cube;
		// The real descriptors are unknown offline: assume a 2D float image where the translation could not tell.
		if (si.numeric_class == Prospero::TextureNumericClass::Unsupported) si.numeric_class = Prospero::TextureNumericClass::Float;
		if (si.dimension == SR::Decoder::ImageDimension::Unknown) si.dimension = SR::Decoder::ImageDimension::Dim2D;
		spec.images.push_back(si);
	}
	if (perm && !perm->spec.empty()) {
		SR::IR::ResourceSpecialization ds;
		if (SR::IR::DecodeSpecializationCompat(perm->spec, ds)) spec = ds;
		else std::fprintf(stderr, "warning: spec decode failed for perm %d\n", perm->perm);
	}
	(void)lanes; (void)comments;
	auto result = SR::CompileProgram(std::move(translated), options, spec, 0);
	std::FILE* out = out_path ? std::fopen(out_path, "wb") : stdout;
	std::fwrite(result.spirv.data(), 4, result.spirv.size(), out);
	if (out_path) std::fclose(out);
	if (manifest_path) {
		std::FILE* mf = std::fopen(manifest_path, "wb");
		if (mf) { std::fwrite(result.msl_manifest.data(), 1, result.msl_manifest.size(), mf); std::fclose(mf); }
	}
	std::fprintf(stderr, "// complete=%d unsupported:", result.msl_complete ? 1 : 0);
	for (const auto& u: result.msl_unsupported) std::fprintf(stderr, " %s", u.c_str());
	std::fprintf(stderr, "\n");
	return 0;
}

} // namespace

int main(int argc, char** argv) {
	const char *programs = nullptr, *bin = nullptr, *out = nullptr, *manifest = nullptr;
	long index = -1, perm_sel = -1, link_perm = -1;
	const char* all_dir = nullptr;
	bool all_cs_only = true;
	uint64_t hash = 0;
	bool have_hash = false, list = false, comments = true;
	uint32_t lanes = 0, wave = 64, ud = 16;
	uint32_t stage = static_cast<uint32_t>(ShaderType::Compute), threads[3] = {64, 1, 1};
	for (int i = 1; i < argc; i++) {
		std::string_view a = argv[i];
		auto next = [&]() { return i + 1 < argc ? argv[++i] : ""; };
		if (a == "--programs") programs = next();
		else if (a == "--bin") bin = next();
		else if (a == "--index") index = std::atol(next());
		else if (a == "--hash") { hash = std::strtoull(next(), nullptr, 16); have_hash = true; }
		else if (a == "--list") list = true;
		else if (a == "--perm") perm_sel = std::atol(next());
		else if (a == "--all") all_dir = next();
		else if (a == "--stage-all") all_cs_only = false;
		else if (a == "--link-ps") link_perm = std::atol(next());
		else if (a == "--lanes") lanes = static_cast<uint32_t>(std::atoi(next()));
		else if (a == "--no-comments") comments = false;
		else if (a == "--loop-cap") g_loop_cap = static_cast<uint32_t>(std::strtoul(next(), nullptr, 0));
		else if (a == "--kk-bda") setenv("KYTY_MSL_CONVENTION", "kk-bda", 1); // mslinject: KosmicKrisp binding convention
		else if (a == "--out") out = next();
		else if (a == "--manifest") manifest = next();
		else if (a == "--wave") wave = static_cast<uint32_t>(std::atoi(next()));
		else if (a == "--ud") ud = static_cast<uint32_t>(std::atoi(next()));
		else if (a == "--threads") std::sscanf(next(), "%u,%u,%u", &threads[0], &threads[1], &threads[2]);
		else if (a == "--stage") stage = std::string_view(next()) == "cs" ? static_cast<uint32_t>(ShaderType::Compute) : 0;
		else { std::fprintf(stderr, "unknown option %.*s\n", (int)a.size(), a.data()); return 2; }
	}
	std::vector<Src> srcs;
	std::vector<Perm> perms;
	const Perm* use_perm = nullptr;
	if (programs && all_dir) {
		srcs = LoadPrograms(programs, &perms);
		std::string idx = std::string(all_dir) + "/index.tsv";
		std::FILE* fi = std::fopen(idx.c_str(), "w");
		if (!fi) { std::fprintf(stderr, "cannot write %s\n", idx.c_str()); return 2; }
		for (const auto& pm: perms) {
			const Src* sp = nullptr;
			for (const auto& x: srcs) if (x.digest == pm.digest) { sp = &x; break; }
			if (!sp) continue;
			if (all_cs_only && sp->stage != static_cast<uint32_t>(ShaderType::Compute)) continue;
			char out_path[1024];
			std::snprintf(out_path, sizeof(out_path), "%s/%d.spv", all_dir, pm.perm);
			std::fflush(stdout);
			const pid_t pid = fork();
			if (pid == 0) { const int rc = Translate(*sp, out_path, false, 0, nullptr, &pm, nullptr); std::fflush(stdout); _exit(rc); }
			int st = 0;
			waitpid(pid, &st, 0);
			std::fprintf(fi, "%d\t%016" PRIx64 "\t%u\t%d\n", pm.perm, sp->hash, sp->stage, WIFEXITED(st) ? WEXITSTATUS(st) : 99);
		}
		std::fclose(fi);
		return 0;
	}
	if (programs) {
		srcs = LoadPrograms(programs, &perms);
		if (list) {
			for (size_t i = 0; i < srcs.size(); i++)
				std::printf("%zu hash=%016" PRIx64 " stage=%u wave=%u words=%zu\n", i, srcs[i].hash, srcs[i].stage, srcs[i].wave, srcs[i].code.size());
			return 0;
		}
		long pick = index;
		if (have_hash)
			for (size_t i = 0; i < srcs.size(); i++) if (srcs[i].hash == hash) { pick = static_cast<long>(i); break; }
		if (perm_sel >= 0) {
			for (const auto& pm: perms) if (pm.perm == perm_sel) use_perm = &pm;
			if (!use_perm) { std::fprintf(stderr, "no such perm\n"); return 2; }
			for (size_t i = 0; i < srcs.size(); i++) if (srcs[i].digest == use_perm->digest) { pick = static_cast<long>(i); break; }
		}
		if (pick < 0 || static_cast<size_t>(pick) >= srcs.size()) { std::fprintf(stderr, "no such shader (%zu sources)\n", srcs.size()); return 2; }
		Src s = srcs[static_cast<size_t>(pick)];
		srcs.clear();
		srcs.push_back(std::move(s));
	} else if (bin) {
		std::vector<uint8_t> d;
		if (!ReadAll(bin, d)) { std::fprintf(stderr, "cannot read %s\n", bin); return 2; }
		Src s;
		s.stage = stage; s.wave = wave; s.user_data_count = ud;
		s.code.resize(d.size() / 4);
		std::memcpy(s.code.data(), d.data(), s.code.size() * 4);
		// statics = BuildStageStaticKey layout for a compute shader with the given threads.
		s.statics = {0, wave | (0xc0u << 8u), 64, 0, 0, 0, 0, threads[0], 0, threads[1], 0, threads[2], 0, 0};
		srcs.push_back(std::move(s));
	} else {
		std::fprintf(stderr, "usage: kyty_msl_dump --programs <programs.bin> (--index N|--hash H|--list) | --bin <code.bin> [--threads X,Y,Z --wave 64]\n");
		return 2;
	}
	// The frontend EXITs (handler _exit) on unsupported input: run in a child.
	std::fflush(stdout);
	Src link_src;
	const Src* link_ptr = nullptr;
	if (link_perm >= 0) {
		// own perm list: reloading into `perms` would free the Perm that use_perm points at (flaky SIGSEGV in
		// DecodeSpecialization of --link-ps runs, mslrest3)
		std::vector<Perm> link_perms;
		auto all = LoadPrograms(programs, &link_perms);
		for (const auto& pm: link_perms) if (pm.perm == link_perm) for (const auto& x: all) if (x.digest == pm.digest) { link_src = x; link_ptr = &link_src; }
		if (!link_ptr) { std::fprintf(stderr, "no such link perm\n"); return 2; }
	}
	const pid_t pid = std::getenv("KYTY_MSL_DUMP_NOFORK") ? 0 : fork();
	if (pid == 0) {
		const int rc = Translate(srcs[0], out, comments, lanes, manifest, use_perm, link_ptr);
		std::fflush(stdout);
		_exit(rc);
	}
	int st = 0;
	waitpid(pid, &st, 0);
	return WIFEXITED(st) ? WEXITSTATUS(st) : 1;
}
