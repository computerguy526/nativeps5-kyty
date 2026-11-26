// mslinject: binding convention "kk-bda" = exactly what KosmicKrisp's own (NIR->MSL) compute
// kernels expect, so a B kernel can replace KK's MSL for a pipeline without touching KK's runtime
// binding code (KK patch kk-mslinject, env KK_MSL_OVERRIDE_DIR):
//   buffer(0) = KK root table (kk_cmd_buffer.h kk_root_descriptor_table): cs.base_group@8,
//   push@672,
//               sets[0] (u64 address of the set-0 descriptor buffer)@928
//   buffer(1) = KK sampler table (sampler handles[MSL_MAX_SAMPLERS])
//   set 0     = Kyty's one descriptor set (NativeBinding = kind for compute), laid out as KK does
//               (kk_descriptor_set_layout.c: bindings sorted by number, offset = align(running,
//               stride); storage/uniform buffer 16B {u64 addr,u32 size,u32 0}, storage image 8B
//               {u64 resource id}, sampled image / sampler 64B kk_sampled_image_descriptor {u64 id,
//               u16 sampler_index, ...})
// The kernel body is unchanged: the prelude builds the same `Resources r`, `push`, `tgpig`, `lds`
// locals the msl-native-v1 body reads. Buffer base adjustments (Kyty memory offsets, SPIR-V
// ByteAddress prefix) are folded into r.b<i> (address + prefix, size - prefix). The first source
// line tells KK the threadgroup size, entry point and math mode: "// kk-mslinject v1 local_size X Y
// Z entry main_entrypoint math safe".
#include "graphics/shader/recompiler/CodegenOptions.h"
#include "graphics/shader/recompiler/backend/msl/MslBinding.h"
#include "graphics/shader/recompiler/backend/msl/MslBufferOps.h"
#include "graphics/shader/recompiler/backend/msl/MslImageOps.h"
#include "graphics/shader/recompiler/backend/msl/MslSharedOps.h"
#include "graphics/shader/recompiler/backend/msl/MslXform.h"
#include "graphics/shader/recompiler/backend/msl/mslEmitterInternal.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>

namespace Libs::Graphics::ShaderRecompiler::Msl {

using namespace Mem;
using Kind = IR::DescriptorBindingKind;

bool KkBdaRequested() {
	const char* c = std::getenv("KYTY_MSL_CONVENTION");
	return c != nullptr && std::strcmp(c, "kk-bda") == 0;
}

bool MipStatsPlainRequested() {
	const char* c = std::getenv("KYTY_MSL_MIPSTATS_PLAIN");
	return c != nullptr && c[0] == '1';
}

namespace {

constexpr uint32_t KkRootPush      = 672;
constexpr uint32_t KkRootSets      = 928;
constexpr uint32_t KkRootBaseGroup = 8;

struct KkBinding {
	Kind                  kind;
	uint32_t              binding = 0;
	uint32_t              offset  = 0;
	uint32_t              stride  = 0;
	std::vector<uint32_t> resources;
};

uint32_t KkStride(Kind kind) {
	switch (IR::ImageBindingResourceClass(kind)) {
		case IR::ImageResourceClass::Sampled: return 64u;
		case IR::ImageResourceClass::Storage: return 8u;
		default: break;
	}
	return kind == Kind::Samplers ? 64u : 16u; // every other kind is a storage/uniform buffer
}

std::vector<KkBinding> KkSetLayout(const IR::Program& program) {
	std::vector<KkBinding> v;
	for (const auto& d: program.bindings.descriptors) {
		KkBinding k;
		k.kind      = d.kind;
		k.binding   = IR::NativeBinding(program.stage, d.kind);
		k.stride    = KkStride(d.kind);
		k.resources = d.resources;
		v.push_back(k);
	}
	std::ranges::sort(v, {}, &KkBinding::binding);
	uint32_t size = 0;
	for (auto& k: v) {
		size     = (size + k.stride - 1u) / k.stride * k.stride;
		k.offset = size;
		size += k.stride * static_cast<uint32_t>(std::max<size_t>(k.resources.size(), 1u));
	}
	// mslgfx: a graphics pipeline's set 0 holds the bindings of BOTH stages (VS and PS numbers
	// differ), so one stage cannot compute the offsets alone: KYTY_MSL_KK_SET0="binding:offset,..."
	// (the pipeline's KK set-0 layout, R1b manifest seed kk.set0) overrides them.
	if (const char* e = std::getenv("KYTY_MSL_KK_SET0"); e != nullptr && *e != 0) {
		for (const char* q = e; *q != 0;) {
			char*      end = nullptr;
			const auto bnd = static_cast<uint32_t>(std::strtoul(q, &end, 10));
			if (end == q || *end != ':') break;
			const auto off = static_cast<uint32_t>(std::strtoul(end + 1, &end, 10));
			for (auto& k: v)
				if (k.binding == bnd) k.offset = off;
			q = *end == ',' ? end + 1 : end;
		}
	}
	return v;
}

// Byte offset in set 0 of resource `index` of a binding that `pred` accepts; UINT32_MAX if none.
template <typename Pred>
uint32_t DescOffset(const std::vector<KkBinding>& layout, Pred pred, uint32_t index) {
	for (const auto& k: layout) {
		if (!pred(k.kind)) continue;
		const auto at = std::ranges::find(k.resources, index);
		if (at != k.resources.end())
			return k.offset + k.stride * static_cast<uint32_t>(at - k.resources.begin());
	}
	return UINT32_MAX;
}

uint32_t FirstOf(const std::vector<KkBinding>& layout, Kind kind) {
	for (const auto& k: layout)
		if (k.kind == kind) return k.offset;
	return UINT32_MAX;
}

std::string KU(uint32_t v) {
	return std::to_string(v) + "u";
}

} // namespace

std::string KkBdaPrelude() {
	return "#if __METAL_VERSION__ >= 320\n#define KK_COHERENT coherent\n#else\n#define "
	       "KK_COHERENT\n#endif\nstruct KkRoot {\n    uint64_t contents[1];\n};\nstruct "
	       "KkSamplerTable {\n    sampler handles[4096];\n};\n";
}

// meshpath spill ABI (shared with KK patch 0030): KK_MESH_SCRATCH_* = word 0 ticket, word 1
// overflow count, words 16.. one owner flag per slot, slices from byte 8192, 16 KB apart, 1024
// slots.
constexpr uint32_t KkMeshScratchSlots = 1024, KkMeshScratchFlags = 16, KkMeshScratchHead = 8192,
                   KkMeshScratchStride = 16384, KkMeshScratchTries = 64;
bool               MeshSpillWanted(uint32_t lds_bytes) {
	const char* e = std::getenv("KYTY_MSL_MESH_SPILL");
	return e != nullptr && std::strcmp(e, "1") == 0 && lds_bytes >= 6000u &&
	       lds_bytes <= KkMeshScratchStride;
}

void KkBdaParams(MslCtx& ctx, const BindingModel& m, std::vector<std::string>& params) {
	params.push_back("constant KkRoot& kkroot [[buffer(0)]]");
	if (!m.samplers.empty())
		params.push_back("constant KkSamplerTable& sampler_table [[buffer(1)]]");
	if (ctx.program.stage == ShaderType::Mesh && ctx.mesh && MeshSpillWanted(m.lds_bytes)) {
		ctx.mesh_spill = true;
		params.push_back("device uchar* kk_scr [[buffer(3)]]");
	}
}

std::string KkBdaLayoutComment(const IR::Program& program) {
	std::string s = "// kk-bda set0:";
	for (const auto& k: KkSetLayout(program)) {
		s += " " + std::to_string(k.binding) + "@" + std::to_string(k.offset) + "x" +
		     std::to_string(std::max<size_t>(k.resources.size(), 1u));
	}
	const auto& b = program.bindings;
	s += " push_start=" +
	     (b.UsesPushData() ? std::to_string(b.push_data_start_dword) : std::string("none"));
	s += " mem_off=" + std::to_string(b.memory_offset_dword) + "x" +
	     std::to_string(b.memory_offset_count);
	s += " ud=" + std::to_string(b.user_data_registers.size());
	return s + "\n";
}

// srtexec: the layout file kk_spv_lay reads ("B <binding> <VkDescriptorType> <count> <stage
// flags>", "P <stage flags> <push size>"), straight from program.bindings = what AddLayoutBindings
// gives vkCreateDescriptorSetLayout (binding = NativeBinding, count = max(1, resources), push range
// {stage, 0, NativePushConstantSize}); replaces layout_dump (which decodes 165 of 935 SAROS CS
// permutations). "D 1" = the program binds Gds (kk_spv_lay needs --dispatch-base). Compute stage
// flags (0x20) for compute programs, 0x1/0x10 otherwise.
std::string KkLayoutText(const IR::Program& program) {
	const uint32_t stages = program.stage == ShaderType::Compute ? 0x20u
	                        : program.stage == ShaderType::Pixel ? 0x10u
	                                                             : 0x1u;
	std::string    s;
	bool           gds = false;
	for (const auto& k: KkSetLayout(program)) {
		uint32_t type =
		    7u; // storage buffer: every kind that is not an image / sampler / uniform buffer
		switch (IR::ImageBindingResourceClass(k.kind)) {
			case IR::ImageResourceClass::Sampled: type = 2u; break;
			case IR::ImageResourceClass::Storage: type = 3u; break;
			default:
				if (k.kind == Kind::Samplers)
					type = 0u;
				else if (k.kind == Kind::UniformBuffers)
					type = 6u;
		}
		gds = gds || k.kind == Kind::Gds;
		s += "B " + std::to_string(k.binding) + " " + std::to_string(type) + " " +
		     std::to_string(std::max<size_t>(k.resources.size(), 1u)) + " " +
		     std::to_string(stages) + "\n";
	}
	s += "P " + std::to_string(stages) + " " + std::to_string(IR::NativePushConstantSize) + "\n";
	if (gds) s += "D 1\n";
	return s;
}

void KkBdaLocals(MslCtx& ctx, const BindingModel& m) {
	SetKkSamplerBias(true);
	auto&       b       = ctx.b;
	const auto& program = ctx.program;
	const auto& bl      = program.bindings;
	const auto  layout  = KkSetLayout(program);
	const auto  bad     = [&](const std::string& why) { ctx.unsupported["kk-bda: " + why]++; };

	b.Line("constant uchar* kk_root = (constant uchar*)&kkroot;");
	if (Xf::On(Xf::LoadSelect)) { // the root block is bound for every draw/dispatch and holds the
		                          // set pointer read below
		b.Line("const ulong kk_lsafe = (ulong)&kkroot.contents[0];");
		SetLoadSelectSafe("kk_lsafe");
	}
	b.Line("const ulong kk_set0 = *(constant ulong*)(kk_root + " + KU(KkRootSets) + ");");
	// meshpath: a mesh program is a graphics-pipeline stage (buffer addressing as VS/PS, `gfx`)
	// with a real workgroup (threadgroup LDS and workgroup id as compute, `gfx_lane`). KK 0007:
	// the mesh workgroup id has no base (the root words alias the vertex strides).
	const bool gfx      = program.stage != ShaderType::Compute;
	const bool mesh     = program.stage == ShaderType::Mesh;
	const bool gfx_lane = gfx && !mesh;
	if (!gfx)
		b.Line("const uint3 tgpig = kk_tgpig + uint3(*(constant packed_uint3*)(kk_root + " +
		       KU(KkRootBaseGroup) + "));");
	if (mesh) b.Line("const uint3 tgpig = kk_tgpig; // KK 0007: mesh workgroup id without base");
	SetLdsThread(0u);
	if (gfx_lane) {
		// mslrest4: Kyty gives stages without a workgroup a per-invocation Function LDS array
		// (spirvEmitterMemoryHelpers.cpp LdsDwordCount: pixel max(lds_size_dwords,1), vertex
		// max(lds_size_dwords,1)), bounds-checked like workgroup LDS.
		bool lds = false, lds_atomic = false;
		for (const auto* blk: program.blocks) {
			for (const auto& inst: *blk) {
				const auto sa = IR::SharedAccessOf(inst.GetOpcode());
				if (sa == IR::SharedAccess::None) continue;
				const auto index = inst.Flags<IR::MemoryFlags>().index;
				if (index >= program.memory_info.size() ||
				    program.memory_info[index].kind != IR::ResourceKind::Lds)
					continue;
				lds = true;
				if (sa != IR::SharedAccess::Read && sa != IR::SharedAccess::Write)
					lds_atomic = true;
			}
		}
		if (lds) {
			uint32_t dw = 1u;
			if (program.stage == ShaderType::Pixel && ctx.input_info.pixel != nullptr)
				dw = std::max<uint32_t>(ctx.input_info.pixel->lds_size_dwords, 1u);
			else if (program.stage == ShaderType::Vertex && ctx.input_info.vertex != nullptr)
				dw = std::max<uint32_t>(ctx.input_info.vertex->lds_size_dwords, 1u);
			else
				bad("graphics LDS outside VS/PS");
			if (lds_atomic) bad("graphics LDS atomic");
			if (dw > 1024u) bad("graphics LDS > 4 KB per invocation");
			b.Line("uint kk_lds[" + std::to_string(dw) +
			       "] = {}; thread uint* lds = kk_lds; // Kyty function_lds (per invocation)");
			SetLdsThread(dw * 4u);
		}
	}
	// Kyty's GPU LOD feedback (KYTY_LOD_STATS_MODE=gpu, default): sampling pixel shaders record the
	// finest wanted mip per image in the MipStats buffer (spirvEmitterImage.cpp
	// EmitGatedMipStatsRecord; B: MslImageOps MipStatsRecord). KYTY_MSL_MIPSTATS_PLAIN=1 emits the
	// feedback-free twin (Spirv::EmitProgram mip_stats_records=false, same layout).
	SetMipStats({});
	if (const uint32_t ms = FirstOf(layout, Kind::MipStats); ms != UINT32_MAX) {
		const auto& opt = GetCodegenOptions();
		if (!opt.lod_stats_gate)
			bad("MipStats record mode other than gate (KYTY_LOD_STATS_GATE=1)");
		if (!MipStatsPlainRequested()) {
			b.Line("const ulong kk_ms_a = *(constant ulong*)(kk_set0 + " + KU(ms) +
			       "); const uint kk_ms_n = *(constant uint*)(kk_set0 + " + KU(ms + 8u) + ");");
			SetMipStats({true, bl.MipStatsOffsetDword(), bl.mip_stats_count});
		}
	}
	// Shader data (user data, dispatch extent, memory offsets): push constants or the ShaderData
	// buffer.
	if (bl.UsesPushData()) {
		b.Line("constant uint* push = (constant uint*)(kk_root + " +
		       KU(KkRootPush + 4u * bl.push_data_start_dword) + ");");
	} else {
		const uint32_t sd = FirstOf(layout, Kind::ShaderData);
		if (sd == UINT32_MAX) {
			b.Line("constant uint* push = (constant uint*)(kk_root + " + KU(KkRootPush) + ");");
			if (bl.ShaderDataDwords() != 0u)
				bad("shader data without push data or ShaderData binding");
		} else {
			b.Line("const device uint* push = (const device uint*)(*(constant ulong*)(kk_set0 + " +
			       KU(sd) + "));");
		}
	}
	// meshpath: guest-address loads (FLAT/GLOBAL/scalar reads through a 64-bit guest address) read
	// through the BDA page table like Kyty's SPIR-V (spirvEmitterMemory.cpp GetBdaPointer). Only
	// for MESH programs for now, and only when the program has such a load (other texts unchanged).
	SetKkBdaPagetable(false);
	if (mesh) {
		bool loads = false;
		for (const auto* blk: program.blocks)
			for (const auto& inst: *blk) {
				const auto op = inst.GetOpcode();
				loads |= op == IR::ValueOpcode::LoadAddressU8 ||
				         op == IR::ValueOpcode::LoadAddressU16 ||
				         op == IR::ValueOpcode::LoadAddressU32;
			}
		const uint32_t pt = FirstOf(layout, Kind::BdaPagetable),
		               fb = FirstOf(layout, Kind::FaultBuffer);
		if (loads && pt != UINT32_MAX && fb != UINT32_MAX) {
			b.Line("const ulong kk_bda_pt = *(constant ulong*)(kk_set0 + " + KU(pt) +
			       "); const uint kk_bda_ptn = *(constant uint*)(kk_set0 + " + KU(pt + 8u) + ");");
			b.Line("const ulong kk_bda_fb = *(constant ulong*)(kk_set0 + " + KU(fb) +
			       "); const uint kk_bda_fbn = *(constant uint*)(kk_set0 + " + KU(fb + 8u) + ");");
			SetKkBdaPagetable(true);
		} else if (loads) {
			bad("guest address load without BdaPagetable/FaultBuffer bindings");
		}
	}
	const uint32_t lds_dev = FirstOf(layout, Kind::SharedMemory);
	if (gfx_lane) {
	} else if (mesh && ctx.mesh_spill) {
		// meshpath spill: lane 0 claims a free slot (ticket, then compare-and-swap of its owner
		// flag 0 -> 1), broadcasts it through threadgroup memory; a group that finds none in
		// KkMeshScratchTries tickets counts itself and draws nothing (never a slice it does not
		// own). The epilogue frees the slot (MslMesh.cpp).
		b.Line("threadgroup uint kk_slot_tg;");
		b.Line("if (tid == 0u) {");
		b.Line("  uint s = ~0u;");
		b.Line("  for (uint k = 0u; k < " + KU(KkMeshScratchTries) + "; k++) {");
		b.Line("    const uint t = atomic_fetch_add_explicit((device atomic_uint*)kk_scr, 1u, "
		       "memory_order_relaxed) & " +
		       KU(KkMeshScratchSlots - 1u) + ";");
		b.Line("    uint z = 0u;");
		b.Line("    if (atomic_compare_exchange_weak_explicit((device atomic_uint*)kk_scr + " +
		       KU(KkMeshScratchFlags) +
		       " + t, &z, 1u, memory_order_relaxed, memory_order_relaxed)) { s = t; break; }");
		b.Line("  }");
		b.Line("  if (s == ~0u) atomic_fetch_add_explicit((device atomic_uint*)kk_scr + 1u, 1u, "
		       "memory_order_relaxed);");
		b.Line("  kk_slot_tg = s;");
		b.Line("}");
		b.Line("threadgroup_barrier(mem_flags::mem_threadgroup);");
		b.Line("const uint kk_slot = kk_slot_tg;");
		b.Line("if (kk_slot == ~0u) { if (tid == 0u) mesh_out.set_primitive_count(0u); return; }");
		b.Line("atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, "
		       "thread_scope_device);");
		b.Line("const uint kk_lds_bytes = " + KU(m.lds_bytes) + ";");
		b.Line("KK_COHERENT device uint* lds = (KK_COHERENT device uint*)(kk_scr + " +
		       KU(KkMeshScratchHead) + " + ulong(kk_slot) * " +
		       std::to_string(KkMeshScratchStride) + "ul);");
		SetLdsDevice(true);
	} else if (mesh && m.lds_bytes != 0u && lds_dev != UINT32_MAX) {
		bad("mesh LDS in device memory (lds_storage)");
	} else if (m.lds_bytes != 0u && lds_dev != UINT32_MAX) {
		// Kyty lds_storage (LDS larger than the host's threadgroup memory, e.g. 48 KB > Metal's 32
		// KB): the workgroup's LDS is region x + nx * (y + ny * z) of the SharedMemory device
		// buffer, regions LdsStorageRegionDwords apart; a workgroup the bound range holds no region
		// for sees length 0 (reads 0, writes dropped) -- exactly spirvEmitterProgram.cpp's
		// lds_base_dwords / lds_length.
		const uint32_t dw     = (m.lds_bytes + 3u) / 4u;
		const uint32_t region = LdsStorageRegionDwords(dw);
		SetLdsDevice(true);
		b.Line("const ulong kk_lds_a = *(constant ulong*)(kk_set0 + " + KU(lds_dev) +
		       "); const uint kk_lds_n = *(constant uint*)(kk_set0 + " + KU(lds_dev + 8u) + ");");
		b.Line("const uint kk_lds_wg = tgpig.x + nwg.x * (tgpig.y + nwg.y * tgpig.z);");
		b.Line("const bool kk_lds_in = kk_lds_wg < (kk_lds_n >> 2u) / " + KU(region) + ";");
		b.Line("const uint kk_lds_bytes = kk_lds_in ? " + KU(dw * 4u) + " : 0u;");
		b.Line("KK_COHERENT device uint* lds = (KK_COHERENT device uint*)(kk_lds_a) + (kk_lds_in ? "
		       "kk_lds_wg * " +
		       KU(region) + " : 0u);");
	} else if (m.lds_bytes != 0u) {
		b.Line("threadgroup uint kk_lds[" + std::to_string((m.lds_bytes + 3u) / 4u) + "];");
		b.Line("threadgroup uint* lds = kk_lds;");
	}
	b.Line("Resources r;");
	for (const auto& buf: m.buffers) {
		const auto i   = buf.id;
		const auto off = DescOffset(
		    layout, [](Kind k) { return k == Kind::Buffers || k == Kind::UniformBuffers; }, i);
		if (off == UINT32_MAX) {
			bad("buffer without descriptor");
			continue;
		}
		const std::string d      = "(kk_set0 + " + KU(off) + ")";
		std::string       prefix = "0u";
		if (i < bl.memory_offset_count) {
			// mslfix2 (graphics): Kyty's SPIR-V addresses buffers in dwords: base = offset byte &
			// ~3, element k valid while (offset >> 2) + k < size >> 2 (KK FS 48096ca2: t126 = byte
			// & 0xfffffffc). Compute text unchanged.
			prefix = "((push[" + KU(bl.memory_offset_dword + i / 4u) + "] >> " + KU(8u * (i % 4u)) +
			         ") & " + (gfx ? "252u)" : "255u)");
		}
		b.Line("{ const ulong a = *(constant ulong*)" + d + "; const uint n = *(constant uint*)(" +
		       d + " + 8u)" + (gfx ? " & 0xfffffffcu" : "") + "; const uint p = " + prefix + ";");
		b.Line("  r.b" + std::to_string(i) + " = (" +
		       std::string(buf.written || buf.atomic ? "device" : "const device") +
		       " uchar*)(a + ulong(p)); r.b" + std::to_string(i) + "_size = n > p ? n - p : 0u; }");
	}
	if (m.srt_bytes != 0u) {
		const uint32_t off = FirstOf(layout, Kind::FlattenedSrt);
		if (off == UINT32_MAX)
			bad("ReadConst without FlattenedSrt binding");
		else
			b.Line("r.srt = (const device uint*)(*(constant ulong*)(kk_set0 + " + KU(off) + "));");
	}
	// vs_const_load (MslXform.h): on for this program; with a written buffer also the per-draw
	// alias bool (bound range of a read-only buffer overlaps the bound range of a written one).
	if (Xf::On(Xf::VsConstLoad) && m.gds_bytes == 0u) {
		std::string alias;
		for (const auto& w: m.buffers) {
			if (!w.written && !w.atomic) continue;
			for (const auto& rd: m.buffers) {
				if (rd.written || rd.atomic) continue;
				const std::string R = std::to_string(rd.id), Wn = std::to_string(w.id);
				alias += std::string(alias.empty() ? "" : " || ") + "((ulong)r.b" + R +
				         " < (ulong)r.b" + Wn + " + ulong(r.b" + Wn + "_size) && (ulong)r.b" + Wn +
				         " < (ulong)r.b" + R + " + ulong(r.b" + R + "_size))";
			}
		}
		if (!alias.empty()) b.Line("const bool kk_calias = " + alias + ";");
		SetConstLoad(1, "kk_calias");
	}
	if (m.gds_bytes != 0u) {
		const uint32_t off = FirstOf(layout, Kind::Gds);
		if (off == UINT32_MAX)
			bad("GDS without Gds binding");
		else
			b.Line("r.gds = (device uchar*)(*(constant ulong*)(kk_set0 + " + KU(off) + "));");
	}
	std::string null_check;
	for (const auto& img: m.images) {
		// 1D/1D-array: KK views them as 2D/2D-array (kk_image_view.c remove_1d_view_types, NIR
		// lower_1d: sample y 0.5, load/store y 0) = B's emulated_2d. Cube: Kyty's SPIR-V declares
		// cube images as 2D arrays (face = layer, st - 1, CubeLayer padding), so KK's MSL reads
		// them as texture2d_array, as B does. Depth: KK declares the same depth2d / depth2d_array
		// types from the SPIR-V depth flag. (mslrest: verified by injdiff vs KK.)
		const auto off = DescOffset(
		    layout,
		    [](Kind k) { return IR::ImageBindingResourceClass(k) != IR::ImageResourceClass::None; },
		    img.id);
		if (off == UINT32_MAX) {
			bad("image without descriptor");
			continue;
		}
		const std::string id = "*(constant ulong*)(kk_set0 + " + KU(off) + ")";
		null_check += (null_check.empty() ? "" : " || ") + std::string("(") + id + ") == 0ul";
		b.Line("r.t" + std::to_string(img.id) + " = *(constant " + TextureDecl(img) +
		       "*)(kk_set0 + " + KU(off) + ");");
	}
	for (const auto& s: m.samplers) {
		const auto off = DescOffset(layout, [](Kind k) { return k == Kind::Samplers; }, s.id);
		if (off == UINT32_MAX) {
			bad("sampler without descriptor");
			continue;
		}
		b.Line("r.s" + std::to_string(s.id) +
		       " = sampler_table.handles[*(constant ushort*)(kk_set0 + " + KU(off + 8u) + ")];");
		b.Line("const float kk_sbias" + std::to_string(s.id) +
		       " = float(*(constant half*)(kk_set0 + " + KU(off + 12u) + "));");
	}
	// Null image descriptors (robustness2 nullDescriptor): Metal cannot bind resource id 0; KK
	// guards each access, B skips the dispatch for this thread group instead (no fault; output
	// stays as it was).
	if (!null_check.empty()) b.Line("if (" + null_check + ") { " + ctx.ret + " }");
}

} // namespace Libs::Graphics::ShaderRecompiler::Msl
