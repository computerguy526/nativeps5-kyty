// nps5 meshpath: Kyty mesh programs (ShaderType::Mesh = NGG / merged ES+GS run as a mesh shader)
// as Metal [[mesh]] functions, opt-in KYTY_MSL_MESH=1 (MslEmitter.cpp MeshEnabled).
//
// Mirrors backend/spirv/spirvEmitterMesh.cpp + the KK lowering it feeds:
//  - wave model: the workgroup (compute) lanes model; wave64 = 2 guest lanes per thread, so the
//    Metal threadgroup is the SPIR-V local size (c739f961: 128 guest lanes -> 64 threads = KK's).
//  - MeshDrawParameter(i): push dword i of the root push range (KK root + 672), or, when dword 3 is
//    IR::PushData::MeshIndirectSentinel (native indirect mesh draw, CodegenOptions::
//    mesh_indirect_params, on in the game: KK's text has the branch), dword i of the parameter
//    block at the device address in dwords 0-1.
//  - MeshAllocate(m0): guest lane 0 stores (m0 bits 0..9, bits 12..22) as the vertex / primitive
//    counts (threadgroup).
//  - exports: Position 0 / Parameter / Primitive go to per-lane locals (SPIR-V: Private arrays).
//  - epilogue (EmitMeshEntryPoint): barrier, counts clamped to the declared maxima (KK's lowering
//    clamps), guest lane L < vertices writes vertex L, L < primitives writes primitive L: indices
//    = 10-bit fields of its primitive export (clamped to maxV-1 as KK does), culled = bit 31.
//    The outputs go straight to mesh_out from the lane that owns them: no staging copy (KK stages
//    every output in threadgroup memory, 16 KB for c739f961).
// Interface: the B pair is linked to itself (B's FS reads user(locnL)), as for B vertex shaders.
#include "graphics/shader/recompiler/backend/msl/mslEmitterInternal.h"
#include "graphics/shader/recompiler/ir/opcodes/ValueOpcodes.h"

#include <cstdlib>
#include <fmt/format.h>
#include <set>

namespace Libs::Graphics::ShaderRecompiler::Msl {

namespace {

const ShaderMeshInputInfo& MeshOf(const MslCtx& c) {
	return c.input_info.vertex->mesh;
}

std::string GuestLane(const MslCtx& c, uint32_t h) {
	return c.lanes == 2 ? fmt::format("(tid + (tid & 4294967264u) + {}u)", h * 32u)
	                    : std::string("tid");
}

// ExportExpr (MslStageIo.cpp) for a non-pixel export: compressed pairs are f16x2, disabled
// components read (0, 0, 0, 1).
std::string ExportFloat4(const std::string& d, const IR::ExportInfo& exp) {
	if (exp.compr) {
		std::string f[4] = {"0.0f", "0.0f", "0.0f", "1.0f"};
		for (uint32_t pair = 0; pair < 2; pair++) {
			if ((exp.en & (3u << (pair * 2))) == 0) continue;
			const auto un = fmt::format("float2(as_type<half2>({}[{}]))", d, pair);
			for (uint32_t lane = 0; lane < 2; lane++) {
				const uint32_t comp = pair * 2 + lane;
				if ((exp.en >> comp) & 1u) f[comp] = fmt::format("({}).{}", un, lane ? 'y' : 'x');
			}
		}
		return fmt::format("float4({}, {}, {}, {})", f[0], f[1], f[2], f[3]);
	}
	if (exp.en == 0xfu) return fmt::format("as_type<float4>({})", d);
	std::string r[4] = {"0u", "0u", "0u", "0x3f800000u"};
	for (uint32_t comp = 0; comp < 4; comp++)
		if ((exp.en >> comp) & 1u) r[comp] = fmt::format("{}[{}]", d, comp);
	return fmt::format("as_type<float4>(uint4({}, {}, {}, {}))", r[0], r[1], r[2], r[3]);
}

// KYTY_MSL_KK_PS_READS="loc:mask,..." (the linked B PS's "// kk-bda ps_reads" line, as for B vertex
// shaders): parameters at locations the PS does not read are not exported (KK trims them too). Unset:
// every parameter.
std::set<uint32_t> Params(const MslCtx& c) {
	std::set<uint32_t> p;
	const char*        e = std::getenv("KYTY_MSL_KK_PS_READS");
	std::set<uint32_t> read;
	for (const char* q = e; q != nullptr && *q != 0;) {
		char*      end = nullptr;
		const auto l   = static_cast<uint32_t>(std::strtoul(q, &end, 10));
		if (end == q || *end != ':') break;
		if ((std::strtoul(end + 1, &end, 16) & 0xfu) != 0u) read.insert(l);
		q = *end == ',' ? end + 1 : end;
	}
	for (const auto& o: c.program.info.outputs)
		if (o.kind == IR::StageOutputKind::Parameter && (e == nullptr || read.contains(o.location)))
			p.insert(o.index);
	return p;
}

uint32_t LocationOf(const MslCtx& c, uint32_t index) {
	for (const auto& o: c.program.info.outputs)
		if (o.kind == IR::StageOutputKind::Parameter && o.index == index) return o.location;
	return index;
}

} // namespace

std::string MeshTypes(MslCtx& c) {
	const auto& m = MeshOf(c);
	// KK's ffract (NIR lowering of Kyty's V_FRACT / the V_SIN/V_COS range reduction, SetKkTrig)
	std::string s = "static inline float kyty_kk_fract(float x) {\n  const uint a = as_type<uint>(x) & 0x7fffffffu;\n"
	                "  const float f = (a < 0x7f800000u && a >= 0x4b000000u) ? 0.0f : x - floor(x);\n"
	                "  return a == 0u ? x : f;\n}\n";
	s += "struct MeshVertexOut {\n  float4 pos [[position]];\n";
	for (const auto& o: c.program.info.outputs) {
		switch (o.kind) {
			case IR::StageOutputKind::Position:
			case IR::StageOutputKind::Parameter: break;
			default:
				c.unsupported[fmt::format("mesh: output kind {}", static_cast<uint32_t>(o.kind))]++;
		}
	}
	for (const auto i: Params(c))
		s += fmt::format("  float4 p{} [[user(locn{})]];\n", i, LocationOf(c, i));
	s += "};\nstruct MeshPrimitiveOut {\n  bool cull [[primitive_culled]];\n};\n";
	s += fmt::format("using NpsMesh = metal::mesh<MeshVertexOut, MeshPrimitiveOut, {}, {}, "
	                 "metal::topology::triangle>;\n",
	                 m.max_vertices, m.max_primitives);
	if (m.input_primitive == 0u || m.max_vertices == 0u || m.max_primitives == 0u)
		c.unsupported["mesh: no vertex/primitive maxima"]++;
	return s;
}

void MeshLocals(MslCtx& c) {
	auto& b = c.b;
	b.Line("threadgroup uint ms_cnt[2]; // MeshAllocate: vertices, primitives");
	b.Line("float4 ms_pos[2] = {};");
	for (const auto i: Params(c))
		b.Line(fmt::format("float4 ms_p{}[2] = {{}};", i));
	b.Line("uint ms_prim[2] = {};");
	// The counts are undefined until MeshAllocate; zero them (KK zeroes its own counts too).
	b.Line("if (tid == 0u) { ms_cnt[0] = 0u; ms_cnt[1] = 0u; }");
	b.Line("threadgroup_barrier(mem_flags::mem_threadgroup);");
}

bool MeshEmitInst(MslCtx& c, const IR::Inst& inst) {
	using O = IR::ValueOpcode;
	auto&          b = c.b;
	const uint32_t h = c.half;
	switch (inst.GetOpcode()) {
		case O::MeshDrawParameter: {
			const uint32_t i = inst.Arg(0).U32();
			const auto& m = MeshOf(c);
			if (i >= IR::PushData::MeshDrawDwords(m.split_groups != 0)) {
				c.unsupported["mesh: draw parameter index"]++;
				b.Line(fmt::format("{} = 0u;", c.Name(inst)));
				return true;
			}
			// root push range = KK root + 672 (MslKkBda.cpp KkRootPush)
			b.Line(fmt::format(
			    "{} = *(constant uint*)(kk_root + 684u) == {}u ? *(coherent device uint*)(ulong(*(constant "
			    "uint*)(kk_root + 672u)) | (ulong(*(constant uint*)(kk_root + 676u)) << 32) | {}ul) : "
			    "*(constant uint*)(kk_root + {}u);",
			    c.Name(inst), IR::PushData::MeshIndirectSentinel, i * 4u, 672u + 4u * i));
			return true;
		}
		case O::MeshAllocate:
			b.Line(fmt::format("if ({} == 0u) {{ ms_cnt[0] = extract_bits({}, 0, 10); ms_cnt[1] = "
			                   "extract_bits({}, 12, 11); }}",
			                   GuestLane(c, h), c.Val(inst.Arg(0)), c.Val(inst.Arg(0))));
			return true;
		case O::SetAttribute: {
			const auto& exp = c.program.export_info.at(inst.Flags<IR::ExportFlags>().index);
			if (exp.kind == IR::ExportTargetKind::Null || exp.en == 0u) return true;
			const auto exec = c.Val(inst.Arg(1));
			const auto data = c.Val(inst.Arg(0));
			switch (exp.kind) {
				case IR::ExportTargetKind::Primitive:
					b.Line(fmt::format("if ({}) ms_prim[{}] = ({}).x;", exec, h, data));
					return true;
				case IR::ExportTargetKind::Position:
					if (exp.index != 0) {
						c.unsupported["mesh: auxiliary position export"]++;
						return true;
					}
					b.Line(fmt::format("if ({}) ms_pos[{}] = {};", exec, h, ExportFloat4(data, exp)));
					return true;
				case IR::ExportTargetKind::Parameter:
					if (Params(c).contains(exp.index))
						b.Line(fmt::format("if ({}) ms_p{}[{}] = {};", exec, exp.index, h,
						                   ExportFloat4(data, exp)));
					return true;
				default: c.unsupported["mesh: export kind"]++; return true;
			}
		}
		default: return false;
	}
}

void MeshEpilogue(MslCtx& c) {
	auto&       b = c.b;
	const auto& m = MeshOf(c);
	b.Line("// mesh epilogue (spirvEmitterMesh.cpp EmitMeshEntryPoint): every guest wave is done");
	b.Line("threadgroup_barrier(mem_flags::mem_threadgroup);");
	b.Line(fmt::format("const uint ms_nv = min(ms_cnt[0], {}u); const uint ms_np = min(ms_cnt[1], {}u);",
	                   m.max_vertices, m.max_primitives));
	b.Line("if (tid == 0u) mesh_out.set_primitive_count(ms_np);");
	const auto ps = Params(c);
	for (uint32_t h = 0; h < c.lanes; h++) {
		const auto L = GuestLane(c, h);
		b.Open(fmt::format("if ({} < ms_nv)", L));
		b.Line(fmt::format("MeshVertexOut v; v.pos = ms_pos[{}];", h));
		for (const auto i: ps)
			b.Line(fmt::format("v.p{} = ms_p{}[{}];", i, i, h));
		b.Line(fmt::format("mesh_out.set_vertex({}, v);", L));
		b.Close();
		b.Open(fmt::format("if ({} < ms_np)", L));
		b.Line(fmt::format("const uint q = ms_prim[{}];", h));
		for (uint32_t k = 0; k < 3; k++)
			b.Line(fmt::format("mesh_out.set_index({} * 3u + {}u, min(extract_bits(q, {}, 10), {}u));",
			                   L, k, k * 10u, m.max_vertices - 1u));
		b.Line("MeshPrimitiveOut p; p.cull = (q & 0x80000000u) != 0u;");
		b.Line(fmt::format("mesh_out.set_primitive({}, p);", L));
		b.Close();
	}
}

} // namespace Libs::Graphics::ShaderRecompiler::Msl
