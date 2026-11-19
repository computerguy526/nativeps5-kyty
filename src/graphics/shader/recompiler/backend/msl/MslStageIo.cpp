// MSL backend M3: vertex / pixel stage entry points and stage IO.
//
// Wave model (graphics): one Metal thread = one guest lane (lanes = 1) for BOTH wave32 and wave64
// shaders, the same model the SPIR-V emitter uses for graphics stages (ShaderLanesPerInvocation is
// 1 outside workgroup stages). EXEC masks only guard the exports; derivatives / quad ops use
// Metal's native 2x2 quads (helper lanes are real simd lanes, discard_fragment keeps a lane alive
// as a helper), so implicit-LOD sampling and DPP quad_perm keep their GCN meaning. simd_*
// ballot/vote ops see the 32 simd lanes of the draw, which is the same limitation KK has.
//
// Stage IO mapping (mirrors backend/spirv):
//  VS in : Parameter(loc) -> [[stage_in]] attribute(loc) float4/uint4/int4 (Metal does the vertex
//  fetch
//          and format conversion from the MTLVertexDescriptor; scalar class from the V# format as
//          the SPIR-V emitter's VertexParameterScalarKind), VertexIndex/InstanceIndex/BaseInstance
//          builtins.
//  VS out: Position -> [[position]], Parameter(i) -> user(locnL), ClipDistance[], PointSize, Layer,
//  Viewport. PS in : Parameter -> user(locnL) with flat / center|centroid|sample x
//  perspective|no_perspective,
//          FragCoord/FrontFacing/SampleId/PrimitiveId/Layer builtins, barycentrics
//          [[barycentric_coord]] (centroid/sample I/J fall back to center, differs only on MSAA).
//  PS out: Mrt(i) -> [[color(loc)]] float4 (uint4 for target_output_mode 7), depth [[depth(any)]],
//          sample mask, discard via the export valid mask (SetAttribute vm), early_fragment_tests.
#include "graphics/shader/recompiler/BufferFormat.h"
#include "graphics/shader/recompiler/CodegenOptions.h"
#include "graphics/shader/recompiler/LodStatsEarlyTests.h"
#include "graphics/shader/recompiler/backend/msl/mslEmitterInternal.h"
#include "graphics/shader/recompiler/ir/opcodes/ValueOpcodes.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fmt/format.h>
#include <map>
#include <set>

namespace Libs::Graphics::ShaderRecompiler::Msl {

namespace {

enum class VKind { Float, Uint, Sint };

VKind VertexKind(const ShaderVertexInputInfo& v, uint32_t loc) {
	using F = Prospero::BufferFormat;
	if (loc >= static_cast<uint32_t>(ShaderVertexInputInfo::RES_MAX) ||
	    loc >= static_cast<uint32_t>(v.resources_num))
		return VKind::Float;
	switch (v.resources[loc].Format()) {
		case F::k8UInt:
		case F::k16UInt:
		case F::k8_8UInt:
		case F::k32UInt:
		case F::k16_16UInt:
		case F::k8_8_8_8UInt:
		case F::k32_32UInt:
		case F::k16_16_16_16UInt:
		case F::k32_32_32UInt:
		case F::k32_32_32_32UInt: return VKind::Uint;
		case F::k8SInt:
		case F::k16SInt:
		case F::k8_8SInt:
		case F::k32SInt:
		case F::k16_16SInt:
		case F::k8_8_8_8SInt:
		case F::k32_32SInt:
		case F::k16_16_16_16SInt:
		case F::k32_32_32SInt:
		case F::k32_32_32_32SInt: return VKind::Sint;
		default: return VKind::Float;
	}
}

bool IsLinear(IR::InterpolationMode m) {
	return m == IR::InterpolationMode::LinearCenter || m == IR::InterpolationMode::LinearCentroid ||
	       m == IR::InterpolationMode::LinearSample;
}
bool IsCentroid(IR::InterpolationMode m) {
	return m == IR::InterpolationMode::PerspectiveCentroid ||
	       m == IR::InterpolationMode::LinearCentroid;
}
bool IsSample(IR::InterpolationMode m) {
	return m == IR::InterpolationMode::PerspectiveSample ||
	       m == IR::InterpolationMode::LinearSample;
}

// mslgfx (kk-bda VS): KosmicKrisp pulls vertices itself (kk_nir_lower_vbo.c): attribute slot k
// (rank of the location among the attributes the VS reads) has its base address (buffer + attribute
// offset) at root 136 + 8k and its last valid element at root 392 + 4k (robustness2: out-of-range
// elements read 0), binding b's stride at root 8 + 4b. The raw components are loaded in the
// format's interchange type and converted as KK's NIR does (MSL text as KK prints it, math_mode
// fast = KK's library default for VS). VkFormat numbers: the pipeline's vertex input state.
struct KkVtxFmt {
	const char* elem;  // interchange element type (uint / ushort / uchar)
	uint32_t    size;  // bytes per element
	uint32_t    comps; // components loaded
	int         kind; // 0 float bits, 1 half, 2 unorm, 3 snorm, 4 uint, 5 rgb10a2 unorm, 6 rg11b10f
	uint32_t    bits; // unorm/snorm channel bits
};

bool KkVertexFormat(uint32_t vk, KkVtxFmt& f) {
	switch (vk) {
		case 100: f = {"uint", 4, 1, 0, 32}; return true;  // R32_SFLOAT
		case 103: f = {"uint", 4, 2, 0, 32}; return true;  // R32G32_SFLOAT
		case 106: f = {"uint", 4, 3, 0, 32}; return true;  // R32G32B32_SFLOAT
		case 109: f = {"uint", 4, 4, 0, 32}; return true;  // R32G32B32A32_SFLOAT
		case 98: f = {"uint", 4, 1, 4, 32}; return true;   // R32_UINT
		case 101: f = {"uint", 4, 2, 4, 32}; return true;  // R32G32_UINT
		case 104: f = {"uint", 4, 3, 4, 32}; return true;  // R32G32B32_UINT
		case 107: f = {"uint", 4, 4, 4, 32}; return true;  // R32G32B32A32_UINT
		case 76: f = {"ushort", 2, 1, 1, 16}; return true; // R16_SFLOAT
		case 83: f = {"ushort", 2, 2, 1, 16}; return true; // R16G16_SFLOAT
		case 97: f = {"ushort", 2, 4, 1, 16}; return true; // R16G16B16A16_SFLOAT
		case 70: f = {"ushort", 2, 1, 2, 16}; return true; // R16_UNORM
		case 77: f = {"ushort", 2, 2, 2, 16}; return true; // R16G16_UNORM
		case 91: f = {"ushort", 2, 4, 2, 16}; return true; // R16G16B16A16_UNORM
		case 71: f = {"ushort", 2, 1, 3, 16}; return true; // R16_SNORM
		case 78: f = {"ushort", 2, 2, 3, 16}; return true; // R16G16_SNORM
		case 92: f = {"ushort", 2, 4, 3, 16}; return true; // R16G16B16A16_SNORM
		case 74: f = {"ushort", 2, 1, 4, 16}; return true; // R16_UINT
		case 81: f = {"ushort", 2, 2, 4, 16}; return true; // R16G16_UINT
		case 95: f = {"ushort", 2, 4, 4, 16}; return true; // R16G16B16A16_UINT
		case 9: f = {"uchar", 1, 1, 2, 8}; return true;    // R8_UNORM
		case 16: f = {"uchar", 1, 2, 2, 8}; return true;   // R8G8_UNORM
		case 37: f = {"uchar", 1, 4, 2, 8}; return true;   // R8G8B8A8_UNORM
		case 38: f = {"uchar", 1, 4, 3, 8}; return true;   // R8G8B8A8_SNORM
		case 41: f = {"uchar", 1, 4, 4, 8}; return true;   // R8G8B8A8_UINT
		case 64: f = {"uint", 4, 1, 5, 10}; return true;   // A2B10G10R10_UNORM_PACK32
		case 122: f = {"uint", 4, 1, 6, 0}; return true;   // B10G11R11_UFLOAT_PACK32
		default: return false;
	}
}

// vsperf (opt-in KYTY_MSL_VS_PERF=1): KK's own vertex-fetch form. (1) The root buffer is read
// through an integer address (`(ulong)&kkroot.contents[0]`, as KK prints it): Apple's compiler
// then hoists the vertex-buffer pointer / stride / count reads into the per-draw constant program;
// through a `constant uchar*` derived from the argument it reloaded them for every vertex (G-buffer
// 9d2710dd: 20 root loads per vertex, spills). (2) The attribute is loaded unconditionally and the
// out-of-range select happens afterwards (KK: load, then `oob ? 0 : v`); `oob ? 0 : *(p)` makes
// the load conditional. Same values: the vertex buffer is bound with KK's robustness padding.
bool VsPerfFetch() {
	static const bool on = [] {
		const char* e = std::getenv("KYTY_MSL_VS_PERF");
		return e != nullptr && std::strcmp(e, "0") != 0;
	}();
	return on;
}

// vsperf: varying width for a read mask (KYTY_MSL_KK_PS_READS / "// kk-bda ps_reads"): floatN with N =
// highest read component + 1, at least 2 (component indexing). Metal stores every component of a VS
// output the FS declares, read or not (G-buffer 9d2710dd: 53 -> 43 vertex_store per vertex; KK: 41).
// VS and FS derive N from the same mask, so the interface types agree.
uint32_t NarrowWidth(uint32_t mask) {
	uint32_t n = 0;
	for (uint32_t m = mask & 0xfu; m != 0u; m >>= 1u)
		n++;
	return std::max(2u, n);
}

// Statements filling `float4|uint4 name` for one attribute (see KkVertexFormat).
std::string KkVertexFetch(const std::string& name, const KkVtxFmt& f, uint32_t slot,
                          uint32_t binding) {
	const bool perf = VsPerfFetch();
	const auto rt   = [&](const char* type, uint32_t off) {
        return perf ? fmt::format("*(constant {}*)(kk_rtu + ulong({}u))", type, off)
                      : fmt::format("*(constant {}*)(kk_rt + {}u)", type, off);
	};
	const std::string vt = f.comps == 1 ? f.elem : fmt::format("{}{}", f.elem, f.comps);
	const std::string load =
	    f.comps == 1 ? fmt::format("(*(((constant {}*)kk_vb) + kk_el))", f.elem)
	                 : fmt::format("{}(*(constant packed_{}*)(((constant {}*)kk_vb) + kk_el))", vt,
	                               vt, f.elem);
	std::string s = "{\n";
	s += fmt::format("    const ulong kk_vb = {};\n", rt("ulong", 136u + 8u * slot));
	s += fmt::format("    const uint kk_el = gl_vertex_id * ({} / {}u);\n",
	                 rt("uint", 8u + 4u * binding), f.size);
	s += fmt::format("    const bool kk_oob = {} < gl_vertex_id;\n", rt("uint", 392u + 4u * slot));
	if (perf) {
		s += fmt::format("    const {} kk_ld = {};\n", vt, load);
		s += fmt::format("    const {} r = kk_oob ? {}(0u) : kk_ld;\n", vt, vt);
	} else {
		s += fmt::format("    const {} r = kk_oob ? {}(0u) : {};\n", vt, vt, load);
	}
	std::string c[4];
	const char* sw = "xyzw";
	for (uint32_t i = 0; i < 4; i++)
		c[i] = f.comps == 1 ? std::string("r") : fmt::format("r.{}", sw[i]);
	std::string v[4];
	const bool  isint = f.kind == 4;
	for (uint32_t i = 0; i < 4; i++)
		v[i] = i == 3 ? (isint ? "1u" : "1.0f") : (isint ? "0u" : "0.0f");
	switch (f.kind) {
		case 0:
			for (uint32_t i = 0; i < f.comps; i++)
				v[i] = "as_type<float>(" + c[i] + ")";
			break;
		case 1:
			for (uint32_t i = 0; i < f.comps; i++)
				v[i] = "float(as_type<half>(" + c[i] + "))";
			break;
		case 2:
			for (uint32_t i = 0; i < f.comps; i++)
				v[i] = fmt::format("float({}) / float({:.1f})", c[i], double((1u << f.bits) - 1u));
			break;
		case 3: {
			const char* st = f.bits == 16 ? "short" : "char";
			for (uint32_t i = 0; i < f.comps; i++)
				v[i] = fmt::format("fmax(float(as_type<{}>({})) / float({:.1f}), float(-1.0))", st,
				                   c[i], double((1u << (f.bits - 1u)) - 1u));
			break;
		}
		case 4:
			for (uint32_t i = 0; i < f.comps; i++)
				v[i] = "uint(" + c[i] + ")";
			break;
		case 5:
			v[0] = "float(r & uint(1023u)) / float(1023.0)";
			v[1] = "float((r << uint(12u)) >> uint(22u)) / float(1023.0)";
			v[2] = "float((r << uint(2u)) >> uint(22u)) / float(1023.0)";
			v[3] = "float(r >> uint(30u)) / float(3.0)";
			break;
		case 6:
			v[0] = "float(as_type<half>(ushort((r & uint(2047u)) << uint(4u))))";
			v[1] = "float(as_type<half>(ushort((r & uint(4192256u)) >> uint(7u))))";
			v[2] = "float(as_type<half>(ushort((r & uint(4290772992u)) >> uint(17u))))";
			break;
		default: break;
	}
	s += "    {\n      #pragma METAL fp math_mode(fast)\n";
	s += fmt::format("      {} = {}4({}, {}, {}, {});\n    }}\n", name, isint ? "uint" : "float",
	                 v[0], v[1], v[2], v[3]);
	s += "  }";
	return s;
}

} // namespace

namespace {
// mslrest4: aggressive (control-dependence aware) dead code for the kk-bda PS read set. Mesa/NIR
// (KK) removes a branch whose region does no live work (no live side effect, no live value leaving
// it through a phi), and with it the values that only fed its condition -- e.g. a varying read only
// by such conditions; KK's linked VS then drops that varying and the attribute loads that fed it,
// renumbering the attribute slots. Live roots: side effects / void instructions except Reference
// (pins a branch condition; emits nothing) and except the given dead operands (exports to no
// attachment). Returns the live instruction set. Opt-in KYTY_MSL_KK_PS_ADCE=1 (default: the old
// use-only liveness).
std::set<const IR::Inst*> PsLiveAdce(const IR::Program&                                  p,
                                     const std::set<std::pair<const IR::Inst*, size_t>>& dead_use) {
	const size_t                       n = p.blocks.size();
	std::map<const IR::Block*, size_t> bi;
	for (size_t i = 0; i < n; i++)
		bi[p.blocks[i]] = i;
	std::vector<std::vector<size_t>> succ(n);
	for (size_t i = 0; i < n; i++)
		for (auto* s: p.blocks[i]->ImmSuccessors())
			if (auto it = bi.find(s); it != bi.end() && std::find(succ[i].begin(), succ[i].end(),
			                                                      it->second) == succ[i].end())
				succ[i].push_back(it->second);
	// post-dominator sets (bitsets); exits = blocks without successors
	const size_t                       w = (n + 63) / 64;
	std::vector<std::vector<uint64_t>> pd(n, std::vector<uint64_t>(w, ~0ull));
	for (size_t i = 0; i < n; i++)
		if (succ[i].empty()) {
			std::fill(pd[i].begin(), pd[i].end(), 0ull);
			pd[i][i / 64] |= 1ull << (i % 64);
		}
	for (bool ch = true; ch;) {
		ch = false;
		for (size_t k = n; k-- > 0;) {
			if (succ[k].empty()) continue;
			std::vector<uint64_t> t(w, ~0ull);
			for (auto s: succ[k])
				for (size_t j = 0; j < w; j++)
					t[j] &= pd[s][j];
			t[k / 64] |= 1ull << (k % 64);
			if (t != pd[k]) {
				pd[k] = t;
				ch    = true;
			}
		}
	}
	const auto cnt = [&](const std::vector<uint64_t>& v) {
		size_t c = 0;
		for (auto x: v)
			c += static_cast<size_t>(__builtin_popcountll(x));
		return c;
	};
	std::vector<size_t> ipdom(n, SIZE_MAX);
	for (size_t i = 0; i < n; i++) {
		const size_t c = cnt(pd[i]);
		for (size_t d = 0; d < n && c > 1; d++)
			if (d != i && ((pd[i][d / 64] >> (d % 64)) & 1u) != 0u && cnt(pd[d]) == c - 1) {
				ipdom[i] = d;
				break;
			}
	}
	// region(b): blocks reachable from b's successors before ipdom(b)
	std::vector<std::vector<size_t>> region(n);
	for (size_t b = 0; b < n; b++) {
		if (succ[b].size() < 2) continue;
		std::vector<char>   seen(n, 0);
		std::vector<size_t> st(succ[b].begin(), succ[b].end());
		while (!st.empty()) {
			const size_t c = st.back();
			st.pop_back();
			if (c == ipdom[b] || seen[c]) continue;
			seen[c] = 1;
			region[b].push_back(c);
			for (auto s: succ[c])
				st.push_back(s);
		}
	}
	std::set<const IR::Inst*>    live;
	std::vector<const IR::Inst*> work;
	std::vector<char> blive(n, 0), brlive(n, 0); // block holds live work; block's branch is live
	const auto        mark = [&](const IR::Inst* i) {
		if (i != nullptr && live.insert(i).second) work.push_back(i);
	};
	for (size_t b = 0; b < n; b++) {
		if (b < p.block_info.size() && !p.block_info[b].indirect_target.IsEmpty()) {
			brlive[b] = 1;
			mark(p.block_info[b].indirect_target.TryInstruction());
		}
		for (const auto& inst: *p.blocks[b])
			if ((inst.MayHaveSideEffects() || inst.GetType() == IR::Type::Void) &&
			    inst.GetOpcode() != IR::ValueOpcode::Reference)
				mark(&inst);
	}
	for (bool grew = true; grew;) {
		grew = false;
		while (!work.empty()) {
			const auto* i = work.back();
			work.pop_back();
			if (auto* par = i->Parent(); par != nullptr)
				if (auto it = bi.find(par); it != bi.end()) blive[it->second] = 1;
			for (size_t a = 0; a < i->NumArgs(); a++) {
				if (dead_use.contains({i, a})) continue;
				if (!i->Arg(a).IsImmediate()) mark(i->Arg(a).TryInstruction());
			}
		}
		for (size_t b = 0; b < n; b++) {
			if (brlive[b] || succ[b].size() < 2) continue;
			bool need = false;
			for (auto c: region[b])
				need = need || blive[c] || brlive[c];
			if (!need) { // a live phi taking a value from this branch's region (or the branch block
				         // itself)
				std::set<const IR::Block*> rb {p.blocks[b]};
				for (auto c: region[b])
					rb.insert(p.blocks[c]);
				for (const auto* li: live)
					if (li->GetOpcode() == IR::ValueOpcode::Phi)
						for (size_t k = 0; k < li->NumPhiBlocks() && !need; k++)
							need = rb.contains(li->PhiBlock(k));
			}
			if (need) {
				brlive[b] = 1;
				blive[b]  = 1;
				grew      = true;
				if (b < p.block_info.size())
					mark(p.block_info[b].condition.IsEmpty() ||
					             p.block_info[b].condition.IsImmediate()
					         ? nullptr
					         : p.block_info[b].condition.TryInstruction());
			}
		}
		grew = grew || !work.empty();
	}
	return live;
}
} // namespace

uint32_t KkLiveComponents(const IR::Inst&                                 composite,
                          const std::function<uint32_t(const IR::Inst&)>& user_mask) {
	uint32_t live = 0u;
	for (const auto& u: composite.Uses())
		live |= user_mask(*u.user);
	return live & 0xfu;
}

struct GfxIo {
	bool                            vs = false, ps = false, kk = false;
	const ShaderVertexInputInfo*    vinfo = nullptr;
	const ShaderPixelInputInfo*     pinfo = nullptr;
	std::map<uint32_t, VKind>       vkind;     // vertex attribute location -> scalar class
	std::map<uint32_t, uint32_t>    ps_loc;    // pixel parameter attr -> interface location
	std::map<uint32_t, std::string> ps_member; // attr -> stage_in member
	// mslfix: KYTY_PS_PER_VERTEX=unpack field slots: (pixel input, field code) -> (interface
	// location, component)
	std::map<std::pair<uint32_t, uint32_t>, std::pair<uint32_t, uint32_t>> ps_field;
	uint32_t                                                               clip_count     = 0;
	bool                                                                   has_valid_mask = false;
	bool                     has_instance = false, has_base_instance = false;
	std::vector<std::string> params;
	std::string              ret = "return out;";
	std::string              out_type, entry_attr, entry_kw;
	std::string              bary_name[3]; // 0 smooth, 1 noperspective
	// mslgfx kk-bda VS: KK vertex pulling (location -> fetch statements), zero-position guard
	// plane, clip_z
	std::map<uint32_t, std::string> kk_fetch;
	std::set<uint32_t>           kk_drop_exports; // parameter exports KK's linked VS does not write
	std::map<uint32_t, uint32_t> kk_export_mask; // parameter export -> component mask kept (!= 0xf)
	std::map<uint32_t, uint32_t> narrow_p; // vsperf VS: parameter export -> floatN width (< 4)
	std::map<uint32_t, uint32_t> narrow_l, narrow_f; // vsperf PS: L<loc> / F<loc> input -> floatN width
	uint32_t                     kk_guard_plane = UINT32_MAX;
	std::set<uint32_t>
	    kk_drop_mrt; // PS MRT exports to no colour attachment (KYTY_MSL_KK_COLOR_MASK)
	// mslfix2: KK blends in the fragment shader (kk_shader.c kk_lower_fs_blend = nir_lower_blend on
	// every FS; Metal blending and write masks stay off), so a replacement FS must do the same: per
	// MRT index the pipeline's VkPipelineColorBlendAttachmentState + format class, from
	// KYTY_MSL_KK_BLEND.
	struct KkBlend {
		uint32_t en = 0, sc = 1, dc = 0, co = 0, sa = 1, da = 0, ao = 0, wm = 15;
		char     cls = 'f';
		uint32_t nch = 4;
	};
	std::map<uint32_t, KkBlend>
	    kk_blend; // only MRTs that need a dst read (blend or partial write mask)
};

namespace {

// mslfix2: nir_lower_blend (nir_color_blend + nir_color_mask) for one colour store, in the same op
// order: inputs clamped to the format, dst channels missing from the format read 0 (alpha 1), per
// channel op(src*sf, dst*df), then the write mask picks blended or dst. KK prints these NIR ops
// outside any math_mode pragma = its library's fast mode.
std::string KkBlendStore(const GfxIo::KkBlend& k, uint32_t idx) {
	const bool replace = k.co == 0 && k.sc == 1 && k.dc == 0 && k.ao == 0 && k.sa == 1 && k.da == 0;
	const bool blend   = k.en != 0 && k.cls != 'i' && !replace;
	std::string r      = "{\n#pragma METAL fp math_mode(fast)\n";
	r += fmt::format("const float4 kd = fin.kk_d{};\n", idx);
	if (blend) {
		r += k.cls == 'u'   ? "const float4 ks = saturate(xv);\n"
		     : k.cls == 's' ? "const float4 ks = clamp(xv, -1.0f, 1.0f);\n"
		                    : "const float4 ks = xv;\n";
		r += fmt::format("const float4 kq = float4({}, {}, {}, {});\n", k.nch > 0 ? "kd.x" : "0.0f",
		                 k.nch > 1 ? "kd.y" : "0.0f", k.nch > 2 ? "kd.z" : "0.0f",
		                 k.nch > 3 ? "kd.w" : "1.0f");
		auto fv = [](uint32_t f, uint32_t ch) -> std::string { // Vk factor without the inversion
			const char* cc = "xyzw";
			switch (f) {
				case 1: return "1.0f";
				case 2: return fmt::format("ks.{}", cc[ch]);
				case 4: return fmt::format("kq.{}", cc[ch]);
				case 6: return "ks.w";
				case 8: return "kq.w";
				case 14: return ch < 3 ? "fmin(ks.w, (1.0f - kq.w))" : "1.0f";
				default: return "0.0f";
			}
		};
		auto factor = [&](uint32_t f, uint32_t ch, const std::string& x) -> std::string {
			if (f == 0) return "0.0f";
			if (f == 3 || f == 5 || f == 7 || f == 9)
				return fmt::format("({} * (-{} + 1.0f))", x, fv(f - 1, ch));
			return fmt::format("({} * {})", x, fv(f, ch));
		};
		const char* cc = "xyzw";
		std::string ch[4];
		for (uint32_t c = 0; c < 4; c++) {
			const uint32_t    op = c < 3 ? k.co : k.ao, sf = c < 3 ? k.sc : k.sa,
			                  df = c < 3 ? k.dc : k.da;
			const std::string s = fmt::format("ks.{}", cc[c]), d = fmt::format("kq.{}", cc[c]);
			if (op == 3)
				ch[c] = fmt::format("fmin({}, {})", s, d);
			else if (op == 4)
				ch[c] = fmt::format("fmax({}, {})", s, d);
			else if (op == 0 && (sf == 2 || sf == 4 || sf == 6 || sf == 8) && df == sf + 1) {
				// mslfix3: nir_opt_algebraic '~' rule (a*b) + (c*(1-b)) -> flrp(c, a, b), printed
				// by KK as mix(c, a, b) (isatest blend_alpha_f16: 1-ulp f16 words vs the unfused
				// form).
				ch[c] = fmt::format("mix({}, {}, {})", d, s, fv(sf, c));
			} else if (op == 0 && (df == 2 || df == 4 || df == 6 || df == 8) && sf == df + 1) {
				ch[c] = fmt::format("mix({}, {}, {})", s, d, fv(df, c));
			} else if (op == 0 && sf == 1 && (df == 3 || df == 5 || df == 7 || df == 9) &&
			           fv(df - 1, c) == s) {
				// a + c*(1-a) -> flrp(c, 1.0, a) (KK alpha of SRC_ALPHA/ONE_MINUS_SRC_ALPHA with
				// srcA ONE: mix(d.w, 1.0, s.w))
				ch[c] = fmt::format("mix({}, 1.0f, {})", d, s);
			} else if (op == 0 && df == 1 && (sf == 3 || sf == 5 || sf == 7 || sf == 9) &&
			           fv(sf - 1, c) == d) {
				ch[c] = fmt::format("mix({}, 1.0f, {})", s, d);
			} else {
				const std::string ps = factor(sf, c, s), pd = factor(df, c, d);
				ch[c] = op == 0   ? fmt::format("({} + {})", ps, pd)
				        : op == 1 ? fmt::format("({} - {})", ps, pd)
				                  : fmt::format("({} - {})", pd, ps);
			}
		}
		r += fmt::format("const float4 kb = float4({}, {}, {}, {});\n", ch[0], ch[1], ch[2], ch[3]);
	} else {
		r += "const float4 kb = xv;\n";
	}
	r += fmt::format("out.c{} = float4({}, {}, {}, {});\n}}", idx, (k.wm & 1u) ? "kb.x" : "kd.x",
	                 (k.wm & 2u) ? "kb.y" : "kd.y", (k.wm & 4u) ? "kb.z" : "kd.z",
	                 (k.wm & 8u) ? "kb.w" : "kd.w");
	return r;
}

IR::InterpolationMode ParamInterp(const MslCtx& c, const GfxIo& g, uint32_t loc_attr) {
	if (!GetCodegenOptions().interp_modes) return IR::InterpolationMode::Unknown;
	bool any = false, linear = false, persp = false, center = false, centroid = false,
	     sample = false;
	for (const auto* block: c.program.blocks) {
		for (const auto& inst: *block) {
			if (inst.GetOpcode() != IR::ValueOpcode::GetAttribute || inst.Arg(0).U32() != loc_attr)
				continue;
			const auto mode = static_cast<IR::InterpolationMode>(inst.Flags<uint32_t>());
			if (mode == IR::InterpolationMode::Unknown) return IR::InterpolationMode::Unknown;
			any                                                              = true;
			(IsLinear(mode) ? linear : persp)                                = true;
			(IsCentroid(mode) ? centroid : IsSample(mode) ? sample : center) = true;
		}
	}
	(void)g;
	if (!any || (linear && persp) || (sample && (center || centroid)))
		return IR::InterpolationMode::Unknown;
	if (sample)
		return linear ? IR::InterpolationMode::LinearSample
		              : IR::InterpolationMode::PerspectiveSample;
	if (centroid && !center)
		return linear ? IR::InterpolationMode::LinearCentroid
		              : IR::InterpolationMode::PerspectiveCentroid;
	return linear ? IR::InterpolationMode::LinearCenter : IR::InterpolationMode::PerspectiveCenter;
}

} // namespace

bool GfxIsStage(ShaderType t) {
	return t == ShaderType::Vertex || t == ShaderType::Pixel;
}

std::shared_ptr<GfxIo> GfxSetup(MslCtx& c) {
	auto        g = std::make_shared<GfxIo>();
	const auto& p = c.program;
	g->vs         = p.stage == ShaderType::Vertex;
	g->kk         = c.kk_bda;
	g->ps         = p.stage == ShaderType::Pixel;
	if (g->vs)
		g->vinfo = c.input_info.vertex;
	else
		g->pinfo = c.input_info.pixel;
	auto&       b     = c.b;
	const auto& info  = p.info;
	auto        unsup = [&](const std::string& s) { c.unsupported[s]++; };

	for (const auto* blk: p.blocks)
		for (const auto& inst: *blk)
			if (inst.GetOpcode() == IR::ValueOpcode::SetAttribute && g->ps &&
			    p.export_info.at(inst.Flags<IR::ExportFlags>().index).vm)
				g->has_valid_mask = true;

	if (g->vs) {
		if (g->vinfo->clip_space.enabled) unsup("gfx:vs_clip_space_transform");
		if (g->vinfo->user_clip_planes != 0) unsup("gfx:user_clip_planes");
		if (c.kk_bda) {
			// KK vertex pulling; KYTY_MSL_KK_VA="loc:binding:vkformat,..." = the pipeline's vertex
			// input state.
			std::map<uint32_t, std::pair<uint32_t, uint32_t>> va;
			if (const char* e = std::getenv("KYTY_MSL_KK_VA"); e != nullptr) {
				uint32_t l = 0, bnd = 0, f = 0;
				for (const char* q = e; *q != 0;) {
					int n = 0;
					if (std::sscanf(q, "%u:%u:%u%n", &l, &bnd, &f, &n) != 3) break;
					va[l] = {bnd, f};
					q += n;
					if (*q == ',') q++;
				}
			}
			// KK links the VS to its FS: parameters the FS does not read are not written, and
			// attribute loads that only fed them disappear, which renumbers the attribute slots.
			// KYTY_MSL_KK_VS_OUTS="loc,..." = the parameter locations KK's VS writes (its
			// user(vary_NN) outputs); B drops the other parameter exports and the instructions only
			// they use, and numbers slots over the attributes still read. "loc[:n],...": n =
			// components KK keeps (its trimmed floatN/uintN vary_NN; NIR drops unread components).
			// mslrest3: KYTY_MSL_KK_PS_READS="loc:mask,..." (the linked B PS's "// kk-bda ps_reads"
			// line) = the components the PS reads per interface location. KK's nir_opt_varyings
			// repacks the varyings it keeps (vary_NN do not map to locations), but the B pair is
			// linked to itself, so only the attribute slots must agree with KK: drop what the PS
			// does not read and number the slots over the attributes still live. Preferred over
			// KYTY_MSL_KK_VS_OUTS.
			if (const char* e = std::getenv("KYTY_MSL_KK_PS_READS"); e != nullptr) {
				std::map<uint32_t, uint32_t> reads;
				for (const char* q = e; *q != 0;) {
					char*      end = nullptr;
					const auto l   = static_cast<uint32_t>(std::strtoul(q, &end, 10));
					if (end == q || *end != ':') break;
					reads[l] = static_cast<uint32_t>(std::strtoul(end + 1, &end, 16)) & 0xfu;
					q        = *end == ',' ? end + 1 : end;
				}
				for (const auto& o: info.outputs) {
					if (o.kind != IR::StageOutputKind::Parameter) continue;
					const auto k = reads.find(o.location);
					if (k == reads.end() || k->second == 0u)
						g->kk_drop_exports.insert(o.index);
					else if (k->second != 0xfu)
						g->kk_export_mask[o.index] = k->second;
				}
			} else if (const char* e = std::getenv("KYTY_MSL_KK_VS_OUTS"); e != nullptr) {
				std::map<uint32_t, uint32_t> keep;
				for (const char* q = e; *q != 0;) {
					char*      end = nullptr;
					const auto v   = static_cast<uint32_t>(std::strtoul(q, &end, 10));
					if (end == q) break;
					uint32_t n = 4;
					if (*end == ':') n = static_cast<uint32_t>(std::strtoul(end + 1, &end, 10));
					keep[v] = std::clamp(n, 1u, 4u);
					q       = *end == ',' ? end + 1 : end;
				}
				for (const auto& o: info.outputs) {
					if (o.kind != IR::StageOutputKind::Parameter) continue;
					const auto k = keep.find(o.location);
					if (k == keep.end())
						g->kk_drop_exports.insert(o.index);
					else if (k->second < 4u)
						g->kk_export_mask[o.index] = (1u << k->second) - 1u;
				}
			}
			std::set<const IR::Inst*> dead;
			std::set<std::pair<const IR::Inst*, size_t>>
			    dead_use; // operands of kept exports' dropped components
			// bcover: a component of an exported CompositeConstructU32x4 is dead only when NO user
			// of the composite reads it. A shader can export one vector to two parameters (e.g. loc
			// 1 .xyz and loc 2 .w of the same uint4); the old per-export rule unioned the dropped
			// components of both and zeroed a live UV, which also dropped its attribute load (slot
			// renumbering).
			auto export_mask = [&](const IR::Inst& inst) -> uint32_t { // components a user reads
				if (inst.GetOpcode() != IR::ValueOpcode::SetAttribute) return 0xfu;
				const auto& exp = p.export_info.at(inst.Flags<IR::ExportFlags>().index);
				if (exp.kind != IR::ExportTargetKind::Parameter) return 0xfu;
				if (g->kk_drop_exports.contains(exp.index)) return 0u;
				const auto nc = g->kk_export_mask.find(exp.index);
				if (nc == g->kk_export_mask.end() || exp.compr) return 0xfu;
				return nc->second;
			};
			for (const auto* blk: p.blocks)
				for (const auto& inst: *blk)
					if (inst.GetOpcode() == IR::ValueOpcode::SetAttribute) {
						const auto& exp = p.export_info.at(inst.Flags<IR::ExportFlags>().index);
						if (exp.kind != IR::ExportTargetKind::Parameter) continue;
						if (g->kk_drop_exports.contains(exp.index)) {
							dead.insert(&inst);
							continue;
						}
						const auto nc = g->kk_export_mask.find(exp.index);
						if (nc == g->kk_export_mask.end() || exp.compr || inst.Arg(0).IsImmediate())
							continue;
						const IR::Inst* v = inst.Arg(0).TryInstruction();
						if (v == nullptr ||
						    v->GetOpcode() != IR::ValueOpcode::CompositeConstructU32x4)
							continue;
						const uint32_t live = KkLiveComponents(*v, export_mask);
						for (size_t i = 0; i < v->NumArgs(); i++)
							if (((live >> i) & 1u) == 0u) dead_use.insert({v, i});
					}
			for (bool grew = true;
			     grew;) { // also unused values (NIR DCE drops an attribute load nothing reads)
				grew = false;
				for (const auto* blk: p.blocks)
					for (const auto& inst: *blk) {
						if (dead.contains(&inst) || inst.MayHaveSideEffects() ||
						    inst.GetType() == IR::Type::Void)
							continue;
						bool all = true;
						for (const auto& u: inst.Uses())
							all = all &&
							      (dead.contains(u.user) || dead_use.contains({u.user, u.operand}));
						if (all) {
							dead.insert(&inst);
							grew = true;
						}
					}
			}
			std::set<uint32_t> read_locs;
			for (const auto* blk: p.blocks)
				for (const auto& inst: *blk)
					if (inst.GetOpcode() == IR::ValueOpcode::GetAttribute && !dead.contains(&inst))
						read_locs.insert(inst.Arg(0).U32());
			uint32_t slot     = 0;
			bool     any_attr = false;
			for (const auto loc: read_locs) { // slots = rank among the attributes the VS reads
				                              // (kk_cmd_draw.c attribs_read)
				const auto it = va.find(loc);
				KkVtxFmt   f {};
				if (it == va.end() || !KkVertexFormat(it->second.second, f)) {
					unsup(it == va.end()
					          ? "kk-bda: vertex attribute without KYTY_MSL_KK_VA"
					          : fmt::format("kk-bda: vertex format {}", it->second.second));
					slot++;
					continue;
				}
				g->vkind[loc] = f.kind == 4 ? VKind::Uint : VKind::Float;
				g->kk_fetch[loc] =
				    KkVertexFetch(fmt::format("vin_a{}", loc), f, slot++, it->second.first);
				any_attr = true;
			}
			bool has_vid = false;
			for (const auto& in: info.inputs) {
				if (in.kind == IR::StageInputKind::Parameter) continue;
				if (in.kind == IR::StageInputKind::VertexIndex)
					has_vid = true;
				else if (in.kind == IR::StageInputKind::InstanceIndex)
					g->has_instance = true;
				else if (in.kind == IR::StageInputKind::BaseInstance)
					g->has_base_instance = true;
				else
					unsup(fmt::format("gfx:vs_input_kind{}", static_cast<int>(in.kind)));
			}
			if (has_vid || any_attr) g->params.push_back("uint gl_vertex_id [[vertex_id]]");
		} else {
			b.Open("struct VIn");
			bool any_attr = false;
			for (const auto& in: info.inputs) {
				if (in.kind == IR::StageInputKind::Parameter) {
					const auto k          = VertexKind(*g->vinfo, in.location);
					g->vkind[in.location] = k;
					b.Line(fmt::format("{} a{} [[attribute({})]];",
					                   k == VKind::Float  ? "float4"
					                   : k == VKind::Uint ? "uint4"
					                                      : "int4",
					                   in.location, in.location));
					any_attr = true;
				} else if (in.kind == IR::StageInputKind::VertexIndex) {
					g->params.push_back("uint gl_vertex_id [[vertex_id]]");
				} else if (in.kind == IR::StageInputKind::InstanceIndex) {
					g->has_instance = true;
				} else if (in.kind == IR::StageInputKind::BaseInstance) {
					g->has_base_instance = true;
				} else {
					unsup(fmt::format("gfx:vs_input_kind{}", static_cast<int>(in.kind)));
				}
			}
			if (!any_attr) b.Line("float4 unused_ [[attribute(0)]];"); // keep the struct valid
			b.Close(";");
			if (any_attr) g->params.insert(g->params.begin(), "VIn vin [[stage_in]]");
		}
		if (g->has_instance) g->params.push_back("uint gl_instance_id [[instance_id]]");
		if (g->has_instance || g->has_base_instance)
			g->params.push_back("uint gl_base_instance [[base_instance]]");
		// outputs
		b.Open("struct VOut");
		// mslaudit (NXNative "Position invariance across shaders"): a depth pre-pass VS and the
		// G-buffer VS compute the same position with different code; without [[invariant]] +
		// preserveInvariance (-fpreserve-invariance) Metal may round them differently and
		// EQUAL/LEQUAL depth tests fail in blocks. Libraries holding this VS must be compiled with
		// MTLCompileOptions.preserveInvariance = YES (else the attribute is ignored).
		// KYTY_MSL_INVARIANT=0 turns it off.
		{
			const char* inv = std::getenv("KYTY_MSL_INVARIANT");
			b.Line(inv != nullptr && std::strcmp(inv, "0") == 0
			           ? "float4 pos [[position]];"
			           : "float4 pos [[position, invariant]];");
		}
		std::set<uint32_t> seen_p;
		for (const auto& o: info.outputs) {
			switch (o.kind) {
				case IR::StageOutputKind::Parameter:
					if (seen_p.insert(o.index).second) {
						uint32_t   n = 4;
						const auto m = g->kk_export_mask.find(o.index);
						if (c.kk_bda && VsPerfFetch() && m != g->kk_export_mask.end() && m->second != 0u)
							n = NarrowWidth(m->second);
						if (n < 4u) g->narrow_p[o.index] = n;
						b.Line(fmt::format("float{} p{} [[user(locn{})]];", n, o.index, o.location));
					}
					break;
				case IR::StageOutputKind::ClipDistance:
					g->clip_count = std::max(g->clip_count, o.index + 1);
					break;
				case IR::StageOutputKind::PointSize: b.Line("float psize [[point_size]];"); break;
				case IR::StageOutputKind::Layer:
					b.Line("uint layer [[render_target_array_index]];");
					break;
				case IR::StageOutputKind::ViewportIndex:
					b.Line("uint vport [[viewport_array_index]];");
					break;
				case IR::StageOutputKind::CullDistance: unsup("gfx:cull_distance"); break;
				default: break;
			}
		}
		// kk-bda: the plane Kyty's SPIR-V reserves for the zero-position clip guard
		// (spirvEmitterModule.cpp).
		bool has_pos = false;
		for (const auto& o: info.outputs)
			has_pos |= o.kind == IR::StageOutputKind::Position;
		if (c.kk_bda && GetCodegenOptions().clip_guard && g->clip_count < 8u && has_pos)
			g->kk_guard_plane = g->clip_count++;
		if (g->clip_count) b.Line(fmt::format("float clip [[clip_distance]] [{}];", g->clip_count));
		b.Close(";");
		g->out_type = "VOut";
		g->entry_kw = "vertex";
		g->ret      = "return out;";
		if (c.kk_bda) {
			// Guard computed as KK prints Kyty's all(position == 0) (|w|+|z|+|y|+|x| == 0,
			// relaxed), then KK's clip_z (depth clip control) on the final position: z' = z + w*c -
			// z*c (root 536, fast math = KK's VS default).
			b.Raw("static inline float kk_clip_guard(float4 p) {\n  float s;\n  {\n    #pragma "
			      "METAL fp math_mode(relaxed)\n"
			      "    s = precise::abs(p.w) + precise::abs(p.z);\n    s = s + "
			      "precise::abs(p.y);\n    s = s + precise::abs(p.x);\n  }\n"
			      "  return (!isnan(s) && s == 0.0f) ? -1.0f : 0.0f;\n}\n"
			      "static inline float kk_clip_z(float4 p, float c) {\n  #pragma METAL fp "
			      "math_mode(fast)\n  float t = p.w * c + p.z;\n  return -(p.z * c) + t;\n}\n");
			std::string r = "{ ";
			if (g->kk_guard_plane != UINT32_MAX)
				r += fmt::format("out.clip[{}] = kk_clip_guard(out.pos); ", g->kk_guard_plane);
			if (has_pos)
				r += "out.pos.z = kk_clip_z(out.pos, *(constant float*)((constant uchar*)&kkroot + "
				     "536u)); ";
			g->ret = r + "return out; }";
		}
	} else {
		const auto& pi = *g->pinfo;
		if (pi.dual_source_blending) unsup("gfx:dual_source");
		std::vector<uint32_t> active;
		for (const auto& in: info.inputs)
			if (in.kind == IR::StageInputKind::Parameter) active.push_back(in.location);
		b.Open("struct FIn");
		bool any_param = false;
		// One member per interface location. Inputs that alias one location (flat + smooth read of
		// one export, the "raw vertex" aliasing of ShaderPixelParameterLocation) share it:
		// smooth/interpolated wins, all-flat stays flat.
		std::map<uint32_t, std::vector<uint32_t>> by_loc;
		for (const auto& in: info.inputs) {
			if (in.kind != IR::StageInputKind::Parameter) continue;
			// per_vertex (raw vertex reads) are read as the interpolated value =
			// KYTY_PS_PER_VERTEX=0 semantics (approximation)
			const auto loc =
			    ShaderPixelParameterLocation(pi, {active.data(), active.size()}, in.location);
			g->ps_loc[in.location] = loc;
			by_loc[loc].push_back(in.location);
		}
		for (const auto& [loc, attrs]: by_loc) {
			bool all_flat = true;
			for (auto a: attrs)
				all_flat = all_flat && ShaderPixelParameterIsFlat(pi, a);
			std::string q;
			if (all_flat)
				q = ", flat";
			else {
				uint32_t pick = attrs[0];
				for (auto a: attrs)
					if (!ShaderPixelParameterIsFlat(pi, a)) {
						pick = a;
						break;
					}
				auto mode = ParamInterp(c, *g, pick);
				bool lin = pi.ps_no_perspective, cen = false, smp = false;
				if (mode != IR::InterpolationMode::Unknown) {
					lin = IsLinear(mode);
					cen = IsCentroid(mode);
					smp = IsSample(mode);
				}
				q = std::string(", ") +
				    (cen   ? "centroid_"
				     : smp ? "sample_"
				           : "center_") +
				    (lin ? "no_perspective" : "perspective");
				if (!cen && !smp && !lin) q.clear();
			}
			b.Line(fmt::format("float4 L{} [[user(locn{}){}]];", loc, loc, q));
			any_param = true;
		}
		// mslfix: unpacked raw-vertex fields (KYTY_PS_PER_VERTEX=unpack, RawVertexUnpack.h) at the
		// spare locations the linked vertex shader exports them to; interpolated like the shader's
		// own I/J (center), Raw fields flat. Mirrors the SPIR-V emitter (spirvEmitterModule.cpp
		// ParameterField inputs).
		for (const auto& in: info.inputs) {
			if (in.kind != IR::StageInputKind::ParameterField) continue;
			const auto slot = ShaderParameterFieldSlot::Unpack(in.location, in.component_count);
			for (uint32_t comp = 0; comp < 4u; comp++)
				if (slot.codes[comp] != 0u)
					g->ps_field[{slot.input, slot.codes[comp]}] = {in.location, comp};
			const char* q = slot.Flat()            ? ", flat"
			                : pi.ps_no_perspective ? ", center_no_perspective"
			                                       : "";
			b.Line(fmt::format("float4 F{} [[user(locn{}){}]];", in.location, in.location, q));
			any_param = true;
		}
		// mslfix2: KYTY_MSL_KK_BLEND="en:sc:dc:co:sa:da:ao:wm:cls:nch;..." per colour attachment
		// (Vk enums; cls f|u|s|i = float/unorm(+srgb)/snorm/integer format, nch = format channels).
		// Unset = old text (no in-shader blend).
		if (const char* e = std::getenv("KYTY_MSL_KK_BLEND"); c.kk_bda && e != nullptr && *e != 0) {
			uint32_t idx = 0;
			for (const char* q = e; *q != 0; idx++) {
				GfxIo::KkBlend kb;
				uint32_t       v[8] = {0, 1, 0, 0, 1, 0, 0, 15};
				const char*    t    = q;
				for (int k = 0; k < 8; k++) {
					char* end = nullptr;
					v[k]      = static_cast<uint32_t>(std::strtoul(t, &end, 0));
					t         = (*end == ':') ? end + 1 : end;
				}
				kb.en = v[0];
				kb.sc = v[1];
				kb.dc = v[2];
				kb.co = v[3];
				kb.sa = v[4];
				kb.da = v[5];
				kb.ao = v[6];
				kb.wm = v[7] & 15u;
				if (*t != 0 && *t != ';') {
					kb.cls = *t;
					t++;
					if (*t == ':') {
						char* end = nullptr;
						kb.nch    = static_cast<uint32_t>(std::strtoul(t + 1, &end, 0));
						t         = end;
					}
				}
				while (*t != 0 && *t != ';')
					t++;
				q                  = (*t == ';') ? t + 1 : t;
				const bool replace = kb.co == 0 && kb.sc == 1 && kb.dc == 0 && kb.ao == 0 &&
				                     kb.sa == 1 && kb.da == 0;
				const bool blend   = kb.en != 0 && kb.cls != 'i' && !replace;
				if (!blend && kb.wm == 15u) continue;
				g->kk_blend[idx] = kb;
			}
			for (const auto& o: info.outputs) {
				if (o.kind != IR::StageOutputKind::Mrt) continue;
				const auto kb = g->kk_blend.find(o.index);
				if (kb == g->kk_blend.end()) continue;
				const bool u = o.index < std::size(pi.target_output_mode) &&
				               pi.target_output_mode[o.index] == 7u;
				if (u || kb->second.cls == 'i') unsup("gfx:kk_blend_int");
				if (pi.alpha_blend_source != ShaderAlphaBlendSource::None)
					unsup("gfx:kk_blend_dual_source");
				const auto& k = kb->second;
				if (k.en != 0) {
					for (uint32_t f: {k.sc, k.dc, k.sa, k.da})
						if ((f >= 10u && f <= 13u) || f >= 15u)
							unsup("gfx:kk_blend_factor_const_or_src1");
					if (k.co > 4u || k.ao > 4u) unsup("gfx:kk_blend_advanced");
				}
				b.Line(fmt::format("float4 kk_d{} [[color({})]];", o.index, o.location));
			}
		}
		if (!any_param) b.Line("float4 unused_ [[user(locn31)]];");
		b.Close(";");
		g->params.push_back("FIn fin [[stage_in]]");
		if (c.kk_bda) { // mslrest3: components read per interface location -> KYTY_MSL_KK_PS_READS
			            // of the linked B VS
			// KYTY_MSL_KK_COLOR_MASK = the pipeline's colour attachments (bit i = colour(i) has a
			// format): Mesa removes fragment outputs without an attachment, and with them the
			// varyings (and VS attribute loads) only they used.
			uint32_t cmask = 0xffu;
			if (const char* e = std::getenv("KYTY_MSL_KK_COLOR_MASK"); e != nullptr && *e != 0)
				cmask = static_cast<uint32_t>(std::strtoul(e, nullptr, 0));
			std::set<const IR::Inst*>                    pdead;
			std::set<std::pair<const IR::Inst*, size_t>> pdead_use;
			for (const auto* blk: p.blocks)
				for (const auto& inst: *blk) {
					if (inst.GetOpcode() != IR::ValueOpcode::SetAttribute) continue;
					const auto& exp = p.export_info.at(inst.Flags<IR::ExportFlags>().index);
					if (exp.kind != IR::ExportTargetKind::Mrt || ((cmask >> exp.index) & 1u) != 0u)
						continue;
					if (exp.index == 1u && pi.alpha_blend_source != ShaderAlphaBlendSource::None)
						continue;
					g->kk_drop_mrt.insert(exp.index);
					pdead_use.insert(
					    {&inst, 0}); // the value; the exec operand still feeds the valid mask
				}
			// mslrest4: KYTY_MSL_KK_WRITE_MASK="m0,m1,.." = the pipeline's colour write mask per
			// attachment (VkPipeline colorWriteMask, bit c = component c): KK drops the masked-off
			// output components, and with them the values (and varyings) that only fed them.
			// Composite operand c (packed: operand c/2, half c%2 of a PackFloat2x16Rtz). (opt-in:
			// set KYTY_MSL_KK_WRITE_MASK; did not explain the 28 slot-mismatch pipes, see
			// handoff-mslrest4.md)
			if (const char* e = std::getenv("KYTY_MSL_KK_WRITE_MASK"); e != nullptr && *e != 0) {
				std::vector<uint32_t> wm;
				for (const char* q = e; *q != 0;) {
					char* end = nullptr;
					wm.push_back(static_cast<uint32_t>(std::strtoul(q, &end, 0)) & 0xfu);
					if (end == q) break;
					q = *end == ',' ? end + 1 : end;
				}
				for (const auto* blk: p.blocks)
					for (const auto& inst: *blk) {
						if (inst.GetOpcode() != IR::ValueOpcode::SetAttribute) continue;
						const auto& exp = p.export_info.at(inst.Flags<IR::ExportFlags>().index);
						if (exp.kind != IR::ExportTargetKind::Mrt ||
						    g->kk_drop_mrt.contains(exp.index))
							continue;
						if (pi.alpha_blend_source != ShaderAlphaBlendSource::None)
							continue; // dual source: keep it simple
						// export components that reach the attachment: enabled (exp.en; SPIR-V
						// writes constants for the others) and selected by a written attachment
						// component through the export mapping
						const uint32_t a = exp.index < wm.size() ? wm[exp.index] : 0xfu;
						uint32_t       m = 0u;
						for (uint32_t c = 0; c < 4; c++) {
							if (((a >> c) & 1u) == 0u) continue;
							const uint32_t k = exp.index < pi.target_export_mapping.size()
							                       ? pi.target_export_mapping[exp.index].Map(c)
							                       : c;
							m |= k < 4u ? 1u << k : 0u;
						}
						m &= exp.en & 0xfu;
						if (m == 0xfu) continue;
						if (inst.Arg(0).IsImmediate()) continue;
						const IR::Inst* v = inst.Arg(0).TryInstruction();
						if (v == nullptr ||
						    v->GetOpcode() != IR::ValueOpcode::CompositeConstructU32x4)
							continue;
						if (!exp.compr) {
							for (size_t c = 0; c < 4 && c < v->NumArgs(); c++)
								if (((m >> c) & 1u) == 0u) pdead_use.insert({v, c});
							continue;
						}
						for (size_t w = 0; w < 2 && w < v->NumArgs(); w++) {
							const uint32_t hm = (m >> (2u * w)) & 3u;
							if (hm == 0u) {
								pdead_use.insert({v, w});
								continue;
							}
							if (hm == 3u || v->Arg(w).IsImmediate()) continue;
							const IR::Inst* pk = v->Arg(w).TryInstruction();
							if (pk == nullptr ||
							    pk->GetOpcode() != IR::ValueOpcode::PackFloat2x16Rtz ||
							    pk->UseCount() != 1u)
								continue;
							pdead_use.insert({pk, hm == 1u ? 1u : 0u});
						}
					}
			}
			static const bool adce = [] {
				const char* e = std::getenv("KYTY_MSL_KK_PS_ADCE");
				return e != nullptr && std::strcmp(e, "1") == 0;
			}(); // opt-in (mslrest4)
			if (adce) {
				const auto live = PsLiveAdce(p, pdead_use);
				for (const auto* blk: p.blocks)
					for (const auto& inst: *blk)
						if (!live.contains(&inst)) pdead.insert(&inst);
			}
			for (bool grew = !adce; grew;) {
				grew = false;
				for (const auto* blk: p.blocks)
					for (const auto& inst: *blk) {
						if (pdead.contains(&inst) || inst.MayHaveSideEffects() ||
						    inst.GetType() == IR::Type::Void)
							continue;
						bool all = true;
						for (const auto& u: inst.Uses())
							all = all && (pdead.contains(u.user) ||
							              pdead_use.contains({u.user, u.operand}));
						if (all) {
							pdead.insert(&inst);
							grew = true;
						}
					}
			}
			std::map<uint32_t, uint32_t> reads;
			for (const auto* blk: p.blocks)
				for (const auto& inst: *blk) {
					const auto op = inst.GetOpcode();
					if (op == IR::ValueOpcode::GetParameterField && !inst.Uses().empty() &&
					    !pdead.contains(&inst)) {
						const auto f = g->ps_field.find({inst.Arg(0).U32(), inst.Arg(1).U32()});
						if (f != g->ps_field.end())
							reads[f->second.first] |= 1u << f->second.second;
						continue;
					}
					if (op != IR::ValueOpcode::GetAttribute &&
					    op != IR::ValueOpcode::GetInterpolationParameter)
						continue;
					if (inst.Uses().empty() || pdead.contains(&inst)) continue;
					const auto l = g->ps_loc.find(inst.Arg(0).U32());
					if (l == g->ps_loc.end()) continue;
					reads[l->second] |=
					    inst.Arg(1).IsImmediate() ? (1u << (inst.Arg(1).U32() & 3u)) : 0xfu;
				}
			std::string r = "// kk-bda ps_reads";
			for (const auto& [l, m]: reads)
				r += fmt::format(" {}:{:x}", l, m);
			if (VsPerfFetch())
				for (const auto& [l, m]: reads) {
					const uint32_t n = NarrowWidth(m);
					if (m == 0u || n >= 4u) continue;
					if (b.ReplaceFirst(fmt::format("float4 L{} [[user(locn{})", l, l),
					                   fmt::format("float{} L{} [[user(locn{})", n, l, l)))
						g->narrow_l[l] = n;
					if (b.ReplaceFirst(fmt::format("float4 F{} [[user(locn{})", l, l),
					                   fmt::format("float{} F{} [[user(locn{})", n, l, l)))
						g->narrow_f[l] = n;
				}
			b.Raw(r + "\n");
		}
		for (const auto& in: info.inputs) {
			switch (in.kind) {
				case IR::StageInputKind::Parameter: break;
				case IR::StageInputKind::ParameterField: break; // mslfix: FIn F<loc> members
				case IR::StageInputKind::FragCoord:
					g->params.push_back("float4 frag_coord [[position]]");
					break;
				case IR::StageInputKind::FrontFacing:
					g->params.push_back("bool front_facing [[front_facing]]");
					break;
				case IR::StageInputKind::SampleId:
					g->params.push_back("uint sample_id [[sample_id]]");
					break;
				case IR::StageInputKind::PrimitiveId:
					g->params.push_back("uint prim_id [[primitive_id]]");
					break;
				case IR::StageInputKind::Layer:
					g->params.push_back("uint layer_in [[render_target_array_index]]");
					break;
				case IR::StageInputKind::BaryCoordSmooth:
				case IR::StageInputKind::BaryCoordSmoothCentroid:
				case IR::StageInputKind::BaryCoordSmoothSample:
					g->params.push_back("float3 bary_s [[barycentric_coord]]");
					break;
				case IR::StageInputKind::BaryCoordNoPerspective:
				case IR::StageInputKind::BaryCoordNoPerspectiveCentroid:
				case IR::StageInputKind::BaryCoordNoPerspectiveSample:
					g->params.push_back(
					    "float3 bary_n [[barycentric_coord, center_no_perspective]]");
					break;
				default:
					unsup(fmt::format("gfx:ps_input_kind{}", static_cast<int>(in.kind)));
					break;
			}
		}
		// dedupe params added twice (several bary kinds of one family)
		{
			std::set<std::string>    s;
			std::vector<std::string> u;
			for (auto& x: g->params)
				if (s.insert(x).second) u.push_back(x);
			g->params = u;
		}
		b.Open("struct FOut");
		bool any_out = false;
		for (const auto& o: info.outputs) {
			if (o.kind == IR::StageOutputKind::Mrt && g->kk_drop_mrt.contains(o.index)) continue;
			if (o.kind == IR::StageOutputKind::Mrt) {
				const bool u = o.index < std::size(pi.target_output_mode) &&
				               pi.target_output_mode[o.index] == 7u;
				b.Line(fmt::format("{} c{} [[color({})]];", u ? "uint4" : "float4", o.index,
				                   o.location));
				any_out = true;
			} else if (o.kind == IR::StageOutputKind::Depth) {
				b.Line("float depth [[depth(any)]];");
				any_out = true;
			} else if (o.kind == IR::StageOutputKind::SampleMask) {
				b.Line("uint smask [[sample_mask]];");
				any_out = true;
			}
		}
		if (!any_out) b.Line("float4 unused_ [[color(0)]];");
		b.Close(";");
		g->out_type = "FOut";
		g->entry_kw = "fragment";
		bool early  = pi.ps_early_z && !pi.ps_pixel_kill_enable && !pi.ps_depth_export_enable &&
		              !pi.ps_sample_mask_export_enable;
		// KYTY_LOD_STATS_EARLY_TESTS (LodStatsEarlyTests.h): the instrumented module (kk-bda with
		// MipStats records) is tested early when the feedback is its only write and it exports no
		// depth (spirvEmitterModule.cpp).
		if (!early && c.kk_bda && !MipStatsPlainRequested()) {
			bool depth = false;
			for (const auto& o: info.outputs)
				depth |= o.kind == IR::StageOutputKind::Depth;
			early = !depth && LodStatsEarlyTests(c.program, pi);
		}
		g->entry_attr = early ? "[[early_fragment_tests]] " : "";
		g->ret        = g->has_valid_mask ? "{ if (!pixel_valid) discard_fragment(); return out; }"
		                                  : "return out;";
	}
	c.ret = g->ret;
	return g;
}

std::string GfxEntryPrefix(const GfxIo& g) {
	return g.entry_attr + g.entry_kw + " " + g.out_type + " ";
}
const std::vector<std::string>& GfxParams(const GfxIo& g) {
	return g.params;
}
void GfxLocals(MslCtx& c, const GfxIo& g) {
	c.b.Line(g.out_type + " out = {};");
	if (g.ps)
		for (const auto& [i, kb]: g.kk_blend)
			if (!g.kk_drop_mrt.contains(i))
				c.b.Line(fmt::format("out.c{} = fin.kk_d{}; // mslfix2: KK reads dst first "
				                     "(nir_lower_blend fb fetch)",
				                     i, i));
	if (g.vs)
		c.b.Line("const uint lid = 0u; // vertex stage has no simd lane index (the SPIR-V emitter "
		         "answers lane 0 too)");
	if (g.vs && !g.kk_fetch.empty()) {
		c.b.Line(VsPerfFetch() ? "const ulong kk_rtu = (ulong)&kkroot.contents[0];"
		                       : "constant uchar* kk_rt = (constant uchar*)&kkroot;");
		for (const auto& [loc, code]: g.kk_fetch) {
			const auto k = g.vkind.at(loc);
			c.b.Line(fmt::format("{} vin_a{};", k == VKind::Uint ? "uint4" : "float4", loc));
			c.b.Line(code);
		}
	}
	if (g.has_valid_mask) c.b.Line("bool pixel_valid = true;");
}

std::string GfxBuiltin(MslCtx&, const GfxIo& g, uint32_t kind, uint32_t comp) {
	using K         = IR::StageInputKind;
	comp            = std::min(comp, 3u);
	const auto bits = [](const std::string& f) { return "as_type<uint>(" + f + ")"; };
	switch (static_cast<K>(kind)) {
		case K::VertexIndex: return g.vs ? "gl_vertex_id" : "";
		// kk-bda: Metal's [[instance_id]] already includes the base instance (= Vulkan
		// InstanceIndex, KK's mapping)
		case K::InstanceIndex:
			return g.vs ? (g.kk ? "gl_instance_id" : "(gl_instance_id + gl_base_instance)") : "";
		case K::BaseInstance: return g.vs ? "gl_base_instance" : "";
		case K::FragCoord: return g.ps ? bits(fmt::format("frag_coord[{}]", comp)) : "";
		case K::FrontFacing: {
			if (!g.ps) return "";
			const bool all = g.pinfo->ps_front_face_all_bits;
			return fmt::format("(front_facing ? {}u : {}u)", all ? "0xffffffff" : "0x3f800000",
			                   all ? "0" : "0xbf800000");
		}
		case K::SampleId: return g.ps ? "sample_id" : "";
		case K::PrimitiveId: return g.ps ? "prim_id" : "";
		case K::Layer: return g.ps ? "layer_in" : "";
		case K::BaryCoordSmooth:
		case K::BaryCoordSmoothCentroid:
		case K::BaryCoordSmoothSample:
			return g.ps ? bits(fmt::format("bary_s[{}]", std::min(comp + 1, 2u))) : "";
		case K::BaryCoordNoPerspective:
		case K::BaryCoordNoPerspectiveCentroid:
		case K::BaryCoordNoPerspectiveSample:
			return g.ps ? bits(fmt::format("bary_n[{}]", std::min(comp + 1, 2u))) : "";
		default: return "";
	}
}

namespace {

// float4/uint4 expression of an export's data (SPIR-V ExportVector).
std::string ExportExpr(const GfxIo& g, const std::string& d, const IR::ExportInfo& exp,
                       bool uint_out) {
	if (exp.compr && !uint_out) {
		const uint32_t mode = g.ps && exp.kind == IR::ExportTargetKind::Mrt &&
		                              exp.index < std::size(g.pinfo->target_output_mode)
		                          ? g.pinfo->target_output_mode[exp.index]
		                          : 0u;
		std::string    f[4] = {"0.0f", "0.0f", "0.0f", "1.0f"};
		for (uint32_t pair = 0; pair < 2; pair++) {
			if ((exp.en & (3u << (pair * 2))) == 0) continue;
			const auto un = mode == 5   ? fmt::format("unpack_unorm2x16_to_float({}[{}])", d, pair)
			                : mode == 6 ? fmt::format("unpack_snorm2x16_to_float({}[{}])", d, pair)
			                            : fmt::format("float2(as_type<half2>({}[{}]))", d, pair);
			for (uint32_t lane = 0; lane < 2; lane++) {
				const uint32_t comp = pair * 2 + lane;
				if ((exp.en >> comp) & 1u) f[comp] = fmt::format("({}).{}", un, lane ? 'y' : 'x');
			}
		}
		return fmt::format("float4({}, {}, {}, {})", f[0], f[1], f[2], f[3]);
	}
	std::string raw = d;
	if (exp.compr || exp.en != 0xfu) {
		std::string r[4] = {"0u", "0u", "0u", uint_out ? "1u" : "0x3f800000u"};
		if (exp.compr) {
			for (uint32_t pair = 0; pair < 2; pair++) {
				if ((exp.en & (3u << (pair * 2))) == 0) continue;
				for (uint32_t lane = 0; lane < 2; lane++) {
					const uint32_t comp = pair * 2 + lane;
					if ((exp.en >> comp) & 1u)
						r[comp] = fmt::format("extract_bits({}[{}], {}, 16)", d, pair, lane * 16);
				}
			}
		} else {
			for (uint32_t comp = 0; comp < 4; comp++)
				if ((exp.en >> comp) & 1u) r[comp] = fmt::format("{}[{}]", d, comp);
		}
		raw = fmt::format("uint4({}, {}, {}, {})", r[0], r[1], r[2], r[3]);
	}
	return uint_out ? raw : fmt::format("as_type<float4>({})", raw);
}

} // namespace

bool GfxEmitInst(MslCtx& c, GfxIo& g, const IR::Inst& inst) {
	using O = IR::ValueOpcode;
	auto& b = c.b;
	switch (inst.GetOpcode()) {
		case O::GetParameterField: {
			// mslfix: KYTY_PS_PER_VERTEX=unpack field read (spirvEmitterFlow.cpp
			// EmitGetParameterField); a field that found no location reads zero.
			const auto f = g.ps_field.find({inst.Arg(0).U32(), inst.Arg(1).U32()});
			if (f == g.ps_field.end()) {
				b.Line(fmt::format("{} = 0.0f;", c.Name(inst)));
				return true;
			}
			if (const auto nf = g.narrow_f.find(f->second.first);
			    nf != g.narrow_f.end() && f->second.second >= nf->second) {
				b.Line(fmt::format("{} = 0.0f; // vsperf: component not read (narrowed)", c.Name(inst)));
				return true;
			}
			b.Line(
			    fmt::format("{} = fin.F{}[{}u];", c.Name(inst), f->second.first, f->second.second));
			return true;
		}
		case O::GetInterpolationParameter:
		case O::GetAttribute: {
			const uint32_t attr = inst.Arg(0).U32();
			const auto     chan = fmt::format("({} & 3u)", c.Val(inst.Arg(1)));
			if (g.vs) {
				const auto k = g.vkind.find(attr);
				if (k == g.vkind.end()) {
					b.Line(fmt::format("{} = 0u;", c.Name(inst)));
					return true;
				}
				const auto src = c.kk_bda ? fmt::format("vin_a{}[{}]", attr, chan)
				                          : fmt::format("vin.a{}[{}]", attr, chan);
				b.Line(fmt::format("{} = {};", c.Name(inst),
				                   k->second == VKind::Uint ? src : "as_type<uint>(" + src + ")"));
				return true;
			}
			const auto l = g.ps_loc.find(attr);
			if (l == g.ps_loc.end()) {
				b.Line(fmt::format("{} = 0u;", c.Name(inst)));
				return true;
			}
			if (const auto nl = g.narrow_l.find(l->second); nl != g.narrow_l.end()) {
				if (inst.Arg(1).IsImmediate() && (inst.Arg(1).U32() & 3u) >= nl->second) {
					b.Line(fmt::format("{} = 0u; // vsperf: component not read (narrowed)", c.Name(inst)));
					return true;
				}
				b.Line(fmt::format("{} = as_type<uint>(fin.L{}[min({}, {}u)]);", c.Name(inst), l->second,
				                   chan, nl->second - 1u));
				return true;
			}
			b.Line(fmt::format("{} = as_type<uint>(fin.L{}[{}]);", c.Name(inst), l->second, chan));
			return true;
		}
		case O::SetAttribute: {
			const auto& exp  = c.program.export_info.at(inst.Flags<IR::ExportFlags>().index);
			const auto  exec = c.Val(inst.Arg(1));
			if (g.ps && exp.vm) b.Line(fmt::format("pixel_valid = {};", exec));
			if (exp.kind == IR::ExportTargetKind::Null || exp.en == 0u ||
			    exp.kind == IR::ExportTargetKind::Primitive)
				return true;
			if (g.vs && exp.kind == IR::ExportTargetKind::Parameter &&
			    g.kk_drop_exports.contains(exp.index))
				return true;
			if (g.ps && exp.kind == IR::ExportTargetKind::Mrt && exp.index != 0 &&
			    g.pinfo->alpha_blend_source != ShaderAlphaBlendSource::None)
				return true;
			if (g.ps && exp.kind == IR::ExportTargetKind::Mrt && g.kk_drop_mrt.contains(exp.index))
				return true;
			b.Open(fmt::format("if ({})", exec));
			b.Line(fmt::format("uint4 xd = {};", c.Val(inst.Arg(0))));
			if (exp.kind == IR::ExportTargetKind::Position && exp.index != 0 && g.vs) {
				for (uint32_t comp = 0; comp < 4; comp++) {
					if ((exp.en & (1u << comp)) == 0) continue;
					const auto o = IR::DecodePositionExportComponent(g.vinfo->pa_cl_vs_out_cntl,
					                                                 exp.index, comp);
					if (o.layer) {
						bool has = false;
						for (const auto& x: c.program.info.outputs)
							has |= x.kind == IR::StageOutputKind::Layer;
						if (has) b.Line(fmt::format("out.layer = xd[{}] & 0x7ffu;", comp));
					}
					if (o.viewport) {
						bool has = false;
						for (const auto& x: c.program.info.outputs)
							has |= x.kind == IR::StageOutputKind::ViewportIndex;
						if (has)
							b.Line(fmt::format("out.vport = extract_bits(xd[{}], 16, 4);", comp));
					}
					if (o.layer || o.viewport) continue;
					if (o.point_size) {
						bool has = false;
						for (const auto& x: c.program.info.outputs)
							has |= x.kind == IR::StageOutputKind::PointSize;
						if (has) b.Line(fmt::format("out.psize = as_type<float>(xd[{}]);", comp));
						continue;
					}
					if (o.clip_distance != UINT32_MAX && o.clip_distance < g.clip_count)
						b.Line(fmt::format("out.clip[{}] = as_type<float>(xd[{}]);",
						                   o.clip_distance, comp));
				}
			} else if (exp.kind == IR::ExportTargetKind::MrtZ && g.ps) {
				if (exp.en & 1u) b.Line("out.depth = as_type<float>(xd[0]);");
				if (exp.en & 4u) b.Line("out.smask = xd[2];");
			} else if (exp.kind == IR::ExportTargetKind::Position && g.vs) {
				b.Line(fmt::format("out.pos = {};", ExportExpr(g, "xd", exp, false)));
			} else if (exp.kind == IR::ExportTargetKind::Parameter && g.vs) {
				bool has = false;
				for (const auto& x: c.program.info.outputs)
					has |= x.kind == IR::StageOutputKind::Parameter && x.index == exp.index;
				const auto nc = g.kk_export_mask.find(exp.index);
				if (has && nc != g.kk_export_mask.end() && !exp.compr) {
					b.Line(fmt::format("const float4 xe = {};", ExportExpr(g, "xd", exp, false)));
					for (uint32_t i = 0; i < 4; i++)
						if ((nc->second >> i) & 1u)
							b.Line(fmt::format("out.p{}[{}] = xe[{}];", exp.index, i, i));
				} else if (const auto np = g.narrow_p.find(exp.index); has && np != g.narrow_p.end())
					b.Line(fmt::format("out.p{} = ({}).{};", exp.index, ExportExpr(g, "xd", exp, false),
					                   std::string("xyzw").substr(0, np->second)));
				else if (has)
					b.Line(
					    fmt::format("out.p{} = {};", exp.index, ExportExpr(g, "xd", exp, false)));
			} else if (exp.kind == IR::ExportTargetKind::Mrt && g.ps) {
				const bool u   = exp.index < std::size(g.pinfo->target_output_mode) &&
				                 g.pinfo->target_output_mode[exp.index] == 7u;
				bool       has = false;
				for (const auto& x: c.program.info.outputs)
					has |= x.kind == IR::StageOutputKind::Mrt && x.index == exp.index;
				if (has) {
					std::string v = ExportExpr(g, "xd", exp, u);
					b.Line(fmt::format("{} xv = {};", u ? "uint4" : "float4", v));
					if (exp.index == 0 && !u &&
					    g.pinfo->alpha_blend_source != ShaderAlphaBlendSource::None) {
						bool has1 = false;
						for (const auto& x: c.program.info.outputs)
							has1 |= x.kind == IR::StageOutputKind::Mrt && x.index == 1;
						if (has1) {
							b.Line("float4 xa = xv.wwww;");
							const char* fac = g.pinfo->alpha_blend_source ==
							                          ShaderAlphaBlendSource::SourceAlphaOne
							                      ? "1.0f"
							                  : g.pinfo->alpha_blend_source ==
							                          ShaderAlphaBlendSource::SourceAlphaZero
							                      ? "0.0f"
							                      : nullptr;
							if (fac)
								for (uint32_t comp = 0; comp < 4; comp++)
									if (g.pinfo->target_export_mapping[0].Map(comp) == 3u) {
										b.Line(fmt::format("xa[{}] = {};", comp, fac));
										break;
									}
							b.Line("out.c1 = xa;");
						}
					}
					if (exp.index < g.pinfo->target_export_mapping.size()) {
						const auto m = g.pinfo->target_export_mapping[exp.index];
						if (!m.IsIdentity())
							b.Line(fmt::format("xv = {}4(xv[{}], xv[{}], xv[{}], xv[{}]);",
							                   u ? "uint" : "float", m.Map(0), m.Map(1), m.Map(2),
							                   m.Map(3)));
					}
					if (const auto kb = g.kk_blend.find(exp.index); kb != g.kk_blend.end() && !u)
						b.Line(KkBlendStore(kb->second, exp.index));
					else
						b.Line(fmt::format("out.c{} = xv;", exp.index));
				}
			}
			b.Close();
			return true;
		}
		case O::IsHelperInvocation:
			if (g.ps) {
				b.Line(fmt::format("{} = simd_is_helper_thread();", c.Name(inst)));
				return true;
			}
			return false;
		default: return false;
	}
}

} // namespace Libs::Graphics::ShaderRecompiler::Msl
