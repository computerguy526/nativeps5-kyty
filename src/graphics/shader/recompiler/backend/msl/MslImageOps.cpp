#include "graphics/shader/recompiler/backend/msl/MslImageOps.h"

#include "graphics/shader/recompiler/backend/msl/MslBufferOps.h"
#include "graphics/shader/recompiler/backend/msl/MslXform.h"

namespace Libs::Graphics::ShaderRecompiler::Msl::Mem {

AddrLayout ImageAddressComponentLayout(uint32_t flags, uint32_t component) {
	const auto width = [flags](uint32_t index) {
		if ((flags & SampleFlag::A16) == 0u) return 32u;
		uint32_t cursor = 0;
		if ((flags & SampleFlag::Offset) != 0u && index == cursor++) return 32u;
		if ((flags & SampleFlag::Bias) != 0u && index == cursor++) return 16u;
		if ((flags & SampleFlag::Compare) != 0u && index == cursor) return 32u;
		return 16u;
	};
	uint32_t offset = 0;
	for (uint32_t i = 0; i < component; i++) {
		const auto b = width(i);
		if (b == 32u) offset = (offset + 31u) & ~31u;
		offset += b;
	}
	const auto bits = width(component);
	if (bits == 32u) offset = (offset + 31u) & ~31u;
	return {offset, bits};
}

uint32_t ImageCoordComponents(ImgDim d) {
	switch (d) {
		case ImgDim::D1: return 1;
		case ImgDim::D1Array: return 2;
		case ImgDim::D2:
		case ImgDim::D2Ms: return 2;
		case ImgDim::D2Array:
		case ImgDim::D2MsArray:
		case ImgDim::Cube:
		case ImgDim::D3: return 3;
	}
	return 2;
}
uint32_t ImageSpatialComponents(ImgDim d) {
	switch (d) {
		case ImgDim::D1:
		case ImgDim::D1Array: return 1;
		case ImgDim::D3: return 3;
		default: return 2;
	}
}

namespace {

bool Arrayed(ImgDim d) {
	return d == ImgDim::D1Array || d == ImgDim::D2Array || d == ImgDim::D2MsArray ||
	       d == ImgDim::Cube;
}
bool Ms(ImgDim d) {
	return d == ImgDim::D2Ms || d == ImgDim::D2MsArray;
}
bool Has(const ImageOp& op, uint32_t f) {
	return (op.sample_flags & f) != 0u;
}

const char* NumType(Numeric n) {
	return n == Numeric::Float ? "float" : n == Numeric::Uint ? "uint" : "int";
}

std::string AddrU(const ImageOp& op, const std::vector<std::string>& a, uint32_t comp) {
	const auto     l = ImageAddressComponentLayout(op.sample_flags, comp);
	const uint32_t d = l.bit_offset / 32u;
	if (d >= a.size()) return "0u";
	if (l.bit_width == 16u)
		return "((" + a[d] + (((l.bit_offset & 31u) != 0u) ? " >> 16u" : "") + ") & 0xffffu)";
	return a[d];
}
std::string AddrF(const ImageOp& op, const std::vector<std::string>& a, uint32_t comp) {
	const auto l = ImageAddressComponentLayout(op.sample_flags, comp);
	const auto u = AddrU(op, a, comp);
	return l.bit_width == 16u ? "float(as_type<half>(ushort(" + u + ")))"
	                          : "as_type<float>(" + u + ")";
}

struct Layout {
	int offset = -1, bias = -1, dref = -1, gx = -1, gy = -1, coord = 0, lod = -1, clamp = -1;
};
Layout SampleLayout(const ImageOp& op) {
	Layout         l;
	uint32_t       cur = 0;
	const uint32_t sp  = ImageSpatialComponents(op.img->dim);
	if (Has(op, SampleFlag::Offset)) l.offset = cur++;
	if (Has(op, SampleFlag::Bias)) l.bias = cur++;
	if (Has(op, SampleFlag::Compare)) l.dref = cur++;
	if (Has(op, SampleFlag::Derivative)) {
		l.gx = cur;
		cur += sp;
		l.gy = cur;
		cur += sp;
	}
	l.coord = cur;
	cur += ImageCoordComponents(op.img->dim);
	if (Has(op, SampleFlag::Lod)) l.lod = cur++;
	if (Has(op, SampleFlag::LodClamp)) l.clamp = cur++;
	return l;
}

std::string FloatN(uint32_t n, const std::vector<std::string>& c) {
	if (n == 1) return c[0];
	std::string s = "float" + std::to_string(n) + "(";
	for (uint32_t i = 0; i < n; i++)
		s += (i ? ", " : "") + c[i];
	return s + ")";
}

// Float coords (spatial part) + array layer expression (empty when not arrayed).
struct Coord {
	std::string spatial;
	std::string layer;
};
Coord SampleCoord(MemOut& o, const ImageOp& op, const std::vector<std::string>& a, uint32_t first) {
	const ImgDim             d    = op.img->dim;
	const uint32_t           sp   = ImageSpatialComponents(d);
	const bool               cube = d == ImgDim::Cube;
	std::vector<std::string> c;
	for (uint32_t i = 0; i < sp; i++) {
		std::string v = first + i < op.address_components ? AddrF(op, a, first + i) : "0.0f";
		if (cube) v = "(" + v + " - 1.0f)";
		c.push_back(v);
	}
	Coord r;
	if (sp == 1)
		r.spatial = o.Def("float2", "float2(" + c[0] + ", 0.5f)"); // 1D emulated as 2D, height 1
	else
		r.spatial = o.Def(sp == 2 ? "float2" : "float3", FloatN(sp, c));
	if (Arrayed(d)) {
		const uint32_t li = first + sp;
		std::string    z  = li < op.address_components ? AddrF(op, a, li) : "0.0f";
		if (cube) {
			auto g  = o.Def("uint", "uint(" + z + ")");
			r.layer = o.Def("uint", g + " - ((" + g + " >> 3u) << 1u)");
		} else {
			r.layer = o.Def("uint", "uint(max(rint(" + z + "), 0.0f))");
		}
	}
	return r;
}

std::string ToBits4(const ImageOp& op, Numeric n, const std::string& v) {
	(void)op;
	return n == Numeric::Float  ? "as_type<uint4>(" + v + ")"
	       : n == Numeric::Sint ? "as_type<uint4>(" + v + ")"
	                            : "uint4(" + v + ")";
}

std::string Pack16(MemOut& o, const ImageOp& op, Numeric n, const std::string& vec4, bool gather) {
	// dmask-compacted 16-bit pairs
	std::vector<std::string> comp;
	for (uint32_t c = 0; c < 4; c++)
		if (gather || ((op.dmask >> c) & 1u)) comp.push_back(vec4 + "[" + std::to_string(c) + "]");
	std::string w[4];
	for (uint32_t i = 0; i < 4; i++) {
		const std::string lo =
		    2 * i < comp.size() ? comp[2 * i] : (n == Numeric::Float ? "0.0f" : "0");
		const std::string hi =
		    2 * i + 1 < comp.size() ? comp[2 * i + 1] : (n == Numeric::Float ? "0.0f" : "0");
		w[i] = n == Numeric::Float ? "as_type<uint>(half2(" + lo + ", " + hi + "))"
		                           : "((uint(" + lo + ") & 0xffffu) | (uint(" + hi + ") << 16u))";
	}
	return o.Def("uint4", "uint4(" + w[0] + ", " + w[1] + ", " + w[2] + ", " + w[3] + ")");
}

bool CheckSupported(MemOut& o, const ImageOp& op, const char* what) {
	if (op.converted)
		return o.Fail(std::string(what) +
		              ": converted-format image (conversion_format) unsupported");
	return true;
}

} // namespace

static thread_local bool g_kk_sbias = false;
void                     SetKkSamplerBias(bool on) {
	g_kk_sbias = on;
}
static thread_local MipStatsConfig g_mip_stats;
void                               SetMipStats(const MipStatsConfig& c) {
	g_mip_stats = c;
}

namespace {
// GET_LOD_STATS feedback (Kyty spirvEmitterImage.cpp EmitMipStatsRecord/EmitGatedMipStatsRecord,
// default KYTY_LOD_STATS_GATE=1, LANE_RECORD=0), written as KosmicKrisp's MSL of that SPIR-V reads
// (cache-v11 KK PS texts): per-draw field id = shader data dword MipStatsOffsetDword()+resource;
// bit 15 set = no counter (skip). LOD = sample LOD relative to the view base (implicit/gradient:
// calculate_unclamped_lod [Vulkan ImageQueryLod.y, no sampler bias]
// + shader bias, raised to the _CL clamp; LOD flag: the address LOD; LZ: 0) + T# BASE_LEVEL
// (bits 8..11), clamped 0..14 (fmax/fmin), level = uint(); subgroup simd_min; counted =
// simd_max(clamped < U4.8 threshold bits 16..27). simd_is_first() lane: robust bounds check of the
// MipStats buffer, AtomicUMin only when a relaxed load shows a larger word, count += 1 at [ctr +
// 257]; KK skips both atomics on a helper lane (simd_is_helper_thread), so B does.
void MipStatsRecord(MemOut& o, const ImageOp& op, const std::string& lod) {
	const auto&       m  = g_mip_stats;
	const std::string id = o.Def("uint", "push[" + U(m.offset_dword + op.img->id) + "]");
	o.Line("if ((" + id + " & 0x8000u) == 0u) {");
	const std::string ab = o.Def("float", "(" + lod + ") + float((" + id + " >> 8u) & 15u)");
	std::string       cl = o.Tmp();
	o.Line("float " + cl + ";");
	o.Line("{");
	o.Line("#pragma METAL fp math_mode(relaxed)");
	o.Line(cl + " = precise::fmin(precise::fmax(" + ab + ", 0.0f), 14.0f);");
	o.Line("}");
	const std::string fin = o.Def("uint", "simd_min(uint(" + cl + "))");
	const std::string thr = o.Def("float", "float((" + id + " >> 16u) & 4095u) * 0.00390625f");
	const std::string cnt =
	    o.Def("uint", "simd_max((!isnan(" + cl + ") && " + cl + " < " + thr + ") ? 1u : 0u)");
	o.Line("if (simd_is_first() && !simd_is_helper_thread()) {");
	const std::string slot = o.Def("uint", "(" + id + " & 255u) << 2u");
	o.Line("if (" + slot + " + 3u < kk_ms_n) {");
	const std::string ptr =
	    o.Def("device atomic_uint*", "(device atomic_uint*)(kk_ms_a + ulong(" + slot + "))");
	// perfauto transform mipstats_plain_load (MslXform.h; nxreview #4): the pre-check reads the
	// word with a plain load. The stored minimum only ever decreases and the update itself stays
	// atomic_fetch_min, so the final word is the same; only the cost of the pre-check changes.
	const std::string cur = Xf::On(Xf::MipStatsPlainLoad)
	                            ? "*(device uint*)(kk_ms_a + ulong(" + slot + "))"
	                            : "atomic_load_explicit(" + ptr + ", memory_order_relaxed)";
	o.Line("if (" + fin + " < " + cur + ") atomic_fetch_min_explicit(" + ptr + ", " + fin +
	       ", memory_order_relaxed);");
	o.Line("}");
	o.Line("if (" + cnt + " != 0u && " + slot +
	       " + 1031u < kk_ms_n) atomic_fetch_add_explicit((device atomic_uint*)(kk_ms_a + ulong(" +
	       slot + " + 1028u)), 1u, memory_order_relaxed);");
	o.Line("}");
	o.Line("}");
}
} // namespace

std::string ImageSample(MemOut& o, const ImageOp& op, const std::vector<std::string>& addr,
                        bool gather) {
	const auto& img = *op.img;
	if (!CheckSupported(o, op, "ImageSample")) return "uint4(0u)";
	if (op.samp == nullptr) {
		o.Fail("ImageSample: no sampler");
		return "uint4(0u)";
	}
	const ImgDim d    = img.dim;
	const bool   dref = Has(op, SampleFlag::Compare);
	if (Ms(d)) {
		o.Fail("ImageSample on a multisampled image");
		return "uint4(0u)";
	}
	if (dref && !img.depth) {
		o.Fail("depth compare on a non-depth texture binding");
		return "uint4(0u)";
	}
	const Layout      L = SampleLayout(op);
	const std::string T = TexName(img.id), S = SampName(op.samp->id);
	const Coord       c    = SampleCoord(o, op, addr, static_cast<uint32_t>(L.coord));
	std::string       args = S + ", " + c.spatial + (c.layer.empty() ? "" : ", " + c.layer);

	// Offset: compile-time when the address dword is immediate, else folded into the coordinate.
	std::string off_arg;
	std::string coord_expr = c.spatial;
	if (L.offset >= 0) {
		if (op.const_offset) {
			const uint32_t p = *op.const_offset;
			const auto sx = [](uint32_t v) { return int(v & 0x3fu) - (int(v & 0x20u) ? 64 : 0); };
			const uint32_t sp = ImageSpatialComponents(d);
			off_arg = sp == 1 ? "int2(" + std::to_string(sx(p)) + ", 0)"
			                  : (sp == 2 ? "int2(" : "int3(") + std::to_string(sx(p)) + ", " +
			                        std::to_string(sx(p >> 8)) +
			                        (sp == 3 ? ", " + std::to_string(sx(p >> 16)) : "") + ")";
		} else {
			// dynamic offset: coord += texel offset / size (level 0 size)
			const auto     raw = AddrU(op, addr, static_cast<uint32_t>(L.offset));
			const uint32_t sp  = ImageSpatialComponents(d);
			std::string    offs =
			    sp == 1 ? "float2(float(t2_sx(" + raw + ", 6u)), 0.0f)"
			            : std::string(sp == 2 ? "float2(" : "float3(") + "float(t2_sx(" + raw +
			                  ", 6u)), float(t2_sx(" + raw + " >> 8u, 6u))" +
			                  (sp == 3 ? ", float(t2_sx(" + raw + " >> 16u, 6u))" : "") + ")";
			std::string size = sp == 1   ? "float2(" + T + ".get_width(), 1.0f)"
			                   : sp == 2 ? "float2(" + T + ".get_width(), " + T + ".get_height())"
			                             : "float3(" + T + ".get_width(), " + T +
			                                   ".get_height(), " + T + ".get_depth())";
			coord_expr =
			    o.Def(sp == 3 ? "float3" : "float2", c.spatial + " + " + offs + " / " + size);
			args = S + ", " + coord_expr + (c.layer.empty() ? "" : ", " + c.layer);
		}
	}

	if (gather) {
		if (d == ImgDim::D1 || d == ImgDim::D1Array || d == ImgDim::D3) {
			o.Fail("gather on 1D/3D image unsupported");
			return "uint4(0u)";
		}
		if (Has(op, SampleFlag::GatherHorizontal)) {
			o.Fail("gather with horizontal offsets unsupported");
			return "uint4(0u)";
		}
		std::string call;
		if (dref) {
			call = T + ".gather_compare(" + args + ", " +
			       (L.dref >= 0 ? AddrF(op, addr, static_cast<uint32_t>(L.dref)) : "0.0f") +
			       (off_arg.empty() ? "" : ", " + off_arg) + ")";
		} else {
			const uint32_t     comp     = op.dmask == 0x2u   ? 1
			                              : op.dmask == 0x4u ? 2
			                              : op.dmask == 0x8u ? 3
			                                                 : 0;
			static const char* names[4] = {"x", "y", "z", "w"};
			call = T + ".gather(" + args + ", " + (off_arg.empty() ? "int2(0)" : off_arg) +
			       ", component::" + names[comp] + ")";
		}
		auto v = o.Def(dref ? "float4" : (std::string(NumType(img.numeric)) + "4").c_str(), call);
		const Numeric n = dref ? Numeric::Float : img.numeric;
		return op.data_bits == 16 ? Pack16(o, op, n, v, true) : o.Def("uint4", ToBits4(op, n, v));
	}

	// Sample LOD selection.
	const bool  explicit_lod = Has(op, SampleFlag::Derivative) || Has(op, SampleFlag::Lod) ||
	                           Has(op, SampleFlag::LevelZero) || !op.pixel_stage;
	std::string clamp;
	if (L.clamp >= 0 && static_cast<uint32_t>(L.clamp) < op.address_components &&
	    op.lod_clamp_enabled)
		clamp = AddrF(op, addr, static_cast<uint32_t>(L.clamp));
	std::string       opt;
	const std::string kb = g_kk_sbias ? "kk_sbias" + std::to_string(op.samp->id) : std::string();
	if (Has(op, SampleFlag::Derivative)) {
		const uint32_t sp = ImageSpatialComponents(d);
		// mslfix3: kk-bda descriptor LOD bias on a gradient sample = gradients scaled by
		// exp2(bias), as KK lowers it (KK FS: g * exp2(float(half bias)) before gradient2d); was
		// unsupported (isatest img_sample_d).
		const std::string gs = kb.empty() ? std::string() : o.Def("float", "exp2(" + kb + ")");
		const auto        G  = [&](uint32_t idx) {
			const std::string v = AddrF(op, addr, idx);
			return gs.empty() ? v : o.Def("float", v + " * " + gs);
		};
		std::vector<std::string> gx, gy;
		for (uint32_t i = 0; i < sp; i++)
			gx.push_back(G(static_cast<uint32_t>(L.gx) + i));
		for (uint32_t i = 0; i < sp; i++)
			gy.push_back(G(static_cast<uint32_t>(L.gy) + i));
		if (sp == 1) {
			opt = "gradient2d(float2(" + gx[0] + ", 0.0f), float2(" + gy[0] + ", 0.0f))" +
			      (clamp.empty() ? "" : ", min_lod_clamp(" + clamp + ")");
		} else {
			opt = std::string(sp == 3 ? "gradient3d(" : "gradient2d(") + FloatN(sp, gx) + ", " +
			      FloatN(sp, gy) + ")" + (clamp.empty() ? "" : ", min_lod_clamp(" + clamp + ")");
		}
	} else if (explicit_lod) {
		std::string lod = "0.0f";
		if (Has(op, SampleFlag::Lod) && L.lod >= 0 &&
		    static_cast<uint32_t>(L.lod) < op.address_components)
			lod = AddrF(op, addr, static_cast<uint32_t>(L.lod));
		if (!clamp.empty()) lod = "max(" + lod + ", " + clamp + ")";
		if (!kb.empty()) lod = "(" + lod + " + " + kb + ")";
		opt = "level(" + lod + ")";
	} else if (L.bias >= 0) {
		opt = "bias(" + AddrF(op, addr, static_cast<uint32_t>(L.bias)) +
		      (kb.empty() ? "" : " + " + kb) + ")" +
		      (clamp.empty() ? "" : ", min_lod_clamp(" + clamp + ")");
	} else if (!kb.empty()) {
		opt = "bias(" + kb + ")" + (clamp.empty() ? "" : ", min_lod_clamp(" + clamp + ")");
	} else if (!clamp.empty()) {
		opt = "min_lod_clamp(" + clamp + ")";
	}
	args += (opt.empty() ? "" : ", " + opt) + (off_arg.empty() ? "" : ", " + off_arg);
	// GET_LOD_STATS feedback (after the sample, as Kyty's SPIR-V); gathers record nothing.
	const auto record = [&]() {
		if (!g_mip_stats.on || !op.pixel_stage || op.img->id >= g_mip_stats.count) return;
		std::string lod = "0.0f";
		if (Has(op, SampleFlag::LevelZero)) {
		} else if (Has(op, SampleFlag::Lod) && L.lod >= 0) {
			lod = static_cast<uint32_t>(L.lod) < op.address_components
			          ? AddrF(op, addr, static_cast<uint32_t>(L.lod))
			          : "0.0f";
		} else {
			lod = o.Def("float", T + ".calculate_unclamped_lod(" + S + ", " + c.spatial + ")");
			if (!explicit_lod && L.bias >= 0)
				lod = o.Def("float", lod + " + " + AddrF(op, addr, static_cast<uint32_t>(L.bias)));
			if (!clamp.empty() && (Has(op, SampleFlag::Derivative) || !explicit_lod))
				lod = o.Def("float", "fmax(" + lod + ", " + clamp + ")");
		}
		MipStatsRecord(o, op, lod);
	};
	if (dref) {
		const std::string dv =
		    L.dref >= 0 ? AddrF(op, addr, static_cast<uint32_t>(L.dref)) : "0.0f";
		// sample_compare(s, coord, [layer], compare_value, [options], [offset])
		std::string a2 = S + ", " + coord_expr + (c.layer.empty() ? "" : ", " + c.layer) + ", " +
		                 dv + (opt.empty() ? "" : ", " + opt) +
		                 (off_arg.empty() ? "" : ", " + off_arg);
		auto        v  = o.Def("float", T + ".sample_compare(" + a2 + ")");
		record();
		return op.data_bits == 16
		           ? Pack16(o, op, Numeric::Float, "float4(" + v + ", 0.0f, 0.0f, 0.0f)", false)
		           : o.Def("uint4", "uint4(as_type<uint>(" + v + "), 0u, 0u, 0u)");
	}
	auto v = o.Def((std::string(NumType(img.numeric)) + "4").c_str(), T + ".sample(" + args + ")");
	record();
	return op.data_bits == 16 ? Pack16(o, op, img.numeric, v, false)
	                          : o.Def("uint4", ToBits4(op, img.numeric, v));
}

namespace {

// Integer texel coordinate + layer/sample/lod pieces for read/write/atomic.
struct TexelAddr {
	std::string coord;
	std::string layer;
	std::string sample;
	std::string lod;
};
TexelAddr Texel(MemOut& o, const ImageOp& op, const std::vector<std::string>& a) {
	const ImgDim             d  = op.img->dim;
	const uint32_t           sp = ImageSpatialComponents(d);
	const uint32_t           cc = ImageCoordComponents(d);
	TexelAddr                t;
	std::vector<std::string> c;
	for (uint32_t i = 0; i < sp; i++)
		c.push_back(i < op.address_components ? AddrU(op, a, i) : "0u");
	if (sp == 1)
		t.coord = o.Def("uint2", "uint2(" + c[0] + ", 0u)"); // 1D emulated as 2D, height 1
	else
		t.coord = o.Def(sp == 2 ? "uint2" : "uint3", std::string(sp == 2 ? "uint2(" : "uint3(") +
		                                                 c[0] + ", " + c[1] +
		                                                 (sp == 3 ? ", " + c[2] : "") + ")");
	if (Arrayed(d)) t.layer = cc - 1 < op.address_components ? AddrU(op, a, cc - 1) : "0u";
	if (Ms(d)) t.sample = cc < op.address_components ? AddrU(op, a, cc) : "0u";
	// isatest img_load_mip: the mip operand counts whenever the opcode has one (IMAGE_LOAD_MIP /
	// STORE_MIP), as the SPIR-V emitter (LodU32: image_has_mip only); ImageResource::mip_mode is
	// not set for these and B read level 0.
	if (op.has_mip && (op.dynamic_mip || op.graphics_stage) && !Ms(d))
		t.lod = cc < op.address_components ? AddrU(op, a, cc) : "0u";
	return t;
}

} // namespace

std::string ImageLoad(MemOut& o, const ImageOp& op, const std::vector<std::string>& addr,
                      const std::string& active) {
	if (!CheckSupported(o, op, "ImageLoad")) return "uint4(0u)";
	const auto&       img = *op.img;
	const auto        t   = Texel(o, op, addr);
	const std::string T   = TexName(img.id);
	const std::string tv  = std::string(NumType(img.numeric)) + "4";
	std::string       out = o.Tmp();
	o.Line("uint4 " + out + " = uint4(0u);");
	// robustImageAccess2 (Kyty's Vulkan device, KK): an out-of-range texel / level / layer reads 0;
	// Metal read() is undefined there.
	std::string inb = active;
	if (!Ms(img.dim) && op.graphics_stage) { // mslfix3: graphics only (compute text byte-identical)
		const uint32_t    sp = ImageSpatialComponents(img.dim);
		const std::string lv = t.lod.empty() ? std::string("0u") : t.lod;
		inb = "(" + active + ") && " + lv + " < " + T + ".get_num_mip_levels() && " + t.coord +
		      ".x < " + T + ".get_width(" + lv + ")";
		if (sp >= 2) inb += " && " + t.coord + ".y < " + T + ".get_height(" + lv + ")";
		if (sp == 3) inb += " && " + t.coord + ".z < " + T + ".get_depth(" + lv + ")";
		if (!t.layer.empty()) inb += " && " + t.layer + " < " + T + ".get_array_size()";
	}
	o.Line("if (" + inb + ") {");
	// Metal read(): [coord, layer, lod] or [coord, layer, sample]
	std::string args = t.coord + (t.layer.empty() ? "" : ", " + t.layer);
	if (Ms(img.dim))
		args += ", " + t.sample;
	else if (!t.lod.empty() || true)
		args += ", " + (t.lod.empty() ? std::string("0u") : t.lod);
	auto v = o.Def(tv.c_str(), T + ".read(" + args + ")");
	o.Line("    " + out + " = " + ToBits4(op, img.numeric, v) + ";");
	o.Line("}");
	return out;
}

void ImageStore(MemOut& o, const ImageOp& op, const std::vector<std::string>& addr,
                const std::string& data, const std::string& active) {
	if (!CheckSupported(o, op, "ImageStore")) return;
	const auto& img = *op.img;
	const auto  t   = Texel(o, op, addr);
	// Same texel assembly as the SPIR-V emitter's StoreTexel: texture channel c takes the source
	// channel s with swizzle[s]==4+c, present only if dmask bit s is set (packed index = popcount
	// of lower dmask bits), else 0. 16-bit data: two halves per dword.
	const uint32_t dm = op.dmask != 0u ? op.dmask : 1u;
	std::string    comp[4];
	for (uint32_t c = 0; c < 4u; c++) {
		uint32_t src = 4u;
		for (uint32_t s = 0; s < 4u; s++)
			if (((img.shader_swizzle >> (s * 3u)) & 7u) == 4u + c) {
				src = s;
				break;
			}
		if (src < 4u && ((dm >> src) & 1u) != 0u) {
			uint32_t pi = 0;
			for (uint32_t i = 0; i < src; i++)
				pi += (dm >> i) & 1u;
			if (op.data_bits == 16u) {
				const std::string raw = "((" + data + ")[" + std::to_string(pi / 2u) + "] " +
				                        ((pi & 1u) ? ">> 16u" : "& 0xffffu") + ")";
				comp[c] = img.numeric == Numeric::Float
				              ? "as_type<uint>(float(as_type<half>(ushort(" + raw + "))))"
				              : raw;
			} else
				comp[c] = "(" + data + ")[" + std::to_string(pi) + "]";
		} else
			comp[c] = "0u";
	}
	const std::string packed =
	    "uint4(" + comp[0] + ", " + comp[1] + ", " + comp[2] + ", " + comp[3] + ")";
	const std::string conv = img.numeric == Numeric::Float  ? "as_type<float4>(" + packed + ")"
	                         : img.numeric == Numeric::Sint ? "as_type<int4>(" + packed + ")"
	                                                        : packed;
	std::string       args = conv + ", " + t.coord + (t.layer.empty() ? "" : ", " + t.layer) +
	                         (t.lod.empty() ? "" : ", " + t.lod);
	o.Line("if (" + active + ") { " + TexName(img.id) + ".write(" + args + "); }");
}

std::string ImageAtomic32(MemOut& o, const ImageOp& op, const std::vector<std::string>& addr,
                          int kind, const std::string& value, const std::string& active) {
	// kind: AtomicOp (MslBufferOps.h)
	const auto  aop = static_cast<AtomicOp>(kind);
	const char* fn  = nullptr;
	switch (aop) {
		case AtomicOp::Add: fn = "atomic_fetch_add"; break;
		case AtomicOp::Sub: fn = "atomic_fetch_sub"; break;
		case AtomicOp::And: fn = "atomic_fetch_and"; break;
		case AtomicOp::Or: fn = "atomic_fetch_or"; break;
		case AtomicOp::Xor: fn = "atomic_fetch_xor"; break;
		case AtomicOp::UMin:
		case AtomicOp::SMin: fn = "atomic_fetch_min"; break;
		case AtomicOp::UMax:
		case AtomicOp::SMax: fn = "atomic_fetch_max"; break;
		case AtomicOp::Swap: fn = "atomic_exchange"; break;
		default: o.Fail("image atomic cmpswap/inc/dec/float unsupported on textures"); return "0u";
	}
	const auto        t   = Texel(o, op, addr);
	const std::string res = o.Tmp();
	o.Line("uint " + res + " = 0u;");
	const bool sgn =
	    op.img->numeric == Numeric::Sint || aop == AtomicOp::SMin || aop == AtomicOp::SMax;
	o.Line("if (" + active + ") { " + res + " = " + (sgn ? "as_type<uint>(" : "(") +
	       TexName(op.img->id) + "." + fn + "(" + t.coord +
	       (t.layer.empty() ? "" : ", " + t.layer) + ", " +
	       (sgn ? "int4(as_type<int>(" + value + "))" : "uint4(" + value + ")") + ").x); }");
	return res;
}

std::string ImageQueryDims(MemOut& o, const ImageOp& op, const std::vector<std::string>& addr) {
	const auto&       img = *op.img;
	const ImgDim      d   = img.dim;
	const std::string T   = TexName(img.id);
	const uint32_t    cc  = ImageCoordComponents(d);
	const std::string lod =
	    op.has_mip && cc < op.address_components && !Ms(d) ? AddrU(op, addr, cc) : "0u";
	const std::string lv = Ms(d) ? "" : lod;
	std::string       w  = "uint(" + T + ".get_width(" + lv + "))";
	std::string       h =
	    ImageSpatialComponents(d) >= 2 ? "uint(" + T + ".get_height(" + lv + "))" : "0u";
	std::string z = d == ImgDim::D3 ? "uint(" + T + ".get_depth(" + lv + "))"
	                : Arrayed(d)
	                    ? "uint(" + T + ".get_array_size())" + (d == ImgDim::Cube ? " / 6u" : "")
	                    : "0u";
	std::string m = Ms(d) ? "1u" : "uint(" + T + ".get_num_mip_levels())";
	return o.Def("uint4", "uint4(" + w + ", " + h + ", " + z + ", " + m + ")");
}

std::string ImageQueryLod(MemOut& o, const ImageOp& op, const std::vector<std::string>& addr) {
	if (op.samp == nullptr) {
		o.Fail("ImageQueryLod: no sampler");
		return "uint4(0u)";
	}
	const Coord       c = SampleCoord(o, op, addr, 0);
	const std::string T = TexName(op.img->id), S = SampName(op.samp->id);
	return o.Def("uint4", "uint4(as_type<uint>(" + T + ".calculate_clamped_lod(" + S + ", " +
	                          c.spatial + ")), as_type<uint>(" + T + ".calculate_unclamped_lod(" +
	                          S + ", " + c.spatial + ")), 0u, 0u)");
}

} // namespace Libs::Graphics::ShaderRecompiler::Msl::Mem
