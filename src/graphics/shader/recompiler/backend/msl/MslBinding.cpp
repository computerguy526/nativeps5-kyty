#include "graphics/shader/recompiler/backend/msl/MslBinding.h"

namespace Libs::Graphics::ShaderRecompiler::Msl::Mem {

std::string BufName(uint32_t id) { return "r.b" + std::to_string(id); }
std::string BufSizeName(uint32_t id) { return "r.b" + std::to_string(id) + "_size"; }
std::string TexName(uint32_t id) { return "r.t" + std::to_string(id); }
std::string SampName(uint32_t id) { return "r.s" + std::to_string(id); }

static const char* NumericName(Numeric n) {
	return n == Numeric::Float ? "float" : n == Numeric::Uint ? "uint" : "int";
}

static const char* DimMsl(ImgDim d) {
	switch (d) {
		case ImgDim::D1: return "texture2d";           // Metal 1D textures cannot take level/bias/gradient: emulated as 2D, height 1
		case ImgDim::D1Array: return "texture2d_array";
		case ImgDim::D2: return "texture2d";
		case ImgDim::D2Array:
		case ImgDim::Cube: return "texture2d_array"; // cube bound as a 2D-array view
		case ImgDim::D2Ms: return "texture2d_ms";
		case ImgDim::D2MsArray: return "texture2d_ms_array";
		case ImgDim::D3: return "texture3d";
	}
	return "texture2d";
}

static const char* DimJson(ImgDim d) {
	switch (d) {
		case ImgDim::D1: return "1d";
		case ImgDim::D1Array: return "1d_array";
		case ImgDim::D2: return "2d";
		case ImgDim::D2Array: return "2d_array";
		case ImgDim::D2Ms: return "2d_ms";
		case ImgDim::D2MsArray: return "2d_ms_array";
		case ImgDim::D3: return "3d";
		case ImgDim::Cube: return "cube";
	}
	return "2d";
}

static const char* AccessMsl(Access a) {
	switch (a) {
		case Access::Sample: return "access::sample";
		case Access::Read: return "access::read";
		case Access::Write: return "access::write";
		case Access::ReadWrite:
		case Access::Atomic: return "access::read_write";
	}
	return "access::sample";
}

static const char* AccessJson(Access a) {
	switch (a) {
		case Access::Sample: return "sample";
		case Access::Read: return "read";
		case Access::Write: return "write";
		case Access::ReadWrite: return "read_write";
		case Access::Atomic: return "atomic";
	}
	return "sample";
}

std::string TextureDecl(const ImageBinding& img) {
	std::string s;
	if (img.depth) {
		// depth textures: depth2d / depth2d_array (cube->array view), float only
		const bool arr = img.dim == ImgDim::D2Array || img.dim == ImgDim::Cube;
		s = std::string(img.dim == ImgDim::D2Ms ? "depth2d_ms" : arr ? "depth2d_array" : "depth2d") +
		    "<float, " + AccessMsl(img.access) + ">";
	} else if (img.dim == ImgDim::D2Ms || img.dim == ImgDim::D2MsArray) {
		s = std::string(DimMsl(img.dim)) + "<" + NumericName(img.numeric) + ", access::read>";
	} else {
		s = std::string(DimMsl(img.dim)) + "<" + NumericName(img.numeric) + ", " +
		    AccessMsl(img.access) + ">";
	}
	return s;
}

const char* MslTextureType(const ImageBinding& img) { return DimMsl(img.dim); }

// Natural layout (verified by the harness): device pointer 8B@8, uint 4B@4, texture/sampler 8B@8.
// Field order == declaration order; RootLayout() gives the byte offsets used by the manifest patches.
std::string ArgumentBufferStruct(const BindingModel& m) {
	std::string s = "struct Resources {\n";
	for (const auto& b : m.buffers) {
		s += "    " + std::string(b.written || b.atomic ? "device" : "const device") + " uchar* b" + std::to_string(b.id) + ";\n";
		s += "    uint b" + std::to_string(b.id) + "_size;\n";
	}
	if (m.srt_bytes) s += "    const device uint* srt;\n";
	if (m.gds_bytes) s += "    device uchar* gds;\n";
	for (const auto& t : m.images) s += "    " + TextureDecl(t) + " t" + std::to_string(t.id) + ";\n";
	for (const auto& sm : m.samplers) s += "    sampler s" + std::to_string(sm.id) + ";\n";
	return s + "};\n";
}

RootLayout ComputeRootLayout(const BindingModel& m) {
	RootLayout l;
	uint32_t off = 0;
	const auto al = [&](uint32_t a) { off = (off + a - 1u) / a * a; };
	for (const auto& b : m.buffers) {
		al(8); l.buffer_ptr[b.id] = off; off += 8;
		al(4); l.buffer_size[b.id] = off; off += 4;
	}
	if (m.srt_bytes) { al(8); l.srt_ptr = off; off += 8; }
	if (m.gds_bytes) { al(8); l.gds_ptr = off; off += 8; }
	for (const auto& t : m.images) { al(8); l.texture[t.id] = off; off += 8; }
	for (const auto& s : m.samplers) { al(8); l.sampler[s.id] = off; off += 8; }
	al(8);
	l.bytes = off ? off : 8u;
	return l;
}

std::string EntryParams(const BindingModel& m) {
	std::string s = "    constant Resources& r [[buffer(" + std::to_string(RootBufferIndex) + ")]]";
	if (m.push_bytes != 0u) s += ",\n    constant uint* push [[buffer(" + std::to_string(PushBufferIndex) + ")]]";
	if (m.lds_bytes != 0u) s += ",\n    threadgroup uint* lds [[threadgroup(0)]]";
	return s;
}

static std::string Q(const std::string& s) { return "\"" + s + "\""; }

static const char* HarnessDim(ImgDim d) {
	switch (d) {
		case ImgDim::D1: return "1d";
		case ImgDim::D1Array: return "1darray";
		case ImgDim::D2: return "2d";
		case ImgDim::D2Array: return "2darray";
		case ImgDim::D2Ms: return "2dms";
		case ImgDim::D2MsArray: return "2dmsarray";
		case ImgDim::D3: return "3d";
		case ImgDim::Cube: return "2darray"; // cube bound as a 6-layer 2D-array view
	}
	return "2d";
}

// Manifest in the R2 harness schema (work/harness/manifest-schema.md) + T2 extensions:
//   root block carries buffer byte sizes via init:"hex" (no schema extension needed)
//   texture dims "1darray","2dms","2dmsarray"; format "depth32float"; "emulated_2d" (1D as 2D h=1)
//   sampler "compare":true; top level "convention","stage","entry", dispatch.tg_memory
std::string ManifestJson(const BindingModel& m, const std::string& convention) {
	const RootLayout rl = ComputeRootLayout(m);
	std::string j = "{\n \"schema\": 1, \"schema_minor\": 0, \"convention\": " + Q(convention) + ", \"stage\": " + Q(m.stage) + ",\n";
	j += " \"msl\": {\"version\":\"3.1\",\"math\":\"safe\"},\n \"resources\": [";
	bool first = true;
	const auto sep = [&]() { j += first ? "\n  " : ",\n  "; first = false; };
	// root block: pointers/resource ids are patched by the harness; byte sizes are pre-filled via init:hex.
	std::string hex(rl.bytes * 2u, '0');
	for (const auto& b : m.buffers) {
		const uint32_t off = rl.buffer_size.at(b.id);
		for (uint32_t k = 0; k < 4; k++) {
			static const char* d = "0123456789abcdef";
			const uint32_t byte = (b.bytes >> (8u * k)) & 0xffu;
			hex[(off + k) * 2u] = d[byte >> 4u];
			hex[(off + k) * 2u + 1u] = d[byte & 15u];
		}
	}
	sep();
	j += "{\"name\":\"root\",\"type\":\"buffer\",\"size\":" + std::to_string(rl.bytes) + ",\"init\":\"hex\",\"hex\":\"" + hex + "\",\"role\":\"in\",\"patch\":[";
	bool pf = true;
	const auto patch = [&](const std::string& p) { j += (pf ? "" : ",") + p; pf = false; };
	for (const auto& b : m.buffers) patch("{\"offset\":" + std::to_string(rl.buffer_ptr.at(b.id)) + ",\"ref\":\"b" + std::to_string(b.id) + "\"}");
	if (m.gds_bytes) patch("{\"offset\":" + std::to_string(rl.gds_ptr) + ",\"ref\":\"gds\"}");
	if (m.srt_bytes) patch("{\"offset\":" + std::to_string(rl.srt_ptr) + ",\"ref\":\"srt\"}");
	for (const auto& t : m.images) patch("{\"offset\":" + std::to_string(rl.texture.at(t.id)) + ",\"ref\":\"t" + std::to_string(t.id) + "\"}");
	for (const auto& s : m.samplers) patch("{\"offset\":" + std::to_string(rl.sampler.at(s.id)) + ",\"ref\":\"s" + std::to_string(s.id) + "\"}");
	j += "]}";
	for (const auto& b : m.buffers) {
		const bool out = b.written || b.atomic;
		sep();
		j += "{\"name\":\"b" + std::to_string(b.id) + "\",\"type\":\"buffer\",\"size\":" + std::to_string(b.bytes) +
		     ",\"init\":\"" + b.init + "\",\"role\":\"" + (out ? (b.read ? "inout" : "out") : "in") + "\",\"dtype\":\"u32\",\"stride\":" +
		     std::to_string(b.packed_stride & 0x3fffu) + ",\"packed_stride\":" + std::to_string(b.packed_stride) + ",\"format\":" + std::to_string(b.format) + "}";
	}
	if (m.push_bytes) {
		sep();
		j += "{\"name\":\"push\",\"type\":\"buffer\",\"size\":" + std::to_string(m.push_bytes) + ",\"init\":\"zero\",\"role\":\"in\",\"dtype\":\"u32\",\"push\":true}";
	}
	if (m.gds_bytes) {
		sep();
		j += "{\"name\":\"gds\",\"type\":\"buffer\",\"size\":" + std::to_string(m.gds_bytes) + ",\"init\":\"zero\",\"role\":\"inout\",\"dtype\":\"u32\"}";
	}
	if (m.srt_bytes) {
		sep();
		j += "{\"name\":\"srt\",\"type\":\"buffer\",\"size\":" + std::to_string(m.srt_bytes) + ",\"init\":\"seed\",\"role\":\"in\",\"dtype\":\"u32\"}";
	}
	for (const auto& t : m.images) {
		const bool out = t.access == Access::Write || t.access == Access::ReadWrite || t.access == Access::Atomic;
		sep();
		j += "{\"name\":\"t" + std::to_string(t.id) + "\",\"type\":\"texture\",\"dim\":\"" + HarnessDim(t.dim) + "\",\"format\":\"" +
		     (t.depth ? "depth32float" : t.pixel_format) + "\",\"w\":" + std::to_string(t.width) + ",\"h\":" + std::to_string(t.height) +
		     ",\"d\":" + std::to_string(t.dim == ImgDim::Cube ? 6u : t.depth_or_layers) + ",\"mips\":" + std::to_string(t.mips) +
		     ",\"init\":\"seed\",\"role\":\"" + (out ? "inout" : "in") + "\",\"dtype\":\"" + (t.numeric == Numeric::Float ? "f32" : "u32") +
		     "\",\"numeric\":\"" + NumericName(t.numeric) + "\",\"access\":\"" + AccessJson(t.access) + "\",\"cube\":" + (t.dim == ImgDim::Cube ? "true" : "false") +
		     ",\"emulated_2d\":" + ((t.dim == ImgDim::D1 || t.dim == ImgDim::D1Array) ? "true" : "false") + "}";
	}
	for (const auto& s : m.samplers) {
		sep();
		j += "{\"name\":\"s" + std::to_string(s.id) + "\",\"type\":\"sampler\",\"filter\":\"" + (s.linear ? "linear" : "nearest") + "\",\"address\":\"" + s.address +
		     "\",\"normalized\":true,\"compare\":" + (s.compare ? "true" : "false") + "}";
	}
	j += "\n ],\n \"bind\": [{\"kind\":\"buffer\",\"index\":" + std::to_string(RootBufferIndex) + ",\"res\":\"root\"}";
	if (m.push_bytes) j += ",{\"kind\":\"buffer\",\"index\":" + std::to_string(PushBufferIndex) + ",\"res\":\"push\"}";
	j += "],\n \"dispatch\": {\"entry\":" + Q(m.entry) + ",\"grid\":[1,1,1],\"threadgroup\":[" + std::to_string(m.threadgroup[0]) + "," + std::to_string(m.threadgroup[1]) + "," +
	     std::to_string(m.threadgroup[2]) + "]" + (m.lds_bytes ? ",\"tg_memory\":" + std::to_string(m.lds_bytes) : std::string()) + "}\n}\n";
	return j;
}

} // namespace Libs::Graphics::ShaderRecompiler::Msl::Mem
