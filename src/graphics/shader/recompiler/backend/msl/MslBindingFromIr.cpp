#include "graphics/shader/recompiler/backend/msl/MslBindingFromIr.h"

#include "graphics/shader/recompiler/backend/msl/MslEmitter.h"

namespace Libs::Graphics::ShaderRecompiler::Msl {

using namespace Mem;

uint32_t ProgramLdsBytes(const IR::Program& program, ShaderStageInputInfo input_info) {
	if (const auto* wg = ShaderWorkgroupInput(program.stage, input_info))
		return wg->lds_size_dwords * 4u;
	return 0u;
}

namespace {

ImgDim DimOf(const IR::ImageResource& img) {
	using D = Decoder::ImageDimension;
	if (img.cube) return ImgDim::Cube;
	switch (img.dimension) {
		case D::Dim1D: return ImgDim::D1;
		case D::Dim1DArray: return ImgDim::D1Array;
		case D::Dim2D: return ImgDim::D2;
		case D::Dim2DArray: return ImgDim::D2Array;
		case D::Dim2DMsaa: return ImgDim::D2Ms;
		case D::Dim2DMsaaArray: return ImgDim::D2MsArray;
		case D::Dim3D: return ImgDim::D3;
		default: return ImgDim::D2;
	}
}

} // namespace

BindingModel BuildBindingModel(const IR::Program& program, ShaderStageInputInfo input_info) {
	BindingModel m;
	m.stage      = program.stage == ShaderType::Compute ? "compute"
	               : program.stage == ShaderType::Pixel ? "fragment"
	                                                    : "vertex";
	m.entry      = "main0"; // T1's EmitProgram entry name
	m.lds_bytes  = ProgramLdsBytes(program, input_info);
	m.push_bytes = IR::NativePushConstantSize;
	if (const auto* wg = ShaderWorkgroupInput(program.stage, input_info)) {
		// Metal threads per group, as EmitProgram computes them: wave64 = two guest halves per
		// thread.
		uint32_t total = std::max(wg->threads_num[0], 1u) * std::max(wg->threads_num[1], 1u) *
		                 std::max(wg->threads_num[2], 1u);
		if (program.wave_size == 64u && ChooseLanes(program, input_info) == 2u)
			total = ((total + 63u) / 64u) * 32u;
		m.threadgroup = {total, 1u, 1u};
	}
	for (const auto* blk: program.blocks) {
		for (const auto& inst: *blk) {
			if (inst.GetOpcode() == IR::ValueOpcode::ReadConst) m.srt_bytes = 65536u;
			if (inst.GetOpcode() == IR::ValueOpcode::DataAppend ||
			    inst.GetOpcode() == IR::ValueOpcode::DataConsume)
				m.gds_bytes = 4096u;
		}
	}
	const auto& info = program.info;
	for (uint32_t i = 0; i < info.buffers.size(); i++) {
		const auto&   b = info.buffers[i];
		BufferBinding bb;
		bb.id            = i;
		bb.read          = b.read || b.scalar || !b.written;
		bb.written       = b.written;
		bb.atomic        = b.atomic;
		bb.bytes         = std::max(65536u, (b.max_byte_extent + 255u) & ~255u);
		bb.packed_stride = b.packed_stride;
		bb.format        = static_cast<uint32_t>(b.descriptor_format);
		if (b.indirect_root != IR::BufferResource::NoIndirectBuffer) bb.bytes = 65536u;
		m.buffers.push_back(bb);
	}
	for (uint32_t i = 0; i < info.images.size(); i++) {
		const auto&  im = info.images[i];
		ImageBinding ib;
		ib.id      = i;
		ib.dim     = DimOf(im);
		ib.numeric = im.numeric_class == Prospero::TextureNumericClass::Uint   ? Numeric::Uint
		             : im.numeric_class == Prospero::TextureNumericClass::Sint ? Numeric::Sint
		                                                                       : Numeric::Float;
		ib.depth   = im.depth_compare;
		ib.shader_swizzle = im.shader_swizzle;
		ib.mips           = std::max(im.mip_count, 1u);
		if (im.resource_class == IR::ImageResourceClass::Storage) {
			ib.access = im.atomic                 ? Access::Atomic
			            : (im.read && im.written) ? Access::ReadWrite
			            : im.written              ? Access::Write
			                                      : Access::Read;
		} else {
			ib.access = Access::Sample;
		}
		ib.pixel_format = ib.numeric == Numeric::Float  ? "rgba32float"
		                  : ib.numeric == Numeric::Uint ? "rgba32uint"
		                                                : "rgba32sint";
		m.images.push_back(ib);
	}
	for (uint32_t i = 0; i < info.samplers.size(); i++) {
		SamplerBinding sb;
		sb.id      = i;
		sb.compare = info.samplers[i].depth_compare;
		sb.linear  = !info.samplers[i].force_point_filtering;
		m.samplers.push_back(sb);
	}
	return m;
}

std::string BuildManifestJson(const IR::Program& program, ShaderStageInputInfo input_info,
                              const std::string& convention) {
	return ManifestJson(BuildBindingModel(program, input_info), convention);
}

} // namespace Libs::Graphics::ShaderRecompiler::Msl
