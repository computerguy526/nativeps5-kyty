#include "graphics/shader/perVertexShader.h"

#include "common/assert.h"
#include "graphics/shader/recompiler/CodegenOptions.h"
#include "graphics/shader/recompiler/backend/spirv/SpirvBuilder.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/shader.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>

namespace Libs::Graphics {

namespace {

using ShaderRecompiler::Spirv::Builder;
namespace IR = ShaderRecompiler::IR;

constexpr uint32_t SpirvVersion15 = 0x00010500u;

class PerVertexGeometryEmitter {
public:
	explicit PerVertexGeometryEmitter(const PerVertexGeometryInterface& interface_)
	    : interface(interface_) {}

	std::vector<uint32_t> Emit() {
		builder.AddMemoryModel(spv::AddressingModelLogical, spv::MemoryModelGLSL450);
		builder.RequireCapability(spv::CapabilityShader);
		builder.RequireCapability(spv::CapabilityGeometry);
		if (interface.clip_distances != 0) {
			builder.RequireCapability(spv::CapabilityClipDistance);
		}
		if (interface.cull_distances != 0) {
			builder.RequireCapability(spv::CapabilityCullDistance);
		}

		void_type       = builder.Type(spv::OpTypeVoid);
		uint_type       = builder.Type(spv::OpTypeInt, 32u, 0u);
		int_type        = builder.Type(spv::OpTypeInt, 32u, 1u);
		float_type      = builder.Type(spv::OpTypeFloat, 32u);
		vec4_float_type = builder.Type(spv::OpTypeVector, float_type, 4u);
		const auto function_type = builder.Type(spv::OpTypeFunction, void_type);

		// gl_PerVertex: the position and the vertex shader's clip and cull distances.
		std::vector<uint32_t> members {vec4_float_type};
		if (interface.clip_distances != 0) {
			clip_member = static_cast<uint32_t>(members.size());
			clip_type   = Array(float_type, interface.clip_distances);
			members.push_back(clip_type);
		}
		if (interface.cull_distances != 0) {
			cull_member = static_cast<uint32_t>(members.size());
			cull_type   = Array(float_type, interface.cull_distances);
			members.push_back(cull_type);
		}
		const auto member_span    = std::span<const uint32_t>(members);
		const auto per_vertex_in  = builder.Type(spv::OpTypeStruct, member_span);
		// A distinct type id for the output block, which alone may carry Invariant.
		const auto per_vertex_out =
		    builder.DecoratedType(spv::OpTypeStruct, {{spv::OpDecorate, {spv::DecorationBlock}}},
		                          member_span);
		builder.AddAnnotation(spv::OpDecorate, per_vertex_in, spv::DecorationBlock);
		for (const auto type: {per_vertex_in, per_vertex_out}) {
			builder.AddAnnotation(spv::OpMemberDecorate, type, 0u, spv::DecorationBuiltIn,
			                      spv::BuiltInPosition);
			if (clip_member != 0) {
				builder.AddAnnotation(spv::OpMemberDecorate, type, clip_member,
				                      spv::DecorationBuiltIn, spv::BuiltInClipDistance);
			}
			if (cull_member != 0) {
				builder.AddAnnotation(spv::OpMemberDecorate, type, cull_member,
				                      spv::DecorationBuiltIn, spv::BuiltInCullDistance);
			}
		}
		if (interface.invariant) {
			builder.AddAnnotation(spv::OpMemberDecorate, per_vertex_out, 0u,
			                      spv::DecorationInvariant);
		}

		main = Result(spv::OpFunction, void_type, spv::FunctionControlMaskNone, function_type);
		builder.AddExecutionMode(main, spv::ExecutionModeTriangles);
		builder.AddExecutionMode(main, spv::ExecutionModeInvocations, 1u);
		builder.AddExecutionMode(main, spv::ExecutionModeOutputTriangleStrip);
		builder.AddExecutionMode(main, spv::ExecutionModeOutputVertices, 3u);

		const auto gl_in  = AddInterface(spv::StorageClassInput, Array(per_vertex_in, 3u));
		const auto gl_out = AddInterface(spv::StorageClassOutput, per_vertex_out);
		uint32_t   primitive_id_in  = 0;
		uint32_t   primitive_id_out = 0;
		if (interface.primitive_id) {
			primitive_id_in = AddInterface(spv::StorageClassInput, int_type);
			builder.AddAnnotation(spv::OpDecorate, primitive_id_in, spv::DecorationBuiltIn,
			                      spv::BuiltInPrimitiveId);
			primitive_id_out = AddInterface(spv::StorageClassOutput, int_type);
			builder.AddAnnotation(spv::OpDecorate, primitive_id_out, spv::DecorationBuiltIn,
			                      spv::BuiltInPrimitiveId);
		}

		struct Bound {
			uint32_t                input = 0; // 0: not exported
			std::array<uint32_t, 3> outputs {};
		};
		std::vector<Bound>       bound(interface.parameters.size());
		std::array<uint32_t, 32> inputs_by_location {};
		for (size_t i = 0; i < interface.parameters.size(); i++) {
			const auto& parameter = interface.parameters[i];
			if (parameter.exported) {
				auto& input = inputs_by_location[parameter.input_location & 31u];
				if (input == 0) {
					input = AddInterface(spv::StorageClassInput, Array(vec4_float_type, 3u));
					builder.AddAnnotation(spv::OpDecorate, input, spv::DecorationLocation,
					                      parameter.input_location);
				}
				bound[i].input = input;
			}
			const std::array<uint32_t, 3> locations {parameter.output_location,
			                                         parameter.vertex1_location,
			                                         parameter.vertex2_location};
			for (uint32_t vertex = 0; vertex < (parameter.raw ? 3u : 1u); vertex++) {
				bound[i].outputs[vertex] = AddInterface(spv::StorageClassOutput, vec4_float_type);
				builder.AddAnnotation(spv::OpDecorate, bound[i].outputs[vertex],
				                      spv::DecorationLocation, locations[vertex]);
			}
		}
		builder.AddEntryPoint(spv::ExecutionModelGeometry, main, "main", interfaces);
		ResultWithoutType(spv::OpLabel);

		const auto ptr_in_vec4  = Pointer(spv::StorageClassInput, vec4_float_type);
		const auto ptr_out_vec4 = Pointer(spv::StorageClassOutput, vec4_float_type);
		const auto zero = builder.Constant(spv::OpConstantNull, vec4_float_type);

		// Every input is read once, before the first EmitVertex.
		std::array<uint32_t, 3> positions {};
		std::array<uint32_t, 3> clips {};
		std::array<uint32_t, 3> culls {};
		for (uint32_t vertex = 0; vertex < 3u; vertex++) {
			positions[vertex] = Load(vec4_float_type, Access(ptr_in_vec4, gl_in, Int(vertex), Int(0)));
			if (clip_member != 0) {
				clips[vertex] = Load(clip_type, Access(Pointer(spv::StorageClassInput, clip_type),
				                                       gl_in, Int(vertex), Int(clip_member)));
			}
			if (cull_member != 0) {
				culls[vertex] = Load(cull_type, Access(Pointer(spv::StorageClassInput, cull_type),
				                                       gl_in, Int(vertex), Int(cull_member)));
			}
		}
		std::vector<std::array<uint32_t, 3>> values(interface.parameters.size());
		for (size_t i = 0; i < interface.parameters.size(); i++) {
			for (uint32_t vertex = 0; vertex < 3u; vertex++) {
				values[i][vertex] =
				    bound[i].input != 0
				        ? Load(vec4_float_type, Access(ptr_in_vec4, bound[i].input, Int(vertex)))
				        : zero;
			}
		}
		const auto primitive_id = interface.primitive_id ? Load(int_type, primitive_id_in) : 0u;

		for (uint32_t vertex = 0; vertex < 3u; vertex++) {
			Store(Access(ptr_out_vec4, gl_out, Int(0)), positions[vertex]);
			if (clip_member != 0) {
				Store(Access(Pointer(spv::StorageClassOutput, clip_type), gl_out, Int(clip_member)),
				      clips[vertex]);
			}
			if (cull_member != 0) {
				Store(Access(Pointer(spv::StorageClassOutput, cull_type), gl_out, Int(cull_member)),
				      culls[vertex]);
			}
			for (size_t i = 0; i < interface.parameters.size(); i++) {
				if (interface.parameters[i].raw) {
					// The same three raw vertices at every corner.
					for (uint32_t source = 0; source < 3u; source++) {
						Store(bound[i].outputs[source], values[i][source]);
					}
				} else {
					Store(bound[i].outputs[0], values[i][vertex]);
				}
			}
			if (interface.primitive_id) {
				Store(primitive_id_out, primitive_id);
			}
			builder.AddFunction(spv::OpEmitVertex);
		}
		builder.AddFunction(spv::OpEndPrimitive);
		builder.AddFunction(spv::OpReturn);
		builder.AddFunction(spv::OpFunctionEnd);
		return builder.Build();
	}

private:
	uint32_t Pointer(spv::StorageClass storage, uint32_t type) {
		return builder.Type(spv::OpTypePointer, storage, type);
	}

	uint32_t Array(uint32_t type, uint32_t size) {
		return builder.Type(spv::OpTypeArray, type, builder.Constant(spv::OpConstant, uint_type, size));
	}

	template <typename... Args>
	uint32_t Result(spv::Op opcode, uint32_t type, Args... operands) {
		const auto id = builder.AllocateId();
		builder.AddFunction(opcode, type, id, operands...);
		return id;
	}

	template <typename... Args>
	uint32_t ResultWithoutType(spv::Op opcode, Args... operands) {
		const auto id = builder.AllocateId();
		builder.AddFunction(opcode, id, operands...);
		return id;
	}

	template <typename... Args>
	uint32_t Access(uint32_t pointer_type, uint32_t base, Args... indices) {
		return Result(spv::OpAccessChain, pointer_type, base, indices...);
	}

	uint32_t Load(uint32_t type, uint32_t pointer) { return Result(spv::OpLoad, type, pointer); }

	void Store(uint32_t pointer, uint32_t value) { builder.AddFunction(spv::OpStore, pointer, value); }

	uint32_t Int(uint32_t value) { return builder.Constant(spv::OpConstant, int_type, value); }

	uint32_t AddInterface(spv::StorageClass storage, uint32_t type) {
		const auto variable = builder.DefineGlobalVariable(Pointer(storage, type), storage);
		interfaces.push_back(variable);
		return variable;
	}

	Builder                           builder {SpirvVersion15};
	const PerVertexGeometryInterface& interface;
	std::vector<uint32_t>             interfaces;
	uint32_t                          main            = 0;
	uint32_t                          void_type       = 0;
	uint32_t                          uint_type       = 0;
	uint32_t                          int_type        = 0;
	uint32_t                          float_type      = 0;
	uint32_t                          vec4_float_type = 0;
	uint32_t                          clip_type       = 0;
	uint32_t                          cull_type       = 0;
	uint32_t                          clip_member     = 0;
	uint32_t                          cull_member     = 0;
};

} // namespace

bool PerVertexGeometryNeeded(const ShaderPixelInputInfo* pixel_info) {
	if (pixel_info == nullptr || pixel_info->stage.program == nullptr ||
	    !ShaderRecompiler::PsPerVertexGeometry() ||
	    pixel_info->parameter_mode == ShaderPixelParameterMode::Rectangle) {
		return false;
	}
	return std::ranges::any_of(pixel_info->stage.program->info.inputs, [](const IR::StageInput& input) {
		return input.kind == IR::StageInputKind::Parameter && input.per_vertex;
	});
}

PerVertexGeometryInterface DescribePerVertexGeometry(const ShaderVertexInputInfo& vertex_info,
                                                     const ShaderPixelInputInfo&  pixel_info) {
	EXIT_IF(pixel_info.stage.program == nullptr || vertex_info.stage.program == nullptr);
	const auto& vertex = *vertex_info.stage.program;
	const auto& pixel  = *pixel_info.stage.program;

	PerVertexGeometryInterface result;
	std::vector<uint32_t>      active;
	std::vector<uint32_t>      raw;
	for (const auto& input: pixel.info.inputs) {
		if (input.kind == IR::StageInputKind::Parameter) {
			active.push_back(input.location);
			if (input.per_vertex) {
				raw.push_back(input.location);
			}
		} else if (input.kind == IR::StageInputKind::PrimitiveId) {
			result.primitive_id = true;
		}
	}
	const auto slots = ShaderPixelPerVertexSlots(pixel_info, active, raw);
	std::array<bool, ShaderMaxPerVertexLocations> written {};
	for (const auto& slot: slots) {
		result.parameters.push_back({.input_location   = slot.source,
		                             .output_location  = slot.locations[0],
		                             .raw              = true,
		                             .vertex1_location = slot.locations[1],
		                             .vertex2_location = slot.locations[2],
		                             .exported = (vertex.param_export_mask & (1u << slot.source)) != 0});
		for (const auto location: slot.locations) {
			written[location] = true;
		}
	}
	for (const auto input: active) {
		const auto source   = ShaderPixelParameterMappedLocation(pixel_info, input);
		const auto location = ShaderPixelParameterLocation(pixel_info, active, input);
		if (location >= written.size() || written[location]) {
			continue;
		}
		written[location] = true;
		result.parameters.push_back(
		    {.input_location  = source,
		     .output_location = location,
		     .exported        = (vertex.param_export_mask & (1u << (source & 31u))) != 0});
	}

	// The vertex shader's built-in outputs, as DefineOutputs (spirvEmitterModule.cpp) declares them.
	bool has_position = false;
	for (const auto& output: vertex.info.outputs) {
		switch (output.kind) {
			case IR::StageOutputKind::Position: has_position = true; break;
			case IR::StageOutputKind::ClipDistance:
				result.clip_distances = std::max(result.clip_distances, output.index + 1u);
				break;
			case IR::StageOutputKind::CullDistance:
				result.cull_distances = std::max(result.cull_distances, output.index + 1u);
				break;
			case IR::StageOutputKind::Layer:
			case IR::StageOutputKind::ViewportIndex: result.unsupported = true; break;
			default: break;
		}
	}
	if (has_position && vertex_info.user_clip_planes != 0) {
		result.clip_distances += static_cast<uint32_t>(std::popcount(vertex_info.user_clip_planes));
	}
	if (vertex.stage == ShaderType::Vertex && ShaderRecompiler::GetCodegenOptions().clip_guard &&
	    result.clip_distances + result.cull_distances < 8u && has_position) {
		result.clip_distances++;
	}
	result.invariant =
	    ShaderRecompiler::GetCodegenOptions().mad_mode != ShaderRecompiler::MadMode::Fused;
	return result;
}

std::vector<uint32_t> BuildPerVertexGeometryShader(const PerVertexGeometryInterface& interface) {
	EXIT_IF(interface.unsupported);
	return PerVertexGeometryEmitter(interface).Emit();
}

} // namespace Libs::Graphics
