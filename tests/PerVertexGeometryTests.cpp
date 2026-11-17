// nps5 kart: offline tests of raw-vertex pixel inputs through the generated geometry stage
// (KYTY_PS_PER_VERTEX=gs; graphics/shader/perVertexShader.h), no GPU:
//  - the host flag and the mode switch,
//  - the location layout shared by the pixel shader and the generated stage,
//  - the interface description (parameters, clip/cull distances, primitive id, unsupported cases),
//  - the generated SPIR-V: valid (spirv-val), three vertices, the expected input and output
//    locations, every raw output written with the same vertex's value at each corner.
// nps5 kart2: and of the variant without a geometry stage (KYTY_PS_PER_VERTEX=unpack;
// ir/passes/RawVertexUnpack.h):
//  - field codes and slot packing, the slot layout (also against random interfaces), the link to
//    the vertex shader's parameter aliases,
//  - the vertex export of every field kind, evaluated on edge and random words,
//  - the data flow per variant: a pixel program that unpacks and interpolates raw vertices the way
//    the game's shaders do is evaluated on three raw vertices (the reference: what PerVertexKHR
//    inputs give) and, after the pass, on the fields a host would interpolate from the vertex
//    shader's exports; signed, unsigned, half-float and negated-half words, clamps, integer uses
//    (rounded fields), whole-dword uses, P0-only parameters (flat copies), ordinary parameters
//    (zero differences), packed halves, both signednesses of one word,
//  - what the pass leaves in the program per variant.
// The SPIR-V of both stages is validated on the real shaders by tools/shader-journal/journal_unpack.
#include "graphics/shader/perVertexShader.h"
#include "graphics/shader/recompiler/CodegenOptions.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/RawVertexUnpack.h"
#include "graphics/shader/shader.h"

#include "graphics/shader/recompiler/backend/spirv/SpirvBuilder.h"

#include "spirv-tools/libspirv.hpp"

#include <cstdio>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <memory>
#include <random>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace {

using namespace Libs::Graphics;
namespace IR = Libs::Graphics::ShaderRecompiler::IR;

int g_failures = 0;
int g_checks   = 0;

void Check(bool condition, const std::string& message) {
	g_checks++;
	if (!condition) {
		std::fprintf(stderr, "FAIL: %s\n", message.c_str());
		g_failures++;
	}
}

constexpr uint32_t FlatShade = 0x400u;

// A pixel shader's input info and compiled info: parameter `input` maps to export `offset`.
struct Pixel {
	ShaderPixelInputInfo   info;
	IR::CompiledShaderInfo program;
	void Add(uint32_t input, uint32_t offset, bool per_vertex, bool flat = false, bool custom = false) {
		info.interpolator_settings[input] = offset | (flat ? FlatShade : 0u);
		info.input_num                    = std::max(info.input_num, input + 1u);
		if (custom) info.custom_interpolation_mask |= 1u << input;
		program.info.inputs.push_back({IR::StageInputKind::Parameter, input, 4, "p", per_vertex});
	}
	void Finish() {
		program.stage      = ShaderType::Pixel;
		info.stage.program = &program;
	}
};

struct Vertex {
	ShaderVertexInputInfo  info;
	IR::CompiledShaderInfo program;
	explicit Vertex(uint32_t export_mask) {
		program.stage             = ShaderType::Vertex;
		program.param_export_mask = export_mask;
		program.info.outputs.push_back({IR::StageOutputKind::Position, 0, 0, "pos"});
		info.stage.program = &program;
	}
};

struct Module {
	std::map<uint32_t, uint32_t>              location;  // variable -> location
	std::map<uint32_t, uint32_t>              storage;   // variable -> storage class
	std::vector<std::pair<uint32_t, uint32_t>> stores;    // (pointer, value), in order
	std::map<uint32_t, std::vector<uint32_t>> access;    // result -> (base, indices...)
	std::map<uint32_t, uint32_t>              loads;     // result -> pointer
	std::map<uint32_t, uint32_t>              constants; // id -> value
	std::vector<size_t>                       emits;     // number of stores before each EmitVertex
	std::set<uint32_t>                        builtins;
	uint32_t                                  end_primitive = 0;
};

Module Parse(const std::vector<uint32_t>& spirv) {
	Module m;
	for (size_t at = 5; at < spirv.size();) {
		const auto     count = spirv[at] >> 16u;
		const auto     op    = static_cast<spv::Op>(spirv[at] & 0xffffu);
		const uint32_t* w    = &spirv[at];
		switch (op) {
			case spv::OpDecorate:
				if (w[2] == spv::DecorationLocation) m.location[w[1]] = w[3];
				if (w[2] == spv::DecorationBuiltIn) m.builtins.insert(w[3]);
				break;
			case spv::OpMemberDecorate:
				if (w[3] == spv::DecorationBuiltIn) m.builtins.insert(w[4]);
				break;
			case spv::OpVariable: m.storage[w[2]] = w[3]; break;
			case spv::OpConstant: m.constants[w[2]] = w[3]; break;
			case spv::OpAccessChain: m.access[w[2]] = std::vector<uint32_t>(w + 3, w + count); break;
			case spv::OpLoad: m.loads[w[2]] = w[3]; break;
			case spv::OpStore: m.stores.emplace_back(w[1], w[2]); break;
			case spv::OpEmitVertex: m.emits.push_back(m.stores.size()); break;
			case spv::OpEndPrimitive: m.end_primitive++; break;
			default: break;
		}
		at += count;
	}
	return m;
}

bool Validate(const std::vector<uint32_t>& spirv, const char* what) {
	spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_3);
	std::string          error;
	tools.SetMessageConsumer([&](spv_message_level_t, const char*, const spv_position_t&, const char* message) {
		error += message;
		error += '\n';
	});
	const bool valid = tools.Validate(spirv);
	Check(valid, std::string("spirv-val: ") + what + ": " + error);
	return valid;
}

// The vertex an output variable's stored value comes from at corner `corner`: the constant index
// of the input array element the value was loaded from.
int SourceVertex(const Module& m, uint32_t value) {
	const auto load = m.loads.find(value);
	if (load == m.loads.end()) return -1;
	const auto chain = m.access.find(load->second);
	if (chain == m.access.end() || chain->second.size() != 2) return -1;
	const auto index = m.constants.find(chain->second[1]);
	return index == m.constants.end() ? -1 : static_cast<int>(index->second);
}

void TestMode() {
	using namespace Libs::Graphics::ShaderRecompiler;
	Check(GetHostPsPerVertex(), "the host has per-vertex inputs until the device layer says otherwise");
	Check(!PsPerVertexGeometry(), "no emulation on a host with PerVertexKHR inputs");
	Check(!PsPerVertexUnpack(), "no unpacked fields on a host with PerVertexKHR inputs");
	SetHostPsPerVertex(false);
	Check(!PsPerVertexGeometry(), "a host without PerVertexKHR inputs takes no geometry stage unasked");
	Check(PsPerVertexUnpack(), "a host without PerVertexKHR inputs takes unpacked fields");
	{
		Pixel unpack;
		unpack.Add(0, 0, false, false, true);
		unpack.Finish();
		Check(!ShaderPixelRawVertexInputs(unpack.info), "unpack: no raw-vertex inputs");
		Check(ShaderPixelRawVertexUnpack(unpack.info), "unpack: raw-vertex reads are unpacked");
		Check(!PerVertexGeometryNeeded(&unpack.info), "unpack: no geometry stage");
		unpack.info.parameter_mode = ShaderPixelParameterMode::Rectangle;
		Check(!ShaderPixelRawVertexUnpack(unpack.info), "unpack: not for rectangle lists (unlinked vertex stage)");
	}
	// KYTY_PS_PER_VERTEX=gs for the tests of the generated stage below.
	SetPsPerVertexModeForTests(1);
	Check(PsPerVertexGeometry() && !PsPerVertexUnpack(), "gs mode");
	Pixel pixel;
	pixel.Add(0, 0, true);
	pixel.Finish();
	Check(ShaderPixelRawVertexInputs(pixel.info), "raw vertices are available through the stage");
	Check(PerVertexGeometryNeeded(&pixel.info), "a per-vertex parameter needs the stage");
	pixel.info.parameter_mode = ShaderPixelParameterMode::Rectangle;
	Check(!ShaderPixelRawVertexInputs(pixel.info), "rectangle lists have no generated geometry stage");
	Check(!PerVertexGeometryNeeded(&pixel.info), "rectangle lists do not ask for the stage");
	Pixel plain;
	plain.Add(0, 0, false);
	plain.Finish();
	Check(!PerVertexGeometryNeeded(&plain.info), "no per-vertex parameter, no stage");
	Check(!PerVertexGeometryNeeded(nullptr), "no pixel shader, no stage");
}

void TestSlots() {
	Pixel pixel;
	pixel.Add(0, 0, false);
	pixel.Add(1, 1, false);
	pixel.Add(2, 3, true, false, true); // export 3, raw
	pixel.Add(3, 5, true, false, true); // export 5, raw
	pixel.Add(4, 5, true, true);        // flat alias of export 5: one slot
	pixel.Finish();
	const std::vector<uint32_t> active {0, 1, 2, 3, 4};
	const std::vector<uint32_t> raw {2, 3, 4};
	const auto slots = ShaderPixelPerVertexSlots(pixel.info, active, raw);
	Check(slots.size() == 2, "one slot per raw export");
	if (slots.size() == 2) {
		Check(slots[0].source == 3 && slots[0].locations == std::array<uint32_t, 3> {3, 2, 4},
		      "export 3: vertex 0 in place, vertices 1 and 2 at the lowest free locations");
		Check(slots[1].source == 5 && slots[1].locations == std::array<uint32_t, 3> {5, 6, 7},
		      "export 5: the next free locations");
	}
	std::set<uint32_t> all {0, 1};
	for (const auto& slot: slots) {
		for (const auto location: slot.locations) {
			Check(all.insert(location).second, "no location is used twice");
			Check(location < ShaderMaxPerVertexLocations, "locations stay below the limit");
		}
	}
	// Every layout of 1..10 raw exports among up to 10 parameters is collision-free.
	for (uint32_t mask = 1; mask < 1024; mask++) {
		Pixel many;
		std::vector<uint32_t> a, r;
		for (uint32_t input = 0; input < 10; input++) {
			many.Add(input, input * 2u, (mask >> input & 1u) != 0);
			a.push_back(input);
			if ((mask >> input & 1u) != 0) r.push_back(input);
		}
		many.Finish();
		std::set<uint32_t> used;
		for (uint32_t input = 0; input < 10; input++) {
			if ((mask >> input & 1u) == 0) used.insert(input * 2u);
		}
		bool ok = true;
		for (const auto& slot: ShaderPixelPerVertexSlots(many.info, a, r)) {
			ok = ok && slot.locations[0] == slot.source;
			for (const auto location: slot.locations) ok = ok && used.insert(location).second && location < 32u;
		}
		if (!ok) {
			Check(false, "layout collision for mask " + std::to_string(mask));
			break;
		}
	}
	Check(true, "1023 layouts");
}

void TestInterface() {
	Pixel pixel;
	pixel.Add(0, 0, false);
	pixel.Add(1, 2, true, false, true);
	pixel.Add(2, 4, false, true);
	pixel.program.info.inputs.push_back({IR::StageInputKind::PrimitiveId, 0, 1, "gl_PrimitiveID"});
	pixel.Finish();
	Vertex vertex(0b00101); // exports 0 and 2, not 4
	vertex.program.info.outputs.push_back({IR::StageOutputKind::CullDistance, 1, 0, "cull"});
	vertex.info.user_clip_planes = 0; // (user planes exclude shader distances)
	const auto described = DescribePerVertexGeometry(vertex.info, pixel.info);
	Check(!described.unsupported, "supported");
	Check(described.primitive_id, "the primitive id is forwarded when the pixel shader reads it");
	Check(described.cull_distances == 2, "cull distances up to the highest index");
	Check(described.clip_distances ==
	          (Libs::Graphics::ShaderRecompiler::GetCodegenOptions().clip_guard ? 1u : 0u),
	      "the clip guard's plane is forwarded");
	Check(described.parameters.size() == 3, "three parameters");
	uint32_t raw = 0;
	for (const auto& parameter: described.parameters) {
		if (parameter.raw) {
			raw++;
			Check(parameter.input_location == 2 && parameter.output_location == 2 &&
			          parameter.vertex1_location == 1 && parameter.vertex2_location == 3 && parameter.exported,
			      "raw parameter: export 2 at locations 2, 1, 3");
		} else if (parameter.input_location == 4) {
			Check(!parameter.exported, "export 4 is missing from the vertex shader");
		} else {
			Check(parameter.input_location == 0 && parameter.output_location == 0 && parameter.exported,
			      "plain parameter passes through at its location");
		}
	}
	Check(raw == 1, "one raw parameter");

	Vertex planes(0b00101);
	planes.info.user_clip_planes = 0b101;
	Check(DescribePerVertexGeometry(planes.info, pixel.info).clip_distances ==
	          2u + (Libs::Graphics::ShaderRecompiler::GetCodegenOptions().clip_guard ? 1u : 0u),
	      "user clip planes are forwarded");
	Vertex layered(0b00101);
	layered.program.info.outputs.push_back({IR::StageOutputKind::Layer, 0, 0, "layer"});
	Check(DescribePerVertexGeometry(layered.info, pixel.info).unsupported,
	      "a vertex shader writing the layer cannot be followed by the stage");
}

void TestSpirv() {
	for (uint32_t variant = 0; variant < 16; variant++) {
		PerVertexGeometryInterface interface;
		interface.clip_distances = (variant & 1u) != 0 ? 3 : 0;
		interface.cull_distances = (variant & 2u) != 0 ? 2 : 0;
		interface.primitive_id   = (variant & 4u) != 0;
		interface.invariant      = (variant & 8u) != 0;
		interface.parameters.push_back({.input_location = 0, .output_location = 0, .exported = true});
		interface.parameters.push_back({.input_location   = 3,
		                                .output_location  = 3,
		                                .raw              = true,
		                                .vertex1_location = 1,
		                                .vertex2_location = 2,
		                                .exported         = true});
		interface.parameters.push_back({.input_location = 5, .output_location = 5, .exported = (variant & 1u) != 0});
		interface.parameters.push_back({.input_location   = 6,
		                                .output_location  = 6,
		                                .raw              = true,
		                                .vertex1_location = 4,
		                                .vertex2_location = 7,
		                                .exported         = (variant & 2u) != 0});
		const auto spirv = BuildPerVertexGeometryShader(interface);
		if (!Validate(spirv, ("variant " + std::to_string(variant)).c_str())) continue;
		const auto m = Parse(spirv);
		Check(m.emits.size() == 3 && m.end_primitive == 1, "three vertices, one primitive");
		std::set<uint32_t> in_locations, out_locations;
		for (const auto& [variable, location]: m.location) {
			(m.storage.at(variable) == spv::StorageClassInput ? in_locations : out_locations).insert(location);
		}
		std::set<uint32_t> expected_in {0, 3};
		if ((variant & 1u) != 0) expected_in.insert(5);
		if ((variant & 2u) != 0) expected_in.insert(6);
		Check(in_locations == expected_in, "inputs: the exported parameters");
		Check(out_locations == std::set<uint32_t> {0, 1, 2, 3, 4, 5, 6, 7}, "outputs: every location the pixel shader reads");
		Check(m.builtins.contains(spv::BuiltInPosition), "position");
		Check(m.builtins.contains(spv::BuiltInClipDistance) == (interface.clip_distances != 0), "clip distances");
		Check(m.builtins.contains(spv::BuiltInCullDistance) == (interface.cull_distances != 0), "cull distances");
		Check(m.builtins.contains(spv::BuiltInPrimitiveId) == interface.primitive_id, "primitive id");
		// Per corner: a raw output holds its own vertex (1 -> vertex 1, ...), a plain one the corner's.
		const std::map<uint32_t, int> raw_vertex {{3, 0}, {1, 1}, {2, 2}};
		size_t begin = 0;
		for (uint32_t corner = 0; corner < 3 && m.emits.size() == 3; corner++) {
			for (size_t i = begin; i < m.emits[corner]; i++) {
				const auto [pointer, value] = m.stores[i];
				const auto location         = m.location.find(pointer);
				if (location == m.location.end()) continue;
				if (const auto raw = raw_vertex.find(location->second); raw != raw_vertex.end()) {
					Check(SourceVertex(m, value) == raw->second, "a raw output carries its vertex at every corner");
				} else if (location->second == 0) {
					Check(SourceVertex(m, value) == static_cast<int>(corner), "a plain output carries the corner's value");
				}
			}
			begin = m.emits[corner];
		}
	}
}


// ---------------------------------------------------------------------------------------------
// KYTY_PS_PER_VERTEX=unpack
// ---------------------------------------------------------------------------------------------

using Kind = ShaderParameterFieldKind;
using Libs::Graphics::ShaderRecompiler::SetPsPerVertexModeForTests;
constexpr uint32_t Custom = 1u;

float HalfToFloat(uint16_t h) {
	const uint32_t sign = (h >> 15u) & 1u;
	const uint32_t exp  = (h >> 10u) & 31u;
	const uint32_t man  = h & 1023u;
	float          value;
	if (exp == 0) {
		value = std::ldexp(static_cast<float>(man), -24);
	} else if (exp == 31) {
		value = man == 0 ? INFINITY : NAN;
	} else {
		value = std::ldexp(static_cast<float>(man + 1024u), static_cast<int>(exp) - 25);
	}
	return sign != 0 ? -value : value;
}

uint16_t FloatToHalf(float f) {
	// Nearest half (ties to even), by search over the finite halves of f's sign.
	const uint16_t sign = std::signbit(f) ? 0x8000u : 0u;
	const float    a    = std::fabs(f);
	uint16_t       lo = 0, hi = 0x7c00;
	while (hi - lo > 1) {
		const auto mid = static_cast<uint16_t>((lo + hi) / 2);
		(HalfToFloat(mid) <= a ? lo : hi) = mid;
	}
	const float dl = a - HalfToFloat(lo);
	const float dh = HalfToFloat(hi) - a;
	const auto  pick = dl < dh ? lo : dh < dl ? hi : (lo & 1u) == 0 ? lo : hi;
	return static_cast<uint16_t>(sign | pick);
}

// One block of IR and a sink (an export of four dwords) to evaluate.
struct Shader {
	IR::Program program;
	IR::Block*  block = nullptr;
	IR::Inst*   sink  = nullptr;

	Shader() {
		auto owned = std::make_unique<IR::Block>();
		block      = owned.get();
		program.blocks.push_back(block);
		program.block_info.push_back({.id = 0});
		program.block_storage.push_back(std::move(owned));
		program.stage = ShaderType::Pixel;
	}
	IR::Value Emit(IR::ValueOpcode opcode, std::initializer_list<IR::Value> args) {
		return IR::Value(&block->AppendNewInst(opcode, args));
	}
	IR::Value Read(uint32_t attr, uint32_t chan, uint32_t mode) {
		return Emit(IR::ValueOpcode::GetInterpolationParameter, {IR::Value(attr), IR::Value(chan), IR::Value(mode)});
	}
	// Float results as dwords.
	void Sink(IR::Value a, IR::Value b = IR::Value(uint32_t {0}), IR::Value c = IR::Value(uint32_t {0}),
	          IR::Value d = IR::Value(uint32_t {0})) {
		const auto data = Emit(IR::ValueOpcode::CompositeConstructU32x4, {a, b, c, d});
		program.export_info.push_back({IR::ExportTargetKind::Mrt, 0, 0, 0xf, true, false, false});
		sink = &block->AppendNewInst(IR::ValueOpcode::SetAttribute, {data, IR::Value(true)});
	}
	IR::Value Bits(IR::Value f) { return Emit(IR::ValueOpcode::BitCastU32F32, {f}); }
};

using Words = std::array<uint32_t, 4>;

// Evaluates the IR subset the tests and the pass use.
struct Machine {
	// Raw vertices (0..2) of each pixel input.
	std::map<uint32_t, std::array<Words, 3>> raw;
	uint32_t                                 custom_mask = 0;
	// Barycentric weights of vertices 0..2.
	std::array<double, 3> weight {1.0, 0.0, 0.0};
	// After the pass: an ordinary parameter's P0 is the host-interpolated input.
	bool                                     interpolated_inputs = false;
	std::function<float(uint32_t, uint32_t)> field;
	uint32_t                                 raw_reads = 0;
	std::map<const IR::Inst*, Words>         memo;

	float Blend(float a, float b, float c) const {
		return static_cast<float>(weight[0] * a + weight[1] * b + weight[2] * c);
	}
	Words Eval(const IR::Value& value) {
		if (value.IsImmediate()) {
			switch (value.GetType()) {
				case IR::Type::U1: return {value.U1() ? 1u : 0u, 0, 0, 0};
				case IR::Type::U16: return {value.U16(), 0, 0, 0};
				case IR::Type::F32: return {std::bit_cast<uint32_t>(value.F32Value()), 0, 0, 0};
				default: return {value.U32(), 0, 0, 0};
			}
		}
		const auto* inst = value.Instruction();
		if (const auto found = memo.find(inst); found != memo.end()) {
			return found->second;
		}
		const auto result = Run(*inst);
		memo.emplace(inst, result);
		return result;
	}
	uint32_t U(const IR::Inst& inst, size_t arg) { return Eval(inst.Arg(arg))[0]; }
	float    F(const IR::Inst& inst, size_t arg) { return std::bit_cast<float>(U(inst, arg)); }
	static Words Float(float f) { return {std::bit_cast<uint32_t>(f), 0, 0, 0}; }
	Words Run(const IR::Inst& inst) {
		using O = IR::ValueOpcode;
		switch (inst.GetOpcode()) {
			case O::GetInterpolationParameter: {
				const auto  attr = inst.Arg(0).U32();
				const auto  chan = inst.Arg(1).U32();
				const auto  mode = inst.Arg(2).U32();
				const auto& v    = raw.at(attr);
				raw_reads++;
				if ((custom_mask & (1u << attr)) != 0) {
					return {v[(mode + 1u) % 3u][chan], 0, 0, 0};
				}
				const auto f = [&](uint32_t vertex) { return std::bit_cast<float>(v[vertex][chan]); };
				if (mode == 2u) {
					return Float(interpolated_inputs ? Blend(f(0), f(1), f(2)) : f(0));
				}
				return Float(f(mode + 1u) - f(0));
			}
			case O::GetAttributeF16: {
				const auto& v    = raw.at(inst.Arg(0).U32());
				const auto  half = [&](uint32_t vertex) {
					return HalfToFloat(static_cast<uint16_t>(v[vertex][inst.Arg(1).U32()] >> (inst.Arg(2).U32() * 16u)));
				};
				raw_reads++;
				return Float(Blend(half(0), half(1), half(2)));
			}
			case O::GetParameterField: return Float(field(inst.Arg(0).U32(), inst.Arg(1).U32()));
			case O::BitFieldUExtract: {
				const auto count = U(inst, 2);
				return {(U(inst, 0) >> U(inst, 1)) & (count >= 32u ? ~0u : (1u << count) - 1u), 0, 0, 0};
			}
			case O::BitFieldSExtract: {
				const auto count = U(inst, 2);
				const auto bits  = (U(inst, 0) >> U(inst, 1)) & ((1u << count) - 1u);
				const auto sign  = 1u << (count - 1u);
				return {(bits ^ sign) - sign, 0, 0, 0};
			}
			case O::ConvertF32S32: return Float(static_cast<float>(static_cast<int32_t>(U(inst, 0))));
			case O::ConvertF32U32: return Float(static_cast<float>(U(inst, 0)));
			case O::ConvertS32F32: return {static_cast<uint32_t>(static_cast<int32_t>(F(inst, 0))), 0, 0, 0};
			case O::ConvertU16U32: return {U(inst, 0) & 0xffffu, 0, 0, 0};
			case O::ConvertU32U16: return {U(inst, 0) & 0xffffu, 0, 0, 0};
			case O::BitCastF16U16:
			case O::BitCastU16F16:
			case O::BitCastU32F32:
			case O::BitCastF32U32: return {U(inst, 0), 0, 0, 0};
			case O::ConvertF32F16: return Float(HalfToFloat(static_cast<uint16_t>(U(inst, 0))));
			case O::ConvertF16F32: return {FloatToHalf(F(inst, 0)), 0, 0, 0};
			case O::BitwiseXor32: return {U(inst, 0) ^ U(inst, 1), 0, 0, 0};
			case O::BitwiseAnd32: return {U(inst, 0) & U(inst, 1), 0, 0, 0};
			case O::BitwiseOr32: return {U(inst, 0) | U(inst, 1), 0, 0, 0};
			case O::ShiftLeftLogical32: return {U(inst, 0) << U(inst, 1), 0, 0, 0};
			case O::ShiftRightLogical32: return {U(inst, 0) >> U(inst, 1), 0, 0, 0};
			case O::ShiftRightArithmetic32:
				return {static_cast<uint32_t>(static_cast<int32_t>(U(inst, 0)) >> U(inst, 1)), 0, 0, 0};
			case O::FPMul32: return Float(F(inst, 0) * F(inst, 1));
			case O::FPAdd32: return Float(F(inst, 0) + F(inst, 1));
			case O::FPSub32: return Float(F(inst, 0) - F(inst, 1));
			case O::FPNeg32: return Float(-F(inst, 0));
			case O::FPMax32: return Float(std::max(F(inst, 0), F(inst, 1)));
			case O::FPRoundEven32: return Float(std::nearbyint(F(inst, 0)));
			case O::SelectU32: return {U(inst, 0) != 0 ? U(inst, 1) : U(inst, 2), 0, 0, 0};
			case O::CompositeConstructU32x4: return {U(inst, 0), U(inst, 1), U(inst, 2), U(inst, 3)};
			case O::CompositeExtractU32x4: return {Eval(inst.Arg(0))[inst.Arg(1).U32()], 0, 0, 0};
			default:
				Check(false, std::string("machine: opcode ") + std::string(IR::ValueOpcodeName(inst.GetOpcode())));
				return {};
		}
	}
};

void TestUnpackCodes() {
	SetPsPerVertexModeForTests(2);
	for (uint32_t kind = 1; kind <= 4; kind++) {
		for (uint32_t chan = 0; chan < 4; chan++) {
			for (uint32_t word = 0; word < 2; word++) {
				const auto code = ShaderParameterFieldCode(static_cast<Kind>(kind), chan, word);
				Check(code != 0 && code < 64u, "field code fits six bits");
				Check(ShaderParameterFieldCodeKind(code) == static_cast<Kind>(kind) &&
				          ShaderParameterFieldCodeChannel(code) == chan && ShaderParameterFieldCodeWord(code) == word,
				      "field code round trip");
			}
		}
	}
	std::mt19937 rng(7);
	for (uint32_t i = 0; i < 2000; i++) {
		ShaderParameterFieldSlot slot;
		slot.location = rng() % 32u;
		slot.input    = rng() % 32u;
		for (auto& code: slot.codes) {
			code = rng() % 64u;
		}
		const auto packed = slot.Pack();
		Check(ShaderParameterFieldSlot::Unpack(slot.location, packed) == slot, "slot pack round trip");
		Check((packed & ShaderParamAliasSourceMask) == slot.input, "the low bits are the source");
		Check((ShaderParamAliasFields(packed) != 0) ==
		          (slot.codes[0] != 0 || slot.codes[1] != 0 || slot.codes[2] != 0 || slot.codes[3] != 0),
		      "the field bits tell a field alias from a copy");
	}
	ShaderParameterFieldSlot raw {31, 2, {ShaderParameterFieldCode(Kind::Raw, 0, 0), 0, 0, 0}};
	ShaderParameterFieldSlot smooth {31, 2, {ShaderParameterFieldCode(Kind::S16, 0, 0), 0, 0, 0}};
	Check(raw.Flat() && !smooth.Flat(), "a slot of raw fields is flat");
}

void TestUnpackSlots() {
	Pixel pixel;
	pixel.Add(0, 0, false);
	pixel.Add(1, 1, false);
	pixel.Add(2, 3, false, false, true); // custom, export 3
	pixel.Add(3, 5, false, false, true); // custom, export 5
	pixel.Finish();
	const std::vector<uint32_t> active {0, 1, 2};
	const auto S = [](uint32_t c, uint32_t w) { return ShaderParameterFieldCode(Kind::S16, c, w); };
	const auto U = [](uint32_t c, uint32_t w) { return ShaderParameterFieldCode(Kind::U16, c, w); };
	std::vector<ShaderParameterField> fields {
	    {3, S(0, 0)}, {2, U(0, 1)}, {2, U(0, 0)}, {2, S(2, 0)}, {2, S(2, 1)}, {2, S(3, 0)},
	    {2, U(0, 0)}, // twice
	    {3, ShaderParameterFieldCode(Kind::Raw, 1, 0)},
	};
	uint32_t   dropped = 9;
	const auto slots   = ShaderPixelFieldSlots(pixel.info, active, fields, &dropped);
	Check(dropped == 0, "every field has a location");
	Check(slots.size() == 4, "two slots for input 2 (five fields), one smooth and one flat for input 3");
	if (slots.size() == 4) {
		Check(slots[0] == ShaderParameterFieldSlot {31, 2, {U(0, 0), U(0, 1), S(2, 0), S(2, 1)}}, "slot 0");
		Check(slots[1] == ShaderParameterFieldSlot {30, 2, {S(3, 0), 0, 0, 0}}, "slot 1");
		Check(slots[2] == ShaderParameterFieldSlot {29, 3, {S(0, 0), 0, 0, 0}}, "slot 2");
		Check(slots[3] == ShaderParameterFieldSlot {28, 3, {ShaderParameterFieldCode(Kind::Raw, 1, 0), 0, 0, 0}},
		      "slot 3 (raw apart)");
		Check(slots[3].Flat() && !slots[2].Flat(), "flatness");
	}
	// Any order of the fields gives the same slots.
	std::mt19937 rng(3);
	for (uint32_t i = 0; i < 50; i++) {
		std::shuffle(fields.begin(), fields.end(), rng);
		Check(ShaderPixelFieldSlots(pixel.info, active, fields) == slots, "the layout does not depend on the order");
	}
	// Locations in use are skipped: an input mapped to 31, and a relocated flat alias.
	Pixel busy;
	busy.Add(0, 31, false);
	busy.Add(1, 30, false, false, true);
	busy.Finish();
	const std::vector<uint32_t>             busy_active {0};
	const std::vector<ShaderParameterField> one {{1, S(0, 0)}};
	const auto                              busy_slots = ShaderPixelFieldSlots(busy.info, busy_active, one);
	Check(busy_slots.size() == 1 && busy_slots[0].location == 30, "a mapped location is skipped");
	// No location left: the fields are dropped and counted.
	Pixel full;
	std::vector<uint32_t> full_active;
	for (uint32_t i = 0; i < 31; i++) {
		full.Add(i, i, false);
		full_active.push_back(i);
	}
	full.Add(31, 31, false, false, true);
	full.Finish();
	const std::vector<ShaderParameterField> many {{31, S(0, 0)}, {31, S(1, 0)}, {31, S(2, 0)}, {31, S(3, 0)},
	                                             {31, S(0, 1)}, {31, S(1, 1)}};
	const auto full_slots = ShaderPixelFieldSlots(full.info, full_active, many, &dropped);
	Check(full_slots.size() == 1 && full_slots[0].location == 31 && dropped == 2,
	      "one free location: four fields placed, two dropped");
	Check(ShaderPixelFieldSlots(pixel.info, active, {}).empty(), "no fields, no slots");
}

void TestUnpackSlotsFuzz() {
	std::mt19937 rng(11);
	for (uint32_t round = 0; round < 400; round++) {
		Pixel                 pixel;
		const uint32_t        inputs = 1u + rng() % 12u;
		std::vector<uint32_t> active;
		for (uint32_t input = 0; input < inputs; input++) {
			const bool custom = rng() % 3u == 0;
			pixel.Add(input, rng() % (round % 2u == 0 ? 32u : 10u), false, !custom && rng() % 3u == 0, custom);
			if (rng() % 4u != 0) {
				active.push_back(input);
			}
		}
		pixel.Finish();
		std::vector<ShaderParameterField> fields;
		const uint32_t                    count = rng() % 40u;
		for (uint32_t i = 0; i < count; i++) {
			fields.push_back({rng() % inputs, ShaderParameterFieldCode(static_cast<Kind>(1u + rng() % 4u), rng() % 4u, rng() % 2u)});
		}
		uint32_t   dropped = 0;
		const auto slots   = ShaderPixelFieldSlots(pixel.info, active, fields, &dropped);
		std::set<uint32_t> taken;
		for (const auto input: active) {
			taken.insert(ShaderPixelParameterMappedLocation(pixel.info, input));
			taken.insert(ShaderPixelParameterLocation(pixel.info, active, input));
		}
		std::set<ShaderParameterField> placed;
		bool                           ok = true;
		for (const auto& slot: slots) {
			ok = ok && slot.location < 32u && taken.insert(slot.location).second;
			bool gap = false;
			for (const auto code: slot.codes) {
				if (code == 0) {
					gap = true;
					continue;
				}
				ok = ok && !gap; // fields fill a slot from component 0
				ok = ok && (ShaderParameterFieldCodeKind(code) == Kind::Raw) == slot.Flat();
				ok = ok && placed.insert({slot.input, code}).second;
			}
			ok = ok && slot.codes[0] != 0;
		}
		const std::set<ShaderParameterField> wanted(fields.begin(), fields.end());
		ok = ok && placed.size() + dropped == wanted.size();
		for (const auto& field: placed) {
			ok = ok && wanted.contains(field);
		}
		ok = ok && (dropped == 0 || taken.size() == 32u);
		Check(ok, "fuzz: unique free locations, every field once, raw apart, drops only when full (round " +
		              std::to_string(round) + ")");
	}
}

// A pixel program info with field slots, as ShaderInfoCollection publishes them.
void AddSlot(Pixel& pixel, const ShaderParameterFieldSlot& slot) {
	pixel.program.info.inputs.push_back({IR::StageInputKind::ParameterField, slot.location, slot.Pack(), "f", false});
}

void TestUnpackLink() {
	SetPsPerVertexModeForTests(2);
	const auto S = [](uint32_t c, uint32_t w) { return ShaderParameterFieldCode(Kind::S16, c, w); };
	{
		Pixel pixel;
		pixel.Add(0, 0, false);
		pixel.Add(2, 7, false, false, true); // custom input 2 reads export 7
		const ShaderParameterFieldSlot slot {31, 2, {S(0, 0), S(0, 1), S(1, 0), 0}};
		AddSlot(pixel, slot);
		pixel.Finish();
		Check(ShaderPixelHasFieldSlots(pixel.info), "the program reads field slots");
		ShaderVertexInputInfo vertex;
		vertex.param_alias_mask = 0x1234; // stale
		ShaderLinkVertexPixelParameters(vertex, pixel.info);
		Check(vertex.param_alias_mask == (1u << 31), "one alias: the slot's location");
		const auto source = vertex.param_alias_source[31];
		Check((source & ShaderParamAliasSourceMask) == 7, "the alias copies from the export the input maps to");
		Check(ShaderParamAliasFields(source) != 0, "a field alias");
		const auto linked = ShaderParameterFieldSlot::Unpack(31, source);
		Check(linked.codes == slot.codes && linked.input == 7, "the fields reach the vertex shader");
	}
	{
		// With a flat/smooth relocation of the same interface: both kinds of alias.
		Pixel pixel;
		pixel.Add(0, 4, false);              // smooth read of export 4
		pixel.Add(1, 4, false, true);        // flat read of export 4: relocated
		pixel.Add(2, 5, false, false, true); // custom
		const ShaderParameterFieldSlot slot {31, 2, {S(3, 1), 0, 0, 0}};
		AddSlot(pixel, slot);
		pixel.Finish();
		ShaderVertexInputInfo vertex;
		ShaderLinkVertexPixelParameters(vertex, pixel.info);
		const std::vector<uint32_t> active {0, 1, 2};
		const auto relocated = ShaderPixelParameterLocation(pixel.info, active, 1);
		Check(relocated != 4, "the flat alias moved");
		Check(vertex.param_alias_mask == ((1u << 31) | (1u << relocated)), "a copy alias and a field alias");
		Check(vertex.param_alias_source[relocated] == 4, "the copy alias has no fields");
		Check(ShaderParameterFieldSlot::Unpack(31, vertex.param_alias_source[31]) ==
		          ShaderParameterFieldSlot {31, 5, slot.codes},
		      "the field alias");
		// The slot layout never takes the relocated location.
		const std::vector<ShaderParameterField> fields {{2, S(3, 1)}};
		for (const auto& placed: ShaderPixelFieldSlots(pixel.info, active, fields)) {
			Check(placed.location != relocated && placed.location != 4 && placed.location != 5, "slot location is free");
		}
	}
	{
		Pixel pixel;
		pixel.Add(0, 0, false);
		pixel.Finish();
		Check(!ShaderPixelHasFieldSlots(pixel.info), "no field slots");
		ShaderVertexInputInfo vertex;
		vertex.param_alias_mask = 0xff;
		ShaderLinkVertexPixelParameters(vertex, pixel.info);
		Check(vertex.param_alias_mask == 0, "no aliases without slots or relocation");
		ShaderPixelInputInfo unbound;
		Check(!ShaderPixelHasFieldSlots(unbound), "no program, no slots");
	}
}

// The floats the vertex shader exports for a slot, from one vertex's dwords.
Words ExportOf(const Words& dwords, const ShaderParameterFieldSlot& slot) {
	Shader     shader;
	const auto data = shader.Emit(IR::ValueOpcode::CompositeConstructU32x4,
	                              {IR::Value(dwords[0]), IR::Value(dwords[1]), IR::Value(dwords[2]), IR::Value(dwords[3])});
	const auto out  = IR::EmitParameterFieldExport(*shader.block, shader.block->end(), data, slot);
	Machine    machine;
	return machine.Eval(out);
}

void TestUnpackExport() {
	const uint32_t edge[] = {0x0000, 0x0001, 0x7fff, 0x8000, 0x8001, 0xffff, 0x3c00, 0xbc00, 0x1234};
	std::mt19937   rng(5);
	for (uint32_t round = 0; round < 300; round++) {
		Words dwords;
		for (auto& dword: dwords) {
			const uint32_t lo = round < 81 ? edge[round % 9] : rng() & 0xffffu;
			const uint32_t hi = round < 81 ? edge[(round / 9) % 9] : rng() & 0xffffu;
			dword             = lo | (hi << 16u);
		}
		for (uint32_t chan = 0; chan < 4; chan++) {
			for (uint32_t word = 0; word < 2; word++) {
				const auto bits = static_cast<uint16_t>(dwords[chan] >> (word * 16u));
				const ShaderParameterFieldSlot slot {31, 0,
				                                     {ShaderParameterFieldCode(Kind::S16, chan, word),
				                                      ShaderParameterFieldCode(Kind::U16, chan, word),
				                                      ShaderParameterFieldCode(Kind::F16, chan, word),
				                                      ShaderParameterFieldCode(Kind::Raw, chan, word)}};
				const auto out = ExportOf(dwords, slot);
				bool       ok  = std::bit_cast<float>(out[0]) == static_cast<float>(static_cast<int16_t>(bits));
				ok             = ok && std::bit_cast<float>(out[1]) == static_cast<float>(bits);
				const auto half = HalfToFloat(bits);
				ok = ok && (std::isnan(half) ? std::isnan(std::bit_cast<float>(out[2]))
				                             : std::bit_cast<float>(out[2]) == half);
				ok = ok && out[3] == dwords[chan];
				Check(ok, "vertex export: signed, unsigned, half and raw of a word");
			}
		}
		const ShaderParameterFieldSlot sparse {30, 0, {ShaderParameterFieldCode(Kind::U16, 2, 1), 0, 0, 0}};
		const auto                     out = ExportOf(dwords, sparse);
		Check(std::bit_cast<float>(out[0]) == static_cast<float>(dwords[2] >> 16u) && out[1] == 0 && out[2] == 0 &&
		          out[3] == 0,
		      "vertex export: components without a field are zero");
	}
	Check(FloatToHalf(1.0f) == 0x3c00 && FloatToHalf(-2.0f) == 0xc000 && HalfToFloat(0x3800) == 0.5f, "half helpers");
}

// How a test shader turns one raw read into a float.
using Unpack = std::function<IR::Value(Shader&, IR::Value)>;

struct Flow {
	std::string name;
	// Builds the program: reads of input 0 (and others), a sink of floats.
	std::function<void(Shader&)> build;
	uint32_t                     inputs      = 1;
	uint32_t                     custom_mask = 1;
	// Words of the raw vertices: generator of one 16-bit word (kind of data the shader expects).
	std::function<uint32_t(std::mt19937&)> word;
	double                                 tolerance = 4e-6;
	// Expected counts of the pass.
	std::function<void(const IR::RawVertexUnpackStats&, const std::string&)> expect;
	// After the pass nothing reads raw vertices of custom parameters.
	bool raw_free = true;
	// The weights the builder baked into the program (I: vertex 1, J: vertex 2).
	float i = 0.0f, j = 0.0f;
	// Compare bit for bit.
	bool exact = false;
	// Packed words also for an ordinary parameter (packed halves).
	bool words = false;
};

// The shader's own interpolation of an unpacked value: f(P0) + i*(f(P10) - f(P0)) + j*(f(P20) - f(P0)),
// with I = weight of vertex 1 and J = weight of vertex 2 (baked into the program as immediates).
IR::Value Interpolate(Shader& s, const Unpack& unpack, uint32_t attr, uint32_t chan, float i, float j) {
	const auto p0  = unpack(s, s.Read(attr, chan, 2));
	const auto p10 = unpack(s, s.Read(attr, chan, 0));
	const auto p20 = unpack(s, s.Read(attr, chan, 1));
	const auto d1  = s.Emit(IR::ValueOpcode::FPSub32, {p10, p0});
	const auto d2  = s.Emit(IR::ValueOpcode::FPSub32, {p20, p0});
	const auto a   = s.Emit(IR::ValueOpcode::FPAdd32, {p0, s.Emit(IR::ValueOpcode::FPMul32, {d1, IR::Value::F32(i)})});
	return s.Emit(IR::ValueOpcode::FPAdd32, {a, s.Emit(IR::ValueOpcode::FPMul32, {d2, IR::Value::F32(j)})});
}

Unpack SignedWord(uint32_t word, bool clamp) {
	return [=](Shader& s, IR::Value read) {
		auto v = s.Emit(IR::ValueOpcode::ConvertF32S32,
		                {s.Emit(IR::ValueOpcode::BitFieldSExtract, {read, IR::Value(word * 16u), IR::Value(uint32_t {16})})});
		if (clamp) {
			v = s.Emit(IR::ValueOpcode::FPMax32, {IR::Value::F32(-32767.0f), v});
		}
		return s.Emit(IR::ValueOpcode::FPMul32, {IR::Value::F32(1.0f / 32767.0f), v});
	};
}
Unpack UnsignedWord(uint32_t word, bool signed_convert) {
	return [=](Shader& s, IR::Value read) {
		const auto v = s.Emit(signed_convert ? IR::ValueOpcode::ConvertF32S32 : IR::ValueOpcode::ConvertF32U32,
		                      {s.Emit(IR::ValueOpcode::BitFieldUExtract, {read, IR::Value(word * 16u), IR::Value(uint32_t {16})})});
		return s.Emit(IR::ValueOpcode::FPMul32, {IR::Value::F32(1.0f / 65535.0f), v});
	};
}
Unpack HalfWord(uint32_t word, bool negate) {
	return [=](Shader& s, IR::Value read) {
		auto v = s.Emit(IR::ValueOpcode::BitFieldUExtract, {read, IR::Value(word * 16u), IR::Value(uint32_t {16})});
		if (negate) {
			v = s.Emit(IR::ValueOpcode::BitwiseXor32, {v, IR::Value(uint32_t {0x8000})});
		}
		return s.Emit(IR::ValueOpcode::ConvertF32F16,
		              {s.Emit(IR::ValueOpcode::BitCastF16U16, {s.Emit(IR::ValueOpcode::ConvertU16U32, {v})})});
	};
}
// The word used as an integer first (a mask the pass does not see through): rounded fields.
Unpack IntegerWord(uint32_t word, bool sign) {
	return [=](Shader& s, IR::Value read) {
		const auto e = s.Emit(sign ? IR::ValueOpcode::BitFieldSExtract : IR::ValueOpcode::BitFieldUExtract,
		                      {read, IR::Value(word * 16u), IR::Value(uint32_t {16})});
		const auto m = s.Emit(IR::ValueOpcode::BitwiseOr32, {e, IR::Value(uint32_t {0})});
		const auto v = s.Emit(IR::ValueOpcode::ConvertF32S32, {m});
		return s.Emit(IR::ValueOpcode::FPMul32, {IR::Value::F32(1.0f / 32767.0f), v});
	};
}
// The dword shifted arithmetically by hand: the signed high word.
Unpack ShiftedSignedWord() {
	return [](Shader& s, IR::Value read) {
		const auto hi = s.Emit(IR::ValueOpcode::ShiftRightArithmetic32, {read, IR::Value(uint32_t {16})});
		return s.Emit(IR::ValueOpcode::FPMul32,
		              {IR::Value::F32(1.0f / 32767.0f), s.Emit(IR::ValueOpcode::ConvertF32S32, {hi})});
	};
}
// The dword shifted by hand instead of a word extract.
Unpack ShiftedWord() {
	return [](Shader& s, IR::Value read) {
		const auto hi = s.Emit(IR::ValueOpcode::ShiftRightLogical32, {read, IR::Value(uint32_t {16})});
		return s.Emit(IR::ValueOpcode::FPMul32,
		              {IR::Value::F32(1.0f / 65535.0f), s.Emit(IR::ValueOpcode::ConvertF32U32, {hi})});
	};
}

void RunFlow(const Flow& flow, uint32_t trials, uint32_t seed) {
	const float i = flow.i, j = flow.j;
	ShaderPixelInputInfo pixel;
	pixel.input_num                 = flow.inputs;
	pixel.custom_interpolation_mask = flow.custom_mask;
	for (uint32_t input = 0; input < flow.inputs; input++) {
		pixel.interpolator_settings[input] = input + 2u; // exports 2, 3, ...
	}
	Shader reference;
	flow.build(reference);
	Shader unpacked;
	flow.build(unpacked);
	const auto stats = IR::UnpackRawVertexParameters(unpacked.program, pixel);
	Check(IR::LastRawVertexUnpackStats().raw_reads == stats.raw_reads, flow.name + ": the thread's last counts");
	if (flow.expect) {
		flow.expect(stats, flow.name);
	}
	// The fields the program reads, and their slots (as ShaderInfoCollection lays them out).
	std::vector<ShaderParameterField> fields;
	std::vector<uint32_t>             active;
	for (const auto& inst: *unpacked.block) {
		if (inst.GetOpcode() == IR::ValueOpcode::GetParameterField && inst.HasUses()) {
			fields.push_back({inst.Arg(0).U32(), inst.Arg(1).U32()});
		}
	}
	uint32_t   dropped = 0;
	const auto slots   = ShaderPixelFieldSlots(pixel, active, fields, &dropped);
	Check(dropped == 0, flow.name + ": every field has a slot");

	std::mt19937 rng(seed);
	double       worst = 0.0;
	uint32_t     raw_reads_after = 0;
	bool         missing = false;
	for (uint32_t trial = 0; trial < trials; trial++) {
		std::map<uint32_t, std::array<Words, 3>> raw;
		for (uint32_t input = 0; input < flow.inputs; input++) {
			auto& vertices = raw[input];
			for (uint32_t vertex = 0; vertex < 3; vertex++) {
				for (auto& dword: vertices[vertex]) {
					if ((flow.custom_mask & (1u << input)) != 0 || flow.words) {
						dword = flow.word(rng) | (flow.word(rng) << 16u);
					} else {
						dword = std::bit_cast<uint32_t>(static_cast<float>(static_cast<int32_t>(rng() % 2001u) - 1000) / 64.0f);
					}
				}
				// Sometimes the triangle is constant: the result must then be exact.
				if (trial % 7u == 3u && vertex > 0) {
					vertices[vertex] = vertices[0];
				}
			}
		}
		const std::array<double, 3> weight {1.0 - static_cast<double>(i) - static_cast<double>(j), i, j};
		// What the host hands the pixel shader: each slot's exports, interpolated (flat: vertex 0).
		std::map<uint32_t, Words> slot_values;
		for (const auto& slot: slots) {
			auto vertex_slot  = slot;
			vertex_slot.input = ShaderPixelParameterMappedLocation(pixel, slot.input);
			std::array<Words, 3> exported;
			for (uint32_t vertex = 0; vertex < 3; vertex++) {
				exported[vertex] = ExportOf(raw[slot.input][vertex], vertex_slot);
			}
			Words value {};
			for (uint32_t c = 0; c < 4; c++) {
				if (slot.Flat()) {
					value[c] = exported[0][c];
				} else {
					value[c] = std::bit_cast<uint32_t>(static_cast<float>(
					    weight[0] * std::bit_cast<float>(exported[0][c]) + weight[1] * std::bit_cast<float>(exported[1][c]) +
					    weight[2] * std::bit_cast<float>(exported[2][c])));
				}
			}
			slot_values[slot.location] = value;
		}
		Machine before;
		before.raw         = raw;
		before.custom_mask = flow.custom_mask;
		before.weight      = weight;
		const auto want    = before.Eval(reference.sink->Arg(0));
		Machine after;
		after.raw                 = raw;
		after.custom_mask         = flow.custom_mask;
		after.weight              = weight;
		after.interpolated_inputs = true;
		after.field               = [&](uint32_t attr, uint32_t code) {
			for (const auto& slot: slots) {
				for (uint32_t c = 0; c < 4; c++) {
					if (slot.input == attr && slot.codes[c] == code) {
						return std::bit_cast<float>(slot_values[slot.location][c]);
					}
				}
			}
			missing = true;
			return 0.0f;
		};
		const auto got = after.Eval(unpacked.sink->Arg(0));
		raw_reads_after += after.raw_reads;
		const bool constant = trial % 7u == 3u;
		if (flow.exact) {
			Check(want == got, flow.name + ": bit for bit");
			continue;
		}
		for (uint32_t c = 0; c < 4; c++) {
			const double a = std::bit_cast<float>(want[c]);
			const double b = std::bit_cast<float>(got[c]);
			const double d = std::fabs(a - b);
			worst          = std::max(worst, d);
			if (constant && d > 1e-6) {
				Check(false, flow.name + ": a constant triangle is exact");
			}
		}
	}
	Check(!missing, flow.name + ": every field read finds its slot component");
	Check(worst <= flow.tolerance, flow.name + ": the unpacked program matches raw vertices (worst " +
	                                   std::to_string(worst) + ", allowed " + std::to_string(flow.tolerance) + ")");
	if (flow.raw_free) {
		Check(raw_reads_after == 0, flow.name + ": no raw-vertex read is left");
	}
}

void TestUnpackFlow() {
	const auto any_word    = [](std::mt19937& rng) { return rng() & 0xffffu; };
	const auto positive_word = [](std::mt19937& rng) { return rng() & 0x7fffu; };
	const auto no_min_word = [](std::mt19937& rng) {
		const auto w = rng() & 0xffffu;
		return w == 0x8000u ? 0x8001u : w;
	};
	// Halves the test's fixed-point checks can compare: multiples of 1/256 in [-4, 4].
	const auto half_word = [](std::mt19937& rng) {
		return uint32_t {FloatToHalf(static_cast<float>(static_cast<int32_t>(rng() % 2049u) - 1024) / 256.0f)};
	};
	const float points[][2] = {{0.25f, 0.5f}, {0.0f, 0.0f}, {1.0f, 0.0f}, {0.0f, 1.0f}, {0.125f, 0.125f}, {0.7f, 0.2f}};
	std::vector<Flow> flows;
	const auto        add = [&](Flow flow, float i, float j) {
		flow.i = i;
		flow.j = j;
		flows.push_back(std::move(flow));
	};
	const auto        direct = [](uint32_t fields, uint32_t s16, uint32_t u16, uint32_t f16) {
		return [=](const IR::RawVertexUnpackStats& stats, const std::string& name) {
			Check(stats.float_fields == fields && stats.signed_fields == s16 && stats.unsigned_fields == u16 &&
			          stats.half_fields == f16 && stats.rounded_reads == 0 && stats.stale_reads == 0 &&
			          stats.flat_reads == 0,
			      name + ": fields read as floats, nothing rounded");
		};
	};
	for (uint32_t word = 0; word < 2; word++) {
		for (uint32_t chan = 0; chan < 4; chan += 3) {
			const auto tag = " chan " + std::to_string(chan) + " word " + std::to_string(word);
			for (const auto& p: points) {
				const float i = p[0], j = p[1];
				add({"signed" + tag,
				                 [=](Shader& s) { s.Sink(s.Bits(Interpolate(s, SignedWord(word, false), 0, chan, i, j))); },
				                 1, 1, any_word, 4e-6, direct(3, 3, 0, 0)}, i, j);
				// The clamp to -32767 is applied after the interpolation instead of per vertex: at most
				// one step of the field's scale apart, and only with a -32768 vertex.
				add({"signed clamped" + tag,
				                 [=](Shader& s) { s.Sink(s.Bits(Interpolate(s, SignedWord(word, true), 0, chan, i, j))); },
				                 1, 1, any_word, 1.0 / 32767.0 + 4e-6, direct(3, 3, 0, 0)}, i, j);
				add({"signed clamped, no -32768" + tag,
				                 [=](Shader& s) { s.Sink(s.Bits(Interpolate(s, SignedWord(word, true), 0, chan, i, j))); },
				                 1, 1, no_min_word, 4e-6, direct(3, 3, 0, 0)}, i, j);
				add({"unsigned" + tag,
				                 [=](Shader& s) { s.Sink(s.Bits(Interpolate(s, UnsignedWord(word, false), 0, chan, i, j))); },
				                 1, 1, any_word, 4e-6, direct(3, 0, 3, 0)}, i, j);
				add({"unsigned, signed convert" + tag,
				                 [=](Shader& s) { s.Sink(s.Bits(Interpolate(s, UnsignedWord(word, true), 0, chan, i, j))); },
				                 1, 1, any_word, 4e-6, direct(3, 0, 3, 0)}, i, j);
				add({"half" + tag,
				                 [=](Shader& s) { s.Sink(s.Bits(Interpolate(s, HalfWord(word, false), 0, chan, i, j))); },
				                 1, 1, half_word, 8e-6, direct(3, 0, 0, 3)}, i, j);
				add({"negated half" + tag,
				                 [=](Shader& s) { s.Sink(s.Bits(Interpolate(s, HalfWord(word, true), 0, chan, i, j))); },
				                 1, 1, half_word, 8e-6, direct(3, 0, 0, 3)}, i, j);
				// Integer uses: the field rounded to the word's grid (half a step of 1/32767).
				for (const bool sign: {true, false}) {
					add({std::string("integer use, ") + (sign ? "signed" : "unsigned") + tag,
					                 [=](Shader& s) { s.Sink(s.Bits(Interpolate(s, IntegerWord(word, sign), 0, chan, i, j))); },
					                 1, 1, any_word, 0.5 / 32767.0 + 3e-6,
					                 [](const IR::RawVertexUnpackStats& stats, const std::string& name) {
						                 Check(stats.float_fields == 0 && stats.rounded_reads == 3,
						                       name + ": three reads rebuilt from rounded fields");
					                 }}, i, j);
				}
			}
		}
	}
	for (const auto& p: points) {
		const float i = p[0], j = p[1];
		// The shape of the game's shaders: two channels, both words, signed and unsigned, mixed
		// with a half channel of another input and an ordinary input.
		add({"material",
		                 [=](Shader& s) {
			                 s.Sink(s.Bits(Interpolate(s, UnsignedWord(0, false), 0, 0, i, j)),
			                        s.Bits(Interpolate(s, UnsignedWord(1, false), 0, 0, i, j)),
			                        s.Bits(Interpolate(s, SignedWord(1, true), 0, 2, i, j)),
			                        s.Bits(Interpolate(s, SignedWord(0, true), 0, 3, i, j)));
		                 },
		                 1, 1, no_min_word, 4e-6, direct(12, 6, 6, 0)}, i, j);
		// One word read both signed and unsigned: two fields.
		add({"both signednesses of one word",
		                 [=](Shader& s) {
			                 s.Sink(s.Bits(Interpolate(s, UnsignedWord(1, false), 0, 1, i, j)),
			                        s.Bits(Interpolate(s, SignedWord(1, false), 0, 1, i, j)));
		                 },
		                 1, 1, any_word, 4e-6, direct(6, 3, 3, 0)}, i, j);
		// The dword used without a word extract: both words rebuilt from rounded fields (signed
		// by default: words of one sign here, so the high word shifted down is the same number).
		add({"whole dword",
		     [=](Shader& s) { s.Sink(s.Bits(Interpolate(s, ShiftedWord(), 0, 1, i, j))); },
		     1, 1, positive_word, 0.5 / 65535.0 + 3e-6,
		     [](const IR::RawVertexUnpackStats& stats, const std::string& name) {
			     Check(stats.float_fields == 0 && stats.rounded_reads == 3 && stats.stale_reads == 0,
			           name + ": rebuilt dwords");
		     }}, i, j);
		// The same with the signed high word, any words: a word of unknown use is taken signed.
		add({"whole dword, signed",
		     [=](Shader& s) { s.Sink(s.Bits(Interpolate(s, ShiftedSignedWord(), 0, 1, i, j))); },
		     1, 1, any_word, 0.5 / 32767.0 + 3e-6,
		     [](const IR::RawVertexUnpackStats& stats, const std::string& name) {
			     Check(stats.float_fields == 0 && stats.rounded_reads == 3, name + ": rebuilt dwords");
		     }}, i, j);
		// An ordinary (not custom) parameter read by V_INTERP_MOV: P0 + i*P10 + j*P20 with
		// differences: the host interpolates, the differences are zero.
		add({"ordinary parameter",
		                 [=](Shader& s) {
			                 const auto p0  = s.Emit(IR::ValueOpcode::BitCastF32U32, {s.Read(0, 1, 2)});
			                 const auto p10 = s.Emit(IR::ValueOpcode::BitCastF32U32, {s.Read(0, 1, 0)});
			                 const auto p20 = s.Emit(IR::ValueOpcode::BitCastF32U32, {s.Read(0, 1, 1)});
			                 const auto a   = s.Emit(IR::ValueOpcode::FPAdd32,
			                                         {p0, s.Emit(IR::ValueOpcode::FPMul32, {p10, IR::Value::F32(i)})});
			                 s.Sink(s.Bits(s.Emit(IR::ValueOpcode::FPAdd32,
			                                      {a, s.Emit(IR::ValueOpcode::FPMul32, {p20, IR::Value::F32(j)})})));
		                 },
		                 1, 0, any_word, 2e-5,
		                 [](const IR::RawVertexUnpackStats& stats, const std::string& name) {
			                 Check(stats.zero_deltas == 2 && stats.raw_reads == 0 && stats.float_fields == 0,
			                       name + ": two zero differences");
		                 },
		                 false}, i, j);
		// Packed halves interpolated by the hardware (V_INTERP_P2_F16).
		add({"packed half attribute",
		                 [=](Shader& s) {
			                 s.Sink(s.Bits(s.Emit(IR::ValueOpcode::GetAttributeF16,
			                                      {IR::Value(uint32_t {0}), IR::Value(uint32_t {2}), IR::Value(uint32_t {1})})),
			                        s.Bits(s.Emit(IR::ValueOpcode::GetAttributeF16,
			                                      {IR::Value(uint32_t {0}), IR::Value(uint32_t {2}), IR::Value(uint32_t {0})})));
		                 },
		                 1, 0, half_word, 8e-6,
		                 [](const IR::RawVertexUnpackStats& stats, const std::string& name) {
			                 Check(stats.half_reads == 2, name + ": two halves");
		                 }}, i, j);
		flows.back().words = true;
	}
	// A custom parameter only read at P0 (an id): the provoking vertex's dword, bit for bit.
	add({"flat custom parameter",
	                 [](Shader& s) {
		                 const auto id = s.Read(0, 3, 2);
		                 s.Sink(id, s.Emit(IR::ValueOpcode::BitwiseAnd32, {id, IR::Value(uint32_t {0xff00ff})}));
	                 },
	                 1, 1, any_word, 0.0,
	                 [](const IR::RawVertexUnpackStats& stats, const std::string& name) {
		                 Check(stats.flat_reads == 1 && stats.raw_reads == 1 && stats.float_fields == 0,
		                       name + ": one flat read");
	                 }}, 0.0f, 0.0f);
	flows.back().exact = true;
	uint32_t seed = 1000;
	for (const auto& flow: flows) {
		RunFlow(flow, 160, seed++);
	}
}

uint32_t CountOpcode(const Shader& shader, IR::ValueOpcode opcode, bool used_only = true) {
	uint32_t count = 0;
	for (const auto& inst: *shader.block) {
		count += inst.GetOpcode() == opcode && (!used_only || inst.HasUses()) ? 1u : 0u;
	}
	return count;
}

void TestUnpackStructure() {
	ShaderPixelInputInfo pixel;
	pixel.input_num                 = 3;
	pixel.custom_interpolation_mask = 1u; // input 0 custom, 1 ordinary, 2 flat
	pixel.interpolator_settings[0]  = 4;
	pixel.interpolator_settings[1]  = 5;
	pixel.interpolator_settings[2]  = 6 | FlatShade;
	{
		// A read kept only as the old value of an overwritten register (the false side of a
		// select) takes the words whose kind is known and no others.
		Shader s;
		const auto p0    = s.Read(0, 0, 2);
		const auto p10   = s.Read(0, 0, 0);
		const auto p20   = s.Read(0, 0, 1);
		const auto f     = SignedWord(1, false);
		const auto value = s.Emit(IR::ValueOpcode::FPAdd32, {f(s, p0), s.Emit(IR::ValueOpcode::FPAdd32, {f(s, p10), f(s, p20)})});
		const auto stale = s.Emit(IR::ValueOpcode::SelectU32, {IR::Value(true), s.Bits(value), p0});
		s.Sink(stale);
		const auto stats = IR::UnpackRawVertexParameters(s.program, pixel);
		Check(stats.raw_reads == 3 && stats.float_fields == 3 && stats.stale_reads == 1 && stats.rounded_reads == 0,
		      "stale read: counted apart");
		std::set<uint32_t> codes;
		for (const auto& inst: *s.block) {
			if (inst.GetOpcode() == IR::ValueOpcode::GetParameterField && inst.HasUses()) {
				codes.insert(inst.Arg(1).U32());
			}
		}
		Check(codes == std::set<uint32_t> {ShaderParameterFieldCode(Kind::S16, 0, 1)},
		      "stale read: only the known word is a field");
		// What the export consumes reads no raw vertex (dead extracts still name the old reads here:
		// the recompiler's dead code elimination follows the pass).
		Machine machine;
		machine.raw[0]      = {};
		machine.custom_mask = 1;
		machine.field       = [](uint32_t, uint32_t) { return 3.0f; };
		const auto out      = machine.Eval(s.sink->Arg(0));
		Check(machine.raw_reads == 0, "stale read: no raw read left");
		Check(std::bit_cast<float>(out[0]) == 9.0f * (1.0f / 32767.0f), "stale read: the live value is the fields' sum");
	}
	{
		// Flat and ordinary parameters keep their P0 read (the emitter reads the input); a flat
		// parameter's packed half stays with the flat input.
		Shader s;
		const auto flat     = s.Read(2, 0, 2);
		const auto ordinary = s.Read(1, 0, 2);
		const auto half     = s.Emit(IR::ValueOpcode::GetAttributeF16,
		                             {IR::Value(uint32_t {2}), IR::Value(uint32_t {1}), IR::Value(uint32_t {0})});
		s.Sink(flat, ordinary, s.Bits(half));
		const auto stats = IR::UnpackRawVertexParameters(s.program, pixel);
		Check(!stats.Any(), "flat and ordinary P0 reads are left alone");
		Check(CountOpcode(s, IR::ValueOpcode::GetInterpolationParameter) == 2 &&
		          CountOpcode(s, IR::ValueOpcode::GetAttributeF16) == 1 &&
		          CountOpcode(s, IR::ValueOpcode::GetParameterField, false) == 0,
		      "nothing rewritten");
	}
	{
		// A vertex program is not touched.
		Shader s;
		s.program.stage = ShaderType::Vertex;
		s.Sink(s.Read(0, 0, 2), s.Read(0, 0, 0));
		Check(!IR::UnpackRawVertexParameters(s.program, pixel).Any() &&
		          CountOpcode(s, IR::ValueOpcode::GetInterpolationParameter) == 2,
		      "only pixel programs");
	}
	{
		// The field codes a program asks for: kind, channel and word of each use.
		Shader s;
		s.Sink(s.Bits(Interpolate(s, UnsignedWord(0, false), 0, 2, 0.5f, 0.25f)),
		       s.Bits(Interpolate(s, HalfWord(1, true), 0, 3, 0.5f, 0.25f)));
		(void)IR::UnpackRawVertexParameters(s.program, pixel);
		std::set<uint32_t> codes;
		for (const auto& inst: *s.block) {
			if (inst.GetOpcode() == IR::ValueOpcode::GetParameterField && inst.HasUses()) {
				codes.insert(inst.Arg(1).U32());
				Check(inst.Arg(0).U32() == 0, "the field names the pixel input");
			}
		}
		Check(codes == std::set<uint32_t> {ShaderParameterFieldCode(Kind::U16, 2, 0), ShaderParameterFieldCode(Kind::F16, 3, 1)},
		      "field codes");
		Check(CountOpcode(s, IR::ValueOpcode::FPNeg32) == 3, "a flipped sign bit is a negation of the field");
	}
	SetPsPerVertexModeForTests(-1);
}

} // namespace

int main() {
	TestMode();
	TestSlots();
	TestInterface();
	TestSpirv();
	const int gs_checks = g_checks;
	TestUnpackCodes();
	TestUnpackSlots();
	TestUnpackSlotsFuzz();
	TestUnpackLink();
	TestUnpackExport();
	TestUnpackFlow();
	TestUnpackStructure();
	std::printf("per_vertex_geometry_tests: unpack %d checks\n", g_checks - gs_checks);
	std::printf("per_vertex_geometry_tests: %d checks, %d failures\n", g_checks, g_failures);
	return g_failures == 0 ? 0 : 1;
}
