#include "common/assert.h"
#include "graphics/shader/recompiler/CodegenOptions.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/shader.h"

#include <algorithm>
#include <cstdlib>
#include <vector>

namespace Libs::Graphics {

namespace {

constexpr uint32_t PsInputOffsetMask = 0x0000001fu;
constexpr uint32_t PsInputFlatShade  = 0x00000400u;

} // namespace

bool ShaderCbShaderMaskExports() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_CB_SHADER_MASK_EXPORTS");
		return value != nullptr && value[0] != '\0' && !(value[0] == '0' && value[1] == '\0');
	}();
	return enabled;
}

bool ShaderObjprimIdEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_OBJPRIM_ID");
		return value != nullptr && value[0] == '1' && value[1] == '\0';
	}();
	return enabled;
}

uint32_t ShaderPixelExportTarget(uint32_t shader_mask, uint32_t export_index) {
	for (uint32_t target = 0; target < 8u; target++) {
		if (((shader_mask >> (target * 4u)) & 0xfu) == 0u) {
			continue;
		}
		if (export_index == 0u) {
			return target;
		}
		export_index--;
	}
	return UINT32_MAX;
}

uint32_t ShaderPixelParameterMappedLocation(const ShaderPixelInputInfo& info, uint32_t input) {
	return input < info.input_num ? info.interpolator_settings[input] & PsInputOffsetMask : input;
}

uint32_t ShaderPixelParameterLocation(const ShaderPixelInputInfo& info,
                                      std::span<const uint32_t> active_inputs, uint32_t input) {
	// Outside rectangle lists, flat and smooth aliases of one vertex export read raw vertices at
	// its location (ShaderInfoCollection). KYTY_PS_PER_VERTEX=0 has no raw-vertex inputs, so
	// differing aliases keep separate locations there as rectangles do.
	if (info.parameter_mode != ShaderPixelParameterMode::Rectangle &&
	    ShaderPixelRawVertexInputs(info)) {
		return ShaderPixelParameterMappedLocation(info, input);
	}
	std::array<bool, 32> used_locations {};
	for (const auto active_input: active_inputs) {
		used_locations[ShaderPixelParameterMappedLocation(info, active_input)] = true;
	}
	std::array<uint32_t, 64> group_locations;
	group_locations.fill(UINT32_MAX);
	for (const auto active_input: active_inputs) {
		const auto mapped = ShaderPixelParameterMappedLocation(info, active_input);
		const auto group  = mapped * 2u + ShaderPixelParameterIsFlat(info, active_input);
		auto&      location = group_locations[group];
		if (location == UINT32_MAX) {
			location = mapped;
			// Rectangle expansion broadcasts flat aliases through separate outputs.
			if (group_locations[group ^ 1u] != UINT32_MAX) {
				location = 0;
				while (location < used_locations.size() && used_locations[location]) {
					location++;
				}
				EXIT_NOT_IMPLEMENTED(location >= used_locations.size());
			}
			used_locations[location] = true;
		}

		if (active_input == input) {
			return location;
		}
	}
	return ShaderPixelParameterMappedLocation(info, input);
}

void ShaderLinkVertexPixelParameters(ShaderVertexInputInfo& vertex,
                                    const ShaderPixelInputInfo& pixel) {
	std::array<uint32_t, 32> active {};
	uint32_t count = 0;
	// Field slots of the pixel program (KYTY_PS_PER_VERTEX=unpack), by location.
	std::array<uint32_t, 32> field_slots {};
	uint32_t                 field_mask = 0;
	if (pixel.stage.program != nullptr) {
		for (const auto& input: pixel.stage.program->info.inputs) {
			if (input.kind == ShaderRecompiler::IR::StageInputKind::Parameter) {
				active[count++] = input.location;
			} else if (input.kind == ShaderRecompiler::IR::StageInputKind::ParameterField) {
				field_mask |= 1u << (input.location & 31u);
				field_slots[input.location & 31u] = input.component_count;
			}
		}
	}
	ShaderLinkVertexPixelParameters(vertex, pixel, std::span(active).first(count));
	if (field_mask == 0) {
		return;
	}
	if (vertex.param_alias_mask == 0) {
		vertex.param_alias_source.fill(0);
	}
	for (uint32_t location = 0; location < 32u; location++) {
		if ((field_mask & (1u << location)) != 0) {
			// The slot names the pixel input; the vertex shader copies from the export it maps to.
			auto slot  = ShaderParameterFieldSlot::Unpack(location, field_slots[location]);
			slot.input = ShaderPixelParameterMappedLocation(pixel, slot.input) & 31u;
			vertex.param_alias_mask |= 1u << location;
			vertex.param_alias_source[location] = slot.Pack();
		}
	}
}

void ShaderLinkVertexPixelParameters(ShaderVertexInputInfo& vertex,
                                     const ShaderPixelInputInfo& pixel,
                                     std::span<const uint32_t> active_inputs) {
	// Runs for every draw. Without a location read both flat and smooth no input moves
	// (ShaderPixelParameterLocation returns the mapped location of each), so no aliases.
	uint32_t flat_locations   = 0;
	uint32_t smooth_locations = 0;
	for (const auto input: active_inputs) {
		const auto bit = 1u << (ShaderPixelParameterMappedLocation(pixel, input) & 31u);
		(ShaderPixelParameterIsFlat(pixel, input) ? flat_locations : smooth_locations) |= bit;
	}
	vertex.param_alias_mask = 0;
	if ((flat_locations & smooth_locations) == 0) {
		return;
	}
	vertex.param_alias_source.fill(0);
	for (const auto input: active_inputs) {
		const auto source = ShaderPixelParameterMappedLocation(pixel, input);
		const auto destination = ShaderPixelParameterLocation(pixel, active_inputs, input);
		if (destination != source) {
			vertex.param_alias_mask |= 1u << destination;
			vertex.param_alias_source[destination] = source;
		}
	}
}

bool ShaderPixelRawVertexUnpack(const ShaderPixelInputInfo& info) {
	return ShaderRecompiler::PsPerVertexUnpack() &&
	       info.parameter_mode != ShaderPixelParameterMode::Rectangle;
}

bool ShaderPixelHasFieldSlots(const ShaderPixelInputInfo& info) {
	if (info.stage.program == nullptr) {
		return false;
	}
	for (const auto& input: info.stage.program->info.inputs) {
		if (input.kind == ShaderRecompiler::IR::StageInputKind::ParameterField) {
			return true;
		}
	}
	return false;
}

std::vector<ShaderParameterFieldSlot>
ShaderPixelFieldSlots(const ShaderPixelInputInfo& info, std::span<const uint32_t> active_inputs,
                      std::span<const ShaderParameterField> fields, uint32_t* dropped) {
	std::array<bool, 32> used {};
	for (const auto input: active_inputs) {
		used[ShaderPixelParameterMappedLocation(info, input) & 31u]          = true;
		used[ShaderPixelParameterLocation(info, active_inputs, input) & 31u] = true;
	}
	std::vector<ShaderParameterField> sorted(fields.begin(), fields.end());
	std::ranges::sort(sorted, [](const ShaderParameterField& a, const ShaderParameterField& b) {
		const bool a_raw = ShaderParameterFieldCodeKind(a.code) == ShaderParameterFieldKind::Raw;
		const bool b_raw = ShaderParameterFieldCodeKind(b.code) == ShaderParameterFieldKind::Raw;
		if (a.input != b.input) return a.input < b.input;
		if (a_raw != b_raw) return b_raw;
		// Channel, word, kind.
		const auto key = [](uint32_t code) {
			return (ShaderParameterFieldCodeChannel(code) << 4u) |
			       (ShaderParameterFieldCodeWord(code) << 3u) |
			       static_cast<uint32_t>(ShaderParameterFieldCodeKind(code));
		};
		return key(a.code) < key(b.code);
	});
	sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());
	std::vector<ShaderParameterFieldSlot> slots;
	int32_t                               next = 31;
	uint32_t                              lost = 0;
	bool                                  full = false;
	for (const auto& field: sorted) {
		if (ShaderParameterFieldCodeKind(field.code) == ShaderParameterFieldKind::None) {
			continue;
		}
		const bool raw  = ShaderParameterFieldCodeKind(field.code) == ShaderParameterFieldKind::Raw;
		auto*      slot = slots.empty() ? nullptr : &slots.back();
		uint32_t   index = 0;
		if (slot != nullptr && slot->input == field.input && slot->Flat() == raw) {
			while (index < 4u && slot->codes[index] != 0) {
				index++;
			}
		}
		if (slot == nullptr || slot->input != field.input || slot->Flat() != raw || index == 4u) {
			while (next >= 0 && used[static_cast<uint32_t>(next)]) {
				next--;
			}
			if (next < 0 || full) {
				full = true;
				lost++;
				continue;
			}
			used[static_cast<uint32_t>(next)] = true;
			slots.push_back({static_cast<uint32_t>(next), field.input, {}});
			slot  = &slots.back();
			index = 0;
		}
		slot->codes[index] = field.code;
	}
	if (dropped != nullptr) {
		*dropped = lost;
	}
	return slots;
}

bool ShaderPixelParameterIsFlat(const ShaderPixelInputInfo& info, uint32_t input) {
	return input < info.input_num && (info.interpolator_settings[input] & PsInputFlatShade) != 0 &&
	       !ShaderPixelParameterIsCustom(info, input);
}

bool ShaderPixelRawVertexInputs(const ShaderPixelInputInfo& info) {
	// KYTY_PS_PER_VERTEX=unpack: no raw vertices either (the reads take unpacked fields), so the
	// interface is laid out as with KYTY_PS_PER_VERTEX=0.
	if (!ShaderRecompiler::GetCodegenOptions().ps_per_vertex ||
	    ShaderRecompiler::PsPerVertexUnpack()) {
		return false;
	}
	return !(ShaderRecompiler::PsPerVertexGeometry() &&
	         info.parameter_mode == ShaderPixelParameterMode::Rectangle);
}

std::vector<ShaderPerVertexSlot>
ShaderPixelPerVertexSlots(const ShaderPixelInputInfo& info, std::span<const uint32_t> active_inputs,
                          std::span<const uint32_t> per_vertex_inputs) {
	std::array<bool, ShaderMaxPerVertexLocations> used {};
	for (const auto input: active_inputs) {
		used[ShaderPixelParameterMappedLocation(info, input) & 31u] = true;
	}
	std::array<bool, 32> raw {};
	for (const auto input: per_vertex_inputs) {
		raw[ShaderPixelParameterMappedLocation(info, input) & 31u] = true;
	}
	std::vector<ShaderPerVertexSlot> slots;
	uint32_t                         next = 0;
	const auto                       take = [&] {
		while (next < used.size() && used[next]) {
			next++;
		}
		EXIT_NOT_IMPLEMENTED(next >= used.size());
		used[next] = true;
		return next;
	};
	for (uint32_t source = 0; source < raw.size(); source++) {
		if (raw[source]) {
			const auto second = take();
			slots.push_back({source, {source, second, take()}});
		}
	}
	return slots;
}

bool ShaderPixelParameterIsCustom(const ShaderPixelInputInfo& info, uint32_t input) {
	return input < 32u && (info.custom_interpolation_mask & (1u << input)) != 0;
}

} // namespace Libs::Graphics
