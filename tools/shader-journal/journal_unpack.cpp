// nps5 kart2: offline check of KYTY_PS_PER_VERTEX=unpack on a shader journal (the corpus of a
// title's shaders, _PipelineCache/<title>.shaders.journal). No GPU, no title run.
//   journal_unpack <journal copy> [<out dir>]
// Compiles every journaled pixel shader with raw-vertex reads in unpack mode (first permutation of
// each source), prints what the pass did (fields read as floats, by signedness; reads rebuilt from
// rounded fields; flat reads; zero differences; halves) and the field slots, validates the SPIR-V,
// then compiles every ordinary vertex shader with the field aliases of each distinct slot layout
// its exports can serve and validates those too. With an out dir the modules are written there
// (ps_<hash>.spv, vs_<hash>_<n>.spv).
// Work on a COPY of the journal: the journal class owns the file it opens.
#include "graphics/host_gpu/renderer/pipeline/shaderPrecompile.h"
#include "graphics/shader/recompiler/CodegenOptions.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/ir/ProgramCodec.h"
#include "graphics/shader/recompiler/ir/passes/RawVertexUnpack.h"
#include "graphics/shader/shader.h"

#include "spirv-tools/libspirv.hpp"

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

using namespace Libs::Graphics;
namespace IR = Libs::Graphics::ShaderRecompiler::IR;

namespace {

bool Validate(const std::vector<uint32_t>& spirv, std::string& error) {
	spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_3);
	tools.SetMessageConsumer([&](spv_message_level_t, const char*, const spv_position_t&, const char* message) {
		error += message;
		error += '\n';
	});
	return tools.Validate(spirv);
}

void WriteModule(const char* dir, const std::string& name, const std::vector<uint32_t>& spirv) {
	if (dir == nullptr) return;
	const auto path = std::string(dir) + "/" + name;
	if (std::FILE* f = std::fopen(path.c_str(), "wb"); f != nullptr) {
		std::fwrite(spirv.data(), 4, spirv.size(), f);
		std::fclose(f);
	}
}

const char* KindName(ShaderParameterFieldKind kind) {
	switch (kind) {
		case ShaderParameterFieldKind::S16: return "s16";
		case ShaderParameterFieldKind::U16: return "u16";
		case ShaderParameterFieldKind::F16: return "f16";
		case ShaderParameterFieldKind::Raw: return "raw";
		default: return "-";
	}
}

struct Layout {
	uint32_t                 mask = 0;
	std::array<uint32_t, 32> source {};
	bool operator<(const Layout& other) const {
		return mask != other.mask ? mask < other.mask : source < other.source;
	}
};

} // namespace

int main(int argc, char** argv) {
	if (argc < 2) {
		std::fprintf(stderr, "usage: journal_unpack <journal copy> [<out dir>]\n");
		return 2;
	}
	const char* out_dir = argc >= 3 ? argv[2] : nullptr;
	ShaderRecompiler::SetPsPerVertexModeForTests(2);

	ShaderJournal::Settings settings;
	settings.path              = argv[1];
	settings.background_writer = false;
	if (!ShaderJournal::ReadIdentity(settings.path, settings.identity)) {
		std::fprintf(stderr, "not a shader journal\n");
		return 1;
	}
	ShaderJournal journal(settings);
	const auto&   sources = journal.Sources();
	const auto&   entries = journal.Entries();
	std::vector<const ShaderJournal::Entry*> first(sources.size(), nullptr);
	for (const auto& entry: entries) {
		if (entry.source < first.size() && first[entry.source] == nullptr) first[entry.source] = &entry;
	}

	IR::RawVertexUnpackStats total;
	std::set<Layout>         layouts;
	uint32_t total_final_fields = 0, total_final_words = 0;
	uint32_t pixel_count = 0, pixel_unpacked = 0, pixel_invalid = 0, slots_total = 0, max_slots = 0;
	for (size_t index = 0; index < sources.size(); index++) {
		const auto& source = sources[index];
		if (source.kind != ShaderJournal::Kind::Pixel || first[index] == nullptr) continue;
		auto info = std::make_unique<ShaderPixelInputInfo>();
		if (source.input_info.size() != sizeof(ShaderPixelInputInfo)) continue;
		std::memcpy(static_cast<void*>(info.get()), source.input_info.data(), sizeof(ShaderPixelInputInfo));
		info->stage = {};
		ShaderSanitizeStoredInputInfo(*info);
		IR::ResourceSpecialization specialization;
		if (!IR::DecodeSpecializationCompat(first[index]->specialization, specialization)) continue;
		ShaderRecompiler::CompileOptions options;
		options.stage            = ShaderType::Pixel;
		options.shader_hash      = source.hash;
		options.wave_size        = info->wave_size;
		options.dump_ir          = false;
		options.input_info.pixel = info.get();
		std::array<uint32_t, 40> user_data {};
		options.user_data = std::span(user_data).first(source.user_data_count);
		pixel_count++;
		if (const char* dump = std::getenv("JOURNAL_UNPACK_DUMP");
		    dump != nullptr && std::strtoull(dump, nullptr, 16) == source.hash) {
			// The IR before (raw-vertex inputs) and after the pass.
			for (const int mode: {0, 2}) {
				ShaderRecompiler::SetPsPerVertexModeForTests(mode);
				auto dump_options    = options;
				dump_options.dump_ir = true;
				auto dumped = ShaderRecompiler::CompileProgram(
				    ShaderRecompiler::TranslateProgram(source.code, dump_options), dump_options, specialization,
				    first[index]->push_data_cursor);
				std::printf("==== IR of %016" PRIx64 " mode %d\n%s\n", source.hash, mode, dumped.ir_dump.c_str());
			}
		}
		auto translated = ShaderRecompiler::TranslateProgram(source.code, options);
		if (translated.skip_dispatch) continue;
		(void)IR::UnpackRawVertexParameters(translated.program, *info); // clears the thread's counts
		translated  = ShaderRecompiler::TranslateProgram(source.code, options);
		auto result = ShaderRecompiler::CompileProgram(std::move(translated), options, specialization,
		                                               first[index]->push_data_cursor);
		const auto stats = IR::LastRawVertexUnpackStats();
		if (!stats.Any()) continue;
		pixel_unpacked++;
		total.raw_reads += stats.raw_reads;
		total.float_fields += stats.float_fields;
		total.rounded_reads += stats.rounded_reads;
		total.flat_reads += stats.flat_reads;
		total.zero_deltas += stats.zero_deltas;
		total.half_reads += stats.half_reads;
		total.signed_fields += stats.signed_fields;
		total.unsigned_fields += stats.unsigned_fields;
		total.half_fields += stats.half_fields;
		total.stale_reads += stats.stale_reads;

		std::string error;
		const bool  valid = Validate(result.spirv, error);
		if (!valid) pixel_invalid++;
		// What the compiled program reads: fields, and words put back together from a field
		// (rounded, or re-packed as a half) for integer uses.
		uint32_t final_fields = 0, final_words = 0;
		for (const auto* block: result.program.blocks) {
			for (const auto& inst: *block) {
				if (inst.GetOpcode() == IR::ValueOpcode::GetParameterField) final_fields++;
				if ((inst.GetOpcode() == IR::ValueOpcode::FPRoundEven32 ||
				     inst.GetOpcode() == IR::ValueOpcode::ConvertF16F32) &&
				    !inst.Arg(0).IsImmediate() && inst.Arg(0).TryInstruction() != nullptr &&
				    inst.Arg(0).TryInstruction()->GetOpcode() == IR::ValueOpcode::GetParameterField) {
					final_words++;
				}
			}
		}
		total_final_fields += final_fields;
		total_final_words += final_words;
		WriteModule(out_dir, "ps_" + std::to_string(index) + ".spv", result.spirv);
		std::printf("ps %3zu %016" PRIx64 " raw %u float %u (s %u u %u h %u) rounded %u stale %u flat %u zero %u half %u; compiled: field reads %u, words rebuilt %u %s\n",
		            index, source.hash, stats.raw_reads, stats.float_fields, stats.signed_fields,
		            stats.unsigned_fields, stats.half_fields, stats.rounded_reads, stats.stale_reads,
		            stats.flat_reads, stats.zero_deltas,
		            stats.half_reads, final_fields, final_words, valid ? "valid" : "INVALID");
		if (!valid) std::printf("    %s", error.c_str());
		// The vertex side of this program's slots.
		IR::CompiledShaderInfo compiled;
		compiled.stage      = ShaderType::Pixel;
		compiled.info       = result.program.info;
		info->stage.program = &compiled;
		uint32_t slots = 0;
		for (const auto& input: result.program.info.inputs) {
			if (input.kind != IR::StageInputKind::ParameterField) continue;
			slots++;
			const auto slot = ShaderParameterFieldSlot::Unpack(input.location, input.component_count);
			std::printf("    slot loc %2u input %u (export %u):", slot.location, slot.input,
			            ShaderPixelParameterMappedLocation(*info, slot.input));
			for (const auto code: slot.codes) {
				if (code != 0) {
					std::printf(" %s.%u.%u", KindName(ShaderParameterFieldCodeKind(code)),
					            ShaderParameterFieldCodeChannel(code), ShaderParameterFieldCodeWord(code));
				}
			}
			std::printf("\n");
		}
		slots_total += slots;
		max_slots = std::max(max_slots, slots);
		auto vertex = std::make_unique<ShaderVertexInputInfo>();
		ShaderLinkVertexPixelParameters(*vertex, *info);
		Layout layout;
		layout.mask = vertex->param_alias_mask;
		for (uint32_t i = 0; i < 32; i++) {
			if ((layout.mask & (1u << i)) != 0) layout.source[i] = vertex->param_alias_source[i];
		}
		if (layout.mask != 0) layouts.insert(layout);
	}
	std::printf("pixel shaders %u, with raw-vertex reads %u, invalid %u; raw reads %u: float fields %u "
	            "(signed %u, unsigned %u, half %u), rounded reads %u, stale reads %u, flat reads %u; zero differences %u, "
	            "halves %u; compiled programs: %u field reads, %u words rebuilt from a field; "
	            "slots %u (max %u per shader), %zu alias layouts\n",
	            pixel_count, pixel_unpacked, pixel_invalid, total.raw_reads, total.float_fields,
	            total.signed_fields, total.unsigned_fields, total.half_fields, total.rounded_reads,
	            total.stale_reads, total.flat_reads,
	            total.zero_deltas, total.half_reads, total_final_fields, total_final_words, slots_total, max_slots, layouts.size());

	uint32_t vertex_count = 0, vertex_variants = 0, vertex_invalid = 0;
	for (size_t index = 0; index < sources.size(); index++) {
		const auto& source = sources[index];
		if (source.kind != ShaderJournal::Kind::Vertex || first[index] == nullptr ||
		    source.stage != static_cast<uint32_t>(ShaderType::Vertex) || !source.back_code.empty()) {
			continue;
		}
		auto info = std::make_unique<ShaderVertexInputInfo>();
		constexpr auto stored = ShaderStoredInputInfoSize<ShaderVertexInputInfo>();
		if (source.input_info.size() != stored) continue;
		std::memcpy(static_cast<void*>(info.get()), source.input_info.data(), stored);
		info->stage            = {};
		info->param_alias_mask = 0;
		if (info->logical_stage != ShaderType::Vertex) continue;
		IR::ResourceSpecialization specialization;
		if (!IR::DecodeSpecializationCompat(first[index]->specialization, specialization)) continue;
		ShaderRecompiler::CompileOptions options;
		options.stage             = ShaderType::Vertex;
		options.shader_hash       = source.hash;
		options.wave_size         = info->wave_size;
		options.user_data_base    = 8;
		options.dump_ir           = false;
		options.input_info.vertex = info.get();
		std::array<uint32_t, 40> user_data {};
		options.user_data = std::span(user_data).first(source.user_data_count);
		auto plain = ShaderRecompiler::TranslateProgram(source.code, options);
		if (plain.skip_dispatch) continue;
		auto base = ShaderRecompiler::CompileProgram(std::move(plain), options, specialization,
		                                             first[index]->push_data_cursor);
		uint32_t exports = 0;
		for (const auto& output: base.program.info.outputs) {
			if (output.kind == IR::StageOutputKind::Parameter) exports |= 1u << (output.index & 31u);
		}
		vertex_count++;
		uint32_t variant = 0;
		for (const auto& layout: layouts) {
			bool served = true;
			for (uint32_t i = 0; i < 32; i++) {
				if ((layout.mask & (1u << i)) != 0 &&
				    (exports & (1u << (layout.source[i] & ShaderParamAliasSourceMask))) == 0) {
					served = false;
				}
			}
			if (!served) continue;
			info->param_alias_mask   = layout.mask;
			info->param_alias_source = layout.source;
			auto result = ShaderRecompiler::CompileProgram(ShaderRecompiler::TranslateProgram(source.code, options),
			                                               options, specialization, first[index]->push_data_cursor);
			info->param_alias_mask = 0;
			std::string error;
			const bool  valid = Validate(result.spirv, error);
			vertex_variants++;
			if (!valid) {
				vertex_invalid++;
				std::printf("vs %3zu %016" PRIx64 " layout mask %08x INVALID\n    %s", index, source.hash,
				            layout.mask, error.c_str());
			}
			WriteModule(out_dir, "vs_" + std::to_string(index) + "_" + std::to_string(variant++) + ".spv",
			            result.spirv);
		}
	}
	std::printf("vertex shaders %u, field-alias variants compiled %u, invalid %u\n", vertex_count,
	            vertex_variants, vertex_invalid);
	return pixel_invalid != 0 || vertex_invalid != 0 ? 1 : 0;
}
