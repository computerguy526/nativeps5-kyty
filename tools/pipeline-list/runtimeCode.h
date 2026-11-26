// The `<runtime>` shaders of a pipeline list (pipelineList.h) in kyty_pipeline_list: export --runtime-code keeps the
// journaled codes found nowhere in the game files as shaders named by their hashes; check reports how many of
// them a per-player code store (pipelineCodeStore.h) resolves; verify demands that a --shaders journal covers every one,
// so the no-code proof has seen their code.
#ifndef KYTY_TOOLS_PIPELINE_LIST_RUNTIMECODE_H_
#define KYTY_TOOLS_PIPELINE_LIST_RUNTIMECODE_H_

#include "graphics/host_gpu/renderer/pipeline/pipelineList.h"

#include <cstdint>
#include <cstdio>
#include <set>
#include <string>
#include <vector>
#include <utility>

namespace PipelineListTool {

using HashKey = std::pair<uint64_t, uint64_t>;

// Wraps an export's locator: a code `inner` does not find becomes a `<runtime>` shader (offset 0 in the list's
// `<runtime>` Files entry, added to the list on first use). `kept` collects the distinct codes that did.
[[nodiscard]] inline Libs::Graphics::PipelineList::Builder::LocateFn
KeepRuntimeCode(Libs::Graphics::PipelineList::Builder::LocateFn inner, Libs::Graphics::PipelineList::Builder& builder,
                std::set<HashKey>& kept) {
	using namespace Libs::Graphics;
	return [inner = std::move(inner), &builder, &kept](
	           std::span<const uint32_t> code) -> std::optional<PipelineList::Builder::Location> {
		if (auto found = inner(code); found.has_value()) return found;
		if (code.empty()) return std::nullopt;
		const auto hash = PipelineList::CodeHash({reinterpret_cast<const uint8_t*>(code.data()), code.size_bytes()});
		kept.insert({hash.low, hash.high});
		return PipelineList::Builder::Location {builder.AddFile(PipelineList::RuntimeFileName), 0};
	};
}

// "runtime: X resolved, Y pending"; empty when the list has no `<runtime>` shader.
[[nodiscard]] inline std::string RuntimeLine(const Libs::Graphics::PipelineList::ResolveStats& stats) {
	if (stats.runtime_shaders == 0) return {};
	return "runtime: " + std::to_string(stats.runtime_resolved) + " resolved, " + std::to_string(stats.runtime_pending) +
	       " pending";
}

// --require-runtime <percent>: at least that share of the list's runtime shaders resolved (pending is otherwise not an
// error). A list without runtime shaders passes.
[[nodiscard]] inline bool MeetsRuntimeRequirement(const Libs::Graphics::PipelineList::ResolveStats& stats, uint32_t percent) {
	return stats.runtime_resolved * 100u >= static_cast<uint64_t>(percent) * stats.runtime_shaders;
}

// The runtime shaders of `file` (a code of a merged stage counts with both halves) a --shaders journal does not hold
// the code of; `journaled`: the XXH3-128 of every journaled code (and back half).
[[nodiscard]] inline uint64_t RuntimeShadersNotCovered(const Libs::Graphics::PipelineList::File& file,
                                                       const std::set<HashKey>&                  journaled,
                                                       uint64_t*                                 runtime_total = nullptr) {
	using namespace Libs::Graphics;
	const auto runtime = PipelineList::RuntimeFileOf(file);
	uint64_t   total = 0, uncovered = 0;
	if (runtime != PipelineList::CodeFile::Executable) {
		for (const auto& shader: file.shaders) {
			if (shader.file != runtime) continue;
			total++;
			bool covered = journaled.contains({shader.code_hash_low, shader.code_hash_high});
			if (covered && shader.back_code_size != 0) covered = journaled.contains({shader.back_hash_low, shader.back_hash_high});
			uncovered += covered ? 0 : 1;
		}
	}
	if (runtime_total != nullptr) *runtime_total = total;
	return uncovered;
}

// Merged stages (a front and a back half) of which exactly one half was kept as `<runtime>` code, so the halves sit in
// different files and the export drops the shader. v1.4.2 drops them; they are counted so the export can say so.
[[nodiscard]] inline uint64_t CountMixedMergedStages(const std::vector<Libs::Graphics::ShaderJournal::Source>& sources,
                                                     const std::set<HashKey>&                                  runtime_kept) {
	using namespace Libs::Graphics;
	uint64_t mixed = 0;
	for (const auto& source: sources) {
		if (source.code.empty() || source.back_code.empty()) continue;
		const auto front = PipelineList::CodeHash({reinterpret_cast<const uint8_t*>(source.code.data()), source.code.size() * 4});
		const auto back =
		    PipelineList::CodeHash({reinterpret_cast<const uint8_t*>(source.back_code.data()), source.back_code.size() * 4});
		mixed += runtime_kept.contains({front.low, front.high}) != runtime_kept.contains({back.low, back.high}) ? 1 : 0;
	}
	return mixed;
}

} // namespace PipelineListTool

#endif // KYTY_TOOLS_PIPELINE_LIST_RUNTIMECODE_H_
