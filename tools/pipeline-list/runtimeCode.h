// The `<runtime>` shaders of a pipeline list (pipelineList.h) in kyty_pipeline_list: export --runtime-code keeps the
// journaled codes found nowhere in the game files as shaders named by their hashes; check reports how many of
// them a per-player code store (pipelineCodeStore.h) resolves; verify demands that a --shaders journal covers every one,
// so the no-code proof has seen their code.
#ifndef KYTY_TOOLS_PIPELINE_LIST_RUNTIMECODE_H_
#define KYTY_TOOLS_PIPELINE_LIST_RUNTIMECODE_H_

#include "graphics/host_gpu/renderer/pipeline/pipelineList.h"

#include <cstdint>
#include <cstdio>
#include <map>
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

// --require-runtime value: decimal digits only, 0..100. False for anything else (the tool then exits 2: a harvest gate
// must never pass because of a typo).
[[nodiscard]] inline bool ParseRequirePercent(const std::string& text, int& out) {
	if (text.empty() || text.size() > 3) return false;
	int value = 0;
	for (const char c: text) {
		if (c < '0' || c > '9') return false;
		value = value * 10 + (c - '0');
	}
	if (value > 100) return false;
	out = value;
	return true;
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

// ---- drop: programs the build of this game cannot compile (harvest compile probe) -----------------------------------
// The launch phase builds every listed pipeline, and one program the recompiler refuses (e.g. a software ray-tracing kernel
// with writable global addresses) ends the whole process. A probe run with the title's real settings logs such programs by
// hash; `drop` removes them, with their permutations and the pipelines that use them, and writes what it dropped.

struct DropStats {
	uint64_t asked = 0, shaders = 0, permutations = 0, pipelines = 0, pipelines_dropped = 0;
};

// Every `0x` + 16 hex digits of `text` (a probe log, or a dropped.tsv: its first column) is a source hash; the value is the
// first line that named it (cut to 160 characters), kept as the reason.
[[nodiscard]] inline std::map<uint64_t, std::string> ParseDropHashes(const std::string& text) {
	std::map<uint64_t, std::string> out;
	size_t                          line_begin = 0;
	while (line_begin < text.size()) {
		size_t line_end = text.find('\n', line_begin);
		if (line_end == std::string::npos) line_end = text.size();
		const std::string line = text.substr(line_begin, line_end - line_begin);
		for (size_t at = line.find("0x"); at != std::string::npos; at = line.find("0x", at + 2)) {
			if (at + 18 > line.size()) break;
			uint64_t value = 0;
			bool     ok    = true;
			for (size_t i = 0; i < 16 && ok; i++) {
				const char c = line[at + 2 + i];
				const int  d = c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : c >= 'A' && c <= 'F' ? c - 'A' + 10 : -1;
				ok           = d >= 0;
				value        = (value << 4) | static_cast<uint64_t>(d < 0 ? 0 : d);
			}
			if (ok && !out.contains(value)) {
				std::string reason = line.substr(0, 160);
				for (auto& c: reason)
					if (c == '\t' || c == '\r') c = ' ';
				out.emplace(value, std::move(reason));
			}
		}
		line_begin = line_end + 1;
	}
	return out;
}

// `file` without the shaders whose source hash is in `drop`, their permutations and the pipelines using them. `dropped` (optional)
// gets one tsv row per dropped shader: hash, kind, runtime|executable, reason.
[[nodiscard]] inline Libs::Graphics::PipelineList::File DropShaders(const Libs::Graphics::PipelineList::File&   file,
                                                                    const std::map<uint64_t, std::string>&      drop,
                                                                    DropStats& stats, std::string* dropped = nullptr) {
	using namespace Libs::Graphics;
	PipelineList::Resolved resolved;
	const auto             n = file.shaders.size();
	resolved.usable.assign(n, true);
	resolved.pending.assign(n, false);
	resolved.offsets.resize(n);
	resolved.back_offsets.resize(n);
	const auto runtime = PipelineList::RuntimeFileOf(file);
	stats.asked        = drop.size();
	static const char* const Kinds[3] = {"vertex", "pixel", "compute"};
	for (size_t i = 0; i < n; i++) {
		const auto& shader      = file.shaders[i];
		resolved.offsets[i]      = shader.offset;
		resolved.back_offsets[i] = shader.back_offset;
		const auto found         = drop.find(shader.hash);
		if (found == drop.end()) continue;
		resolved.usable[i] = false;
		stats.shaders++;
		if (dropped != nullptr) {
			char hash[32];
			std::snprintf(hash, sizeof(hash), "0x%016llx", static_cast<unsigned long long>(shader.hash));
			const auto kind = static_cast<size_t>(shader.kind);
			*dropped += std::string(hash) + "\t" + (kind < 3 ? Kinds[kind] : "?") + "\t" +
			            (runtime != PipelineList::CodeFile::Executable && shader.file == runtime ? "runtime" : "executable") + "\t" +
			            found->second + "\n";
		}
	}
	PipelineList::RetargetStats rs;
	auto                        out = PipelineList::Retarget(file, resolved, rs);
	stats.permutations              = file.permutations.size() - out.permutations.size();
	stats.pipelines                 = file.pipelines.size();
	stats.pipelines_dropped         = file.pipelines.size() - out.pipelines.size();
	return out;
}

} // namespace PipelineListTool

#endif // KYTY_TOOLS_PIPELINE_LIST_RUNTIMECODE_H_
