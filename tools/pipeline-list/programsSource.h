// kyty_pipeline_list export --programs <programs.bin> (kytyPipelineList.cpp): the persistent program cache (KPRC, tools/msl-dump/kprcLoader.h)
// as a source of shaders and permutations, next to the journals.
//
// Why: the shader journal records only what a session compiled itself; programs that reach programs.bin through the shader precompile / compute
// prewarm path (and every program of a session that started from a list) are never journaled. A harvest that reads journals alone misses them
// (v1.4: 70 compute shaders / 78 permutations and 5 mesh shaders of the game were in programs.bin and in no journal or list).
// A program source key holds the guest code and the static state (BuildStageStaticKey of the stage input info), a permutation record holds the
// push-data cursor and the encoded specialization: everything a list needs except the stored input info struct. That struct is rebuilt from the
// static state for compute (every field the key does not hold is a runtime value and stays 0) and taken from a journal source with the same
// identity for the other stages; a graphics source that no journal knows is counted and left out (its translation inputs are not all known).
#ifndef KYTY_TOOLS_PIPELINE_LIST_PROGRAMS_SOURCE_H_
#define KYTY_TOOLS_PIPELINE_LIST_PROGRAMS_SOURCE_H_

#include "../msl-dump/kprcLoader.h"
#include "exportBase.h"
#include "graphics/host_gpu/renderer/pipeline/shaderPrecompile.h"
#include "graphics/shader/shader.h"

#include <cstdint>
#include <cstring>
#include <functional>
#include <map>
#include <span>
#include <string>
#include <tuple>
#include <vector>

namespace PipelineListTool {
using Libs::Graphics::ShaderJournal;

// Inverse of BuildStageStaticKey(const ShaderComputeInputInfo&) (graphics/shader/shaderStaticKey.cpp): the 14 key words give back every field the
// key holds; dispatch_threads_num / workgroup_counts / the runtime pointers are not in it (0). False when the key is not a compute key.
inline bool ComputeInfoFromStaticKey(std::span<const uint32_t> k, Libs::Graphics::ShaderComputeInputInfo& info) {
	if (k.size() != 14) return false;
	info.workgroup_register          = static_cast<int>(k[0]);
	info.wave_size                   = k[1] & 0xffu;
	info.float_mode                  = static_cast<uint8_t>(k[1] >> 8u);
	info.host_subgroup_size          = k[2];
	info.thread_ids_num              = static_cast<int>(k[3]);
	info.lds_size_dwords             = k[4] & 0x7fffffffu;
	info.lds_storage                 = (k[4] & 0x80000000u) != 0;
	info.scratch_size_dwords         = k[5];
	info.dispatch_thread_dimensions  = (k[6] & 1u) != 0;
	info.async_compute               = (k[6] & 2u) != 0;
	info.dispatch_dimensions_indirect = (k[6] & 4u) != 0;
	for (int i = 0; i < 3; i++) {
		info.threads_num[i] = k[7 + 2 * i];
		info.group_id[i]    = k[8 + 2 * i] != 0;
	}
	info.tg_size_en = k[13] != 0;
	return true;
}

// The stored form of a compute input info (journals and lists: the bytes as in memory, runtime pointers cleared, padding 0).
inline std::vector<uint8_t> StoredComputeInfo(std::span<const uint32_t> key) {
	alignas(Libs::Graphics::ShaderComputeInputInfo) uint8_t raw[sizeof(Libs::Graphics::ShaderComputeInputInfo)];
	std::memset(raw, 0, sizeof(raw));
	auto* info = new (raw) Libs::Graphics::ShaderComputeInputInfo;
	if (!ComputeInfoFromStaticKey(key, *info)) return {};
	info->stage = {};
	std::vector<uint8_t> out(Libs::Graphics::ShaderStoredInputInfoSize<Libs::Graphics::ShaderComputeInputInfo>());
	std::memcpy(out.data(), raw, out.size());
	return out;
}

struct ProgramsStats {
	uint64_t sources = 0, permutations = 0;
	uint64_t sources_no_info = 0;          // graphics sources no journal gave an input info for (left out with their permutations)
	uint64_t permutations_unreadable = 0;  // specialization that reads as no known layout
	uint64_t permutations_orphan = 0;      // permutation whose source record is not in the file
	uint64_t sources_with_variant = 0;     // SAROS variant keys (other title family): left out
};

using ShaderIdentityKey = std::tuple<uint32_t, uint64_t, uint32_t, uint32_t, std::vector<uint32_t>, uint32_t, uint32_t, bool>;
inline ShaderIdentityKey IdentityKeyOf(const ShaderJournal::Source& s) {
	return {s.stage, s.hash, s.user_data_count, static_cast<uint32_t>(s.code.size()), s.static_state, s.wave_size, s.user_data_base, s.plain_mip_stats_variant};
}

// `lookup_info(source)` returns the stored input info of a graphics source from a journal ({} = unknown). Appends to sources/entries.
inline ProgramsStats LoadPrograms(const std::vector<uint8_t>& image, const std::function<std::vector<uint8_t>(const ShaderJournal::Source&)>& lookup_info,
                                  std::vector<ShaderJournal::Source>& sources, std::vector<ShaderJournal::Entry>& entries, std::string* error) {
	ProgramsStats st;
	std::vector<Kprc::Perm> perms;
	std::string             err;
	const auto              parsed = Kprc::Parse(image, &perms, &err);
	if (!err.empty() && error != nullptr) *error = err;
	std::map<std::string, uint32_t> index_of_digest; // digest -> index in `sources`, UINT32_MAX = left out
	for (const auto& p: parsed) {
		st.sources++;
		if (p.has_variant && p.variant != 0) {
			st.sources_with_variant++;
			index_of_digest[p.digest] = UINT32_MAX;
			continue;
		}
		ShaderJournal::Source s;
		s.stage                   = p.stage;
		s.hash                    = p.hash;
		s.user_data_count         = p.user_data_count;
		s.wave_size               = p.wave;
		s.user_data_base          = p.user_data_base;
		s.plain_mip_stats_variant = p.plain != 0;
		s.static_state            = p.statics;
		s.code                    = p.code;
		s.back_code               = p.back_code;
		s.code_size               = static_cast<uint32_t>(p.code.size());
		const auto type           = static_cast<Libs::Graphics::ShaderType>(p.stage);
		if (type == Libs::Graphics::ShaderType::Compute) {
			s.kind       = ShaderJournal::Kind::Compute;
			s.input_info = StoredComputeInfo(p.statics);
		} else {
			s.kind       = type == Libs::Graphics::ShaderType::Pixel ? ShaderJournal::Kind::Pixel : ShaderJournal::Kind::Vertex;
			s.input_info = lookup_info ? lookup_info(s) : std::vector<uint8_t> {};
		}
		if (s.input_info.empty() || s.code.empty()) {
			st.sources_no_info++;
			index_of_digest[p.digest] = UINT32_MAX;
			continue;
		}
		index_of_digest[p.digest] = static_cast<uint32_t>(sources.size());
		sources.push_back(std::move(s));
	}
	for (const auto& pm: perms) {
		st.permutations++;
		const auto it = index_of_digest.find(pm.digest);
		if (it == index_of_digest.end()) {
			st.permutations_orphan++;
			continue;
		}
		if (it->second == UINT32_MAX) continue;
		std::vector<uint8_t> spec = pm.spec;
		size_t               at   = 0;
		uint32_t             n    = 0;
		if (!ReadsAsSpecialization(spec, false, &at, &n) && !UpgradeInt21Specialization(spec)) {
			st.permutations_unreadable++;
			continue;
		}
		if (!ReadsAsSpecialization(spec, false, &at, &n)) {
			st.permutations_unreadable++;
			continue;
		}
		entries.push_back({it->second, pm.cursor, std::move(spec)});
	}
	return st;
}

} // namespace PipelineListTool

#endif /* KYTY_TOOLS_PIPELINE_LIST_PROGRAMS_SOURCE_H_ */
