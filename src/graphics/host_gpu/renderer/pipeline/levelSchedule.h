#pragma once

// nps5 levelexec: the glue between the level database / save tiers (levelDb.h, levelTier.h) and the pipeline journal
// replay (pipelineCache.cpp JournalReplay::BuildTierOrder), header only and free of the GPU so the wiring runs in
// tests/LevelScheduleTests.cpp: gate parsing, the database + save text -> tier 1 hold set + background jobs.

#include "graphics/host_gpu/renderer/pipeline/bgQueue.h"
#include "graphics/host_gpu/renderer/pipeline/levelDb.h"
#include "graphics/host_gpu/renderer/pipeline/levelTier.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace Libs::Graphics::LevelSchedule {

// KYTY_SHADER_PREPARE value: only the exact text "tier" selects the tier replay (unset / anything else = off).
inline bool IsTierMode(const char* value) {
	return value != nullptr && std::strcmp(value, "tier") == 0;
}

// One replay item the schedule orders: the caller's index and its pipeline key.
struct ItemKey {
	uint32_t index = 0;
	uint64_t key   = 0;
};

struct Schedule {
	std::vector<uint32_t>      tier1;        // item indices that hold the game, in prepare order
	std::vector<BgQueue::Job>  jobs;         // the rest, in prepare order (background queue)
	std::vector<std::string>   scene_names;  // BgQueue job scene index -> scene name (tier 1 scenes, then tier 2)
	size_t                     tier1_scenes = 0;
	size_t                     unattributed = 0;
	LevelTier::Plan            plan;
};

// `items` = the decodable replay items (several items may share a key). Items of one key keep their relative order.
// all_rows=false (default): a key with a first-seen or used row is placed by those rows only (KYTY_SHADER_PREPARE_ALL_ROWS=0).
inline Schedule Build(const LevelDb::File& db, const std::vector<LevelTier::SlotProgress>& slots, const LevelTier::Config& config,
                      std::string_view boot_scene, const std::vector<ItemKey>& items, bool strict_first = true) {
	Schedule out;
	out.plan = LevelTier::PlanTiers(db, slots, config, boot_scene);
	std::vector<uint64_t>        keys;
	std::unordered_set<uint64_t> seen;
	for (const auto& item: items) {
		if (seen.insert(item.key).second) keys.push_back(item.key);
	}
	const auto                             pp = LevelTier::PlanPipelines(db, out.plan, keys, strict_first);
	std::unordered_map<uint64_t, uint32_t> position;
	std::vector<uint32_t>                  scene_of(pp.entries.size(), 0);
	for (uint32_t i = 0; i < pp.entries.size(); i++) {
		position[pp.entries[i].key] = i;
		scene_of[i]                 = pp.entries[i].scene;
	}
	std::vector<ItemKey> sorted = items;
	std::stable_sort(sorted.begin(), sorted.end(), [&](const ItemKey& a, const ItemKey& b) { return position[a.key] < position[b.key]; });
	out.scene_names = out.plan.tier1;
	out.scene_names.insert(out.scene_names.end(), out.plan.tier2.begin(), out.plan.tier2.end());
	out.tier1_scenes = out.plan.tier1.size();
	out.unattributed = pp.unattributed;
	for (const auto& item: sorted) {
		const auto pos = position[item.key];
		if (pp.entries[pos].tier == 1) {
			out.tier1.push_back(item.index);
		} else {
			out.jobs.push_back({item.key, item.index, scene_of[pos]});
		}
	}
	return out;
}

} // namespace Libs::Graphics::LevelSchedule
