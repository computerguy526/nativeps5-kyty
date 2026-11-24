#pragma once

// nps5 levelprep: save-slot aware prepare tiers, header only (tests/LevelTierTests.cpp).
//
// Tier 1 (before the first frame): the scenes every start needs (boot, menus, hub, cutscenes), the earliest levels
// (default 2), and for each of the player's save slots (up to 3) the level it stands in and the next levels ahead
// (default 2) plus every level it has unlocked and not cleared. Tier 2 (behind the game): everything else, ordered by
// how soon a slot could reach it. Pipelines no scene claims stay in tier 1 (the safe direction).
//
// Save metadata: the title's save is a text blob with one <UserN> block per slot (N = 1..3); each holds
// <PlayingLevelId>, <LastEnteredLevelId> and a <Level> list of <id> entries with IsUnlocked / IsEverEntered /
// IsCleared flags (levels in game order). Only those flags are read; nothing else of the save is kept.

#include "graphics/host_gpu/renderer/pipeline/levelDb.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace Libs::Graphics::LevelTier {

using LevelDb::File;
using LevelDb::SceneClass;

struct LevelState {
	std::string id;
	bool        unlocked = false, entered = false, cleared = false;
};

struct SlotProgress {
	int                     slot = 0; // 1..3
	std::string             playing, last_entered;
	std::vector<LevelState> levels;
};

namespace detail {
inline std::string_view Between(std::string_view text, std::string_view open, std::string_view close) {
	const auto a = text.find(open);
	if (a == std::string_view::npos) return {};
	const auto from = a + open.size();
	const auto b    = text.find(close, from);
	if (b == std::string_view::npos) return {};
	return text.substr(from, b - from);
}
inline bool SafeId(std::string_view s) {
	if (s.empty() || s.size() > 96) return false;
	for (const char c: s) {
		if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_')) return false;
	}
	return true;
}
} // namespace detail

// Reads the slots of a save blob. Slots without a <Level> list (or with no level ever entered and no unlocked level
// other than the hub) are still reported; callers decide what an empty slot means. Garbage, truncation and unknown
// tags give fewer slots, never a crash.
inline std::vector<SlotProgress> ParseSaveSlots(std::string_view text) {
	std::vector<SlotProgress> out;
	for (int n = 1; n <= 3; n++) {
		const std::string open  = "<User" + std::to_string(n) + ">";
		const std::string close = "</User" + std::to_string(n) + ">";
		auto              block = detail::Between(text, open, close);
		if (block.empty()) {
			// A blob cut before the closing tag: read to the next user block or the end.
			const auto a = text.find(open);
			if (a == std::string_view::npos) continue;
			block = text.substr(a + open.size());
			const auto next = block.find("<User");
			if (next != std::string_view::npos) block = block.substr(0, next);
		}
		SlotProgress slot;
		slot.slot = n;
		if (const auto p = detail::Between(block, "<PlayingLevelId>", "</PlayingLevelId>"); detail::SafeId(p)) slot.playing = std::string(p);
		if (const auto p = detail::Between(block, "<LastEnteredLevelId>", "</LastEnteredLevelId>"); detail::SafeId(p))
			slot.last_entered = std::string(p);
		auto levels = detail::Between(block, "<Level>", "</Level>");
		if (levels.empty()) {
			const auto a = block.find("<Level>");
			if (a != std::string_view::npos) levels = block.substr(a + 7);
		}
		size_t pos = 0;
		while (pos < levels.size()) {
			const auto lt = levels.find('<', pos);
			if (lt == std::string_view::npos) break;
			const auto gt = levels.find('>', lt);
			if (gt == std::string_view::npos) break;
			const auto tag = levels.substr(lt + 1, gt - lt - 1);
			pos            = gt + 1;
			if (tag.empty() || tag.front() == '/' || !detail::SafeId(tag)) continue;
			const std::string close_tag = "</" + std::string(tag) + ">";
			const auto        end       = levels.find(close_tag, pos);
			if (end == std::string_view::npos) break;
			const auto body = levels.substr(pos, end - pos);
			pos             = end + close_tag.size();
			if (body.find("<IsUnlocked>") == std::string_view::npos && body.find("<IsEverEntered>") == std::string_view::npos) continue;
			LevelState s;
			s.id       = std::string(tag);
			s.unlocked = detail::Between(body, "<IsUnlocked>", "</IsUnlocked>") == "true";
			s.entered  = detail::Between(body, "<IsEverEntered>", "</IsEverEntered>") == "true";
			s.cleared  = detail::Between(body, "<IsCleared>", "</IsCleared>") == "true";
			slot.levels.push_back(std::move(s));
		}
		out.push_back(std::move(slot));
	}
	return out;
}

// A slot with nothing but the hub entered is a new game (or an unused slot).
inline bool SlotInProgress(const SlotProgress& s) {
	for (const auto& l: s.levels) {
		if (l.entered && l.id.rfind("hub_", 0) != 0) return true;
	}
	return !s.last_entered.empty();
}

struct Config {
	unsigned earliest = 2; // the first N story levels are always tier 1
	unsigned ahead    = 2; // levels ahead of a slot's furthest level that are tier 1
};

struct Plan {
	std::vector<std::string> tier1;       // scene names, in the order they matter
	std::vector<std::string> tier2;       // the rest, in prepare order
	std::set<std::string>    tier1_set;
	[[nodiscard]] bool       InTier1(std::string_view scene) const { return tier1_set.count(std::string(scene)) != 0; }
};

namespace detail {
inline const LevelDb::Scene* BySaveId(const File& db, std::string_view id) {
	for (const auto& s: db.scenes) {
		if (!s.save_id.empty() && s.save_id == id) return &s;
	}
	return nullptr;
}
} // namespace detail

// The scenes a set of save slots makes tier 1 (besides the always-classes and the earliest levels): per slot the scene
// it last stood in (`last_entered`, else `playing`), the unlocked-not-cleared levels and the `ahead` levels after its
// furthest entered one.
inline Plan PlanTiers(const File& db, const std::vector<SlotProgress>& slots, const Config& config, std::string_view target_scene = {}) {
	Plan                  plan;
	std::vector<uint16_t> level_orders; // sorted orders of every Level scene that has one
	for (const auto& s: db.scenes) {
		if (s.cls == SceneClass::Level && s.order != LevelDb::UnknownOrder) level_orders.push_back(s.order);
	}
	std::sort(level_orders.begin(), level_orders.end());
	const auto by_order = [&](uint16_t order) -> const LevelDb::Scene* {
		for (const auto& s: db.scenes) {
			if (s.cls == SceneClass::Level && s.order == order) return &s;
		}
		return nullptr;
	};
	std::map<std::string, int> rank; // scene -> priority (lower = sooner), tier 1 members only
	const auto                 add1 = [&](const std::string& name, int priority) {
        const auto it = rank.find(name);
        if (it == rank.end()) rank[name] = priority;
        else it->second = std::min(it->second, priority);
	};
	for (const auto& s: db.scenes) {
		if (LevelDb::IsAlwaysClass(s.cls)) add1(s.name, 0);
	}
	if (!target_scene.empty() && LevelDb::FindScene(db, target_scene) >= 0) add1(std::string(target_scene), 1);
	for (unsigned i = 0; i < config.earliest && i < level_orders.size(); i++) add1(by_order(level_orders[i])->name, 2 + static_cast<int>(i));
	// Distance of every level order from the nearest slot frontier (for tier 2 order).
	std::map<uint16_t, int> distance;
	for (const auto& slot: slots) {
		if (!SlotInProgress(slot)) continue;
		uint16_t furthest = 0;
		for (const auto& l: slot.levels) {
			const auto* scene = detail::BySaveId(db, l.id);
			if (!scene) continue;
			if (l.entered && scene->order != LevelDb::UnknownOrder) furthest = std::max(furthest, scene->order);
		}
		const auto current_id = !slot.last_entered.empty() ? slot.last_entered : slot.playing;
		if (const auto* cur = detail::BySaveId(db, current_id)) {
			add1(cur->name, 10);
			if (cur->order != LevelDb::UnknownOrder) furthest = std::max(furthest, cur->order);
		}
		for (const auto& l: slot.levels) {
			const auto* scene = detail::BySaveId(db, l.id);
			if (scene && l.unlocked && !l.cleared && scene->cls != SceneClass::Hub) add1(scene->name, 11);
		}
		// `ahead` next level scenes after the furthest one.
		unsigned taken = 0;
		for (const auto o: level_orders) {
			if (o <= furthest) continue;
			if (taken >= config.ahead) break;
			add1(by_order(o)->name, 12 + static_cast<int>(taken));
			taken++;
		}
		for (const auto o: level_orders) {
			const int d = o > furthest ? o - furthest : furthest - o + 1; // forward first
			auto      it = distance.find(o);
			if (it == distance.end()) distance[o] = d;
			else it->second = std::min(it->second, d);
		}
	}
	// Tier 1 in priority order, ties by story order then name.
	std::vector<std::pair<int, const LevelDb::Scene*>> t1;
	for (const auto& s: db.scenes) {
		const auto it = rank.find(s.name);
		if (it != rank.end()) t1.push_back({it->second, &s});
	}
	const auto story = [](const LevelDb::Scene* s) { return s->order == LevelDb::UnknownOrder ? 0xFFFF : s->order; };
	std::sort(t1.begin(), t1.end(), [&](const auto& a, const auto& b) {
		if (a.first != b.first) return a.first < b.first;
		if (story(a.second) != story(b.second)) return story(a.second) < story(b.second);
		return a.second->name < b.second->name;
	});
	for (const auto& [p, s]: t1) {
		plan.tier1.push_back(s->name);
		plan.tier1_set.insert(s->name);
	}
	// Tier 2: levels by distance from a slot frontier (no slot: story order), bonus / unknown after by name.
	std::vector<const LevelDb::Scene*> rest;
	for (const auto& s: db.scenes) {
		if (!plan.tier1_set.count(s.name)) rest.push_back(&s);
	}
	const auto dist = [&](const LevelDb::Scene* s) {
		if (s->order == LevelDb::UnknownOrder) return 0x7FFF;
		const auto it = distance.find(s->order);
		return it == distance.end() ? 0x4000 + s->order : it->second;
	};
	std::sort(rest.begin(), rest.end(), [&](const auto* a, const auto* b) {
		if (dist(a) != dist(b)) return dist(a) < dist(b);
		if (story(a) != story(b)) return story(a) < story(b);
		return a->name < b->name;
	});
	for (const auto* s: rest) plan.tier2.push_back(s->name);
	return plan;
}

// One pipeline in prepare order.
struct Entry {
	uint64_t key   = 0;
	uint8_t  tier  = 1;
	uint32_t scene = 0xFFFFFFFFu; // index into Plan order (tier1 then tier2) of the best scene that uses it; none: unattributed
};

struct PipelinePlan {
	std::vector<Entry> entries;       // tier 1 first, then tier 2 in prepare order
	size_t             tier1_count = 0;
	size_t             unattributed = 0; // keys no scene claims (kept in tier 1)
};

// Orders `all_keys` (the list's distinct pipeline keys). A key is in the tier of the soonest scene that uses it; keys
// with no row stay in tier 1. Inside a scene: first_ms, then key.
//
// Which rows count: `strict_first` (default) = a key with a first-seen or used row is placed by those rows only, and
// its inferred rows (a superset: state variants of used shader sets) matter only for keys with no other row, the ones
// that were untagged. false = every row counts (a bigger tier 1, fewer live compiles at a first level entry).
inline PipelinePlan PlanPipelines(const File& db, const Plan& plan, const std::vector<uint64_t>& all_keys, bool strict_first = true) {
	PipelinePlan out;
	std::map<std::string, uint32_t> scene_pos;
	uint32_t                        pos = 0;
	for (const auto& s: plan.tier1) scene_pos[s] = pos++;
	for (const auto& s: plan.tier2) scene_pos[s] = pos++;
	struct Item {
		uint32_t scene;
		uint32_t ms;
		uint64_t key;
	};
	std::vector<Item> t1, t2;
	for (const auto key: all_keys) {
		const auto it   = db.pipelines.find(key);
		uint32_t   best = 0xFFFFFFFFu, ms = LevelDb::UnknownMs;
		if (it != db.pipelines.end()) {
			bool direct = false;
			for (const auto& r: it->second) direct = direct || (r.flags & (LevelDb::FirstSeen | LevelDb::Used)) != 0;
			for (const auto& r: it->second) {
				if (r.scene >= db.scenes.size()) continue;
				if (strict_first && direct && (r.flags & (LevelDb::FirstSeen | LevelDb::Used)) == 0) continue;
				const auto sp = scene_pos.find(db.scenes[r.scene].name);
				if (sp == scene_pos.end()) continue;
				if (sp->second < best || (sp->second == best && r.first_ms < ms)) {
					best = sp->second;
					ms   = r.first_ms;
				}
			}
		}
		if (best == 0xFFFFFFFFu) {
			out.unattributed++;
			t1.push_back({0, 0, key});
			continue;
		}
		(best < plan.tier1.size() ? t1 : t2).push_back({best, ms, key});
	}
	const auto less = [](const Item& a, const Item& b) {
		if (a.scene != b.scene) return a.scene < b.scene;
		if (a.ms != b.ms) return a.ms < b.ms;
		return a.key < b.key;
	};
	std::sort(t1.begin(), t1.end(), less);
	std::sort(t2.begin(), t2.end(), less);
	for (const auto& i: t1) out.entries.push_back({i.key, 1, i.scene});
	out.tier1_count = out.entries.size();
	for (const auto& i: t2) out.entries.push_back({i.key, 2, i.scene});
	return out;
}

} // namespace Libs::Graphics::LevelTier
