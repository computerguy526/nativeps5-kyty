#pragma once

// kyty_pipeline_list scenedb ...: builds, merges, queries and checks the shader level database (levelDb.h).
//
//   scenedb build --out <db> --scenes <levels.tsv> --list <list>... [--sight <log>]... [--title <id>] [--common-runs N]
//       Rows from the scene tags of every list (golden and the per-level run lists exported from the journals) and
//       from [sight] lines of run logs (shader used in scene), then inference: a pipeline whose stage shaders are all
//       used in a scene is used there. Keys, hashes and scene names only.
//   scenedb merge --out <db> <db>...
//   scenedb info <db> [--list <list>]       counts and coverage
//   scenedb query <db> (--scene <name> | --key <hex>)
//   scenedb tiers <db> --list <list> [--save <save blob>] [--earliest N] [--ahead N] [--target <scene>]
//       Tier sizes (distinct pipeline keys, permutations) for the save's slots.
//   scenedb check <db> [--list <list>]      decodes, re-encodes, counts rows the list does not know

#include "graphics/host_gpu/renderer/pipeline/levelDb.h"
#include "graphics/host_gpu/renderer/pipeline/levelTier.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineList.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace SceneDbTool {

using namespace Libs::Graphics;

inline bool Slurp(const std::string& path, std::vector<uint8_t>& out) {
	std::ifstream in(path, std::ios::binary);
	if (!in) return false;
	out.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
	return true;
}

inline bool LoadList(const std::string& path, PipelineList::File& list, std::vector<uint8_t>* raw = nullptr) {
	std::vector<uint8_t> bytes;
	std::string          err;
	if (!Slurp(path, bytes) || !PipelineList::Decode(bytes, list, &err)) {
		std::fprintf(stderr, "cannot read list %s %s\n", path.c_str(), err.c_str());
		return false;
	}
	if (raw) *raw = std::move(bytes);
	return true;
}

inline bool LoadDb(const std::string& path, LevelDb::File& db) {
	std::vector<uint8_t> bytes;
	std::string          err;
	if (!Slurp(path, bytes) || !LevelDb::Decode(bytes, db, &err)) {
		std::fprintf(stderr, "cannot read database %s: %s\n", path.c_str(), err.empty() ? "unreadable" : err.c_str());
		return false;
	}
	return true;
}

inline bool SaveDb(const std::string& path, const LevelDb::File& db) {
	const auto    bytes = LevelDb::Encode(db);
	std::ofstream out(path, std::ios::binary | std::ios::trunc);
	out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
	if (!out) {
		std::fprintf(stderr, "cannot write %s\n", path.c_str());
		return false;
	}
	std::printf("wrote %s: %zu bytes, %zu scenes, %zu pipelines, %zu shaders, %zu sources\n", path.c_str(), bytes.size(), db.scenes.size(),
	            db.pipelines.size(), db.shaders.size(), db.sources.size());
	return true;
}

// Stage shader guest hashes of every distinct key of a list (twins: the union).
inline std::map<uint64_t, std::set<uint64_t>> KeyShaders(const PipelineList::File& list) {
	std::map<uint64_t, std::set<uint64_t>> out;
	for (const auto& p: list.pipelines) {
		auto& set = out[p.key];
		for (const auto& stage: p.stages) {
			if (stage.permutation >= list.permutations.size()) continue;
			const auto shader = list.permutations[stage.permutation].shader;
			if (shader < list.shaders.size()) set.insert(list.shaders[shader].hash);
		}
	}
	return out;
}

inline std::set<uint64_t> KeyPermutations(const PipelineList::File& list, const std::set<uint64_t>& keys) {
	std::set<uint64_t> perms;
	for (const auto& p: list.pipelines) {
		if (!keys.count(p.key)) continue;
		for (const auto& stage: p.stages) perms.insert(stage.permutation);
	}
	return perms;
}

inline void AddList(LevelDb::File& db, const PipelineList::File& list, uint64_t source) {
	db.sources.insert(source);
	if (db.title_id.empty()) db.title_id = list.info.title_id;
	std::set<std::pair<uint64_t, uint16_t>> done;
	std::set<uint16_t>                      scenes_seen;
	for (const auto& p: list.pipelines) {
		for (const auto level: p.levels) {
			if (level >= list.levels.size()) continue;
			const auto scene = LevelDb::AddScene(db, list.levels[level]);
			if (!done.insert({p.key, scene}).second) continue; // the twin layout of the same key
			LevelDb::NotePipeline(db, p.key, list.levels[level], LevelDb::FirstSeen);
			scenes_seen.insert(scene);
		}
	}
	for (const auto s: scenes_seen) db.scenes[s].runs = static_cast<uint16_t>(db.scenes[s].runs + 1);
}

// [sight] lines say "this shader was first drawn at t", with `level=` the run's target scene (not the scene at that time):
// a shader's scenes come from the [level-open] timeline of the same log (same clock): the last non-ui scene opened
// before t, plus the target once it has opened; before any scene opened it belongs to "pre_scene" (boot, title).
//   [sight] level=<target> flip=N t=<s> stage=VS guest=<hex> pipe=<hex>
//   [level-open] <scene> t=<s>
using PreOpen = std::map<uint64_t, std::map<std::string, uint32_t>>; // shader -> target scene -> earliest ms

// Shaders drawn before the run's target scene opened (its load takes up to a minute) are not known to belong to it:
// a shader that shows up before the target opened in at least `common` different targets' runs is a boot / common
// shader (scene "pre_scene"); one that shows up for fewer belongs to its targets (flag PreOpen).
inline size_t ResolvePreOpen(LevelDb::File& db, const PreOpen& pre, size_t common) {
	size_t rows = 0;
	for (const auto& [guest, targets]: pre) {
		if (targets.size() >= common) {
			LevelDb::NoteShader(db, guest, "pre_scene", LevelDb::Used | LevelDb::PreOpen, targets.begin()->second);
			rows++;
			continue;
		}
		for (const auto& [target, ms]: targets) {
			LevelDb::NoteShader(db, guest, target, LevelDb::Used | LevelDb::PreOpen, ms);
			rows++;
		}
	}
	auto& scene = db.scenes[LevelDb::AddScene(db, "pre_scene")];
	if (scene.cls == LevelDb::SceneClass::Unknown) scene.cls = LevelDb::SceneClass::Boot;
	return rows;
}

inline size_t AddSight(LevelDb::File& db, const std::string& text, uint64_t source, PreOpen* pre = nullptr) {
	db.sources.insert(source);
	struct Open {
		double      t;
		std::string name;
	};
	std::vector<Open> opens;
	std::string       target;
	{
		std::istringstream in(text);
		std::string        line;
		while (std::getline(in, line)) {
			if (line.rfind("[level-open] ", 0) == 0) {
				const auto sp = line.find(" t=", 13);
				if (sp == std::string::npos) continue;
				opens.push_back({std::atof(line.c_str() + sp + 3), line.substr(13, sp - 13)});
			} else if (target.empty() && line.rfind("[sight] KYTY_SIGHT_LOG on: level=", 0) == 0) {
				target = line.substr(33);
			}
		}
	}
	std::sort(opens.begin(), opens.end(), [](const Open& a, const Open& b) { return a.t < b.t; });
	const auto open_time = [&](const std::string& name, double& t) {
		for (const auto& o: opens) {
			if (o.name == name) {
				t = o.t;
				return true;
			}
		}
		return false;
	};
	std::istringstream                          in(text);
	std::string                                 line;
	size_t                                      rows = 0;
	std::set<std::pair<uint64_t, std::string>>  seen;
	std::set<std::string>                       scenes;
	while (std::getline(in, line)) {
		if (line.rfind("[sight] level=", 0) != 0) continue;
		const auto tpos = line.find(" t=");
		const auto gpos = line.find(" guest=");
		if (tpos == std::string::npos || gpos == std::string::npos) continue;
		const double   t     = std::atof(line.c_str() + tpos + 3);
		const uint64_t guest = std::strtoull(line.c_str() + gpos + 7, nullptr, 16);
		if (guest == 0) continue;
		std::vector<std::pair<std::string, double>> in_scene; // scene, open time
		std::string last;
		double      last_t = 0;
		for (const auto& o: opens) {
			if (o.t > t) break;
			if (o.name.rfind("ui_", 0) == 0) continue;
			last   = o.name;
			last_t = o.t;
		}
		if (last.empty() && pre != nullptr && !target.empty()) {
			double tt = 0;
			if (!open_time(target, tt) || tt > t) { // the target has not opened yet: decided once every log is in
				auto& ms = (*pre)[guest][target];
				ms       = ms == 0 ? static_cast<uint32_t>(t * 1000.0) : std::min(ms, static_cast<uint32_t>(t * 1000.0));
				scenes.insert(target);
				continue;
			}
		}
		if (last.empty()) in_scene.push_back({"pre_scene", 0});
		else in_scene.push_back({last, last_t});
		double tt = 0;
		if (!target.empty() && target != last && open_time(target, tt) && tt <= t) in_scene.push_back({target, tt});
		for (const auto& [scene, at]: in_scene) {
			if (!seen.insert({guest, scene}).second) continue;
			scenes.insert(scene);
			const double rel = std::max(0.0, t - at);
			LevelDb::NoteShader(db, guest, scene, LevelDb::Used, static_cast<uint32_t>(rel * 1000.0));
			rows++;
		}
	}
	for (const auto& s: scenes) {
		auto& scene = db.scenes[LevelDb::AddScene(db, s)];
		scene.runs  = static_cast<uint16_t>(scene.runs + 1);
		if (s == "pre_scene" && scene.cls == LevelDb::SceneClass::Unknown) scene.cls = LevelDb::SceneClass::Boot;
	}
	return rows;
}

struct Coverage {
	size_t keys = 0, with_scene = 0, with_used_or_inferred = 0, first_seen_only = 0, unattributed = 0;
};
inline Coverage CoverageOf(const LevelDb::File& db, const std::vector<uint64_t>& keys) {
	Coverage c;
	c.keys = keys.size();
	for (const auto key: keys) {
		const auto it = db.pipelines.find(key);
		if (it == db.pipelines.end() || it->second.empty()) {
			c.unattributed++;
			continue;
		}
		c.with_scene++;
		bool used = false;
		for (const auto& r: it->second) used = used || (r.flags & (LevelDb::Used | LevelDb::Inferred));
		(used ? c.with_used_or_inferred : c.first_seen_only)++;
	}
	return c;
}

inline int Build(const std::vector<std::string>& a) {
	std::string                out, scenes_path, title;
	size_t                     common = 3;
	std::vector<std::string>   lists, sights;
	for (size_t i = 0; i < a.size(); i++) {
		const auto& arg = a[i];
		const auto  v   = [&]() { return i + 1 < a.size() ? a[++i] : std::string(); };
		if (arg == "--out") out = v();
		else if (arg == "--scenes") scenes_path = v();
		else if (arg == "--list") lists.push_back(v());
		else if (arg == "--sight") sights.push_back(v());
		else if (arg == "--title") title = v();
		else if (arg == "--common-runs") common = static_cast<size_t>(std::max(1, std::atoi(v().c_str())));
	}
	if (out.empty() || lists.empty()) {
		std::fprintf(stderr, "scenedb build needs --out and at least one --list\n");
		return 2;
	}
	LevelDb::File db;
	db.producer = "kyty_pipeline_list scenedb build";
	db.title_id = title;
	PipelineList::File                  base; // the first list: the inference base (golden)
	std::map<uint64_t, std::set<uint64_t>> key_shaders;
	for (size_t i = 0; i < lists.size(); i++) {
		PipelineList::File   list;
		std::vector<uint8_t> raw;
		if (!LoadList(lists[i], list, &raw)) return 1;
		AddList(db, list, LevelDb::Fnv64(raw.data(), raw.size()));
		for (const auto& [k, s]: KeyShaders(list)) key_shaders[k].insert(s.begin(), s.end());
		if (i == 0) base = std::move(list);
	}
	size_t sight_lines = 0;
	PreOpen pre;
	for (const auto& path: sights) {
		std::vector<uint8_t> raw;
		if (!Slurp(path, raw)) {
			std::fprintf(stderr, "cannot read %s\n", path.c_str());
			return 1;
		}
		sight_lines += AddSight(db, std::string(raw.begin(), raw.end()), LevelDb::Fnv64(raw.data(), raw.size()), &pre);
	}
	sight_lines += ResolvePreOpen(db, pre, common);
	std::vector<uint64_t> all_keys;
	for (const auto& [k, s]: key_shaders) all_keys.push_back(k);
	const auto before = CoverageOf(db, all_keys);
	std::vector<LevelDb::PipelineShaders> ps;
	for (const auto& [k, s]: key_shaders) ps.push_back({k, std::vector<uint64_t>(s.begin(), s.end())});
	const auto inferred = LevelDb::Infer(db, ps);
	if (!scenes_path.empty()) {
		std::vector<uint8_t> raw;
		std::vector<LevelDb::Scene> meta;
		std::string          err;
		if (!Slurp(scenes_path, raw) || !LevelDb::ParseScenesTsv(std::string_view(reinterpret_cast<const char*>(raw.data()), raw.size()), meta, &err)) {
			std::fprintf(stderr, "scenes %s: %s\n", scenes_path.c_str(), err.c_str());
			return 1;
		}
		LevelDb::ApplyScenes(db, meta);
	}
	const auto after = CoverageOf(db, all_keys);
	std::printf("%zu list(s), %zu sight log(s) with %zu shader-in-scene rows, %zu inferred rows\n", lists.size(), sights.size(), sight_lines, inferred);
	std::printf("keys %zu: before inference %zu with a scene (%zu unattributed); after %zu with a scene (%zu seen/inferred as used, %zu tag only), "
	            "%zu unattributed\n",
	            before.keys, before.with_scene, before.unattributed, after.with_scene, after.with_used_or_inferred, after.first_seen_only,
	            after.unattributed);
	return SaveDb(out, db) ? 0 : 1;
}

inline int MergeCmd(const std::vector<std::string>& a) {
	std::string              out;
	std::vector<std::string> in;
	for (size_t i = 0; i < a.size(); i++) {
		if (a[i] == "--out" && i + 1 < a.size()) out = a[++i];
		else in.push_back(a[i]);
	}
	if (out.empty() || in.empty()) {
		std::fprintf(stderr, "scenedb merge needs --out and inputs\n");
		return 2;
	}
	LevelDb::File db;
	for (const auto& p: in) {
		LevelDb::File one;
		if (!LoadDb(p, one)) return 1;
		LevelDb::Merge(db, one);
	}
	return SaveDb(out, db) ? 0 : 1;
}

inline int Info(const std::vector<std::string>& a) {
	std::string path, list_path;
	for (size_t i = 0; i < a.size(); i++) {
		if (a[i] == "--list" && i + 1 < a.size()) list_path = a[++i];
		else path = a[i];
	}
	LevelDb::File db;
	if (path.empty() || !LoadDb(path, db)) return 1;
	std::printf("%s: title %s, producer %s, %zu scenes, %zu sources, %zu pipeline keys, %zu shaders\n", path.c_str(), db.title_id.c_str(),
	            db.producer.c_str(), db.scenes.size(), db.sources.size(), db.pipelines.size(), db.shaders.size());
	std::map<uint16_t, size_t> per;
	for (const auto& [k, rows]: db.pipelines) {
		for (const auto& r: rows) per[r.scene]++;
	}
	for (size_t i = 0; i < db.scenes.size(); i++) {
		const auto& s = db.scenes[i];
		std::printf("  %-34s %-8s order %5s runs %3u keys %zu\n", s.name.c_str(), LevelDb::SceneClassName(s.cls),
		            s.order == LevelDb::UnknownOrder ? "-" : std::to_string(s.order).c_str(), s.runs, per[static_cast<uint16_t>(i)]);
	}
	if (!list_path.empty()) {
		PipelineList::File list;
		if (!LoadList(list_path, list)) return 1;
		std::vector<uint64_t> keys;
		for (const auto& [k, s]: KeyShaders(list)) keys.push_back(k);
		const auto c = CoverageOf(db, keys);
		std::printf("coverage of %s: %zu keys, %zu with a scene (%zu used/inferred, %zu tag only), %zu unattributed\n", list_path.c_str(), c.keys,
		            c.with_scene, c.with_used_or_inferred, c.first_seen_only, c.unattributed);
	}
	return 0;
}

inline int Query(const std::vector<std::string>& a) {
	std::string path, scene, key;
	for (size_t i = 0; i < a.size(); i++) {
		if (a[i] == "--scene" && i + 1 < a.size()) scene = a[++i];
		else if (a[i] == "--key" && i + 1 < a.size()) key = a[++i];
		else path = a[i];
	}
	LevelDb::File db;
	if (path.empty() || !LoadDb(path, db)) return 1;
	if (!scene.empty()) {
		const auto keys = LevelDb::KeysInScene(db, scene);
		std::printf("scene %s: %zu pipeline keys\n", scene.c_str(), keys.size());
		for (const auto k: keys) std::printf("%016llx\n", static_cast<unsigned long long>(k));
		return keys.empty() ? 1 : 0;
	}
	if (!key.empty()) {
		const auto k      = std::strtoull(key.c_str(), nullptr, 16);
		const auto scenes = LevelDb::ScenesOfKey(db, k);
		std::printf("key %016llx: %zu scenes\n", static_cast<unsigned long long>(k), scenes.size());
		for (const auto& s: scenes) std::printf("%s\n", s.c_str());
		return scenes.empty() ? 1 : 0;
	}
	std::fprintf(stderr, "scenedb query needs --scene or --key\n");
	return 2;
}

inline int Tiers(const std::vector<std::string>& a) {
	std::string path, list_path, save_path, target;
	LevelTier::Config config;
	bool              all_rows = false, weak_pre = false;
	for (size_t i = 0; i < a.size(); i++) {
		const auto v = [&]() { return i + 1 < a.size() ? a[++i] : std::string(); };
		if (a[i] == "--list") list_path = v();
		else if (a[i] == "--save") save_path = v();
		else if (a[i] == "--target") target = v();
		else if (a[i] == "--all-rows") all_rows = true;
		else if (a[i] == "--weak-pre-scene") weak_pre = true;
		else if (a[i] == "--progression") config.progression = true;
		else if (a[i] == "--earliest") config.earliest = static_cast<unsigned>(std::atoi(v().c_str()));
		else if (a[i] == "--ahead") config.ahead = static_cast<unsigned>(std::atoi(v().c_str()));
		else path = a[i];
	}
	LevelDb::File      db;
	PipelineList::File list;
	if (path.empty() || list_path.empty() || !LoadDb(path, db) || !LoadList(list_path, list)) return 1;
	std::vector<LevelTier::SlotProgress> slots;
	if (!save_path.empty()) {
		std::vector<uint8_t> raw;
		if (!Slurp(save_path, raw)) {
			std::fprintf(stderr, "cannot read %s\n", save_path.c_str());
			return 1;
		}
		slots = LevelTier::ParseSaveSlots(std::string_view(reinterpret_cast<const char*>(raw.data()), raw.size()));
	}
	for (const auto& s: slots) {
		std::printf("slot %d: last '%s', %s\n", s.slot, s.last_entered.c_str(), LevelTier::SlotInProgress(s) ? "in progress" : "new");
	}
	const auto plan = LevelTier::PlanTiers(db, slots, config, target);
	std::vector<uint64_t> keys;
	for (const auto& [k, s]: KeyShaders(list)) keys.push_back(k);
	const auto pp = LevelTier::PlanPipelines(db, plan, keys, !all_rows, weak_pre);
	std::set<uint64_t> t1, all;
	for (const auto& e: pp.entries) {
		all.insert(e.key);
		if (e.tier == 1) t1.insert(e.key);
	}
	const auto p1 = KeyPermutations(list, t1).size(), pa = KeyPermutations(list, all).size();
	std::printf("tier 1: %zu scenes, %zu of %zu pipeline keys (%.1f %%), %zu of %zu permutations (%.1f %%), %zu unattributed kept\n", plan.tier1.size(),
	            t1.size(), all.size(), 100.0 * static_cast<double>(t1.size()) / static_cast<double>(std::max<size_t>(1, all.size())), p1, pa,
	            100.0 * static_cast<double>(p1) / static_cast<double>(std::max<size_t>(1, pa)), pp.unattributed);
	std::printf("tier 1 scenes:");
	for (const auto& s: plan.tier1) std::printf(" %s", s.c_str());
	std::printf("\n");
	return 0;
}

inline int Check(const std::vector<std::string>& a) {
	std::string path, list_path;
	for (size_t i = 0; i < a.size(); i++) {
		if (a[i] == "--list" && i + 1 < a.size()) list_path = a[++i];
		else path = a[i];
	}
	std::vector<uint8_t> raw;
	LevelDb::File        db;
	std::string          err;
	if (path.empty() || !Slurp(path, raw) || !LevelDb::Decode(raw, db, &err)) {
		std::printf("check: FAILED %s\n", err.c_str());
		return 1;
	}
	if (LevelDb::Encode(db) != raw) {
		std::printf("check: FAILED re-encoding differs\n");
		return 1;
	}
	size_t unknown = 0;
	if (!list_path.empty()) {
		PipelineList::File list;
		if (!LoadList(list_path, list)) return 1;
		std::set<uint64_t> keys;
		for (const auto& [k, s]: KeyShaders(list)) keys.insert(k);
		for (const auto& [k, rows]: db.pipelines) unknown += keys.count(k) ? 0 : 1;
	}
	std::printf("check: ok, %zu pipeline keys%s\n", db.pipelines.size(),
	            list_path.empty() ? "" : (", " + std::to_string(unknown) + " not in the list (ignored at load)").c_str());
	return 0;
}

inline int Main(const std::vector<std::string>& args) {
	if (args.empty()) {
		std::fprintf(stderr, "usage: kyty_pipeline_list scenedb build|merge|info|query|tiers|check ...\n");
		return 2;
	}
	const std::vector<std::string> rest(args.begin() + 1, args.end());
	if (args[0] == "build") return Build(rest);
	if (args[0] == "merge") return MergeCmd(rest);
	if (args[0] == "info") return Info(rest);
	if (args[0] == "query") return Query(rest);
	if (args[0] == "tiers") return Tiers(rest);
	if (args[0] == "check") return Check(rest);
	std::fprintf(stderr, "unknown scenedb command %s\n", args[0].c_str());
	return 2;
}

} // namespace SceneDbTool
