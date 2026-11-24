#include "graphics/host_gpu/renderer/pipeline/levelSchedule.h"


#include <cstdio>
#include <set>

using namespace Libs::Graphics;
using namespace Libs::Graphics::LevelDb;
using namespace Libs::Graphics::LevelTier;

static int g_failed = 0;
static int g_checks = 0;
#define CHECK(cond)                                                       \
	do {                                                                  \
		g_checks++;                                                       \
		if (!(cond)) {                                                    \
			std::printf("FAILED %s:%d: %s\n", __FILE__, __LINE__, #cond); \
			g_failed++;                                                   \
		}                                                                 \
	} while (0)

// A synthetic game (same shape as LevelTierTests): menu, hub, cutscene, six story levels l1..l6 (save ids s1..s6), a bonus level.
static File Game() {
	File               f;
	std::vector<Scene> meta;
	std::string        err;
	const char*        tsv = "title\tmenu\nhub\thub\t-\thub_id\nintro\tcutscene\n"
	                         "l1\tlevel\t1\ts1\nl2\tlevel\t2\ts2\nl3\tlevel\t3\ts3\nl4\tlevel\t4\ts4\nl5\tlevel\t5\ts5\nl6\tlevel\t6\ts6\n"
	                         "b1\tbonus\t-\tbonus1\n";
	if (!ParseScenesTsv(tsv, meta, &err)) std::printf("bad tsv %s\n", err.c_str());
	ApplyScenes(f, meta);
	NotePipeline(f, 10, "title", FirstSeen);
	NotePipeline(f, 20, "hub", FirstSeen);
	NotePipeline(f, 30, "intro", FirstSeen);
	for (int n = 1; n <= 6; n++) {
		const std::string name = "l" + std::to_string(n);
		NotePipeline(f, 100 + 10 * n, name, FirstSeen, 5000);
		NotePipeline(f, 101 + 10 * n, name, FirstSeen, 100);
	}
	NotePipeline(f, 200, "b1", FirstSeen);
	return f;
}
static std::string Lv(const char* id, bool unlocked, bool entered, bool cleared) {
	auto b = [](bool v) { return v ? "true" : "false"; };
	return std::string("<") + id + ">\n<IsUnlocked>" + b(unlocked) + "</IsUnlocked>\n<IsEverEntered>" + b(entered) +
	       "</IsEverEntered>\n<IsCleared>" + b(cleared) + "</IsCleared>\n</" + id + ">\n";
}
static std::string SaveAt(int entered_up_to, int cleared_up_to) {
	std::string levels = Lv("hub_id", true, true, true);
	for (int n = 1; n <= 6; n++) {
		const std::string id = "s" + std::to_string(n);
		levels += Lv(id.c_str(), n <= entered_up_to + 1, n <= entered_up_to, n <= cleared_up_to);
	}
	levels += Lv("bonus1", false, false, false);
	const std::string last = "s" + std::to_string(entered_up_to);
	return "<User1>\n<Common><PlayingLevelId>worldmap</PlayingLevelId><LastEnteredLevelId>" + last +
	       "</LastEnteredLevelId></Common>\n<Level>\n" + levels + "</Level>\n</User1>\n";
}

// The replay items: key 1 untagged, the keys of Game(), key 111 twice (two snapshot layouts), the undecodable item is simply absent.
static std::vector<LevelSchedule::ItemKey> Items() {
	std::vector<uint64_t> keys = {1, 10, 20, 30, 200};
	for (int n = 1; n <= 6; n++) {
		keys.push_back(100 + 10 * n);
		keys.push_back(101 + 10 * n);
	}
	keys.push_back(111);
	std::vector<LevelSchedule::ItemKey> items;
	for (uint32_t i = 0; i < keys.size(); i++) items.push_back({i * 2 + 1, keys[i]}); // item indices are sparse: gaps = undecodable
	return items;
}
static bool Contains(const std::vector<uint32_t>& v, uint32_t x) { return std::find(v.begin(), v.end(), x) != v.end(); }
static uint32_t IndexOfKey(const std::vector<LevelSchedule::ItemKey>& items, uint64_t key) {
	for (const auto& i: items)
		if (i.key == key) return i.index;
	return 0xFFFFFFFFu;
}

int main() {
	// ---- the gate: only the exact text "tier" ----
	CHECK(LevelSchedule::IsTierMode("tier"));
	CHECK(!LevelSchedule::IsTierMode(nullptr));
	CHECK(!LevelSchedule::IsTierMode(""));
	CHECK(!LevelSchedule::IsTierMode("scene"));
	CHECK(!LevelSchedule::IsTierMode("Tier"));
	CHECK(!LevelSchedule::IsTierMode("tier2"));

	// ---- the database round trip the replay does (Encode -> file bytes -> Decode) feeds the schedule ----
	const auto           game = Game();
	const auto           bytes = Encode(game);
	File                 db;
	std::string          err;
	CHECK(Decode(bytes, db, &err));
	CHECK(!db.pipelines.empty());

	const auto items = Items();
	// ---- no save: tier 1 = untagged + menu/hub/cutscene + l1, l2; the rest behind ----
	{
		const auto s = LevelSchedule::Build(db, {}, Config {}, "", items);
		CHECK(s.tier1.size() + s.jobs.size() == items.size()); // every item exactly once
		std::set<uint32_t> seen(s.tier1.begin(), s.tier1.end());
		for (const auto& j: s.jobs) seen.insert(j.index);
		CHECK(seen.size() == items.size());
		CHECK(Contains(s.tier1, IndexOfKey(items, 1)));   // untagged stays in tier 1 (safe direction)
		CHECK(Contains(s.tier1, IndexOfKey(items, 10)));  // title
		CHECK(Contains(s.tier1, IndexOfKey(items, 20)));  // hub
		CHECK(Contains(s.tier1, IndexOfKey(items, 110))); // l1
		CHECK(Contains(s.tier1, IndexOfKey(items, 120))); // l2
		CHECK(!Contains(s.tier1, IndexOfKey(items, 130))); // l3 behind the game
		CHECK(!Contains(s.tier1, IndexOfKey(items, 200))); // bonus last
		CHECK(s.tier1_scenes == s.plan.tier1.size());
		CHECK(s.scene_names.size() == db.scenes.size());
		// the job scene index points into scene_names and names the job's own scene
		bool scene_ok = true;
		for (const auto& j: s.jobs) {
			scene_ok = scene_ok && j.scene < s.scene_names.size() && j.scene >= s.tier1_scenes;
			if (j.id == 130 || j.id == 131) scene_ok = scene_ok && s.scene_names[j.scene] == "l3";
		}
		CHECK(scene_ok);
		// both snapshot layouts of key 111 (two items) land in the same tier
		uint32_t twin = 0;
		for (const auto& i: items) twin += i.key == 111 ? 1 : 0;
		CHECK(twin == 2);
		size_t in_tier1 = 0, in_jobs = 0;
		for (const auto& i: items) {
			if (i.key != 111) continue;
			in_tier1 += Contains(s.tier1, i.index) ? 1 : 0;
			for (const auto& j: s.jobs) in_jobs += j.index == i.index ? 1 : 0;
		}
		CHECK(in_tier1 == 2 && in_jobs == 0);
		// jobs are in prepare order: story order l3..l6 then the bonus level
		CHECK(!s.jobs.empty() && s.scene_names[s.jobs.front().scene] == "l3" && s.jobs.back().id == 200);
		// inside a scene: earliest first-seen first (key 131 at 100 ms before 130 at 5000 ms)
		CHECK(s.jobs.front().id == 131 && s.jobs[1].id == 130);
	}
	// ---- a save at level 4: its level and 2 ahead hold the game, l3 (cleared, behind) goes to the queue ----
	{
		const auto slots = ParseSaveSlots(SaveAt(4, 3));
		CHECK(slots.size() == 1);
		const auto s = LevelSchedule::Build(db, slots, Config {}, "", items);
		CHECK(Contains(s.tier1, IndexOfKey(items, 140)) && Contains(s.tier1, IndexOfKey(items, 150)) && Contains(s.tier1, IndexOfKey(items, 160)));
		CHECK(!Contains(s.tier1, IndexOfKey(items, 130)));
		CHECK(s.tier1.size() + s.jobs.size() == items.size());
	}
	// ---- the schedule drives the background queue: budget, drain, promotion by scene key ----
	{
		auto s = LevelSchedule::Build(db, {}, Config {}, "", items);
		const auto njobs = s.jobs.size();
		BgQueue::Queue q;
		q.Reset(s.jobs, 2, 1);
		std::unordered_set<uint64_t> l5_keys = {150, 151};
		CHECK(q.PromoteIds(l5_keys, 0) == 2);
		BgQueue::Job first;
		CHECK(q.Acquire(first) == BgQueue::Result::Job);
		CHECK(first.id == 150 || first.id == 151); // the promoted scene's pipelines go first
		q.Complete(first);
		size_t got = 1;
		BgQueue::Job j;
		while (true) {
			const auto r = q.Acquire(j);
			if (r == BgQueue::Result::Done) break;
			if (r == BgQueue::Result::Wait) { CHECK(false); break; }
			q.Complete(j);
			got++;
		}
		CHECK(got == njobs);
	}
	// ---- an empty item list is a valid (empty) schedule ----
	{
		const auto s = LevelSchedule::Build(db, {}, Config {}, "", {});
		CHECK(s.tier1.empty() && s.jobs.empty());
	}
	std::printf("LevelSchedule: %d checks, %d failed\n", g_checks, g_failed);
	return g_failed == 0 ? 0 : 1;
}
