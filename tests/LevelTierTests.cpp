#include "graphics/host_gpu/renderer/pipeline/levelTier.h"

#include <cstdio>

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

// A synthetic game: title menu, hub, an intro cutscene, six story levels l1..l6 (save ids s1..s6), one bonus level.
static File Game() {
	File f;
	std::vector<Scene> meta;
	std::string        err;
	const char* tsv =
	    "title\tmenu\n"
	    "hub\thub\t-\thub_id\n"
	    "intro\tcutscene\n"
	    "l1\tlevel\t1\ts1\nl2\tlevel\t2\ts2\nl3\tlevel\t3\ts3\nl4\tlevel\t4\ts4\nl5\tlevel\t5\ts5\nl6\tlevel\t6\ts6\n"
	    "b1\tbonus\t-\tbonus1\n";
	if (!ParseScenesTsv(tsv, meta, &err)) std::printf("bad tsv %s\n", err.c_str());
	ApplyScenes(f, meta);
	// keys: 1 untagged; 10 title; 20 hub; 30 intro; 100+10*n level n (two keys each); 200 bonus; 300 shared by l5 and l2
	NotePipeline(f, 10, "title", FirstSeen);
	NotePipeline(f, 20, "hub", FirstSeen);
	NotePipeline(f, 30, "intro", FirstSeen);
	for (int n = 1; n <= 6; n++) {
		const std::string name = "l" + std::to_string(n);
		NotePipeline(f, 100 + 10 * n, name, FirstSeen, 5000);
		NotePipeline(f, 101 + 10 * n, name, FirstSeen, 100);
	}
	NotePipeline(f, 200, "b1", FirstSeen);
	NotePipeline(f, 300, "l5", Used);
	NotePipeline(f, 300, "l2", Used);
	return f;
}

static std::string Slot(int n, const char* playing, const char* last, const std::string& levels) {
	return "<User" + std::to_string(n) + ">\n<Common><PlayingLevelId>" + playing + "</PlayingLevelId><LastEnteredLevelId>" + last +
	       "</LastEnteredLevelId></Common>\n<Level>\n" + levels + "</Level>\n</User" + std::to_string(n) + ">\n";
}
static std::string Lv(const char* id, bool unlocked, bool entered, bool cleared) {
	auto b = [](bool v) { return v ? "true" : "false"; };
	return std::string("<") + id + ">\n<IsSilhouette>false</IsSilhouette>\n<IsUnlocked>" + b(unlocked) + "</IsUnlocked>\n<IsEverEntered>" +
	       b(entered) + "</IsEverEntered>\n<IsCleared>" + b(cleared) + "</IsCleared>\n</" + id + ">\n";
}
static std::string Levels(int entered_up_to, int cleared_up_to) {
	std::string out = Lv("hub_id", true, true, true);
	for (int n = 1; n <= 6; n++) {
		const std::string id = "s" + std::to_string(n);
		out += Lv(id.c_str(), n <= entered_up_to + 1, n <= entered_up_to, n <= cleared_up_to);
	}
	out += Lv("bonus1", false, false, false);
	return out;
}
static bool Has(const std::vector<std::string>& v, const char* s) { return std::find(v.begin(), v.end(), s) != v.end(); }

int main() {
	const auto db = Game();
	// ---- save parsing ----
	{
		const auto text = "junk" + Slot(1, "worldmap", "s3", Levels(3, 2)) + Slot(3, "s2", "s2", Levels(2, 1));
		const auto slots = ParseSaveSlots(text);
		CHECK(slots.size() == 2 && slots[0].slot == 1 && slots[1].slot == 3);
		CHECK(slots[0].last_entered == "s3" && slots[0].playing == "worldmap");
		CHECK(slots[0].levels.size() == 8 && slots[0].levels[0].id == "hub_id");
		CHECK(slots[0].levels[3].id == "s3" && slots[0].levels[3].entered && !slots[0].levels[3].cleared);
		CHECK(slots[0].levels[4].id == "s4" && slots[0].levels[4].unlocked && !slots[0].levels[4].entered);
		CHECK(SlotInProgress(slots[0]) && SlotInProgress(slots[1]));
		CHECK(ParseSaveSlots("").empty() && ParseSaveSlots("<User1>").size() == 1);
		CHECK(ParseSaveSlots("\xff\xfe<<<>>><User2><Level><a>").size() == 1);
		// a blob cut in the middle of a level: what was complete is kept
		auto cut = Slot(1, "x", "s2", Levels(2, 1));
		cut.resize(cut.size() - 120);
		const auto c = ParseSaveSlots(cut);
		CHECK(c.size() == 1 && c[0].levels.size() >= 5);
		// bad ids are dropped
		CHECK(ParseSaveSlots("<User1><LastEnteredLevelId>../x</LastEnteredLevelId></User1>")[0].last_entered.empty());
		// a new game: hub only
		const auto fresh = ParseSaveSlots(Slot(2, "worldmap", "", Levels(0, 0)));
		CHECK(fresh.size() == 1 && !SlotInProgress(fresh[0]));
	}
	// ---- 0 slots: always scenes + earliest levels ----
	{
		const auto plan = PlanTiers(db, {}, Config {});
		CHECK(Has(plan.tier1, "title") && Has(plan.tier1, "hub") && Has(plan.tier1, "intro"));
		CHECK(Has(plan.tier1, "l1") && Has(plan.tier1, "l2") && !Has(plan.tier1, "l3"));
		CHECK(plan.tier2.front() == "l3" && plan.tier2.back() == "b1"); // story order, bonus last
		CHECK(plan.tier1.size() + plan.tier2.size() == db.scenes.size());
		const auto nobody = PlanTiers(db, {}, Config {0, 2});
		CHECK(!Has(nobody.tier1, "l1") && Has(nobody.tier1, "hub"));
		const auto more = PlanTiers(db, {}, Config {4, 2});
		CHECK(Has(more.tier1, "l4") && !Has(more.tier1, "l5"));
		CHECK(PlanTiers(db, {}, Config {99, 2}).tier2.size() == 1); // only the bonus level left
	}
	// ---- one slot at the start (a new game): same as no slot ----
	{
		const auto slots = ParseSaveSlots(Slot(1, "worldmap", "", Levels(0, 0)));
		const auto plan  = PlanTiers(db, slots, Config {});
		CHECK(!Has(plan.tier1, "l3"));
		CHECK(plan.tier1.size() == PlanTiers(db, {}, Config {}).tier1.size());
	}
	// ---- one slot mid-game: level 4 entered, 3 cleared ----
	{
		const auto slots = ParseSaveSlots(Slot(1, "worldmap", "s4", Levels(4, 3)));
		const auto plan  = PlanTiers(db, slots, Config {});
		CHECK(Has(plan.tier1, "l4") && Has(plan.tier1, "l5") && Has(plan.tier1, "l6")); // current + 2 ahead (l5 is unlocked, l6 ahead)
		CHECK(Has(plan.tier1, "l1") && Has(plan.tier1, "l2")); // earliest
		CHECK(!Has(plan.tier1, "l3")); // cleared, behind: tier 2
		CHECK(plan.tier2.front() == "l3"); // nearest behind the frontier first, then the rest
		CHECK(plan.tier1.size() + plan.tier2.size() == db.scenes.size());
		const auto zero = PlanTiers(db, slots, Config {2, 0});
		CHECK(Has(zero.tier1, "l4") && Has(zero.tier1, "l5") && !Has(zero.tier1, "l6")); // l5 is unlocked, not ahead
		const auto three = PlanTiers(db, slots, Config {2, 3});
		CHECK(Has(three.tier1, "l6"));
	}
	// ---- one slot at the end of the game ----
	{
		const auto slots = ParseSaveSlots(Slot(1, "worldmap", "s6", Levels(6, 6)));
		const auto plan  = PlanTiers(db, slots, Config {});
		CHECK(Has(plan.tier1, "l6") && Has(plan.tier1, "l1") && Has(plan.tier1, "l2"));
		CHECK(!Has(plan.tier1, "l4")); // all cleared: nothing ahead, nothing unlocked-uncleared
		CHECK(plan.tier2.front() == "l5"); // nearest to the frontier (behind it)
	}
	// ---- three slots: union ----
	{
		const auto text  = Slot(1, "worldmap", "s1", Levels(1, 0)) + Slot(2, "worldmap", "s4", Levels(4, 3)) + Slot(3, "worldmap", "", Levels(0, 0));
		const auto slots = ParseSaveSlots(text);
		CHECK(slots.size() == 3);
		const auto plan = PlanTiers(db, slots, Config {});
		CHECK(Has(plan.tier1, "l1") && Has(plan.tier1, "l2") && Has(plan.tier1, "l3")); // slot 1 ahead = l2,l3
		CHECK(Has(plan.tier1, "l4") && Has(plan.tier1, "l5") && Has(plan.tier1, "l6"));
		CHECK(plan.tier2.size() == 1 && plan.tier2[0] == "b1");
		// the -lvl target joins tier 1 whatever the slots say
		const auto tgt = PlanTiers(db, ParseSaveSlots(Slot(1, "worldmap", "s1", Levels(1, 0))), Config {1, 0}, "l5");
		CHECK(Has(tgt.tier1, "l5") && Has(tgt.tier1, "l1") && !Has(tgt.tier1, "l6"));
		CHECK(PlanTiers(db, {}, Config {}, "no_such_scene").tier1.size() == PlanTiers(db, {}, Config {}).tier1.size());
	}
	// ---- last entered unknown to the database ----
	{
		const auto slots = ParseSaveSlots(Slot(1, "worldmap", "zzz", Levels(2, 1)));
		const auto plan  = PlanTiers(db, slots, Config {});
		CHECK(Has(plan.tier1, "l2") && Has(plan.tier1, "l3") && Has(plan.tier1, "l4"));
	}
	// ---- pipeline plan ----
	{
		const auto slots = ParseSaveSlots(Slot(1, "worldmap", "s4", Levels(4, 3)));
		const auto plan  = PlanTiers(db, slots, Config {});
		const std::vector<uint64_t> all = {1, 10, 20, 30, 110, 111, 120, 121, 130, 131, 140, 141, 150, 151, 160, 161, 200, 300, 999};
		const auto pp = PlanPipelines(db, plan, all);
		CHECK(pp.entries.size() == all.size());
		CHECK(pp.unattributed == 2); // 1 and 999 have no scene: kept in tier 1
		size_t t1 = 0;
		for (const auto& e: pp.entries) t1 += e.tier == 1;
		CHECK(t1 == pp.tier1_count);
		std::map<uint64_t, uint8_t> tier;
		for (const auto& e: pp.entries) tier[e.key] = e.tier;
		CHECK(tier[1] == 1 && tier[999] == 1 && tier[10] == 1 && tier[20] == 1 && tier[30] == 1);
		CHECK(tier[110] == 1 && tier[120] == 1 && tier[140] == 1 && tier[150] == 1 && tier[160] == 1);
		CHECK(tier[130] == 2 && tier[131] == 2 && tier[200] == 2);
		CHECK(tier[300] == 1); // shared by l5 (tier 1) and l2 (tier 1)
		// tier 1 first, then tier 2
		bool seen2 = false, ok = true;
		for (const auto& e: pp.entries) {
			if (e.tier == 2) seen2 = true;
			else if (seen2) ok = false;
		}
		CHECK(ok);
		// inside a scene: earliest first_ms first (key 111 at 100 ms before 110 at 5000 ms)
		size_t p110 = 0, p111 = 0;
		for (size_t i = 0; i < pp.entries.size(); i++) {
			if (pp.entries[i].key == 110) p110 = i;
			if (pp.entries[i].key == 111) p111 = i;
		}
		CHECK(p111 < p110);
		// a key used by a tier-1 scene and a tier-2 scene is tier 1
		File g = db;
		NotePipeline(g, 130, "l5", Used);
		CHECK(PlanPipelines(g, PlanTiers(g, slots, Config {}), {130}).entries[0].tier == 1);
		// inferred rows: ignored for a key with a direct row (strict), counted for keys with none, and with all rows
		File inf = db;
		NotePipeline(inf, 130, "l4", Inferred); // l4 is in tier 1 for this slot; 130 is tagged l3 (tier 2)
		NotePipeline(inf, 1, "l4", Inferred);   // key 1 has no direct row: placed by the inferred one
		const auto strict = PlanPipelines(inf, PlanTiers(inf, slots, Config {}), {130, 1});
		const auto loose  = PlanPipelines(inf, PlanTiers(inf, slots, Config {}), {130, 1}, false);
		std::map<uint64_t, uint8_t> ts, tl;
		for (const auto& e: strict.entries) ts[e.key] = e.tier;
		for (const auto& e: loose.entries) tl[e.key] = e.tier;
		CHECK(ts[130] == 2 && tl[130] == 1);
		CHECK(strict.unattributed == 0 && ts[1] == 1);
		// empty database: everything is unattributed tier 1 (safe)
		const auto none = PlanPipelines(File {}, Plan {}, all);
		CHECK(none.tier1_count == all.size() && none.unattributed == all.size());
		// deterministic
		const auto again = PlanPipelines(db, PlanTiers(db, slots, Config {}), all);
		bool same = again.entries.size() == pp.entries.size();
		for (size_t i = 0; same && i < pp.entries.size(); i++) same = again.entries[i].key == pp.entries[i].key;
		CHECK(same);
	}
	std::printf("LevelTierTests: %d checks, %d failed\n", g_checks, g_failed);
	return g_failed == 0 ? 0 : 1;
}
