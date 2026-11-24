#include "graphics/host_gpu/renderer/pipeline/levelDb.h"

#include <cstdio>

using namespace Libs::Graphics::LevelDb;

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

static File Make(uint64_t source) {
	File f;
	f.title_id = "TEST";
	f.producer = "tests";
	f.sources.insert(source);
	NotePipeline(f, 0x11, "hub", FirstSeen, 100);
	NotePipeline(f, 0x11, "level_a", FirstSeen | Used, 2000);
	NotePipeline(f, 0x22, "level_a", Used, 500);
	NotePipeline(f, 0x33, "level_b", FirstSeen);
	NoteShader(f, 0xaaaa, "level_a", Used, 100);
	f.scenes[0].runs = 1;
	f.scenes[1].runs = 1;
	return f;
}

static bool Same(const File& a, const File& b) { return Encode(a) == Encode(b); }

int main() {
	// Rows, queries.
	{
		const auto f = Make(1);
		CHECK(f.scenes.size() == 3);
		CHECK(KeysInScene(f, "level_a") == (std::vector<uint64_t> {0x11, 0x22}));
		CHECK(KeysInScene(f, "hub") == (std::vector<uint64_t> {0x11}));
		CHECK(KeysInScene(f, "nowhere").empty());
		CHECK(ScenesOfKey(f, 0x11) == (std::vector<std::string> {"hub", "level_a"}));
		CHECK(ScenesOfKey(f, 0x99).empty());
		CHECK(UnattributedKeys(f, {0x11, 0x44, 0x55}) == (std::vector<uint64_t> {0x44, 0x55}));
		// a key used twice in one scene keeps one row, runs add, first_ms takes the minimum
		File g;
		NotePipeline(g, 7, "s", Used, 900);
		NotePipeline(g, 7, "s", FirstSeen, 300);
		CHECK(g.pipelines[7].size() == 1 && g.pipelines[7][0].runs == 2 && g.pipelines[7][0].first_ms == 300);
		CHECK(g.pipelines[7][0].flags == (Used | FirstSeen));
	}
	// Round trip and damage.
	{
		const auto f     = Make(1);
		auto       bytes = Encode(f);
		File       back;
		std::string err;
		CHECK(Decode(bytes, back, &err));
		CHECK(Same(f, back) && back.title_id == "TEST" && back.pipelines.size() == 3 && back.sources.count(1));
		CHECK(back.pipelines[0x11][1].first_ms == 2000);
		// every single-byte corruption and every truncation is rejected (or decodes to the same file)
		int accepted_wrong = 0;
		for (size_t i = 0; i < bytes.size(); i++) {
			auto bad = bytes;
			bad[i] ^= 0x5a;
			File out;
			if (Decode(bad, out) && !Same(out, f)) accepted_wrong++;
		}
		CHECK(accepted_wrong == 0);
		for (size_t n = 0; n < bytes.size(); n += 7) {
			File out;
			CHECK(!Decode(bytes.data(), n, out) && out.pipelines.empty());
		}
		File out;
		CHECK(!Decode(std::vector<uint8_t> {}, out));
		CHECK(!Decode(std::vector<uint8_t>(64, 0), out));
		auto longer = bytes;
		longer.push_back(0);
		CHECK(!Decode(longer, out, &err)); // trailing bytes
	}
	// Versioning.
	{
		auto bytes = Encode(Make(1));
		File out;
		std::string err;
		bytes[8] = static_cast<uint8_t>(FormatVersion + 1); // a newer writer
		CHECK(!Decode(bytes, out, &err) && err == "unsupported version");
		bytes[8] = 0;
		CHECK(!Decode(bytes, out, &err));
		bytes[8] = static_cast<uint8_t>(FormatVersion);
		CHECK(Decode(bytes, out, &err));
		bytes[0] = 'X';
		CHECK(!Decode(bytes, out, &err) && err == "not a KYLVDB1 file");
	}
	// Merge: idempotent, commutative, associative, partial overlap.
	{
		const auto a = Make(1);
		File       b;
		b.sources.insert(2);
		NotePipeline(b, 0x11, "level_a", Used, 100);
		NotePipeline(b, 0x44, "level_c", FirstSeen);
		b.scenes[0].runs = 1;
		File c;
		c.sources.insert(3);
		NotePipeline(c, 0x44, "level_c", Used, 50);
		c.scenes[0].runs = 2;

		File ab = a;
		Merge(ab, b);
		File ab2 = ab;
		Merge(ab2, b); // same source again
		CHECK(Same(ab, ab2));
		File aa = a;
		Merge(aa, a);
		CHECK(Same(aa, a));
		const auto& rows = ab.pipelines[0x11];
		CHECK(rows.size() == 2);
		CHECK(rows[1].runs == 2 && rows[1].first_ms == 100 && (rows[1].flags & Used)); // level_a: runs add, min time
		CHECK(ab.sources.size() == 2 && ab.pipelines.count(0x44));
		// commutative
		File ba = b;
		Merge(ba, a);
		CHECK(KeysInScene(ab, "level_a") == KeysInScene(ba, "level_a"));
		CHECK(ab.pipelines.size() == ba.pipelines.size());
		for (const auto& [k, r]: ab.pipelines) {
			const auto& o = ba.pipelines[k];
			CHECK(r.size() == o.size());
			for (const auto& x: r) {
				bool found = false;
				for (const auto& y: o) {
					if (ab.scenes[x.scene].name != ba.scenes[y.scene].name) continue;
					found = x.runs == y.runs && x.flags == y.flags && x.first_ms == y.first_ms;
				}
				CHECK(found);
			}
		}
		// associative: (a+b)+c == a+(b+c) by content
		File l = ab;
		Merge(l, c);
		File bc = b;
		Merge(bc, c);
		File r = a;
		Merge(r, bc);
		CHECK(l.pipelines.size() == r.pipelines.size() && l.sources == r.sources);
		CHECK(KeysInScene(l, "level_c") == KeysInScene(r, "level_c"));
		CHECK(l.scenes[static_cast<size_t>(FindScene(l, "level_c"))].runs == 2);
		CHECK(r.scenes[static_cast<size_t>(FindScene(r, "level_c"))].runs == 2);
		// partial overlap: shared source 1 + new source 4: flags fold, runs do not double
		File d = a;
		d.sources.insert(4);
		NotePipeline(d, 0x22, "level_a", FirstSeen, 10);
		File e = a;
		Merge(e, d);
		CHECK(e.pipelines[0x22][0].runs == 2 && (e.pipelines[0x22][0].flags & FirstSeen) && e.pipelines[0x22][0].first_ms == 10);
		// sourceless files are always folded (no id to dedupe by)
		File s;
		NotePipeline(s, 5, "x", Used);
		File t;
		Merge(t, s);
		CHECK(t.pipelines.count(5));
	}
	// Inference: all stage shaders used in a scene.
	{
		File f;
		NoteShader(f, 1, "lvl", Used, 100);
		NoteShader(f, 2, "lvl", Used, 900);
		NoteShader(f, 2, "other", Used, 10);
		NoteShader(f, 3, "other", Used, 10);
		NoteShader(f, 4, "lvl", FirstSeen); // tagged but never seen: not a use
		std::vector<PipelineShaders> ps = {{0xA, {1, 2}}, {0xB, {2, 3}}, {0xC, {1, 9}}, {0xD, {1, 4}}, {0xE, {}}};
		CHECK(Infer(f, ps) == 2);
		CHECK(KeysInScene(f, "lvl") == (std::vector<uint64_t> {0xA}));
		CHECK(KeysInScene(f, "other") == (std::vector<uint64_t> {0xB}));
		CHECK(f.pipelines[0xA][0].first_ms == 900 && (f.pipelines[0xA][0].flags & Inferred)); // as late as its last shader
		CHECK(Infer(f, ps) == 0); // idempotent
		CHECK(f.pipelines.count(0xC) == 0 && f.pipelines.count(0xD) == 0); // an unknown or only-tagged shader: no row
		// a pipeline used in two scenes gets both rows
		NoteShader(f, 1, "other", Used, 5);
		NoteShader(f, 2, "other", Used, 5);
		Infer(f, ps);
		CHECK(KeysInScene(f, "other").size() == 2);
	}
	// scenes.tsv.
	{
		std::vector<Scene> s;
		std::string        err;
		CHECK(ParseScenesTsv("# c\nworldmap\tmenu\n\nhub_a\thub\t-\thub_a\nlvl_1\tlevel\t1\tone\nlvl_2\tlevel\t2\r\n", s, &err));
		CHECK(s.size() == 4 && s[0].cls == SceneClass::Menu && s[0].order == UnknownOrder && s[2].order == 1 && s[2].save_id == "one" &&
		      s[3].order == 2);
		CHECK(!ParseScenesTsv("a\tnope\n", s, &err) && s.empty());
		CHECK(!ParseScenesTsv("a\tlevel\t1\nb\tlevel\t1\n", s, &err) && err.find("duplicate order") != std::string::npos);
		CHECK(!ParseScenesTsv("a\tlevel\nA\tmenu\na\tmenu\n", s, &err) && err.find("duplicate scene") != std::string::npos);
		CHECK(!ParseScenesTsv("a\tlevel\tx\n", s, &err));
		CHECK(!ParseScenesTsv("a\tlevel\t0\n", s, &err));
		CHECK(!ParseScenesTsv("justone\n", s, &err));
		File f = Make(1);
		std::vector<Scene> meta;
		CHECK(ParseScenesTsv("level_a\tlevel\t1\tsave_a\nlevel_z\tlevel\t2\n", meta));
		ApplyScenes(f, meta);
		CHECK(f.scenes[static_cast<size_t>(FindScene(f, "level_a"))].order == 1 && FindScene(f, "level_z") >= 0);
		CHECK(f.scenes[static_cast<size_t>(FindScene(f, "level_a"))].save_id == "save_a");
		CHECK(SceneClassFromName("hub") == SceneClass::Hub && SceneClassFromName("zzz") == SceneClass::Unknown);
		CHECK(IsAlwaysClass(SceneClass::Menu) && !IsAlwaysClass(SceneClass::Level) && !IsAlwaysClass(SceneClass::Bonus));
	}
	std::printf("LevelDbTests: %d checks, %d failed\n", g_checks, g_failed);
	return g_failed == 0 ? 0 : 1;
}
