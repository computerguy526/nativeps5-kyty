// nps5 perfmode: unit tests of the scenery keys and the scene-gated draw skip (renderer/sceneSkip.h, header only core). Standalone:
//   clang++ -std=c++20 -O2 -Isrc tests/SceneSkipTests.cpp -o /tmp/scene_skip_tests && /tmp/scene_skip_tests
// CMake target: scene_skip_tests. SYNTHETIC data only: invented scene names and program hashes.
#include "graphics/host_gpu/renderer/sceneSkip.h"

#include <cstdio>
#include <string>

using namespace Libs::Graphics::SceneSkip;

static int g_checks = 0, g_failed = 0;
#define CHECK(c)                                                                                   \
	do {                                                                                           \
		g_checks++;                                                                                \
		if (!(c)) {                                                                                \
			g_failed++;                                                                            \
			std::printf("FAILED %s:%d %s\n", __FILE__, __LINE__, #c);                              \
		}                                                                                          \
	} while (0)

static const char* kTable =
    "# test table\n"
    "preset\tbalanced\tflowers=50,grass=50\n"
    "preset\tperformance\tflowers=0,grass=0,moss=0,other=0\n"
    "start\tflowers=100,grass=100,moss=100,other=100\n"
    "skip\tlevel_a\tflowers\t00000000000000a1\t00000000000000b1\n"
    "skip\tlevel_a\tflowers\t00000000000000a2\t-\n"
    "skip\tlevel_a\tgrass+moss\t00000000000000a3\t00000000000000b3\n"
    "skip\tlevel_b\tgrass\t00000000000000a4\t00000000000000b4\n"
    "level\tlevel_a\tflowers=both,grass=entry,moss=both\n"
    "level\tlevel_c\tother=entry\n";

static Mask M(int f, int g, int m, int o) {
	return {static_cast<uint8_t>(f), static_cast<uint8_t>(g), static_cast<uint8_t>(m), static_cast<uint8_t>(o)};
}

static void TestParse() {
	const auto t = ParseTable(kTable);
	CHECK(t.bad_lines == 0 && t.dropped == 0);
	CHECK(t.presets.size() == 2 && t.presets[0].name == "balanced" && t.presets[1].name == "performance");
	CHECK(t.presets[0].mask == M(50, 50, 100, 100));
	CHECK(t.presets[1].mask == M(0, 0, 0, 0));
	CHECK(t.start == kAllOn);
	CHECK(t.skips.size() == 4);
	CHECK(t.skips[0].scene == "level_a" && t.skips[0].cats == 1 && t.skips[0].vs == 0xa1 && t.skips[0].ps == 0xb1 && t.skips[0].has_ps);
	CHECK(!t.skips[1].has_ps);
	CHECK(t.skips[2].cats == ((1u << Grass) | (1u << Moss)));
	// CRLF and missing trailing newline
	const auto c = ParseTable("start\tflowers=0\r\nskip\ts\tgrass\tff\t-");
	CHECK(c.bad_lines == 0 && c.start == M(0, 100, 100, 100) && c.skips.size() == 1 && c.skips[0].vs == 0xff);
	// bad forms are counted and ignored, the good lines around them still count
	const char* bad[] = {"preset\tfull\tflowers=0",       "preset\tcustom\tflowers=0",     "preset\tx\tflowers=101",       "preset\tx\ttrees=0",
	                     "preset\tx\tflowers",            "preset\tx",                     "start\tgrass=1x",              "start\t",
	                     "skip\ts\tgrass\t0\t-",          "skip\ts\ttrees\tab\t-",         "skip\ts\tgrass\tzz\t-",        "skip\ts\tgrass\tab\tzz",
	                     "skip\ts\tgrass\tab",            "skip\t../x\tgrass\tab\t-",      "skip\ts\tgrass+\tab\t-",       "wat\t1",
	                     "skip\ts\tgrass\t11111111111111111\t-"};
	for (const auto* b: bad) {
		const auto t2 = ParseTable(std::string(b) + "\nstart\tflowers=0\n");
		CHECK(t2.bad_lines == 1);
		CHECK(t2.start == M(0, 100, 100, 100));
		CHECK(t2.skips.empty() && t2.presets.empty());
	}
	CHECK(ParseTable("").skips.empty());
	CHECK(ParseTable("\n\n# only comments\n").bad_lines == 0);
}

static void TestLimits() {
	std::string text;
	for (int i = 0; i < 70; i++) {
		char line[96];
		std::snprintf(line, sizeof(line), "skip\tbig\tgrass\t%x\t-\n", 0x100 + i);
		text += line;
	}
	text += "skip\tother_scene\tgrass\t5\t-\n";
	const auto t = ParseTable(text);
	CHECK(t.skips.size() == kMaxPairs + 1); // 64 of "big" and the one of the other scene
	CHECK(t.dropped == 6);
	CHECK(SelectActive(t, "big", M(100, 0, 100, 100)).size() == kMaxPairs);
	// presets over the limit
	std::string p;
	for (int i = 0; i < 12; i++) p += "preset\tp" + std::to_string(i) + "\tflowers=0\n";
	const auto tp = ParseTable(p);
	CHECK(tp.presets.size() == kMaxPresets && tp.bad_lines == 4);
}

static void TestSelect() {
	const auto t = ParseTable(kTable);
	// everything on: nothing is skipped
	CHECK(SelectActive(t, "level_a", kAllOn).empty());
	// flowers off: the two flower pairs of level_a only
	auto a = SelectActive(t, "level_a", M(0, 100, 100, 100));
	CHECK(a.size() == 2 && a[0].vs == 0xa1 && a[1].vs == 0xa2);
	// a pair that draws grass AND moss is skipped only while both are off
	CHECK(SelectActive(t, "level_a", M(100, 0, 100, 100)).empty());
	CHECK(SelectActive(t, "level_a", M(100, 100, 0, 100)).empty());
	a = SelectActive(t, "level_a", M(100, 0, 0, 100));
	CHECK(a.size() == 1 && a[0].vs == 0xa3);
	// a thinned category is not off
	CHECK(SelectActive(t, "level_a", M(50, 100, 100, 100)).empty());
	CHECK(SelectActive(t, "level_a", M(50, 50, 50, 50)).empty());
	// another scene = no skip of level_a's pairs; unknown or empty scene = none
	CHECK(SelectActive(t, "level_b", M(0, 100, 100, 100)).empty());
	a = SelectActive(t, "level_b", M(100, 0, 100, 100));
	CHECK(a.size() == 1 && a[0].vs == 0xa4);
	CHECK(SelectActive(t, "", M(0, 0, 0, 0)).empty());
	CHECK(SelectActive(t, "nowhere", M(0, 0, 0, 0)).empty());
}

static void TestActiveSet() {
	ActiveSet s;
	CHECK(s.Count() == 0);
	CHECK(!s.Contains(0xa1, 0xb1, true));
	const auto t = ParseTable(kTable);
	s.Publish(SelectActive(t, "level_a", M(0, 100, 100, 100)));
	CHECK(s.Count() == 2);
	CHECK(s.Contains(0xa1, 0xb1, true));   // the pair
	CHECK(!s.Contains(0xa1, 0xb9, true));  // same vertex program, other pixel program: a different object, not skipped
	CHECK(!s.Contains(0xa1, 0, false));    // no pixel program: a row with one never matches
	CHECK(s.Contains(0xa2, 0x1234, true)); // ps=- : the vertex program alone
	CHECK(s.Contains(0xa2, 0, false));
	CHECK(!s.Contains(0xb1, 0xa1, true));  // vs and ps are not interchangeable
	s.Publish({});                         // another scene / everything on again
	CHECK(s.Count() == 0 && !s.Contains(0xa1, 0xb1, true));
	// the limit
	std::vector<SkipRow> many;
	for (uint64_t i = 1; i <= 100; i++) many.push_back({"x", 1, i, 0, false});
	s.Publish(many);
	CHECK(s.Count() == kMaxPairs && s.Contains(64, 0, false) && !s.Contains(65, 0, false));
	// the global one starts empty (nothing is active unless the runtime publishes)
	CHECK(!Active() && !Skips(1, 1, true));
}

static void TestCycle() {
	const auto  t = ParseTable(kTable);
	Controller  c;
	c.Init(t);
	CHECK(c.PresetName() == "full" && c.Seq() == 0);
	// 5: full -> balanced -> performance -> full
	auto r = c.OnKey(true, Key::Preset, false, true);
	CHECK(r.consumed && r.changed && c.PresetName() == "balanced" && c.Current() == M(50, 50, 100, 100) && c.Seq() == 1);
	CHECK(r.headline == "Scenery: balanced");
	r = c.OnKey(true, Key::Preset, false, true);
	CHECK(c.PresetName() == "performance" && c.Current() == M(0, 0, 0, 0) && c.Seq() == 2);
	r = c.OnKey(true, Key::Preset, false, true);
	CHECK(c.PresetName() == "full" && c.Current() == kAllOn && c.Seq() == 3 && r.headline == "Scenery: full");
	// from a custom mask 5 goes to full
	c.OnKey(true, Key::Flowers, false, true);
	CHECK(c.PresetName() == "custom");
	c.OnKey(true, Key::Preset, false, true);
	CHECK(c.PresetName() == "full");
	// a table without presets: 5 stays at full (no change, no seq)
	Controller e;
	e.Init(ParseTable("start\tflowers=100\n"));
	const auto before = e.Seq();
	r = e.OnKey(true, Key::Preset, false, true);
	CHECK(r.consumed && !r.changed && e.Seq() == before);
}

static void TestToggles() {
	Controller c;
	c.Init(ParseTable(kTable));
	auto r = c.OnKey(true, Key::Flowers, false, true);
	CHECK(r.changed && r.cat == Flowers && c.Current() == M(0, 100, 100, 100) && r.headline == "Scenery: flowers OFF");
	r = c.OnKey(true, Key::Flowers, false, true);
	CHECK(c.Current() == kAllOn && r.headline == "Scenery: flowers ON");
	c.OnKey(true, Key::Grass, false, true);
	CHECK(c.Current() == M(100, 0, 100, 100));
	c.OnKey(true, Key::Moss, false, true);
	CHECK(c.Current() == M(100, 0, 0, 100));
	c.OnKey(true, Key::Other, false, true);
	CHECK(c.Current() == M(100, 0, 0, 0));
	CHECK(c.Seq() == 5);
	// a thinned category (balanced) goes OFF on its key
	Controller b;
	b.Init(ParseTable(kTable));
	b.OnKey(true, Key::Preset, false, true);
	CHECK(b.Current()[Flowers] == 50);
	b.OnKey(true, Key::Flowers, false, true);
	CHECK(b.Current()[Flowers] == 0 && b.Current()[Grass] == 50);
	// Shift+5 shows the state: no change, no seq
	const auto seq = b.Seq();
	r = b.OnKey(true, Key::Preset, true, true);
	CHECK(r.consumed && !r.changed && r.show && b.Seq() == seq && r.headline == "Scenery: custom");
	CHECK(b.MakeToast(r.headline, 0, true).line2 == "F- G~ M+ O+");
}

static void TestKeyMap() {
	Controller c;
	c.Init(ParseTable(kTable));
	// keys off: nothing is consumed, nothing changes, for every key, down and up
	for (const auto k: {Key::Help, Key::Preset, Key::Flowers, Key::Grass, Key::Moss, Key::Other, Key::None}) {
		for (const bool pressed: {true, false}) {
			const auto r = c.OnKey(false, k, false, pressed);
			CHECK(!r.consumed && !r.changed);
		}
	}
	CHECK(c.Current() == kAllOn && c.Seq() == 0);
	// keys on: a key that is not ours is not consumed
	CHECK(!c.OnKey(true, Key::None, false, true).consumed);
	// release (and repeat, which the window passes as pressed=false) is consumed without effect
	const auto r = c.OnKey(true, Key::Flowers, false, false);
	CHECK(r.consumed && !r.changed && c.Current() == kAllOn && c.Seq() == 0);
}

static void TestDigitKeys() {
	// physical digits 4..9 map to the keys; 1-3 (keyboard-pad setting cycles), 0 and anything else are not ours
	CHECK(KeyFromDigit(4) == Key::Help && KeyFromDigit(5) == Key::Preset && KeyFromDigit(6) == Key::Flowers);
	CHECK(KeyFromDigit(7) == Key::Grass && KeyFromDigit(8) == Key::Moss && KeyFromDigit(9) == Key::Other);
	for (const int d: {-1, 0, 1, 2, 3, 10}) CHECK(KeyFromDigit(d) == Key::None);
	// passthrough: with the keys off no digit is consumed or changes anything; with them on 1-3 still reach the game
	Controller c;
	c.Init(ParseTable(kTable));
	for (int d = 0; d <= 9; d++) {
		CHECK(!c.OnKey(false, KeyFromDigit(d), false, true).consumed);
		if (d <= 3) CHECK(!c.OnKey(true, KeyFromDigit(d), false, true).consumed);
	}
	CHECK(c.Current() == kAllOn && c.Seq() == 0);
}

static void TestTexts() {
	Controller c;
	c.Init(ParseTable(kTable));
	CHECK(c.StateText() == "seq=0\npreset=full\nflowers=100\ngrass=100\nmoss=100\nother=100\n");
	c.OnKey(true, Key::Preset, false, true);
	CHECK(c.StateText() == "seq=1\npreset=balanced\nflowers=50\ngrass=50\nmoss=100\nother=100\n");
	c.OnKey(true, Key::Preset, false, true);
	CHECK(c.StateText() == "seq=2\npreset=performance\nflowers=0\ngrass=0\nmoss=0\nother=0\n");
	CHECK(c.RowText() == "F- G- M- O- (performance)");
	c.OnKey(true, Key::Preset, false, true);
	c.OnKey(true, Key::Flowers, false, true);
	CHECK(c.RowText() == "F- G+ M+ O+ (custom)");
	CHECK(c.LogLine(0, 2, "level_a") == "[scenery] seq 4: flowers OFF (instant: 2 shader pairs in level_a), grass on, moss on, other on");
	CHECK(c.LogLine(0, 1, "") == "[scenery] seq 4: flowers OFF (instant: 1 shader pair in no scene), grass on, moss on, other on");
	CHECK(MaskText(M(0, 50, 100, 7)) == "flowers=0,grass=50,moss=100,other=7");
	Mask m;
	CHECK(ParseMask(MaskText(M(0, 50, 100, 7)), m) && m == M(0, 50, 100, 7));
	CHECK(!ParseMask("", m) && !ParseMask("flowers=", m) && !ParseMask("flowers=-1", m) && !ParseMask("tree=1", m) && !ParseMask("flowers=1000", m));
}

static void TestToast() {
	Controller c;
	c.Init(ParseTable(kTable));
	auto r = c.OnKey(true, Key::Flowers, false, true);
	// level data differs from what is loaded: re-entry text; with instant pairs it says hidden now
	CHECK(c.MakeToast(r.headline, 0, false).line2 == "applies when you re-enter the level");
	CHECK(c.MakeToast(r.headline, 2, false).line2 == "hidden now; re-enter the level to apply fully");
	// a level entered after the press: the loaded mask follows
	c.MarkLoaded();
	CHECK(!c.DiffersFromLoaded());
	CHECK(c.MakeToast(r.headline, 2, false).line2 == "same as the level in memory");
	// pressing back to the loaded mask
	c.OnKey(true, Key::Flowers, false, true);
	CHECK(c.DiffersFromLoaded());
	c.OnKey(true, Key::Flowers, false, true);
	CHECK(!c.DiffersFromLoaded());
}

static std::string Join(const std::vector<PanelLine>& v) {
	std::string s;
	for (const auto& l: v) s += l.text + "\n";
	return s;
}

static void TestPanel() {
	const auto t = ParseTable(kTable);
	CHECK(t.bad_lines == 0 && t.levels.size() == 2);
	// how: level lines plus instant from skip rows (level_b has no level line, only a skip row)
	auto h = HowFor(t, "level_a");
	CHECK(h.how[Flowers] == (kHowInstant | kHowEntry) && h.how[Grass] == (kHowEntry | kHowInstant) && h.how[Moss] == (kHowInstant | kHowEntry) && h.how[Other] == 0);
	CHECK(h.pairs[Flowers] == 2 && h.pairs[Grass] == 1 && h.pairs[Moss] == 1 && h.pairs[Other] == 0);
	h = HowFor(t, "level_b");
	CHECK(h.how[Grass] == kHowInstant && h.how[Flowers] == 0 && h.pairs[Grass] == 1);
	h = HowFor(t, "level_c");
	CHECK(h.how[Other] == kHowEntry && h.how[Grass] == 0);
	h = HowFor(t, "");
	CHECK((h.how == std::array<uint8_t, kCatCount> {}));
	// bad level lines are counted
	for (const char* b: {"level\ts\tgrass=sometimes", "level\ts\ttrees=entry", "level\ts", "level\t../x\tgrass=entry", "level\ts\tgrass"}) CHECK(ParseTable(b).bad_lines == 1);
	CHECK(ParseTable("level\ts\tgrass=entry\nlevel\ts\tgrass=instant").levels.size() == 1);
	CHECK(ParseTable("level\ts\tgrass=entry\nlevel\ts\tgrass=instant").levels[0].how[Grass] == (kHowEntry | kHowInstant));

	Controller c;
	c.Init(t);
	const auto p = c.MakePanel(t, "level_b");
	CHECK(p.size() >= 7 && p[0].kind == PanelKind::Title);
	const auto txt = Join(p);
	CHECK(txt.find("Level: level_b") != std::string::npos);
	CHECK(txt.find("Preset: full   5 cycles full > balanced > performance > full") != std::string::npos);
	CHECK(txt.find("6  flowers    n/a   not in this level") != std::string::npos);
	CHECK(txt.find("7  grass      ON    instant  (1 shader pair)") != std::string::npos);
	CHECK(txt.find("8  moss") == std::string::npos || txt.find("n/a") != std::string::npos);
	CHECK(txt.find("re-enter") == std::string::npos); // instant only: nothing to re-enter
	// a level with both
	c.OnKey(true, Key::Flowers, false, true);
	c.OnKey(true, Key::Grass, false, true);
	const auto a = Join(c.MakePanel(t, "level_a"));
	CHECK(a.find("6  flowers    OFF   instant + fully on re-entry  (2 shader pairs)") != std::string::npos);
	CHECK(a.find("7  grass      OFF   instant + fully on re-entry  (1 shader pair)") != std::string::npos);
	CHECK(a.find("9  other") == std::string::npos || a.find("n/a   not in this level") != std::string::npos);
	CHECK(a.find("Re-enter the level to apply the on re-entry changes.") != std::string::npos);
	CHECK(a.find("Changed since this level loaded.") != std::string::npos);
	c.MarkLoaded();
	CHECK(Join(c.MakePanel(t, "level_a")).find("Changed since") == std::string::npos);
	// entry only, thinned
	c.OnKey(true, Key::Preset, false, true); // custom -> full
	c.OnKey(true, Key::Preset, false, true); // balanced
	const auto cc = Join(c.MakePanel(t, "level_c"));
	CHECK(cc.find("9  other      ON    on re-entry") != std::string::npos);
	CHECK(cc.find("Preset: balanced") != std::string::npos);
	// a level with no rows and no level yet
	const auto n = Join(c.MakePanel(t, "level_z"));
	CHECK(n.find("Nothing to switch here.") != std::string::npos && n.find("Level: level_z") != std::string::npos);
	CHECK(Join(c.MakePanel(t, "")).find("not in a level yet") != std::string::npos);
	// no names beyond what the table carries; every line is short enough for the overlay buffer
	for (const auto& l: c.MakePanel(t, "level_a")) CHECK(l.text.size() < 118);
	CHECK(c.MakePanel(t, "level_a").size() <= 16);

	// 4 key: consumed (down, up), toggles the panel, never changes the mask; ignored when the keys are off
	Controller k;
	k.Init(t);
	auto r = k.OnKey(true, Key::Help, false, true);
	CHECK(r.consumed && r.panel && !r.changed && k.Seq() == 0 && k.Current() == kAllOn);
	r = k.OnKey(true, Key::Help, false, false);
	CHECK(r.consumed && !r.panel);
	r = k.OnKey(false, Key::Help, false, true);
	CHECK(!r.consumed && !r.panel);
	r = k.OnKey(true, Key::Flowers, false, true);
	CHECK(r.consumed && !r.panel && r.changed);

	// the auto-hide clock
	PanelClock pc;
	CHECK(!pc.Open(0) && pc.Alpha(10) == 0.0f);
	CHECK(pc.Toggle(1000) && pc.Open(1000) && pc.Open(1000 + kPanelMs - 1) && !pc.Open(1000 + kPanelMs));
	CHECK(pc.Alpha(1000) == 1.0f && pc.Alpha(1000 + kPanelMs - kPanelFadeMs / 2) > 0.0f && pc.Alpha(1000 + kPanelMs - kPanelFadeMs / 2) < 1.0f);
	pc.Extend(5000);
	CHECK(pc.Open(5000 + kPanelMs - 1) && !pc.Open(5000 + kPanelMs));
	CHECK(!pc.Toggle(6000) && !pc.Open(6000)); // 4 again closes
	pc.Extend(7000);
	CHECK(!pc.Open(7000)); // a key press while closed does not open it
	CHECK(pc.Toggle(8000) && pc.Open(8000));
}

int main() {
	TestParse();
	TestLimits();
	TestSelect();
	TestActiveSet();
	TestCycle();
	TestToggles();
	TestKeyMap();
	TestDigitKeys();
	TestTexts();
	TestToast();
	TestPanel();
	std::printf("SceneSkipTests: %d checks, %d failed\n", g_checks, g_failed);
	return g_failed == 0 ? 0 : 1;
}
