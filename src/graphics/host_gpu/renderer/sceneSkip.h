#pragma once

// nps5 perfmode: scene-gated draw skip + the scenery hotkeys (flowers / grass / moss / other decoration).
//
// Two switches, both default off (nothing is read, nothing runs, the keys reach the game as before):
//   KYTY_SCENE_SKIP_FILE=<table>   the instant draw skip: vertex/pixel program pairs of the CURRENT scene (the level the runtime's VFS names,
//                                  nps5_vfs_current_scene) whose scenery categories are all switched off are not recorded. Pairs, not single
//                                  programs: a program alone is shared with other objects. One relaxed atomic load per draw when nothing is active.
//   KYTY_SCENE_KEYS=1              F3 cycles the preset (full -> the table's presets -> full), F4 flowers, F5 grass, F10 small moss, F12 other
//                                  decoration; Shift+F3 shows the state again. Each press writes KYTY_SCENE_STATE_FILE (atomically) for the launcher,
//                                  which stages the level-data part (the overlay links the runtime reads at the next level load), and shows a toast.
//                                  F1 toggles the help panel (what each key does in the current level; auto-hides). F1 is ours only with this switch.
//
// Table (KYTY_SCENE_SKIP_FILE; written by `nps5 run`, tab separated, # comments):
//   preset <name> <cat=keep,...>        a preset of the F3 cycle (keep = percent of the category's objects that stay; 100 = untouched)
//   start  <cat=keep,...>               the mask the launch starts with
//   skip   <scene> <cat+cat> <vs hex> <ps hex or ->   one program pair that draws only objects of those categories in that scene
//   level  <scene> <cat=how,...>        (F1 help panel) how each category applies in that level: instant | entry | both (entry = the level
//                                       data of the next level load); a category not named is not in that level. A `skip` row also makes its
//                                       categories instant for its scene.
//
// This header is pure (no emulator, no GPU, no SDL) so tests/SceneSkipTests.cpp is a plain executable. The runtime half (environment, files,
// scene polling, toast) is sceneSkip.cpp.

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>
#include <vector>

namespace Libs::Graphics::SceneSkip {

enum Cat : uint8_t { Flowers = 0, Grass = 1, Moss = 2, Other = 3 };
inline constexpr size_t      kCatCount         = 4;
inline constexpr const char* kCatNames[]       = {"flowers", "grass", "moss", "other"};
inline constexpr char        kCatLetters[]     = {'F', 'G', 'M', 'O'};
inline constexpr size_t      kMaxPairs         = 64; // per scene (the shape of the diagnostic DrawSkipSet)
inline constexpr size_t      kMaxPresets       = 8;
inline constexpr uint64_t    kToastMs          = 4000;
inline constexpr uint64_t    kToastFadeMs      = 500;
inline constexpr uint64_t    kPanelMs          = 15000; // the F1 panel hides itself after this long
inline constexpr uint64_t    kPanelFadeMs      = 500;
inline constexpr size_t      kMaxLevels        = 64;
inline constexpr uint8_t     kHowInstant       = 1;
inline constexpr uint8_t     kHowEntry         = 2;

// Percent of each category's objects that stay: 100 = untouched, 0 = all removed, in between = thinned.
using Mask = std::array<uint8_t, kCatCount>;
inline constexpr Mask kAllOn {100, 100, 100, 100};

inline bool CatFromName(std::string_view name, Cat& out) {
	for (size_t i = 0; i < kCatCount; i++) {
		if (name == kCatNames[i]) {
			out = static_cast<Cat>(i);
			return true;
		}
	}
	return false;
}

struct Preset {
	std::string name;
	Mask        mask = kAllOn;
};

struct SkipRow {
	std::string scene;
	uint8_t     cats   = 0; // bit per Cat
	uint64_t    vs     = 0;
	uint64_t    ps     = 0;
	bool        has_ps = false;
};

// How each category applies in one level (bits kHowInstant | kHowEntry per category; 0 = not in that level).
struct LevelInfo {
	std::string                 scene;
	std::array<uint8_t, kCatCount> how {};
};

struct Table {
	std::vector<Preset>  presets;
	std::vector<LevelInfo> levels;
	Mask                 start = kAllOn;
	std::vector<SkipRow> skips;
	uint32_t             bad_lines = 0; // lines that did not parse (ignored)
	uint32_t             dropped   = 0; // rows over kMaxPairs for their scene (ignored)
};

namespace Detail {

inline std::vector<std::string_view> Split(std::string_view s, char sep) {
	std::vector<std::string_view> out;
	size_t                        a = 0;
	for (;;) {
		const auto b = s.find(sep, a);
		if (b == std::string_view::npos) {
			out.push_back(s.substr(a));
			return out;
		}
		out.push_back(s.substr(a, b - a));
		a = b + 1;
	}
}

inline bool ParseHex(std::string_view s, uint64_t& out) {
	if (s.empty() || s.size() > 16) return false;
	uint64_t v = 0;
	for (const char c: s) {
		unsigned d;
		if (c >= '0' && c <= '9') d = static_cast<unsigned>(c - '0');
		else if (c >= 'a' && c <= 'f') d = static_cast<unsigned>(c - 'a') + 10u;
		else if (c >= 'A' && c <= 'F') d = static_cast<unsigned>(c - 'A') + 10u;
		else return false;
		v = (v << 4) | d;
	}
	out = v;
	return true;
}

inline bool PlainName(std::string_view s) {
	if (s.empty() || s.size() > 63) return false;
	for (const char c: s) {
		const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '-' || c == '.';
		if (!ok) return false;
	}
	return s[0] != '.';
}

} // namespace Detail

// "flowers=50,grass=0" -> mask (categories not named stay 100). false on anything else.
inline bool ParseMask(std::string_view spec, Mask& out) {
	Mask m = kAllOn;
	if (spec.empty()) return false;
	for (const auto item: Detail::Split(spec, ',')) {
		const auto eq = item.find('=');
		Cat        c;
		if (eq == std::string_view::npos || !CatFromName(item.substr(0, eq), c)) return false;
		const auto pct = item.substr(eq + 1);
		if (pct.empty() || pct.size() > 3) return false;
		unsigned v = 0;
		for (const char d: pct) {
			if (d < '0' || d > '9') return false;
			v = v * 10 + static_cast<unsigned>(d - '0');
		}
		if (v > 100) return false;
		m[c] = static_cast<uint8_t>(v);
	}
	out = m;
	return true;
}

inline std::string MaskText(const Mask& m) {
	std::string out;
	for (size_t i = 0; i < kCatCount; i++) {
		if (i != 0) out += ',';
		out += kCatNames[i];
		out += '=';
		out += std::to_string(m[i]);
	}
	return out;
}

inline Table ParseTable(std::string_view text) {
	Table t;
	for (auto line: Detail::Split(text, '\n')) {
		while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.remove_suffix(1);
		if (line.empty() || line.front() == '#') continue;
		const auto f = Detail::Split(line, '\t');
		if (f[0] == "preset" && f.size() == 3) {
			Mask m;
			if (!Detail::PlainName(f[1]) || f[1] == "full" || f[1] == "custom" || !ParseMask(f[2], m) || t.presets.size() >= kMaxPresets) {
				t.bad_lines++;
				continue;
			}
			t.presets.push_back({std::string(f[1]), m});
		} else if (f[0] == "start" && f.size() == 2) {
			Mask m;
			if (!ParseMask(f[1], m)) {
				t.bad_lines++;
				continue;
			}
			t.start = m;
		} else if (f[0] == "skip" && f.size() == 5) {
			SkipRow r;
			bool    ok = Detail::PlainName(f[1]);
			r.scene    = std::string(f[1]);
			for (const auto c: Detail::Split(f[2], '+')) {
				Cat cat;
				if (!CatFromName(c, cat)) ok = false;
				else r.cats |= static_cast<uint8_t>(1u << cat);
			}
			ok = ok && r.cats != 0 && Detail::ParseHex(f[3], r.vs) && r.vs != 0;
			if (f[4] == "-") r.has_ps = false;
			else {
				r.has_ps = true;
				ok       = ok && Detail::ParseHex(f[4], r.ps) && r.ps != 0;
			}
			if (!ok) {
				t.bad_lines++;
				continue;
			}
			size_t in_scene = 0;
			for (const auto& s: t.skips) in_scene += s.scene == r.scene ? 1u : 0u;
			if (in_scene >= kMaxPairs) {
				t.dropped++;
				continue;
			}
			t.skips.push_back(std::move(r));
		} else if (f[0] == "level" && f.size() == 3) {
			LevelInfo li;
			bool      ok = Detail::PlainName(f[1]) && t.levels.size() < kMaxLevels;
			li.scene     = std::string(f[1]);
			for (const auto item: Detail::Split(f[2], ',')) {
				const auto eq = item.find('=');
				Cat        c;
				if (eq == std::string_view::npos || !CatFromName(item.substr(0, eq), c)) {
					ok = false;
					continue;
				}
				const auto h = item.substr(eq + 1);
				if (h == "instant") li.how[c] = kHowInstant;
				else if (h == "entry") li.how[c] = kHowEntry;
				else if (h == "both") li.how[c] = kHowInstant | kHowEntry;
				else ok = false;
			}
			if (!ok) {
				t.bad_lines++;
				continue;
			}
			bool merged = false;
			for (auto& l: t.levels) {
				if (l.scene != li.scene) continue;
				for (size_t i = 0; i < kCatCount; i++) l.how[i] |= li.how[i];
				merged = true;
			}
			if (!merged) t.levels.push_back(std::move(li));
		} else {
			t.bad_lines++;
		}
	}
	return t;
}

// How each category applies in `scene`: the `level` lines, plus instant for every category a `skip` row of the scene names.
struct SceneHow {
	std::array<uint8_t, kCatCount>  how {};
	std::array<uint32_t, kCatCount> pairs {}; // `skip` rows of the scene that name the category
};
inline SceneHow HowFor(const Table& table, std::string_view scene) {
	SceneHow out;
	if (scene.empty()) return out;
	for (const auto& l: table.levels) {
		if (l.scene == scene) out.how = l.how;
	}
	for (const auto& r: table.skips) {
		if (r.scene != scene) continue;
		for (size_t c = 0; c < kCatCount; c++) {
			if ((r.cats & (1u << c)) != 0) {
				out.how[c] |= kHowInstant;
				out.pairs[c]++;
			}
		}
	}
	return out;
}

// The rows of `scene` whose categories are ALL off (keep 0) under `mask`: a pair that draws grass and another category stays while any of
// them is on. Thinned categories (keep 1..99) are not off: no draw skip can keep "half".
inline std::vector<SkipRow> SelectActive(const Table& table, std::string_view scene, const Mask& mask) {
	std::vector<SkipRow> out;
	if (scene.empty()) return out;
	for (const auto& r: table.skips) {
		if (r.scene != scene) continue;
		bool all_off = true;
		for (size_t c = 0; c < kCatCount; c++) {
			if ((r.cats & (1u << c)) != 0 && mask[c] != 0) all_off = false;
		}
		if (all_off && out.size() < kMaxPairs) out.push_back(r);
	}
	return out;
}

// The pairs the draw path checks: a fixed lock-free array, published by one thread at a time (a flip, a key press). A draw that races a
// publish sees the old or the new list or a mix for one draw: harmless (one draw more or less skipped for a frame).
struct ActiveSet {
	std::array<std::atomic<uint64_t>, kMaxPairs> vs {};
	std::array<std::atomic<uint64_t>, kMaxPairs> ps {};
	std::array<std::atomic<uint8_t>, kMaxPairs>  has_ps {};
	std::atomic<uint32_t>                        count {0};

	void Publish(const std::vector<SkipRow>& rows) {
		count.store(0, std::memory_order_release);
		uint32_t n = 0;
		for (const auto& r: rows) {
			if (n >= kMaxPairs) break;
			vs[n].store(r.vs, std::memory_order_relaxed);
			ps[n].store(r.ps, std::memory_order_relaxed);
			has_ps[n].store(r.has_ps ? 1 : 0, std::memory_order_relaxed);
			n++;
		}
		count.store(n, std::memory_order_release);
	}
	[[nodiscard]] uint32_t Count() const { return count.load(std::memory_order_relaxed); }
	// ps_known: the draw has a pixel program (ps_hash valid). A row with a pixel program never matches a draw without one.
	[[nodiscard]] bool Contains(uint64_t vs_hash, uint64_t ps_hash, bool ps_known) const {
		const auto n = count.load(std::memory_order_acquire);
		for (uint32_t i = 0; i < n; i++) {
			if (vs[i].load(std::memory_order_relaxed) != vs_hash) continue;
			if (has_ps[i].load(std::memory_order_relaxed) == 0) return true;
			if (ps_known && ps[i].load(std::memory_order_relaxed) == ps_hash) return true;
		}
		return false;
	}
};

enum class Key : uint8_t { None, F1, F3, F4, F5, F10, F12 };

struct Toast {
	std::string line1;
	std::string line2;
};

// The F1 panel: lines of text with a kind (the overlay picks the colour).
enum class PanelKind : uint8_t { Title, Heading, On, Off, Dim, Note };
struct PanelLine {
	std::string text;
	PanelKind   kind = PanelKind::Dim;
};

// The F1 panel's open/close state with its auto-hide deadline (pure: the caller supplies the clock).
struct PanelClock {
	uint64_t until_ms = 0;
	[[nodiscard]] bool Open(uint64_t now) const { return now < until_ms; }
	// F1 pressed: closes an open panel, opens a closed one for kPanelMs. -> open now.
	bool Toggle(uint64_t now) {
		until_ms = Open(now) ? 0 : now + kPanelMs;
		return until_ms != 0;
	}
	// Any other key press while open keeps it up for as long as the player is using the keys.
	void Extend(uint64_t now) {
		if (Open(now)) until_ms = now + kPanelMs;
	}
	[[nodiscard]] float Alpha(uint64_t now) const {
		if (!Open(now)) return 0.0f;
		return until_ms - now >= kPanelFadeMs ? 1.0f : static_cast<float>(until_ms - now) / static_cast<float>(kPanelFadeMs);
	}
};

// The scenery state machine: pure; the runtime feeds it key presses. Not thread safe (the runtime holds a mutex).
class Controller {
public:
	void Init(const Table& table) {
		m_presets = table.presets;
		m_mask    = table.start;
		m_loaded  = table.start;
		m_seq     = 0;
	}

	[[nodiscard]] const Mask& Current() const { return m_mask; }
	[[nodiscard]] const Mask& Loaded() const { return m_loaded; }
	[[nodiscard]] uint32_t    Seq() const { return m_seq; }
	// The level data now on screen is the one the launcher staged before the level was entered: a scene change makes `loaded` the mask.
	void MarkLoaded() { m_loaded = m_mask; }
	[[nodiscard]] bool DiffersFromLoaded() const { return m_mask != m_loaded; }

	[[nodiscard]] std::string PresetName() const { return PresetNameOf(m_mask); }
	[[nodiscard]] std::string PresetNameOf(const Mask& m) const {
		if (m == kAllOn) return "full";
		for (const auto& p: m_presets) {
			if (p.mask == m) return p.name;
		}
		return "custom";
	}
	// The next name of the cycle full -> presets in file order -> full. From a custom mask: full.
	[[nodiscard]] std::string NextPresetName() const {
		const auto cur = PresetName();
		if (cur == "custom") return "full";
		if (cur == "full") return m_presets.empty() ? std::string("full") : m_presets.front().name;
		for (size_t i = 0; i < m_presets.size(); i++) {
			if (m_presets[i].name == cur) return i + 1 < m_presets.size() ? m_presets[i + 1].name : std::string("full");
		}
		return "full";
	}
	[[nodiscard]] Mask MaskOfPreset(const std::string& name) const {
		for (const auto& p: m_presets) {
			if (p.name == name) return p.mask;
		}
		return kAllOn;
	}

	struct Result {
		bool        consumed = false; // the key is ours (down, up and repeat of it never reach the game)
		bool        changed  = false; // the mask changed (stage, publish, toast)
		bool        show     = false; // show the state without a change (Shift+F3)
		bool        panel    = false; // F1: toggle the help panel
		int         cat      = -1;    // the one category a toggle changed
		std::string headline;         // toast line 1
	};

	// A key event. `enabled`: KYTY_SCENE_KEYS=1; when false nothing is consumed (the keys reach the game exactly as before).
	Result OnKey(bool enabled, Key key, bool shift, bool pressed) {
		Result r;
		if (!enabled || key == Key::None) return r;
		r.consumed = true;
		if (!pressed) return r;
		if (key == Key::F1) {
			r.panel = true;
			return r;
		}
		if (key == Key::F3) {
			if (shift) {
				r.show     = true;
				r.headline = "Scenery: " + PresetName();
				return r;
			}
			const auto next = NextPresetName();
			const Mask target = next == "full" ? kAllOn : MaskOfPreset(next);
			r.changed         = target != m_mask;
			m_mask            = target;
			r.headline        = "Scenery: " + next;
		} else {
			const Cat c = key == Key::F4 ? Flowers : key == Key::F5 ? Grass : key == Key::F10 ? Moss : Other;
			m_mask[c]   = m_mask[c] == 0 ? 100 : 0; // off <-> on; a thinned category goes off
			r.changed   = true;
			r.cat       = c;
			r.headline  = std::string("Scenery: ") + kCatNames[c] + (m_mask[c] == 0 ? " OFF" : " ON");
		}
		if (r.changed) m_seq++;
		return r;
	}

	// "seq=4\npreset=balanced\nflowers=50\n..." (what the launcher's watcher reads).
	[[nodiscard]] std::string StateText() const {
		std::string s = "seq=" + std::to_string(m_seq) + "\npreset=" + PresetName() + "\n";
		for (size_t i = 0; i < kCatCount; i++) s += std::string(kCatNames[i]) + "=" + std::to_string(m_mask[i]) + "\n";
		return s;
	}

	// The F2 overlay row: "F- G+ M+ O+ (balanced)"; ~ = thinned.
	[[nodiscard]] std::string RowText() const {
		std::string s;
		for (size_t i = 0; i < kCatCount; i++) {
			if (i != 0) s += ' ';
			s += kCatLetters[i];
			s += m_mask[i] == 100 ? '+' : m_mask[i] == 0 ? '-' : '~';
		}
		return s + " (" + PresetName() + ")";
	}

	// "[scenery] seq 4: flowers OFF (instant: 2 shader pairs in hero_ape_escape), grass on, moss on, other on"
	[[nodiscard]] std::string LogLine(int changed_cat, uint32_t pairs, std::string_view scene) const {
		std::string s = "[scenery] seq " + std::to_string(m_seq) + ": ";
		const std::string note = "(instant: " + std::to_string(pairs) + " shader pair" + (pairs == 1 ? "" : "s") + " in " + (scene.empty() ? std::string("no scene") : std::string(scene)) + ")";
		for (size_t i = 0; i < kCatCount; i++) {
			if (i != 0) s += ", ";
			s += kCatNames[i];
			s += m_mask[i] == 100 ? " on" : m_mask[i] == 0 ? " OFF" : " " + std::to_string(m_mask[i]) + "%";
			if (changed_cat == static_cast<int>(i)) s += " " + note;
		}
		if (changed_cat < 0) s += " " + note + " preset " + PresetName();
		return s;
	}

	// The F1 panel for `scene` under the current mask: what each key does in this level and how it applies. `scene` empty = no level yet.
	[[nodiscard]] std::vector<PanelLine> MakePanel(const Table& table, std::string_view scene) const {
		std::vector<PanelLine> out;
		out.push_back({"Scenery options  (F1 closes)", PanelKind::Title});
		out.push_back({"Level: " + (scene.empty() ? std::string("not in a level yet") : std::string(scene)), PanelKind::Heading});
		out.push_back({"Preset: " + PresetName() + "   F3 cycles " + CycleText() + "   Shift+F3 shows state", PanelKind::Heading});
		const auto  h    = HowFor(table, scene);
		const char* keys[] = {"F4", "F5", "F10", "F12"};
		bool        any_entry = false, any_instant = false;
		for (size_t i = 0; i < kCatCount; i++) {
			std::string line = std::string(keys[i]) + "  " + kCatNames[i];
			while (line.size() < 14) line += ' ';
			if (h.how[i] == 0) {
				line += "n/a   not in this level";
				out.push_back({line, PanelKind::Dim});
				continue;
			}
			const uint8_t v = m_mask[i];
			line += v == 100 ? "ON    " : v == 0 ? "OFF   " : (std::to_string(v) + "%   ");
			if ((h.how[i] & kHowInstant) != 0 && (h.how[i] & kHowEntry) != 0) line += "instant + fully on re-entry";
			else if ((h.how[i] & kHowInstant) != 0) line += "instant";
			else line += "on re-entry";
			if (h.pairs[i] != 0) line += "  (" + std::to_string(h.pairs[i]) + " shader pair" + (h.pairs[i] == 1 ? "" : "s") + ")";
			any_entry   = any_entry || (h.how[i] & kHowEntry) != 0;
			any_instant = any_instant || (h.how[i] & kHowInstant) != 0;
			out.push_back({line, v == 100 ? PanelKind::On : PanelKind::Off});
		}
		if (!any_entry && !any_instant) out.push_back({"Nothing to switch here.", PanelKind::Note});
		else if (any_entry) out.push_back({"Re-enter the level to apply the on re-entry changes.", PanelKind::Note});
		if (DiffersFromLoaded()) out.push_back({"Changed since this level loaded.", PanelKind::Note});
		return out;
	}

	// "full > balanced > performance > full"
	[[nodiscard]] std::string CycleText() const {
		std::string s = "full";
		for (const auto& p: m_presets) s += " > " + p.name;
		return s + " > full";
	}

	// The toast: line 1 the headline, line 2 what applies when. `pairs` = the draw skip pairs active in the current scene.
	[[nodiscard]] Toast MakeToast(const std::string& headline, uint32_t pairs, bool show_only) const {
		Toast t {headline, {}};
		if (show_only) {
			for (size_t i = 0; i < kCatCount; i++) {
				if (i != 0) t.line2 += ' ';
				t.line2 += kCatLetters[i];
				t.line2 += m_mask[i] == 100 ? '+' : m_mask[i] == 0 ? '-' : '~';
			}
		} else if (!DiffersFromLoaded()) {
			t.line2 = "same as the level in memory";
		} else if (pairs != 0) {
			t.line2 = "hidden now; re-enter the level to apply fully";
		} else {
			t.line2 = "applies when you re-enter the level";
		}
		return t;
	}

private:
	std::vector<Preset> m_presets;
	Mask                m_mask   = kAllOn;
	Mask                m_loaded = kAllOn;
	uint32_t            m_seq    = 0;
};

// ---- runtime half (sceneSkip.cpp) ----
inline ActiveSet g_active; // read by the draw path

// The draw path: one relaxed load when nothing is active.
[[nodiscard]] inline bool Active() {
	return g_active.Count() != 0;
}
[[nodiscard]] inline bool Skips(uint64_t vs_hash, uint64_t ps_hash, bool ps_known) {
	return g_active.Contains(vs_hash, ps_hash, ps_known);
}

[[nodiscard]] bool KeysEnabled() noexcept; // KYTY_SCENE_KEYS=1 (read once)
// Window thread: a key event. true = consumed (keys on and a scenery key); false = the game sees it as before.
bool HandleKey(Key key, bool shift, bool pressed);
// Command-processor thread, every guest flip: reads the scene, publishes the pairs, counts. Cheap when the feature is off.
void OnCpFlip();
// The toast (systemOverlay): active until it fades; the text is copied under a lock.
[[nodiscard]] bool     ToastWanted() noexcept;
[[nodiscard]] uint32_t ToastRevision() noexcept;
bool                   ToastView(char* line1, size_t cap1, char* line2, size_t cap2, float& alpha);
// The F2 overlay row text ("F- G+ M+ O+ (balanced)"); empty when the keys are off.
[[nodiscard]] std::string OverlayRow();
// The F1 panel (systemOverlay): open until it hides itself (kPanelMs) or F1 again. Lines are copied into the caller's fixed buffer.
struct PanelText {
	char    text[120];
	uint8_t kind; // PanelKind
};
inline constexpr size_t kPanelMaxLines = 16;
[[nodiscard]] bool PanelWanted() noexcept;
size_t             PanelView(PanelText* out, size_t cap, float& alpha); // lines written; 0 = closed

} // namespace Libs::Graphics::SceneSkip
