#pragma once

// nps5 levelprep: the shader level database (pipelines.kylvdb), header only so
// tests/LevelDbTests.cpp needs no emulator, no zstd and no game.
//
// Which scenes ("levels": hub, menus, a level of the game) USE which pipeline, by the pipeline
// list's key (PipelineList::Pipeline::key) and which shader, by its guest hash
// (PipelineList::Shader::hash). Keys, hashes and scene names only: no shader code, nothing a
// translator made, so a file may ship next to the list.
//
// File: magic "KYLVDB1\0", u32 format version, u32 payload size, u64 FNV-1a of the payload, then
// the payload (little endian): info strings, scenes, sources, pipeline rows, shader rows. A newer
// version, a bad checksum, a truncated or over-long field = rejected, never partly read.
//
// A row is (scene, flags, runs, first_ms): flags say how it is known (first seen = the journal's
// scene tag, used = a sight log saw it in the scene, inferred = every stage shader of the pipeline
// was seen in the scene), runs = how many recorded runs saw it (frequency = runs / scene runs),
// first_ms = the earliest offset from the scene open (0xFFFFFFFF unknown). Merge is commutative,
// associative and idempotent: a source (a journal, log or list folded in, named by a 64 bit id) is
// counted once, however often it is merged.

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace Libs::Graphics::LevelDb {

inline constexpr char     Magic[8]      = {'K', 'Y', 'L', 'V', 'D', 'B', '1', '\0'};
inline constexpr uint32_t FormatVersion = 1;
inline constexpr uint32_t UnknownMs     = 0xFFFFFFFFu;
inline constexpr uint16_t UnknownOrder  = 0xFFFFu;

enum RowFlag : uint8_t {
	FirstSeen = 1, // the journal / list tags the pipeline with the scene (created there first)
	Used      = 2, // a log saw it (shader level: a sight line) in the scene
	Inferred  = 4, // every stage shader of the pipeline was seen in the scene
	PreOpen   = 8, // needed during the load, before the scene's first picture
	Live      = 16 // was compiled live at a draw (it was missing from a prepare)
};

enum class SceneClass : uint8_t {
	Unknown  = 0,
	Boot     = 1,
	Menu     = 2,
	Hub      = 3,
	Cutscene = 4,
	Level    = 5,
	Bonus    = 6
};

[[nodiscard]] inline const char* SceneClassName(SceneClass c) {
	switch (c) {
		case SceneClass::Boot: return "boot";
		case SceneClass::Menu: return "menu";
		case SceneClass::Hub: return "hub";
		case SceneClass::Cutscene: return "cutscene";
		case SceneClass::Level: return "level";
		case SceneClass::Bonus: return "bonus";
		default: return "unknown";
	}
}
[[nodiscard]] inline SceneClass SceneClassFromName(std::string_view name) {
	for (uint8_t i = 1; i <= 6; i++) {
		if (name == SceneClassName(static_cast<SceneClass>(i))) return static_cast<SceneClass>(i);
	}
	return SceneClass::Unknown;
}
// Scenes that are always part of tier 1: the game's own start path (boot, menus, hub, cutscenes).
[[nodiscard]] inline bool IsAlwaysClass(SceneClass c) {
	return c == SceneClass::Boot || c == SceneClass::Menu || c == SceneClass::Hub ||
	       c == SceneClass::Cutscene;
}

struct Scene {
	std::string name;
	SceneClass  cls = SceneClass::Unknown;
	uint16_t order  = UnknownOrder; // story order of a level (1 = first level); UnknownOrder: none
	uint16_t runs   = 0;            // recorded runs that visited the scene
	std::string save_id;            // the id of the level in the game's save data, empty: none
};

struct Row {
	uint16_t scene    = 0; // index into File::scenes
	uint8_t  flags    = 0;
	uint16_t runs     = 0;
	uint32_t first_ms = UnknownMs;
};

struct File {
	std::string                          title_id, producer;
	std::vector<Scene>                   scenes;
	std::set<uint64_t>                   sources;
	std::map<uint64_t, std::vector<Row>> pipelines; // list key -> rows, sorted by scene
	std::map<uint64_t, std::vector<Row>> shaders;   // guest hash -> rows
};

// ---- scenes and rows
// -----------------------------------------------------------------------------------------------

[[nodiscard]] inline int FindScene(const File& f, std::string_view name) {
	for (size_t i = 0; i < f.scenes.size(); i++) {
		if (f.scenes[i].name == name) return static_cast<int>(i);
	}
	return -1;
}

inline uint16_t AddScene(File& f, std::string_view name) {
	const int found = FindScene(f, name);
	if (found >= 0) return static_cast<uint16_t>(found);
	Scene s;
	s.name = std::string(name);
	f.scenes.push_back(std::move(s));
	return static_cast<uint16_t>(f.scenes.size() - 1);
}

namespace detail {
inline Row* RowFor(std::vector<Row>& rows, uint16_t scene) {
	auto it = std::lower_bound(rows.begin(), rows.end(), scene,
	                           [](const Row& r, uint16_t s) { return r.scene < s; });
	if (it != rows.end() && it->scene == scene) return &*it;
	it = rows.insert(it, Row {scene, 0, 0, UnknownMs});
	return &*it;
}
inline void Fold(Row& dst, const Row& src, bool add_runs) {
	dst.flags |= src.flags;
	if (add_runs)
		dst.runs =
		    static_cast<uint16_t>(std::min<uint32_t>(0xFFFFu, uint32_t {dst.runs} + src.runs));
	else
		dst.runs = std::max(dst.runs, src.runs);
	dst.first_ms = std::min(dst.first_ms, src.first_ms);
}
} // namespace detail

// Notes that `key` (pipeline) is used in `scene`; one run counts once per call.
inline void NotePipeline(File& f, uint64_t key, std::string_view scene, uint8_t flags,
                         uint32_t first_ms = UnknownMs, uint16_t runs = 1) {
	auto* row = detail::RowFor(f.pipelines[key], AddScene(f, scene));
	detail::Fold(*row, Row {0, flags, runs, first_ms}, true);
}
inline void NoteShader(File& f, uint64_t hash, std::string_view scene, uint8_t flags,
                       uint32_t first_ms = UnknownMs, uint16_t runs = 1) {
	auto* row = detail::RowFor(f.shaders[hash], AddScene(f, scene));
	detail::Fold(*row, Row {0, flags, runs, first_ms}, true);
}

// A pipeline and the guest hashes of its stages' shaders (the list resolved: stage -> permutation
// -> shader).
struct PipelineShaders {
	uint64_t              key = 0;
	std::vector<uint64_t> shader_hashes;
};

// Inference: a pipeline is used in a scene when every one of its stage shaders is used there. This
// is a superset (state variants of a used shader set), the safe direction for a prepare. Returns
// the rows added or changed.
inline size_t Infer(File& f, const std::vector<PipelineShaders>& pipelines) {
	size_t changed = 0;
	for (const auto& p: pipelines) {
		if (p.shader_hashes.empty()) continue;
		std::map<uint16_t, uint32_t> hits; // scene -> stages seen
		std::map<uint16_t, uint32_t> first;
		bool                         all_known = true;
		for (const auto hash: p.shader_hashes) {
			const auto it = f.shaders.find(hash);
			if (it == f.shaders.end()) {
				all_known = false;
				break;
			}
			for (const auto& r: it->second) {
				if ((r.flags & (Used | PreOpen)) == 0) continue;
				hits[r.scene]++;
				first[r.scene] = std::max(first.count(r.scene) ? first[r.scene] : 0u,
				                          r.first_ms == UnknownMs ? 0u : r.first_ms);
			}
		}
		if (!all_known) continue;
		for (const auto& [scene, n]: hits) {
			if (n < p.shader_hashes.size()) continue;
			auto&         rows   = f.pipelines[p.key];
			auto*         row    = detail::RowFor(rows, scene);
			const uint8_t before = row->flags;
			row->flags |= Inferred;
			if (row->runs == 0) row->runs = 1;
			if (row->first_ms == UnknownMs && first[scene] != 0) row->first_ms = first[scene];
			if (row->flags != before) changed++;
		}
	}
	return changed;
}

// A pipeline whose twin (same key, another snapshot layout) is tagged is the same pipeline: callers
// fold twins by key before they note anything, so nothing here is needed beyond keys.

// ---- queries
// -------------------------------------------------------------------------------------------------------

[[nodiscard]] inline std::vector<uint64_t> KeysInScene(const File& f, std::string_view scene) {
	std::vector<uint64_t> out;
	const int             index = FindScene(f, scene);
	if (index < 0) return out;
	for (const auto& [key, rows]: f.pipelines) {
		for (const auto& r: rows) {
			if (r.scene == index) {
				out.push_back(key);
				break;
			}
		}
	}
	return out;
}

[[nodiscard]] inline std::vector<std::string> ScenesOfKey(const File& f, uint64_t key) {
	std::vector<std::string> out;
	const auto               it = f.pipelines.find(key);
	if (it == f.pipelines.end()) return out;
	for (const auto& r: it->second) {
		if (r.scene < f.scenes.size()) out.push_back(f.scenes[r.scene].name);
	}
	return out;
}

// Keys with no scene at all (recorded before any scene opened, or never attributed).
[[nodiscard]] inline std::vector<uint64_t> UnattributedKeys(const File&                  f,
                                                            const std::vector<uint64_t>& all_keys) {
	std::vector<uint64_t> out;
	for (const auto key: all_keys) {
		const auto it = f.pipelines.find(key);
		if (it == f.pipelines.end() || it->second.empty()) out.push_back(key);
	}
	return out;
}

// ---- merge
// ---------------------------------------------------------------------------------------------------------

// Folds `src` into `dst`. A source id both files carry is counted once: when every source of `src`
// is already in `dst` nothing changes; with a partial overlap flags and first times fold, the run
// counts do not add.
inline void Merge(File& dst, const File& src) {
	bool all_known = !src.sources.empty();
	bool any_known = false;
	for (const auto id: src.sources) {
		if (dst.sources.count(id))
			any_known = true;
		else
			all_known = false;
	}
	if (all_known) return;
	const bool add_runs = !any_known;
	if (dst.title_id.empty()) dst.title_id = src.title_id;
	if (dst.producer.empty()) dst.producer = src.producer;
	std::vector<uint16_t> map(src.scenes.size());
	for (size_t i = 0; i < src.scenes.size(); i++) {
		const auto& s = src.scenes[i];
		const auto  d = AddScene(dst, s.name);
		map[i]        = d;
		auto& ds      = dst.scenes[d];
		if (ds.cls == SceneClass::Unknown) ds.cls = s.cls;
		if (ds.order == UnknownOrder) ds.order = s.order;
		if (ds.save_id.empty()) ds.save_id = s.save_id;
		ds.runs =
		    add_runs
		        ? static_cast<uint16_t>(std::min<uint32_t>(0xFFFFu, uint32_t {ds.runs} + s.runs))
		        : std::max(ds.runs, s.runs);
	}
	const auto fold = [&](std::map<uint64_t, std::vector<Row>>&       to,
	                      const std::map<uint64_t, std::vector<Row>>& from) {
		for (const auto& [key, rows]: from) {
			auto& out = to[key];
			for (const auto& r: rows) {
				if (r.scene >= map.size()) continue;
				detail::Fold(*detail::RowFor(out, map[r.scene]), r, add_runs);
			}
		}
	};
	fold(dst.pipelines, src.pipelines);
	fold(dst.shaders, src.shaders);
	dst.sources.insert(src.sources.begin(), src.sources.end());
}

// A stable 64 bit id of a source's bytes (a journal, log or list): FNV-1a.
[[nodiscard]] inline uint64_t Fnv64(const void* data, size_t size,
                                    uint64_t seed = 0xcbf29ce484222325ull) {
	auto h = seed;
	auto p = static_cast<const uint8_t*>(data);
	for (size_t i = 0; i < size; i++) {
		h ^= p[i];
		h *= 0x100000001b3ull;
	}
	return h;
}

// ---- file format
// ---------------------------------------------------------------------------------------------------

namespace detail {
struct Writer {
	std::vector<uint8_t> out;
	void                 U8(uint8_t v) { out.push_back(v); }
	void                 U16(uint16_t v) { U8(v & 0xff), U8(v >> 8); }
	void                 U32(uint32_t v) { U16(v & 0xffff), U16(v >> 16); }
	void U64(uint64_t v) { U32(static_cast<uint32_t>(v)), U32(static_cast<uint32_t>(v >> 32)); }
	void Str(const std::string& s) {
		U16(static_cast<uint16_t>(std::min<size_t>(s.size(), 0xffff)));
		out.insert(out.end(), s.begin(), s.begin() + std::min<size_t>(s.size(), 0xffff));
	}
};
struct Reader {
	const uint8_t* p;
	size_t         left;
	bool           ok = true;
	uint64_t       Take(unsigned bytes) {
		if (left < bytes) {
			ok = false;
			return 0;
		}
		uint64_t v = 0;
		for (unsigned i = 0; i < bytes; i++)
			v |= static_cast<uint64_t>(p[i]) << (8 * i);
		p += bytes, left -= bytes;
		return v;
	}
	uint8_t     U8() { return static_cast<uint8_t>(Take(1)); }
	uint16_t    U16() { return static_cast<uint16_t>(Take(2)); }
	uint32_t    U32() { return static_cast<uint32_t>(Take(4)); }
	uint64_t    U64() { return Take(8); }
	std::string Str() {
		const auto n = U16();
		if (!ok || left < n) {
			ok = false;
			return {};
		}
		std::string s(reinterpret_cast<const char*>(p), n);
		p += n, left -= n;
		return s;
	}
};
inline void PutRows(Writer& w, const std::map<uint64_t, std::vector<Row>>& m) {
	w.U32(static_cast<uint32_t>(m.size()));
	for (const auto& [key, rows]: m) {
		w.U64(key);
		w.U16(static_cast<uint16_t>(rows.size()));
		for (const auto& r: rows)
			w.U16(r.scene), w.U8(r.flags), w.U16(r.runs), w.U32(r.first_ms);
	}
}
inline bool GetRows(Reader& r, std::map<uint64_t, std::vector<Row>>& m, size_t scene_count) {
	const auto n = r.U32();
	for (uint32_t i = 0; i < n && r.ok; i++) {
		const auto key   = r.U64();
		const auto count = r.U16();
		if (r.left < size_t {count} * 9) return false;
		auto& rows = m[key];
		for (uint16_t j = 0; j < count; j++) {
			Row row;
			row.scene    = r.U16();
			row.flags    = r.U8();
			row.runs     = r.U16();
			row.first_ms = r.U32();
			if (row.scene >= scene_count) return false;
			rows.push_back(row);
		}
		if (!std::is_sorted(rows.begin(), rows.end(),
		                    [](const Row& a, const Row& b) { return a.scene < b.scene; }))
			return false;
	}
	return r.ok;
}
} // namespace detail

[[nodiscard]] inline std::vector<uint8_t> Encode(const File& f) {
	detail::Writer w;
	w.Str(f.title_id);
	w.Str(f.producer);
	w.U32(static_cast<uint32_t>(f.scenes.size()));
	for (const auto& s: f.scenes) {
		w.Str(s.name);
		w.U8(static_cast<uint8_t>(s.cls));
		w.U16(s.order);
		w.U16(s.runs);
		w.Str(s.save_id);
	}
	w.U32(static_cast<uint32_t>(f.sources.size()));
	for (const auto id: f.sources)
		w.U64(id);
	detail::PutRows(w, f.pipelines);
	detail::PutRows(w, f.shaders);
	detail::Writer h;
	h.out.insert(h.out.end(), Magic, Magic + 8);
	h.U32(FormatVersion);
	h.U32(static_cast<uint32_t>(w.out.size()));
	h.U64(Fnv64(w.out.data(), w.out.size()));
	h.out.insert(h.out.end(), w.out.begin(), w.out.end());
	return h.out;
}

// False (and `error`) for another format, a newer version or damage; `f` is then empty.
[[nodiscard]] inline bool Decode(const uint8_t* bytes, size_t size, File& f,
                                 std::string* error = nullptr) {
	f               = File {};
	const auto fail = [&](const char* why) {
		if (error) *error = why;
		f = File {};
		return false;
	};
	if (size < 24 || std::memcmp(bytes, Magic, 8) != 0) return fail("not a KYLVDB1 file");
	detail::Reader h {bytes + 8, size - 8};
	const auto     version = h.U32();
	const auto     length  = h.U32();
	const auto     sum     = h.U64();
	if (version == 0 || version > FormatVersion) return fail("unsupported version");
	if (h.left != length) return fail("size mismatch");
	if (Fnv64(h.p, h.left) != sum) return fail("checksum mismatch");
	detail::Reader r {h.p, h.left};
	f.title_id        = r.Str();
	f.producer        = r.Str();
	const auto scenes = r.U32();
	if (scenes > 0xffff || r.left < size_t {scenes} * 9) return fail("bad scene count");
	for (uint32_t i = 0; i < scenes && r.ok; i++) {
		Scene s;
		s.name    = r.Str();
		s.cls     = static_cast<SceneClass>(r.U8());
		s.order   = r.U16();
		s.runs    = r.U16();
		s.save_id = r.Str();
		f.scenes.push_back(std::move(s));
	}
	const auto sources = r.U32();
	if (!r.ok || r.left < size_t {sources} * 8) return fail("bad source count");
	for (uint32_t i = 0; i < sources; i++)
		f.sources.insert(r.U64());
	if (!detail::GetRows(r, f.pipelines, f.scenes.size()) ||
	    !detail::GetRows(r, f.shaders, f.scenes.size()) || !r.ok || r.left != 0)
		return fail("damaged rows");
	return true;
}
[[nodiscard]] inline bool Decode(const std::vector<uint8_t>& bytes, File& f,
                                 std::string* error = nullptr) {
	return Decode(bytes.data(), bytes.size(), f, error);
}

// ---- scenes.tsv
// ----------------------------------------------------------------------------------------------------

// "name<TAB>class<TAB>order<TAB>save_id" per line ('#' comments, blank lines ignored; order and
// save_id may be "-"). Returns false (and `error`) on a bad row, a duplicate scene name or a
// duplicate story order of two levels.
inline bool ParseScenesTsv(std::string_view text, std::vector<Scene>& out,
                           std::string* error = nullptr) {
	out.clear();
	std::set<std::string> names;
	std::set<uint16_t>    orders;
	size_t                pos  = 0;
	int                   line = 0;
	const auto            fail = [&](const std::string& why) {
		if (error) *error = "line " + std::to_string(line) + ": " + why;
		out.clear();
		return false;
	};
	while (pos <= text.size()) {
		auto end = text.find('\n', pos);
		if (end == std::string_view::npos) end = text.size();
		auto row = text.substr(pos, end - pos);
		pos      = end + 1;
		line++;
		while (!row.empty() && (row.back() == '\r' || row.back() == ' '))
			row.remove_suffix(1);
		if (row.empty() || row.front() == '#') continue;
		std::vector<std::string> cols;
		size_t                   c = 0;
		while (true) {
			const auto tab = row.find('\t', c);
			cols.emplace_back(
			    row.substr(c, tab == std::string_view::npos ? std::string_view::npos : tab - c));
			if (tab == std::string_view::npos) break;
			c = tab + 1;
		}
		if (cols.size() < 2 || cols.size() > 4 || cols[0].empty())
			return fail("expected name, class[, order[, save_id]]");
		Scene s;
		s.name = cols[0];
		s.cls  = SceneClassFromName(cols[1]);
		if (s.cls == SceneClass::Unknown) return fail("unknown class '" + cols[1] + "'");
		if (cols.size() > 2 && cols[2] != "-" && !cols[2].empty()) {
			if (cols[2].find_first_not_of("0123456789") != std::string::npos || cols[2].size() > 4)
				return fail("bad order");
			const auto o = std::stoul(cols[2]);
			if (o == 0 || o >= UnknownOrder) return fail("bad order");
			s.order = static_cast<uint16_t>(o);
			if (!orders.insert(s.order).second) return fail("duplicate order " + cols[2]);
		}
		if (cols.size() > 3 && cols[3] != "-") s.save_id = cols[3];
		if (!names.insert(s.name).second) return fail("duplicate scene '" + s.name + "'");
		out.push_back(std::move(s));
	}
	return true;
}

// Applies scene metadata (class, order, save id) to the file's scenes, adding the scenes it does
// not have.
inline void ApplyScenes(File& f, const std::vector<Scene>& meta) {
	for (const auto& m: meta) {
		auto& s   = f.scenes[AddScene(f, m.name)];
		s.cls     = m.cls;
		s.order   = m.order;
		s.save_id = m.save_id;
	}
}

} // namespace Libs::Graphics::LevelDb
