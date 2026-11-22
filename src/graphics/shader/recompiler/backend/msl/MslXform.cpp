#include "graphics/shader/recompiler/backend/msl/MslXform.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <string_view>
#include <utility>

namespace Libs::Graphics::ShaderRecompiler::Msl::Xf {

namespace {

constexpr Info Registry[Count] = {
    {"vs_root_int", 1, ShaderType::Vertex, "structural",
     "vertex fetch: root buffer through an integer address (constant-program hoisting)"},
    {"vs_fetch_select", 1, ShaderType::Vertex, "structural",
     "vertex fetch: unconditional attribute load, then the out-of-range select"},
    {"vs_no_barrier", 1, ShaderType::Vertex, "structural",
     "vertex function: no device-flush barrier for S_BARRIER"},
    {"bounds_u32", 1, ShaderType::Unknown, "semantic-exact",
     "buffer bounds check in 32 bits (no 64-bit temporary per access)"},
    {"minmax_select", 1, ShaderType::Unknown, "semantic-exact",
     "min/max/med3 helpers with select() instead of ?: (no branch per call)"},
    {"mipstats_plain_load", 1, ShaderType::Pixel, "semantic-exact",
     "MipStats record: plain load in the pre-check before atomic_fetch_min"},
    {"load_group", 1, ShaderType::Unknown, "semantic-exact",
     "dword reads with a common dynamic base: one range check + packed vector loads"},
    {"movrel_switch", 1, ShaderType::Unknown, "semantic-exact",
     "V_MOVRELS select chain (>= 4 links on one M0) as one switch on M0"},
    {"movrel_tree", 1, ShaderType::Unknown, "semantic-exact",
     "V_MOVRELS select chain (>= 4 links on one M0) as a balanced compare tree"},
    {"load_select", 1, ShaderType::Unknown, "structural",
     "checked buffer load: unconditional load from (in range ? address : root), then the select"},
    {"vs_load_group", 1, ShaderType::Vertex, "semantic-exact",
     "vertex: dword reads with a common dynamic base: one range check + packed vector loads"},
    {"vs_const_load", 1, ShaderType::Vertex, "preconditioned",
     "vertex: read-only buffer loads from the constant address space (alias-guarded after a "
     "store/atomic)"},
    {"loop_cont", 1, ShaderType::Unknown, "structural",
     "loop with a side-effect-free continue block: no do/flag wrapper, block printed per edge"},
    {"bounds_lim", 1, ShaderType::Unknown, "semantic-exact",
     "buffer bounds check as one compare against a uniform per-buffer limit"},
    {"guard_exit", 1, ShaderType::Unknown, "preconditioned",
     "gpuguard loop budget checked inside the loop's own exit test (one exit edge)"},
};

constexpr uint32_t AllMask = (1u << Count) - 1u;
// `all` in a KYTY_MSL_XF list / table row: the transforms up to load_select. The foltrans ids are
// per-key choices (mode tokens vgroup / cload, or their xf= bits) and never part of `all`.
constexpr uint32_t ListAllMask =
    AllMask & ~((1u << VsLoadGroup) | (1u << VsConstLoad) | (1u << LoopCont) | (1u << BoundsLim) |
                (1u << GuardExit));

thread_local uint32_t   g_active = 0;
thread_local ShaderType g_stage  = ShaderType::Unknown;

bool Truthy(const char* e) {
	return e != nullptr && *e != 0 && std::strcmp(e, "0") != 0;
}

// "all" | "none" | "-" | ids separated by ',' ; "-id" removes. Unknown ids are reported once and
// ignored (a newer table read by an older emitter must not silently change other transforms).
uint32_t ParseList(std::string_view s, uint32_t mask) {
	while (!s.empty()) {
		const size_t     c = s.find(',');
		std::string_view t = s.substr(0, c);
		s                  = c == std::string_view::npos ? std::string_view {} : s.substr(c + 1);
		while (!t.empty() && (t.front() == ' ' || t.front() == '\t'))
			t.remove_prefix(1);
		while (!t.empty() && (t.back() == ' ' || t.back() == '\t' || t.back() == '\r'))
			t.remove_suffix(1);
		if (const size_t at = t.find('@'); at != std::string_view::npos)
			t = t.substr(0, at); // "id@version" as ActiveList prints it
		if (t.empty() || t == "none" || t == "-") continue;
		if (t == "all") {
			mask |= ListAllMask;
			continue;
		}
		const bool remove = t.front() == '-';
		if (remove) t.remove_prefix(1);
		bool found = false;
		for (uint32_t i = 0; i < Count; i++) {
			if (t == Registry[i].id) {
				mask  = remove ? mask & ~(1u << i) : mask | (1u << i);
				found = true;
			}
		}
		if (!found)
			std::fprintf(stderr, "kyty msl xf: unknown transform '%.*s' ignored\n",
			             static_cast<int>(t.size()), t.data());
	}
	return mask;
}

uint32_t BaseMask() {
	static const uint32_t base = [] {
		uint32_t m = 0;
		if (const char* e = std::getenv("KYTY_MSL_XF"); e != nullptr) m = ParseList(e, m);
		if (Truthy(std::getenv("KYTY_MSL_VS_PERF")))
			m |= (1u << VsRootInt) | (1u << VsFetchSelect) | (1u << VsNoBarrier);
		return m;
	}();
	return base;
}

char StageChar(ShaderType s) {
	switch (s) {
		case ShaderType::Vertex: return 'v';
		case ShaderType::Pixel: return 'p';
		case ShaderType::Compute: return 'c';
		default: return '?';
	}
}

// (stage char, guest hash) -> mask, from KYTY_MSL_XF_TABLE.
const std::map<std::pair<char, uint64_t>, uint32_t>& Table() {
	static const std::map<std::pair<char, uint64_t>, uint32_t> table = [] {
		std::map<std::pair<char, uint64_t>, uint32_t> t;
		const char*                                   path = std::getenv("KYTY_MSL_XF_TABLE");
		if (path == nullptr || *path == 0) return t;
		std::FILE* fp = std::fopen(path, "r");
		if (fp == nullptr) {
			std::fprintf(stderr, "kyty msl xf: cannot read table %s\n", path);
			return t;
		}
		char line[1024];
		while (std::fgets(line, sizeof(line), fp) != nullptr) {
			char stage[8] = {}, hash[32] = {}, list[900] = {};
			if (line[0] == '#' || std::sscanf(line, "%7s %31s %899s", stage, hash, list) < 3)
				continue;
			char*          end = nullptr;
			const uint64_t h   = std::strtoull(hash, &end, 16);
			if (end == hash || *end != 0) continue;
			t[{stage[0], h}] = ParseList(list, 0);
		}
		std::fclose(fp);
		return t;
	}();
	return table;
}

} // namespace

const Info& InfoOf(Id id) {
	return Registry[id < Count ? id : 0];
}

void Begin(ShaderType stage, uint64_t shader_hash) {
	g_stage       = stage;
	g_active      = BaseMask();
	if (EnvOn("KYTY_MSL_XF_LSEL")) // mode token `lsel`; read per call
		g_active |= 1u << LoadSelect;
	if (EnvOn("KYTY_MSL_XF_VGROUP")) // mode token `vgroup`; read per call
		g_active |= 1u << VsLoadGroup;
	if (EnvOn("KYTY_MSL_XF_CLOAD")) // mode token `cload`; read per call
		g_active |= 1u << VsConstLoad;
	if (EnvOn("KYTY_MSL_XF_LCONT")) // mode token `lcont`; read per call
		g_active |= 1u << LoopCont;
	if (EnvOn("KYTY_MSL_XF_BLIM")) // mode token `blim`; read per call
		g_active |= 1u << BoundsLim;
	if (EnvOn("KYTY_MSL_XF_GEXIT")) // mode token `gexit`; read per call
		g_active |= 1u << GuardExit;
	const auto& t = Table();
	if (const auto it = t.find({StageChar(stage), shader_hash}); it != t.end())
		g_active = it->second;
	for (uint32_t i = 0; i < Count; i++) {
		if (Registry[i].stage != ShaderType::Unknown && Registry[i].stage != stage)
			g_active &= ~(1u << i);
	}
}

bool On(Id id) {
	return id < Count && (g_active & (1u << id)) != 0;
}

std::string ActiveList() {
	std::string s;
	for (uint32_t i = 0; i < Count; i++) {
		if ((g_active & (1u << i)) == 0) continue;
		if (!s.empty()) s += ',';
		s += Registry[i].id;
		s += '@';
		s += std::to_string(Registry[i].version);
	}
	return s;
}

std::string Catalogue() {
	std::string s;
	for (uint32_t i = 0; i < Count; i++) {
		s += Registry[i].id;
		s += '@';
		s += std::to_string(Registry[i].version);
		s += ' ';
		if (Registry[i].stage == ShaderType::Unknown) {
			s += "any ";
		} else {
			s += StageChar(Registry[i].stage);
			s += "s ";
		}
		s += Registry[i].cls;
		s += "  ";
		s += Registry[i].doc;
		s += '\n';
	}
	return s;
}

namespace {
bool SetOnce(const char* name, const std::string& value, std::string& error) {
	const char* had = std::getenv(name);
	if (had != nullptr && value != had) {
		error = std::string(name) + " is already set to '" + had + "'";
		return false;
	}
	setenv(name, value.c_str(), 1);
	return true;
}
} // namespace

bool EnvOn(const char* name) {
	return Truthy(std::getenv(name));
}

bool ApplyMode(std::string_view mode, std::string& error) {
	// review L1: every switch that changes the emitted text, whether this mode sets it or not
	for (const char* v:
	     {"KYTY_MSL_FIX6", "KYTY_MSL_VS_PERF", "KYTY_MSL_GFX_SPLIT", "KYTY_MSL_GFX_REASSOC",
	      "KYTY_MSL_GFX_MINMAX", "KYTY_MSL_GFX_MATH", "KYTY_MSL_GFX_FTZ", "KYTY_MSL_XF",
	      "KYTY_MSL_XF_LSEL", "KYTY_MSL_XF_VGROUP", "KYTY_MSL_XF_CLOAD", "KYTY_MSL_XF_LCONT",
	      "KYTY_MSL_XF_BLIM", "KYTY_MSL_XF_GEXIT", "KYTY_MSL_VS_MODES", "KYTY_MSL_XF_TABLE", "KYTY_MSL_LOADGROUP",
	      "KYTY_MSL_LOADGROUP_MUTANT"}) {
		if (std::getenv(v) != nullptr) {
			error = std::string(v) +
			        " is set in the environment; --mode does not mix with hand-set switches";
			return false;
		}
	}
	if (mode.empty() || mode == "base") return true;
	{ // review L2: perfr only switches off parts of fix6
		bool fix6 = false, perfr = false;
		for (size_t a = 0; a <= mode.size();) {
			const size_t e = std::min(mode.find('+', a), mode.size());
			fix6 |= mode.substr(a, e - a) == "fix6";
			perfr |= mode.substr(a, e - a) == "perfr";
			a = e + 1;
		}
		if (perfr && !fix6) {
			error = "perfr needs fix6 (it switches off parts of the fix6 chain; alone it is the "
			        "base text)";
			return false;
		}
	}
	std::map<std::string, bool> seen;
	size_t                      at = 0;
	while (at <= mode.size()) {
		const size_t      end = std::min(mode.find('+', at), mode.size());
		const std::string tok(mode.substr(at, end - at));
		at                    = end + 1;
		const auto        eq  = tok.find('=');
		const std::string key = tok.substr(0, eq),
		                  val = eq == std::string::npos ? "" : tok.substr(eq + 1);
		if (!seen.emplace(key, true).second) {
			error = "mode token '" + key + "' given twice";
			return false;
		}
		bool ok = true;
		if (key == "fix6" && val.empty()) {
			ok = SetOnce("KYTY_MSL_FIX6", "1", error);
		} else if (key == "vsperf" && val.empty()) {
			ok = SetOnce("KYTY_MSL_VS_PERF", "1", error);
		} else if (key == "lsel" && val.empty()) {
			ok = SetOnce("KYTY_MSL_XF_LSEL", "1", error);
		} else if (key == "vgroup" && val.empty()) {
			ok = SetOnce("KYTY_MSL_XF_VGROUP", "1", error);
		} else if (key == "cload" && val.empty()) {
			ok = SetOnce("KYTY_MSL_XF_CLOAD", "1", error);
		} else if (key == "lcont" && val.empty()) {
			ok = SetOnce("KYTY_MSL_XF_LCONT", "1", error);
		} else if (key == "blim" && val.empty()) {
			ok = SetOnce("KYTY_MSL_XF_BLIM", "1", error);
		} else if (key == "gexit" && val.empty()) {
			ok = SetOnce("KYTY_MSL_XF_GEXIT", "1", error);
		} else if (key == "vsmodes" && val.empty()) {
			ok = SetOnce("KYTY_MSL_VS_MODES", "1", error);
		} else if (key == "perfr" && val.empty()) {
			ok = SetOnce("KYTY_MSL_GFX_SPLIT", "0", error) &&
			     SetOnce("KYTY_MSL_GFX_REASSOC", "0", error) &&
			     SetOnce("KYTY_MSL_GFX_MINMAX", "0", error);
		} else if (key == "gfxmath" && (val == "safe" || val == "relaxed" || val == "fast")) {
			ok = SetOnce("KYTY_MSL_GFX_MATH", val, error);
		} else if (key == "xf" && !val.empty() && val.size() <= 8 &&
		           val.find_first_not_of("0123456789abcdef") == std::string::npos) {
			const uint32_t mask = static_cast<uint32_t>(std::strtoul(val.c_str(), nullptr, 16));
			if ((mask & ~AllMask) != 0u) {
				error = "xf=" + val + " names a transform this emitter does not have";
				return false;
			}
			std::string list;
			for (uint32_t i = 0; i < Count; i++)
				if ((mask >> i) & 1u) list += std::string(list.empty() ? "" : ",") + Registry[i].id;
			ok = SetOnce("KYTY_MSL_XF", list.empty() ? "none" : list, error);
		} else {
			error = "unknown mode token '" + tok + "'";
			return false;
		}
		if (!ok) return false;
	}
	return true;
}

std::string ModeCatalogue() {
	std::string t =
	    "base\tthe v1.1.2 / v1.2 text\n"
	    "fix6\tKK-mimicry chain: KK-exact min/max, NIR re-association + statement split, bcover\n"
	    "perfr\trelaxed pixel math without KK mimicry (GFX_SPLIT/REASSOC/MINMAX=0)\n"
	    "vsperf\tvertex fetch transforms + varying narrowing (KYTY_MSL_VS_PERF=1)\n"
	    "gfxmath=safe|relaxed|fast\tfragment math mode header (KYTY_MSL_GFX_MATH)\n"
	    "lsel\tchecked buffer loads as unconditional load + select (KYTY_MSL_XF_LSEL=1 = transform "
	    "load_select)\n"
	    "vgroup\tvertex: grouped buffer reads, one range check + packed loads "
	    "(KYTY_MSL_XF_VGROUP=1 = transform vs_load_group)\n"
	    "cload\tvertex: read-only buffer loads from the constant address space "
	    "(KYTY_MSL_XF_CLOAD=1 = transform vs_const_load)\n"
	    "lcont\tloops with a side-effect-free continue block without the do/flag wrapper "
	    "(KYTY_MSL_XF_LCONT=1 = transform loop_cont)\n"
	    "blim\tbuffer bounds check as one compare against a uniform limit (KYTY_MSL_XF_BLIM=1 = "
	    "transform bounds_lim)\n"
	    "gexit\tgpuguard loop budget checked inside the loop's own exit test (KYTY_MSL_XF_GEXIT=1 "
	    "= transform guard_exit)\n"
	    "vsmodes\tvertex: KK's per-op float modes (position slice safe, rest relaxed, MAD as two statements; KYTY_MSL_VS_MODES=1)\n"
	    "xf=HEX\ttransform bit mask in registry order:\n";
	for (uint32_t i = 0; i < Count; i++) {
		char line[96];
		std::snprintf(line, sizeof line, "  %x\t%s@%u\n", 1u << i, Registry[i].id,
		              Registry[i].version);
		t += line;
	}
	return t;
}

} // namespace Libs::Graphics::ShaderRecompiler::Msl::Xf
