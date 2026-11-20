#include "graphics/shader/recompiler/backend/msl/MslXform.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
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
};

constexpr uint32_t AllMask = (1u << Count) - 1u;

thread_local uint32_t  g_active = 0;
thread_local ShaderType g_stage = ShaderType::Unknown;

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
		if (t.empty() || t == "none" || t == "-")
			continue;
		if (t == "all") {
			mask |= AllMask;
			continue;
		}
		const bool remove = t.front() == '-';
		if (remove)
			t.remove_prefix(1);
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
		if (const char* e = std::getenv("KYTY_MSL_XF"); e != nullptr)
			m = ParseList(e, m);
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
		if (path == nullptr || *path == 0)
			return t;
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
			if (end == hash || *end != 0)
				continue;
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
	g_stage  = stage;
	g_active = BaseMask();
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
		if ((g_active & (1u << i)) == 0)
			continue;
		if (!s.empty())
			s += ',';
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

} // namespace Libs::Graphics::ShaderRecompiler::Msl::Xf
