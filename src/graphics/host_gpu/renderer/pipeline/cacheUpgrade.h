#pragma once

// nps5 v14prep: the shader-cache upgrade decision (header only, no GPU, unit-tested by
// tests/CacheUpgradeTests.cpp).
//
// The driver pipeline cache (<title>.bin) keeps entries keyed on the shader code of the build that
// wrote it. A new Kyty codegen (any change of the recompiler sources or its runtime state:
// ShaderRecompiler::CodegenFingerprint) or another driver makes most of them dead weight that still
// counts against the replay's size limit and the file cap, so a cache carried over from an older
// release (a v1.3.x cache under v1.4) stopped the launch-phase replay after a few dozen pipelines
// and everything else compiled during play. The stamp file next to the cache records which stack
// the cache was prepared for and whether the preparation finished:
//
//   <title>.cacheprep   text, two lines:   identity=<text>\nstate=prepared|preparing\n
//
// Decide() turns (stamp on disk, current identity, whether cache data exists) into an action:
//   Fresh    no stamp, no cache data: the normal first start (the launch phase builds the list);
//   the stamp is written when it finishes. Warm     stamp = current identity and prepared: nothing
//   special (the launch phase hits the cache). Resume   stamp = current identity but preparing (an
//   earlier start quit midway): keep what is there, replay the whole list again. Upgrade  another
//   identity, or cache data without any stamp (a v1.3.x cache): the old driver cache is set aside
//   (one generation,
//            ApplyUpgrade), the list is replayed in full behind the loading card, the stamp is
//            written "prepared" only after that replay completed, and the set-aside copy is removed
//            then.
// KYTY_CACHE_UPGRADE=0 turns the whole thing off (Decide returns Warm and nothing is read or
// written).

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>

namespace Libs::Graphics::CacheUpgrade {

enum class Action { Fresh, Warm, Resume, Upgrade };

struct Stamp {
	std::string identity;
	bool        prepared = false;
	std::string
	     note; // prepared only: how the replay ended ("failed=2;beyond_limit=0"); informational
	bool operator==(const Stamp&) const = default;
};

inline const char* ActionName(Action a) {
	switch (a) {
		case Action::Fresh: return "fresh";
		case Action::Warm: return "warm";
		case Action::Resume: return "resume";
		case Action::Upgrade: return "upgrade";
	}
	return "?";
}

// FNV-1a 64 as 16 hex digits (an identity string is built from fingerprints, this only shortens
// them).
inline std::string Hash16(std::string_view data) {
	uint64_t h = 1469598103934665603ull;
	for (const char c: data) {
		h ^= static_cast<uint8_t>(c);
		h *= 1099511628211ull;
	}
	char buf[17];
	std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(h));
	return buf;
}

inline std::string Format(const Stamp& s) {
	return "identity=" + s.identity + "\nstate=" + (s.prepared ? "prepared" : "preparing") + "\n" +
	       (s.note.empty() ? "" : "note=" + s.note + "\n");
}

// nullopt for text that is not a stamp (torn, foreign, empty): the caller treats it as "no stamp".
inline std::optional<Stamp> Parse(std::string_view text) {
	Stamp  s;
	bool   have_identity = false, have_state = false;
	size_t pos = 0;
	while (pos < text.size()) {
		auto end = text.find('\n', pos);
		if (end == std::string_view::npos) return std::nullopt; // no final newline = a torn write
		const auto line = text.substr(pos, end - pos);
		pos             = end + 1;
		if (line.rfind("identity=", 0) == 0) {
			s.identity    = std::string(line.substr(9));
			have_identity = !s.identity.empty();
		} else if (line.rfind("note=", 0) == 0) {
			s.note = std::string(line.substr(5));
		} else if (line.rfind("state=", 0) == 0) {
			const auto v = line.substr(6);
			if (v == "prepared")
				s.prepared = true;
			else if (v != "preparing")
				return std::nullopt;
			have_state = true;
		}
	}
	if (!have_identity || !have_state) return std::nullopt;
	return s;
}

// `cache_loaded`: the driver cache file was read and accepted. A "prepared" stamp over a cache that
// is missing or was rejected (another driver, a damaged file) is not Warm: the preparation is done
// again (Resume keeps what is there).
inline Action Decide(const std::optional<Stamp>& disk, std::string_view current,
                     bool cache_has_data, bool disabled, bool cache_loaded = true) {
	if (disabled) return Action::Warm;
	if (!disk) return cache_has_data ? Action::Upgrade : Action::Fresh;
	if (disk->identity != current) return Action::Upgrade;
	if (!disk->prepared) return Action::Resume;
	return cache_has_data && cache_loaded ? Action::Warm : Action::Resume;
}

// The upgrade decision needs a pipeline list with pipelines that resolved against the game
// (m_list->pipelines holds only those) and a driver cache to work on. Without them nothing is read,
// set aside or stamped.
inline bool CanDecide(uint64_t usable_list_pipelines, bool driver_cache_present,
                      bool stamp_path_known) {
	return stamp_path_known && usable_list_pipelines != 0 && driver_cache_present;
}

// Whether the stamp's start has to run the full replay: everything but Warm.
inline bool NeedsFullReplay(Action a) {
	return a != Action::Warm;
}

// ---- files
inline std::optional<Stamp> Read(const std::filesystem::path& path) {
	std::ifstream in(path, std::ios::binary);
	if (!in) return std::nullopt;
	std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
	return Parse(text);
}

// Atomic: a temporary file next to it, then rename.
inline bool Write(const std::filesystem::path& path, const Stamp& s) {
	auto tmp = path;
	tmp += ".tmp";
	{
		std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
		if (!out) return false;
		out << Format(s);
		if (!out) return false;
	}
	std::error_code ec;
	std::filesystem::rename(tmp, path, ec);
	return !ec;
}

inline std::filesystem::path AsidePath(const std::filesystem::path& file) {
	auto p = file;
	p += ".pre-upgrade";
	return p;
}

// Upgrade: the old driver cache is set aside under ONE fixed name (a second upgrade before the
// first finished replaces it, so only one generation exists). Returns true when a file was moved.
// The cache is not deleted: it stays until the new one is prepared.
inline bool SetAside(const std::filesystem::path& cache_file) {
	std::error_code ec;
	if (!std::filesystem::exists(cache_file, ec)) return false;
	const auto aside = AsidePath(cache_file);
	std::filesystem::remove(aside, ec);
	ec.clear();
	std::filesystem::rename(cache_file, aside, ec);
	return !ec;
}

// After the new cache is prepared: drop the set-aside generation. Returns its size (0 when there
// was none).
inline uint64_t RemoveAside(const std::filesystem::path& cache_file) {
	std::error_code ec;
	const auto      aside = AsidePath(cache_file);
	const auto      size =
	    std::filesystem::exists(aside, ec) ? std::filesystem::file_size(aside, ec) : 0;
	if (ec) return 0;
	std::filesystem::remove(aside, ec);
	return ec ? 0 : size;
}

// The replay limit default follows the file cap (7/8: the replay must stop before the next start
// would discard the file).
inline uint64_t DefaultReplayLimitMb(uint64_t cap_mb) {
	return cap_mb / 8u * 7u;
}

// A replay is complete when it was not stopped: every record was attempted. Records that failed or
// that the cache limit cut are counted in the stamp's note; running the replay again would not
// change them.
inline bool ReplayComplete(bool stopped, uint64_t /*beyond_limit*/) {
	return !stopped;
}

} // namespace Libs::Graphics::CacheUpgrade
