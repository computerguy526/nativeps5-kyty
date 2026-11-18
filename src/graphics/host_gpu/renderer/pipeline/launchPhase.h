#pragma once

// nps5 zerocomp: pure helpers of the launch compile phase (KYTY_PIPELINE_REPLAY_ALL) and of the
// prepare's delta pass, header only so tests/LaunchPhaseTests.cpp needs no emulator.
//
// - Key files (KYTY_PIPELINE_DELTA_OUT / KYTY_PIPELINE_REPLAY_KEYS): one 16-digit hex pipeline key
// per line. Pass 1
//   of a prepare writes the keys it built that the driver cache did not know before; pass 2 (after
//   the metallib build) replays only those, so only the delta is re-recorded under the prebuilt
//   libraries.
// - Scene selection (KYTY_SHADER_PREPARE=scene): the hold covers the records of the scene this
// start goes to, plus
//   untagged records (recorded before any scene opened: title, menus) and the "always" scenes
//   (KYTY_SHADER_PREPARE_ALWAYS, comma-separated names, a trailing '*' is a prefix: e.g.
//   "worldmap,hub_*,ui_*").
// - Last scene (<cache>/<title>.last_scene): the last scene of a session that is not an "always"
// scene; the next
//   start's target when it names no level (-lvl) of its own.

#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace Libs::Graphics::LaunchPhase {

// "0123456789abcdef\n" per key; blank lines, '#' comments and malformed lines are ignored.
inline std::unordered_set<uint64_t> ParseKeyFile(std::string_view text) {
	std::unordered_set<uint64_t> keys;
	size_t                       pos = 0;
	while (pos < text.size()) {
		auto end = text.find('\n', pos);
		if (end == std::string_view::npos) end = text.size();
		auto line = text.substr(pos, end - pos);
		pos       = end + 1;
		while (!line.empty() && (line.back() == '\r' || line.back() == ' ' || line.back() == '\t'))
			line.remove_suffix(1);
		while (!line.empty() && (line.front() == ' ' || line.front() == '\t'))
			line.remove_prefix(1);
		if (line.empty() || line.front() == '#' || line.size() > 16) continue;
		uint64_t value = 0;
		bool     ok    = true;
		for (const char c: line) {
			const int digit = c >= '0' && c <= '9'   ? c - '0'
			                  : c >= 'a' && c <= 'f' ? c - 'a' + 10
			                  : c >= 'A' && c <= 'F' ? c - 'A' + 10
			                                         : -1;
			if (digit < 0) {
				ok = false;
				break;
			}
			value = value << 4 | static_cast<uint64_t>(digit);
		}
		if (ok && value != 0) keys.insert(value);
	}
	return keys;
}

inline std::string FormatKeyFile(const std::vector<uint64_t>& keys) {
	std::string out;
	out.reserve(keys.size() * 17);
	char line[24];
	for (const auto key: keys) {
		std::snprintf(line, sizeof(line), "%016llx\n", static_cast<unsigned long long>(key));
		out += line;
	}
	return out;
}

// "a,b*, c" -> {"a", "b*", "c"}.
inline std::vector<std::string> ParsePatterns(std::string_view text) {
	std::vector<std::string> out;
	size_t                   pos = 0;
	while (pos <= text.size()) {
		auto end = text.find(',', pos);
		if (end == std::string_view::npos) end = text.size();
		auto item = text.substr(pos, end - pos);
		while (!item.empty() && (item.front() == ' ' || item.front() == '\t'))
			item.remove_prefix(1);
		while (!item.empty() && (item.back() == ' ' || item.back() == '\t'))
			item.remove_suffix(1);
		if (!item.empty()) out.emplace_back(item);
		pos = end + 1;
	}
	return out;
}

inline bool MatchesAny(std::string_view name, const std::vector<std::string>& patterns) {
	for (const auto& p: patterns) {
		if (!p.empty() && p.back() == '*') {
			if (name.substr(0, p.size() - 1) == std::string_view(p).substr(0, p.size() - 1))
				return true;
		} else if (name == p) {
			return true;
		}
	}
	return false;
}

// Whether a record belongs to the scene hold: its scene tags (journal: one level marker, list: the
// union of its records' markers; empty = recorded before any scene opened) name the target or an
// "always" scene.
inline bool InSceneHold(const std::vector<std::string_view>& tags, std::string_view target,
                        const std::vector<std::string>& always) {
	if (tags.empty()) return true;
	for (const auto tag: tags) {
		if (tag.empty() || (!target.empty() && tag == target) || MatchesAny(tag, always))
			return true;
	}
	return false;
}

// The last-scene memory: a scene that is an "always" scene (world map, hub, menus) never replaces
// the remembered one, so a session that ends on the world map still names the level played last.
inline std::string NextLastScene(const std::string& remembered, std::string_view current,
                                 const std::vector<std::string>& always) {
	if (current.empty() || MatchesAny(current, always)) return remembered;
	return std::string(current);
}

// The last_scene file: one line, the scene name (printable ASCII, at most 127 characters); anything
// else = none.
inline std::string ParseLastScene(std::string_view text) {
	while (!text.empty() && (text.back() == '\n' || text.back() == '\r' || text.back() == ' '))
		text.remove_suffix(1);
	if (text.empty() || text.size() > 127) return {};
	for (const char c: text) {
		if (c <= ' ' || c > '~') return {};
	}
	return std::string(text);
}

// Compile fan-out (user, 2026-11-17): logical cores - 4, at least 2; an override
// (NPS5_COMPILE_THREADS) wins, clamped to 1..16; 8 when the core count is unknown (0).
inline uint64_t CompileFanoutFor(uint64_t logical_cores, uint64_t override_threads) {
	if (override_threads != 0)
		return override_threads < 1 ? 1 : override_threads > 16 ? 16 : override_threads;
	if (logical_cores == 0) return 8;
	const uint64_t n = logical_cores > 6 ? logical_cores - 4 : 2;
	return n > 16 ? 16 : n;
}

// Compile gate (compileqos): the pools of a prepare / launch phase (the pipeline replay workers and
// the compute prewarm threads, plus whatever else joins) are each sized to the fan-out, so together
// they ran up to 14 Metal compiles at once on a 14-core Mac and starved WindowServer. Every driver
// pipeline build of those pools passes through this gate; at most `capacity` run at once (the
// fan-out), the rest wait. Capacity 0 (the default until a prepare / launch phase configures it) =
// no limit: live draw compiles are never gated.
class CompileGate {
public:
	static CompileGate& Get() {
		static CompileGate gate;
		return gate;
	}
	void Configure(uint64_t capacity) {
		{
			std::lock_guard lock(m_mutex);
			m_capacity = capacity;
		}
		m_wake.notify_all();
	}
	void Acquire() {
		std::unique_lock lock(m_mutex);
		m_wake.wait(lock, [this] { return m_capacity == 0 || m_running < m_capacity; });
		++m_running;
	}
	void Release() {
		{
			std::lock_guard lock(m_mutex);
			if (m_running != 0) --m_running;
		}
		m_wake.notify_one();
	}
	uint64_t Running() {
		std::lock_guard lock(m_mutex);
		return m_running;
	}
	struct Slot {
		Slot() { Get().Acquire(); }
		~Slot() { Get().Release(); }
		Slot(const Slot&)            = delete;
		Slot& operator=(const Slot&) = delete;
	};

private:
	std::mutex              m_mutex;
	std::condition_variable m_wake;
	uint64_t                m_capacity = 0;
	uint64_t                m_running  = 0;
};

} // namespace Libs::Graphics::LaunchPhase
