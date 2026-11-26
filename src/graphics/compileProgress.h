#pragma once

// Window-title feedback for shader and pipeline compiles (nps5 hosted Kyty).
//
// The pipeline cache, the shader precompile replay and the pipeline journal replay publish plain
// relaxed counters here; the window's main thread reads them (at most every kTitleIntervalMs) and
// appends a status to the title ("compiling shaders N/M", "compiling shader (live) K",
// "(draw waiting)", "loading..."). Hot paths only do relaxed fetch_add/sub on these counters.
// KYTY_PROGRESS_TITLE=0 turns the title text off; KYTY_PROGRESS_TITLE_LOG=1 logs each title change
// to stderr. Header-only so the unit test (tests/CompileProgressTests.cpp) needs no emulator.

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <string>

namespace Libs::Graphics::CompileProgress {

inline constexpr uint64_t kTitleIntervalMs = 250;

struct Counters {
	// Replay of the shader precompile journal and of the pipeline journal (done counts finished
	// entries).
	std::atomic<uint64_t> shader_total {0}, shader_done {0};
	std::atomic<uint64_t> pipeline_total {0}, pipeline_done {0};
	// The loaded pipeline list: entries in the list (known when the list is loaded; every entry
	// resolved counts in pipeline_done: a driver/archive cache hit at once, a compile when it
	// finishes). Shader permutations (Metal libraries) are a sub-step and stay in shader_*; they
	// are never added to the list count.
	std::atomic<uint64_t> list_total {0};
	// Pipeline/program compiles in flight that are not replay work, and draws blocked on a compile.
	std::atomic<int32_t> live {0};
	std::atomic<int32_t> waiting {0};
	// load card: cumulative live compiles started / finished (the load card's N / M).
	std::atomic<uint64_t> live_started {0}, live_finished {0};
	// card3: compute prewarm (requested / finished builds). Counted as replay work for Pending().
	std::atomic<uint64_t> prewarm_total {0}, prewarm_done {0};
	// card3: ONE progress unit for the opening card, fed by every compile path (library replay,
	// pipeline list replay, compute prewarm): total only grows within a session, done counts
	// completions (cache/archive hits at once, skipped/failed/dropped too). A new session starts
	// (both reset) when work is added after all finished.
	std::atomic<uint64_t> unit_total {0}, unit_done {0};
	// zerocomp: the launch phase's hold of the game (pipelineCache.cpp, KYTY_PIPELINE_REPLAY_ALL):
	// 0 = no hold this session, 1 = the game is held, 2 = the hold is over (with
	// KYTY_SHADER_PREPARE=scene that is before the unit is done: the rest builds behind the game
	// and the opening card must not wait for it).
	std::atomic<int> launch_hold {0};
};

inline Counters& Get() {
	static Counters counters;
	return counters;
}

inline void UnitAddTotal(uint64_t n) {
	if (n == 0) return;
	auto& c = Get();
	auto  t = c.unit_total.load(std::memory_order_relaxed);
	if (const auto d = c.unit_done.load(std::memory_order_relaxed); d >= t && (t != 0 || d != 0)) {
		// Everything finished: a new session. (Benign race: a lost reset only keeps the old,
		// finished counts.)
		if (c.unit_total.compare_exchange_strong(t, 0, std::memory_order_relaxed))
			c.unit_done.store(0, std::memory_order_relaxed);
	}
	c.unit_total.fetch_add(n, std::memory_order_relaxed);
}
inline void UnitAddDone(uint64_t n) {
	if (n != 0) Get().unit_done.fetch_add(n, std::memory_order_relaxed);
}
inline void PrewarmRequested() {
	Get().prewarm_total.fetch_add(1, std::memory_order_relaxed);
	UnitAddTotal(1);
}
inline void PrewarmFinished(uint64_t n = 1) {
	Get().prewarm_done.fetch_add(n, std::memory_order_relaxed);
	UnitAddDone(n);
}

// True on the replay/prewarm worker threads: their compiles are not "live".
inline thread_local bool t_replay_thread = false;

inline void MarkReplayThread() {
	t_replay_thread = true;
}

// A compile in flight on a non-replay thread.
struct LiveScope {
	bool counted;
	explicit LiveScope(bool count = true): counted(count && !t_replay_thread) {
		if (counted) {
			Get().live_started.fetch_add(1, std::memory_order_relaxed);
			Get().live.fetch_add(1, std::memory_order_relaxed);
		}
	}
	~LiveScope() {
		if (counted) {
			Get().live.fetch_sub(1, std::memory_order_relaxed);
			Get().live_finished.fetch_add(1, std::memory_order_relaxed);
		}
	}
	LiveScope(const LiveScope&)            = delete;
	LiveScope& operator=(const LiveScope&) = delete;
};

// A draw (command processor / draw thread) blocked on a compile.
struct WaitScope {
	bool counted;
	explicit WaitScope(bool count = true): counted(count && !t_replay_thread) {
		if (counted) Get().waiting.fetch_add(1, std::memory_order_relaxed);
	}
	~WaitScope() {
		if (counted) Get().waiting.fetch_sub(1, std::memory_order_relaxed);
	}
	WaitScope(const WaitScope&)            = delete;
	WaitScope& operator=(const WaitScope&) = delete;
};

// A replay finished or stopped: it no longer counts.
inline void ClearShaderReplay() {
	auto&      c = Get();
	const auto t = c.shader_total.load(std::memory_order_relaxed),
	           d = c.shader_done.load(std::memory_order_relaxed);
	if (t > d) UnitAddDone(t - d); // entries that never ran (stopped): the unit still completes
	c.shader_total.store(0, std::memory_order_relaxed);
	c.shader_done.store(0, std::memory_order_relaxed);
}
inline void ClearPipelineReplay() {
	auto&      c = Get();
	const auto t = c.pipeline_total.load(std::memory_order_relaxed),
	           d = c.pipeline_done.load(std::memory_order_relaxed);
	if (t > d) UnitAddDone(t - d);
	c.pipeline_total.store(0, std::memory_order_relaxed);
	c.pipeline_done.store(0, std::memory_order_relaxed);
	c.list_total.store(0, std::memory_order_relaxed);
}

struct Snapshot {
	uint64_t replay_done  = 0;
	uint64_t replay_total = 0;
	int32_t  live         = 0;
	int32_t  waiting      = 0;
	// Pipeline list unit (0 total = no list loaded) and the library sub-step.
	uint64_t list_done = 0, list_total = 0, lib_done = 0, lib_total = 0;
};

inline Snapshot Read() {
	const auto& c = Get();
	Snapshot    s;
	const auto  pw_total = c.prewarm_total.load(std::memory_order_relaxed),
	            pw_done  = c.prewarm_done.load(std::memory_order_relaxed);
	s.replay_total       = c.shader_total.load(std::memory_order_relaxed) +
	                       c.pipeline_total.load(std::memory_order_relaxed) + pw_total;
	s.replay_done        = c.shader_done.load(std::memory_order_relaxed) +
	                       c.pipeline_done.load(std::memory_order_relaxed) + pw_done;
	s.live               = c.live.load(std::memory_order_relaxed);
	s.waiting            = c.waiting.load(std::memory_order_relaxed);
	// card3: the card's unit = every compile path (never shrinks, done never above total, never
	// goes backwards).
	s.list_total = c.unit_total.load(std::memory_order_relaxed);
	s.list_done  = std::min<uint64_t>(c.unit_done.load(std::memory_order_relaxed), s.list_total);
	s.lib_total  = c.shader_total.load(std::memory_order_relaxed);
	s.lib_done   = c.shader_done.load(std::memory_order_relaxed);
	return s;
}

// The status text ("" when idle): "compiling shaders 312/510", "compiling shader (live) 3",
// "compiling shaders 312/510 + live 3", each followed by " (draw waiting)" while a draw waits.
inline std::string FormatStatus(const Snapshot& s) {
	const bool replay = s.replay_total != 0 && s.replay_done < s.replay_total;
	int32_t    live   = s.live > 0 ? s.live : 0;
	if (s.waiting > 0 && live < s.waiting) live = s.waiting;
	std::string text;
	if (replay) {
		text = "compiling shaders " + std::to_string(s.replay_done) + "/" +
		       std::to_string(s.replay_total);
		if (live > 0) text += " + live " + std::to_string(live);
	} else if (live > 0) {
		text = "compiling shader (live) " + std::to_string(live);
	}
	if (s.waiting > 0 && !text.empty()) text += " (draw waiting)";
	return text;
}

// The whole title: base title, an em dash and the status; before the first frame the base is
// `name` and an idle status reads "loading...".
inline std::string ComposeTitle(const std::string& base, const Snapshot& s, bool first_frame_seen) {
	auto status = FormatStatus(s);
	if (!first_frame_seen && status.empty()) status = "loading\xE2\x80\xA6";
	if (status.empty()) return base;
	return base + " \xE2\x80\x94 " + status;
}

// A window title must be valid UTF-8: SDL's Cocoa backend builds an NSString from it, which is nil
// for invalid bytes, and
// -[NSWindow setTitle:] then throws NSInternalInconsistencyException ('aString != nil') on the main
// thread (a v1.4 laptop crash: the title was set while the process was already tearing down).
// Invalid bytes become U+FFFD; embedded NULs end the title.
inline std::string ValidUtf8Title(const std::string& in) {
	std::string out;
	out.reserve(in.size());
	size_t i = 0, n = in.size();
	while (i < n) {
		const auto c = static_cast<unsigned char>(in[i]);
		if (c == 0) break;
		size_t len = 0;
		if (c < 0x80)
			len = 1;
		else if (c >= 0xC2 && c <= 0xDF)
			len = 2;
		else if (c >= 0xE0 && c <= 0xEF)
			len = 3;
		else if (c >= 0xF0 && c <= 0xF4)
			len = 4;
		bool ok = len != 0 && i + len <= n;
		for (size_t k = 1; ok && k < len; k++)
			ok = (static_cast<unsigned char>(in[i + k]) & 0xC0) == 0x80;
		if (ok && len == 3) {
			const auto c1 = static_cast<unsigned char>(in[i + 1]);
			if ((c == 0xE0 && c1 < 0xA0) || (c == 0xED && c1 >= 0xA0))
				ok = false; // overlong, surrogate
		}
		if (ok && len == 4) {
			const auto c1 = static_cast<unsigned char>(in[i + 1]);
			if ((c == 0xF0 && c1 < 0x90) || (c == 0xF4 && c1 >= 0x90)) ok = false;
		}
		if (ok) {
			out.append(in, i, len);
			i += len;
		} else {
			out += "\xEF\xBF\xBD";
			i++;
		}
	}
	return out;
}

// Sets the title at most every kTitleIntervalMs and only when the text changed.
class TitleThrottle {
public:
	// True when `text` must be applied now (then it is remembered as the current title).
	bool ShouldSet(uint64_t now_ms, const std::string& text) {
		if (m_have && text == m_last) return false;
		if (m_have && now_ms - m_last_ms < kTitleIntervalMs) return false;
		m_have    = true;
		m_last    = text;
		m_last_ms = now_ms;
		return true;
	}

private:
	bool        m_have = false;
	std::string m_last;
	uint64_t    m_last_ms = 0;
};

inline bool Enabled() {
	static const bool enabled = [] {
		const char* v = std::getenv("KYTY_PROGRESS_TITLE");
		return v == nullptr || v[0] != '0';
	}();
	return enabled;
}

inline bool LogEnabled() {
	static const bool enabled = [] {
		const char* v = std::getenv("KYTY_PROGRESS_TITLE_LOG");
		return v != nullptr && v[0] == '1';
	}();
	return enabled;
}

} // namespace Libs::Graphics::CompileProgress
