#pragma once

// Window-title feedback for shader and pipeline compiles (nps5 hosted Kyty).
//
// The pipeline cache, the shader precompile replay and the pipeline journal replay publish plain
// relaxed counters here; the window's main thread reads them (at most every kTitleIntervalMs) and
// appends a status to the title ("compiling shaders N/M", "compiling shader (live) K",
// "(draw waiting)", "loading..."). Hot paths only do relaxed fetch_add/sub on these counters.
// KYTY_PROGRESS_TITLE=0 turns the title text off; KYTY_PROGRESS_TITLE_LOG=1 logs each title change
// to stderr. Header-only so the unit test (tests/CompileProgressTests.cpp) needs no emulator.

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <string>

namespace Libs::Graphics::CompileProgress {

inline constexpr uint64_t kTitleIntervalMs = 250;

struct Counters {
	// Replay of the shader precompile journal and of the pipeline journal (done counts finished entries).
	std::atomic<uint64_t> shader_total {0}, shader_done {0};
	std::atomic<uint64_t> pipeline_total {0}, pipeline_done {0};
	// Pipeline/program compiles in flight that are not replay work, and draws blocked on a compile.
	std::atomic<int32_t> live {0};
	std::atomic<int32_t> waiting {0};
};

inline Counters& Get() {
	static Counters counters;
	return counters;
}

// True on the replay/prewarm worker threads: their compiles are not "live".
inline thread_local bool t_replay_thread = false;

inline void MarkReplayThread() { t_replay_thread = true; }

// A compile in flight on a non-replay thread.
struct LiveScope {
	bool counted;
	explicit LiveScope(bool count = true) : counted(count && !t_replay_thread) {
		if (counted) Get().live.fetch_add(1, std::memory_order_relaxed);
	}
	~LiveScope() {
		if (counted) Get().live.fetch_sub(1, std::memory_order_relaxed);
	}
	LiveScope(const LiveScope&)            = delete;
	LiveScope& operator=(const LiveScope&) = delete;
};

// A draw (command processor / draw thread) blocked on a compile.
struct WaitScope {
	bool counted;
	explicit WaitScope(bool count = true) : counted(count && !t_replay_thread) {
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
	auto& c = Get();
	c.shader_total.store(0, std::memory_order_relaxed);
	c.shader_done.store(0, std::memory_order_relaxed);
}
inline void ClearPipelineReplay() {
	auto& c = Get();
	c.pipeline_total.store(0, std::memory_order_relaxed);
	c.pipeline_done.store(0, std::memory_order_relaxed);
}

struct Snapshot {
	uint64_t replay_done  = 0;
	uint64_t replay_total = 0;
	int32_t  live         = 0;
	int32_t  waiting      = 0;
};

inline Snapshot Read() {
	const auto& c = Get();
	Snapshot    s;
	s.replay_total = c.shader_total.load(std::memory_order_relaxed) + c.pipeline_total.load(std::memory_order_relaxed);
	s.replay_done  = c.shader_done.load(std::memory_order_relaxed) + c.pipeline_done.load(std::memory_order_relaxed);
	s.live         = c.live.load(std::memory_order_relaxed);
	s.waiting      = c.waiting.load(std::memory_order_relaxed);
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
		text = "compiling shaders " + std::to_string(s.replay_done) + "/" + std::to_string(s.replay_total);
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
	bool        m_have    = false;
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
