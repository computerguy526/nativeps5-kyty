#pragma once

// nps5 levelprep: the background prepare queue (tier 2), header only so tests/BgQueueTests.cpp can drive it with fake
// compile jobs.
//
// A fixed job order, a worker budget (KYTY_SHADER_BG_WORKERS, default 2), scene promotion (the player entered a scene:
// its unbuilt jobs go first and the budget grows by `promote_boost` until they are done) and a pause switch that the
// frame-time guard flips. Workers ask `Acquire`; it never blocks, so the caller owns the sleeping and the thread QoS.

#include <algorithm>
#include <cstdint>
#include <deque>
#include <mutex>
#include <unordered_set>
#include <vector>

namespace Libs::Graphics::BgQueue {

struct Job {
	uint64_t id    = 0; // the caller's pipeline key
	uint32_t index = 0; // the caller's own index (e.g. into its replay items)
	uint32_t scene = 0; // the scene the job belongs to (promotion)
};

enum class Result { Job, Wait, Done };

class Queue {
public:
	// `workers`: jobs that may run at once; `promote_boost`: extra jobs allowed while a promoted scene is unfinished.
	void Reset(std::vector<Job> jobs, unsigned workers, unsigned promote_boost = 1) {
		std::scoped_lock lock(m_mutex);
		m_pending.assign(jobs.begin(), jobs.end());
		m_workers   = std::max(1u, workers);
		m_boost     = promote_boost;
		m_running   = 0;
		m_completed = 0;
		m_promoted.clear();
		m_paused = false;
	}
	void SetWorkers(unsigned workers) {
		std::scoped_lock lock(m_mutex);
		m_workers = std::max(1u, workers);
	}
	void SetPaused(bool paused) {
		std::scoped_lock lock(m_mutex);
		m_paused = paused;
	}
	[[nodiscard]] bool Paused() {
		std::scoped_lock lock(m_mutex);
		return m_paused;
	}

	// Moves the unbuilt jobs of `scene` to the front (their relative order kept) and raises the budget until they are
	// done. Returns the number of jobs promoted; a scene with no pending job changes nothing.
	size_t Promote(uint32_t scene) {
		return PromoteIf([scene](const Job& j) { return j.scene == scene; }, scene);
	}
	// The same for the jobs whose id (pipeline key) is in `ids`: what a scene entered by the player needs, whichever
	// scene each job was filed under. `tag` names the promotion (a scene index); a tag already promoted keeps its
	// boost accounting.
	size_t PromoteIds(const std::unordered_set<uint64_t>& ids, uint32_t tag) {
		return PromoteIf([&ids](const Job& j) { return ids.count(j.id) != 0; }, tag);
	}
	[[nodiscard]] bool PromotedPending() {
		std::scoped_lock lock(m_mutex);
		return PromotedPendingLocked();
	}

	[[nodiscard]] unsigned Budget() {
		std::scoped_lock lock(m_mutex);
		return BudgetLocked();
	}

	// A worker asks for work. Wait: paused or over budget (try again later). Done: nothing left to start.
	Result Acquire(Job& out) {
		std::scoped_lock lock(m_mutex);
		if (m_pending.empty()) return Result::Done; // promotion only reorders: nothing new will come
		if (m_paused && !PromotedPendingLocked()) return Result::Wait;
		if (m_running >= BudgetLocked()) return Result::Wait;
		out = m_pending.front();
		m_pending.pop_front();
		m_running++;
		return Result::Job;
	}

	void Complete(const Job& job) {
		std::scoped_lock lock(m_mutex);
		if (m_running != 0) m_running--;
		m_completed++;
		m_promoted.erase(job.index);
	}

	[[nodiscard]] size_t Pending() {
		std::scoped_lock lock(m_mutex);
		return m_pending.size();
	}
	[[nodiscard]] unsigned Running() {
		std::scoped_lock lock(m_mutex);
		return m_running;
	}
	[[nodiscard]] uint64_t Completed() {
		std::scoped_lock lock(m_mutex);
		return m_completed;
	}
	[[nodiscard]] bool Finished() {
		std::scoped_lock lock(m_mutex);
		return m_pending.empty() && m_running == 0;
	}

private:
	template<class Pred>
	size_t PromoteIf(Pred pred, uint32_t /*tag*/) {
		std::scoped_lock lock(m_mutex);
		std::deque<Job>  front, back;
		for (const auto& j: m_pending) (pred(j) ? front : back).push_back(j);
		if (front.empty()) return 0;
		m_pending    = std::move(front);
		const auto n = m_pending.size();
		m_pending.insert(m_pending.end(), back.begin(), back.end());
		for (const auto& j: m_pending) {
			if (pred(j)) m_promoted.insert(j.index);
		}
		return n;
	}
	bool     PromotedPendingLocked() const { return !m_promoted.empty(); }
	unsigned BudgetLocked() const { return m_workers + (PromotedPendingLocked() ? m_boost : 0); }

	std::mutex             m_mutex;
	std::deque<Job>        m_pending;
	std::unordered_set<uint32_t> m_promoted; // indices of promoted jobs not completed yet
	unsigned               m_workers   = 2;
	unsigned               m_boost     = 1;
	unsigned               m_running   = 0;
	uint64_t               m_completed = 0;
	bool                   m_paused    = false;
};

// ---- the pause guard ---------------------------------------------------------------------------------------------

// What the guard looks at (the caller reads them from its frame timer; all times are steady-clock nanoseconds).
struct Sample {
	uint64_t now_ns          = 0;
	double   last_frame_ms   = 0; // the most recent frame
	double   median_frame_ms = 0; // median of the last ~120 frames; 0: unknown
	uint64_t last_slow_ns    = 0; // time of the last frame above the slow limit (0: never)
	uint64_t last_load_ns    = 0; // time of the last loading sign (0: never)
	bool     calm_scene      = false; // title / menu / hub: the game is not in a level
	bool     promoted        = false; // a promoted scene's jobs are pending (loading is what hides them)
};

struct GuardConfig {
	double   spike_ratio = 1.25;       // pause when the last frame is above ratio x median
	uint64_t window_ns   = 2'000'000'000ull; // a slow frame / load sign this recent pauses
	uint64_t resume_ns   = 500'000'000ull;   // calm for this long before it resumes (hysteresis)
};

// Pure verdict: may a new background compile start now? Calm scenes (menu, hub) are never throttled, a loading sign
// does not pause while promoted jobs of the scene being loaded are pending.
[[nodiscard]] inline bool Allows(const Sample& s, const GuardConfig& c = {}) {
	if (s.calm_scene) return true;
	if (s.median_frame_ms > 0 && s.last_frame_ms > c.spike_ratio * s.median_frame_ms) return false;
	if (s.last_slow_ns != 0 && s.now_ns >= s.last_slow_ns && s.now_ns - s.last_slow_ns < c.window_ns) return false;
	if (!s.promoted && s.last_load_ns != 0 && s.now_ns >= s.last_load_ns && s.now_ns - s.last_load_ns < c.window_ns) return false;
	return true;
}

// Hysteresis around Allows: pauses at once, resumes after `resume_ns` of calm.
class Guard {
public:
	explicit Guard(GuardConfig config = {}): m_config(config) {}
	// Returns true while compiles may start.
	bool Update(const Sample& s) {
		if (!Allows(s, m_config)) {
			m_paused     = true;
			m_calm_since = 0;
			return false;
		}
		if (!m_paused) return true;
		if (m_calm_since == 0) m_calm_since = s.now_ns;
		if (s.now_ns - m_calm_since >= m_config.resume_ns) {
			m_paused = false;
			return true;
		}
		return false;
	}
	[[nodiscard]] bool Paused() const { return m_paused; }

private:
	GuardConfig m_config;
	bool        m_paused     = false;
	uint64_t    m_calm_since = 0;
};

// Median of the last N frame times (ms), for the guard.
class FrameMedian {
public:
	explicit FrameMedian(size_t window = 120): m_window(std::max<size_t>(1, window)) {}
	void Add(double ms) {
		if (m_ring.size() < m_window) m_ring.push_back(ms);
		else m_ring[m_next++ % m_window] = ms;
	}
	[[nodiscard]] double Median() const {
		if (m_ring.empty()) return 0;
		auto v = m_ring;
		std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(v.size() / 2), v.end());
		return v[v.size() / 2];
	}

private:
	size_t              m_window;
	size_t              m_next = 0;
	std::vector<double> m_ring;
};

// Worker count from the environment text (KYTY_SHADER_BG_WORKERS): 1..3 as asked; anything else (empty, 0, junk)
// gives `fallback`; more than `max_allowed` is clamped (the machine's cores).
[[nodiscard]] inline unsigned WorkersFromText(const char* text, unsigned fallback = 2, unsigned max_allowed = 16) {
	if (!text || !*text) return fallback;
	unsigned value = 0;
	for (const char* p = text; *p; p++) {
		if (*p < '0' || *p > '9' || value > 1000) return fallback;
		value = value * 10 + static_cast<unsigned>(*p - '0');
	}
	if (value == 0) return fallback;
	return std::min(value, std::max(1u, max_allowed));
}

} // namespace Libs::Graphics::BgQueue
