#pragma once

#include <atomic>
#include <cstdint>

namespace Libs::Graphics::RtSession {

// Session state of the software ray-tracing kernels (KYTY_RT_SOFTWARE, graphics/shader/recompiler).
//
// A GPU or driver that fails to build the pipeline of an RT kernel (an error result, an exception,
// a fault inside the driver) must not take the emulator down: software RT is marked failed for the
// rest of the session and every later dispatch of an RT kernel is skipped, as with
// KYTY_RT_SOFTWARE=0 (the tiled lighting and GI passes then leave their outputs untouched: black
// lighting). The next launch tries again; KYTY_RT_SOFTWARE=0 skips them from the start.

[[nodiscard]] inline std::atomic<bool>& FailedFlag() {
	static std::atomic<bool> failed {false};
	return failed;
}

// True once a RT pipeline build failed.
[[nodiscard]] inline bool Failed() {
	return FailedFlag().load(std::memory_order_acquire);
}

// Marks software RT failed; true for the caller that did it (that one logs the reason once).
inline bool MarkFailed() {
	return !FailedFlag().exchange(true, std::memory_order_acq_rel);
}

// Tests.
inline void ResetForTest() {
	FailedFlag().store(false, std::memory_order_release);
}

enum class DispatchAction : uint8_t {
	Run,           // the pipeline is ready
	WaitBounded,   // first dispatch that finds it still building: wait up to the budget
	Skip,          // not ready (or software RT failed): this dispatch is dropped
};

// What the command processor does with a dispatch of an RT kernel. The CP never waits for a
// pipeline build for long: a cold compile of a 300k-word kernel took 17 s on an RTX 50 GPU, and a
// stalled CP freezes the whole game (and trips the hang watchdog), while a skipped dispatch only
// leaves the lighting it feeds one frame behind. The first dispatch of a kernel waits up to
// `wait_budget_ms` (a cached pipeline builds in well under that, so a warm launch loses no frame
// of lighting); after that, and for every dispatch of a kernel whose build is still running, the
// dispatch is skipped. A budget of UINT32_MAX waits for the build (KYTY_RT_PIPELINE_ASYNC=0).
[[nodiscard]] inline DispatchAction DecideDispatch(bool session_failed, bool ready, bool first_request,
                                                   uint32_t wait_budget_ms) {
	if (session_failed) return DispatchAction::Skip;
	if (ready) return DispatchAction::Run;
	if (first_request && wait_budget_ms != 0) return DispatchAction::WaitBounded;
	return DispatchAction::Skip;
}

// What kind of background-built program a dispatch runs.
enum class KernelClass : uint8_t {
	Direct,    // a software RT kernel run as one dispatch (uses_bvh, no call split)
	CallSplit, // a call-split program (KYTY_SHADER_CALLS_SPLIT): its chain resumes past it
	Huge,      // KYTY_COMPUTE_BACKGROUND_WORDS: a huge compute program, not an RT kernel
};

// The first-dispatch wait budgets: kernel_ms for direct RT kernels, other_ms for huge programs.
// Defaults: KYTY_RT_KERNEL_WAIT_MS (120000) and KYTY_RT_PIPELINE_WAIT_MS (250).
//
// A direct RT kernel waits for its build at its first dispatch. Skipping it is not always one frame
// of lag: some kernels are dispatched only for a short window and never again (Astro Bot, Sky
// Garden: four kernels 304 times each while the level loads, one about 135 times right after it
// starts). With a cold driver cache such a kernel (125k-230k SPIR-V words) takes 3-10 s to build;
// when that outlasts its window it never runs, and the GI lighting it computes stays missing for
// the whole level (shadowed surfaces much too dark). A warm cache builds them in milliseconds (and
// the shader precompile starts them before the level), so the wait costs nothing there. A
// call-split program never waits: its chain resumes past callees that are not built (another
// title's reflections: some 200 of them per area, about 3 s each).
[[nodiscard]] inline uint32_t FirstWaitBudgetMs(KernelClass kind, uint32_t kernel_ms, uint32_t other_ms) {
	switch (kind) {
		case KernelClass::Direct: return kernel_ms;
		case KernelClass::CallSplit: return 0;
		case KernelClass::Huge: return other_ms;
	}
	return other_ms;
}

// Budget guard (KYTY_RT_BUDGET_GUARD, default on). A RT kernel that traverses an invalid BVH (level
// still streaming, stale upload) runs every lane to KYTY_RT_NODE_BUDGET: bounded, but seconds of
// GPU time per dispatch, and the dispatches of the next frames queue behind it (GPU stalls and, by
// correlation, GPU watchdog stops at level entry). The exhausted-lane counter lives in mapped
// GDS memory and rises while such a dispatch is still running, so the command processor reads it
// before every RT dispatch: a rise of `trip_lanes` or more since the previous look holds all RT
// dispatches for `hold_ms` (they are skipped, as while a pipeline builds: the lighting they feed
// lags). When the hold ends one dispatch probes; another rise doubles the hold up to `max_hold_ms`,
// `calm_ms` without a rise resets it. The guard cannot stop the first bad dispatch: the node budget
// bounds that one.
struct BudgetGuard {
	uint32_t trip_lanes  = 1024;
	uint64_t hold_ms     = 500;
	uint64_t max_hold_ms = 8000;
	uint64_t calm_ms     = 30000;

	// True when a RT dispatch may run now. `exhausted` = the GDS counter (wraps), `now_ms`
	// monotonic.
	[[nodiscard]] bool Allow(uint32_t exhausted, uint64_t now_ms) {
		const uint32_t rise = exhausted - m_seen; // unsigned: correct across a wrap
		m_seen              = exhausted;
		if (!m_primed) {
			// Lanes exhausted before the first look are not evidence against this dispatch.
			m_primed = true;
			return true;
		}
		if (m_holding && now_ms < m_hold_until) {
			// The running bad dispatch keeps counting: not new evidence, the hold is not extended.
			return false;
		}
		if (rise >= trip_lanes) {
			const bool repeat = m_trips != 0 && now_ms - m_last_trip < calm_ms + m_cur_hold;
			m_cur_hold =
			    repeat ? (m_cur_hold * 2 > max_hold_ms ? max_hold_ms : m_cur_hold * 2) : hold_ms;
			m_holding    = true;
			m_hold_until = now_ms + m_cur_hold;
			m_last_trip  = now_ms;
			m_trips++;
			return false;
		}
		m_holding = false;
		return true;
	}
	[[nodiscard]] uint64_t Trips() const { return m_trips; }
	[[nodiscard]] uint64_t CurrentHoldMs() const { return m_cur_hold; }

private:
	uint32_t m_seen       = 0;
	bool     m_primed     = false;
	bool     m_holding    = false;
	uint64_t m_hold_until = 0;
	uint64_t m_cur_hold   = 0;
	uint64_t m_last_trip  = 0;
	uint64_t m_trips      = 0;
};

// Load hold (KYTY_RT_LOAD_HOLD, default on): the proactive half of the budget guard. The invalid
// BVH shows up while a level streams in, and the first dispatch on it already costs seconds (the
// guard above only stops the following ones). Software-RT dispatches are skipped
//   - for `phase_quiet_ms` after the runtime's level phase changed (nps5_vfs_phase: the n-th
//   DISTINCT
//     levels/<name> opened, so a UI scene counts once per session, not every time it is shown), and
//   - while the game reads files fast (>= `rate_bytes_s`, sampled every `sample_ms` or slower) and
//   for
//     `quiet_ms` after the last such sample,
// whichever ends later. One hold lasts at most `max_hold_ms` after the last level phase change;
// after that RT runs again until every sign has paused (a long stream of loads cannot keep the
// lighting off). Observe() must be called from a path that runs every frame (every compute
// dispatch), not only at RT dispatches: a direct level boot has no RT dispatch before the level
// opens, and a VFS-only trigger missed that case (a GPU watchdog stop with the hold nominally on).
struct LoadHold {
	uint64_t rate_bytes_s   = 48ull << 20;
	uint64_t sample_ms      = 250;
	uint64_t quiet_ms       = 1500;
	uint64_t phase_quiet_ms = 5000;
	uint64_t max_hold_ms    = 60000;
	bool     use_phase      = true;

	// True when RT dispatches are skipped now. `bytes` = the runtime's VFS read counter, `phase`
	// its level phase, `now_ms` monotonic.
	[[nodiscard]] bool Hold(uint64_t bytes, int phase, uint64_t now_ms) {
		if (!m_primed) {
			m_primed = true;
			m_t      = now_ms;
			m_bytes  = bytes;
			m_phase  = phase;
			return false;
		}
		if (phase != m_phase) {
			m_phase = phase;
			if (use_phase) {
				Until(now_ms + phase_quiet_ms);
				// A newly opened level starts a new hold budget: hub -> level is two loads in a
				// row, and a budget counted from the hub's start ran out in the middle of the
				// level's load (the RT kernel then ran on invalid data and the GPU watchdog ended
				// the session). Distinct levels are finite, so this stays bounded.
				if (m_holding) m_since = now_ms;
			}
		}
		if (now_ms - m_t >= sample_ms) {
			if (bytes >= m_bytes && (bytes - m_bytes) * 1000 >= rate_bytes_s * (now_ms - m_t)) {
				Until(now_ms + quiet_ms);
			}
			m_t     = now_ms;
			m_bytes = bytes;
		}
		if (now_ms >= m_until) {
			m_holding = false;
			return false;
		}
		if (!m_holding) {
			m_holding = true;
			m_since   = now_ms;
			m_holds++;
		}
		return now_ms - m_since < max_hold_ms;
	}
	[[nodiscard]] uint64_t Holds() const { return m_holds; }

private:
	void Until(uint64_t t) {
		if (t > m_until) m_until = t;
	}
	bool     m_primed  = false;
	bool     m_holding = false;
	int      m_phase   = 0;
	uint64_t m_t       = 0;
	uint64_t m_bytes   = 0;
	uint64_t m_until   = 0;
	uint64_t m_since   = 0;
	uint64_t m_holds   = 0;
};

// Probe gate (KYTY_RT_PROBE, default off since kyty-rebase-best): the direct validity check. The only reliable statement
// about the BVH a RT kernel is about to traverse comes from the kernel itself: lanes that run to
// KYTY_RT_NODE_BUDGET found cyclic or garbage nodes. The command processor records commands ahead
// of the GPU, so the budget guard sees that counter rise only after a frame or two of RT dispatches
// (dozens) are already queued on the bad data; that queue, not one dispatch, is what stalls the
// GPU. The gate therefore admits exactly ONE RT dispatch (the probe) whenever validity is unknown:
// at the first RT dispatch of the session, after every load hold, and after every guard trip. Every
// other RT dispatch is skipped until the GPU has completed the probe (`completed`, the scheduler's
// tick). Then the counter is exact: no rise = the BVH is valid, RT runs freely; a rise = still
// invalid, RT stays closed for `retry_ms` (doubling to `max_retry_ms`) and the next dispatch probes
// again. A probe the GPU has not completed after `timeout_ms` counts as invalid. Timing (the load
// hold) only decides when to start probing; it no longer decides that the data is good.
struct ProbeGate {
	uint32_t trip_lanes   = 1024;
	uint64_t retry_ms     = 1000;
	uint64_t max_retry_ms = 8000;
	uint64_t timeout_ms   = 4000;

	enum class State : uint8_t { Open, Armed, Waiting, Closed };

	// Validity is unknown again (a load hold ended, a level opened): the next dispatch probes.
	void Arm() {
		if (m_state == State::Open) m_state = State::Armed;
	}
	// The budget guard saw lanes exhaust while the gate was open: closed, then probe.
	void Close(uint64_t now_ms) {
		if (m_state == State::Open || m_state == State::Armed) Fail(now_ms);
	}
	// True when this RT dispatch may run. Ask only for a dispatch that would really run (pipeline
	// ready). `probe_sent` is set when this dispatch is the probe: the caller notes its tick and
	// passes `completed` = "the GPU finished that tick" from then on.
	[[nodiscard]] bool Allow(uint32_t exhausted, bool completed, uint64_t now_ms,
	                         bool& probe_sent) {
		probe_sent = false;
		if (m_state == State::Closed) {
			if (now_ms < m_until) return false;
			m_state = State::Armed;
		}
		switch (m_state) {
			case State::Open: return true;
			case State::Armed:
				m_baseline = exhausted;
				m_sent     = now_ms;
				m_state    = State::Waiting;
				m_probes++;
				probe_sent = true;
				return true;
			case State::Waiting: {
				if (!completed && now_ms - m_sent < timeout_ms) return false;
				const uint32_t rise = exhausted - m_baseline; // unsigned: correct across a wrap
				if (!completed || rise >= trip_lanes) {
					Fail(now_ms);
					return false;
				}
				m_state       = State::Open;
				m_consecutive = 0;
				m_opened++;
				return true;
			}
			case State::Closed: break;
		}
		return false;
	}
	[[nodiscard]] State    Current() const { return m_state; }
	[[nodiscard]] uint64_t Probes() const { return m_probes; }
	[[nodiscard]] uint64_t Failed() const { return m_failed; }
	[[nodiscard]] uint64_t Opened() const { return m_opened; }
	[[nodiscard]] uint64_t RetryMs() const { return m_cur_retry; }

private:
	void Fail(uint64_t now_ms) {
		m_cur_retry = retry_ms;
		for (uint32_t i = 0; i < m_consecutive && m_cur_retry < max_retry_ms; i++)
			m_cur_retry *= 2;
		if (m_cur_retry > max_retry_ms) m_cur_retry = max_retry_ms;
		m_consecutive++;
		m_failed++;
		m_until = now_ms + m_cur_retry;
		m_state = State::Closed;
	}
	State    m_state       = State::Armed; // the first RT dispatch of a session is a probe
	uint32_t m_baseline    = 0;
	uint32_t m_consecutive = 0;
	uint64_t m_sent        = 0;
	uint64_t m_until       = 0;
	uint64_t m_cur_retry   = 0;
	uint64_t m_probes      = 0;
	uint64_t m_failed      = 0;
	uint64_t m_opened      = 0;
};

} // namespace Libs::Graphics::RtSession
