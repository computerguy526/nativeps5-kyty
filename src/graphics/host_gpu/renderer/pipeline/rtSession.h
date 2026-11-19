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

// Budget guard (KYTY_RT_BUDGET_GUARD, default on). A RT kernel that traverses an invalid BVH (level
// still streaming, stale upload) runs every lane to KYTY_RT_NODE_BUDGET: bounded, but seconds of GPU
// time per dispatch, and the dispatches of the next frames queue behind it (GPU watchdog at Sky
// Garden level entry, work/research/wdog-2026-11-19.md). The exhausted-lane counter lives in mapped
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

	// True when a RT dispatch may run now. `exhausted` = the GDS counter (wraps), `now_ms` monotonic.
	[[nodiscard]] bool Allow(uint32_t exhausted, uint64_t now_ms) {
		const uint32_t rise = exhausted - m_seen;   // unsigned: correct across a wrap
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
			m_cur_hold        = repeat ? (m_cur_hold * 2 > max_hold_ms ? max_hold_ms : m_cur_hold * 2) : hold_ms;
			m_holding         = true;
			m_hold_until      = now_ms + m_cur_hold;
			m_last_trip       = now_ms;
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

} // namespace Libs::Graphics::RtSession
