#ifndef KYTY_HYBRID_FOOTPRINTGOVERNOR_H_
#define KYTY_HYBRID_FOOTPRINTGOVERNOR_H_

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>

// nps5 footprint governor (KYTY_FOOTPRINT_GOVERNOR=1, default off). The GPU collectors react to device
// usage against the planning budget, never to the process footprint, and the self memory limit
// (StartSelfMemoryLimit, exit code 86) can only end the process. The governor sits between: while the
// footprint is above a soft mark (KYTY_FOOTPRINT_SOFT_PCT of the self limit, default 86 %) it publishes a
// budget factor (KYTY_FOOTPRINT_FACTOR_PCT, default 60 %) that GraphicContext::GetTotalMemoryBudget applies,
// so the image/buffer collectors and the native image pool (which keeps nothing at the mark) run now, and a
// one-shot pool flush. It never changes the self limit. Pure state machine, header only:
// tests/FootprintGovernorTests.cpp (work/research/memgrow-2026-11-25.md section 4).
namespace Hybrid {

struct FootprintGovernorConfig {
	bool     enabled         = false;
	uint64_t limit_mib       = 0;    // the self memory limit; 0 = off
	uint32_t soft_pct        = 86;   // enter at this share of the limit
	uint32_t leave_gap_pct   = 4;    // leave below soft_pct - leave_gap_pct ...
	uint32_t hold_ms         = 5000; // ... for this long
	uint32_t factor_permille = 600;  // budget factor while active
};

class FootprintGovernor {
public:
	enum class Transition : uint8_t { None, Activated, Deactivated };

	explicit FootprintGovernor(const FootprintGovernorConfig& config): m_config(config) {
		// The soft mark never sits closer than 512 MiB to the limit.
		constexpr uint64_t MinHeadroom = 512;
		if (m_config.limit_mib > MinHeadroom) {
			m_soft = std::min(m_config.limit_mib * m_config.soft_pct / 100, m_config.limit_mib - MinHeadroom);
		} else {
			m_soft = m_config.limit_mib / 2;
		}
		const uint64_t gap = m_config.limit_mib * m_config.leave_gap_pct / 100;
		m_leave            = m_soft > gap ? m_soft - gap : 0;
	}

	[[nodiscard]] bool     Usable() const { return m_config.enabled && m_config.limit_mib != 0; }
	[[nodiscard]] uint64_t SoftMib() const { return m_soft; }
	[[nodiscard]] uint64_t LeaveMib() const { return m_leave; }
	[[nodiscard]] bool     Active() const { return m_active; }

	// The budget factor now (1000 = unchanged).
	[[nodiscard]] uint32_t FactorPermille() const {
		return Usable() && m_active ? std::min<uint32_t>(m_config.factor_permille, 1000) : 1000;
	}

	// One footprint sample (MiB) at now_ms (monotonic). Activates at/above the soft mark; deactivates once
	// the footprint stayed below the leave mark for hold_ms.
	Transition Step(uint64_t footprint_mib, uint64_t now_ms) {
		if (!Usable()) {
			return Transition::None;
		}
		if (!m_active) {
			if (footprint_mib >= m_soft) {
				m_active      = true;
				m_below_valid = false;
				return Transition::Activated;
			}
			return Transition::None;
		}
		if (footprint_mib >= m_leave) {
			m_below_valid = false;
			return Transition::None;
		}
		if (!m_below_valid) {
			m_below_valid = true;
			m_below_since = now_ms;
		}
		if (now_ms - m_below_since >= m_config.hold_ms) {
			m_active      = false;
			m_below_valid = false;
			return Transition::Deactivated;
		}
		return Transition::None;
	}

private:
	FootprintGovernorConfig m_config;
	uint64_t                m_soft        = 0;
	uint64_t                m_leave       = 0;
	bool                    m_active      = false;
	bool                    m_below_valid = false;
	uint64_t                m_below_since = 0;
};

// budget scaled by permille: never raised, and not below 1 GiB (or the budget itself when that is smaller).
[[nodiscard]] inline uint64_t ScaledBudget(uint64_t budget, uint32_t permille) {
	constexpr uint64_t Floor = 1ull << 30u;
	if (permille >= 1000 || budget == 0) {
		return budget;
	}
	const uint64_t scaled = budget / 1000 * permille + budget % 1000 * permille / 1000;
	return std::min(budget, std::max(scaled, std::min(budget, Floor)));
}

// At most one log line per interval (state changes are already hysteresis-limited; this bounds a
// pathological flap).
class LogGate {
public:
	explicit LogGate(uint64_t interval_ms): m_interval(interval_ms) {}
	[[nodiscard]] bool Allow(uint64_t now_ms) {
		if (m_have && now_ms - m_last < m_interval) {
			return false;
		}
		m_have = true;
		m_last = now_ms;
		return true;
	}

private:
	uint64_t m_interval;
	uint64_t m_last = 0;
	bool     m_have = false;
};

// Published by the poll thread (kytyGpuBridge.cpp), read by the graphics layer.
inline std::atomic<uint32_t> g_footprint_budget_permille {1000};
inline std::atomic<bool>     g_footprint_pool_flush {false};

[[nodiscard]] inline uint64_t FootprintScaledBudget(uint64_t budget) {
	return ScaledBudget(budget, g_footprint_budget_permille.load(std::memory_order_relaxed));
}

// Env: unset / empty / not a number in [lo, hi] = fallback.
[[nodiscard]] inline uint32_t FootprintEnvUint(const char* text, uint32_t fallback, uint32_t lo, uint32_t hi) {
	if (text == nullptr || text[0] == '\0') {
		return fallback;
	}
	char*      end   = nullptr;
	const auto value = std::strtoul(text, &end, 10);
	return end == text || *end != '\0' || value < lo || value > hi ? fallback : static_cast<uint32_t>(value);
}

[[nodiscard]] inline FootprintGovernorConfig FootprintConfigFromEnv(const char* gate, const char* soft_pct,
                                                                    const char* hold_ms, const char* factor_pct,
                                                                    uint64_t limit_mib) {
	FootprintGovernorConfig c;
	c.enabled         = gate != nullptr && gate[0] == '1' && gate[1] == '\0';
	c.limit_mib       = limit_mib;
	c.soft_pct        = FootprintEnvUint(soft_pct, 86, 50, 98);
	c.hold_ms         = FootprintEnvUint(hold_ms, 5000, 0, 600000);
	c.factor_permille = FootprintEnvUint(factor_pct, 60, 20, 100) * 10;
	return c;
}

} // namespace Hybrid

#endif // KYTY_HYBRID_FOOTPRINTGOVERNOR_H_
