#pragma once
// KYTY_CP_OP_STATS=1 (diagnostic, default off): the "MeshIndirect 10s:" console line. Header only
// and free of renderer includes so a unit test can link it alone. Nothing here feeds a draw
// decision; the counters only say what the indirect mesh path did in the last window.
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>

namespace MeshIndirectStats {

struct Snapshot {
	uint64_t draws              = 0; // conversions recorded
	uint64_t always_empty       = 0; // consumed without a read
	uint64_t checks             = 0; // verify comparisons completed
	uint64_t mismatches         = 0;
	uint64_t status_overflow    = 0;
	uint64_t status_limits      = 0;
	uint64_t status_index_range = 0;
	uint64_t stale              = 0;
	uint64_t declined_off       = 0; // mode off, or mode empty for a draw that is not provably empty
	uint64_t declined_multi     = 0;
	uint64_t declined_restart   = 0;
	uint64_t declined_clear     = 0;
	uint64_t declined_program   = 0;
};

// Decline reasons of DrawIndirectNative's mesh block (relaxed counters, GPU thread writes).
struct Declines {
	std::atomic<uint64_t> off {0};
	std::atomic<uint64_t> multi {0};
	std::atomic<uint64_t> restart {0};
	std::atomic<uint64_t> clear {0};
	std::atomic<uint64_t> program {0};
};
inline Declines& GetDeclines() {
	static Declines declines;
	return declines;
}

// now - before, field by field; a counter that went backwards (a reset) reads 0.
[[nodiscard]] inline Snapshot Delta(const Snapshot& now, const Snapshot& before) {
	const auto sub = [](uint64_t a, uint64_t b) { return a >= b ? a - b : uint64_t {0}; };
	Snapshot   d;
	d.draws              = sub(now.draws, before.draws);
	d.always_empty       = sub(now.always_empty, before.always_empty);
	d.checks             = sub(now.checks, before.checks);
	d.mismatches         = sub(now.mismatches, before.mismatches);
	d.status_overflow    = sub(now.status_overflow, before.status_overflow);
	d.status_limits      = sub(now.status_limits, before.status_limits);
	d.status_index_range = sub(now.status_index_range, before.status_index_range);
	d.stale              = sub(now.stale, before.stale);
	d.declined_off       = sub(now.declined_off, before.declined_off);
	d.declined_multi     = sub(now.declined_multi, before.declined_multi);
	d.declined_restart   = sub(now.declined_restart, before.declined_restart);
	d.declined_clear     = sub(now.declined_clear, before.declined_clear);
	d.declined_program   = sub(now.declined_program, before.declined_program);
	return d;
}

// One line, NUL terminated, truncated when `n` is too small; returns the length written (without
// the NUL), or 0 when n == 0.
inline int Format(char* out, size_t n, const Snapshot& d, const char* mode) {
	if (out == nullptr || n == 0) {
		return 0;
	}
	const int len = std::snprintf(
	    out, n,
	    "MeshIndirect 10s: mode=%s draws=%llu empty=%llu declined(off=%llu multi=%llu restart=%llu "
	    "clear=%llu program=%llu) verify(checks=%llu mismatches=%llu) status(overflow=%llu "
	    "limits=%llu index_range=%llu) stale=%llu",
	    mode != nullptr ? mode : "?", static_cast<unsigned long long>(d.draws),
	    static_cast<unsigned long long>(d.always_empty), static_cast<unsigned long long>(d.declined_off),
	    static_cast<unsigned long long>(d.declined_multi), static_cast<unsigned long long>(d.declined_restart),
	    static_cast<unsigned long long>(d.declined_clear), static_cast<unsigned long long>(d.declined_program),
	    static_cast<unsigned long long>(d.checks), static_cast<unsigned long long>(d.mismatches),
	    static_cast<unsigned long long>(d.status_overflow), static_cast<unsigned long long>(d.status_limits),
	    static_cast<unsigned long long>(d.status_index_range), static_cast<unsigned long long>(d.stale));
	if (len < 0) {
		out[0] = '\0';
		return 0;
	}
	return len < static_cast<int>(n) ? len : static_cast<int>(n) - 1;
}

} // namespace MeshIndirectStats
