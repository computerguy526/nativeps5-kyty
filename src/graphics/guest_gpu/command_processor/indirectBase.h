// nps5 indirectfix: the CP's SET_BASE (indirect-args base) state and the decisions taken from it.
//
// Real PM4 semantics: SET_BASE (base_index 1) loads a CP register that persists until the next
// SET_BASE; a CP reset (new submission slice, DISPATCH reset packet) does not clear it, like the
// index base. The title may set the base in an earlier command buffer and draw in a later one.
// Zeroing it on Reset turned that into a fatal assert (rc65: "Apes on the Loose"). A draw whose
// base is still 0 is skipped (counted, rate-limited log), never a process exit.
// Header only, unit tested without a GPU (tests/IndirectBaseTests.cpp).
#pragma once

#include <cstdint>
#include <cstdlib>

namespace Libs::Graphics::IndirectBase {

struct State {
	uint64_t draw_base     = 0;
	uint64_t dispatch_base = 0;
	// Diagnostics: the last SET_BASE seen (shader_type 0 = draw, 1 = dispatch) and how many.
	uint64_t last_set_addr      = 0;
	uint32_t last_set_type      = 0xffffffffu;
	uint64_t set_count          = 0;
	uint64_t skipped_draws      = 0;
	uint64_t skipped_dispatches = 0;

	void Set(uint32_t shader_type, uint64_t addr) {
		(shader_type == 0u ? draw_base : dispatch_base) = addr;
		last_set_addr                                   = addr;
		last_set_type                                   = shader_type;
		set_count++;
	}
	// A CP reset keeps the bases (the registers persist on the real CP). Kept as a named no-op so
	// the decision is explicit and testable.
	void OnReset() {}
};

enum class DrawVerdict { Draw, SkipNoBase, SkipBadInitiator };

// nps5 indcrash (2026-11-28, a title's first visit of a new level): no guest mapping lives in the
// first 64 KiB. A title that sets the base from a null object's member (SET_BASE 0x20: null + a
// field offset) passed the old `base == 0` test, and the CPU read of the record at guest 0x20
// faulted on the GPU thread (ExecDrawIndirect, ReadGuestForCp -> GuestReadPtr). A base or record
// address in that page is "no base": skipped, counted, never read.
inline constexpr uint64_t kGuestNullPageEnd = 0x10000;

// A SET_BASE value that can address indirect records (draw or dispatch).
[[nodiscard]] inline bool BaseUsable(uint64_t base) {
	return base >= kGuestNullPageEnd;
}

// The `bytes` of indirect records at guest `addr` may be read: not in the null page, no
// wrap-around. Checked where the command processor reads records on the CPU (every path:
// plain, run replays, DrawPrep-prepared commits, multi-draws), so a base that reached the back
// some other way still never faults.
[[nodiscard]] inline bool ArgsReadable(uint64_t addr, uint64_t bytes) {
	return addr >= kGuestNullPageEnd && addr + bytes >= addr;
}

[[nodiscard]] inline DrawVerdict ClassifyDraw(uint64_t base, uint32_t draw_initiator,
                                              bool indexed) {
	// Indexed records select the DMA index source (0), auto-index records generated indices (2).
	if ((draw_initiator & ~0x20u) != (indexed ? 0u : 2u)) {
		return DrawVerdict::SkipBadInitiator;
	}
	return BaseUsable(base) ? DrawVerdict::Draw : DrawVerdict::SkipNoBase;
}

// SET_BASE (base_index 1) body words as CpOpSetBase decodes them: the address (8-byte aligned,
// 48 bits).
[[nodiscard]] inline uint64_t DecodeSetBaseAddr(uint32_t word1, uint32_t word2) {
	return (word1 & ~0x7ull) | (static_cast<uint64_t>(word2 & 0xffffu) << 32u);
}

// Rate limit for the skip log lines: the first 8, then every power of two.
[[nodiscard]] inline bool ShouldLogSkip(uint64_t count) {
	return count <= 8 || (count & (count - 1)) == 0;
}

// KYTY_INDIRECT_DIAG=1: print every indirect op's offset, initiator, submit id and the last
// SET_BASE.
[[nodiscard]] inline bool DiagEnabled() {
	static const bool enabled = [] {
		const char* v = std::getenv("KYTY_INDIRECT_DIAG");
		return v != nullptr && v[0] == '1';
	}();
	return enabled;
}

} // namespace Libs::Graphics::IndirectBase
