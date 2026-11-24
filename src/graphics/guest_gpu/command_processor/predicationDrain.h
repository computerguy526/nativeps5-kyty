#ifndef EMULATOR_SRC_GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_PREDICATIONDRAIN_H_
#define EMULATOR_SRC_GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_PREDICATIONDRAIN_H_

// KYTY_PREDICATION_NO_DRAIN decision (graphicsRun.cpp ExecPredication, memory predicate, op 0x03),
// split out of the command processor so it can be unit tested alone
// (tests/PredicationDrainTests.cpp).
//
//   mode 0 (default): a waiting SET_PREDICATION always flushes the GPU and waits before the read.
//   mode 1 (upstream 840b9f575): never drains; the wait selector applies only to Z-pass queries.
//   mode 2 (nps5 saroscp): drains unless nothing can still be writing the predicate, i.e. no label
//          deferred to completion covers it AND the guest view of the 8 bytes is clean.
//
// No Kyty types: pure functions of plain values.

#include <cstring>

namespace Libs::Graphics::PredicationDrain {

// "1" -> 1, "2" -> 2, anything else (unset, "0", junk) -> 0.
[[nodiscard]] inline int ParseMode(const char* value) noexcept {
	if (value == nullptr) {
		return 0;
	}
	return std::strcmp(value, "1") == 0 ? 1 : std::strcmp(value, "2") == 0 ? 2 : 0;
}

// True when the CP must flush the GPU and wait before reading the predicate.
// deferred_label: a label deferred to completion covers the predicate bytes.
// clean_read:     a strict clean read of the 8 bytes succeeded (no GPU-dirty bytes, no pending
//                 publication, no GPU-modified image).
[[nodiscard]] inline bool ShouldDrain(int mode, bool wait_op_set, bool deferred_label,
                                      bool clean_read) noexcept {
	if (!wait_op_set) {
		return false;
	}
	switch (mode) {
		case 1: return false;
		case 2: return deferred_label || !clean_read;
		default: return true;
	}
}

} // namespace Libs::Graphics::PredicationDrain

#endif
