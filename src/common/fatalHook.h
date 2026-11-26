#ifndef KYTY_COMMON_FATALHOOK_H_
#define KYTY_COMMON_FATALHOOK_H_

// One optional callback that runs on the failing thread when a fatal error ends the process
// (Common::DbgReport in assert.cpp: EXIT / EXIT_IF / EXIT_NOT_IMPLEMENTED), before the report is
// printed and the process exits. Used by the refused-program record (pipeline/refusedPrograms.h) to
// note which program the failing thread was translating. Header only so tests need no emulator. The
// callback must not throw and should not take locks other threads may hold while they translate.

#include <atomic>
#include <string_view>

namespace Common {

using FatalHookFn = void (*)(std::string_view text);

inline std::atomic<FatalHookFn>& FatalHookSlot() {
	static std::atomic<FatalHookFn> slot {nullptr};
	return slot;
}

// Replaces the hook (nullptr: none); returns the previous one.
inline FatalHookFn SetFatalHook(FatalHookFn fn) {
	return FatalHookSlot().exchange(fn, std::memory_order_acq_rel);
}

inline void RunFatalHook(std::string_view text) {
	if (const auto fn = FatalHookSlot().load(std::memory_order_acquire)) fn(text);
}

} // namespace Common

#endif
