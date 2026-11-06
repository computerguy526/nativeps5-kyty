#ifndef EMULATOR_SRC_COMMON_ASTROHYBRID_H_
#define EMULATOR_SRC_COMMON_ASTROHYBRID_H_

#include <cstdio>
#include <cstdlib>
#include <cstring>

// AstroHybrid experimental switches (startup, each default 0 = the original Kyty behaviour).
//   ASTROHYBRID_OPT_1=1  PageManager::ApplySpan skips the region host_lock when the span is already
//                        settled (pageManager.cpp).
//   ASTROHYBRID_OPT_2=1  TextureBindingMemo defaults to 32768 entries (textureBindingMemo.cpp).
namespace AstroHybrid {

[[nodiscard]] inline bool ReadSwitch(const char* name, const char* what) {
	const auto* value = std::getenv(name);
	const bool  on    = value != nullptr && std::strcmp(value, "1") == 0;
	std::printf("AstroHybrid: %s=%d (%s)\n", name, on ? 1 : 0, what);
	std::fflush(stdout);
	return on;
}

[[nodiscard]] inline bool Opt1() {
	static const bool on = ReadSwitch("ASTROHYBRID_OPT_1", "deferred unprotect skips host_lock when settled");
	return on;
}

[[nodiscard]] inline bool Opt2() {
	static const bool on = ReadSwitch("ASTROHYBRID_OPT_2", "texture binding memo default 32768 entries");
	return on;
}

} // namespace AstroHybrid

#endif // EMULATOR_SRC_COMMON_ASTROHYBRID_H_
