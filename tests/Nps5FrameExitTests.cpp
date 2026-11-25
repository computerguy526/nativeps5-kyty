// Nps5Frame::LastLoadSignNs must stay usable while the process exits (saver threads call it during static destruction).
#include "common/nps5Frame.h"

#include <cstdio>
#include <exception>
#include <unistd.h>

namespace {
// Constructed before main (so destroyed after the function-local statics first used in main): calls the probe during static destruction.
struct LateCaller {
	~LateCaller() {
		try {
			Nps5Frame::LastLoadSignNs(1);
		} catch (const std::exception& e) {
			std::printf("FAILED: LastLoadSignNs threw during static destruction: %s\n", e.what());
			std::fflush(stdout);
			_exit(1);
		}
		std::printf("nps5_frame_exit_tests: all passed\n");
	}
} g_late;
} // namespace

int main() {
	Nps5Frame::LastLoadSignNs(0); // constructs the function-local statics (the mutex) after g_late
	return 0;
}
