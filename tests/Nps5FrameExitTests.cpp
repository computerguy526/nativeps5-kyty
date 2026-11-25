// Nps5Frame::LastLoadSignNs must stay usable while the process exits (saver threads call it during static destruction).
#include "common/nps5Frame.h"

#include <cstdio>

namespace {
// Constructed before main (so destroyed after the function-local statics first used in main): calls the probe during static destruction.
struct LateCaller {
	~LateCaller() {
		Nps5Frame::LastLoadSignNs(1); // old code: 'mutex lock failed: Invalid argument' -> terminate -> abort (the test process dies, exit != 0)
		std::printf("nps5_frame_exit_tests: all passed\n");
	}
} g_late;
} // namespace

int main() {
	Nps5Frame::LastLoadSignNs(0); // constructs the function-local statics (the mutex) after g_late
	return 0;
}
