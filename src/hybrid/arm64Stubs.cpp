// nps5: kyty_gpu.dylib on arm64. Stand-ins for what the rest of Kyty references from the sources
// that only exist for x86-64 guest code in the process (CMakeLists.txt KYTY_GPU_X86_ONLY_SOURCES:
// guest printf/libc, the guest instruction patcher and the x64 instruction emulator). The host
// runtime (nativeps5) owns the guest's libc and code; none of these may be reached.

#include "common/common.h"

#include <cstdio>
#include <cstdlib>

namespace {

[[noreturn, maybe_unused]] void Unreachable(const char* what) {
	std::fprintf(stderr, "kyty_gpu: %s is not available in the arm64 hybrid build\n", what);
	std::abort();
}

} // namespace

#include "common/guestLoad.h"
#include "loader/guestInstructionPatcher.h"
#include "loader/symbolDatabase.h"
#include "loader/x64InstructionEmulator.h"

namespace Libs {

namespace LibcInternalExt {
void RunThreadAtexitDestructors() {} // guest libc thread-exit destructors: the host runtime's
} // namespace LibcInternalExt

namespace LibC {
void InitLibC_1(Loader::SymbolDatabase* /*s*/) {} // no guest libc exports from this library
int  GetArgc() {
	return 0;
}
const char** GetArgv() {
	return nullptr;
}
} // namespace LibC

} // namespace Libs

namespace Loader {

GuestInstructionHostFeatures GetGuestInstructionHostFeatures() {
	return {};
}
void                        RegisterGuestInstructionPatchModule(void*, uint64_t, void*, uint64_t) {}
void                        UnregisterGuestInstructionPatchModule(void*) {}
GuestInstructionPatchResult PatchGuestInstructions(uint64_t, uint64_t, std::span<const uintptr_t>,
                                                   bool, bool, GuestInstructionHostFeatures,
                                                   uint64_t) {
	Unreachable("the guest instruction patcher");
}
bool DecodeEhFrameFunctionStarts(uint64_t, uint64_t, std::vector<uintptr_t>*) {
	return false;
}

namespace X64InstructionEmulator {
bool TryEmulate(void*) {
	return false;
}
bool DecodeGuestLoad(void*, uint64_t, GuestLoad::Load&) {
	return false;
}
bool ApplyGuestLoad(void*, const GuestLoad::Load&, const void*) {
	return false;
}
void                ConfigureReciprocalSqrtStats() {}
ReciprocalSqrtStats GetReciprocalSqrtStats() {
	return {};
}
} // namespace X64InstructionEmulator

} // namespace Loader
