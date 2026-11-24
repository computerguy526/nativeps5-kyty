// nps5 one: unit tests that link SrtWalker.cpp without kernel/memory.cpp get
// the not-hosted answer of Libs::LibKernel::Memory::GuestReadPtr (the guest
// address itself; tests run with a zero guest bias).
#include <cstdint>

namespace Libs::LibKernel::Memory {
[[nodiscard]] const void *GuestReadPtr(uint64_t vaddr,
                                       uint64_t /*size*/) noexcept {
  return reinterpret_cast<const void *>(static_cast<uintptr_t>(vaddr));
}
} // namespace Libs::LibKernel::Memory
