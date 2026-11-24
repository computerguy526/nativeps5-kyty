// nps5 one: unit tests that link pageManager.cpp without kernel/memory.cpp run
// as a stand-alone (not hosted) process.
namespace Libs::LibKernel::Memory {
bool HostedActive() noexcept { return false; }
} // namespace Libs::LibKernel::Memory
