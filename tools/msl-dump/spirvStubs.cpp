// Link stubs for the standalone MSL build (KYTY_MSL_STANDALONE): the SPIR-V backend is not linked.
#include "graphics/shader/recompiler/backend/spirv/SpirvEmitter.h"

namespace Libs::Graphics::ShaderRecompiler::Spirv {
std::vector<uint32_t> EmitProgram(const IR::Program&, ShaderStageInputInfo, bool) { return {}; }
void              SetHostFloatControls(const HostFloatControls&) {}
HostFloatControls GetHostFloatControls() { return {}; }
HostBufferRobustness GetHostBufferRobustness() { return {}; }
HostImageFeatures GetHostImageFeatures() { return {}; }
HostGeometryGuard GetHostGeometryGuard() { return {}; }
HostShaderClock   GetHostShaderClock() { return {}; }
HostWaveClusters  GetHostWaveClusters() { return {}; }
bool WaveClustersActive(const IR::Program&) { return false; }
} // namespace Libs::Graphics::ShaderRecompiler::Spirv

namespace Libs::LibKernel::Memory {
const void* GuestReadPtr(unsigned long long, unsigned long long) { return nullptr; }
} // namespace Libs::LibKernel::Memory
