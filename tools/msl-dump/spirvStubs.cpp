// Link stubs for the standalone MSL build (KYTY_MSL_STANDALONE): the SPIR-V
// backend is not linked.
#include "graphics/shader/recompiler/backend/spirv/SpirvEmitter.h"

namespace Libs::Graphics::ShaderRecompiler::Spirv {
std::vector<uint32_t> EmitProgram(const IR::Program &, ShaderStageInputInfo,
                                  bool) {
  return {};
}
void SetHostFloatControls(const HostFloatControls &) {}
HostFloatControls GetHostFloatControls() { return {}; }
HostBufferRobustness GetHostBufferRobustness() { return {}; }
HostImageFeatures GetHostImageFeatures() { return {}; }
HostGeometryGuard GetHostGeometryGuard() { return {}; }
HostShaderClock GetHostShaderClock() { return {}; }
HostWaveClusters GetHostWaveClusters() { return {}; }
bool WaveClustersActive(const IR::Program &) { return false; }
// Upstream int22 host-feature queries the recompiler reads: the defaults of an unconfigured
// device (as the real ones before SetHost* runs) and no KYTY_* switch (metal.clean_env).
bool GetHostBufferPairAtomics() { return false; }
bool GetHostMeshLayerDrop() { return false; }
bool GetHostFragmentInterlock() { return false; }
bool GetHostComputeDerivativesLinear() { return false; }
uint64_t IndirectFormatCountAddress() { return 0; }
CoherentLoadMode GetCoherentLoadMode() { return CoherentLoadMode::Plain; }
const char *CoherentLoadModeName(CoherentLoadMode) { return "plain"; }
const char *MeshPassesUnsupported(const IR::Program &) { return nullptr; }
} // namespace Libs::Graphics::ShaderRecompiler::Spirv

namespace Libs::LibKernel::Memory {
const void *GuestReadPtr(unsigned long long, unsigned long long) {
  return nullptr;
}
} // namespace Libs::LibKernel::Memory
