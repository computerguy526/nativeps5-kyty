#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_KKTAG_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_KKTAG_H_

#include "graphics/host_gpu/vulkanCommon.h"

#include <cstdint>

namespace Libs::Graphics {

// nps5 memtrk diagnostic (KYTY_KK_TAG=1): KosmicKrisp's nps5_kk_tag() names the guest program of
// the compute pipeline the calling thread creates next (KK prints it with each loop-cap key under
// NPS5_KK_LOOP_KEYS=1). Pipelines are created on several threads (fast-first, optimizer,
// precompile), so the program hash is registered per shader module and each creation site sets
// the tag from its create info's module for the duration of the call. No-ops when off.
void KkTagRegisterModule(vk::ShaderModule module, uint64_t program_hash);
class KkTagScope final {
public:
	explicit KkTagScope(vk::ShaderModule module);
	~KkTagScope();
	KkTagScope(const KkTagScope&)            = delete;
	KkTagScope& operator=(const KkTagScope&) = delete;

private:
	bool m_set = false;
};

} // namespace Libs::Graphics

#endif
