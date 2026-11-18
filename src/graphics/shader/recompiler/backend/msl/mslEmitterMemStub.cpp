// Weak default implementations of the T2 hooks (see mslEmitterInternal.h).
#include "graphics/shader/recompiler/backend/msl/mslEmitterInternal.h"

namespace Libs::Graphics::ShaderRecompiler::Msl {

__attribute__((weak)) bool EmitMemoryInst(MslCtx&, const IR::Inst&) {
	return false;
}
__attribute__((weak)) void EmitResourceParams(MslCtx&, std::vector<std::string>&) {}
__attribute__((weak)) void EmitResourcePrelude(MslCtx&) {}
__attribute__((weak)) void EmitResourceLocals(MslCtx&) {}

} // namespace Libs::Graphics::ShaderRecompiler::Msl
