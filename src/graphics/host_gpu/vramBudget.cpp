#include "graphics/host_gpu/vramBudget.h"

#include "common/liveSwitch.h"

namespace Libs::Graphics::VramBudget {

namespace {

Live::Switch g_vram_gc_budget("KYTY_VRAM_GC_BUDGET", Live::ParseDefaultOff);

Live::Switch g_texture_gc_reuse("KYTY_TEXTURE_GC_REUSE", Live::ParseDefaultOff);

} // namespace

bool GcEnabled() noexcept {
	return g_vram_gc_budget.On();
}

bool ReuseEnabled() noexcept {
	return g_texture_gc_reuse.On();
}

} // namespace Libs::Graphics::VramBudget
