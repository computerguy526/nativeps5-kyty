// stubs for the standalone build of MipLayoutClampTests.cpp (tile.cpp + textureCommon.cpp + gpu_format.cpp + fmt linked in; see tests/mip_layout_clamp_tests.sh)
#include "common/assert.h"
#include "common/profiler.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include <cstdio>
#include <cstdlib>
namespace Common {
int DbgExitHandler(char const* f, int l, std::string_view t) { std::printf("EXIT %s:%d %.*s\n", f, l, (int)t.size(), t.data()); return 1; }
int DbgExitHandler(char const* f, int l, fmt::text_style, std::string_view t) { return DbgExitHandler(f, l, t); }
int DbgExitIfHandler(char const* e, char const* f, int l) { std::printf("EXIT_IF %s %s:%d\n", e, f, l); return 1; }
int DbgNotImplementedHandler(char const* e, char const* f, int l) { std::printf("NOT_IMPL %s %s:%d\n", e, f, l); return 1; }
void DbgExit(int) { std::abort(); }
}
namespace Profiler { namespace Detail { std::atomic<bool> g_zones{false}; } void ScopedBlock::Begin(const tracy::SourceLocationData*) {} void ScopedBlock::End() noexcept {} }
namespace Libs::Graphics { vk::Format VulkanFormat(Prospero::BufferFormat) { return vk::Format::eR32Sfloat; } }
