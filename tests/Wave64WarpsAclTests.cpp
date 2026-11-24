// nps5 uetrans: KYTY_WAVE64_WARPS_DENY / _ALLOW per-shader gate of the
// split-wave lowering (CodegenOptions), no GPU.
#include "graphics/shader/recompiler/CodegenOptions.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <unistd.h>

using namespace Libs::Graphics::ShaderRecompiler;

namespace {
int g_failures = 0;
void Expect(bool condition, const char *what) {
  if (!condition) {
    std::printf("Wave64WarpsAclTests: failed: %s\n", what);
    g_failures++;
  }
}
} // namespace

int main() {
  CodegenOptions options = GetCodegenOptions();
  options.wave64_warps = false;
  SetCodegenOptions(options);
  Expect(!Wave64WarpsShaderAllowed(0x1234), "switch off: nothing splits");

  options.wave64_warps = true;
  SetCodegenOptions(options);
  Expect(Wave64WarpsShaderAllowed(0x1234),
         "switch on, no list: everything splits");

  options.wave64_warps_deny = {0x1234, 0xb6c61a6c2ebb842dull};
  SetCodegenOptions(options);
  Expect(!Wave64WarpsShaderAllowed(0xb6c61a6c2ebb842dull), "deny listed");
  Expect(Wave64WarpsShaderAllowed(0x5678), "deny: others split");

  options.wave64_warps_allow = {0x5678, 0xb6c61a6c2ebb842dull};
  options.wave64_warps_allow_set = true;
  SetCodegenOptions(options);
  Expect(Wave64WarpsShaderAllowed(0x5678), "allow listed");
  Expect(!Wave64WarpsShaderAllowed(0x9abc), "allow: unlisted stays two-lane");
  Expect(!Wave64WarpsShaderAllowed(0xb6c61a6c2ebb842dull),
         "deny wins over allow");

  options.wave64_warps_allow.clear();
  SetCodegenOptions(options);
  Expect(!Wave64WarpsShaderAllowed(0x5678),
         "empty allow file = nothing splits");

  options.wave64_warps = false;
  SetCodegenOptions(options);
  Expect(!Wave64WarpsShaderAllowed(0x5678),
         "lists without the switch do nothing");

  if (g_failures != 0) {
    std::printf("Wave64WarpsAclTests: %d failure(s)\n", g_failures);
    return 1;
  }
  std::printf("Wave64WarpsAclTests: all passed\n");
  return 0;
}
