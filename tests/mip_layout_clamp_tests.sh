#!/bin/bash
# nps5 mipclamp: builds and runs MipLayoutClampTests standalone against the real tile code.
# usage: tests/mip_layout_clamp_tests.sh <3rdparty dir with fmt, tracy, Vulkan-Headers, VMA, xxHash, magic_enum> <build dir with cmake_config.h and a kyty_gpu object tree>
set -eu
W=$(cd "$(dirname "$0")/.." && pwd); T=${1:?3rdparty dir}; B=${2:?build dir}; O=$(mktemp -d)
INC="-I$W/src/compat/arm64 -I$T/fmt/include -I$B -I$W/src -I$T/tracy/public -I$T/Vulkan-Headers/include -I$T/magic_enum/include/magic_enum -I$T/xxHash -I$T/VulkanMemoryAllocator/include"
FL="-std=c++20 -O1 -DXBYAK_NO_EXCEPTION -DTRACY_ENABLE -DTRACY_MANUAL_LIFETIME -DTRACY_NO_CRASH_HANDLER -DTRACY_ON_DEMAND"
cd "$W"
for f in src/graphics/guest_gpu/tile.cpp src/graphics/host_gpu/renderer/image/textureCommon.cpp tests/MipLayoutClampTests.cpp tests/MipLayoutClampStubs.cpp; do
  c++ $FL $INC -c $f -o $O/$(basename $f).o
done
c++ $O/*.o $B/CMakeFiles/kyty_gpu.dir/src/graphics/guest_gpu/gpu_format.cpp.o $B/3rdparty/fmt/CMakeFiles/fmt.dir/src/format.cc.o -o $O/mlt
$O/mlt
