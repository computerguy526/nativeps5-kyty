#!/bin/sh
# Builds + runs the T2 msl-mem standalone tests, then compiles every emitted kernel with the Metal compiler.
set -e
HERE=$(cd "$(dirname "$0")" && pwd); SRC="$HERE/../../src"; OUT=${1:-/tmp/msl_mem_tests}
CFG=${KYTY_CMAKE_CONFIG_DIR:-"$HERE/../../../build-kbuild"}
mkdir -p "$OUT"; B="$SRC/graphics/shader/recompiler/backend/msl"
clang++ -std=c++20 -O0 -I"$SRC" -I"$CFG" "$HERE/MslMemStandalone.cpp" "$B"/MslMemCommon.cpp "$B"/MslBinding.cpp "$B"/MslBufferOps.cpp "$B"/MslSharedOps.cpp "$B"/MslWaveOps.cpp "$B"/MslImageOps.cpp -o "$OUT/msl_mem_tests"
"$OUT/msl_mem_tests" "$OUT"
for f in "$OUT"/*.metal; do
  xcrun -sdk macosx metal -std=metal3.1 -mmacosx-version-min=15 -c "$f" -o "${f%.metal}.air" 2>"${f%.metal}.log" && echo "metal OK  $(basename $f)" || { echo "metal ERR $(basename $f)"; head -20 "${f%.metal}.log"; exit 1; }
done
# Runtime check on the GPU via the R2 harness (mslh) when available.
MSLH=${MSLH:-"$HERE/../../../../work/mslnative/harness/mslh"}
if [ -x "$MSLH" ]; then
  mkdir -p "$OUT/rt"
  clang++ -std=c++20 -I"$SRC" -I"$CFG" "$HERE/MslMemStandalone.cpp" "$B"/MslMemCommon.cpp "$B"/MslBinding.cpp "$B"/MslBufferOps.cpp "$B"/MslSharedOps.cpp "$B"/MslWaveOps.cpp "$B"/MslImageOps.cpp -o "$OUT/gen"
  "$OUT/gen" "$OUT/rt" rt >/dev/null
  python3 "$HERE/verify_runtime.py" "$OUT/rt" "$MSLH"
fi
