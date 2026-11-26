#!/usr/bin/env bash
# Refreshes tests/fixtures/recomp_kytyGpuApi.h (the recomp copy of the ABI header the header-diff test compares token
# by token with src/hybrid/kytyGpuApi.h). usage: scripts/refresh_recomp_api_fixture.sh <recomp checkout> [rev]
set -euo pipefail
repo=${1:?recomp checkout}; rev=${2:-HEAD}
git -C "$repo" show "$rev:pkg/runtime/modules/libSceAnyPS5/kytyGpuApi.h" > "$(dirname "$0")/../tests/fixtures/recomp_kytyGpuApi.h"
