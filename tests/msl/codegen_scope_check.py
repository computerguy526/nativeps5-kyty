#!/usr/bin/env python3
"""Step-2 gate of the MSL port: the game's SPIR-V must not move.

The game's SPIR-V comes from the frontend, the IR passes and recompiler/backend/spirv; the MSL back end
(recompiler/backend/msl), kyty_msl_dump and the tests never run in it. This script lists the files changed
between two refs and fails when one outside the allowed set changed. ProgramCodec is allowed only when the
diff adds lines and removes none (the offline DecodeSpecializationCompat).

usage: codegen_scope_check.py <base-ref> <head-ref> [repo-dir]      real check (git diff --numstat)
       codegen_scope_check.py --selftest                           synthetic cases, exit 0 = pass
"""
import subprocess, sys

ALLOWED_PREFIX = (
    "src/graphics/shader/recompiler/backend/msl/",
    "tools/msl-dump/",
    "tests/",
)
ADDITIVE_ONLY = ("src/graphics/shader/recompiler/ir/ProgramCodec.cpp", "src/graphics/shader/recompiler/ir/ProgramCodec.h")
GUARDED = ("src/graphics/shader/", "src/graphics/host_gpu/renderer/pipeline/programDiskCache", "src/graphics/guest_gpu/gpu_format")


def check(rows):
    """rows: (added, removed, path). Returns the list of violations."""
    bad = []
    for add, rem, path in rows:
        if path.startswith(ALLOWED_PREFIX):
            continue
        if path in ADDITIVE_ONLY:
            if rem != 0:
                bad.append("%s: %d lines removed (only additions allowed)" % (path, rem))
            continue
        if path.startswith(GUARDED):
            bad.append("%s: codegen input changed (+%d -%d)" % (path, add, rem))
    return bad


def selftest():
    fails = 0

    def expect(name, rows, nbad):
        nonlocal fails
        got = len(check(rows))
        if got != nbad:
            print("FAIL %s: %d violations, want %d" % (name, got, nbad))
            fails += 1
    expect("msl only", [(10, 2, "src/graphics/shader/recompiler/backend/msl/MslXform.cpp"), (1, 1, "tests/MslCoreTests.cpp")], 0)
    expect("codec additive", [(16, 0, "src/graphics/shader/recompiler/ir/ProgramCodec.cpp")], 0)
    expect("codec removes", [(16, 1, "src/graphics/shader/recompiler/ir/ProgramCodec.cpp")], 1)
    expect("spirv emitter", [(1, 1, "src/graphics/shader/recompiler/backend/spirv/spirvEmitterAlu.cpp")], 1)
    expect("pass", [(1, 0, "src/graphics/shader/recompiler/ir/passes/ResourceTracking.cpp")], 1)
    expect("host code is out of scope", [(5, 5, "src/graphics/host_gpu/renderer/image/image.cpp")], 0)
    expect("empty", [], 0)
    print("selftest %s" % ("FAIL" if fails else "PASS"))
    return 1 if fails else 0


def main():
    if sys.argv[1:2] == ["--selftest"]:
        return selftest()
    base, head = sys.argv[1], sys.argv[2]
    repo = sys.argv[3] if len(sys.argv) > 3 else "."
    out = subprocess.run(["git", "-C", repo, "diff", "--numstat", base, head], capture_output=True, text=True, check=True).stdout
    rows = []
    for line in out.splitlines():
        a, r, p = line.split("\t", 2)
        rows.append((int(a) if a != "-" else 0, int(r) if r != "-" else 0, p))
    bad = check(rows)
    for b in bad:
        print("FAIL", b)
    print("codegen scope: %s (%d files changed)" % ("FAIL" if bad else "PASS", len(rows)))
    return 1 if bad else 0


sys.exit(main())
