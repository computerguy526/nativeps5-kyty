#!/bin/bash
# nps5 kytyone: the merge gate for the one Kyty (ASTRO PPSA21564 + SAROS PPSA07631 + Sparta PPSA28997).
# usage: tests/one/run_one_tests.sh <cmake build dir> [--require "astro startcrash saros sparta"] [--msl-ref <kyty_msl_dump> --programs <programs.bin> [--msl-max N]]
#  1. builds and runs the header-only unit tests (kyty_one_tests + the existing nps5 ones), ninja progress shown;
#  2. (a) the built kyty_gpu.dylib exports every graphics NID of each required title (nm, the seam binds by dlsym);
#  3. (b) every title switch in tests/one/switches.txt is read with a default-off idiom (source scan);
#  4. (d) startcrash: PrepareBlankFrame takes present_mutex for the producer-less blank frame;
#  5. (c) optional: the MSL back end emits byte-identical text to a reference kyty_msl_dump for the programs of a cache;
#  6. prints the codegen fingerprint (CodegenVersion changes with any recompiler source edit: new program cache).
set -u
B=${1:?build dir}; shift
SRC=$(cd "$(dirname "$0")/../.." && pwd)
REQUIRE="astro"; MSLREF=""; PROGS=""; MSLMAX=400
while [ $# -gt 0 ]; do case $1 in --require) REQUIRE=$2; shift;; --msl-ref) MSLREF=$2; shift;; --programs) PROGS=$2; shift;; --msl-max) MSLMAX=$2; shift;; esac; shift; done
fail=0; note() { echo "[one] $*"; }; bad() { echo "[one] FAIL: $*"; fail=$((fail+1)); }

TESTS="kyty_one_tests hybrid_seam_tests hosted_guard_tests hosted_ranges_tests cp_lead_bound_tests load_card_tests launch_phase_tests
 read_fault_frame_burst_tests occlusion_draw_skip_tests compile_progress_tests progress_overlay_tests live_switch_tests
 kk_lowering_tests gpu_safety_tests"
# Not in the list: scalar_provenance_tests, resource_tracking_tests, resource_materialization_tests fail to link at the base
# (159eba04: undefined Libs::LibKernel::Memory::GuestReadPtr), a pre-existing CMake gap, not a merge regression.
avail=""; for t in $TESTS; do grep -q "build $t:" "$B/build.ninja" 2>/dev/null && avail="$avail $t" || note "skip $t (no target)"; done
note "building:$avail"
nice -n 10 ninja -C "$B" $avail 2>&1 | grep -E "^\[[0-9]+/[0-9]+\]|error:|FAILED" | awk '/^\[/{if(++n%10==0)print; next}{print}' || true
for t in $avail; do
  if [ ! -x "$B/$t" ]; then bad "$t did not build"; continue; fi
  out=$("$B/$t" 2>&1); rc=$?
  [ $rc = 0 ] && note "PASS $t $(echo "$out" | grep -E 'KytyOneTests: .*(exported|checked)' | tr '\n' ';')" || { bad "$t rc=$rc"; echo "$out" | grep -iE "fail" | head -10; }
done

LIB=$B/kyty_gpu.dylib
if [ -f "$LIB" ]; then
  syms=$(nm -gU "$LIB" 2>/dev/null | awk '{print $3}' | sed 's/^_//' | sort -u)
  for t in astro saros sparta; do
    f=$SRC/tests/one/${t}_nids.inc; tot=0; miss=""
    while read -r nid name; do tot=$((tot+1)); echo "$syms" | grep -qxF -- "$nid" || miss="$miss $name"; done < <(grep -oE '^\{"[^"]+", "[^"]+"' "$f" | sed -E 's/^\{"([^"]+)", "([^"]+)"/\1 \2/')
    n=$(echo $miss | wc -w | tr -d ' ')
    if echo " $REQUIRE " | grep -q " $t "; then [ "$n" = 0 ] && note "PASS dylib exports $t $tot/$tot" || bad "dylib lacks $n $t exports:$miss"
    else note "info dylib exports $t $((tot-n))/$tot (not required)"; fi
  done
else note "no $LIB: export check skipped"; fi

python3 -I - "$SRC" <<'PY' || fail=$((fail+1))
import os, re, sys
src = sys.argv[1]; rows = []
for line in open(os.path.join(src, "tests/one/switches.txt")):
    line = line.split("#")[0].split()
    if line: rows.append(line[0])
ok_idiom = re.compile(r"ParseDefaultOff|::Parse[A-Z][A-Za-z]*\)|nullptr|'\\0'|strcmp\(|atoi\(|strtou?l|== '1'|\[0\] == '1'|EnvFlag|EnvOn|IsOn\(")
on_idiom = re.compile(r"ParseDefaultOn|== nullptr\)\s*return true|== nullptr \? true")
files = []
for d, _, fs in os.walk(os.path.join(src, "src")):
    files += [os.path.join(d, f) for f in fs if f.endswith((".cpp", ".h", ".inc"))]
text = {f: open(f, errors="replace").read().split("\n") for f in files}
bad = 0
for sw in rows:
    sites = 0
    for f, lines in text.items():
        for i, l in enumerate(lines):
            if '"%s"' % sw not in l: continue
            win = "\n".join(lines[i:i + 7])
            if "getenv" not in win and "Switch" not in win and "Env" not in win: continue
            sites += 1
            if on_idiom.search(win) or not ok_idiom.search(win):
                print("[one] FAIL: %s read at %s:%d is not default-off" % (sw, os.path.relpath(f, src), i + 1)); bad += 1
    if sites == 0:
        print("[one] FAIL: switch %s not read anywhere" % sw); bad += 1
print("[one] %s switch defaults: %d switches scanned" % ("PASS" if bad == 0 else "FAIL", len(rows)))
sys.exit(1 if bad else 0)
PY

sc=$SRC/src/graphics/presentation/window/swapchain.cpp
awk '/Presenter::PrepareBlankFrame\(/{f=1} f&&/present_lock.emplace\(m_impl->present_mutex\)/{print "ok"; exit} f&&/^}/{exit}' "$sc" | grep -q ok \
  && note "PASS startcrash: PrepareBlankFrame takes present_mutex" \
  || { echo " $REQUIRE " | grep -q " startcrash " && bad "startcrash fix missing in PrepareBlankFrame" || note "info startcrash fix not present (not required)"; }

if [ -n "$MSLREF" ] && [ -n "$PROGS" ]; then
  ninja -C "$B" kyty_msl_dump >/dev/null 2>&1 || bad "kyty_msl_dump did not build"
  n=$("$B/kyty_msl_dump" --programs "$PROGS" --list 2>/dev/null | grep -cE "^ *[0-9]+") ; [ "$n" -gt "$MSLMAX" ] && n=$MSLMAX
  same=0; diff=0; T=$(mktemp -d)
  for i in $(seq 0 $((n-1))); do
    "$MSLREF" --programs "$PROGS" --index $i --no-comments > $T/a 2>/dev/null; ra=$?
    "$B/kyty_msl_dump" --programs "$PROGS" --index $i --no-comments > $T/b 2>/dev/null; rb=$?
    if [ $ra = $rb ] && cmp -s $T/a $T/b; then same=$((same+1)); else diff=$((diff+1)); echo "[one] msl differs at index $i"; fi
  done; rm -rf $T
  [ $diff = 0 ] && note "PASS msl emit identity $same/$n" || bad "msl emit identity $same/$n"
fi

[ -f "$LIB" ] && note "codegen fingerprint fp-$(strings -a "$LIB" | awk '/^KytyCodegenFingerprint\/1$/{f=1; print prev; next} f&&/^KYTY_/{exit} f{print} {prev=$0}' | shasum | cut -c1-8)"
[ $fail = 0 ] && { note "ALL PASS"; exit 0; } || { note "$fail failure(s)"; exit 1; }
