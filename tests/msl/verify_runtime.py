#!/usr/bin/env python3
"""Checks outputs of the T2 runtime kernels (run via mslh) against numpy references.
usage: verify_runtime.py <dir with r_*.metal/json> <mslh> [seed]"""
import struct, subprocess, sys, os, math
d, mslh = sys.argv[1], sys.argv[2]
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1
M = (1 << 64) - 1
def sm64(s):
    s[0] = (s[0] + 0x9E3779B97F4A7C15) & M
    z = s[0]; z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M; z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M
    return z ^ (z >> 31)
def seeded(name, n):
    h = 1469598103934665603 ^ seed
    for c in name.encode(): h ^= c; h = (h * 1099511628211) & M
    st = [h]; return [sm64(st) & 0xffffffff for _ in range(n // 4)]
def rd(path): return list(struct.unpack("<%dI" % (os.path.getsize(path) // 4), open(path, "rb").read()))
def run(name):
    subprocess.run([mslh, "run", "--msl", f"{d}/{name}.metal", "--manifest", f"{d}/{name}.json", "--seed", str(seed), "--out", f"{d}/o_{name}"], check=True, capture_output=True)
    return rd(f"{d}/o_{name}/b0.bin")
def f32(x): return struct.unpack("<f", struct.pack("<f", x))[0]
def f2u(x): return struct.unpack("<I", struct.pack("<f", x))[0]
bad = 0
def cmp(tag, got, exp):
    global bad
    got = [int(x) & 0xffffffff for x in got]; exp = [int(x) & 0xffffffff for x in exp]
    ok = got == exp
    print(("PASS " if ok else "FAIL ") + tag)
    if not ok:
        bad += 1
        i = [k for k in range(len(got)) if got[k] != exp[k]][:4]; print("   first diffs idx", i, "got", [hex(got[k]) for k in i], "exp", [hex(exp[k]) for k in i])
b1w = seeded("b1", 4096)
b1 = b"".join(struct.pack("<I", x) for x in b1w)
def sel(c, count): return [c[(s - 4) % count] for s in (4, 5, 6, 7)]
def half2f(u): return struct.unpack("<e", struct.pack("<H", u))[0]
def uf(raw, mb):
    e = (raw >> mb) & 31; m = raw & ((1 << mb) - 1)
    if e == 0: return 0.0 if m == 0 else math.ldexp(float(m), -14 - mb)
    if e == 31: return math.inf if m == 0 else math.nan
    return math.ldexp(float((1 << mb) | m), e - 15 - mb)
def s8(x): return x - 256 if x > 127 else x
def s16(x): return x - 65536 if x > 32767 else x
for name in ["8888unorm", "8888snorm", "1616float", "16uint", "88sint", "3232float", "11_11_10float", "2_10_10_10unorm", "16161616snorm"]:
    got = run("r_fmt_" + name)[:128]
    exp = []
    for t in range(32):
        e = b1[t * 16:t * 16 + 16]; w = struct.unpack("<4I", e); h = struct.unpack("<8H", e)
        if name == "8888unorm": cc = [f2u(f32(f32(x) * f32(1.0 / 255.0))) for x in e[:4]]
        elif name == "8888snorm": cc = [f2u(max(f32(f32(s8(x)) * f32(1.0 / 127.0)), -1.0)) for x in e[:4]]
        elif name == "1616float": cc = [f2u(f32(half2f(h[i]))) for i in range(2)]
        elif name == "16uint": cc = [h[0]]
        elif name == "88sint": cc = [s8(e[i]) & 0xffffffff for i in range(2)]
        elif name == "3232float": cc = [w[0], w[1]]
        elif name == "11_11_10float": cc = [f2u(f32(uf(w[0] & 0x7ff, 6))), f2u(f32(uf((w[0] >> 11) & 0x7ff, 6))), f2u(f32(uf((w[0] >> 22) & 0x3ff, 5)))]
        elif name == "2_10_10_10unorm": cc = [f2u(f32(f32(w[0] & 3) * f32(1.0 / 3.0)))] + [f2u(f32(f32((w[0] >> sh) & 1023) * f32(1.0 / 1023.0))) for sh in (2, 12, 22)]
        elif name == "16161616snorm": cc = [f2u(max(f32(f32(s16(h[i])) * f32(1.0 / 32767.0)), -1.0)) for i in range(4)]
        exp += sel(cc, len(cc))
    cmp("format " + name, got, exp)
o = run("r_wave")[:32 * 10]
v = lambda l: 3 * l + 1
exp = []; g = []
for l in range(32):
    g += o[l * 10:l * 10 + 9]
    exp += [v(5), v(0), 0xAAAAAAAA, v(l - 1) if (l & 15) >= 1 else 999, v((l & ~3) | ((0x1b >> (2 * (l & 3))) & 3)), v(l ^ 7), v(l ^ 1), v(31 - l), v((l - 1) & 31)]
cmp("wave ops (readlane/readfirst/ballot/dpp row_shr/quad_perm/half_mirror/swizzle/bpermute/permute)", g, exp)
raw = run("r_lds")
exp = []
for t in range(32): exp += [((t + 1) & 31) * 7, 528, 93, (-31) & 0xffffffff, 2, 0, 248, 0]
cmp("lds load/store/atomics(add,umax,smin,inc,add64 carry)/oob", raw[:256], exp)
cmp("buffer atomics add/or/add64", [raw[256], raw[257], raw[260], raw[261]], [32, 0xffffffff, 0xffffffe0, 31])
print("cmpswap winner:", raw[258], "(expect 100..131)")
if not (100 <= raw[258] <= 131): bad += 1
sys.exit(1 if bad else 0)
