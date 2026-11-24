// KPRC loader tests (tools/msl-dump/kprcLoader.h): plain key, SAROS variant key, truncated record, absurd counts, subset writer.
// Standalone (std only): c++ -std=c++20 -I tools/msl-dump tests/KprcLoaderTests.cpp -o kprc_loader_tests
#include "kprcLoader.h"

#include <cstdio>

using namespace Kprc;
static int g_fail = 0, g_n = 0;
static void Check(bool ok, const char *what) { g_n++; if (!ok) { g_fail++; std::printf("FAIL %s\n", what); } }

template <class T> static void Put(std::vector<uint8_t> &o, T v) { auto *b = reinterpret_cast<const uint8_t *>(&v); o.insert(o.end(), b, b + sizeof(T)); }

static std::vector<uint8_t> Key(uint64_t hash, std::vector<uint32_t> code, std::vector<uint32_t> back, bool variant, uint64_t var = 0,
                                uint32_t code_size_field = 0xffffffffu, bool tail = true) {
  std::vector<uint8_t> k;
  Put<uint32_t>(k, 1); Put<uint32_t>(k, 4); Put<uint64_t>(k, hash); Put<uint32_t>(k, 16);
  Put<uint32_t>(k, code_size_field == 0xffffffffu ? static_cast<uint32_t>(code.size()) : code_size_field);
  Put<uint32_t>(k, 3); for (uint32_t i = 0; i < 3; i++) Put<uint32_t>(k, 100 + i);
  Put<uint32_t>(k, 64); Put<uint32_t>(k, 0); Put<uint8_t>(k, 0);
  if (variant) Put<uint64_t>(k, var);
  Put<uint32_t>(k, static_cast<uint32_t>(code.size())); for (auto w : code) Put<uint32_t>(k, w);
  if (tail) { Put<uint32_t>(k, static_cast<uint32_t>(back.size())); for (auto w : back) Put<uint32_t>(k, w); }
  return k;
}
static void Rec(std::vector<uint8_t> &f, uint32_t kind, const std::vector<uint8_t> &payload) {
  Put<uint32_t>(f, kMagic); Put<uint32_t>(f, kind); Put<uint64_t>(f, payload.size()); Put<uint64_t>(f, 0);
  f.insert(f.end(), payload.begin(), payload.end());
}
static std::vector<uint8_t> SrcPayload(uint8_t dig, const std::vector<uint8_t> &key) {
  std::vector<uint8_t> p(16, dig); Put<uint32_t>(p, static_cast<uint32_t>(key.size())); p.insert(p.end(), key.begin(), key.end());
  Put<uint8_t>(p, 0); Put<uint32_t>(p, 0); return p;
}
static std::vector<uint8_t> PermPayload(uint8_t dig, uint32_t cursor, std::vector<uint8_t> spec) {
  std::vector<uint8_t> p(16, dig); Put<uint32_t>(p, cursor); Put<uint32_t>(p, static_cast<uint32_t>(spec.size())); p.insert(p.end(), spec.begin(), spec.end());
  Put<uint32_t>(p, 0); Put<uint32_t>(p, 0); Put<uint32_t>(p, 0); return p;
}
static std::vector<uint8_t> Head() {
  std::vector<uint8_t> f(16, 0); const char *s = "hdr"; Put<uint64_t>(f, 3); f.insert(f.end(), s, s + 3); Put<uint64_t>(f, 0); return f;
}

int main() {
  // plain key
  { auto f = Head(); Rec(f, 1, SrcPayload(1, Key(0xaa, {1, 2, 3}, {9}, false))); Rec(f, 2, PermPayload(1, 7, {5, 6}));
    std::vector<Perm> pm; std::string e; auto s = Parse(f, &pm, &e);
    Check(e.empty() && s.size() == 1 && pm.size() == 1, "plain loads");
    Check(s[0].hash == 0xaa && !s[0].has_variant && s[0].code == std::vector<uint32_t>({1, 2, 3}) && s[0].back_code == std::vector<uint32_t>({9}), "plain fields");
    Check(s[0].statics.size() == 3 && s[0].wave == 64, "plain statics/wave");
    Check(pm[0].cursor == 7 && pm[0].spec.size() == 2 && pm[0].perm == 0, "perm fields"); }
  // variant key (the 8-byte SarosCacheVariant before the code)
  { auto f = Head(); Rec(f, 1, SrcPayload(2, Key(0xbb, {10, 11}, {}, true, 0x1122334455667788ull))); Rec(f, 2, PermPayload(2, 0, {}));
    std::vector<Perm> pm; std::string e; auto s = Parse(f, &pm, &e);
    Check(e.empty() && s.size() == 1, "variant loads");
    Check(s[0].has_variant && s[0].variant == 0x1122334455667788ull && s[0].code == std::vector<uint32_t>({10, 11}), "variant fields"); }
  // variant whose low word happens to equal code_size: layout resolved by exact key consumption
  { auto k = Key(0xcc, {1, 2}, {}, true, 2); std::string e; Src s; Check(ParseSourceKey(k.data(), k.size(), s, &e) && s.has_variant && s.code.size() == 2, "variant low word == code_size"); }
  // key without the back-code tail (old caches)
  { auto k = Key(0xdd, {1, 2}, {}, false, 0, 0xffffffffu, false); Src s; Check(ParseSourceKey(k.data(), k.size(), s) && s.code.size() == 2, "no tail"); }
  // truncated key / absurd counts: error, no crash
  { auto k = Key(0xee, {1, 2, 3}, {}, false); k.resize(k.size() - 9); Src s; std::string e; Check(!ParseSourceKey(k.data(), k.size(), s, &e) && !e.empty(), "truncated key"); }
  { auto k = Key(0xee, {1, 2, 3}, {}, false); uint32_t big = 0xfffffff0u; std::memcpy(&k[28 + 12 + 8 + 1], &big, 4); Src s; Check(!ParseSourceKey(k.data(), k.size(), s), "absurd code count"); }
  { auto k = Key(0xee, {1, 2, 3}, {}, false); uint32_t big = 0xfffffff0u; std::memcpy(&k[24], &big, 4); Src s; Check(!ParseSourceKey(k.data(), k.size(), s), "absurd static count"); }
  { std::vector<uint8_t> k(10, 0); Src s; Check(!ParseSourceKey(k.data(), k.size(), s), "tiny key"); }
  // file level: record size beyond the file, key length beyond the record, header string beyond the file, short file
  { auto f = Head(); Rec(f, 1, SrcPayload(1, Key(1, {1}, {}, false))); f.resize(f.size() - 5); std::string e; auto s = Parse(f, nullptr, &e); Check(s.empty() && !e.empty(), "truncated record"); }
  { auto f = Head(); auto p = SrcPayload(1, Key(1, {1}, {}, false)); uint32_t big = 0x7fffffffu; std::memcpy(&p[16], &big, 4); Rec(f, 1, p); std::string e; Parse(f, nullptr, &e); Check(!e.empty(), "key length beyond record"); }
  { auto f = Head(); uint64_t big = ~0ull; std::memcpy(&f[16], &big, 8); std::string e; Parse(f, nullptr, &e); Check(!e.empty(), "header length absurd"); }
  { std::vector<uint8_t> f(8); std::string e; Parse(f, nullptr, &e); Check(!e.empty(), "short file"); }
  { auto f = Head(); Rec(f, 2, PermPayload(1, 0, {1})); std::vector<Perm> pm; std::string e; Parse(f, &pm, &e); Check(e.empty() && pm.size() == 1, "perm without source parses"); }
  // subset writer: the one-permutation file loads on its own and keeps its variant
  { auto f = Head(); Rec(f, 1, SrcPayload(1, Key(0x11, {1}, {}, false))); Rec(f, 1, SrcPayload(2, Key(0x22, {7, 8}, {}, true, 5)));
    Rec(f, 2, PermPayload(1, 0, {})); Rec(f, 2, PermPayload(2, 3, {4}));
    std::vector<Perm> pm; std::string e; size_t hl = 0; auto s = Parse(f, &pm, &e, &hl);
    std::vector<uint8_t> sub; Check(BuildSubset(f, hl, s, pm, 1, sub, &e), "subset built");
    std::vector<Perm> pm2; auto s2 = Parse(sub, &pm2, &e);
    Check(e.empty() && s2.size() == 1 && pm2.size() == 1 && s2[0].hash == 0x22 && s2[0].has_variant && pm2[0].cursor == 3 && pm2[0].perm == 0, "subset reloads");
    Check(!BuildSubset(f, hl, s, pm, 9, sub, &e), "subset: no such perm"); }
  std::printf("%d checks, %d failed\n", g_n, g_fail);
  return g_fail ? 1 : 0;
}
