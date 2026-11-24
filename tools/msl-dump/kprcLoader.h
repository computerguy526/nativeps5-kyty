// kprcLoader.h: bounds-checked reader of Kyty's persistent program cache (KPRC, programDiskCache.cpp) for kyty_msl_dump.
// Header-only, std only (unit-tested by tests/KprcLoaderTests.cpp without the recompiler).
// Record = magic u32 'KPRC', kind u32 (1 source, 2 permutation), size u64, checksum u64, payload[size].
// Source payload = digest[16], key (u32 len + bytes), skip u8, plan. Source key (BuildSourceKey):
//   u32 layout, u32 stage, u64 hash, u32 user_data_count, u32 code_size, u32 ns + ns words (static state), u32 wave, u32 user_data_base,
//   u8 plain_mip_stats_variant, [u64 SarosCacheVariant: only when != 0], u32 nc + nc code words, u32 nb + nb back-code words.
// Permutation payload = digest[16], cursor u32, specialization (u32 len + bytes), info, spirv ...
// Every length is checked against the record: a damaged file yields an error string, never a crash.
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace Kprc {

struct Src {
  uint32_t stage = 0, wave = 64, user_data_count = 0, user_data_base = 0;
  uint64_t hash = 0;
  uint64_t variant = 0;      // SarosCacheVariant (0 = key without the variant word)
  bool has_variant = false;
  std::vector<uint32_t> statics, code;
  std::vector<uint32_t> back_code; // meshpath: merged GS back half (key tail)
  std::string digest;
  size_t rec_off = 0, rec_len = 0; // the whole record in the file (subset writer)
};
struct Perm {
  int perm = 0;
  std::string digest;
  std::vector<uint8_t> spec;
  uint32_t cursor = 0; // push-data cursor the permutation was compiled with
  size_t rec_off = 0, rec_len = 0;
};

constexpr uint32_t kMagic = 0x4352504bu;

namespace detail {
struct Rd {
  const uint8_t *p;
  size_t n, pos = 0;
  bool bad = false;
  Rd(const uint8_t *b, size_t len) : p(b), n(len) {}
  template <class T> T get() {
    T v{};
    if (bad || sizeof(T) > n - pos) { bad = true; return v; }
    std::memcpy(&v, p + pos, sizeof(T));
    pos += sizeof(T);
    return v;
  }
  bool words(uint64_t count, std::vector<uint32_t> &out) {
    if (bad || count > (n - pos) / 4) { bad = true; return false; }
    out.resize(static_cast<size_t>(count));
    if (count) std::memcpy(out.data(), p + pos, static_cast<size_t>(count) * 4);
    pos += static_cast<size_t>(count) * 4;
    return true;
  }
};

// One layout attempt: with (variant=true) or without the u64 variant word before the code count.
inline bool TryKey(const uint8_t *key, size_t kl, uint32_t code_size, bool variant, Src &s) {
  Rd r(key, kl);
  r.get<uint32_t>(); // layout
  s.stage = r.get<uint32_t>();
  s.hash = r.get<uint64_t>();
  s.user_data_count = r.get<uint32_t>();
  r.get<uint32_t>(); // code_size (compared by the caller)
  const uint32_t ns = r.get<uint32_t>();
  if (!r.words(ns, s.statics)) return false;
  s.wave = r.get<uint32_t>();
  s.user_data_base = r.get<uint32_t>();
  r.get<uint8_t>();
  s.has_variant = variant;
  s.variant = 0;
  if (variant) s.variant = r.get<uint64_t>();
  const uint32_t nc = r.get<uint32_t>();
  if (r.bad || nc != code_size) return false;
  if (!r.words(nc, s.code)) return false;
  s.back_code.clear();
  if (r.pos == kl) return true; // old keys: no back-code tail
  const uint32_t nb = r.get<uint32_t>();
  if (!r.words(nb, s.back_code)) return false;
  return r.pos == kl; // the key must be consumed exactly (tells the variant layout from the plain one)
}
} // namespace detail

// Parse one source key. Plain layout first; the variant layout when the word at the code position is not code_size.
inline bool ParseSourceKey(const uint8_t *key, size_t kl, Src &s, std::string *err = nullptr) {
  if (kl < 28) { if (err) *err = "source key too short"; return false; }
  uint32_t code_size;
  std::memcpy(&code_size, key + 20, 4);
  Src a;
  if (detail::TryKey(key, kl, code_size, false, a)) { s = std::move(a); return true; }
  Src b;
  if (detail::TryKey(key, kl, code_size, true, b)) { s = std::move(b); return true; }
  if (err) *err = "source key does not parse as plain or variant layout";
  return false;
}

// Parse a whole file image. Stops at the first damaged record and reports it in *err (records before it stay loaded).
inline std::vector<Src> Parse(const std::vector<uint8_t> &d, std::vector<Perm> *perms, std::string *err, size_t *head_len = nullptr) {
  std::vector<Src> out;
  auto fail = [&](const std::string &m) { if (err) *err = m; return out; };
  if (d.size() < 32) return fail("file too short");
  uint64_t n;
  std::memcpy(&n, &d[16], 8);
  if (n > d.size() - 32) return fail("header string length beyond the file");
  size_t off = 16 + 8 + static_cast<size_t>(n) + 8;
  if (off > d.size()) return fail("header beyond the file");
  if (head_len) *head_len = off;
  int perm_idx = 0;
  while (off + 24 <= d.size()) {
    uint32_t magic, kind;
    uint64_t size;
    std::memcpy(&magic, &d[off], 4);
    std::memcpy(&kind, &d[off + 4], 4);
    std::memcpy(&size, &d[off + 8], 8);
    if (magic != kMagic) break; // end of the record stream (the file may carry a tail)
    if (size > d.size() - off - 24) return fail("record " + std::to_string(perm_idx) + " at " + std::to_string(off) + ": size beyond the file");
    const size_t p = off + 24, end = p + static_cast<size_t>(size);
    if (kind == 1) {
      if (size < 20) return fail("source record too short");
      uint32_t kl;
      std::memcpy(&kl, &d[p + 16], 4);
      if (kl > size - 20) return fail("source key length beyond the record");
      Src s;
      std::string e;
      if (!ParseSourceKey(&d[p + 20], kl, s, &e)) return fail("source at " + std::to_string(off) + ": " + e);
      s.digest.assign(reinterpret_cast<const char *>(&d[p]), 16);
      s.rec_off = off;
      s.rec_len = end - off;
      out.push_back(std::move(s));
    } else {
      if (perms) {
        if (size < 24) return fail("permutation record too short");
        Perm pm;
        pm.perm = perm_idx;
        pm.digest.assign(reinterpret_cast<const char *>(&d[p]), 16);
        uint32_t sl;
        std::memcpy(&sl, &d[p + 20], 4);
        if (sl > size - 24) return fail("specialization length beyond the record");
        pm.spec.assign(&d[p + 24], &d[p + 24] + sl);
        std::memcpy(&pm.cursor, &d[p + 16], 4);
        pm.rec_off = off;
        pm.rec_len = end - off;
        perms->push_back(std::move(pm));
      }
      perm_idx++;
    }
    off = end;
  }
  return out;
}

inline bool ReadFile(const char *path, std::vector<uint8_t> &d) {
  std::FILE *fp = std::fopen(path, "rb");
  if (!fp) return false;
  std::fseek(fp, 0, SEEK_END);
  const long sz = std::ftell(fp);
  std::fseek(fp, 0, SEEK_SET);
  if (sz < 0) { std::fclose(fp); return false; }
  d.resize(static_cast<size_t>(sz));
  const size_t got = sz ? std::fread(d.data(), 1, d.size(), fp) : 0;
  std::fclose(fp);
  return got == d.size();
}

// The one-permutation subset file: header + the permutation's source record + the permutation record.
// Readable by --programs OUT --perm 0. Returns false (and *err) when the permutation or its source is missing.
inline bool BuildSubset(const std::vector<uint8_t> &d, size_t head_len, const std::vector<Src> &srcs, const std::vector<Perm> &perms,
                        int perm, std::vector<uint8_t> &out, std::string *err) {
  for (const auto &pm : perms) {
    if (pm.perm != perm) continue;
    for (const auto &s : srcs) {
      if (s.digest != pm.digest) continue;
      out.assign(d.begin(), d.begin() + static_cast<std::ptrdiff_t>(head_len));
      out.insert(out.end(), d.begin() + static_cast<std::ptrdiff_t>(s.rec_off), d.begin() + static_cast<std::ptrdiff_t>(s.rec_off + s.rec_len));
      out.insert(out.end(), d.begin() + static_cast<std::ptrdiff_t>(pm.rec_off), d.begin() + static_cast<std::ptrdiff_t>(pm.rec_off + pm.rec_len));
      return true;
    }
    if (err) *err = "permutation " + std::to_string(perm) + " has no source record";
    return false;
  }
  if (err) *err = "no such perm " + std::to_string(perm);
  return false;
}

} // namespace Kprc
