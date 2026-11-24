#include "common/inputProfile.h"

#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

namespace InputProfile {

namespace Detail {
static bool EnvOn() {
	const char* v = std::getenv("KYTY_INPUT_PROFILE");
	return v != nullptr && v[0] == '1' && v[1] == '\0';
}
bool g_on = EnvOn();
} // namespace Detail

uint32_t Classify(uint32_t v) noexcept {
	if (v == 0) {
		return kZero;
	}
	if (v == 0x80000000u) {
		return kNegZero;
	}
	if (v <= 255u) {
		return kSmallInt;
	}
	if (v <= 65535u) {
		return kInt16;
	}
	if (v >= 0xffff0000u) {
		return kNegInt;
	}
	const uint32_t exp  = (v >> 23u) & 0xffu;
	const uint32_t mant = v & 0x7fffffu;
	if (exp == 0xffu) {
		return mant != 0 ? kNan : kInf;
	}
	if (exp == 0) {
		return kDenorm;
	}
	const int e = static_cast<int>(exp) - 127;
	if (e < -40 || e > 40) {
		return kFloatBig;
	}
	return (v & 0x80000000u) != 0 ? kFloatNeg : kFloatPos;
}

static float AsFloat(uint32_t v) {
	float f = 0.0f;
	std::memcpy(&f, &v, 4);
	return f;
}

static uint32_t FloatBits(float f) {
	uint32_t v = 0;
	std::memcpy(&v, &f, 4);
	return v;
}

void DwordStat::Add(uint32_t v) noexcept {
	const uint32_t c = Classify(v);
	if (n != 0 && v != last) {
		changes++;
	}
	if ((c & (kFloatPos | kFloatNeg | kFloatBig | kDenorm)) != 0) {
		const float f = AsFloat(v);
		if ((flags & (kFloatPos | kFloatNeg | kFloatBig | kDenorm)) == 0) {
			fmin = fmax = f;
		} else {
			fmin = f < fmin ? f : fmin;
			fmax = f > fmax ? f : fmax;
		}
	}
	flags |= c;
	umin = v < umin ? v : umin;
	umax = v > umax ? v : umax;
	last = v;
	n++;
}

static uint32_t EnvU32(const char* name, uint32_t def) {
	const char* v = std::getenv(name);
	return v != nullptr && *v != '\0' ? static_cast<uint32_t>(std::strtoul(v, nullptr, 0)) : def;
}

Config Config::FromEnv() {
	Config c;
	c.on = Detail::g_on;
	if (const char* o = std::getenv("KYTY_INPUT_PROFILE_OUT"); o != nullptr && *o != '\0') {
		c.out = o;
	}
	c.first        = EnvU32("KYTY_INPUT_PROFILE_FIRST", c.first);
	c.max_samples  = EnvU32("KYTY_INPUT_PROFILE_MAX", c.max_samples);
	c.interval_ms  = EnvU32("KYTY_INPUT_PROFILE_INTERVAL_MS", c.interval_ms);
	c.stat_bytes   = EnvU32("KYTY_INPUT_PROFILE_STAT_KB", c.stat_bytes / 1024) * 1024;
	c.read_bytes   = EnvU32("KYTY_INPUT_PROFILE_KB", c.read_bytes / 1024) * 1024;
	c.raw_bytes    = EnvU32("KYTY_INPUT_PROFILE_RAW_KB", c.raw_bytes / 1024) * 1024;
	c.raw          = EnvU32("KYTY_INPUT_PROFILE_RAW", c.raw);
	c.raw_budget   = uint64_t {EnvU32("KYTY_INPUT_PROFILE_RAW_MB", 128)} * 1024 * 1024;
	c.max_slots    = EnvU32("KYTY_INPUT_PROFILE_MAX_SLOTS", c.max_slots);
	c.flush_s      = EnvU32("KYTY_INPUT_PROFILE_FLUSH_S", c.flush_s);
	c.vertex_count = EnvU32("KYTY_INPUT_PROFILE_VERTICES", c.vertex_count);
	if (c.read_bytes > 64u * 1024u) {
		c.read_bytes = 64u * 1024u;
	}
	c.stat_bytes = c.stat_bytes < c.read_bytes ? c.stat_bytes : c.read_bytes;
	c.raw_bytes  = c.raw_bytes < c.read_bytes ? c.raw_bytes : c.read_bytes;
	return c;
}

Registry::Registry(Config cfg): m_cfg(std::move(cfg)) {}

bool Registry::BeginSample(uint64_t hash, uint32_t stage, uint32_t gate, uint64_t now_ms) {
	std::lock_guard lock(m_mutex);
	auto& g = m_gates[{hash, stage, gate}];
	g.draws++;
	bool take = false;
	if (g.samples < m_cfg.first) {
		take = true;
	} else if (g.samples < m_cfg.max_samples && now_ms - g.last_ms >= m_cfg.interval_ms) {
		take = true;
	}
	if (take) {
		g.samples++;
		g.last_ms = now_ms;
	}
	return take;
}

Slot* Registry::Find(const Key& key) {
	auto it = m_slots.find(key);
	if (it != m_slots.end()) {
		return it->second.get();
	}
	if (m_slots.size() >= m_cfg.max_slots) {
		m_dropped++;
		return nullptr;
	}
	auto [ins, ok] = m_slots.emplace(key, std::make_unique<Slot>());
	return ins->second.get();
}

void Registry::AddMeta(Slot& s, const uint32_t* meta, uint32_t meta_n) {
	if (meta == nullptr || meta_n == 0) {
		return;
	}
	std::vector<uint32_t> m(meta, meta + meta_n);
	for (const auto& x: s.metas) {
		if (x == m) {
			return;
		}
	}
	if (s.metas.size() < 4) {
		s.metas.push_back(std::move(m));
	}
}

void Registry::AddData(uint64_t hash, uint32_t stage, char kind, uint32_t slot, const void* data,
                       size_t bytes, uint64_t bound_bytes, const uint32_t* meta, uint32_t meta_n,
                       uint32_t info, uint32_t stride, uint32_t valid_chunks) {
	std::lock_guard lock(m_mutex);
	Slot* s = Find({hash, stage, kind, slot});
	if (s == nullptr) {
		return;
	}
	s->dirty = true;
	s->samples++;
	s->info |= info;
	s->bytes_max = bound_bytes > s->bytes_max ? bound_bytes : s->bytes_max;
	AddMeta(*s, meta, meta_n);
	if (data == nullptr || bytes < 4) {
		s->unread++;
		return;
	}
	const auto* d     = static_cast<const uint8_t*>(data);
	const auto  valid = [&](size_t byte) {
		const size_t c = byte / kChunkBytes;
		return ((valid_chunks >> (c < 31 ? c : 31)) & 1u) != 0;
	};
	if (stride >= 4 && stride % 4 == 0 && stride <= 1024) {
		s->stride            = stride;
		const size_t per_vtx = stride / 4;
		const size_t words   = bytes / 4;
		if (s->dw.size() < per_vtx) {
			s->dw.resize(per_vtx);
		}
		for (size_t w = 0; w < words; w++) {
			if (!valid(4 * w)) {
				continue;
			}
			uint32_t v = 0;
			std::memcpy(&v, d + 4 * w, 4);
			s->dw[w % per_vtx].Add(v);
		}
	} else {
		const size_t read  = bytes < m_cfg.read_bytes ? bytes : m_cfg.read_bytes;
		const size_t words = read / 4;
		const size_t stat  = (m_cfg.stat_bytes / 4) < words ? m_cfg.stat_bytes / 4 : words;
		if (s->dw.size() < stat) {
			s->dw.resize(stat);
		}
		const size_t blocks = (words - stat + kBlockDwords - 1) / kBlockDwords;
		if (s->blk.size() < blocks) {
			s->blk.resize(blocks);
		}
		for (size_t w = 0; w < words; w++) {
			if (!valid(4 * w)) {
				continue;
			}
			uint32_t v = 0;
			std::memcpy(&v, d + 4 * w, 4);
			if (w < stat) {
				s->dw[w].Add(v);
			} else {
				s->blk[(w - stat) / kBlockDwords].Add(v);
			}
		}
	}
	const size_t raw = bytes < m_cfg.raw_bytes ? bytes : m_cfg.raw_bytes;
	bool         all = raw != 0;
	for (size_t o = 0; o < raw && all; o += kChunkBytes) {
		all = valid(o);
	}
	if (all && s->raw.size() < m_cfg.raw) {
		// A new snapshot only when it differs from the stored ones.
		bool same = false;
		for (const auto& r: s->raw) {
			same = same || (r.size() == raw && std::memcmp(r.data(), d, raw) == 0);
		}
		if (!same && m_raw_bytes.load(std::memory_order_relaxed) + raw <= m_cfg.raw_budget) {
			s->raw.emplace_back(reinterpret_cast<const char*>(d), raw);
			m_raw_bytes.fetch_add(raw, std::memory_order_relaxed);
		}
	}
}

void Registry::AddDescriptor(uint64_t hash, uint32_t stage, char kind, uint32_t slot,
                             const uint32_t* meta, uint32_t meta_n, uint64_t bound_bytes,
                             uint32_t info) {
	std::lock_guard lock(m_mutex);
	Slot* s = Find({hash, stage, kind, slot});
	if (s == nullptr) {
		return;
	}
	const size_t before = s->metas.size();
	s->samples++;
	s->info |= info;
	s->bytes_max = bound_bytes > s->bytes_max ? bound_bytes : s->bytes_max;
	AddMeta(*s, meta, meta_n);
	// Only a new distinct descriptor changes what the line says beyond the counts; counts are
	// refreshed anyway when the slot is serialised for another reason.
	if (s->metas.size() != before || s->samples <= 1) {
		s->dirty = true;
	}
}

std::string Base64(const void* data, size_t bytes) {
	static const char* t = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
	const auto*        p = static_cast<const uint8_t*>(data);
	std::string        o;
	o.reserve((bytes + 2) / 3 * 4);
	size_t i = 0;
	for (; i + 2 < bytes; i += 3) {
		const uint32_t v = (p[i] << 16u) | (p[i + 1] << 8u) | p[i + 2];
		o += t[(v >> 18u) & 63u];
		o += t[(v >> 12u) & 63u];
		o += t[(v >> 6u) & 63u];
		o += t[v & 63u];
	}
	if (i < bytes) {
		uint32_t v = p[i] << 16u;
		if (i + 1 < bytes) {
			v |= p[i + 1] << 8u;
		}
		o += t[(v >> 18u) & 63u];
		o += t[(v >> 12u) & 63u];
		o += i + 1 < bytes ? t[(v >> 6u) & 63u] : '=';
		o += '=';
	}
	return o;
}

static void AppendU(std::string& o, uint64_t v) {
	char b[24];
	std::snprintf(b, sizeof(b), "%" PRIu64, v);
	o += b;
}

void Registry::SerializeSlot(const Key& key, Slot& s) {
	const auto& [hash, stage, kind, slot] = key;
	std::string o;
	o.reserve(64 + s.dw.size() * 40);
	char head[160];
	std::snprintf(head, sizeof(head),
	              "{\"h\":\"%016" PRIx64 "\",\"st\":%u,\"k\":\"%c\",\"s\":%u,\"n\":%u,\"unread\":%u,\"sz\":%" PRIu64
	              ",\"info\":%u,\"stride\":%u",
	              hash, stage, kind, slot, s.samples, s.unread, s.bytes_max, s.info, s.stride);
	o += head;
	o += ",\"meta\":[";
	for (size_t i = 0; i < s.metas.size(); i++) {
		o += i ? ",[" : "[";
		for (size_t j = 0; j < s.metas[i].size(); j++) {
			if (j) {
				o += ',';
			}
			AppendU(o, s.metas[i][j]);
		}
		o += ']';
	}
	o += "]";
	const auto stats = [&o](const std::vector<DwordStat>& v) {
		size_t used = v.size();
		while (used > 0 && v[used - 1].n == 0) {
			used--;
		}
		for (size_t i = 0; i < used; i++) {
			const auto& d = v[i];
			if (i) {
				o += ',';
			}
			o += '[';
			AppendU(o, d.flags);
			o += ',';
			AppendU(o, d.umin);
			o += ',';
			AppendU(o, d.umax);
			o += ',';
			AppendU(o, FloatBits(d.fmin));
			o += ',';
			AppendU(o, FloatBits(d.fmax));
			o += ',';
			AppendU(o, d.changes);
			o += ',';
			AppendU(o, d.n);
			o += ']';
		}
	};
	// Trailing never-seen positions trimmed. Per position/block: flags,umin,umax,fmin bits,fmax bits,changes,n.
	o += ",\"dw\":[";
	stats(s.dw);
	o += "],\"blk\":[";
	stats(s.blk);
	o += "],\"raw\":[";
	for (size_t i = 0; i < s.raw.size(); i++) {
		o += i ? ",\"" : "\"";
		o += Base64(s.raw[i].data(), s.raw[i].size());
		o += '"';
	}
	o += "]}\n";
	s.line  = std::move(o);
	s.dirty = false;
}

std::string Registry::Serialize() {
	std::vector<Key> keys;
	std::string      out;
	{
		std::lock_guard lock(m_mutex);
		keys.reserve(m_slots.size());
		for (const auto& [k, v]: m_slots) {
			keys.push_back(k);
		}
		char head[256];
		std::snprintf(head, sizeof(head),
		              "{\"k\":\"hdr\",\"version\":1,\"first\":%u,\"max\":%u,\"interval_ms\":%u,\"read_bytes\":%u,"
		              "\"stat_bytes\":%u,\"block_dwords\":%u,\"slots\":%zu,\"dropped\":%u,\"raw_bytes\":%" PRIu64 "}\n",
		              m_cfg.first, m_cfg.max_samples, m_cfg.interval_ms, m_cfg.read_bytes, m_cfg.stat_bytes, kBlockDwords, m_slots.size(),
		              m_dropped, RawBytes());
		out += head;
		for (auto& [k, g]: m_gates) {
			const auto& [hash, stage, gate] = k;
			char line[160];
			std::snprintf(line, sizeof(line),
			              "{\"h\":\"%016" PRIx64 "\",\"st\":%u,\"k\":\"P\",\"gate\":%u,\"draws\":%" PRIu64 ",\"n\":%u}\n",
			              hash, stage, gate, g.draws, g.samples);
			out += line;
		}
	}
	// One slot at a time under the lock: draws wait at most one slot's serialisation.
	for (const auto& k: keys) {
		std::lock_guard lock(m_mutex);
		auto&           s = *m_slots[k];
		if (s.dirty || s.line.empty()) {
			SerializeSlot(k, s);
		}
		out += s.line;
	}
	return out;
}

bool Registry::WriteFile(const std::string& path) {
	const std::string text = Serialize();
	const std::string tmp  = path + ".tmp";
	FILE*             f    = std::fopen(tmp.c_str(), "wb");
	if (f == nullptr) {
		return false;
	}
	const bool ok = std::fwrite(text.data(), 1, text.size(), f) == text.size();
	std::fclose(f);
	if (!ok || std::rename(tmp.c_str(), path.c_str()) != 0) {
		return false;
	}
	return true;
}

size_t Registry::SlotCount() {
	std::lock_guard lock(m_mutex);
	return m_slots.size();
}

uint64_t NowMs() noexcept {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}

static Registry* g_registry = nullptr;

static void FlushNow(const char* why) {
	if (g_registry == nullptr) {
		return;
	}
	const auto t0 = NowMs();
	const bool ok = g_registry->WriteFile(g_registry->Cfg().out);
	std::fprintf(stderr, "[inprof] %s flush %s: %zu slots, raw %" PRIu64 " KiB, %" PRIu64 " ms\n", why,
	             ok ? "ok" : "FAILED", g_registry->SlotCount(), g_registry->RawBytes() / 1024, NowMs() - t0);
}

Registry& Global() {
	static Registry* reg = [] {
		auto* r    = new Registry(Config::FromEnv()); // never destroyed: the exit flush may run late
		g_registry = r;
		std::fprintf(stderr,
		             "[inprof] KYTY_INPUT_PROFILE on (LOCAL research capture): out %s, first %u, max %u, "
		             "interval %u ms, read %u B, raw %u, flush %u s\n",
		             r->Cfg().out.c_str(), r->Cfg().first, r->Cfg().max_samples, r->Cfg().interval_ms,
		             r->Cfg().read_bytes, r->Cfg().raw, r->Cfg().flush_s);
		std::atexit([] { FlushNow("exit"); });
		if (r->Cfg().flush_s > 0) {
			std::thread([s = r->Cfg().flush_s] {
				for (;;) {
					std::this_thread::sleep_for(std::chrono::seconds(s));
					FlushNow("periodic");
				}
			}).detach();
		}
		return r;
	}();
	return *reg;
}

} // namespace InputProfile
