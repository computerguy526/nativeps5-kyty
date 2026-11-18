#include "graphics/host_gpu/renderer/flightRecDraw.h"

#include "common/guestAddress.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/image/image.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/shader.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace Libs::Graphics::FlightRecDraw {

namespace {

// pixid-2's helmet material pipelines (NOTES 2026-11-18 12:44), vs and ps.
constexpr const char* kHelmetKeys[] = {"06c0b6a2", "b96c2898", "7644cdfe", "5401c251", "a0a69f9b",
                                       "c1af3275", "4e28f908", "c6469900", "2335532b", "4d72ec7f"};

struct Key {
	uint64_t value;
	uint64_t mask;
};
struct Config {
	std::vector<Key> keys;
	uint64_t         hash_bytes = 1024 * 1024;
	uint32_t         rehash     = 3;
};

void AddKey(Config& c, const char* text, size_t n) {
	if (n == 0 || n > 16) {
		return;
	}
	char buf[17] {};
	std::memcpy(buf, text, n);
	char*      end = nullptr;
	const auto v   = std::strtoull(buf, &end, 16);
	if (end != buf + n) {
		return;
	}
	const unsigned shift = static_cast<unsigned>(64 - 4 * n);
	c.keys.push_back({v << shift, n == 16 ? ~0ull : ~((1ull << shift) - 1)});
}

const Config& Cfg() {
	static const Config cfg = [] {
		Config      c;
		const char* keys = std::getenv("KYTY_FLIGHTREC_KEYS");
		if (keys != nullptr) {
			std::string s(keys);
			size_t      pos = 0;
			while (pos <= s.size()) {
				auto end = s.find(',', pos);
				if (end == std::string::npos) {
					end = s.size();
				}
				const std::string k = s.substr(pos, end - pos);
				if (k == "helmet") {
					for (const char* h: kHelmetKeys) {
						AddKey(c, h, std::strlen(h));
					}
				} else {
					AddKey(c, k.c_str(), k.size());
				}
				pos = end + 1;
			}
		}
		if (const char* kb = std::getenv("KYTY_FLIGHTREC_HASH_KB"); kb != nullptr && *kb != '\0') {
			c.hash_bytes = std::strtoull(kb, nullptr, 0) * 1024;
		}
		if (const char* r = std::getenv("KYTY_FLIGHTREC_REHASH"); r != nullptr && *r != '\0') {
			c.rehash = static_cast<uint32_t>(std::strtoul(r, nullptr, 0));
		}
		if (!c.keys.empty()) {
			std::fprintf(stderr, "[flightrec] %zu draw keys, hash %llu KiB, rehash %u flips\n", c.keys.size(),
			             static_cast<unsigned long long>(c.hash_bytes / 1024), c.rehash);
		}
		return c;
	}();
	return cfg;
}

bool Match(const Config& c, uint64_t hash) {
	for (const auto& k: c.keys) {
		if ((hash & k.mask) == k.value) {
			return true;
		}
	}
	return false;
}

struct Pending {
	uint64_t addr;
	uint64_t size;
	uint64_t hash;
	uint32_t flip;
};
std::mutex           g_mutex;
std::vector<Pending> g_pending;

thread_local uint32_t t_flip       = 0;
thread_local uint32_t t_index      = 0;
thread_local uint32_t t_draws      = 0;
thread_local uint32_t t_keyed_n    = 0;
thread_local uint32_t t_rehash_n   = 0;

uint64_t GuestHash(uint64_t addr, uint64_t size, uint64_t cap) {
	if (addr == 0 || size == 0) {
		return 0;
	}
	return FlightRec::Hash(Guest::Ptr<const uint8_t>(addr), size < cap ? size : cap);
}

void NewFlip(uint32_t flip, const Config& cfg) {
	FlightRec::Rec(FlightRec::EvFrameSum, t_flip, t_draws, t_keyed_n, t_rehash_n);
	t_flip     = flip;
	t_index    = 0;
	t_draws    = 0;
	t_keyed_n  = 0;
	t_rehash_n = 0;
	if (cfg.keys.empty()) {
		return;
	}
	std::lock_guard lock(g_mutex);
	size_t          keep = 0;
	for (auto& p: g_pending) {
		const uint32_t age = flip - p.flip;
		if (age == 0) {
			g_pending[keep++] = p;
			continue;
		}
		const uint64_t now = GuestHash(p.addr, p.size, cfg.hash_bytes);
		t_rehash_n++;
		if (now != p.hash) {
			FlightRec::Rec(FlightRec::EvRehash, p.addr, p.size, p.hash, now, static_cast<uint8_t>(age));
			continue; // changed: reported once
		}
		if (age < cfg.rehash) {
			g_pending[keep++] = p;
		}
	}
	g_pending.resize(keep);
}

uint64_t Id(Common::SlotId id) {
	return id ? (static_cast<uint64_t>(id.index) << 32u) | (id.generation & 0x3fffffffu) : 0;
}

} // namespace

void Begin(char kind, std::span<PreparedBindings* const> stages, uint64_t tick) noexcept {
	if (!FlightRec::On()) [[likely]] {
		return;
	}
	const auto&    cfg  = Cfg();
	const uint32_t flip = FlightRec::Detail::g_flip.load(std::memory_order_relaxed);
	if (flip != t_flip) {
		NewFlip(flip, cfg);
	}
	if (cfg.keys.empty() && !FlightRec::Verbose()) [[likely]] {
		// Light mode: draws are only tallied (EvFrameSum per flip + the 1 Hz counters).
		t_draws++;
		FlightRec::Detail::g_count[kind == 'X' ? FlightRec::EvDispatch : FlightRec::EvDraw].store(
		    FlightRec::Detail::g_count[kind == 'X' ? FlightRec::EvDispatch : FlightRec::EvDraw].load(std::memory_order_relaxed) + 1,
		    std::memory_order_relaxed);
		return;
	}
	uint64_t hashes[2] = {0, 0};
	bool     keyed     = false;
	int      n         = 0;
	for (auto* stage: stages) {
		if (stage == nullptr || stage->runtime == nullptr || !*stage->runtime) {
			continue;
		}
		const uint64_t h = stage->runtime->program->shader_hash;
		if (n < 2) {
			hashes[n] = h;
		}
		n++;
		keyed = keyed || (!cfg.keys.empty() && Match(cfg, h));
	}
	// Draws: the last two stages are vs, ps (a single stage: ps 0).
	if (n > 2) {
		hashes[0] = stages[stages.size() - 2] != nullptr && stages[stages.size() - 2]->runtime != nullptr &&
		                    *stages[stages.size() - 2]->runtime
		                ? stages[stages.size() - 2]->runtime->program->shader_hash
		                : 0;
		hashes[1] = stages.back() != nullptr && stages.back()->runtime != nullptr && *stages.back()->runtime
		                ? stages.back()->runtime->program->shader_hash
		                : 0;
	}
	const uint64_t index = t_index++;
	t_draws++;
	if (keyed) {
		t_keyed_n++;
	}
	FlightRec::Rec(kind == 'X' ? FlightRec::EvDispatch : FlightRec::EvDraw, hashes[0], hashes[1],
	               (index << 32u) | (keyed ? 1ull << 31u : 0) | static_cast<uint64_t>(n), tick);
	Detail::t_keyed = keyed;
}

void End(TextureCache& cache, std::span<PreparedBindings* const> stages,
         std::span<const RenderColorInfo> colors) noexcept {
	if (!Detail::t_keyed) [[likely]] {
		return;
	}
	Detail::t_keyed = false;
	const auto&    cfg  = Cfg();
	const uint32_t flip = FlightRec::Detail::g_flip.load(std::memory_order_relaxed);
	std::vector<Pending> fresh;
	for (auto* stage: stages) {
		if (stage == nullptr || stage->runtime == nullptr || !*stage->runtime) {
			continue;
		}
		const auto     st      = static_cast<uint64_t>(stage->runtime->program->stage) & 0xff;
		const auto&    program = *stage->runtime->program;
		for (size_t slot = 0; slot < stage->buffer_sources.size(); slot++) {
			const auto& src = stage->buffer_sources[slot];
			if (src.address == 0) {
				continue;
			}
			const bool     written = slot < program.info.buffers.size() && program.info.buffers[slot].written;
			const uint64_t h       = GuestHash(src.address, src.size, cfg.hash_bytes);
			FlightRec::Rec(FlightRec::EvBind, src.address,
			               (src.size & 0xffffffffffffull) | (st << 56u) | ((slot & 0xff) << 48u),
			               Id(src.id) | (written ? 1ull << 30u : 0), h);
			if (!written) {
				fresh.push_back({src.address, src.size, h, flip});
			}
		}
		for (size_t b = 0; b < stage->images.size(); b++) {
			const auto& binding = stage->images[b];
			auto*       image   = binding.image_id ? cache.TryGetImage(binding.image_id) : nullptr;
			const bool  storage = binding.desc.type == TextureCache::BindingType::Storage;
			uint64_t    addr = 0, fmt = 0;
			if (image != nullptr) {
				addr = image->info.data.address;
				fmt  = (static_cast<uint64_t>(image->info.pixel_format) << 32u) |
				      ((image->info.extent.width & 0xffffu) << 16u) | (image->info.extent.height & 0xffffu);
			}
			FlightRec::Rec(FlightRec::EvTexture, addr,
			               (st << 56u) | (storage ? 1ull << 55u : 0) | ((b & 0x7f) << 48u) |
			                   (static_cast<uint64_t>(binding.image_id.index & 0xffffffffu) << 16u) |
			                   (binding.image_id.generation & 0xffffu),
			               fmt, (static_cast<uint64_t>(binding.desc.view_info.base_level) << 16u) |
			                        binding.desc.view_info.base_layer);
		}
	}
	for (const auto& color: colors) {
		auto*    image = color.image_id ? cache.TryGetImage(color.image_id) : nullptr;
		uint64_t addr = 0, fmt = 0;
		if (image != nullptr) {
			addr = image->info.data.address;
			fmt  = (static_cast<uint64_t>(image->info.pixel_format) << 32u) |
			      ((image->info.extent.width & 0xffffu) << 16u) | (image->info.extent.height & 0xffffu);
		}
		FlightRec::Rec(FlightRec::EvTarget, addr,
		               (static_cast<uint64_t>(color.target_slot & 0xffff) << 48u) |
		                   (static_cast<uint64_t>(color.image_id.index & 0xffffffffu) << 16u) |
		                   (color.image_id.generation & 0xffffu),
		               fmt, (static_cast<uint64_t>(color.guest_mip_level) << 16u) | color.guest_array_layer);
	}
	if (!fresh.empty() && cfg.rehash > 0) {
		std::lock_guard lock(g_mutex);
		for (const auto& f: fresh) {
			bool dup = false;
			for (size_t i = g_pending.size(); i > 0 && i + 512 > g_pending.size() && !dup; i--) {
				const auto& p = g_pending[i - 1];
				dup = p.addr == f.addr && p.size == f.size && p.flip == f.flip;
			}
			if (!dup && g_pending.size() < 65536) {
				g_pending.push_back(f);
			}
		}
	}
}

} // namespace Libs::Graphics::FlightRecDraw
