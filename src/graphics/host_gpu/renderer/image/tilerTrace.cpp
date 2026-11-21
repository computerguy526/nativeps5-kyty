#include "graphics/host_gpu/renderer/image/tilerTrace.h"

#include "graphics/host_gpu/renderer/image/image.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <tuple>
#include <utility>

namespace Libs::Graphics::TilerTrace {

namespace {

constinit thread_local const char* t_reason = nullptr;

double WindowSeconds() noexcept {
	static const double seconds = [] {
		const char* value = std::getenv("KYTY_TILER_TRACE");
		if (value == nullptr || value[0] == 0 || std::strcmp(value, "0") == 0) {
			return 0.0;
		}
		const double parsed = std::atof(value);
		return parsed > 0.0 ? parsed : 10.0;
	}();
	return seconds;
}

struct Key {
	uint8_t     to_image; // 0 tile, 1 detile, 2 event (Event / BufferWrite)
	std::string reason;
	uint64_t    address;
	uint32_t    format;
	uint32_t    tile;
	uint32_t    width;
	uint32_t    height;
	uint32_t    layers;
	uint32_t    flags;
	uint32_t    vk_format;
	auto        Tie() const {
		return std::tie(to_image, reason, address, format, tile, width, height, layers, flags,
		                vk_format);
	}
	bool operator<(const Key& other) const { return Tie() < other.Tie(); }
};

struct Value {
	uint64_t count = 0;
	uint64_t bytes = 0;
};

struct State {
	std::mutex                            mutex;
	std::map<Key, Value>                  window;
	std::map<std::string, Value>          counters;
	std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
	uint64_t                              index = 0;
};

State& GetState() {
	static State state;
	return state;
}

void FlushLocked(State& state, double seconds) {
	uint64_t total_count = 0;
	uint64_t total_bytes = 0;
	uint64_t event_count = 0;
	for (const auto& [key, value]: state.window) {
		// Events (kind 2: bwrite / coh / rp diagnostics) are not tiler passes: the header counts
		// passes only, events separately.
		(key.to_image == 2 ? event_count : total_count) += value.count;
		if (key.to_image != 2) {
			total_bytes += value.bytes;
		}
	}
	std::fprintf(stderr,
	             "[tilertrace] window %" PRIu64 " %.1f s: %zu keys, %" PRIu64 " passes, %" PRIu64
	             " bytes (passes only; %" PRIu64 " events)\n",
	             state.index, seconds, state.window.size(), total_count, total_bytes, event_count);
	for (const auto& [key, value]: state.window) {
		std::fprintf(stderr,
		             "[tilertrace] %s %s addr=0x%016" PRIx64 " fmt=%u tile=%u %ux%u layers=%u "
		             "flags=%c%c%c%c%c vk=%u n=%" PRIu64 " bytes=%" PRIu64 "\n",
		             key.to_image == 2 ? "event" : (key.to_image != 0 ? "detile" : "tile"),
		             key.reason.c_str(), key.address, key.format, key.tile, key.width, key.height,
		             key.layers, (key.flags & 1u) != 0 ? 'G' : '-',
		             (key.flags & 2u) != 0 ? 'B' : '-', (key.flags & 4u) != 0 ? 'C' : '-',
		             (key.flags & 8u) != 0 ? 'K' : '-', (key.flags & 16u) != 0 ? 'O' : '-',
		             key.vk_format, value.count, value.bytes);
	}
	for (const auto& [what, value]: state.counters) {
		std::fprintf(stderr, "[tilertrace] count %s n=%" PRIu64 " bytes=%" PRIu64 "\n",
		             what.c_str(), value.count, value.bytes);
	}
	std::fflush(stderr);
	state.window.clear();
	state.counters.clear();
	state.index++;
}

} // namespace

bool Enabled() noexcept {
	return WindowSeconds() > 0.0;
}

Reason::Reason(const char* why) noexcept: m_previous(t_reason) {
	t_reason = why;
}

Reason::~Reason() {
	t_reason = m_previous;
}

namespace {
void Add(uint8_t kind, std::string reason, const Image& image, uint64_t bytes) noexcept {
	const double window = WindowSeconds();
	if (window <= 0.0) {
		return;
	}
	const auto& info  = image.info;
	uint32_t    flags = 0;
	flags |= image.IsGpuModified() ? 1u : 0u;
	flags |= image.IsBufferModified() ? 2u : 0u;
	flags |= image.IsCpuDirty() ? 4u : 0u;
	flags |= image.HasKeptDirty() ? 8u : 0u;
	flags |= image.alias_owner ? 16u : 0u;
	Key              key {kind,
	                      std::move(reason),
	                      info.data.address,
	                      static_cast<uint32_t>(info.guest_format),
	                      static_cast<uint32_t>(info.tile_mode),
	                      info.extent.width,
	                      info.extent.height,
	                      info.resources.layers,
	                      flags,
	                      static_cast<uint32_t>(image.backing.format)};
	auto&            state = GetState();
	std::scoped_lock lock(state.mutex);
	if (state.window.empty() && state.index == 0) {
		std::fprintf(stderr, "[tilertrace] first pass (window %.1f s)\n", window);
	}
	auto& value = state.window[key];
	value.count++;
	value.bytes += bytes;
	const auto   now     = std::chrono::steady_clock::now();
	const double elapsed = std::chrono::duration<double>(now - state.start).count();
	if (elapsed >= window) {
		FlushLocked(state, elapsed);
		state.start = now;
	}
}
} // namespace

void Note(bool to_image, const Image& image, uint64_t bytes) noexcept {
	if (WindowSeconds() <= 0.0) {
		return;
	}
	Add(to_image ? 1 : 0, t_reason != nullptr ? t_reason : "?", image, bytes);
}

void Count(const char* what, uint64_t n, uint64_t bytes) noexcept {
	if (WindowSeconds() <= 0.0) {
		return;
	}
	// Flushed with the next image pass's window (Add).
	auto&            state = GetState();
	std::scoped_lock lock(state.mutex);
	auto&            value = state.counters[what];
	value.count += n;
	value.bytes += bytes;
}

void Event(const char* what, const Image& image, uint64_t bytes) noexcept {
	if (WindowSeconds() <= 0.0) {
		return;
	}
	Add(2, what, image, bytes);
}

void EventWithReason(const char* prefix, const Image& image, uint64_t bytes) noexcept {
	if (WindowSeconds() <= 0.0) {
		return;
	}
	std::string what = prefix;
	what += '.';
	what += t_reason != nullptr ? t_reason : "?";
	Add(2, std::move(what), image, bytes);
}

void BufferWrite(const Image& image, uint64_t address, uint64_t size) noexcept {
	if (WindowSeconds() <= 0.0) {
		return;
	}
	const auto&    data  = image.info.data;
	const char*    cover = address == data.address && size == data.size              ? "exact"
	                       : address >= data.address && address + size <= data.End() ? "inside"
	                       : address <= data.address && address + size >= data.End() ? "covers"
	                                                                                 : "partial";
	const uint64_t begin = std::max(address, data.address);
	const uint64_t end   = std::min(address + size, data.End());
	std::string    what  = "bwrite.";
	what += t_reason != nullptr ? t_reason : "?";
	what += '.';
	what += cover;
	Add(2, std::move(what), image, end > begin ? end - begin : 0);
}

} // namespace Libs::Graphics::TilerTrace
