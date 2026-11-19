#include "graphics/host_gpu/renderer/image/tilerTrace.h"

#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <tuple>

#include "graphics/host_gpu/renderer/image/image.h"

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
	bool        to_image;
	std::string reason;
	uint64_t    address;
	uint32_t    format;
	uint32_t    tile;
	uint32_t    width;
	uint32_t    height;
	uint32_t    layers;
	uint32_t    flags;
	auto        Tie() const {
		return std::tie(to_image, reason, address, format, tile, width, height, layers, flags);
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
	for (const auto& [key, value]: state.window) {
		total_count += value.count;
		total_bytes += value.bytes;
	}
	std::fprintf(stderr, "[tilertrace] window %" PRIu64 " %.1f s: %zu keys, %" PRIu64 " passes, %" PRIu64
	     " bytes\n",
	     state.index, seconds, state.window.size(), total_count, total_bytes);
	for (const auto& [key, value]: state.window) {
		std::fprintf(stderr, "[tilertrace] %s %s addr=0x%016" PRIx64 " fmt=%u tile=%u %ux%u layers=%u "
		     "flags=%c%c%c%c n=%" PRIu64 " bytes=%" PRIu64 "\n",
		     key.to_image ? "detile" : "tile", key.reason.c_str(), key.address, key.format,
		     key.tile, key.width, key.height, key.layers, (key.flags & 1u) != 0 ? 'G' : '-',
		     (key.flags & 2u) != 0 ? 'B' : '-', (key.flags & 4u) != 0 ? 'C' : '-',
		     (key.flags & 8u) != 0 ? 'K' : '-', value.count, value.bytes);
	}
	std::fflush(stderr);
	state.window.clear();
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

void Note(bool to_image, const Image& image, uint64_t bytes) noexcept {
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
	Key key {to_image,
	         t_reason != nullptr ? t_reason : "?",
	         info.data.address,
	         static_cast<uint32_t>(info.guest_format),
	         static_cast<uint32_t>(info.tile_mode),
	         info.extent.width,
	         info.extent.height,
	         info.resources.layers,
	         flags};
	auto&            state = GetState();
	std::scoped_lock lock(state.mutex);
	auto&            value = state.window[key];
	value.count++;
	value.bytes += bytes;
	const auto   now     = std::chrono::steady_clock::now();
	const double elapsed = std::chrono::duration<double>(now - state.start).count();
	if (elapsed >= window) {
		FlushLocked(state, elapsed);
		state.start = now;
	}
}

} // namespace Libs::Graphics::TilerTrace
