#pragma once

// v1.4 review M6: the upstream pick 6506ef9f6 (6dcdcd44, "preserve the depth array when rebinding
// an HTile prefix") dropped three checks from TextureCache::ResolveDepthOverlap's
// retain_cached_layout and admitted an HTile request. A depth request at another base address (a
// suffix or middle slice with the same slice size and fewer layers) is then bound to the cached
// layout, whose info is replaced wholesale, so it could read the wrong layer. The upstream test
// covers only a same-address prefix, and there is no SAROS evidence. So it is behind
// KYTY_HTILE_PREFIX_REBIND=1 (default OFF = the v1.3.1 checks). NEEDS an ASTRO + SAROS regression
// before it can be default-on (and ideally the address/layer-boundary check kept).

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace Libs::Graphics {

// Test hook: -1 = follow the switch, 0/1 = forced (a scoped case in the compute GPU suite; the
// default-off path keeps its coverage).
inline std::atomic<int>& HtilePrefixRebindOverride() {
	static std::atomic<int> value {-1};
	return value;
}

inline bool HtilePrefixRebindEnabled() {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_HTILE_PREFIX_REBIND");
		return v != nullptr && std::strcmp(v, "1") == 0;
	}();
	const int forced = HtilePrefixRebindOverride().load(std::memory_order_relaxed);
	return forced < 0 ? on : forced != 0;
}

// The terms of retain_cached_layout the pick changed. Off: bytes_per_block equal, same base
// address, cached layers != 0 and a request with no metadata at all. On (upstream): none of the
// first three, and a request with no metadata or HTile metadata.
inline bool PrefixLayoutGuard(bool gate_on, bool bpp_equal, bool address_equal,
                              bool cached_layers_nonzero, bool request_no_metadata,
                              bool request_htile) {
	if (!gate_on) return bpp_equal && address_equal && cached_layers_nonzero && request_no_metadata;
	return request_no_metadata || request_htile;
}

} // namespace Libs::Graphics
