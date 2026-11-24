#pragma once
// Pure aggregation of the tiler's BUFFER-mode passes (TileManager::Detile / Tile / TileImage) for
// KYTY_TILER_TRACE: key = (op, caller reason, element size, block family, extent, depth), value =
// pass count, tile infos, bytes. Header only so a unit test needs no Vulkan.
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <tuple>

namespace Libs::Graphics::TilerTrace {

struct BufferKey {
	std::string op; // "detile.buf" / "tile.buf" / "tile_image.buf"
	std::string reason;
	uint32_t    element = 0;
	uint32_t    family  = 0;
	uint32_t    width   = 0;
	uint32_t    height  = 0;
	uint32_t    depth   = 0;
	bool operator<(const BufferKey& o) const {
		return std::tie(op, reason, element, family, width, height, depth) <
		       std::tie(o.op, o.reason, o.element, o.family, o.width, o.height, o.depth);
	}
};

struct BufferValue {
	uint64_t passes = 0;
	uint64_t infos  = 0;
	uint64_t bytes  = 0;
};

class BufferAggregate {
public:
	void Add(const BufferKey& key, uint64_t infos, uint64_t bytes) {
		auto& v = m_map[key];
		v.passes++;
		v.infos += infos;
		v.bytes += bytes;
	}
	[[nodiscard]] uint64_t TotalBytes() const {
		uint64_t sum = 0;
		for (const auto& [k, v]: m_map) {
			sum += v.bytes;
		}
		return sum;
	}
	[[nodiscard]] size_t Size() const { return m_map.size(); }
	[[nodiscard]] const std::map<BufferKey, BufferValue>& Map() const { return m_map; }
	void                                                  Clear() { m_map.clear(); }
	// One line per key; the first info's geometry stands for the pass.
	[[nodiscard]] static std::string Format(const BufferKey& k, const BufferValue& v) {
		char line[320];
		std::snprintf(line, sizeof(line),
		              "[tilertrace] %s %s elem=%u family=%u %ux%ux%u passes=%" PRIu64
		              " infos=%" PRIu64 " bytes=%" PRIu64,
		              k.op.c_str(), k.reason.c_str(), k.element, k.family, k.width, k.height,
		              k.depth, v.passes, v.infos, v.bytes);
		return line;
	}

private:
	std::map<BufferKey, BufferValue> m_map;
};

} // namespace Libs::Graphics::TilerTrace
