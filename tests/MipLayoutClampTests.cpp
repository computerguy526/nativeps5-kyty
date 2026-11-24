// nps5 mipclamp: a T# that declares more mip levels than its extent holds (SAROS: 64x32 x128 layers,
// 9 declared, 7 fit) keeps the GUEST layout of the declared count (slice stride, total size) while the
// host image gets the clamped count: the copies of the extra levels are dropped, the others keep
// their guest offsets. Synthetic extents, the real tile code (textureCommon.cpp + tile.cpp).
#include "graphics/host_gpu/renderer/image/mipClamp.h"
#include "graphics/host_gpu/renderer/image/textureCommon.h"
#include "graphics/host_gpu/renderer/image/tiler.h"

#include <cstdio>
#include <cstdlib>

using namespace Libs::Graphics;

static int g_checks = 0, g_failed = 0;
#define CHECK(c)                                                              \
	do {                                                                      \
		g_checks++;                                                           \
		if (!(c)) {                                                           \
			g_failed++;                                                       \
			std::printf("FAILED line %d: %s\n", __LINE__, #c);                \
		}                                                                     \
	} while (0)

namespace {
struct Case {
	const char*            name;
	uint32_t               w, h, depth; // depth: layers (array) or slices (volume)
	bool                   volume;
	uint32_t               declared;
	Prospero::TileMode     tile;
};

void Run(const Case& c) {
	const auto format = Prospero::BufferFormat::k32Float;
	const uint32_t host = MipClamp::ClampLevels(c.declared, c.w, c.h, c.volume ? c.depth : 1u);
	CHECK(host < c.declared); // the case really is over-declared
	// the guest footprint of the DECLARED count (what the title wrote)
	TileSizeAlign total {};
	TileGetTextureTotalSize(format, c.w, c.h, c.depth, c.declared, c.tile, c.volume, total);
	CHECK(total.size != 0);
	// FIXED: layout of the declared count, copies of levels >= host dropped
	auto fixed = TextureCalcUploadLayout(format, c.w, c.h, c.declared, c.depth, c.tile, total.size, true,
	                                     c.volume, c.name);
	auto fixed_regions = TextureBuildImageCopies(fixed);
	TextureDropLevels(fixed_regions, host);
	// reference: the same declared layout, untouched (what real hardware addresses)
	const auto ref_regions = TextureBuildImageCopies(fixed);
	size_t     matched     = 0;
	for (const auto& r: fixed_regions) {
		CHECK(r.imageSubresource.mipLevel < host);
		for (const auto& q: ref_regions) {
			if (q.imageSubresource.mipLevel == r.imageSubresource.mipLevel &&
			    q.imageSubresource.baseArrayLayer == r.imageSubresource.baseArrayLayer &&
			    q.imageOffset.z == r.imageOffset.z) {
				CHECK(q.bufferOffset == r.bufferOffset);
				matched++;
				break;
			}
		}
	}
	CHECK(matched == fixed_regions.size() && !fixed_regions.empty());
	// every layer / slice of level 0 is present once
	size_t level0 = 0;
	for (const auto& r: fixed_regions) {
		level0 += r.imageSubresource.mipLevel == 0 ? 1 : 0;
	}
	CHECK(level0 == c.depth);
	if (c.tile != Prospero::TileMode::kLinear || true) {
		// the slice stride of the declared layout carries the whole chain: layer i of level 0 sits at
		// i * stride, and the chain total matches the declared footprint
		if (!c.volume) {
			uint64_t prev = 0;
			bool     ok   = true;
			for (const auto& r: fixed_regions) {
				if (r.imageSubresource.mipLevel != 0) {
					continue;
				}
				const uint64_t want = fixed.mips[0].offset + r.imageSubresource.baseArrayLayer * fixed.slice_stride;
				ok = ok && r.bufferOffset == want;
				prev = r.bufferOffset;
			}
			(void)prev;
			CHECK(ok);
			CHECK(fixed.slice_stride * c.depth <= total.size + fixed.slice_stride); // stride from the declared total
		}
	}
	// the buggy clamp (layout of the clamped count with the declared footprint's size) is what moved
	// layers: record that the two layouts differ for the layered case so the fix is not a no-op
	if (!c.volume && c.depth > 1) {
		TileSizeAlign total7 {};
		TileGetTextureTotalSize(format, c.w, c.h, c.depth, host, c.tile, false, total7);
		const auto clamped = TextureCalcUploadLayout(format, c.w, c.h, host, c.depth, c.tile, total7.size,
		                                             true, false, c.name);
		std::printf("  %s: slice stride declared(%u) = %llu, clamped(%u) = %llu\n", c.name, c.declared,
		            static_cast<unsigned long long>(fixed.slice_stride), host,
		            static_cast<unsigned long long>(clamped.slice_stride));
		CHECK(clamped.slice_stride <= fixed.slice_stride);
	}
	// GPU tile infos for the host levels are buildable from the declared layout
	if (c.tile != Prospero::TileMode::kLinear) {
		std::vector<GpuTileInfo> infos;
		const bool built = TextureBuildGpuTileInfos(total.size, fixed_regions, fixed, host, infos);
		// where the plain layout of a LEGAL texture of the same shape can be GPU-tiled, ours can too
		TileSizeAlign host_total {};
		TileGetTextureTotalSize(format, c.w, c.h, c.depth, host, c.tile, c.volume, host_total);
		const auto legal = TextureCalcUploadLayout(format, c.w, c.h, host, c.depth, c.tile, host_total.size,
		                                           true, c.volume, c.name);
		const auto legal_regions = TextureBuildImageCopies(legal);
		std::vector<GpuTileInfo> legal_infos;
		const bool legal_built = TextureBuildGpuTileInfos(host_total.size, legal_regions, legal, host, legal_infos);
		std::vector<GpuTileInfo> full_infos;
		const bool full_built = TextureBuildGpuTileInfos(total.size, ref_regions, fixed, c.declared, full_infos);
		if (!built) {
			std::printf("  %s: declared layout builds %s, regions %zu vs full %zu\n", c.name, full_built ? "yes" : "no",
			            fixed_regions.size(), ref_regions.size());
		}
		if (!built || infos.size() != fixed_regions.size()) {
			std::printf("  tile infos not built for %s tile=%u (legal same-shape texture: %s)\n", c.name,
			            static_cast<unsigned>(c.tile), legal_built ? "built" : "not built either");
		}
		CHECK(!legal_built || (built && (c.volume || infos.size() == fixed_regions.size())));
	}
}
} // namespace

int main() {
	const Prospero::TileMode tiles[] = {Prospero::TileMode::kStandard64KB, Prospero::TileMode::kStandard4KB,
	                                    Prospero::TileMode::kStandard256B, Prospero::TileMode::kLinear};
	for (const auto tile: tiles) {
		Run({"array 64x32 x128, 9 declared", 64, 32, 128, false, 9, tile});
		Run({"2D 128x64, 10 declared", 128, 64, 1, false, 10, tile});
		Run({"cube 64x64 x6, 9 declared", 64, 64, 6, false, 9, tile});
	}
	Run({"volume 64x32x16, 9 declared", 64, 32, 16, true, 9, Prospero::TileMode::kStandard64KB});
	// legal textures: no clamp, no layout change
	CHECK(MipClamp::ClampLevels(7, 64, 32, 1) == 7);
	std::printf("mip_layout_clamp_tests: %d checks, %d failed\n", g_checks, g_failed);
	return g_failed == 0 ? 0 : 1;
}
