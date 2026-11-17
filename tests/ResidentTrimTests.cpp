// nps5 kmem: offline tests of the trimmed-image level arithmetic (residentTrim.h) together with the
// minimum-LOD view lowering it relies on (imageViewMinLod.h). No GPU.
#include "graphics/host_gpu/renderer/image/imageViewMinLod.h"
#include "graphics/host_gpu/renderer/image/residentTrim.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>

namespace {

using namespace Libs::Graphics;

uint64_t g_checks   = 0;
int      g_failures = 0;

void Check(bool condition, const char* message) {
	g_checks++;
	if (!condition) {
		if (g_failures < 40) {
			std::fprintf(stderr, "FAIL: %s\n", message);
		}
		g_failures++;
	}
}

struct View {
	uint32_t base_level  = 0;
	uint32_t level_count = 0;
	uint32_t min_lod     = 0; // 8.8 fixed point, relative to base_level
};

// TextureCache::RequestedFirstLevel for a sampled view.
uint32_t RequestedFirst(const View& view, uint32_t levels) {
	return std::min(view.base_level + (view.min_lod >> 8u), levels - 1u);
}

void Edges() {
	Check(ResidentTrim::LevelExtent(4096, 0) == 4096, "extent level 0");
	Check(ResidentTrim::LevelExtent(4096, 3) == 512, "extent level 3");
	Check(ResidentTrim::LevelExtent(4096, 12) == 1, "extent last level");
	Check(ResidentTrim::LevelExtent(4096, 13) == 1, "extent below 1 clamps");
	Check(ResidentTrim::LevelExtent(5, 1) == 2, "odd extent floors");
	Check(ResidentTrim::LevelExtent(1, 31) == 1, "shift 31");
	Check(ResidentTrim::LevelExtent(0xffffffffu, 32) == 1, "shift 32 is not undefined");
	Check(ResidentTrim::LevelExtent(0xffffffffu, 200) == 1, "huge shift");
	Check(ResidentTrim::NativeLevels(13, 0) == 13, "no trim keeps all levels");
	Check(ResidentTrim::NativeLevels(13, 4) == 9, "trim 4 of 13");
	Check(ResidentTrim::NativeLevels(13, 13) == 0, "trim all");
	Check(ResidentTrim::NativeLevels(13, 99) == 0, "trim beyond");
	Check(ResidentTrim::TrimFirst(0, 13) == 0, "trim first 0");
	Check(ResidentTrim::TrimFirst(12, 13) == 12, "trim to the last level");
	Check(ResidentTrim::TrimFirst(13, 13) == 0, "a trim that leaves nothing is no trim");
	Check(ResidentTrim::HasLevel(4, 13, 4) && !ResidentTrim::HasLevel(3, 13, 4) &&
	          !ResidentTrim::HasLevel(13, 13, 4),
	      "HasLevel bounds");
	// Identity when untrimmed.
	const auto same = ResidentTrim::NativeView(2, 5, 13, 0);
	Check(same.base == 2 && same.count == 5 && !same.clamped, "untrimmed view is unchanged");
	const auto moved = ResidentTrim::NativeView(6, 7, 13, 4);
	Check(moved.base == 2 && moved.count == 7 && !moved.clamped, "view moves by native_first");
	const auto cut = ResidentTrim::NativeView(2, 11, 13, 4);
	Check(cut.base == 0 && cut.count == 9 && cut.clamped, "a view from below is cut");
	const auto below = ResidentTrim::NativeView(0, 2, 13, 4);
	Check(below.base == 0 && below.count == 1 && below.clamped, "a view wholly below gets one level");
	Check(ResidentTrim::NativeView(0, 0, 13, 4).count == 0, "empty view");
	Check(ResidentTrim::NativeView(13, 1, 13, 4).count == 0, "view past the chain");
	Check(ResidentTrim::NativeView(5, 9, 13, 4).count == 0, "view longer than the chain");
	Check(ResidentTrim::KeptFraction(4096, 4096, 13, 0) == 1.0, "kept fraction untrimmed");
	const double two = ResidentTrim::KeptFraction(4096, 4096, 13, 2);
	Check(two > 0.0624 && two < 0.0626, "two levels trimmed keep 1/16");
	Check(ResidentTrim::KeptFraction(0, 0, 0, 0) == 1.0, "kept fraction of nothing");
}

// Every sampled view of a trimmed image, after the KosmicKrisp minimum-LOD lowering, maps onto
// native levels with the same guest levels and the same extents, never clamped, for every
// resident_first the cache can hold for it (resident_first <= the view's requested first level).
void ViewsExhaustive() {
	for (uint32_t levels = 1; levels <= 14; ++levels) {
		for (uint32_t base = 0; base < levels; ++base) {
			for (uint32_t count = 1; count <= levels - base; ++count) {
				for (uint32_t whole = 0; whole <= levels + 1; ++whole) {
					for (uint32_t fraction: {0u, 1u, 128u, 255u}) {
						View view {base, count, (whole << 8u) | fraction};
						const uint32_t requested = RequestedFirst(view, levels);
						auto           lowered   = view;
						ImageViewOps::ApplyMinLodWithoutExtension(lowered, false);
						Check(lowered.min_lod == 0, "lowering removes the clamp");
						Check(lowered.base_level + lowered.level_count == base + count,
						      "lowering keeps the view's last level");
						for (uint32_t native_first = 0; native_first <= requested; ++native_first) {
							if (lowered.base_level < native_first) {
								// The lowering stops at the view's last level; the cache's request
								// stops at the image's. A view cut short of the chain can start
								// below: NativeView cuts it, inside the image.
								const auto cut = ResidentTrim::NativeView(
								    lowered.base_level, lowered.level_count, levels, native_first);
								Check(cut.clamped && cut.count >= 1 &&
								          cut.base + cut.count <=
								              ResidentTrim::NativeLevels(levels, native_first),
								      "a short view below the trim stays inside the image");
								Check(count < levels - base, "only a view cut short starts below");
								continue;
							}
							const auto native = ResidentTrim::NativeView(
							    lowered.base_level, lowered.level_count, levels, native_first);
							Check(!native.clamped && native.count == lowered.level_count,
							      "view keeps its level count");
							Check(native.base + native_first == lowered.base_level,
							      "view keeps its guest base level");
							Check(native.base + native.count <=
							          ResidentTrim::NativeLevels(levels, native_first),
							      "view inside the native image");
						}
					}
				}
			}
		}
	}
}

// The native image's level n has the extent of guest level native_first + n, for every size.
void ExtentsRandom() {
	std::mt19937 rng(0x6b6d656d);
	for (int round = 0; round < 200000; ++round) {
		const uint32_t width  = 1u + rng() % 16384u;
		const uint32_t height = 1u + rng() % 16384u;
		uint32_t       levels = 1;
		while ((std::max(width, height) >> levels) != 0) {
			levels++;
		}
		const uint32_t first    = rng() % levels;
		const uint32_t native_w = ResidentTrim::LevelExtent(width, first);
		const uint32_t native_h = ResidentTrim::LevelExtent(height, first);
		const uint32_t level    = first + rng() % (levels - first);
		Check(ResidentTrim::LevelExtent(native_w, level - first) ==
		          ResidentTrim::LevelExtent(width, level),
		      "native level width equals the guest level's");
		Check(ResidentTrim::LevelExtent(native_h, level - first) ==
		          ResidentTrim::LevelExtent(height, level),
		      "native level height equals the guest level's");
		// A full chain of the trimmed extent has at least the kept levels (Vulkan's mipLevels limit).
		uint32_t native_chain = 1;
		while ((std::max(native_w, native_h) >> native_chain) != 0) {
			native_chain++;
		}
		Check(ResidentTrim::NativeLevels(levels, first) <= native_chain,
		      "kept levels fit the trimmed extent's chain");
		const double kept = ResidentTrim::KeptFraction(width, height, levels, first);
		Check(kept > 0.0 && kept <= 1.0, "kept fraction in (0, 1]");
		if (first != 0) {
			Check(kept < ResidentTrim::KeptFraction(width, height, levels, first - 1) ||
			          ResidentTrim::LevelExtent(width, first - 1) *
			                  ResidentTrim::LevelExtent(height, first - 1) == 0,
			      "kept fraction falls with every trimmed level");
		}
	}
}

// A growth sequence (resident_first only falls) keeps every live view valid at each step.
void GrowthRandom() {
	std::mt19937 rng(0x74726d);
	for (int round = 0; round < 100000; ++round) {
		const uint32_t levels = 2u + rng() % 13u;
		uint32_t       native = rng() % levels;
		for (int step = 0; step < 8; ++step) {
			View view {rng() % levels, 0, (rng() % (levels + 2u)) << 8u};
			view.level_count         = levels - view.base_level; // sampled views run to the last level
			const uint32_t requested = RequestedFirst(view, levels);
			if (requested < native) {
				native = ResidentTrim::TrimFirst(requested, levels); // EnsureResidency + ReplaceBacking
			}
			auto lowered = view;
			ImageViewOps::ApplyMinLodWithoutExtension(lowered, false);
			const auto result =
			    ResidentTrim::NativeView(lowered.base_level, lowered.level_count, levels, native);
			Check(!result.clamped && result.count == lowered.level_count &&
			          result.base + native == lowered.base_level,
			      "a full-length view is exact after growth");
		}
	}
}

} // namespace

// nps5 mipskip: KYTY_TEX_MIP_SKIP level math (exhaustive + random).
void MipSkip() {
	using namespace ResidentTrim;
	Check(MipSkipLevels(0, 1024, 4096, 4096, 13, 16, true) == 0, "skip off");
	Check(MipSkipLevels(1, 1024, 4096, 4096, 13, 16, true) == 1, "bc7 4096 skip 1");
	Check(MipSkipLevels(2, 1024, 4096, 2048, 12, 16, true) == 2, "bc7 4096x2048 skip 2");
	Check(MipSkipLevels(1, 1024, 512, 512, 10, 16, true) == 0, "below min dim");
	Check(MipSkipLevels(1, 1024, 1024, 8, 10, 16, true) == 1, "longest side counts");
	Check(MipSkipLevels(1, 1024, 2048, 2048, 3, 4, false) == 0, "3 levels: nothing to skip");
	Check(MipSkipLevels(2, 1024, 2048, 2048, 4, 4, false) == 1, "4 levels: keep three");
	Check(MipSkipLevels(1, 1024, 2048, 2048, 12, 1, false) == 0, "R8 stays whole");
	Check(MipSkipLevels(1, 1024, 2048, 2048, 12, 1, true) == 1, "block compressed 8-bit-block ok");
	Check(MipSkipLevels(1, 1024, 2050, 2048, 12, 16, true) == 0, "bc not block aligned after skip");
	Check(MipSkipLevels(1, 1024, 2050, 2048, 12, 4, false) == 1, "uncompressed odd width ok");
	Check(SkipFloor(0, 1, 13) == 1 && SkipFloor(3, 1, 13) == 3 && SkipFloor(0, 2, 13) == 2, "floor raises only");
	Check(SkipFloor(20, 1, 13) == 12, "floor clamps to last level");
	std::mt19937 rng(0x6d697073);
	for (int round = 0; round < 200000; ++round) {
		const uint32_t w = 1u + rng() % 8192u, h = 1u + rng() % 8192u, lv = 1u + rng() % 14u, req = rng() % 4u;
		const bool bc = rng() & 1u;
		const uint32_t bpb = bc ? 8u << (rng() & 1u) : 1u + rng() % 16u;
		const uint32_t skip = MipSkipLevels(req, 1024, w, h, lv, bpb, bc);
		Check(skip <= req, "skip never above request");
		if (skip == 0) continue;
		Check(lv >= skip + 3u, "three levels remain");
		Check(std::max(w, h) >= 1024u, "min dim honoured");
		Check((w >> skip) != 0 && (h >> skip) != 0, "skipped extent nonzero");
		Check(!bc || ((((w >> skip) & 3u) == 0) && (((h >> skip) & 3u) == 0)), "bc whole blocks");
		Check(bc || bpb >= 2u, "no 1-byte uncompressed");
		// the native image of a floor request holds the remaining levels, level 0 = guest level skip
		for (uint32_t view_first = 0; view_first < lv; ++view_first) {
			const uint32_t first = SkipFloor(view_first, skip, lv);
			Check(first >= skip && first >= view_first && first < lv, "floor in range");
			Check(NativeLevels(lv, first) >= 1u, "native image holds a level");
			// a view at guest base b (>= first) maps to b - first on the native image
			const auto nv = NativeView(view_first, 1u, lv, first);
			Check(nv.count == 1u && nv.base == (view_first >= first ? view_first - first : 0u), "view maps into native image");
		}
	}
}

int main() {
	MipSkip();
	Edges();
	ViewsExhaustive();
	ExtentsRandom();
	GrowthRandom();
	std::printf("resident_trim_tests: %llu checks, %d failures\n",
	            static_cast<unsigned long long>(g_checks), g_failures);
	return g_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
