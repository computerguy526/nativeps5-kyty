#pragma once
// KYTY_ALIAS_REBUILD_PARTIAL (TextureCache::InitializeImage): which 64 KiB blocks of a
// render-target-tiled surface a rebuild must read back from the buffer.
//
// A GPU-modified image that owns only some of its bytes (another image or a bounded buffer write
// took the others) and is buffer-modified used to be rebuilt as: tile all its owned bytes out to
// the buffer, then detile the whole image back from it. Its native contents are current exactly
// where it owns the bytes, so only the blocks it does not (fully) own need the buffer's bytes;
// blocks it owns completely keep their native texels (the tiler passes are raw bit copies, so the
// round trip was an identity there).
//
// Render-target blocks are row-major: block (column, row) is at (row * columns + column) * size.
// The tiler addresses a sub-surface from its first block with the texel position relative to the
// sub-surface. That is exact when the swizzle inside a block depends only on the position inside
// the block, except: 8- and 16-byte elements also read y bit 6 (the parity of the block row), and
// 16-byte elements x bit 6 (the parity of the block column) (gpu_tiler_render_target.comp
// rt_offset). So a segment is a run of columns in one block row, or a run of whole block rows; it
// starts on a multiple of RowAlign rows and ColumnAlign columns (tests/OwnedRebuildPlanTests.cpp
// checks the periods against every rt_offset variant). Blocks a segment covers that the image owns
// (all or in part) are "preserved": the caller moves their owned bytes to the buffer first, exactly
// as the old rebuild did for every owned byte. Header-only so it is unit-tested without a GPU
// (tests/OwnedRebuildPlanTests.cpp).
#include <cstdint>
#include <vector>

namespace Libs::Graphics::OwnedRebuild {

enum class BlockState : uint8_t { Owned, NotOwned, Mixed };

struct Surface {
	uint64_t base         = 0; // guest address of block 0
	uint32_t width        = 0; // texels
	uint32_t height       = 0;
	uint32_t block_width  = 0; // texels per block
	uint32_t block_height = 0;
	uint64_t block_size   = 0; // bytes per block
	uint32_t columns      = 0; // blocks per row (pitch / block_width)
	uint32_t rows         = 0; // block rows
};

// Blocks [first_column, first_column + columns) of rows [first_row, first_row + rows). A segment
// of more than one row always spans every column.
struct Segment {
	uint32_t first_row    = 0;
	uint32_t row_count    = 0;
	uint32_t first_column = 0;
	uint32_t column_count = 0;
};

struct Plan {
	std::vector<Segment> segments;
	uint64_t             refresh_blocks  = 0; // blocks the segments read from the buffer
	uint64_t             preserve_blocks = 0; // of those, blocks with owned bytes (moved out first)
};

// Block rows a segment must start on (see above).
[[nodiscard]] constexpr uint32_t RowAlign(uint32_t bytes_per_element) {
	return bytes_per_element == 8 || bytes_per_element == 16 ? 2u : 1u;
}
[[nodiscard]] constexpr uint32_t ColumnAlign(uint32_t bytes_per_element) {
	return bytes_per_element == 16 ? 2u : 1u;
}

// Texel rectangle of a segment, clipped to the surface (empty when the blocks are all padding).
struct Rect {
	uint32_t x = 0, y = 0, width = 0, height = 0;
};
[[nodiscard]] inline Rect SegmentRect(const Surface& s, const Segment& g) {
	Rect           r;
	const uint64_t x0 = uint64_t {g.first_column} * s.block_width;
	const uint64_t y0 = uint64_t {g.first_row} * s.block_height;
	const uint64_t x1 = uint64_t {g.first_column + g.column_count} * s.block_width;
	const uint64_t y1 = uint64_t {g.first_row + g.row_count} * s.block_height;
	if (x0 >= s.width || y0 >= s.height) {
		return r;
	}
	r.x      = static_cast<uint32_t>(x0);
	r.y      = static_cast<uint32_t>(y0);
	r.width  = static_cast<uint32_t>((x1 < s.width ? x1 : s.width) - x0);
	r.height = static_cast<uint32_t>((y1 < s.height ? y1 : s.height) - y0);
	return r;
}
// Tiled byte offset (from base) of a segment's first block, and its tiled size.
[[nodiscard]] inline uint64_t SegmentOffset(const Surface& s, const Segment& g) {
	return (uint64_t {g.first_row} * s.columns + g.first_column) * s.block_size;
}
[[nodiscard]] inline uint64_t SegmentBytes(const Surface& s, const Segment& g) {
	return uint64_t {g.row_count} * g.column_count * s.block_size;
}

// state(address, size) -> BlockState of one block. Returns false (plan unusable) on an invalid
// surface or when more than max_segments segments would be needed.
template <typename StateFn>
[[nodiscard]] bool Build(const Surface& s, StateFn&& state, Plan& plan, uint32_t row_align = 1,
                         uint32_t column_align = 1, size_t max_segments = 64) {
	plan = {};
	if (s.width == 0 || s.height == 0 || s.block_width == 0 || s.block_height == 0 ||
	    s.block_size == 0 || s.columns == 0 || s.rows == 0 || row_align == 0 || column_align == 0 ||
	    uint64_t {s.columns} * s.block_width < s.width ||
	    uint64_t {s.rows} * s.block_height < s.height) {
		return false;
	}
	std::vector<BlockState> states(size_t {s.columns} * row_align);
	const auto              add_rows = [&](uint32_t first, uint32_t count) {
		for (uint32_t r = 0; r < count; ++r) {
			for (uint32_t c = 0; c < s.columns; ++c) {
				plan.preserve_blocks += states[size_t {r} * s.columns + c] != BlockState::NotOwned;
			}
		}
		plan.refresh_blocks += uint64_t {count} * s.columns;
		if (!plan.segments.empty()) {
			auto& last = plan.segments.back();
			if (last.first_column == 0 && last.column_count == s.columns &&
			    last.first_row + last.row_count == first) {
				last.row_count += count;
				return;
			}
		}
		plan.segments.push_back({first, count, 0, s.columns});
	};
	for (uint32_t group = 0; group < s.rows; group += row_align) {
		const uint32_t count           = s.rows - group < row_align ? s.rows - group : row_align;
		bool           first_row_only  = true;
		bool           any             = false;
		uint32_t       first_not_owned = 0;
		for (uint32_t r = 0; r < count; ++r) {
			for (uint32_t c = 0; c < s.columns; ++c) {
				const auto st = state(
				    s.base + (uint64_t {group + r} * s.columns + c) * s.block_size, s.block_size);
				states[size_t {r} * s.columns + c] = st;
				if (st != BlockState::Owned) {
					any            = true;
					first_row_only = first_row_only && r == 0;
					first_not_owned += r == 0 && st == BlockState::NotOwned;
				}
			}
		}
		if (!any) {
			continue;
		}
		if (!first_row_only) {
			// A block of a later row of the group: the whole group (from its aligned first row).
			add_rows(group, count);
		} else if (first_not_owned == s.columns) {
			add_rows(group, 1);
		} else {
			// Column runs of the group's first row (an aligned row).
			for (uint32_t c = 0; c < s.columns;) {
				if (states[c] == BlockState::Owned) {
					c++;
					continue;
				}
				uint32_t end = c + 1;
				while (end < s.columns && states[end] != BlockState::Owned) {
					end++;
				}
				// An aligned start: the owned blocks before the run join it (preserved). They
				// follow the previous run's end, so runs never overlap.
				c -= c % column_align;
				for (uint32_t k = c; k < end; ++k) {
					plan.preserve_blocks += states[k] != BlockState::NotOwned;
				}
				plan.refresh_blocks += end - c;
				plan.segments.push_back({group, 1, c, end - c});
				c = end;
			}
		}
		if (plan.segments.size() > max_segments) {
			return false;
		}
	}
	return true;
}

} // namespace Libs::Graphics::OwnedRebuild
