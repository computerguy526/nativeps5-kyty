#pragma once
// nps5 decalfix: KYTY_GBCOPY_REMAP pair parsing / derivation / remap (see descriptors.cpp for the story). No GPU, no globals.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace Libs::Graphics {

struct GbCopyRemap {
	uint32_t from_w = 0, from_h = 0, to_w = 0, to_h = 0;
};
namespace GbCopyChoices {
// Mirrors Nps5ChoiceValue / Nps5ChoiceDesc of the recomp runtime (pkg/runtime/core/nps5_patches.h, NPS5_CHOICE_ABI 1).
struct Value {
	uint32_t        slot, f32;
	uint64_t        site;
	const uint64_t* values;
};
struct Desc {
	const char *        key, *title, *description, *env;
	uint32_t            noptions, def, nvalues;
	const char* const * names;
	const char* const * labels;
	const char* const * setenv;
	const Value*        values;
};
using CountFn = int (*)();
using DescFn  = const Desc* (*)(int);
using GetFn   = int (*)(int);
inline bool SizeOf(uint64_t v, uint32_t& w, uint32_t& h) {
	w = static_cast<uint32_t>(v & 0xffffffffu);
	h = static_cast<uint32_t>(v >> 32u);
	return w >= 64 && w <= 16384 && h >= 64 && h <= 16384;
}
inline void DeriveFrom(CountFn count, DescFn desc, GetFn get, std::vector<GbCopyRemap>& out) {
	if (count == nullptr || desc == nullptr || get == nullptr) {
		std::printf("[decalfix] KYTY_GBCOPY_REMAP=auto: no patch choices in this program (nps5_choices_* not found): off\n");
		return;
	}
	int seen = 0;
	for (int i = 0; i < count(); i++) {
		const Desc* d = desc(i);
		if (d == nullptr || d->key == nullptr || d->values == nullptr || d->noptions == 0 || d->noptions > 16 ||
		    (std::strcmp(d->key, "upscaling") != 0 && std::strcmp(d->key, "render_scale") != 0)) {
			continue;
		}
		seen++;
		const int o = get(i);
		if (o < 0 || static_cast<uint32_t>(o) >= d->noptions) {
			continue;
		}
		for (uint32_t k = 0; k < d->nvalues; k++) {
			const Value& v = d->values[k];
			if (v.f32 != 0 || v.values == nullptr) {
				continue;
			}
			GbCopyRemap r {};
			bool        sizes = true;
			for (uint32_t n = 0; n < d->noptions && sizes; n++) {
				uint32_t w = 0, h = 0;
				sizes = SizeOf(v.values[n], w, h);
				if (sizes && static_cast<uint64_t>(w) * h > static_cast<uint64_t>(r.from_w) * r.from_h) {
					r.from_w = w;
					r.from_h = h;
				}
			}
			if (!sizes || !SizeOf(v.values[o], r.to_w, r.to_h) ||
			    (r.to_w == r.from_w && r.to_h == r.from_h) || r.to_w > r.from_w || r.to_h > r.from_h) {
				continue;
			}
			bool dup = false;
			for (const auto& e: out) {
				dup = dup || (e.from_w == r.from_w && e.from_h == r.from_h);
			}
			if (!dup) {
				out.push_back(r);
				std::printf("[decalfix] KYTY_GBCOPY_REMAP=auto: patch %s = %s: native %ux%u -> %ux%u\n", d->key,
				            d->names != nullptr && d->names[o] != nullptr ? d->names[o] : "?", r.from_w, r.from_h,
				            r.to_w, r.to_h);
			}
		}
	}
	if (out.empty()) {
		std::printf("[decalfix] KYTY_GBCOPY_REMAP=auto: %d render-size choice%s, none smaller than native: off\n", seen,
		            seen == 1 ? "" : "s");
	}
}
} // namespace GbCopyChoices

// Explicit pairs "WxH=wxh,..." (to <= from, non-zero); false when parsing stopped early.
inline bool ParseGbCopyPairs(const char* value, std::vector<GbCopyRemap>& out) {
	const char* p = value;
	while (*p != '\0') {
		GbCopyRemap r {};
		int         used = 0;
		if (std::sscanf(p, "%ux%u=%ux%u%n", &r.from_w, &r.from_h, &r.to_w, &r.to_h, &used) == 4 && used > 0 &&
		    r.to_w != 0 && r.to_h != 0 && r.to_w <= r.from_w && r.to_h <= r.from_h) {
			out.push_back(r);
			p += used;
		} else {
			std::printf("[decalfix] KYTY_GBCOPY_REMAP: cannot parse '%s', ignored from there\n", p);
			return false;
		}
		if (*p == ',') {
			p++;
		}
	}
	return true;
}

// The extent a native-size copy gets (false: unchanged).
inline bool ApplyGbCopyRemap(const std::vector<GbCopyRemap>& remaps, uint32_t& width, uint32_t& height) {
	for (const auto& r: remaps) {
		if (width == r.from_w && height == r.from_h) {
			width  = r.to_w;
			height = r.to_h;
			return true;
		}
	}
	return false;
}

} // namespace Libs::Graphics
