// nps5: MetalFX in place of Astro Bot's temporal upscale pass (metalfx.h).
#include "graphics/host_gpu/renderer/metalfx.h"

#include "common/guestAddress.h"

#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/image/image.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"

#if defined(__APPLE__)
#include <dlfcn.h>
#include <mach-o/dyld.h>
#endif

#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace Libs::Graphics {

namespace MetalFx {

namespace {

// The driver command's argument (work/s3/kk-mfx, src/kosmickrisp/vulkan/kk_cmd_fx.c).
struct KkMetalFxInfo {
	uint32_t size;
	uint32_t mode; // 1 temporal, 2 spatial
	VkImage  color, depth, motion, output;
	uint32_t in_w, in_h;
	float    jitter[2];
	float    mv_scale[2];
	float    mv_bias[2];
	uint32_t reset;
	uint32_t depth_reversed;
	uint32_t auto_exposure;
	uint32_t reserved;
};
using KkCmdMetalFx = int (*)(VkCommandBuffer, const KkMetalFxInfo*);

enum class Mode { Off, Temporal, Spatial };

enum class Fallback : uint8_t { NoDriver, Bindings, Sizes, NoDepth, NoMotionVectors, Refused, Count };
constexpr std::array<const char*, static_cast<size_t>(Fallback::Count)> FALLBACK_NAMES {
    "no-driver-command", "bindings", "sizes", "no-depth", "no-mv", "metalfx-refused"};

struct DepthPair {
	ImageId color;
	ImageId depth;
};

// 2936e5eca99d75e7: the pass in 01.019 (ASTROHYBRID_DLSS).
constexpr std::array<uint64_t, 1> KNOWN_PASSES {0x2936e5eca99d75e7ull};

struct State {
	Mode     mode           = Mode::Off;
	bool     find           = false;
	bool     pass_auto      = false;
	uint64_t pass_hash      = 0; // 0: KNOWN_PASSES
	float    jitter_sign    = 1.0f;
	float    mv_jitter      = 1.0f;
	bool     depth_reversed = false;
	std::array<DepthPair, 16> depth_pairs {};
	uint32_t                  depth_next = 0;
	bool                      lookup_tried = false;
	KkCmdMetalFx              command      = nullptr;
	uint32_t last_frame = 0;
	uint32_t in_w = 0, in_h = 0, out_w = 0, out_h = 0;
	float    last_jitter[2] {};
	uint64_t evaluations = 0;
	std::array<uint64_t, static_cast<size_t>(Fallback::Count)> fallbacks {};
	int64_t  print_ns = 0;
	std::array<uint64_t, 16> logged_hashes {};
	uint32_t                 logged_count = 0;
	uint32_t                 find_cb_logged = 0;
};

State& Get() {
	static State state = [] {
		State s;
		if (const char* value = std::getenv("KYTY_METALFX"); value != nullptr) {
			s.mode = (std::strcmp(value, "1") == 0 || std::strcmp(value, "temporal") == 0) ? Mode::Temporal
			         : std::strcmp(value, "spatial") == 0                                  ? Mode::Spatial
			                                                                               : Mode::Off;
		}
		if (const char* value = std::getenv("KYTY_METALFX_FIND"); value != nullptr) {
			s.find = std::strcmp(value, "1") == 0;
		}
		if (const char* value = std::getenv("KYTY_METALFX_PASS"); value != nullptr) {
			if (std::strcmp(value, "auto") == 0) {
				s.pass_auto = true;
			} else {
				s.pass_hash = std::strtoull(value, nullptr, 16);
			}
		}
		if (const char* value = std::getenv("KYTY_METALFX_JITTER_SIGN"); value != nullptr) {
			s.jitter_sign = std::strtof(value, nullptr) < 0 ? -1.0f : 1.0f;
		}
		if (const char* value = std::getenv("KYTY_METALFX_MV_JITTER"); value != nullptr) {
			const float v = std::strtof(value, nullptr);
			s.mv_jitter   = v < 0 ? -1.0f : v > 0 ? 1.0f : 0.0f;
		}
		if (const char* value = std::getenv("KYTY_METALFX_DEPTH_REVERSED"); value != nullptr) {
			s.depth_reversed = std::strcmp(value, "0") != 0;
		}
		return s;
	}();
	return state;
}

int64_t NowNs() {
	return std::chrono::duration_cast<std::chrono::nanoseconds>(
	           std::chrono::steady_clock::now().time_since_epoch())
	    .count();
}

void Summary(State& s) {
	const auto now = NowNs();
	if (s.print_ns == 0) {
		s.print_ns = now;
	}
	if (now - s.print_ns < 10'000'000'000LL) {
		return;
	}
	s.print_ns = now;
	std::string text;
	for (size_t i = 0; i < s.fallbacks.size(); i++) {
		if (s.fallbacks[i] != 0) {
			text += " " + std::string(FALLBACK_NAMES[i]) + "=" + std::to_string(s.fallbacks[i]);
		}
	}
	std::printf("MetalFX 10s: %s evaluations %llu (%ux%u -> %ux%u, jitter %.3f %.3f px) | game pass kept:%s\n",
	            s.mode == Mode::Spatial ? "spatial" : "temporal",
	            static_cast<unsigned long long>(s.evaluations), s.in_w, s.in_h, s.out_w, s.out_h,
	            static_cast<double>(s.last_jitter[0]), static_cast<double>(s.last_jitter[1]),
	            text.empty() ? " none" : text.c_str());
	std::fflush(stdout);
	s.evaluations = 0;
	s.fallbacks   = {};
}

bool Fail(State& s, Fallback reason) {
	s.fallbacks[static_cast<size_t>(reason)]++;
	Summary(s);
	return false;
}

// The command is an exported symbol of the loaded KosmicKrisp driver, not a Vulkan entry point.
KkCmdMetalFx Lookup() {
#if defined(__APPLE__)
	const uint32_t count = _dyld_image_count();
	for (uint32_t i = 0; i < count; i++) {
		const char* name = _dyld_get_image_name(i);
		if (name == nullptr || std::strstr(name, "libvulkan_kosmickrisp") == nullptr) {
			continue;
		}
		if (void* handle = dlopen(name, RTLD_NOLOAD | RTLD_LAZY); handle != nullptr) {
			auto* symbol = dlsym(handle, "nps5_kk_cmd_metalfx");
			std::printf("MetalFX: driver %s %s nps5_kk_cmd_metalfx\n", name,
			            symbol != nullptr ? "has" : "has NO");
			std::fflush(stdout);
			return reinterpret_cast<KkCmdMetalFx>(symbol);
		}
	}
	std::printf("MetalFX: no KosmicKrisp driver loaded; the game's own upscale pass stays\n");
	std::fflush(stdout);
#endif
	return nullptr;
}

bool LogOnce(State& s, uint64_t hash) {
	for (uint32_t i = 0; i < s.logged_count; i++) {
		if (s.logged_hashes[i] == hash) {
			return false;
		}
	}
	if (s.logged_count == s.logged_hashes.size()) {
		return false;
	}
	s.logged_hashes[s.logged_count++] = hash;
	return true;
}

} // namespace

bool Requested() {
	static const bool requested = Get().mode != Mode::Off || Get().find;
	return requested;
}

} // namespace MetalFx

void RenderExecutor::NoteMetalFxTargets(std::span<const RenderColorInfo> colors, ImageId depth) {
	if (!depth) {
		return;
	}
	auto& s = MetalFx::Get();
	for (const auto& color: colors) {
		if (!color.image_id) {
			continue;
		}
		bool found = false;
		for (auto& pair: s.depth_pairs) {
			if (pair.color == color.image_id) {
				pair.depth = depth;
				found      = true;
				break;
			}
		}
		if (!found) {
			s.depth_pairs[s.depth_next] = {color.image_id, depth};
			s.depth_next                = (s.depth_next + 1) % s.depth_pairs.size();
		}
	}
}

bool RenderExecutor::TryMetalFxUpscale(CommandBuffer& buffer, PreparedBindings& pixel, ImageId output,
                                       uint32_t color_count) {
	using namespace MetalFx;
	if (pixel.runtime == nullptr || !*pixel.runtime) {
		return false;
	}
	auto&      s    = Get();
	const auto hash = pixel.runtime->program->shader_hash;
	auto&      cache = m_context.GetTextureCache();

	bool known = false;
	if (s.pass_hash != 0) {
		known = hash == s.pass_hash;
	} else {
		for (const auto h: KNOWN_PASSES) {
			known = known || hash == h;
		}
	}
	// KYTY_METALFX_PASS=auto / KYTY_METALFX_FIND: the pass by its shape, from the cached images
	// alone (another draw's constants are not ours to read): two colour targets, motion and colour
	// inputs of one size, a larger output.
	const auto shape_matches = [&] {
		if (color_count != 2 || pixel.images.size() < 3 || !output || !pixel.images[0].image_id ||
		    !pixel.images[1].image_id || cache.m_slot_images.try_get(pixel.images[0].image_id) == nullptr ||
		    cache.m_slot_images.try_get(pixel.images[1].image_id) == nullptr ||
		    cache.m_slot_images.try_get(output) == nullptr) {
			return false;
		}
		const auto& mv  = cache.GetImage(pixel.images[0].image_id);
		const auto& in  = cache.GetImage(pixel.images[1].image_id);
		const auto& out = cache.GetImage(output);
		return mv.info.extent.width == in.info.extent.width &&
		       mv.info.extent.height == in.info.extent.height && in.info.extent.width >= 320 &&
		       out.info.extent.width > in.info.extent.width &&
		       out.info.extent.height > in.info.extent.height && !out.info.IsDepth() &&
		       !in.info.IsDepth() && in.backing.format == out.backing.format &&
		       out.backing.format == vk::Format::eR16G16B16A16Sfloat;
	};
	const bool structural = (s.find || s.pass_auto) && !known && shape_matches();
	// The pass's constants (metalfx.h).
	const bool has_cb = (known || structural) && pixel.images.size() >= 2 && output &&
	                    !pixel.buffer_sources.empty() && pixel.buffer_sources[0].address != 0 &&
	                    pixel.buffer_sources[0].size >= 56;
	// nps5: guest bytes are at guest + bias (dlss.cpp reads the guest address itself: Windows identity map).
	const auto* cb = has_cb ? Guest::Ptr<const uint32_t>(pixel.buffer_sources[0].address) : nullptr;
	if ((s.find || s.pass_auto) && (known || structural) && LogOnce(s, hash)) {
		std::string images;
		for (const auto& binding: pixel.images) {
			if (binding.image_id && cache.m_slot_images.try_get(binding.image_id) != nullptr) {
				const auto& image = cache.GetImage(binding.image_id);
				images += fmt::format(" {}x{}:{}", image.info.extent.width, image.info.extent.height,
				                      vk::to_string(image.backing.format));
			} else {
				images += " -";
			}
		}
		std::string sources;
		for (const auto& source: pixel.buffer_sources) {
			sources += fmt::format(" {:x}+{}", source.address, source.size);
		}
		const auto& out = cache.GetImage(output);
		std::printf("MetalFX find: pixel shader %016llx (%s) targets %u, output %ux%u %s, buffers:%s, images:%s\n",
		            static_cast<unsigned long long>(hash), known ? "known" : "shape match", color_count,
		            out.info.extent.width, out.info.extent.height,
		            vk::to_string(out.backing.format).c_str(), sources.c_str(), images.c_str());
		std::fflush(stdout);
	}
	if (s.find && has_cb && cb[4] == cache.GetImage(output).info.extent.width && s.find_cb_logged < 4) {
		s.find_cb_logged++;
		const auto* f = reinterpret_cast<const float*>(cb);
		std::printf("MetalFX find: constants of %016llx: out %ux%u in %ux%u frame %u mv %u jitter %.6f %.6f "
		            "prev %.6f %.6f\n",
		            static_cast<unsigned long long>(hash), cb[4], cb[5], cb[6], cb[7], cb[11], cb[13],
		            static_cast<double>(f[0]), static_cast<double>(f[1]), static_cast<double>(f[2]),
		            static_cast<double>(f[3]));
		std::fflush(stdout);
	}
	if (s.mode == Mode::Off || !(known || (s.pass_auto && structural))) {
		return false;
	}
	if (!has_cb) {
		return Fail(s, Fallback::Bindings);
	}
	if (!s.lookup_tried) {
		s.lookup_tried = true;
		s.command      = Lookup();
	}
	if (s.command == nullptr) {
		return Fail(s, Fallback::NoDriver);
	}

	float jitter_uv[4];
	std::memcpy(jitter_uv, cb, sizeof(jitter_uv));
	const uint32_t out_w = cb[4], out_h = cb[5], in_w = cb[6], in_h = cb[7], frame = cb[11];
	const bool     temporal = s.mode == Mode::Temporal;
	if (temporal && cb[13] == 0) {
		return Fail(s, Fallback::NoMotionVectors);
	}
	const auto mv_id    = pixel.images[0].image_id;
	const auto color_id = pixel.images[1].image_id;
	ImageId    depth_id {};
	for (const auto& pair: s.depth_pairs) {
		if (pair.color == mv_id) {
			depth_id = pair.depth;
		}
	}
	if (!color_id || cache.m_slot_images.try_get(color_id) == nullptr ||
	    cache.m_slot_images.try_get(output) == nullptr) {
		return Fail(s, Fallback::Bindings);
	}
	if (temporal && (!mv_id || cache.m_slot_images.try_get(mv_id) == nullptr)) {
		return Fail(s, Fallback::Bindings);
	}
	if (temporal && (!depth_id || cache.m_slot_images.try_get(depth_id) == nullptr)) {
		return Fail(s, Fallback::NoDepth);
	}
	auto& color = cache.GetImage(color_id);
	auto& out   = cache.GetImage(output);
	const auto same = [](const Image& image, uint32_t w, uint32_t h) {
		return image.info.extent.width == w && image.info.extent.height == h &&
		       image.backing.image != nullptr && image.info.samples == 1;
	};
	if (!same(color, in_w, in_h) || !same(out, out_w, out_h)) {
		return Fail(s, Fallback::Sizes);
	}
	Image* mv    = temporal ? &cache.GetImage(mv_id) : nullptr;
	Image* depth = temporal ? &cache.GetImage(depth_id) : nullptr;
	if (temporal && (!same(*mv, in_w, in_h) || !same(*depth, in_w, in_h) || !depth->info.IsDepth())) {
		return Fail(s, Fallback::Sizes);
	}

	// Native recording: outside rendering, the recorder drained (CommandBuffer::Handle).
	buffer.EndRendering();
	const auto cmd = buffer.Handle();
	color.Transit(vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits2::eShaderSampledRead, {},
	              cmd);
	if (temporal) {
		mv->Transit(vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits2::eShaderSampledRead, {},
		            cmd);
		depth->Transit(vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits2::eShaderSampledRead,
		               {}, cmd);
	}
	out.Transit(vk::ImageLayout::eGeneral,
	            vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite, {},
	            cmd);

	KkMetalFxInfo info {};
	info.size   = sizeof(info);
	info.mode   = temporal ? 1 : 2;
	info.color  = static_cast<VkImage>(color.backing.image);
	info.depth  = temporal ? static_cast<VkImage>(depth->backing.image) : VK_NULL_HANDLE;
	info.motion = temporal ? static_cast<VkImage>(mv->backing.image) : VK_NULL_HANDLE;
	info.output = static_cast<VkImage>(out.backing.image);
	info.in_w   = in_w;
	info.in_h   = in_h;
	// Jitter in input pixels; motion vectors in UV from previous to current, including the jitter
	// (dlss.h): MetalFX wants pixels pointing at the previous position, without the jitter.
	info.jitter[0]      = s.jitter_sign * jitter_uv[0] * static_cast<float>(in_w);
	info.jitter[1]      = s.jitter_sign * jitter_uv[1] * static_cast<float>(in_h);
	info.mv_scale[0]    = -static_cast<float>(in_w);
	info.mv_scale[1]    = -static_cast<float>(in_h);
	info.mv_bias[0]     = s.mv_jitter * (jitter_uv[0] - jitter_uv[2]);
	info.mv_bias[1]     = s.mv_jitter * (jitter_uv[1] - jitter_uv[3]);
	info.reset          = (s.last_frame != 0 && frame != s.last_frame + 1) ? 1 : 0;
	info.depth_reversed = s.depth_reversed ? 1 : 0;
	info.auto_exposure  = 1;
	const int result = s.command(static_cast<VkCommandBuffer>(cmd), &info);
	buffer.ForgetBoundState();
	if (result != 1) {
		// Nothing was encoded: the game's pass draws (its targets are bound again by the caller).
		return Fail(s, Fallback::Refused);
	}
	s.last_frame     = frame;
	s.last_jitter[0] = info.jitter[0];
	s.last_jitter[1] = info.jitter[1];
	s.in_w           = in_w;
	s.in_h           = in_h;
	s.out_w          = out_w;
	s.out_h          = out_h;
	// The output now holds MetalFX's image: a GPU write of the target, as the draw's would be.
	out.NoteContentWrite();
	cache.CommitGpuWrite(out);
	s.evaluations++;
	Summary(s);
	return true;
}

} // namespace Libs::Graphics
