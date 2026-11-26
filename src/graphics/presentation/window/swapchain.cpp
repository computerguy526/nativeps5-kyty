#include "common/assert.h"
#include "common/flightRec.h"
#include "common/nps5Frame.h"
#ifdef __APPLE__
#include <mach/mach.h>
#include <sys/resource.h>
#include <sys/sysctl.h>
#endif
#include "common/common.h"
#include "common/debugCounters.h"
#include "common/emulatorConfig.h"
#include "common/hangWatchdog.h"
#include "common/latTrace.h"
#include "common/logging/log.h"
#include "common/presentInterval.h"
#include "common/profiler.h"
#include "common/threads.h"
#include "gpu_blit_shaders/gpu_blit_fs_triangle_spv.h"
#include "gpu_blit_shaders/gpu_video_out_downscale_spv.h"
#include "gpu_blit_shaders/gpu_video_out_overlay_spv.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/host_gpu/renderer/metalfx.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/loadCard.h"
#include "graphics/presentation/openxrHeadset.h"
#include "graphics/presentation/performanceOverlay.h"
#include "graphics/presentation/presenter.h"
#include "graphics/presentation/systemOverlay.h"
#include "graphics/presentation/videoOut.h"
#include "graphics/presentation/window/lowLatency.h"
#include "graphics/presentation/window/windowInternal.h"
#include "graphics/progressOverlay.h"
#include "kernel/memory.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <vector>
#include <vulkan/vk_platform.h>
#include <vulkan/vulkan_format_traits.hpp>

// IWYU pragma: no_include <intrin.h>

// nps5 cachesave: KYTY_MEMLOG=1 -> a 1 Hz '[mem]' line (footprint / resident / compressed / swap /
// guest direct memory) and a '[slowframe]' line for every frame over 500 ms with the page-fault
// deltas. Default off.
namespace Nps5MemLog {
static bool Enabled() {
	static const bool v = [] {
		const char* e = std::getenv("KYTY_MEMLOG");
		return e != nullptr && e[0] != '\0' && e[0] != '0';
	}();
	return v;
}
static double Mb(uint64_t b) {
	return static_cast<double>(b) / 1048576.0;
}
static void Line() {
#ifdef __APPLE__
	// One sampler for both: the always-on flight recorder's 1 Hz EvMem (common/flightRec.cpp) reads
	// the same numbers.
	FlightRec::MemSample ms {};
	FlightRec::ReadMem(ms);
	const uint64_t foot = ms.footprint, res = ms.resident, comp = ms.compressed;
	xsw_usage      sw {};
	sw.xsu_used = ms.swap_used;
	rusage ru {};
	ru.ru_majflt  = static_cast<long>(ms.majflt);
	ru.ru_minflt  = static_cast<long>(ms.minflt);
	const auto ds = Libs::LibKernel::Memory::GetDebugStats();
	std::fprintf(stderr,
	             "[mem] footprint %.0f MB resident %.0f MB compressed %.0f MB swap_used %.0f MB "
	             "direct_total %.0f MB direct_alloc %.0f MB "
	             "direct_mapped %.0f MB flex_used %.0f MB majflt %ld minflt %ld frame_ms %u\n",
	             Mb(foot), Mb(res), Mb(comp), Mb(sw.xsu_used), Mb(ds.direct_total),
	             Mb(ds.direct_allocated), Mb(ds.direct_mapped), Mb(ds.flexible_used), ru.ru_majflt,
	             ru.ru_minflt, Nps5Frame::g_last_frame_ms.load(std::memory_order_relaxed));
	std::fflush(stderr);
#endif
}
static void ExtraForRecorder(FlightRec::MemExtra& e) {
#ifdef __APPLE__
	const auto ds      = Libs::LibKernel::Memory::GetDebugStats();
	e.direct_alloc_mb  = static_cast<uint64_t>(Mb(ds.direct_allocated));
	e.direct_mapped_mb = static_cast<uint64_t>(Mb(ds.direct_mapped));
	e.flexible_mb      = static_cast<uint64_t>(Mb(ds.flexible_used));
	e.value            = static_cast<uint64_t>(Mb(ds.direct_total));
#else
	(void)e;
#endif
}
static void Start() {
	static std::once_flag once;
	std::call_once(once, [] {
		FlightRec::SetMemExtraProvider(
		    &ExtraForRecorder); // the recorder samples it at 1 Hz, KYTY_MEMLOG or not
		if (!Enabled()) return;
		std::thread([] {
			for (;;) {
				std::this_thread::sleep_for(std::chrono::seconds(1));
				Line();
			}
		}).detach();
	});
}
// Called once per Present().
static void NoteFrame() {
	using clock    = std::chrono::steady_clock;
	const auto now = static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now().time_since_epoch())
	        .count());
	const auto prev = Nps5Frame::g_last_present_ns.exchange(now, std::memory_order_relaxed);
#ifdef __APPLE__
	static long last_maj = 0, last_min = 0;
	long        dmaj = 0, dmin = 0;
	if (Enabled()) {
		rusage ru {};
		getrusage(RUSAGE_SELF, &ru);
		dmaj     = ru.ru_majflt - last_maj;
		dmin     = ru.ru_minflt - last_min;
		last_maj = ru.ru_majflt;
		last_min = ru.ru_minflt;
	}
#endif
	if (prev == 0) return;
	const auto ms = static_cast<uint32_t>((now - prev) / 1'000'000ull);
	Nps5Frame::g_last_frame_ms.store(ms, std::memory_order_relaxed);
	if (ms > 100) Nps5Frame::g_slow_frame_ns.store(now, std::memory_order_relaxed);
#ifdef __APPLE__
	if (ms > 500 && Enabled()) {
		std::fprintf(stderr, "[slowframe] %u ms majflt +%ld minflt +%ld\n", ms, dmaj, dmin);
		std::fflush(stderr);
	}
#endif
}
} // namespace Nps5MemLog

namespace Libs::Graphics {

struct Presenter::Frame {
	VulkanImage   image;
	vk::ImageView view         = nullptr;
	uint64_t      present_tick = 0;
	bool          busy         = false;
	bool          hdr          = false;
	bool          blank        = false;

	void Configure(GraphicContext& graphics, vk::Extent2D extent, vk::Format format);
	void Transit(vk::CommandBuffer command, vk::ImageLayout layout, vk::AccessFlags2 access);
	void CopyFrom(CommandBuffer& command, Image& source);
	void Clear(CommandBuffer& command, const vk::ClearColorValue& color);
};

class FramePool final {
public:
	FramePool(WindowContext& window, CommandScheduler& scheduler)
	    : m_window(window), m_scheduler(scheduler) {}
	~FramePool() {
		m_scheduler.Wait(m_scheduler.CurrentTick() - 1);
		for (auto& frame: m_frames) {
			m_window.graphic_ctx.device.destroyImageView(frame->view, nullptr);
			if (frame->image.image != nullptr) {
				m_window.graphic_ctx.DeleteImage(frame->image);
			}
		}
	}
	KYTY_CLASS_NO_COPY(FramePool);

	void Initialize(uint32_t count, vk::Format format) {
		Common::LockGuard lock(m_mutex);
		m_format = format;
		m_frames.reserve(count);
		m_free.reserve(count);
		for (uint32_t i = 0; i < count; i++) {
			auto frame = std::make_unique<Presenter::Frame>();
			m_free.push_back(frame.get());
			m_frames.push_back(std::move(frame));
		}
	}

	void SetFormat(vk::Format format) {
		Common::LockGuard lock(m_mutex);
		m_format = format;
	}

	vk::Format GetFormat() {
		Common::LockGuard lock(m_mutex);
		return m_format;
	}

	Presenter::Frame* Acquire(vk::Extent2D extent, vk::Format format) {
		m_mutex.Lock();
		// A synchronized flip may need more frames than the swapchain has images.
		// Queue capacity bounds growth; waiting here can prevent its master from recording.
		if (m_free.empty()) {
			auto frame = std::make_unique<Presenter::Frame>();
			m_free.push_back(frame.get());
			m_frames.push_back(std::move(frame));
		}
		const auto compatible = std::ranges::find_if(m_free, [&](const auto* frame) {
			return frame->image.extent.width == extent.width &&
			       frame->image.extent.height == extent.height && frame->image.format == format;
		});
		auto*      frame      = compatible == m_free.end() ? m_free.back() : *compatible;
		if (compatible != m_free.end()) {
			*compatible = m_free.back();
		}
		m_free.pop_back();
		frame->busy = true;
		m_mutex.Unlock();

		m_scheduler.Wait(frame->present_tick);
		return frame;
	}

	void Release(Presenter::Frame* frame) {
		Common::LockGuard lock(m_mutex);
		if (!frame->busy) {
			EXIT("prepared frame was released twice\n");
		}
		frame->busy = false;
		m_free.push_back(frame);
	}

private:
	WindowContext&                                 m_window;
	CommandScheduler&                              m_scheduler;
	Common::Mutex                                  m_mutex;
	std::vector<std::unique_ptr<Presenter::Frame>> m_frames;
	std::vector<Presenter::Frame*>                 m_free;
	vk::Format                                     m_format = vk::Format::eUndefined;
};

void Presenter::Frame::Configure(GraphicContext& graphics, vk::Extent2D extent, vk::Format format) {
	if (extent.width == 0 || extent.height == 0 || format == vk::Format::eUndefined) {
		EXIT("unsupported prepared frame, extent=%ux%u format=%d\n", extent.width, extent.height,
		     static_cast<int>(format));
	}
	auto&      dst        = image;
	const bool compatible = dst.image != nullptr && dst.extent.width == extent.width &&
	                        dst.extent.height == extent.height && dst.format == format;
	if (compatible) {
		return;
	}

	const auto features = graphics.GetFormatProperties(format).optimalTilingFeatures;
	const auto required =
	    vk::FormatFeatureFlagBits::eBlitSrc | vk::FormatFeatureFlagBits::eSampledImageFilterLinear |
	    vk::FormatFeatureFlagBits::eSampledImage | vk::FormatFeatureFlagBits::eTransferSrc |
	    vk::FormatFeatureFlagBits::eTransferDst;
	if ((features & required) != required) {
		EXIT("prepared presentation format lacks optimal blit support: format=%d features=0x%x\n",
		     static_cast<int>(format), static_cast<vk::FormatFeatureFlags::MaskType>(features));
	}

	if (dst.image != nullptr) {
		graphics.device.destroyImageView(view, nullptr);
		view = nullptr;
		graphics.DeleteImage(dst);
	}

	vk::ImageCreateInfo create {};
	create.sType         = vk::StructureType::eImageCreateInfo;
	create.imageType     = vk::ImageType::e2D;
	create.extent        = {extent.width, extent.height, 1};
	create.mipLevels     = 1;
	create.arrayLayers   = 1;
	create.format        = format;
	create.tiling        = vk::ImageTiling::eOptimal;
	create.initialLayout = vk::ImageLayout::eUndefined;
	create.usage = vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst |
	               vk::ImageUsageFlagBits::eSampled;
	create.sharingMode = vk::SharingMode::eExclusive;
	create.samples     = vk::SampleCountFlagBits::e1;
	if (!graphics.CreateImage(create, dst)) {
		EXIT("failed to allocate prepared presentation image, extent=%ux%u format=%d\n",
		     extent.width, extent.height, static_cast<int>(format));
	}
}

void Presenter::Frame::Transit(vk::CommandBuffer command, vk::ImageLayout layout,
                               vk::AccessFlags2 access) {
	const auto     stage  = access == vk::AccessFlagBits2::eTransferRead ||
	                                access == vk::AccessFlagBits2::eTransferWrite
	                            ? vk::PipelineStageFlagBits2::eTransfer
	                            : vk::PipelineStageFlagBits2::eAllCommands;
	constexpr auto writes = vk::AccessFlagBits2::eTransferWrite |
	                        vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eMemoryWrite;
	if (image.state.layout == layout && image.state.access_mask == access &&
	    !static_cast<bool>(image.state.access_mask & writes)) {
		return;
	}
	vk::ImageMemoryBarrier2 barrier {};
	barrier.srcStageMask                    = image.state.pl_stage;
	barrier.srcAccessMask                   = image.state.access_mask;
	barrier.dstStageMask                    = stage;
	barrier.dstAccessMask                   = access;
	barrier.oldLayout                       = image.state.layout;
	barrier.newLayout                       = layout;
	barrier.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
	barrier.image                           = image.image;
	barrier.subresourceRange.aspectMask     = vk::ImageAspectFlagBits::eColor;
	barrier.subresourceRange.baseMipLevel   = 0;
	barrier.subresourceRange.levelCount     = VK_REMAINING_MIP_LEVELS;
	barrier.subresourceRange.baseArrayLayer = 0;
	barrier.subresourceRange.layerCount     = VK_REMAINING_ARRAY_LAYERS;
	vk::DependencyInfo dependency {};
	dependency.imageMemoryBarrierCount = 1;
	dependency.pImageMemoryBarriers    = &barrier;
	command.pipelineBarrier2(dependency);
	image.state = {stage, access, layout};
}

void Presenter::Frame::CopyFrom(CommandBuffer& command_buffer, Image& source) {
	command_buffer.EndRendering();
	auto command = command_buffer.Handle();
	source.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, {},
	               command);
	Transit(command, vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite);
	vk::ImageCopy copy {};
	copy.srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, source.backing.layers};
	copy.dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, image.layers};
	copy.extent         = {std::min(source.backing.extent.width, image.extent.width),
	                       std::min(source.backing.extent.height, image.extent.height), 1};
	EXIT_IF(copy.srcSubresource.layerCount != copy.dstSubresource.layerCount);
	command.copyImage(source.backing.image, vk::ImageLayout::eTransferSrcOptimal, image.image,
	                  vk::ImageLayout::eTransferDstOptimal, copy);
}

void Presenter::Frame::Clear(CommandBuffer& command_buffer, const vk::ClearColorValue& color) {
	command_buffer.EndRendering();
	auto command = command_buffer.Handle();
	Transit(command, vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite);
	const vk::ImageSubresourceRange range {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
	command.clearColorImage(image.image, vk::ImageLayout::eTransferDstOptimal, &color, 1, &range);
}

// nps5 load card: "is the game frame black?" probe. The presented game frame is blitted to a 32x18
// RGBA8 image and copied to a host buffer on the present command buffer; the result is read at the
// next present (after the tick of the previous one completed: never a stall). The card then also
// covers the game's own black loading frames and stops on the first frame with content
// (LoadCard::NoteGameFlip(black)). Probed every present while the state is unknown/black, every
// 16th otherwise. Skipped for the filtered present path (state stays unknown: flips count as
// content after kUnknownFlips).
class LumaProbe final {
public:
	static LumaProbe& Instance() {
		static LumaProbe probe;
		return probe;
	}
	[[nodiscard]] bool Enabled() const noexcept { return m_enabled; }
	// Reads the previous probe if its tick completed (call before recording the next present).
	void Collect(CommandScheduler& scheduler) {
		if (!m_pending) {
			return;
		}
		scheduler.Wait(m_tick);
		m_pending       = false;
		const auto* px  = static_cast<const uint8_t*>(m_mapped);
		unsigned    max = 0;
		for (uint32_t i = 0; i < kW * kH * 4u; i += 4) {
			max = std::max<unsigned>(max, std::max({px[i], px[i + 1], px[i + 2]}));
		}
		const bool black = LoadCard::FrameIsBlack(max);
		if (LoadCard::LogEnabled() &&
		    LoadCard::State().probe_state.load(std::memory_order_relaxed) != (black ? 1 : 2)) {
			std::fprintf(stderr, "[loadcard] probe: game frame is %s (max %u)\n",
			             black ? "black" : "content", max);
		}
		LoadCard::NoteProbe(black);
	}
	[[nodiscard]] bool Wanted() noexcept {
		if (!m_enabled) {
			return false;
		}
		const auto state = LoadCard::State().probe_state.load(std::memory_order_relaxed);
		return state != 2 || (++m_count % 16) == 0 || LoadCard::Active();
	}
	void Record(GraphicContext& graphics, vk::CommandBuffer command, vk::Image source,
	            vk::Extent2D source_extent) {
		Ensure(graphics);
		const vk::ImageSubresourceRange range {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
		vk::ImageMemoryBarrier          barrier {};
		barrier.dstAccessMask       = vk::AccessFlagBits::eTransferWrite;
		barrier.oldLayout           = vk::ImageLayout::eUndefined;
		barrier.newLayout           = vk::ImageLayout::eTransferDstOptimal;
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.image               = m_image;
		barrier.subresourceRange    = range;
		command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
		                        vk::PipelineStageFlagBits::eTransfer, vk::DependencyFlags {}, 0,
		                        nullptr, 0, nullptr, 1, &barrier);
		vk::ImageBlit region {};
		region.srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
		region.srcOffsets[1]  = vk::Offset3D {static_cast<int>(source_extent.width),
		                                      static_cast<int>(source_extent.height), 1};
		region.dstSubresource = region.srcSubresource;
		region.dstOffsets[1]  = vk::Offset3D {static_cast<int>(kW), static_cast<int>(kH), 1};
		command.blitImage(source, vk::ImageLayout::eTransferSrcOptimal, m_image,
		                  vk::ImageLayout::eTransferDstOptimal, 1, &region, vk::Filter::eLinear);
		barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
		barrier.dstAccessMask = vk::AccessFlagBits::eTransferRead;
		barrier.oldLayout     = vk::ImageLayout::eTransferDstOptimal;
		barrier.newLayout     = vk::ImageLayout::eTransferSrcOptimal;
		command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
		                        vk::PipelineStageFlagBits::eTransfer, vk::DependencyFlags {}, 0,
		                        nullptr, 0, nullptr, 1, &barrier);
		vk::BufferImageCopy copy {};
		copy.imageSubresource = region.srcSubresource;
		copy.imageExtent      = vk::Extent3D {kW, kH, 1};
		command.copyImageToBuffer(m_image, vk::ImageLayout::eTransferSrcOptimal, m_buffer, 1,
		                          &copy);
		m_recorded = true;
	}
	// After Submit: the tick of the command buffer that carries the probe.
	void Submitted(uint64_t tick) noexcept {
		if (m_recorded) {
			m_recorded = false;
			m_pending  = true;
			m_tick     = tick;
		}
	}

private:
	static constexpr uint32_t
	    kW = 64,
	    kH = 36; // card3: black = every pixel <= LoadCard::kBlackMax (max luma, not mean)
	LumaProbe(): m_enabled(LoadCard::BlackEnabled()) {}
	void Ensure(GraphicContext& graphics) {
		if (m_image != nullptr) {
			return;
		}
		auto       device      = graphics.device;
		const auto memory_type = [&](uint32_t bits, vk::MemoryPropertyFlags wanted) {
			const auto& memory = graphics.physical_device_memory_properties;
			for (uint32_t i = 0; i < memory.memoryTypeCount; i++) {
				if ((bits & (1u << i)) != 0 &&
				    (memory.memoryTypes[i].propertyFlags & wanted) == wanted) {
					return i;
				}
			}
			EXIT("luma probe: no memory type\n");
			return 0u;
		};
		vk::ImageCreateInfo create {};
		create.imageType   = vk::ImageType::e2D;
		create.extent      = vk::Extent3D {kW, kH, 1};
		create.mipLevels   = 1;
		create.arrayLayers = 1;
		create.format      = vk::Format::eB8G8R8A8Unorm;
		create.usage = vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst;
		create.samples = vk::SampleCountFlagBits::e1;
		RequireVulkanSuccess(device.createImage(&create, nullptr, &m_image), "luma probe image");
		vk::MemoryRequirements requirements {};
		device.getImageMemoryRequirements(m_image, &requirements);
		vk::MemoryAllocateInfo allocate {};
		allocate.allocationSize  = requirements.size;
		allocate.memoryTypeIndex = memory_type(requirements.memoryTypeBits, {});
		RequireVulkanSuccess(device.allocateMemory(&allocate, nullptr, &m_image_memory),
		                     "luma probe image memory");
		device.bindImageMemory(m_image, m_image_memory, 0);
		vk::BufferCreateInfo buffer {};
		buffer.size  = static_cast<vk::DeviceSize>(kW) * kH * 4u;
		buffer.usage = vk::BufferUsageFlagBits::eTransferDst;
		RequireVulkanSuccess(device.createBuffer(&buffer, nullptr, &m_buffer), "luma probe buffer");
		device.getBufferMemoryRequirements(m_buffer, &requirements);
		allocate.allocationSize = requirements.size;
		allocate.memoryTypeIndex =
		    memory_type(requirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible |
		                                                 vk::MemoryPropertyFlagBits::eHostCoherent);
		RequireVulkanSuccess(device.allocateMemory(&allocate, nullptr, &m_buffer_memory),
		                     "luma probe buffer memory");
		device.bindBufferMemory(m_buffer, m_buffer_memory, 0);
		RequireVulkanSuccess(device.mapMemory(m_buffer_memory, 0, VK_WHOLE_SIZE, {}, &m_mapped),
		                     "luma probe map");
	}
	bool             m_enabled       = false;
	bool             m_recorded      = false;
	bool             m_pending       = false;
	uint64_t         m_tick          = 0;
	uint64_t         m_count         = 0;
	vk::Image        m_image         = nullptr;
	vk::DeviceMemory m_image_memory  = nullptr;
	vk::Buffer       m_buffer        = nullptr;
	vk::DeviceMemory m_buffer_memory = nullptr;
	void*            m_mapped        = nullptr;
};

// nps5: KYTY_DUMP_DIR=<directory> (default: working directory) for the frame and target dumps.
static std::string DumpPath(const char* name) {
	const char* dir = std::getenv("KYTY_DUMP_DIR");
	return dir != nullptr && dir[0] != 0 ? std::string(dir) + "/" + name : std::string(name);
}

// nps5: KYTY_DUMP_FRAMES=<n> [KYTY_DUMP_FRAMES_EVERY=<m>] [KYTY_DUMP_FRAMES_FROM=<present>] saves
// every m-th presented swapchain image (what the window shows: after the blit/filter and the
// overlays) as frame_<index>.bmp in the working directory, 960x540 (KYTY_DUMP_SCALE=1: swapchain
// size) 32-bit top-down BGRA, the format and naming of AnyPS5's APS5_DUMP_FRAMES. The read-back is
// synchronous on the present thread.
class FrameDump final {
public:
	static FrameDump& Instance() {
		static FrameDump dump;
		return dump;
	}
	[[nodiscard]] bool Enabled() const noexcept { return m_limit > 0; }
	// True when the next Next() call will select a frame (no state change).
	[[nodiscard]] bool WouldDump() const noexcept {
		Arm();
		return m_dumped < m_limit && m_presents + 1 >= m_from &&
		       (m_presents + 1) % static_cast<uint64_t>(m_every) == 0;
	}
	[[nodiscard]] bool Next() {
		if (m_dumped >= m_limit) {
			return false;
		}
		Arm();
		return ++m_presents >= m_from && m_presents % static_cast<uint64_t>(m_every) == 0;
	}
	// fmsel6: KYTY_DUMP_FRAMES_TRIGGER=<file>: no dumps until the file exists; then
	// KYTY_DUMP_FRAMES_FROM counts from the present at which it appeared (a runner creates it on a
	// game-time event).
	void Arm() const noexcept {
		if (m_trigger.empty() || m_armed) {
			return;
		}
		m_from = UINT64_MAX;
		struct stat st {};
		if (::stat(m_trigger.c_str(), &st) == 0) {
			m_armed = true;
			m_from  = m_presents + 1 + m_from_after;
			std::fprintf(stderr, "[kyty-dump] trigger %s at present %llu: dumps from %llu\n",
			             m_trigger.c_str(), static_cast<unsigned long long>(m_presents),
			             static_cast<unsigned long long>(m_from));
		}
	}
	void Record(GraphicContext& graphics, vk::CommandBuffer command, vk::Image swapchain_image,
	            vk::Format swapchain_format, vk::Extent2D swapchain_extent,
	            bool from_attachment = false) {
		const bool         srgb   = swapchain_format == vk::Format::eB8G8R8A8Srgb ||
		                            swapchain_format == vk::Format::eR8G8B8A8Srgb;
		const auto         format = srgb ? vk::Format::eB8G8R8A8Srgb : vk::Format::eB8G8R8A8Unorm;
		const vk::Extent2D extent = m_full ? swapchain_extent : vk::Extent2D {960, 540};
		Ensure(graphics, format, extent);
		const vk::ImageSubresourceRange       range {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
		std::array<vk::ImageMemoryBarrier, 2> barriers {};
		barriers[0].srcAccessMask = from_attachment ? vk::AccessFlagBits::eColorAttachmentWrite
		                                            : vk::AccessFlagBits::eMemoryRead;
		barriers[0].dstAccessMask = vk::AccessFlagBits::eTransferRead;
		barriers[0].oldLayout     = from_attachment ? vk::ImageLayout::eColorAttachmentOptimal
		                                            : vk::ImageLayout::ePresentSrcKHR;
		barriers[0].newLayout     = vk::ImageLayout::eTransferSrcOptimal;
		barriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barriers[0].image               = swapchain_image;
		barriers[0].subresourceRange    = range;
		barriers[1]                     = barriers[0];
		barriers[1].srcAccessMask       = {};
		barriers[1].dstAccessMask       = vk::AccessFlagBits::eTransferWrite;
		barriers[1].oldLayout           = vk::ImageLayout::eUndefined;
		barriers[1].newLayout           = vk::ImageLayout::eTransferDstOptimal;
		barriers[1].image               = m_image;
		command.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
		                        vk::PipelineStageFlagBits::eTransfer, vk::DependencyFlags {}, 0,
		                        nullptr, 0, nullptr, 2, barriers.data());
		vk::ImageBlit region {};
		region.srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
		region.srcOffsets[1]  = vk::Offset3D {static_cast<int>(swapchain_extent.width),
		                                      static_cast<int>(swapchain_extent.height), 1};
		region.dstSubresource = region.srcSubresource;
		region.dstOffsets[1] =
		    vk::Offset3D {static_cast<int>(extent.width), static_cast<int>(extent.height), 1};
		command.blitImage(swapchain_image, vk::ImageLayout::eTransferSrcOptimal, m_image,
		                  vk::ImageLayout::eTransferDstOptimal, 1, &region, vk::Filter::eLinear);
		barriers[0].srcAccessMask = vk::AccessFlagBits::eTransferRead;
		barriers[0].dstAccessMask = from_attachment ? vk::AccessFlagBits::eColorAttachmentRead |
		                                                  vk::AccessFlagBits::eColorAttachmentWrite
		                                            : vk::AccessFlagBits::eMemoryRead;
		barriers[0].oldLayout     = vk::ImageLayout::eTransferSrcOptimal;
		barriers[0].newLayout     = from_attachment ? vk::ImageLayout::eColorAttachmentOptimal
		                                            : vk::ImageLayout::ePresentSrcKHR;
		barriers[1].srcAccessMask = vk::AccessFlagBits::eTransferWrite;
		barriers[1].dstAccessMask = vk::AccessFlagBits::eTransferRead;
		barriers[1].oldLayout     = vk::ImageLayout::eTransferDstOptimal;
		barriers[1].newLayout     = vk::ImageLayout::eTransferSrcOptimal;
		command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
		                        vk::PipelineStageFlagBits::eAllCommands, vk::DependencyFlags {}, 0,
		                        nullptr, 0, nullptr, 2, barriers.data());
		vk::BufferImageCopy copy {};
		copy.imageSubresource = region.srcSubresource;
		copy.imageExtent      = vk::Extent3D {extent.width, extent.height, 1};
		command.copyImageToBuffer(m_image, vk::ImageLayout::eTransferSrcOptimal, m_buffer, 1,
		                          &copy);
		m_recorded = true;
	}
	/// True when the command buffer just submitted carries a read-back; the caller waits for it.
	[[nodiscard]] bool TakeRecorded() noexcept {
		const bool recorded = m_recorded;
		m_recorded          = false;
		return recorded;
	}
	void Write() {
		char name[64];
		std::snprintf(name, sizeof(name), "frame_%03d.bmp", m_dumped++);
		const uint32_t bytes = m_extent.width * m_extent.height * 4u;
		uint8_t        header[54] {};
		const auto put = [&](size_t at, uint32_t value) { std::memcpy(header + at, &value, 4); };
		header[0]      = 'B';
		header[1]      = 'M';
		put(2, 54u + bytes);
		put(10, 54u);
		put(14, 40u);
		put(18, m_extent.width);
		put(22, static_cast<uint32_t>(-static_cast<int32_t>(m_extent.height)));
		header[26] = 1;
		header[28] = 32;
		put(34, bytes);
		put(38, 2835u);
		put(42, 2835u);
		if (Common::LatTrace::Enabled() && std::getenv("KYTY_DUMP_STAT") != nullptr) {
			// nps5 KYTY_DUMP_STAT=1 (with KYTY_LAT_TRACE): no file; "[klat] diff" = mean absolute
			// difference of every 4th pixel against the previous presented frame (x100, 0 =
			// identical), to find the first frame an input changed.
			static std::vector<uint8_t> prev;
			const auto*                 cur = static_cast<const uint8_t*>(m_mapped);
			uint64_t                    acc = 0, cnt = 0;
			if (prev.size() == bytes) {
				for (uint32_t i = 0; i < bytes; i += 16) {
					acc += static_cast<uint64_t>(std::abs(int(cur[i]) - int(prev[i]))) +
					       std::abs(int(cur[i + 1]) - int(prev[i + 1])) +
					       std::abs(int(cur[i + 2]) - int(prev[i + 2]));
					cnt += 3;
				}
			}
			prev.assign(cur, cur + bytes);
			Common::LatTrace::Log("diff", cnt != 0 ? static_cast<long long>(acc * 100 / cnt) : -1);
			return;
		}
		std::vector<uint8_t> pixels(bytes);
		std::memcpy(pixels.data(), m_mapped, bytes);
		uint64_t sum = 0;
		for (uint32_t i = 0; i < bytes; i += 4) {
			sum += pixels[i] + pixels[i + 1] + pixels[i + 2];
			pixels[i + 3] = 0xff;
		}
		if (auto* file = std::fopen(DumpPath(name).c_str(), "wb"); file != nullptr) {
			std::fwrite(header, 1, sizeof(header), file);
			std::fwrite(pixels.data(), 1, bytes, file);
			std::fclose(file);
		}
		ScanGuest(name);
		std::fprintf(stderr, "[kyty-dump] %s: present %llu, %ux%u, mean %.1f\n", name,
		             static_cast<unsigned long long>(m_presents), m_extent.width, m_extent.height,
		             static_cast<double>(sum) / (static_cast<double>(bytes) * 0.75));
	}

	// KYTY_DUMP_GUEST_RANGE=<hex address>:<hex size> (diagnostic): with each dump, the guest bytes
	// of that range (e.g. the display buffer) as runs of 64 KiB blocks: zero, one repeated 32-bit
	// word, or mixed (first word shown).
	static void ScanGuest(const char* name) {
		const char* text = std::getenv("KYTY_DUMP_GUEST_RANGE");
		if (text == nullptr) {
			return;
		}
		char*          end  = nullptr;
		const uint64_t base = std::strtoull(text, &end, 16);
		const uint64_t size =
		    end != nullptr && *end == ':' ? std::strtoull(end + 1, nullptr, 16) : 0;
		const char* block_text = std::getenv("KYTY_DUMP_GUEST_BLOCK"); // hex bytes, default 64 KiB
		const uint64_t Block   = block_text != nullptr
		                             ? std::max<uint64_t>(std::strtoull(block_text, nullptr, 16), 4)
		                             : 0x10000;
		int            last_kind = -1;
		uint32_t       last_word = 0;
		uint64_t       run_start = base;
		const auto     flush     = [&](uint64_t at) {
			if (last_kind >= 0) {
				std::fprintf(stderr, "[kyty-dump] %s guest 0x%llx +0x%llx: %s 0x%08x\n", name,
				             static_cast<unsigned long long>(run_start),
				             static_cast<unsigned long long>(at - run_start),
				             last_kind == 0   ? "zero"
				             : last_kind == 1 ? "uniform"
				                              : "mixed",
				             last_word);
			}
			run_start = at;
		};
		for (uint64_t at = base; at + Block <= base + size; at += Block) {
			const auto* words =
			    static_cast<const uint32_t*>(Libs::LibKernel::Memory::GuestReadPtr(at, Block));
			uint32_t first = 0;
			std::memcpy(&first, words, 4);
			bool uniform = true;
			for (uint64_t i = 1; i < Block / 4 && uniform; i++) {
				uint32_t word = 0;
				std::memcpy(&word, words + i, 4);
				uniform = word == first;
			}
			const int kind = !uniform ? 2 : first == 0 ? 0 : 1;
			if (kind != last_kind || (kind == 1 && first != last_word)) {
				flush(at);
				last_kind = kind;
				last_word = first;
			}
		}
		flush(base + size);
	}

private:
	FrameDump() {
		const auto number = [](const char* name, int fallback) {
			const char* value = std::getenv(name);
			return value != nullptr ? std::atoi(value) : fallback;
		};
		m_limit = number("KYTY_DUMP_FRAMES", 0);
		m_every = std::max(number("KYTY_DUMP_FRAMES_EVERY", 1), 1);
		m_full  = number("KYTY_DUMP_SCALE", 0) == 1;
		m_from  = static_cast<uint64_t>(std::max(number("KYTY_DUMP_FRAMES_FROM", 0), 0));
		if (const char* trigger = std::getenv("KYTY_DUMP_FRAMES_TRIGGER");
		    trigger != nullptr && trigger[0] != '\0') {
			m_trigger    = trigger;
			m_from_after = m_from;
			m_from       = UINT64_MAX;
		}
	}
	void Ensure(GraphicContext& graphics, vk::Format format, vk::Extent2D extent) {
		if (m_image != nullptr && format == m_format && extent == m_extent) {
			return;
		}
		auto device = graphics.device;
		if (m_image != nullptr) {
			static_cast<void>(device.waitIdle());
			device.unmapMemory(m_buffer_memory);
			device.destroyBuffer(m_buffer, nullptr);
			device.freeMemory(m_buffer_memory, nullptr);
			device.destroyImage(m_image, nullptr);
			device.freeMemory(m_image_memory, nullptr);
		}
		m_format               = format;
		m_extent               = extent;
		const auto memory_type = [&](uint32_t bits, vk::MemoryPropertyFlags wanted) {
			const auto& memory = graphics.physical_device_memory_properties;
			for (uint32_t i = 0; i < memory.memoryTypeCount; i++) {
				if ((bits & (1u << i)) != 0 &&
				    (memory.memoryTypes[i].propertyFlags & wanted) == wanted) {
					return i;
				}
			}
			EXIT("frame dump: no memory type\n");
			return 0u;
		};
		vk::ImageCreateInfo create {};
		create.imageType   = vk::ImageType::e2D;
		create.extent      = vk::Extent3D {extent.width, extent.height, 1};
		create.mipLevels   = 1;
		create.arrayLayers = 1;
		create.format      = format;
		create.usage = vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst;
		create.samples = vk::SampleCountFlagBits::e1;
		RequireVulkanSuccess(device.createImage(&create, nullptr, &m_image), "frame dump image");
		vk::MemoryRequirements requirements {};
		device.getImageMemoryRequirements(m_image, &requirements);
		vk::MemoryAllocateInfo allocate {};
		allocate.allocationSize  = requirements.size;
		allocate.memoryTypeIndex = memory_type(requirements.memoryTypeBits, {});
		RequireVulkanSuccess(device.allocateMemory(&allocate, nullptr, &m_image_memory),
		                     "frame dump image memory");
		device.bindImageMemory(m_image, m_image_memory, 0);
		vk::BufferCreateInfo buffer {};
		buffer.size  = static_cast<vk::DeviceSize>(extent.width) * extent.height * 4u;
		buffer.usage = vk::BufferUsageFlagBits::eTransferDst;
		RequireVulkanSuccess(device.createBuffer(&buffer, nullptr, &m_buffer), "frame dump buffer");
		device.getBufferMemoryRequirements(m_buffer, &requirements);
		allocate.allocationSize = requirements.size;
		allocate.memoryTypeIndex =
		    memory_type(requirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible |
		                                                 vk::MemoryPropertyFlagBits::eHostCoherent);
		RequireVulkanSuccess(device.allocateMemory(&allocate, nullptr, &m_buffer_memory),
		                     "frame dump buffer memory");
		device.bindBufferMemory(m_buffer, m_buffer_memory, 0);
		RequireVulkanSuccess(device.mapMemory(m_buffer_memory, 0, VK_WHOLE_SIZE, {}, &m_mapped),
		                     "frame dump map");
	}

	int              m_limit    = 0;
	int              m_every    = 1;
	int              m_dumped   = 0;
	bool             m_full     = false;
	bool             m_recorded = false;
	uint64_t         m_presents = 0;
	mutable uint64_t m_from     = 0; // KYTY_DUMP_FRAMES_FROM: first present considered
	std::string      m_trigger;      // KYTY_DUMP_FRAMES_TRIGGER
	mutable bool     m_armed      = false;
	uint64_t         m_from_after = 0; // with a trigger: presents after it before the first dump
	vk::Format       m_format   = vk::Format::eUndefined;
	vk::Extent2D     m_extent {};
	vk::Image        m_image         = nullptr;
	vk::DeviceMemory m_image_memory  = nullptr;
	vk::Buffer       m_buffer        = nullptr;
	vk::DeviceMemory m_buffer_memory = nullptr;
	void*            m_mapped        = nullptr;
};

// nps5 diagnostic: KYTY_DUMP_TARGETS=<n> [KYTY_DUMP_TARGETS_FROM=<flip>]
// [KYTY_DUMP_TARGETS_EVERY=<m>] [KYTY_DUMP_TARGETS_MINW=<px>, default 512]
// [KYTY_DUMP_TARGETS_ONLYW=<px>: images of exactly that width] [KYTY_DUMP_TARGETS_MAXW=<px>,
// default 960]: at such a flip, level 0 / layer 0 of every cached colour image at least that wide
// is blitted to an sRGB-encoded BGRA image of at most MAXW px width and saved as
// k3_<dump>_<address>_<w>x<h>_<format>.bmp in the working directory, with one [kyty-target] line
// per image (address, extent, formats, usage, state). The files are written KYTY_DUMP_TARGETS_DELAY
// (20) flips later, when the recording command buffer has long completed.
class TargetDump final {
public:
	static TargetDump& Instance() {
		static TargetDump dump;
		return dump;
	}
	void OnFlip(GraphicContext& graphics, TextureCache& cache, CommandBuffer& command_buffer) {
		if (m_limit <= 0) {
			return;
		}
		++m_flips;
		if (!m_pending.empty() && m_flips >= m_pending_flip + m_delay) {
			Write(graphics);
		}
		// memtrk: KYTY_DUMP_TARGETS_AFTER_S=<s>: the series starts at the first flip at least s
		// seconds after the first flip (KYTY_DUMP_TARGETS_FROM is then counted from that flip).
		if (m_after_s > 0 && !m_after_armed) {
			const auto now = std::chrono::steady_clock::now();
			if (m_first_flip_time == std::chrono::steady_clock::time_point {}) {
				m_first_flip_time = now;
			}
			if (std::chrono::duration<double>(now - m_first_flip_time).count() < m_after_s) {
				return;
			}
			m_after_armed = true;
			m_from += m_flips - 1;
			std::fprintf(
			    stderr,
			    "[kyty-target] time trigger %d s reached at flip %llu: series from flip %llu\n",
			    m_after_s, static_cast<unsigned long long>(m_flips),
			    static_cast<unsigned long long>(m_from));
		}
		// A series with KYTY_DUMP_TARGETS_EVERY below the delay (consecutive flips for a flicker
		// check) is recorded whole and written after its last dump.
		// KYTY_DUMP_TARGETS_BURST=<k> (default 1): k consecutive flips at each dump point (two give
		// a frame-to-frame difference per point), written together after the last.
		const auto phase = m_flips < m_from ? m_every : (m_flips - m_from) % m_every;
		if ((!m_pending.empty() && m_every >= m_delay && phase == 0) || m_dumped >= m_limit ||
		    m_flips < m_from || phase >= m_burst) {
			return;
		}
		command_buffer.EndRendering();
		auto      command = command_buffer.Handle();
		const int index   = m_dumped++;
		m_pending_flip    = m_flips;
		cache.ForEachImageDebug([&](auto, Image& image) {
			const auto& info    = image.info;
			const auto  format  = image.backing.format;
			const auto  name    = vk::to_string(format);
			// sarostex2: KYTY_DUMP_BC=1 (default off): raw level-0 / layer-0 texel blocks of
			// block-compressed images, and of the images named in KYTY_DUMP_BC_ADDRS (any
			// uncompressed format), to a host buffer, written as .bin + .json (+ .dds for BC) next
			// to the BMPs.
			if (m_bc && image.backing.image != nullptr && info.samples == 1 && !info.IsDepth() &&
			    BcWanted(info.data.address, info.data.size, image.backing.extent.width,
			             info.IsBlock())) {
				const uint32_t bw   = image.backing.extent.width;
				const uint32_t bh   = image.backing.extent.height;
				const bool     bcf  = IsBcFormat(format);
				const uint64_t bpb  = vk::blockSize(format);
				const uint64_t cols = bcf ? (bw + 3) / 4 : bw;
				const uint64_t rows = bcf ? (bh + 3) / 4 : bh;
				// saroslight: volumes (the baked-lighting bricks) are dumped whole, all depth
				// slices of level 0.
				const uint32_t slices =
				    info.IsVolume() ? std::max(image.backing.extent.depth, 1u) : 1u;
				const uint64_t bytes = cols * rows * bpb * slices;
				if (bpb != 0 && bytes <= (96ull << 20) && m_bc_bytes + bytes <= (768ull << 20) &&
				    m_pending.size() < m_bc_max) {
					m_bc_bytes += bytes;
					Entry entry {};
					entry.bc            = true;
					entry.bc_bytes      = bytes;
					entry.bc_bcf        = bcf;
					entry.bc_format     = format;
					entry.extent.width  = bw;
					entry.extent.height = bh;
					entry.address       = info.data.address;
					entry.guest_format  = static_cast<unsigned>(info.guest_format);
					entry.tile_mode     = static_cast<unsigned>(info.tile_mode);
					entry.levels        = info.resources.levels;
					entry.layers        = info.IsVolume() ? slices : info.resources.layers;
					std::snprintf(entry.name, sizeof(entry.name), "bc_%02d_%llx_%ux%u%s%u_%s",
					              index, static_cast<unsigned long long>(info.data.address), bw, bh,
					              info.IsVolume() ? "x" : "_L", info.IsVolume() ? slices : 1u,
					              name.c_str());
					Create(graphics, entry);
					image.Transit(vk::ImageLayout::eTransferSrcOptimal,
					              vk::AccessFlagBits2::eTransferRead, {}, command);
					vk::BufferImageCopy copy {};
					copy.imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
					copy.imageExtent      = vk::Extent3D {bw, bh, slices};
					command.copyImageToBuffer(image.backing.image,
					                          vk::ImageLayout::eTransferSrcOptimal, entry.buffer, 1,
					                          &copy);
					m_pending.push_back(entry);
					std::fprintf(stderr,
					             "[kyty-bc] %d flip %llu image 0x%llx +0x%llx %ux%u %s "
					             "guest-format %u tile %u levels %u layers %u "
					             "%llu bytes\n",
					             index, static_cast<unsigned long long>(m_flips),
					             static_cast<unsigned long long>(info.data.address),
					             static_cast<unsigned long long>(info.data.size), bw, bh,
					             name.c_str(), entry.guest_format, entry.tile_mode, entry.levels,
					             entry.layers, static_cast<unsigned long long>(bytes));
				} else {
					std::fprintf(stderr,
					             "[kyty-bc] skipped image 0x%llx %ux%u %s (%llu bytes, budget)\n",
					             static_cast<unsigned long long>(info.data.address), bw, bh,
					             name.c_str(), static_cast<unsigned long long>(bytes));
				}
				return;
			}
			const bool  integer = name.find("int") != std::string::npos;
			const bool  size_ok = image.backing.image != nullptr && !info.IsBlock() &&
			                      !info.IsVolume() && image.backing.extent.width >= m_min_width &&
			                      (m_only_width == 0 || image.backing.extent.width == m_only_width);
			// memtrk: KYTY_DUMP_TARGETS_DEPTH=1 also dumps depth images (depth aspect, level 0 /
			// layer 0, full resolution, as raw values to a host buffer, written as a min..max grey
			// BMP).
			const auto depth_fmt  = DepthAspectTransferFormat(format);
			const auto depth_size = DepthAspectTransferBytes(format);
			const bool depth_dump = m_depth && size_ok && info.IsDepth() && info.samples == 1 &&
			                        depth_size != 0 && m_pending.size() < 256;
			const bool wanted     = size_ok && (!info.IsDepth() || m_depth);
			const bool blit =
			    wanted && !info.IsDepth() && info.samples == 1 && !integer &&
			    static_cast<bool>(graphics.GetFormatProperties(format).optimalTilingFeatures &
			                      vk::FormatFeatureFlagBits::eBlitSrc) &&
			    m_pending.size() < 256;
			if (wanted) {
				std::fprintf(
				    stderr,
				    "[kyty-target] %d flip %llu image 0x%llx +0x%llx %ux%u %s guest-format %u tile "
				    "%u samples %u "
				    "levels %u layers %u usage%s%s%s%s%s gpu-modified %d cpu-dirty %d "
				    "buffer-modified %d writes %llu "
				    "metadata %u 0x%llx clear-word 0x%08x%s\n",
				    index, static_cast<unsigned long long>(m_flips),
				    static_cast<unsigned long long>(info.data.address),
				    static_cast<unsigned long long>(info.data.size), image.backing.extent.width,
				    image.backing.extent.height, name.c_str(),
				    static_cast<unsigned>(info.guest_format), static_cast<unsigned>(info.tile_mode),
				    info.samples, info.resources.levels, info.resources.layers,
				    image.usage.texture ? " texture" : "", image.usage.storage ? " storage" : "",
				    image.usage.render_target ? " rt" : "",
				    image.usage.depth_target ? " depth" : "",
				    image.usage.video_out ? " video-out" : "", image.IsGpuModified() ? 1 : 0,
				    image.IsCpuDirty() ? 1 : 0, image.IsBufferModified() ? 1 : 0,
				    static_cast<unsigned long long>(image.DefiniteWrites()),
				    static_cast<unsigned>(info.metadata.kind),
				    static_cast<unsigned long long>(info.metadata.range.address),
				    info.metadata.dcc_clear_word, blit || depth_dump ? "" : " (not dumped)");
			}
			if (depth_dump) {
				Entry entry {};
				entry.depth         = true;
				entry.depth_format  = depth_fmt;
				entry.texel_bytes   = depth_size;
				entry.extent.width  = image.backing.extent.width;
				entry.extent.height = image.backing.extent.height;
				std::snprintf(entry.name, sizeof(entry.name), "k3_%02d_%llx_%ux%u_%s.bmp", index,
				              static_cast<unsigned long long>(info.data.address),
				              image.backing.extent.width, image.backing.extent.height,
				              name.c_str());
				Create(graphics, entry);
				image.Transit(vk::ImageLayout::eTransferSrcOptimal,
				              vk::AccessFlagBits2::eTransferRead, {}, command);
				vk::BufferImageCopy copy {};
				copy.imageSubresource = {vk::ImageAspectFlagBits::eDepth, 0, 0, 1};
				copy.imageExtent      = vk::Extent3D {entry.extent.width, entry.extent.height, 1};
				command.copyImageToBuffer(image.backing.image, vk::ImageLayout::eTransferSrcOptimal,
				                          entry.buffer, 1, &copy);
				m_pending.push_back(entry);
				// nanite: the stencil aspect too (raw bytes: nonzero share and value range,
				// "_stencil"), e.g. the Nanite bit a depth export sets for the material passes'
				// stencil test.
				if (format == vk::Format::eD32SfloatS8Uint ||
				    format == vk::Format::eD24UnormS8Uint ||
				    format == vk::Format::eD16UnormS8Uint) {
					Entry stencil {};
					stencil.raw         = true;
					stencil.texel_bytes = 1;
					stencil.extent      = entry.extent;
					std::snprintf(stencil.name, sizeof(stencil.name),
					              "k3_%02d_%llx_%ux%u_stencil.bmp", index,
					              static_cast<unsigned long long>(info.data.address),
					              image.backing.extent.width, image.backing.extent.height);
					Create(graphics, stencil);
					vk::BufferImageCopy scopy {};
					scopy.imageSubresource = {vk::ImageAspectFlagBits::eStencil, 0, 0, 1};
					scopy.imageExtent =
					    vk::Extent3D {stencil.extent.width, stencil.extent.height, 1};
					command.copyImageToBuffer(image.backing.image,
					                          vk::ImageLayout::eTransferSrcOptimal, stencil.buffer,
					                          1, &scopy);
					m_pending.push_back(stencil);
				}
				return;
			}
			// memtrk: KYTY_DUMP_TARGETS_RAW=1 also dumps single-sample integer colour images (e.g.
			// a 64-bit visibility buffer): raw texels to a host buffer, written as a BMP that
			// colours each nonzero texel by a hash of its low 32 bits, plus the share of nonzero
			// texels.
			uint32_t raw_bytes = 0;
			switch (format) {
				case vk::Format::eR8Uint: raw_bytes = 1; break;
				case vk::Format::eR16Uint: raw_bytes = 2; break;
				case vk::Format::eR32Uint:
				case vk::Format::eR8G8B8A8Uint: raw_bytes = 4; break;
				case vk::Format::eR32G32Uint: raw_bytes = 8; break;
				case vk::Format::eR32G32B32A32Uint: raw_bytes = 16; break;
				default: break;
			}
			if (m_raw && wanted && integer && !info.IsDepth() && info.samples == 1 &&
			    raw_bytes != 0 && m_pending.size() < 256) {
				Entry entry {};
				entry.raw           = true;
				entry.texel_bytes   = raw_bytes;
				entry.extent.width  = image.backing.extent.width;
				entry.extent.height = image.backing.extent.height;
				std::snprintf(entry.name, sizeof(entry.name), "k3_%02d_%llx_%ux%u_%s.bmp", index,
				              static_cast<unsigned long long>(info.data.address),
				              image.backing.extent.width, image.backing.extent.height,
				              name.c_str());
				Create(graphics, entry);
				image.Transit(vk::ImageLayout::eTransferSrcOptimal,
				              vk::AccessFlagBits2::eTransferRead, {}, command);
				vk::BufferImageCopy copy {};
				copy.imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
				copy.imageExtent      = vk::Extent3D {entry.extent.width, entry.extent.height, 1};
				command.copyImageToBuffer(image.backing.image, vk::ImageLayout::eTransferSrcOptimal,
				                          entry.buffer, 1, &copy);
				m_pending.push_back(entry);
				return;
			}
			if (!blit) {
				return;
			}
			Entry entry {};
			entry.extent.width  = std::min(image.backing.extent.width, m_max_width);
			entry.extent.height = std::max(1u, image.backing.extent.height * entry.extent.width /
			                                       image.backing.extent.width);
			std::snprintf(entry.name, sizeof(entry.name), "k3_%02d_%llx_%ux%u_%s.bmp", index,
			              static_cast<unsigned long long>(info.data.address),
			              image.backing.extent.width, image.backing.extent.height, name.c_str());
			Create(graphics, entry);
			image.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead,
			              {}, command);
			const vk::ImageSubresourceRange range {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
			vk::ImageMemoryBarrier          barrier {};
			barrier.dstAccessMask       = vk::AccessFlagBits::eTransferWrite;
			barrier.oldLayout           = vk::ImageLayout::eUndefined;
			barrier.newLayout           = vk::ImageLayout::eTransferDstOptimal;
			barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			barrier.image               = entry.image;
			barrier.subresourceRange    = range;
			command.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
			                        vk::PipelineStageFlagBits::eTransfer, vk::DependencyFlags {}, 0,
			                        nullptr, 0, nullptr, 1, &barrier);
			vk::ImageBlit region {};
			region.srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
			region.srcOffsets[1]  = vk::Offset3D {static_cast<int>(image.backing.extent.width),
			                                      static_cast<int>(image.backing.extent.height), 1};
			region.dstSubresource = region.srcSubresource;
			region.dstOffsets[1]  = vk::Offset3D {static_cast<int>(entry.extent.width),
			                                      static_cast<int>(entry.extent.height), 1};
			command.blitImage(image.backing.image, vk::ImageLayout::eTransferSrcOptimal,
			                  entry.image, vk::ImageLayout::eTransferDstOptimal, 1, &region,
			                  vk::Filter::eNearest);
			barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
			barrier.dstAccessMask = vk::AccessFlagBits::eTransferRead;
			barrier.oldLayout     = vk::ImageLayout::eTransferDstOptimal;
			barrier.newLayout     = vk::ImageLayout::eTransferSrcOptimal;
			command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
			                        vk::PipelineStageFlagBits::eAllCommands, vk::DependencyFlags {},
			                        0, nullptr, 0, nullptr, 1, &barrier);
			vk::BufferImageCopy copy {};
			copy.imageSubresource = region.srcSubresource;
			copy.imageExtent      = vk::Extent3D {entry.extent.width, entry.extent.height, 1};
			command.copyImageToBuffer(entry.image, vk::ImageLayout::eTransferSrcOptimal,
			                          entry.buffer, 1, &copy);
			m_pending.push_back(entry);
		});
		std::fprintf(stderr, "[kyty-target] %d flip %llu: %zu images recorded\n", index,
		             static_cast<unsigned long long>(m_flips), m_pending.size());
	}

private:
	struct Entry {
		char             name[160] {};
		vk::Extent2D     extent {};
		vk::Image        image         = nullptr;
		vk::DeviceMemory image_memory  = nullptr;
		vk::Buffer       buffer        = nullptr;
		vk::DeviceMemory buffer_memory = nullptr;
		void*            mapped        = nullptr;
		bool             depth         = false; // memtrk: raw depth aspect in `buffer`, no `image`
		bool             raw = false; // memtrk: raw integer texels in `buffer`, no `image`
		bool             bc  = false; // sarostex2: raw level-0 texel blocks in `buffer`, no `image`
		bool             bc_bcf       = false;
		uint64_t         bc_bytes     = 0;
		vk::Format       bc_format    = vk::Format::eUndefined;
		uint64_t         address      = 0;
		unsigned         guest_format = 0, tile_mode = 0;
		uint32_t         levels = 0, layers = 0;
		vk::Format       depth_format = vk::Format::eUndefined;
		uint32_t         texel_bytes  = 4;
	};
	TargetDump() {
		const auto number = [](const char* name, int fallback) {
			const char* value = std::getenv(name);
			return value != nullptr ? std::atoi(value) : fallback;
		};
		m_limit      = number("KYTY_DUMP_TARGETS", 0);
		m_from       = static_cast<uint64_t>(std::max(number("KYTY_DUMP_TARGETS_FROM", 1), 1));
		m_every      = static_cast<uint64_t>(std::max(number("KYTY_DUMP_TARGETS_EVERY", 1), 1));
		m_delay      = static_cast<uint64_t>(std::max(number("KYTY_DUMP_TARGETS_DELAY", 20), 1));
		m_min_width  = static_cast<uint32_t>(std::max(number("KYTY_DUMP_TARGETS_MINW", 512), 1));
		m_max_width  = static_cast<uint32_t>(std::max(number("KYTY_DUMP_TARGETS_MAXW", 960), 16));
		m_only_width = static_cast<uint32_t>(std::max(number("KYTY_DUMP_TARGETS_ONLYW", 0), 0));
		m_burst      = static_cast<uint64_t>(std::max(number("KYTY_DUMP_TARGETS_BURST", 1), 1));
		m_depth      = number("KYTY_DUMP_TARGETS_DEPTH", 0) != 0;
		m_raw        = number("KYTY_DUMP_TARGETS_RAW", 0) != 0;
		m_after_s    = std::max(number("KYTY_DUMP_TARGETS_AFTER_S", 0), 0);
		m_bc         = number("KYTY_DUMP_BC", 0) != 0;
		m_bc_max     = static_cast<size_t>(std::max(number("KYTY_DUMP_BC_MAX", 400), 1));
		m_bc_all     = number("KYTY_DUMP_BC_ALL", 0) != 0;
		m_bc_minw    = static_cast<uint32_t>(std::max(number("KYTY_DUMP_BC_MINW", 16), 1));
		m_bc_maxw    = static_cast<uint32_t>(std::max(number("KYTY_DUMP_BC_MAXW", 4096), 16));
		if (const char* list = std::getenv("KYTY_DUMP_BC_ADDRS"); m_bc && list != nullptr) {
			for (const char* at = list; *at != '\0';) {
				char*          end = nullptr;
				const uint64_t a   = std::strtoull(at, &end, 16);
				if (end == at) {
					at++;
					continue;
				}
				m_bc_addrs.push_back(a);
				at = end;
			}
		}
	}
	static void Create(GraphicContext& graphics, Entry& entry) {
		auto       device      = graphics.device;
		const auto memory_type = [&](uint32_t bits, vk::MemoryPropertyFlags wanted) {
			const auto& memory = graphics.physical_device_memory_properties;
			for (uint32_t i = 0; i < memory.memoryTypeCount; i++) {
				if ((bits & (1u << i)) != 0 &&
				    (memory.memoryTypes[i].propertyFlags & wanted) == wanted) {
					return i;
				}
			}
			EXIT("target dump: no memory type\n");
			return 0u;
		};
		vk::MemoryRequirements requirements {};
		vk::MemoryAllocateInfo allocate {};
		if (!entry.depth && !entry.raw && !entry.bc) {
			vk::ImageCreateInfo create {};
			create.imageType   = vk::ImageType::e2D;
			create.extent      = vk::Extent3D {entry.extent.width, entry.extent.height, 1};
			create.mipLevels   = 1;
			create.arrayLayers = 1;
			create.format      = vk::Format::eB8G8R8A8Srgb;
			create.usage =
			    vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst;
			create.samples = vk::SampleCountFlagBits::e1;
			RequireVulkanSuccess(device.createImage(&create, nullptr, &entry.image),
			                     "target dump image");
			device.getImageMemoryRequirements(entry.image, &requirements);
			allocate.allocationSize  = requirements.size;
			allocate.memoryTypeIndex = memory_type(requirements.memoryTypeBits, {});
			RequireVulkanSuccess(device.allocateMemory(&allocate, nullptr, &entry.image_memory),
			                     "target dump image memory");
			device.bindImageMemory(entry.image, entry.image_memory, 0);
		}
		vk::BufferCreateInfo buffer {};
		buffer.size  = entry.bc
		                   ? entry.bc_bytes
		                   : static_cast<vk::DeviceSize>(entry.extent.width) * entry.extent.height *
		                         (entry.depth || entry.raw ? entry.texel_bytes : 4u);
		buffer.usage = vk::BufferUsageFlagBits::eTransferDst;
		RequireVulkanSuccess(device.createBuffer(&buffer, nullptr, &entry.buffer),
		                     "target dump buffer");
		device.getBufferMemoryRequirements(entry.buffer, &requirements);
		allocate.allocationSize = requirements.size;
		allocate.memoryTypeIndex =
		    memory_type(requirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible |
		                                                 vk::MemoryPropertyFlagBits::eHostCoherent);
		RequireVulkanSuccess(device.allocateMemory(&allocate, nullptr, &entry.buffer_memory),
		                     "target dump buffer memory");
		device.bindBufferMemory(entry.buffer, entry.buffer_memory, 0);
		RequireVulkanSuccess(
		    device.mapMemory(entry.buffer_memory, 0, VK_WHOLE_SIZE, {}, &entry.mapped),
		    "target dump map");
	}
	void Write(GraphicContext& graphics) {
		auto device = graphics.device;
		for (auto& entry: m_pending) {
			if (entry.bc) {
				WriteBc(device, entry);
				continue;
			}
			if (entry.depth) {
				WriteDepth(device, entry);
				continue;
			}
			if (entry.raw) {
				WriteRaw(device, entry);
				continue;
			}
			const uint32_t bytes = entry.extent.width * entry.extent.height * 4u;
			uint8_t        header[54] {};
			const auto     put = [&](size_t at, uint32_t value) {
				std::memcpy(header + at, &value, 4);
			};
			header[0] = 'B';
			header[1] = 'M';
			put(2, 54u + bytes);
			put(10, 54u);
			put(14, 40u);
			put(18, entry.extent.width);
			put(22, static_cast<uint32_t>(-static_cast<int32_t>(entry.extent.height)));
			header[26] = 1;
			header[28] = 32;
			put(34, bytes);
			put(38, 2835u);
			put(42, 2835u);
			std::vector<uint8_t> pixels(bytes);
			std::memcpy(pixels.data(), entry.mapped, bytes);
			uint64_t sum = 0, alpha = 0;
			for (uint32_t i = 0; i < bytes; i += 4) {
				sum += pixels[i] + pixels[i + 1] + pixels[i + 2];
				alpha += pixels[i + 3];
				pixels[i + 3] = 0xff;
			}
			if (auto* file = std::fopen(DumpPath(entry.name).c_str(), "wb"); file != nullptr) {
				std::fwrite(header, 1, sizeof(header), file);
				std::fwrite(pixels.data(), 1, bytes, file);
				std::fclose(file);
			}
			std::fprintf(stderr, "[kyty-target] %s: mean %.1f alpha %.1f\n", entry.name,
			             static_cast<double>(sum) / (static_cast<double>(bytes) * 0.75),
			             static_cast<double>(alpha) / (static_cast<double>(bytes) * 0.25));
			device.unmapMemory(entry.buffer_memory);
			device.destroyBuffer(entry.buffer, nullptr);
			device.freeMemory(entry.buffer_memory, nullptr);
			device.destroyImage(entry.image, nullptr);
			device.freeMemory(entry.image_memory, nullptr);
		}
		m_pending.clear();
	}
	static uint32_t BcDxgi(vk::Format f) {
		switch (f) {
			case vk::Format::eBc1RgbUnormBlock:
			case vk::Format::eBc1RgbaUnormBlock: return 71;
			case vk::Format::eBc1RgbSrgbBlock:
			case vk::Format::eBc1RgbaSrgbBlock: return 72;
			case vk::Format::eBc2UnormBlock: return 74;
			case vk::Format::eBc2SrgbBlock: return 75;
			case vk::Format::eBc3UnormBlock: return 77;
			case vk::Format::eBc3SrgbBlock: return 78;
			case vk::Format::eBc4UnormBlock: return 80;
			case vk::Format::eBc4SnormBlock: return 81;
			case vk::Format::eBc5UnormBlock: return 83;
			case vk::Format::eBc5SnormBlock: return 84;
			case vk::Format::eBc6HUfloatBlock: return 95;
			case vk::Format::eBc6HSfloatBlock: return 96;
			case vk::Format::eBc7UnormBlock: return 98;
			case vk::Format::eBc7SrgbBlock: return 99;
			default: return 0;
		}
	}
	static bool IsBcFormat(vk::Format f) { return BcDxgi(f) != 0; }
	// sarostex2: an image to dump: named in KYTY_DUMP_BC_ADDRS (any format; an address inside the
	// image counts) or, with no list, any block-compressed image of width KYTY_DUMP_BC_MINW..MAXW
	// (16..4096); KYTY_DUMP_BC_ALL=1 adds that range to the list.
	bool BcWanted(uint64_t address, uint64_t size, uint32_t width, bool block) const {
		for (const auto a: m_bc_addrs) {
			if (address == a || (size != 0 && a > address && a < address + size)) {
				return true;
			}
		}
		return (m_bc_addrs.empty() || m_bc_all) && block && width >= m_bc_minw &&
		       width <= m_bc_maxw;
	}
	// sarostex2: <name>.bin = raw level-0 blocks, <name>.json = format/size/tiling, <name>.dds for
	// BC (DX10 header).
	void WriteBc(vk::Device device, Entry& entry) {
		const auto*    src    = static_cast<const uint8_t*>(entry.mapped);
		const uint32_t dxgi   = BcDxgi(entry.bc_format);
		uint64_t       groups = 0, nonzero = 0;
		for (uint64_t i = 0; i + 4 <= entry.bc_bytes; i += 16) {
			uint8_t x = 0;
			for (uint64_t k = 0; k < 16 && i + k < entry.bc_bytes; k++) {
				x |= src[i + k];
			}
			nonzero += x != 0;
			groups++;
		}
		if (auto* file =
		        std::fopen(DumpPath((std::string(entry.name) + ".bin").c_str()).c_str(), "wb");
		    file != nullptr) {
			std::fwrite(src, 1, entry.bc_bytes, file);
			std::fclose(file);
		}
		if (dxgi != 0) {
			if (auto* file =
			        std::fopen(DumpPath((std::string(entry.name) + ".dds").c_str()).c_str(), "wb");
			    file != nullptr) {
				uint32_t h[37] {};  // magic, 124-byte header (31 dwords), DX10 header (5 dwords)
				h[0] = 0x20534444u; // "DDS "
				h[1] = 124;
				h[2] =
				    0x1 | 0x2 | 0x4 | 0x1000 | 0x80000; // caps height width pixelformat linearsize
				h[3]  = entry.extent.height;
				h[4]  = entry.extent.width;
				h[5]  = static_cast<uint32_t>(entry.bc_bytes);
				h[7]  = 1;           // mip count
				h[19] = 32;          // pixel format size
				h[20] = 0x4;         // DDPF_FOURCC
				h[21] = 0x30315844u; // "DX10"
				h[27] = 0x1000;      // DDSCAPS_TEXTURE
				h[32] = dxgi;        // DX10 header: format, 2D, misc, array size, alpha mode
				h[33] = 3;
				h[35] = 1;
				std::fwrite(h, 4, 37, file);
				std::fwrite(src, 1, entry.bc_bytes, file);
				std::fclose(file);
			}
		}
		if (auto* file =
		        std::fopen(DumpPath((std::string(entry.name) + ".json").c_str()).c_str(), "wb");
		    file != nullptr) {
			std::fprintf(file,
			             "{\"address\": \"0x%llx\", \"format\": \"%s\", \"guest_format\": %u, "
			             "\"tile_mode\": %u, "
			             "\"width\": %u, \"height\": %u, \"levels\": %u, \"layers\": %u, \"bc\": "
			             "%s, \"dxgi\": %u, "
			             "\"bytes\": %llu}\n",
			             static_cast<unsigned long long>(entry.address),
			             vk::to_string(entry.bc_format).c_str(), entry.guest_format,
			             entry.tile_mode, entry.extent.width, entry.extent.height, entry.levels,
			             entry.layers, entry.bc_bcf ? "true" : "false", dxgi,
			             static_cast<unsigned long long>(entry.bc_bytes));
			std::fclose(file);
		}
		std::fprintf(
		    stderr, "[kyty-bc] %s: written, %.1f%% of 16-byte groups nonzero\n", entry.name,
		    groups != 0 ? 100.0 * static_cast<double>(nonzero) / static_cast<double>(groups) : 0.0);
		device.unmapMemory(entry.buffer_memory);
		device.destroyBuffer(entry.buffer, nullptr);
		device.freeMemory(entry.buffer_memory, nullptr);
	}
	// memtrk: a depth entry as a grey BMP (black = min, white = max of the values that are neither
	// 0 nor 1) and one stats line: the share of pixels at exactly 0 and 1 (cleared near/far) tells
	// whether a pass wrote depth at all.
	void WriteDepth(vk::Device device, Entry& entry) {
		const uint64_t pixels_n = static_cast<uint64_t>(entry.extent.width) * entry.extent.height;
		std::vector<float> values(pixels_n);
		const auto*        src = static_cast<const uint8_t*>(entry.mapped);
		for (uint64_t i = 0; i < pixels_n; i++) {
			if (entry.depth_format == vk::Format::eD16Unorm) {
				uint16_t v = 0;
				std::memcpy(&v, src + i * 2, 2);
				values[i] = static_cast<float>(v) / 65535.0f;
			} else if (entry.depth_format == vk::Format::eX8D24UnormPack32) {
				uint32_t v = 0;
				std::memcpy(&v, src + i * 4, 4);
				values[i] = static_cast<float>(v & 0x00ffffffu) / 16777215.0f;
			} else {
				std::memcpy(&values[i], src + i * 4, 4);
			}
		}
		uint64_t zeros = 0, ones = 0, other = 0;
		float    lo = 1e30f, hi = -1e30f;
		for (const auto v: values) {
			if (v == 0.0f) {
				zeros++;
			} else if (v == 1.0f) {
				ones++;
			} else {
				other++;
				lo = std::min(lo, v);
				hi = std::max(hi, v);
			}
		}
		const uint32_t bytes = static_cast<uint32_t>(pixels_n * 4u);
		uint8_t        header[54] {};
		const auto put = [&](size_t at, uint32_t value) { std::memcpy(header + at, &value, 4); };
		header[0]      = 'B';
		header[1]      = 'M';
		put(2, 54u + bytes);
		put(10, 54u);
		put(14, 40u);
		put(18, entry.extent.width);
		put(22, static_cast<uint32_t>(-static_cast<int32_t>(entry.extent.height)));
		header[26] = 1;
		header[28] = 32;
		put(34, bytes);
		put(38, 2835u);
		put(42, 2835u);
		std::vector<uint8_t> out(bytes);
		const float          span = hi > lo ? hi - lo : 1.0f;
		for (uint64_t i = 0; i < pixels_n; i++) {
			const float v = values[i];
			uint8_t     g = 0;
			if (v == 1.0f) {
				g = 255;
			} else if (v != 0.0f) {
				g = static_cast<uint8_t>(std::clamp((v - lo) / span, 0.0f, 1.0f) * 253.0f + 1.0f);
			}
			out[i * 4 + 0] = g;
			out[i * 4 + 1] = g;
			out[i * 4 + 2] = g;
			out[i * 4 + 3] = 0xff;
		}
		if (auto* file = std::fopen(DumpPath(entry.name).c_str(), "wb"); file != nullptr) {
			std::fwrite(header, 1, sizeof(header), file);
			std::fwrite(out.data(), 1, bytes, file);
			std::fclose(file);
		}
		std::fprintf(stderr,
		             "[kyty-target] %s: depth %ux%u zero %.1f%% one %.1f%% other %.1f%% (min %.6g "
		             "max %.6g)\n",
		             entry.name, entry.extent.width, entry.extent.height,
		             100.0 * static_cast<double>(zeros) / static_cast<double>(pixels_n),
		             100.0 * static_cast<double>(ones) / static_cast<double>(pixels_n),
		             100.0 * static_cast<double>(other) / static_cast<double>(pixels_n),
		             other != 0 ? lo : 0.0f, other != 0 ? hi : 0.0f);
		device.unmapMemory(entry.buffer_memory);
		device.destroyBuffer(entry.buffer, nullptr);
		device.freeMemory(entry.buffer_memory, nullptr);
	}

	// memtrk: a raw integer entry: nonzero texels coloured by a hash of their low 32 bits (black =
	// zero), plus the share of nonzero texels and, for 8-byte texels, the range of the high words.
	void WriteRaw(vk::Device device, Entry& entry) {
		const uint64_t pixels_n = static_cast<uint64_t>(entry.extent.width) * entry.extent.height;
		const auto*    src      = static_cast<const uint8_t*>(entry.mapped);
		const uint32_t bytes    = static_cast<uint32_t>(pixels_n * 4u);
		std::vector<uint8_t> out(bytes);
		uint64_t             nonzero = 0;
		uint32_t             hi_min = UINT32_MAX, hi_max = 0, lo_min = UINT32_MAX, lo_max = 0;
		for (uint64_t i = 0; i < pixels_n; i++) {
			const uint8_t* t   = src + i * entry.texel_bytes;
			uint32_t       low = 0;
			std::memcpy(&low, t, std::min<uint32_t>(entry.texel_bytes, 4u));
			bool any = false;
			for (uint32_t b = 0; b < entry.texel_bytes; b++) {
				any |= t[b] != 0;
			}
			if (entry.texel_bytes >= 8) {
				uint32_t hi = 0;
				std::memcpy(&hi, t + 4, 4);
				if (any) {
					hi_min = std::min(hi_min, hi);
					hi_max = std::max(hi_max, hi);
				}
			}
			uint8_t r = 0, g = 0, b = 0;
			if (any) {
				nonzero++;
				lo_min     = std::min(lo_min, low);
				lo_max     = std::max(lo_max, low);
				uint32_t h = low * 2654435761u;
				h ^= h >> 15;
				r = static_cast<uint8_t>(64 + (h & 0xbf));
				g = static_cast<uint8_t>(64 + ((h >> 8) & 0xbf));
				b = static_cast<uint8_t>(64 + ((h >> 16) & 0xbf));
			}
			out[i * 4 + 0] = b;
			out[i * 4 + 1] = g;
			out[i * 4 + 2] = r;
			out[i * 4 + 3] = 0xff;
		}
		uint8_t    header[54] {};
		const auto put = [&](size_t at, uint32_t value) { std::memcpy(header + at, &value, 4); };
		header[0]      = 'B';
		header[1]      = 'M';
		put(2, 54u + bytes);
		put(10, 54u);
		put(14, 40u);
		put(18, entry.extent.width);
		put(22, static_cast<uint32_t>(-static_cast<int32_t>(entry.extent.height)));
		header[26] = 1;
		header[28] = 32;
		put(34, bytes);
		put(38, 2835u);
		put(42, 2835u);
		if (auto* file = std::fopen(DumpPath(entry.name).c_str(), "wb"); file != nullptr) {
			std::fwrite(header, 1, sizeof(header), file);
			std::fwrite(out.data(), 1, bytes, file);
			std::fclose(file);
		}
		std::fprintf(stderr,
		             "[kyty-target] %s: raw %ux%u %u B/texel nonzero %.2f%% high words "
		             "0x%08x..0x%08x low 0x%x..0x%x\n",
		             entry.name, entry.extent.width, entry.extent.height, entry.texel_bytes,
		             100.0 * static_cast<double>(nonzero) / static_cast<double>(pixels_n),
		             nonzero != 0 && entry.texel_bytes >= 8 ? hi_min : 0u, hi_max,
		             nonzero != 0 ? lo_min : 0u, lo_max);
		device.unmapMemory(entry.buffer_memory);
		device.destroyBuffer(entry.buffer, nullptr);
		device.freeMemory(entry.buffer_memory, nullptr);
	}

	bool     m_depth   = false;
	bool     m_raw     = false;
	int      m_after_s = 0;
	bool     m_bc      = false; // sarostex2: KYTY_DUMP_BC
	size_t   m_bc_max  = 400;   // KYTY_DUMP_BC_MAX
	bool     m_bc_all  = false; // KYTY_DUMP_BC_ALL=1: the width range also with an address list
	uint32_t m_bc_minw = 16, m_bc_maxw = 4096;
	uint64_t m_bc_bytes = 0;
	std::vector<uint64_t> m_bc_addrs; // KYTY_DUMP_BC_ADDRS=0x..,0x.. (hex, any separator)
	bool                  m_after_armed = false;
	std::chrono::steady_clock::time_point m_first_flip_time {};
	int                m_limit        = 0;
	int                m_dumped       = 0;
	uint64_t           m_from         = 1;
	uint64_t           m_every        = 1;
	uint64_t           m_delay        = 20;
	uint32_t           m_min_width    = 512;
	uint32_t           m_max_width    = 960;
	uint32_t           m_only_width   = 0;
	uint64_t           m_burst        = 1;
	uint64_t           m_flips        = 0;
	uint64_t           m_pending_flip = 0;
	std::vector<Entry> m_pending;
};

class Swapchain final {
public:
	enum class Status : uint8_t { Success, Recreate, SurfaceLost, Minimized };

	explicit Swapchain(WindowContext& window): m_window(window) {}
	~Swapchain();
	KYTY_CLASS_NO_COPY(Swapchain);

	/// Creates the Vulkan swapchain; sets m_minimized when the window or surface extent is zero.
	void                 Create();
	/// Destroys then re-creates the swapchain, optionally recreating the Vulkan surface first.
	void                 Recreate(bool surface_lost = false);
	/// Returns true when the swapchain must be recreated before the next present.
	[[nodiscard]] bool   NeedsResize() const;
	[[nodiscard]] Status AcquireNextImage(CommandScheduler& scheduler);
	[[nodiscard]] bool   PrepareSystemOverlay();
	void     RecordPresentCommands(CommandBuffer& command, Presenter::Frame* source,
	                               const Presenter::Layer& overlay, bool draw_system_overlay);
	uint64_t Submit(CommandScheduler& scheduler);
	[[nodiscard]] Status Present();

	[[nodiscard]] uint32_t ImageCount() const noexcept {
		return static_cast<uint32_t>(m_images.size());
	}
	[[nodiscard]] vk::Format Format() const noexcept { return m_format; }
	[[nodiscard]] bool       IsMinimized() const noexcept { return m_minimized; }
	[[nodiscard]] bool IsHdrSupported() const noexcept { return m_hdr_supported.load(); }
	bool SetHdr(bool hdr) {
		const bool changed = m_hdr_active != (hdr && IsHdrSupported());
		m_hdr_requested = hdr;
		return changed;
	}

private:
	vk::SurfaceFormatKHR SelectFormat();
	/// Destroys the swapchain Vulkan objects and resets all members.
	void Destroy();
	void DrawLayer(vk::CommandBuffer command, const Presenter::Layer& layer,
	               const vk::Rect2D& rect);

	WindowContext&   m_window;
	vk::SwapchainKHR m_handle = nullptr;
	vk::Format       m_format = vk::Format::eUndefined;
	std::atomic_bool m_hdr_supported {false};
	bool             m_hdr_requested = false;
	bool             m_hdr_active    = false;
	vk::Extent2D     m_extent {};
	// Drawable pixel size observed when this swapchain was created.
	vk::Extent2D                   m_window_extent {};
	bool                           m_minimized         = false;
	/// True when m_minimized was set because the surface/drawable extent was {0,0} but
	/// WindowContext::minimized is false (e.g. compositor not ready). Cleared on successful
	/// swapchain creation or Destroy().
	bool                           m_surface_extent_zero = false;
	bool                           m_suboptimal          = false;
	std::vector<vk::Image>         m_images;
	std::vector<vk::ImageView>     m_image_views;
	std::vector<vk::Semaphore>     m_image_acquired;
	std::vector<vk::Semaphore>     m_render_complete;
	std::vector<uint64_t>          m_frame_ticks;
	std::unique_ptr<SystemOverlay> m_system_overlay;
	vk::DescriptorSetLayout        m_overlay_descriptors = nullptr;
	vk::PipelineLayout             m_overlay_layout      = nullptr;
	vk::Pipeline                   m_overlay_pipeline    = nullptr;
	vk::Sampler                    m_overlay_sampler     = nullptr;
	PresentFilter                  m_filter; // KYTY_PRESENT_BOX_DOWNSCALE
	uint32_t                       m_image_index         = 0;
	uint32_t                       m_frame_index         = 0;
};

struct Presenter::Impl {
	explicit Impl(WindowContext& owner)
	    : renderer(*owner.render_context), window(owner), swapchain(owner),
	      present_scheduler(renderer, owner.graphic_ctx, CommandScheduler::Role::Presenter),
	      frames(owner, present_scheduler) {
		swapchain.Create();
		const uint32_t image_count = swapchain.ImageCount() > 0 ? swapchain.ImageCount() : 3;
		frames.Initialize(image_count, swapchain.Format());
		if (LoadCard::Enabled()) {
			loadcard_thread = std::jthread([this](std::stop_token stop) { LoadCardLoop(stop); });
		}
	}

	void RecoverSwapchain(Swapchain::Status status) {
		LOGF("Recovering Vulkan swapchain%s\n",
		     status == Swapchain::Status::SurfaceLost ? " and surface" : "");
		Common::DebugCounters::Add(Common::DebugCounters::Counter::SwapchainRecreations);
		swapchain.Recreate(status == Swapchain::Status::SurfaceLost);
		if (!swapchain.IsMinimized()) {
			frames.SetFormat(swapchain.Format());
		}
	}

	Image& ResolveSurface(const ImageInfo& info) {
		TextureCache::ImageDesc desc {};
		desc.info                  = info;
		desc.view_info.format      = info.pixel_format;
		desc.view_info.type        = vk::ImageViewType::e2D;
		desc.view_info.aspect      = vk::ImageAspectFlagBits::eColor;
		desc.view_info.base_level  = 0;
		desc.view_info.level_count = 1;
		desc.view_info.base_layer  = 0;
		desc.view_info.layer_count = 1;
		desc.view_info.usage       = vk::ImageUsageFlagBits::eTransferSrc;
		desc.type                  = TextureCache::BindingType::VideoOut;

		auto&      cache      = renderer.GetTextureCache();
		const auto image_id   = cache.FindImage(desc);
		auto&      image      = cache.GetImage(image_id);
		image.usage.video_out = true;
		// nps5 diagnostic, KYTY_FLIP_LOG=1: the state of the flipped image before it is refreshed
		// (a CPU-dirty image is uploaded again from guest bytes), on change and every 120th flip.
		static const bool flip_log = std::getenv("KYTY_FLIP_LOG") != nullptr;
		if (flip_log) {
			static uint64_t flips = 0, dirty_flips = 0;
			static uint32_t last  = ~0u;
			const uint32_t  state = (image.IsCpuDirty() ? 1u : 0u) |
			                        (image.IsGpuModified() ? 2u : 0u) |
			                        (image.IsBufferModified() ? 4u : 0u);
			dirty_flips += state & 1u;
			if (state != last || ++flips % 120 == 0) {
				last = state;
				std::fprintf(stderr,
				             "[kyty-flip] image 0x%llx %ux%u: cpu-dirty %u gpu-modified %u "
				             "buffer-modified %u (%llu of %llu flips cpu-dirty)\n",
				             static_cast<unsigned long long>(info.data.address), info.extent.width,
				             info.extent.height, state & 1u, (state >> 1u) & 1u, (state >> 2u) & 1u,
				             static_cast<unsigned long long>(dirty_flips),
				             static_cast<unsigned long long>(flips));
			}
		}
		cache.UpdateImage(image_id);
		return image;
	}
	void Present();
	// nps5 load card: the watchdog thread (below Presenter::Impl::Present).
	void LoadCardLoop(std::stop_token stop);
	// Presents the last frame with (card = true) or without the load card, only if no game flip
	// happened since `seq` (decided under present_mutex, the lock of every present: no concurrent
	// acquire/queue present, and the card can never land on a game frame). False when nothing was
	// presented.
	bool PresentLoadCard(uint64_t seq, bool card);

	RenderContext&        renderer;
	WindowContext&        window;
	Swapchain             swapchain;
	CommandScheduler      present_scheduler;
	FramePool             frames;
	Common::Mutex         present_mutex;
	std::array<Layer, 2>  layers {};
	std::atomic<uint64_t> presented_overlay_revision {0};
	std::jthread          loadcard_thread; // last member: joined first
};

vk::SurfaceFormatKHR Swapchain::SelectFormat() {
	const auto& formats    = m_window.surface_capabilities.formats;
	auto&       graphics   = m_window.graphic_ctx;
	const auto  hdr_format = std::ranges::find_if(formats, [&](const auto& candidate) {
		return candidate.colorSpace == vk::ColorSpaceKHR::eHdr10St2084EXT &&
		       IsHdrSwapchainFormat(candidate.format) &&
		       static_cast<bool>(
		           graphics.GetFormatProperties(candidate.format).optimalTilingFeatures &
		           vk::FormatFeatureFlagBits::eBlitDst);
	});
	m_hdr_supported        = Config::HdrEnabled() && hdr_format != formats.end();
	m_hdr_active           = m_hdr_requested && IsHdrSupported();
	if (m_hdr_active) {
		return *hdr_format;
	}
	if (Config::HdrEnabled() && !IsHdrSupported()) {
		LOGF("HDR10 surface unavailable; using SDR output\n");
	}
	if (formats.size() == 1 && formats.front().format == vk::Format::eUndefined) {
		return {vk::Format::eR8G8B8A8Unorm, vk::ColorSpaceKHR::eSrgbNonlinear};
	}
	for (const auto& candidate: formats) {
		if (candidate.colorSpace != vk::ColorSpaceKHR::eSrgbNonlinear) {
			continue;
		}
		switch (candidate.format) {
			case vk::Format::eB8G8R8A8Unorm:
			case vk::Format::eR8G8B8A8Unorm: return candidate;
			default: break;
		}
	}
	EXIT("no supported UNORM swapchain format\n");
}

/// Creates the Vulkan swapchain for the current window surface.
/// Sets m_minimized and returns early when the window or surface extent is zero.
/// Sets m_surface_extent_zero when the early return is due to a zero surface/drawable extent
/// and WindowContext::minimized is false, so NeedsResize() can poll instead of spinning.
void Swapchain::Create() {
	auto& graphics = m_window.graphic_ctx;
	EXIT_IF(graphics.device == nullptr);
	EXIT_IF(m_window.surface == nullptr);

	m_window.RefreshSurfaceCapabilities();
	const auto& surface = m_window.surface_capabilities;
	EXIT_NOT_IMPLEMENTED(surface.formats.empty());

	const auto format = SelectFormat();
	m_format                      = format.format;
	const auto swapchain_features = graphics.GetFormatProperties(m_format).optimalTilingFeatures;
	if (!static_cast<bool>(swapchain_features & vk::FormatFeatureFlagBits::eBlitDst)) {
		EXIT("swapchain format cannot be a blit destination: format=%d\n",
		     static_cast<int>(m_format));
	}

	if (m_window.minimized.load(std::memory_order_acquire)) {
		// Window is genuinely minimized; m_surface_extent_zero is not the cause.
		m_minimized           = true;
		m_surface_extent_zero = false;
		return;
	}

	// Capture the size before querying surface capabilities. If the window changes
	// during creation, the next presentation will detect the newer size.
	{
		Common::LockGuard lock(m_window.mutex);
		m_window_extent = {graphics.screen_width, graphics.screen_height};
	}
	if (m_window_extent.width == 0 || m_window_extent.height == 0) {
		LOGF("Swapchain::Create(): drawable extent is zero while window is not minimized; "
		     "waiting for surface to become ready\n");
		m_minimized           = true;
		m_surface_extent_zero = true;
		return;
	}

	m_extent = surface.capabilities.currentExtent;
	if (m_extent.width == std::numeric_limits<uint32_t>::max()) {
		// winresize: the surface lets the swapchain choose: the drawable size clamped to the
		// surface limits and to KYTY_SWAPCHAIN_MAX_MB (default 512 MiB of images; the self memory
		// limit is never raised for a big window).
		static const uint64_t budget = [] {
			const char* v  = std::getenv("KYTY_SWAPCHAIN_MAX_MB");
			const long  mb = v != nullptr && *v != '\0' ? std::strtol(v, nullptr, 10) : 512;
			return static_cast<uint64_t>(mb > 0 ? mb : 512) << 20u;
		}();
		const auto& caps   = surface.capabilities;
		auto        chosen = PresentGeometry::ClampExtent(
		    {m_window_extent.width, m_window_extent.height}, {m_extent.width, m_extent.height},
		    {caps.minImageExtent.width, caps.minImageExtent.height},
		    {caps.maxImageExtent.width, caps.maxImageExtent.height});
		chosen = PresentGeometry::FitToBudget(chosen, std::max(caps.minImageCount + 1, 3u), budget);
		m_extent.width  = chosen.w;
		m_extent.height = chosen.h;
	}
	if (m_extent.width == 0 || m_extent.height == 0) {
		LOGF("Swapchain::Create(): surface currentExtent is zero while window is not minimized; "
		     "waiting for surface to become ready\n");
		m_minimized           = true;
		m_surface_extent_zero = true;
		return;
	}

	m_minimized           = false;
	m_surface_extent_zero = false;
	m_suboptimal          = false;

	uint32_t image_count = surface.capabilities.minImageCount + 1;
	if (surface.capabilities.maxImageCount != 0) {
		image_count = std::min(image_count, surface.capabilities.maxImageCount);
	}
	const auto transform =
	    surface.capabilities.supportedTransforms & vk::SurfaceTransformFlagBitsKHR::eIdentity
	        ? vk::SurfaceTransformFlagBitsKHR::eIdentity
	        : surface.capabilities.currentTransform;
	const auto composite =
	    surface.capabilities.supportedCompositeAlpha & vk::CompositeAlphaFlagBitsKHR::eOpaque
	        ? vk::CompositeAlphaFlagBitsKHR::eOpaque
	        : vk::CompositeAlphaFlagBitsKHR::eInherit;

	vk::SwapchainCreateInfoKHR create_info {};
	create_info.sType            = vk::StructureType::eSwapchainCreateInfoKHR;
	create_info.surface          = m_window.surface;
	create_info.minImageCount    = image_count;
	create_info.imageFormat      = format.format;
	create_info.imageColorSpace  = format.colorSpace;
	create_info.imageExtent      = m_extent;
	create_info.imageArrayLayers = 1;
	create_info.imageUsage =
	    vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferDst;
	if (FrameDump::Instance().Enabled()) {
		create_info.imageUsage |= vk::ImageUsageFlagBits::eTransferSrc; // KYTY_DUMP_FRAMES
	}
	create_info.imageSharingMode = vk::SharingMode::eExclusive;
	create_info.preTransform     = transform;
	create_info.compositeAlpha   = composite;
	switch (Config::GetPresentMode()) {
		case Config::PresentMode::Mailbox:
			create_info.presentMode = vk::PresentModeKHR::eMailbox;
			break;
		case Config::PresentMode::Immediate:
			create_info.presentMode = vk::PresentModeKHR::eImmediate;
			break;
		case Config::PresentMode::Fifo:
		default: create_info.presentMode = vk::PresentModeKHR::eFifo; break;
	}
	// nps5 KYTY_PRESENT_MODE=fifo|immediate|mailbox overrides the configured mode (KosmicKrisp
	// offers immediate and fifo only; immediate = CAMetalLayer displaySyncEnabled=NO: the frame is
	// shown at once instead of at the next vblank).
	if (const char* mode = std::getenv("KYTY_PRESENT_MODE"); mode != nullptr) {
		if (std::strcmp(mode, "immediate") == 0)
			create_info.presentMode = vk::PresentModeKHR::eImmediate;
		else if (std::strcmp(mode, "mailbox") == 0)
			create_info.presentMode = vk::PresentModeKHR::eMailbox;
		else if (std::strcmp(mode, "fifo") == 0)
			create_info.presentMode = vk::PresentModeKHR::eFifo;
	}
	if (std::find(surface.present_modes.begin(), surface.present_modes.end(),
	              create_info.presentMode) == surface.present_modes.end()) {
		LOGF("warning: requested present mode is unavailable; falling back to Fifo\n");
		create_info.presentMode = vk::PresentModeKHR::eFifo;
	}
	create_info.clipped = VK_TRUE;
#if defined(_WIN32)
	// KYTY_FULLSCREEN_EXCLUSIVE (GraphicContext::full_screen_exclusive): exclusive fullscreen
	// disallowed or allowed. (VkSurfaceFullScreenExclusiveInfoEXT is vulkan_win32.h's, not included
	// here: its layout.)
	struct {
		VkStructureType sType         = VK_STRUCTURE_TYPE_SURFACE_FULL_SCREEN_EXCLUSIVE_INFO_EXT;
		const void*     pNext         = nullptr;
		int32_t full_screen_exclusive = 0; // VkFullScreenExclusiveEXT: 1 allowed, 2 disallowed
	} exclusive;
	if (graphics.full_screen_exclusive >= 0) {
		exclusive.pNext                 = create_info.pNext;
		exclusive.full_screen_exclusive = graphics.full_screen_exclusive == 0 ? 2 : 1;
		create_info.pNext               = &exclusive;
	}
#endif
	if (Common::LatTrace::Enabled()) {
		std::fprintf(stderr, "[klat] swapchain present_mode=%s images=%u\n",
		             vk::to_string(create_info.presentMode).c_str(), image_count);
	}
	create_info.pNext = LowLatency::SwapchainCreateNext(create_info.pNext);
	RequireVulkanSuccess(graphics.device.createSwapchainKHR(&create_info, nullptr, &m_handle),
	                     "vkCreateSwapchainKHR");
	EXIT_IF(m_handle == nullptr);
	LowLatency::OnSwapchainCreated(static_cast<VkSwapchainKHR>(m_handle));
	LOGF("Vulkan presentation: %s\n", m_hdr_active ? "HDR10 (PQ)" : "SDR");

	m_images = EnumerateVulkan<vk::Image>(
	    "vkGetSwapchainImagesKHR", [&](uint32_t* count, vk::Image* images) {
		    return graphics.device.getSwapchainImagesKHR(m_handle, count, images);
	    });
	EXIT_NOT_IMPLEMENTED(m_images.empty());
	// Freeze reports depend on it (RTX 50 PCs froze in Mailbox, not in Fifo).
	LOGF("Swapchain: %ux%u, %zu images, present mode %s\n", m_extent.width, m_extent.height,
	     m_images.size(), vk::to_string(create_info.presentMode).c_str());

	m_image_views.resize(m_images.size());
	for (size_t i = 0; i < m_images.size(); i++) {
		vk::ImageViewCreateInfo view {};
		view.sType                           = vk::StructureType::eImageViewCreateInfo;
		view.image                           = m_images[i];
		view.viewType                        = vk::ImageViewType::e2D;
		view.format                          = m_format;
		view.components                      = {};
		view.subresourceRange.aspectMask     = vk::ImageAspectFlagBits::eColor;
		view.subresourceRange.baseArrayLayer = 0;
		view.subresourceRange.baseMipLevel   = 0;
		view.subresourceRange.layerCount     = 1;
		view.subresourceRange.levelCount     = 1;
		RequireVulkanSuccess(graphics.device.createImageView(&view, nullptr, &m_image_views[i]),
		                     "vkCreateImageView");
		EXIT_IF(m_image_views[i] == nullptr);
	}

	vk::SemaphoreCreateInfo semaphore_info {};
	semaphore_info.sType = vk::StructureType::eSemaphoreCreateInfo;
	m_image_acquired.resize(m_images.size());
	m_render_complete.resize(m_images.size());
	m_frame_ticks.assign(m_images.size(), 0);
	for (size_t i = 0; i < m_images.size(); i++) {
		RequireVulkanSuccess(
		    graphics.device.createSemaphore(&semaphore_info, nullptr, &m_image_acquired[i]),
		    "create swapchain image-acquired semaphore");
		RequireVulkanSuccess(
		    graphics.device.createSemaphore(&semaphore_info, nullptr, &m_render_complete[i]),
		    "create swapchain render-complete semaphore");
	}
}

Swapchain::~Swapchain() {
	Destroy();
}

/// Destroys all Vulkan swapchain objects and resets all members to their default state.
void Swapchain::Destroy() {
	m_minimized           = false;
	m_surface_extent_zero = false;
	m_suboptimal          = false;
	if (m_handle == nullptr && m_image_acquired.empty() && m_render_complete.empty() &&
	    m_image_views.empty()) {
		return;
	}
	auto& graphics = m_window.graphic_ctx;

	{
		Common::LockGuard queue_lock(graphics.queue_mutex);
		graphics.submission_queue.DrainPendingLocked();
		RequireVulkanSuccess(graphics.queue.waitIdle(), "wait for swapchain queue");
	}
	if (m_system_overlay != nullptr) {
		m_system_overlay->ReleaseVulkan();
	}
	graphics.device.destroyPipeline(m_overlay_pipeline, nullptr);
	graphics.device.destroyPipelineLayout(m_overlay_layout, nullptr);
	graphics.device.destroyDescriptorSetLayout(m_overlay_descriptors, nullptr);
	graphics.device.destroySampler(m_overlay_sampler, nullptr);
	m_overlay_pipeline    = nullptr;
	m_overlay_layout      = nullptr;
	m_overlay_descriptors = nullptr;
	m_overlay_sampler     = nullptr;
	m_filter.Release(graphics.device);

	for (const auto semaphore: m_image_acquired) {
		if (semaphore != nullptr) {
			graphics.device.destroySemaphore(semaphore, nullptr);
		}
	}
	for (const auto semaphore: m_render_complete) {
		if (semaphore != nullptr) {
			graphics.device.destroySemaphore(semaphore, nullptr);
		}
	}
	for (const auto view: m_image_views) {
		if (view != nullptr) {
			graphics.device.destroyImageView(view, nullptr);
		}
	}
	if (m_handle != nullptr) {
		LowLatency::OnSwapchainDestroyed(static_cast<VkSwapchainKHR>(m_handle));
		graphics.device.destroySwapchainKHR(m_handle, nullptr);
	}

	m_handle        = nullptr;
	m_format        = vk::Format::eUndefined;
	m_extent        = {};
	m_window_extent = {};
	m_frame_index   = 0;
	m_images.clear();
	m_image_views.clear();
	m_image_acquired.clear();
	m_render_complete.clear();
	m_frame_ticks.clear();
}

/// Destroys and re-creates the swapchain, optionally recreating the Vulkan surface first.
void Swapchain::Recreate(bool surface_lost) {
	Destroy();
	if (surface_lost) {
#if defined(__APPLE__)
		// Surface recreation goes through SDL_Vulkan_CreateSurface, which touches the
		// window's view/layer and must run on the main thread on macOS.
		EXIT_IF(!SDL_RunOnMainThread(
		    [](void* window) { static_cast<WindowContext*>(window)->RecreateSurface(); }, &m_window,
		    true));
#else
		m_window.RecreateSurface();
#endif
	}
	Create();
}

/// Returns true when the swapchain must be recreated before the next present.
/// In the m_surface_extent_zero case (compositor not ready, window not minimized) this
/// re-queries the surface capabilities and only returns true when the extent is usable,
/// preventing a recreate-every-frame busy-loop.
bool Swapchain::NeedsResize() const {
	const bool window_minimized = m_window.minimized.load(std::memory_order_acquire);
	if (m_minimized) {
		if (window_minimized) {
			// OS window is minimized; nothing to do yet.
			return false;
		}
		if (!m_surface_extent_zero) {
			// Swapchain was marked minimized because WindowContext::minimized was true at
			// creation time, but it has since been cleared -> recreate now.
			return true;
		}
		// m_surface_extent_zero: the OS window exists but the compositor has not yet
		// assigned a real extent. Re-query and only trigger a recreate once the extent
		// is non-zero; otherwise return false to avoid a spinning recreate loop.
		m_window.RefreshSurfaceCapabilities();
		const auto& caps   = m_window.surface_capabilities.capabilities;
		const auto& extent = caps.currentExtent;
		// Extent 0xFFFFFFFF means the surface size is determined by the swapchain;
		// in that case read the cached drawable size under the window mutex.
		if (extent.width == std::numeric_limits<uint32_t>::max()) {
			Common::LockGuard lock(m_window.mutex);
			const bool drawable_ready = m_window.graphic_ctx.screen_width > 0 &&
			                            m_window.graphic_ctx.screen_height > 0;
			return drawable_ready;
		}
		Common::LockGuard lock(m_window.mutex);
		const bool drawable_ready = m_window.graphic_ctx.screen_width > 0 &&
		                            m_window.graphic_ctx.screen_height > 0;
		return drawable_ready && extent.width > 0 && extent.height > 0;
	}
	if (window_minimized) {
		return true;
	}
	Common::LockGuard lock(m_window.mutex);
	return m_window_extent.width != m_window.graphic_ctx.screen_width ||
	       m_window_extent.height != m_window.graphic_ctx.screen_height;
}

Swapchain::Status Swapchain::AcquireNextImage(CommandScheduler& scheduler) {
	if (m_minimized || m_handle == nullptr) {
		return Status::Minimized;
	}
	if (m_frame_index < m_frame_ticks.size() && m_frame_ticks[m_frame_index] != 0) {
		scheduler.Wait(m_frame_ticks[m_frame_index]);
		m_frame_ticks[m_frame_index] = 0;
	}
	vk::Result result;
	{
		// Hang reports (issue #62, Linux): shows a present thread blocked in the driver's acquire.
		HangWatchdog::Scope acquire(
		    "vkAcquireNextImageKHR",
		    reinterpret_cast<uint64_t>(static_cast<VkSwapchainKHR>(m_handle)), m_frame_index);
		result = m_window.graphic_ctx.device.acquireNextImageKHR(
		    m_handle, std::numeric_limits<uint64_t>::max(), m_image_acquired[m_frame_index],
		    nullptr, &m_image_index);
	}
	switch (result) {
		case vk::Result::eSuccess: break;
		case vk::Result::eSuboptimalKHR:
			LOGF("vkAcquireNextImageKHR returned vk::Result::eSuboptimalKHR\n");
			m_suboptimal = true;
			break;
		case vk::Result::eErrorOutOfDateKHR:
			LOGF("vkAcquireNextImageKHR returned vk::Result::eErrorOutOfDateKHR\n");
			return Status::Recreate;
		case vk::Result::eErrorUnknown:
			LOGF("vkAcquireNextImageKHR returned vk::Result::eErrorUnknown\n");
			return Status::Recreate;
		case vk::Result::eErrorSurfaceLostKHR:
			LOGF("vkAcquireNextImageKHR returned vk::Result::eErrorSurfaceLostKHR\n");
			return Status::SurfaceLost;
		default: EXIT("vkAcquireNextImageKHR failed: %s\n", vk::to_string(result).c_str());
	}
	return Status::Success;
}

bool Swapchain::PrepareSystemOverlay() {
	if (m_system_overlay == nullptr) {
		m_system_overlay = std::make_unique<SystemOverlay>(m_window.graphic_ctx);
	}
	return m_system_overlay->PrepareFrame(m_extent, m_format, ImageCount());
}

// KYTY_PRESENT_BOX_DOWNSCALE=1: present 1-2x downscales with PresentFilter (windowInternal.h).
static bool PresentBoxDownscaleEnabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_PRESENT_BOX_DOWNSCALE");
		return value != nullptr && std::strcmp(value, "1") == 0;
	}();
	return enabled;
}

// A pipeline that draws the fullscreen triangle with `fragment_code` into one `format` attachment.
static vk::Pipeline CreateFullscreenPipeline(vk::Device device, vk::PipelineLayout layout,
                                             std::span<const uint32_t> fragment_code,
                                             vk::Format format) {
	const auto vertex   = CompileSPV(GPU_BLIT_FS_TRIANGLE_SPV, device);
	const auto fragment = CompileSPV(fragment_code, device);
	std::array<vk::PipelineShaderStageCreateInfo, 2> stages {};
	stages[0].stage  = vk::ShaderStageFlagBits::eVertex;
	stages[0].module = vertex;
	stages[0].pName  = "main";
	stages[1].stage  = vk::ShaderStageFlagBits::eFragment;
	stages[1].module = fragment;
	stages[1].pName  = "main";
	vk::PipelineVertexInputStateCreateInfo   vertex_input {};
	vk::PipelineInputAssemblyStateCreateInfo assembly {};
	assembly.topology = vk::PrimitiveTopology::eTriangleList;
	vk::PipelineViewportStateCreateInfo viewport {};
	viewport.viewportCount = 1;
	viewport.scissorCount  = 1;
	vk::PipelineRasterizationStateCreateInfo rasterization {};
	rasterization.lineWidth = 1.0f;
	vk::PipelineMultisampleStateCreateInfo multisample {};
	multisample.rasterizationSamples = vk::SampleCountFlagBits::e1;
	vk::PipelineColorBlendAttachmentState attachment {};
	attachment.blendEnable    = VK_FALSE;
	attachment.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
	                            vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
	vk::PipelineColorBlendStateCreateInfo blend {};
	blend.attachmentCount = 1;
	blend.pAttachments    = &attachment;
	const std::array dynamic_states {vk::DynamicState::eViewport, vk::DynamicState::eScissor};
	vk::PipelineDynamicStateCreateInfo dynamic {};
	dynamic.dynamicStateCount = static_cast<uint32_t>(dynamic_states.size());
	dynamic.pDynamicStates    = dynamic_states.data();
	vk::PipelineRenderingCreateInfo rendering {};
	rendering.colorAttachmentCount    = 1;
	rendering.pColorAttachmentFormats = &format;
	vk::GraphicsPipelineCreateInfo create {};
	create.pNext               = &rendering;
	create.stageCount          = static_cast<uint32_t>(stages.size());
	create.pStages             = stages.data();
	create.pVertexInputState   = &vertex_input;
	create.pInputAssemblyState = &assembly;
	create.pViewportState      = &viewport;
	create.pRasterizationState = &rasterization;
	create.pMultisampleState   = &multisample;
	create.pColorBlendState    = &blend;
	create.pDynamicState       = &dynamic;
	create.layout              = layout;
	vk::Pipeline pipeline      = nullptr;
	RequireVulkanSuccess(device.createGraphicsPipelines(nullptr, 1, &create, nullptr, &pipeline),
	                     "create video-out downscale pipeline");
	device.destroyShaderModule(fragment, nullptr);
	device.destroyShaderModule(vertex, nullptr);
	return pipeline;
}

// KYTY_HDR_TO_SDR (default 1; 0 off): a game that outputs HDR10 (PQ) video-out buffers while the
// window is SDR (HDR off in the launcher or no HDR display) is converted for display: the PQ code
// values were shown unchanged, so the picture looked washed out (Miles Morales outputs PQ whatever
// the reported dynamic range). KYTY_HDR_TO_SDR_WHITE_NITS (default 203, BT.2408 reference white)
// maps to SDR white; highlights roll off up to KYTY_HDR_TO_SDR_PEAK_NITS (default 1000).
static bool PqToSdrEnabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_HDR_TO_SDR");
		return value == nullptr || std::strcmp(value, "0") != 0;
	}();
	return enabled;
}

static float EnvNits(const char* name, float fallback) {
	const auto* value  = std::getenv(name);
	const float parsed = value != nullptr ? std::strtof(value, nullptr) : 0.0f;
	return parsed >= 1.0f && parsed <= 10000.0f ? parsed : fallback;
}

float PqToSdrWhiteNits() {
	static const float nits = EnvNits("KYTY_HDR_TO_SDR_WHITE_NITS", 203.0f);
	return nits;
}

float PqToSdrPeakNits() {
	static const float nits = EnvNits("KYTY_HDR_TO_SDR_PEAK_NITS", 1000.0f);
	return nits;
}

bool PresentNeedsFilter(vk::Extent2D source, vk::Extent2D target) noexcept {
	const bool shrinks = source.width > target.width || source.height > target.height;
	const bool halves =
	    source.width == 2ull * target.width && source.height == 2ull * target.height;
	return shrinks && !halves && source.width <= 2ull * target.width &&
	       source.height <= 2ull * target.height;
}

void PresentFilter::Record(vk::Device device, vk::CommandBuffer command, vk::ImageView source_view,
                           uint32_t level, vk::Extent2D source, vk::ImageView target_view,
                           vk::Format target_format, vk::Extent2D target, bool pq_to_sdr) {
	struct Constants {
		int32_t  size[2];
		float    scale[2];
		uint32_t level;
		uint32_t flags;
		float    white_nits;
		float    peak_nits;
	};
	static_assert(sizeof(Constants) == 32);
	if (m_layout == nullptr) {
		vk::SamplerCreateInfo sampler {};
		sampler.magFilter    = vk::Filter::eLinear;
		sampler.minFilter    = vk::Filter::eLinear;
		sampler.mipmapMode   = vk::SamplerMipmapMode::eNearest;
		sampler.addressModeU = vk::SamplerAddressMode::eClampToEdge;
		sampler.addressModeV = vk::SamplerAddressMode::eClampToEdge;
		sampler.addressModeW = vk::SamplerAddressMode::eClampToEdge;
		sampler.maxLod       = VK_LOD_CLAMP_NONE;
		RequireVulkanSuccess(device.createSampler(&sampler, nullptr, &m_sampler),
		                     "create video-out downscale sampler");
		const vk::DescriptorSetLayoutBinding binding {0, vk::DescriptorType::eCombinedImageSampler,
		                                              1, vk::ShaderStageFlagBits::eFragment};
		vk::DescriptorSetLayoutCreateInfo    descriptors {};
		descriptors.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
		descriptors.bindingCount = 1;
		descriptors.pBindings    = &binding;
		RequireVulkanSuccess(device.createDescriptorSetLayout(&descriptors, nullptr, &m_descriptors),
		                     "create video-out downscale descriptor layout");
		const vk::PushConstantRange  constants {vk::ShaderStageFlagBits::eFragment, 0,
		                                        sizeof(Constants)};
		vk::PipelineLayoutCreateInfo layout {};
		layout.setLayoutCount         = 1;
		layout.pSetLayouts            = &m_descriptors;
		layout.pushConstantRangeCount = 1;
		layout.pPushConstantRanges    = &constants;
		RequireVulkanSuccess(device.createPipelineLayout(&layout, nullptr, &m_layout),
		                     "create video-out downscale pipeline layout");
	}
	if (m_pipeline != nullptr && m_format != target_format) {
		device.destroyPipeline(m_pipeline, nullptr);
		m_pipeline = nullptr;
	}
	if (m_pipeline == nullptr) {
		m_pipeline = CreateFullscreenPipeline(device, m_layout, GPU_VIDEO_OUT_DOWNSCALE_SPV, target_format);
		m_format   = target_format;
	}

	const vk::DescriptorImageInfo image {m_sampler, source_view,
	                                     vk::ImageLayout::eShaderReadOnlyOptimal};
	vk::WriteDescriptorSet        write {};
	write.dstBinding      = 0;
	write.descriptorCount = 1;
	write.descriptorType  = vk::DescriptorType::eCombinedImageSampler;
	write.pImageInfo      = &image;
	command.pushDescriptorSetKHR(vk::PipelineBindPoint::eGraphics, m_layout, 0, 1, &write);
	const Constants constants {
	    {static_cast<int32_t>(source.width), static_cast<int32_t>(source.height)},
	    {static_cast<float>(source.width) / static_cast<float>(target.width),
	     static_cast<float>(source.height) / static_cast<float>(target.height)},
	    level,
	    // The two-texel box covers at most a 2:1 downscale; beyond that (only an HDR10 frame on an
	    // SDR window takes this pass then) one linear tap per pixel, as the blit it replaces.
	    (pq_to_sdr ? 1u : 0u) |
	        (source.width > 2ull * target.width || source.height > 2ull * target.height ? 2u : 0u),
	    PqToSdrWhiteNits(),
	    PqToSdrPeakNits()};
	command.pushConstants(m_layout, vk::ShaderStageFlagBits::eFragment, 0, sizeof(constants),
	                      &constants);
	command.bindPipeline(vk::PipelineBindPoint::eGraphics, m_pipeline);
	vk::RenderingAttachmentInfo attachment {};
	attachment.imageView   = target_view;
	attachment.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
	attachment.loadOp      = vk::AttachmentLoadOp::eDontCare;
	attachment.storeOp     = vk::AttachmentStoreOp::eStore;
	vk::RenderingInfo rendering {};
	rendering.renderArea.extent    = target;
	rendering.layerCount           = 1;
	rendering.colorAttachmentCount = 1;
	rendering.pColorAttachments    = &attachment;
	command.beginRendering(&rendering);
	const vk::Viewport viewport {
	    0.0f, 0.0f, static_cast<float>(target.width), static_cast<float>(target.height), 0.0f, 1.0f};
	const vk::Rect2D scissor {{0, 0}, target};
	command.setViewport(0, 1, &viewport);
	command.setScissor(0, 1, &scissor);
	command.draw(3, 1, 0, 0);
	command.endRendering();
}

void PresentFilter::Release(vk::Device device) {
	device.destroyPipeline(m_pipeline, nullptr);
	device.destroyPipelineLayout(m_layout, nullptr);
	device.destroyDescriptorSetLayout(m_descriptors, nullptr);
	device.destroySampler(m_sampler, nullptr);
	m_pipeline    = nullptr;
	m_layout      = nullptr;
	m_descriptors = nullptr;
	m_sampler     = nullptr;
	m_format      = vk::Format::eUndefined;
}

// The game frame's rectangle in the swapchain image: the window's whole extent with --stretch,
// else the largest one with the frame's aspect ratio, centred (upstream 69fdb03dc; the bars are
// cleared to black).
static vk::Rect2D FitAspect(const vk::Extent3D& source, const vk::Extent2D& target) {
	const uint64_t src_w = std::max(source.width, 1u);
	const uint64_t src_h = std::max(source.height, 1u);
	uint32_t       w     = target.width;
	uint32_t       h     = target.height;
	if (src_w * h > src_h * w) {
		h = std::max<uint32_t>(1, static_cast<uint32_t>(src_h * w / src_w));
	} else {
		w = std::max<uint32_t>(1, static_cast<uint32_t>(src_w * h / src_h));
	}
	return {{static_cast<int32_t>((target.width - w) / 2),
	         static_cast<int32_t>((target.height - h) / 2)},
	        {w, h}};
}

void Swapchain::DrawLayer(vk::CommandBuffer command, const Presenter::Layer& layer,
                          const vk::Rect2D& rect) {
	auto device = m_window.graphic_ctx.device;
	if (m_overlay_pipeline == nullptr) {
		vk::SamplerCreateInfo sampler {};
		sampler.magFilter    = vk::Filter::eLinear;
		sampler.minFilter    = vk::Filter::eLinear;
		sampler.addressModeU = vk::SamplerAddressMode::eClampToEdge;
		sampler.addressModeV = vk::SamplerAddressMode::eClampToEdge;
		sampler.addressModeW = vk::SamplerAddressMode::eClampToEdge;
		RequireVulkanSuccess(device.createSampler(&sampler, nullptr, &m_overlay_sampler),
		                     "create video-out overlay sampler");
		const vk::DescriptorSetLayoutBinding binding {0, vk::DescriptorType::eCombinedImageSampler,
		                                              1, vk::ShaderStageFlagBits::eFragment};
		vk::DescriptorSetLayoutCreateInfo    descriptors {};
		descriptors.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
		descriptors.bindingCount = 1;
		descriptors.pBindings    = &binding;
		RequireVulkanSuccess(
		    device.createDescriptorSetLayout(&descriptors, nullptr, &m_overlay_descriptors),
		    "create video-out overlay descriptor layout");
		const vk::PushConstantRange alpha {vk::ShaderStageFlagBits::eFragment, 0, sizeof(uint32_t)};
		vk::PipelineLayoutCreateInfo layout {};
		layout.setLayoutCount         = 1;
		layout.pSetLayouts            = &m_overlay_descriptors;
		layout.pushConstantRangeCount = 1;
		layout.pPushConstantRanges    = &alpha;
		RequireVulkanSuccess(device.createPipelineLayout(&layout, nullptr, &m_overlay_layout),
		                     "create video-out overlay pipeline layout");
		const auto vertex   = CompileSPV(GPU_BLIT_FS_TRIANGLE_SPV, device);
		const auto fragment = CompileSPV(GPU_VIDEO_OUT_OVERLAY_SPV, device);
		std::array<vk::PipelineShaderStageCreateInfo, 2> stages {};
		stages[0].stage  = vk::ShaderStageFlagBits::eVertex;
		stages[0].module = vertex;
		stages[0].pName  = "main";
		stages[1].stage  = vk::ShaderStageFlagBits::eFragment;
		stages[1].module = fragment;
		stages[1].pName  = "main";
		vk::PipelineVertexInputStateCreateInfo   vertex_input {};
		vk::PipelineInputAssemblyStateCreateInfo assembly {};
		assembly.topology = vk::PrimitiveTopology::eTriangleList;
		vk::PipelineViewportStateCreateInfo viewport {};
		viewport.viewportCount = 1;
		viewport.scissorCount  = 1;
		vk::PipelineRasterizationStateCreateInfo rasterization {};
		rasterization.lineWidth = 1.0f;
		vk::PipelineMultisampleStateCreateInfo multisample {};
		multisample.rasterizationSamples = vk::SampleCountFlagBits::e1;
		vk::PipelineColorBlendAttachmentState attachment {};
		attachment.blendEnable         = VK_TRUE;
		attachment.srcColorBlendFactor = vk::BlendFactor::eOne;
		attachment.dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha;
		attachment.srcAlphaBlendFactor = vk::BlendFactor::eOne;
		attachment.dstAlphaBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha;
		attachment.colorWriteMask = vk::ColorComponentFlagBits::eR |
		                            vk::ColorComponentFlagBits::eG |
		                            vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
		vk::PipelineColorBlendStateCreateInfo blend {};
		blend.attachmentCount = 1;
		blend.pAttachments    = &attachment;
		const std::array dynamic_states {vk::DynamicState::eViewport, vk::DynamicState::eScissor};
		vk::PipelineDynamicStateCreateInfo dynamic {};
		dynamic.dynamicStateCount = static_cast<uint32_t>(dynamic_states.size());
		dynamic.pDynamicStates    = dynamic_states.data();
		vk::PipelineRenderingCreateInfo rendering {};
		rendering.colorAttachmentCount    = 1;
		rendering.pColorAttachmentFormats = &m_format;
		vk::GraphicsPipelineCreateInfo create {};
		create.pNext               = &rendering;
		create.stageCount          = static_cast<uint32_t>(stages.size());
		create.pStages             = stages.data();
		create.pVertexInputState   = &vertex_input;
		create.pInputAssemblyState = &assembly;
		create.pViewportState      = &viewport;
		create.pRasterizationState = &rasterization;
		create.pMultisampleState   = &multisample;
		create.pColorBlendState    = &blend;
		create.pDynamicState       = &dynamic;
		create.layout              = m_overlay_layout;
		RequireVulkanSuccess(
		    device.createGraphicsPipelines(nullptr, 1, &create, nullptr, &m_overlay_pipeline),
		    "create video-out overlay pipeline");
		device.destroyShaderModule(fragment, nullptr);
		device.destroyShaderModule(vertex, nullptr);
	}
	auto& frame = *layer.frame;
	if (frame.view == nullptr) {
		vk::ImageViewCreateInfo view {};
		view.image            = frame.image.image;
		view.viewType         = vk::ImageViewType::e2D;
		view.format           = frame.image.format;
		view.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
		RequireVulkanSuccess(device.createImageView(&view, nullptr, &frame.view),
		                     "create video-out overlay image view");
	}
	const vk::DescriptorImageInfo image {m_overlay_sampler, frame.view,
	                                     vk::ImageLayout::eShaderReadOnlyOptimal};
	vk::WriteDescriptorSet        write {};
	write.dstBinding      = 0;
	write.descriptorCount = 1;
	write.descriptorType  = vk::DescriptorType::eCombinedImageSampler;
	write.pImageInfo      = &image;
	command.pushDescriptorSetKHR(vk::PipelineBindPoint::eGraphics, m_overlay_layout, 0, 1, &write);
	// The main bus ignores alpha.
	const uint32_t flags = (layer.premultiplied_alpha ? 1u : 0u) | (m_hdr_active ? 2u : 0u) |
	                       (frame.hdr ? 4u : 0u) |
	                       (frame.image.format == vk::Format::eR16G16B16A16Sfloat ? 8u : 0u) |
	                       (layer.bus == 0 ? 16u : 0u);
	command.pushConstants(m_overlay_layout, vk::ShaderStageFlagBits::eFragment, 0,
	                      sizeof(flags), &flags);
	command.bindPipeline(vk::PipelineBindPoint::eGraphics, m_overlay_pipeline);
	vk::RenderingAttachmentInfo attachment {};
	attachment.imageView   = m_image_views[m_image_index];
	attachment.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
	attachment.loadOp      = vk::AttachmentLoadOp::eLoad;
	attachment.storeOp     = vk::AttachmentStoreOp::eStore;
	vk::RenderingInfo rendering {};
	rendering.renderArea.extent    = m_extent;
	rendering.layerCount           = 1;
	rendering.colorAttachmentCount = 1;
	rendering.pColorAttachments    = &attachment;
	command.beginRendering(&rendering);
	const vk::Viewport viewport {static_cast<float>(rect.offset.x),
	                             static_cast<float>(rect.offset.y),
	                             static_cast<float>(rect.extent.width),
	                             static_cast<float>(rect.extent.height),
	                             0.0f,
	                             1.0f};
	command.setViewport(0, 1, &viewport);
	command.setScissor(0, 1, &rect);
	command.draw(3, 1, 0, 0);
	command.endRendering();
}

void Swapchain::RecordPresentCommands(CommandBuffer& command, Presenter::Frame* source,
                                      const Presenter::Layer& overlay, bool draw_system_overlay) {
	bool       dump_checked    = false;
	auto       vk_command      = command.Handle();
	const bool draw_overlay    = overlay.frame != nullptr;
	const bool draw_attachment = draw_overlay || draw_system_overlay;
	// KYTY_PRESENT_BOX_DOWNSCALE: shrinking by less than 2x draws with PresentFilter instead of
	// the blit (windowInternal.h).
	const vk::Extent2D source_extent =
	    source != nullptr ? vk::Extent2D {source->image.extent.width, source->image.extent.height}
	                      : vk::Extent2D {};
	// An HDR10 frame on an SDR window is converted by the filter pass (KYTY_HDR_TO_SDR).
	const bool pq_to_sdr = source != nullptr && source->hdr && !m_hdr_active && PqToSdrEnabled();
	const bool filtered =
	    source != nullptr && (pq_to_sdr || (PresentBoxDownscaleEnabled() &&
	                                        PresentNeedsFilter(source_extent, m_extent)));
	if (filtered) {
		if (source->view == nullptr) {
			vk::ImageViewCreateInfo view {};
			view.image            = source->image.image;
			view.viewType         = vk::ImageViewType::e2D;
			view.format           = source->image.format;
			view.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
			RequireVulkanSuccess(
			    m_window.graphic_ctx.device.createImageView(&view, nullptr, &source->view),
			    "create presentation source view");
		}
		source->Transit(vk_command, vk::ImageLayout::eShaderReadOnlyOptimal,
		                vk::AccessFlagBits2::eShaderRead);
	} else if (source != nullptr) {
		source->Transit(vk_command, vk::ImageLayout::eTransferSrcOptimal,
		                vk::AccessFlagBits2::eTransferRead);
	}
	if (draw_overlay) {
		overlay.frame->Transit(vk_command, vk::ImageLayout::eShaderReadOnlyOptimal,
		                       vk::AccessFlagBits2::eShaderRead);
	}

	vk::ImageMemoryBarrier to_transfer {};
	to_transfer.sType = vk::StructureType::eImageMemoryBarrier;
	to_transfer.dstAccessMask =
	    filtered ? vk::AccessFlagBits::eColorAttachmentWrite : vk::AccessFlagBits::eTransferWrite;
	to_transfer.oldLayout = vk::ImageLayout::eUndefined;
	to_transfer.newLayout =
	    filtered ? vk::ImageLayout::eColorAttachmentOptimal : vk::ImageLayout::eTransferDstOptimal;
	to_transfer.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
	to_transfer.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
	to_transfer.image                           = m_images[m_image_index];
	to_transfer.subresourceRange.aspectMask     = vk::ImageAspectFlagBits::eColor;
	to_transfer.subresourceRange.baseMipLevel   = 0;
	to_transfer.subresourceRange.levelCount     = 1;
	to_transfer.subresourceRange.baseArrayLayer = 0;
	to_transfer.subresourceRange.layerCount     = 1;
	// Match the acquire wait stage so the layout transition cannot precede acquisition.
	const auto write_stage = filtered ? vk::PipelineStageFlagBits::eColorAttachmentOutput
	                                  : vk::PipelineStageFlagBits::eTransfer;
	vk_command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, write_stage,
	                           vk::DependencyFlags {}, 0, nullptr, 0, nullptr, 1, &to_transfer);

	// The filter pass (KYTY_HDR_TO_SDR, KYTY_PRESENT_BOX_DOWNSCALE) always fills the whole image.
	const auto* aspect_source = source != nullptr ? source : overlay.frame;
	const auto  game_rect     = !filtered && !Config::StretchEnabled() && aspect_source != nullptr
	                                ? FitAspect(aspect_source->image.extent, m_extent)
	                                : vk::Rect2D {{0, 0}, m_extent};
	if (!filtered && (source == nullptr || game_rect.extent != m_extent)) {
		const vk::ClearColorValue black {};
		vk_command.clearColorImage(m_images[m_image_index], vk::ImageLayout::eTransferDstOptimal,
		                           &black, 1, &to_transfer.subresourceRange);
		if (source != nullptr) {
			vk::ImageMemoryBarrier after_clear = to_transfer;
			after_clear.srcAccessMask          = vk::AccessFlagBits::eTransferWrite;
			after_clear.oldLayout              = vk::ImageLayout::eTransferDstOptimal;
			vk_command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
			                           vk::PipelineStageFlagBits::eTransfer, vk::DependencyFlags {},
			                           0, nullptr, 0, nullptr, 1, &after_clear);
		}
	}
	if (filtered) {
		m_filter.Record(m_window.graphic_ctx.device, vk_command, source->view, 0, source_extent,
		                m_image_views[m_image_index], m_format, m_extent, pq_to_sdr);
		if (!source->blank && LumaProbe::Instance().Wanted()) {
			// The filtered path read the frame as a texture: back to transfer-source for the probe
			// blit.
			source->Transit(vk_command, vk::ImageLayout::eTransferSrcOptimal,
			                vk::AccessFlagBits2::eTransferRead);
			LumaProbe::Instance().Record(m_window.graphic_ctx, vk_command, source->image.image,
			                             source_extent);
		}
	} else if (source != nullptr) {
		vk::ImageBlit region {};
		region.srcSubresource.aspectMask     = vk::ImageAspectFlagBits::eColor;
		region.srcSubresource.mipLevel       = 0;
		region.srcSubresource.baseArrayLayer = 0;
		region.srcSubresource.layerCount     = 1;
		region.srcOffsets[1].x               = static_cast<int>(source->image.extent.width);
		region.srcOffsets[1].y               = static_cast<int>(source->image.extent.height);
		region.srcOffsets[1].z               = 1;
		region.dstSubresource.aspectMask     = vk::ImageAspectFlagBits::eColor;
		region.dstSubresource.mipLevel       = 0;
		region.dstSubresource.baseArrayLayer = 0;
		region.dstSubresource.layerCount     = 1;
		region.dstOffsets[0].x               = game_rect.offset.x;
		region.dstOffsets[0].y               = game_rect.offset.y;
		region.dstOffsets[1].x = game_rect.offset.x + static_cast<int>(game_rect.extent.width);
		region.dstOffsets[1].y = game_rect.offset.y + static_cast<int>(game_rect.extent.height);
		region.dstOffsets[1].z               = 1;
		vk_command.blitImage(source->image.image, vk::ImageLayout::eTransferSrcOptimal,
		                     m_images[m_image_index], vk::ImageLayout::eTransferDstOptimal, 1,
		                     &region, vk::Filter::eLinear);
		if (!source->blank && LumaProbe::Instance().Wanted()) {
			LumaProbe::Instance().Record(m_window.graphic_ctx, vk_command, source->image.image,
			                             source_extent);
		}
	}

	vk::ImageMemoryBarrier to_present {};
	to_present.sType               = vk::StructureType::eImageMemoryBarrier;
	to_present.srcAccessMask =
	    filtered ? vk::AccessFlagBits::eColorAttachmentWrite : vk::AccessFlagBits::eTransferWrite;
	to_present.dstAccessMask       = draw_attachment ? vk::AccessFlagBits::eColorAttachmentRead |
	                                                       vk::AccessFlagBits::eColorAttachmentWrite
	                                                 : vk::AccessFlagBits::eMemoryRead;
	to_present.oldLayout           = to_transfer.newLayout;
	to_present.newLayout           = draw_attachment ? vk::ImageLayout::eColorAttachmentOptimal
	                                                 : vk::ImageLayout::ePresentSrcKHR;
	to_present.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	to_present.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	to_present.image               = m_images[m_image_index];
	to_present.subresourceRange.aspectMask     = vk::ImageAspectFlagBits::eColor;
	to_present.subresourceRange.baseMipLevel   = 0;
	to_present.subresourceRange.levelCount     = 1;
	to_present.subresourceRange.baseArrayLayer = 0;
	to_present.subresourceRange.layerCount     = 1;
	vk_command.pipelineBarrier(write_stage,
	                           draw_attachment ? vk::PipelineStageFlagBits::eColorAttachmentOutput
	                                           : vk::PipelineStageFlagBits::eAllCommands,
	                           vk::DependencyFlagBits::eByRegion, 0, nullptr, 0, nullptr, 1,
	                           &to_present);
	if (draw_attachment) {
		// Each pass blends over the color writes of the previous one.
		to_present.srcAccessMask      = vk::AccessFlagBits::eColorAttachmentWrite;
		to_present.oldLayout          = vk::ImageLayout::eColorAttachmentOptimal;
		const auto attachment_barrier = [&] {
			vk_command.pipelineBarrier(vk::PipelineStageFlagBits::eColorAttachmentOutput,
			                           vk::PipelineStageFlagBits::eColorAttachmentOutput,
			                           vk::DependencyFlagBits::eByRegion, 0, nullptr, 0, nullptr, 1,
			                           &to_present);
		};
		// Upstream 026189166 also draws an HDR10 main frame through DrawLayer here; ours converts
		// it in the filter pass above (KYTY_HDR_TO_SDR).
		if (draw_overlay) {
			DrawLayer(vk_command, overlay, game_rect);
			if (draw_system_overlay) {
				attachment_barrier();
			}
		}
		// load card: dumping never changes the screen. A dumped present that carries the progress
		// card/badge copies the frame BEFORE the overlay is drawn (BMP without the card;
		// KYTY_PROGRESS_OVERLAY_IN_DUMPS=1 dumps it after).
		if (draw_system_overlay && !ProgressOverlay::InDumps() && SystemOverlayProgressWanted()) {
			if (auto& dump = FrameDump::Instance(); dump.Enabled()) {
				dump_checked = true;
				if (dump.Next()) {
					dump.Record(m_window.graphic_ctx, vk_command, m_images[m_image_index], m_format,
					            m_extent, true);
				}
			}
		}
		if (draw_system_overlay) {
			m_system_overlay->Record(vk_command, m_image_views[m_image_index]);
		}
		to_present.dstAccessMask = vk::AccessFlagBits::eMemoryRead;
		to_present.newLayout     = vk::ImageLayout::ePresentSrcKHR;
		vk_command.pipelineBarrier(vk::PipelineStageFlagBits::eColorAttachmentOutput,
		                           vk::PipelineStageFlagBits::eAllCommands,
		                           vk::DependencyFlagBits::eByRegion, 0, nullptr, 0, nullptr, 1,
		                           &to_present);
	}
	if (auto& dump = FrameDump::Instance(); dump.Enabled() && !dump_checked && dump.Next()) {
		dump.Record(m_window.graphic_ctx, vk_command, m_images[m_image_index], m_format, m_extent);
	}
}

uint64_t Swapchain::Submit(CommandScheduler& scheduler) {
	SubmitInfo submit;
	submit.AddWait(m_image_acquired[m_frame_index], 1, vk::PipelineStageFlagBits::eTransfer);
	submit.AddSignal(m_render_complete[m_image_index]);
	const auto tick = scheduler.Submit(submit);
	if (m_frame_index < m_frame_ticks.size()) {
		m_frame_ticks[m_frame_index] = tick;
	}
	return tick;
}

Swapchain::Status Swapchain::Present() {
	if (m_minimized || m_handle == nullptr) {
		return Status::Minimized;
	}
	const auto         ready = m_render_complete[m_image_index];
	vk::PresentInfoKHR present {};
	present.sType              = vk::StructureType::ePresentInfoKHR;
	present.swapchainCount     = 1;
	present.pSwapchains        = &m_handle;
	present.pImageIndices      = &m_image_index;
	present.pWaitSemaphores    = &ready;
	present.waitSemaphoreCount = 1;

	vk::Result result;
	// ASTROHYBRID_LOW_LATENCY (lowLatency.h): the presented flip's frame number as present ID.
	const auto       latency_id = LowLatency::BeginPresent(static_cast<VkSwapchainKHR>(m_handle));
	vk::PresentIdKHR present_id {};
	if (latency_id != 0) {
		present_id.swapchainCount = 1;
		present_id.pPresentIds    = &latency_id;
		present.pNext             = &present_id;
	}
	{
		// The blit this present waits for goes ahead of game batches still waiting for their
		// texture copies: the present never waits for them, nor for a signal not yet submitted.
		// Hang reports (issue #62, Linux): a present blocked in the driver holds the graphics queue
		// lock, so these scopes tell a stuck vkQueuePresentKHR from a stuck GPU.
		const auto queue =
		    reinterpret_cast<uint64_t>(static_cast<VkQueue>(m_window.graphic_ctx.queue));
		HangWatchdog::Scope lock_scope("present-queue-lock", queue, m_image_index);
		Common::LockGuard   lock(m_window.graphic_ctx.queue_mutex);
		m_window.graphic_ctx.submission_queue.DrainReadyForPresentLocked();
		HangWatchdog::Scope present_scope("vkQueuePresentKHR", queue, m_image_index);
		result = m_window.graphic_ctx.queue.presentKHR(&present);
	}
	FlightRec::NotePresent();
	LowLatency::EndPresent(static_cast<VkSwapchainKHR>(m_handle), latency_id);
	Common::LatTrace::Log("present");
	Common::PresentInterval::Note();
	switch (result) {
		case vk::Result::eSuccess: break;
		case vk::Result::eSuboptimalKHR:
			LOGF("vkQueuePresentKHR returned vk::Result::eSuboptimalKHR\n");
			return Status::Recreate;
		case vk::Result::eErrorOutOfDateKHR:
			LOGF("vkQueuePresentKHR returned vk::Result::eErrorOutOfDateKHR\n");
			return Status::Recreate;
		case vk::Result::eErrorSurfaceLostKHR:
			LOGF("vkQueuePresentKHR returned vk::Result::eErrorSurfaceLostKHR\n");
			return Status::SurfaceLost;
		default: EXIT("vkQueuePresentKHR failed: %s\n", vk::to_string(result).c_str());
	}
	m_frame_index = (m_frame_index + 1u) % static_cast<uint32_t>(m_images.size());
	if (m_suboptimal) {
		m_suboptimal = false;
		return Status::Recreate;
	}
	return Status::Success;
}

Presenter::Presenter(WindowContext& window): m_impl(std::make_unique<Impl>(window)) {}

// KYTY_PRESENT_UNLOCKED (default on; =0 keeps the renderer's lock; chenxiao07/KytyPS5 a30144267): the
// present thread records and submits the swapchain blit without the renderer's lock. What it uses is
// the presenter's own: present_scheduler (its pool, timeline and command buffers; only the present
// thread records on it), the presenter's frames (the flip queue hands a frame over only once the GPU
// completed its copy, FlipQueue::Complete) and the swapchain images, and the graphics queue, which
// has its own lock (queue_mutex, also taken by the RenderDoc capture boundaries). Under the
// renderer's lock the GPU thread waited for that whole recording and submission once a frame. (This
// tree has no frame generation, which shares the renderer's state and keeps the lock in theirs.)
static bool PresentUnlocked() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_PRESENT_UNLOCKED");
		return value == nullptr || std::strcmp(value, "0") != 0;
	}();
	return enabled;
}

Presenter::~Presenter() = default;

// KYTY_FRAME_DUMP=<frames>[:<x>,<y>,<w>,<h>] (diagnostic, default off): each time the file
// KYTY_FRAME_DUMP_TRIGGER (default _framedump.go in the working directory) appears, it is removed
// and the next <frames> prepared video-out frames (every prepared frame, also ones that are not
// shown; optional crop) are read back and appended to _framedump.bin ({u32 'KFDM', u32 dump
// ordinal, i32 guest frame, u32 width, u32 height, u32 vk format, u32 bytes per pixel, u64
// steady-clock ns at recording, pixels} per frame), to tell one-frame jumps from motion.
static void GuestFrameDump(RenderContext& renderer, CommandBuffer& command_buffer,
                           Presenter::Frame& frame) {
	struct Request {
		uint32_t    frames = 0;
		uint32_t    x = 0, y = 0, w = 0, h = 0;
		std::string trigger;
	};
	static const Request request = [] {
		Request parsed;
		if (const auto* value = std::getenv("KYTY_FRAME_DUMP"); value != nullptr) {
			char* end     = nullptr;
			parsed.frames = static_cast<uint32_t>(std::strtoul(value, &end, 10));
			if (end != nullptr && *end == ':') {
				std::array<uint32_t, 4> rect {};
				for (auto& v: rect) {
					v = static_cast<uint32_t>(std::strtoul(end + 1, &end, 10));
					if (end == nullptr || *end != ',') break;
				}
				parsed.x = rect[0];
				parsed.y = rect[1];
				parsed.w = rect[2];
				parsed.h = rect[3];
			}
		}
		const auto* trigger = std::getenv("KYTY_FRAME_DUMP_TRIGGER");
		parsed.trigger      = trigger != nullptr && trigger[0] != '\0' ? trigger : "_framedump.go";
		return parsed;
	}();
	if (request.frames == 0) [[likely]] {
		return;
	}
	static uint32_t remaining = 0;
	static uint32_t ordinal   = 0;
	if (remaining == 0) {
		std::error_code error;
		if (!std::filesystem::exists(request.trigger, error)) {
			return;
		}
		std::filesystem::remove(request.trigger, error);
		remaining = request.frames;
	}
	remaining--;
	const auto         format = frame.image.format;
	const uint32_t     bpp    = format == vk::Format::eR16G16B16A16Sfloat ? 8u : 4u;
	const auto         ex     = frame.image.extent;
	const uint32_t     x      = std::min(request.x, ex.width - 1);
	const uint32_t     y      = std::min(request.y, ex.height - 1);
	const uint32_t     w      = request.w == 0 ? ex.width - x : std::min(request.w, ex.width - x);
	const uint32_t     h      = request.h == 0 ? ex.height - y : std::min(request.h, ex.height - y);
	const uint64_t     bytes  = uint64_t {w} * h * bpp;
	constexpr uint32_t Slots  = 16;
	static std::unique_ptr<Buffer> download;
	static uint64_t                slot_bytes = 0;
	if (download == nullptr || bytes > slot_bytes) {
		auto& scheduler = renderer.GetCommandScheduler();
		if (download != nullptr) {
			scheduler.Wait(scheduler.CurrentTick() - 1);
			scheduler.PopPendingOperations();
		}
		slot_bytes = bytes;
		download = std::make_unique<Buffer>(renderer.GetGraphics(), scheduler,
		                                    MemoryUsage::Download, 0, AllFlags, slot_bytes * Slots);
	}
	const auto slot    = ordinal % Slots;
	const auto number  = ordinal++;
	auto       command = command_buffer.Handle();
	frame.Transit(command, vk::ImageLayout::eTransferSrcOptimal,
	              vk::AccessFlagBits2::eTransferRead);
	vk::BufferImageCopy copy {};
	copy.bufferOffset     = slot * slot_bytes;
	copy.imageSubresource = vk::ImageSubresourceLayers {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
	copy.imageOffset      = vk::Offset3D {static_cast<int32_t>(x), static_cast<int32_t>(y), 0};
	copy.imageExtent      = vk::Extent3D {w, h, 1};
	command.copyImageToBuffer(frame.image.image, vk::ImageLayout::eTransferSrcOptimal,
	                          download->Handle(), copy);
	vk::BufferMemoryBarrier barrier {};
	barrier.srcAccessMask       = vk::AccessFlagBits::eTransferWrite;
	barrier.dstAccessMask       = vk::AccessFlagBits::eHostRead;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.buffer              = download->Handle();
	barrier.offset              = copy.bufferOffset;
	barrier.size                = bytes;
	command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost,
	                        {}, {}, barrier, {});
	const auto guest_frame = renderer.GetGpu().GetFrameNum();
	const auto t_ns = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                            std::chrono::steady_clock::now().time_since_epoch())
	                                            .count());
	const auto vk_format = static_cast<uint32_t>(format);
	renderer.GetCommandScheduler().DeferOperation([slot, number, guest_frame, w, h, vk_format, bpp,
	                                               t_ns, bytes] {
		static std::FILE* file = std::fopen("_framedump.bin", "wb");
		if (file == nullptr) return;
		download->Invalidate(slot * slot_bytes, bytes);
		const uint32_t header[7] = {0x4d44464bu, number, static_cast<uint32_t>(guest_frame), w, h,
		                            vk_format,   bpp};
		std::fwrite(header, sizeof(header), 1, file);
		std::fwrite(&t_ns, sizeof(t_ns), 1, file);
		std::fwrite(download->Mapped().data() + slot * slot_bytes, 1, bytes, file);
		std::fflush(file);
	});
}

Presenter::Frame& Presenter::PrepareFrame(CommandBuffer& buffer, const ImageInfo& info, bool hdr,
                                          bool headset) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(buffer.IsInvalid());
	auto frame_format = info.pixel_format;
	switch (frame_format) {
		case vk::Format::eR8G8B8A8Srgb: frame_format = vk::Format::eR8G8B8A8Unorm; break;
		case vk::Format::eB8G8R8A8Srgb: frame_format = vk::Format::eB8G8R8A8Unorm; break;
		default: break;
	}
	auto* frame = m_impl->frames.Acquire({info.extent.width, info.extent.height}, frame_format);
	frame->hdr   = hdr;
	frame->blank = false;
	RenderLockGuard   render_lock(m_impl->renderer.GetMutex());
	auto&             image = m_impl->ResolveSurface(info);
	if (image.backing.format == vk::Format::eUndefined) {
		EXIT("unsupported presentation source, image=%p\n", static_cast<const void*>(&image));
	}
	frame->Configure(m_impl->window.graphic_ctx,
	                 {image.backing.extent.width, image.backing.extent.height}, frame_format);
	frame->CopyFrom(buffer, image);
	GuestFrameDump(m_impl->renderer, buffer,
	               *frame); // upstream KYTY_FRAME_DUMP (renamed: our FrameDump class)
	TargetDump::Instance().OnFlip(m_impl->window.graphic_ctx, m_impl->renderer.GetTextureCache(),
	                              buffer);
	if (MetalFx::Requested()) [[unlikely]] {
		MetalFx::NoteFlip();
	}
	if (headset) {
		// The frame holds the raw display-encoded pixels the headset needs.
		frame->Transit(buffer.Handle(), vk::ImageLayout::eTransferSrcOptimal,
		               vk::AccessFlagBits2::eTransferRead);
		Vr::OpenXrQueueEyeSource(buffer, frame->image);
	}
	return *frame;
}

Presenter::Frame& Presenter::PrepareBlankFrame(uint32_t width, uint32_t height, bool opaque,
                                               CommandBuffer* producer) {
	KYTY_PROFILER_FUNCTION();
	auto              format = m_impl->frames.GetFormat();
	auto*             frame  = m_impl->frames.Acquire({width, height}, format);
	frame->hdr   = false;
	frame->blank = true;
	// Into the GPU thread's command buffer (its flip) under the renderer's lock; the present
	// thread's blank frame (no producer) uses only the presenter's scheduler and frame.
	// present_scheduler has one current command buffer and two recording threads: this (vblank)
	// thread and the load card thread (PresentLoadCard -> Impl::Present). Without present_mutex
	// a blank frame begun while the card's present was recording hit BeginCommand's
	// EXIT_IF(!m_command.IsInvalid()) at startup. Taken before the renderer's lock, the order
	// Presenter::Present -> Impl::Present uses; no caller of this function holds present_mutex.
	std::optional<Common::LockGuard> present_lock;
	if (producer == nullptr) {
		present_lock.emplace(m_impl->present_mutex);
	}
	std::optional<RenderLockGuard> render_lock;
	if (producer != nullptr || !PresentUnlocked()) {
		render_lock.emplace(m_impl->renderer.GetMutex());
	}
	frame->Configure(m_impl->window.graphic_ctx, {width, height}, format);
	vk::ClearColorValue clear {};
	clear.float32[3] = opaque ? 1.0f : 0.0f;
	if (producer != nullptr) {
		EXIT_IF(producer->IsInvalid());
		frame->Clear(*producer, clear);
	} else {
		auto& command = m_impl->present_scheduler.BeginCommand();
		frame->Clear(command, clear);
		frame->present_tick = m_impl->present_scheduler.Submit();
	}
	return *frame;
}

bool Presenter::PresentLastFrame() {
	Common::LockGuard lock(m_impl->present_mutex);
	if (m_impl->layers[0].frame == nullptr && m_impl->layers[1].frame == nullptr) {
		return false;
	}
	m_impl->Present();
	return true;
}

bool Presenter::IsHdrSupported() const noexcept {
	return m_impl->swapchain.IsHdrSupported();
}

bool Presenter::NeedsSystemOverlayRefresh() const noexcept {
	const auto visual = GetSystemOverlayVisualState();
	return visual.active ||
	       visual.revision != m_impl->presented_overlay_revision.load(std::memory_order_acquire);
}

RenderContext& Presenter::Renderer() const noexcept {
	return m_impl->renderer;
}

void Presenter::Present(Frame& frame) {
	const Layer layer {&frame, 0, false};
	Present(std::span(&layer, 1));
}

void Presenter::Present(std::span<const Layer> layers) {
	Common::LockGuard lock(m_impl->present_mutex);
	bool              hold = false;
	for (const auto& layer: layers) {
		// nps5 load card: a real game flip ends the load card before this frame is presented.
		if (layer.bus == 0 && layer.frame != nullptr && !layer.frame->blank) {
			const bool black = LoadCard::BlackNow();
			hold = layers.size() == 1 && LoadCard::HoldGameFrame(black, LoadCard::Active());
			LoadCard::NoteGameFlip(black);
		}
	}
	for (const auto& layer: layers) {
		EXIT_IF(layer.bus < 0 || layer.bus >= static_cast<int>(m_impl->layers.size()));
		auto& previous = m_impl->layers[layer.bus];
		if (layer.hold && previous.frame != nullptr && previous.frame != layer.frame) {
			// KYTY_HOLD_SKIPPED_DRAWS: the frame shown before stays; this one goes back to the
			// pool.
			m_impl->frames.Release(layer.frame);
			continue;
		}
		if (previous.frame != nullptr) {
			m_impl->frames.Release(previous.frame);
		}
		previous = layer;
	}
	if (!hold) {
		m_impl->Present(); // load card: a held black frame is repainted by the load card watchdog
	}
}

void Presenter::ClearLayer(int bus) {
	if (static_cast<size_t>(bus) >= m_impl->layers.size()) {
		return;
	}
	Common::LockGuard lock(m_impl->present_mutex);
	auto& layer = m_impl->layers[bus];
	if (layer.frame != nullptr) {
		m_impl->frames.Release(layer.frame);
		layer = {};
	}
}

void Presenter::Impl::Present() {
	KYTY_PROFILER_FUNCTION();
	using Common::DebugCounters::Counter;
	using Common::DebugCounters::ScopedTimer;
	ScopedTimer present_time(Counter::PresentNs);
	Nps5MemLog::Start();
	Nps5MemLog::NoteFrame();

	if (layers[0].frame != nullptr && !layers[0].frame->blank) {
		ProgressOverlay::GameFrameSeen().store(true, std::memory_order_relaxed);
	}
	const auto overlay_visual = GetSystemOverlayVisualState();
	// Keep black/blank flips in the current color space; only real main-bus frames switch it.
	const auto* main_frame = layers[0].frame;
	const bool hdr_changed = main_frame != nullptr && !main_frame->blank &&
	                         swapchain.SetHdr(main_frame->hdr);
	// Some window systems keep presenting an old swapchain after a resize.
	if (hdr_changed || swapchain.NeedsResize()) {
		RecoverSwapchain(Swapchain::Status::Recreate);
	}
	if (swapchain.IsMinimized()) {
		return;
	}
	for (uint32_t attempt = 0; attempt < 2; attempt++) {
		auto status = [&] {
			ScopedTimer acquire_time(Counter::PresentAcquireNs);
			return swapchain.AcquireNextImage(present_scheduler);
		}();
		if (status == Swapchain::Status::Minimized) {
			return;
		}
		if (status != Swapchain::Status::Success) {
			RecoverSwapchain(status);
			if (swapchain.IsMinimized()) {
				return;
			}
			continue;
		}
		{
			std::optional<RenderLockGuard> render_lock;
			if (!PresentUnlocked()) {
				render_lock.emplace(renderer.GetMutex());
			}
			LumaProbe::Instance().Collect(present_scheduler);
			auto& command = present_scheduler.BeginCommand();
			if (attempt != 0) {
			}
			const bool draw_system_overlay = [&] {
				ScopedTimer overlay_time(Counter::OverlayNs);
				return (overlay_visual.active || PerformanceOverlayEnabled()) &&
				       swapchain.PrepareSystemOverlay();
			}();
			swapchain.RecordPresentCommands(command, layers[0].frame, layers[1],
			                                draw_system_overlay);
			const auto tick = swapchain.Submit(present_scheduler);
			LumaProbe::Instance().Submitted(tick);
			if (auto& dump = FrameDump::Instance(); dump.TakeRecorded()) {
				present_scheduler.Wait(tick);
				dump.Write();
			}
			for (const auto& layer: layers) {
				if (layer.frame != nullptr) {
					layer.frame->present_tick = tick;
				}
			}
		}
		status = swapchain.Present();
		if (status == Swapchain::Status::Minimized) {
			return;
		}
		if (status != Swapchain::Status::Success) {
			RecoverSwapchain(status);
			if (swapchain.IsMinimized()) {
				return;
			}
			continue;
		}

		presented_overlay_revision.store(overlay_visual.revision, std::memory_order_release);
		window.loop.presented_frames.fetch_add(1, std::memory_order_relaxed);
		return;
	}
	LOGF("Vulkan presentation retry exhausted; dropping frame\n");
	Common::DebugCounters::Add(Common::DebugCounters::Counter::DroppedPresents);
}

bool Presenter::Impl::PresentLoadCard(uint64_t seq, bool card) {
	Common::LockGuard lock(present_mutex);
	auto&             shared = LoadCard::State();
	if (shared.flip_seq.load(std::memory_order_acquire) != seq) {
		return false; // the game flipped meanwhile: nothing of ours is inserted
	}
	// Before the very first game frame there is nothing to repaint: the card is drawn over black.
	if (layers[0].frame == nullptr && layers[1].frame == nullptr &&
	    !(card && LoadCard::AnyReason())) {
		return false;
	}
	if (card) {
		shared.active.store(true, std::memory_order_release);
	} else if (shared.active.exchange(false, std::memory_order_acq_rel)) {
		shared.cut_gen.fetch_add(1, std::memory_order_relaxed);
	}
	Present();
	return true;
}

// nps5 load card: polls the flip sequence and the compile counters (LoadCard::Watchdog decides) and
// presents the load card from this thread. Never touches the command processor or the compile
// workers: it only reads relaxed counters and takes present_mutex for its own present.
void Presenter::Impl::LoadCardLoop(std::stop_token stop) {
	using Clock = std::chrono::steady_clock;
	LoadCard::Watchdog    wd(LoadCard::StartMs(), LoadCard::PresentIntervalMs());
	auto&                 shared   = LoadCard::State();
	uint64_t              last_seq = ~uint64_t {0};
	uint64_t              shown    = 0;
	LoadCard::VfsActivity vfs;
	while (!stop.stop_requested()) {
		std::this_thread::sleep_for(std::chrono::milliseconds(20));
		const auto now = static_cast<uint64_t>(
		    std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now().time_since_epoch())
		        .count());
		const auto seq  = shared.flip_seq.load(std::memory_order_acquire);
		const auto snap = CompileProgress::Read();
		if (seq != last_seq) {
			last_seq      = seq;
			const auto& c = CompileProgress::Get();
			shared.base_started.store(c.live_started.load(std::memory_order_relaxed),
			                          std::memory_order_relaxed);
			shared.base_finished.store(c.live_finished.load(std::memory_order_relaxed),
			                           std::memory_order_relaxed);
		}
		// load card: the full-screen card belongs to the session start (before the first content
		// frame). Afterwards only a genuine long stall (>= GapMs without content, compiles pending
		// or VFS bytes still moving) brings it back.
		const bool content_seen = shared.content_seen.load(std::memory_order_acquire);
		const bool video        = LoadCard::VideoPlaying();
		const bool content =
		    seq != 0 &&
		    LoadCard::EndsOpening(video, LoadCard::LaunchPhasePending(),
		                          LoadCard::BlackEnabled() &&
		                              shared.probe_state.load(std::memory_order_relaxed) != 2);
		const auto pol = LoadCard::MakePolicy(content_seen, content, seq,
		                                      false /* upstream dropped the window pause state */,
		                                      LoadCard::AnyReason(), LoadCard::MidgameEnabled(),
		                                      LoadCard::StartMs(), LoadCard::GapMs(), video);
		wd.SetStartMs(pol.start_ms);
		bool work = LoadCard::Pending(snap);
		if (content_seen) {
			static const auto* vfs_bytes = reinterpret_cast<const std::atomic<unsigned long long>*>(
			    dlsym(RTLD_DEFAULT, "nps5_vfs_bytes_read"));
			if (vfs_bytes != nullptr)
				work = vfs.Update(now, vfs_bytes->load(std::memory_order_relaxed)) || work;
		}
		const auto d = wd.Step(now, seq, work, pol.armed, pol.any);
		shared.last_flip_ms.store(wd.LastFlipMs(), std::memory_order_relaxed);
		if (d.started && LoadCard::LogEnabled()) {
			std::fprintf(stderr,
			             "[loadcard] %llu ms: start (no game flip for %llu ms, seq %llu, live %d "
			             "waiting %d replay %llu/%llu)\n",
			             static_cast<unsigned long long>(now),
			             static_cast<unsigned long long>(now - wd.LastFlipMs()),
			             static_cast<unsigned long long>(seq), snap.live, snap.waiting,
			             static_cast<unsigned long long>(snap.replay_done),
			             static_cast<unsigned long long>(snap.replay_total));
		}
		if (d.present_now && !video && PresentLoadCard(seq, true)) {
			shown++;
		}
		if (d.stopped && LoadCard::LogEnabled()) {
			std::fprintf(stderr, "[loadcard] %llu ms: stop (%s) after %llu card presents\n",
			             static_cast<unsigned long long>(now),
			             d.repaint_clean ? "work ended" : "game flipped",
			             static_cast<unsigned long long>(shown));
			shown = 0;
		}
		if (d.repaint_clean) {
			(void)PresentLoadCard(seq, false);
		}
	}
}

void Presenter::Discard(Frame& frame) {
	m_impl->frames.Release(&frame);
}

WindowContext::WindowContext() = default;

} // namespace Libs::Graphics
