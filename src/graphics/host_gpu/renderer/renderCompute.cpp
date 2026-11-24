#include "common/assert.h"
#include "common/common.h"
#include "common/debugCounters.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/liveSwitch.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "graphics/guest_gpu/command_processor/naniteShadeStats.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/pm4.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/host_gpu/renderer/computeImageCopy.h"
#include "graphics/host_gpu/renderer/drawPrep/cpGaps.h"
#include "graphics/host_gpu/renderer/drawPrep/drawPrep.h"
#include "graphics/host_gpu/renderer/flightRecDraw.h"
#include "graphics/host_gpu/renderer/gpuOpProfiler.h"
#include "graphics/host_gpu/renderer/image/imageInfo.h"
#include "graphics/host_gpu/renderer/image/tilerTrace.h"
#include "graphics/host_gpu/renderer/inputProfileDraw.h"
#include "graphics/host_gpu/renderer/metalfx.h"
#include "graphics/host_gpu/renderer/nanProbe.h"
#include "graphics/host_gpu/renderer/passTrace.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/pipeline/rtSession.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/CodegenOptions.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/BindingLayout.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shader.h"
#include "kernel/eventQueue.h"
#include "kernel/memory.h"
#include "kernel/pthread.h"
#include "libs/errno.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <set>
#include <span>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#if defined(__APPLE__) || defined(__linux__)
#include <dlfcn.h>
#endif

namespace {
bool mip_view_samplers() {
	static const bool on = [] {
		const auto* v = std::getenv("KYTY_CS_BIND_LOG");
		return v != nullptr && std::strcmp(v, "2") == 0;
	}();
	return on;
}
} // namespace

namespace Libs::Graphics {
// rtSession.h BudgetGuard: true when this dispatch of a software-RT kernel must be skipped because
// lanes of an earlier RT dispatch are exhausting the BVH node budget (invalid BVH). Command
// processor thread only. KYTY_RT_BUDGET_GUARD=0 turns it off; _LANES / _HOLD_MS tune it.
// rtSession.h LoadHold: true while the game streams a level in (software-RT dispatches are skipped,
// the BVH they would traverse may not be valid yet). Command processor thread only.
// KYTY_RT_LOAD_HOLD=0 off; _PHASE_MS (5000, after a level phase change), _QUIET_MS (1500, after
// fast reads), _MAX_MS (60000, counted from the last level phase change), _VFS_MBS (48), _PHASE=0
// (phase changes do not count). Called for EVERY compute dispatch so the signs are sampled every
// frame; `rt` = this dispatch is a RT kernel.
static bool RtLoadHold(bool rt) {
	static const auto env = [](const char* name, uint64_t fallback) {
		const char* v = std::getenv(name);
		return v != nullptr && v[0] != '\0' ? std::strtoull(v, nullptr, 0) : fallback;
	};
	static const bool enabled = env("KYTY_RT_LOAD_HOLD", 1) != 0;
	if (!enabled) return false;
#if defined(__APPLE__) || defined(__linux__)
	static const auto* bytes = reinterpret_cast<const std::atomic<unsigned long long>*>(
	    dlsym(RTLD_DEFAULT, "nps5_vfs_bytes_read"));
	static const auto* phase =
	    reinterpret_cast<const std::atomic<int>*>(dlsym(RTLD_DEFAULT, "nps5_vfs_phase"));
#else
	static const std::atomic<unsigned long long>* bytes = nullptr;
	static const std::atomic<int>*                phase = nullptr;
#endif
	if (bytes == nullptr) return false;
	static RtSession::LoadHold hold = [] {
		RtSession::LoadHold h;
		h.quiet_ms       = env("KYTY_RT_LOAD_HOLD_QUIET_MS", h.quiet_ms);
		h.phase_quiet_ms = env("KYTY_RT_LOAD_HOLD_PHASE_MS", h.phase_quiet_ms);
		h.max_hold_ms    = env("KYTY_RT_LOAD_HOLD_MAX_MS", h.max_hold_ms);
		h.rate_bytes_s   = env("KYTY_RT_LOAD_HOLD_VFS_MBS", 48) << 20;
		h.use_phase      = env("KYTY_RT_LOAD_HOLD_PHASE", 1) != 0;
		return h;
	}();
	static uint64_t skipped = 0;
	static bool     was     = false;
	const auto      now_ms =
	    static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
	                              std::chrono::steady_clock::now().time_since_epoch())
	                              .count());
	const bool held =
	    hold.Hold(bytes->load(std::memory_order_relaxed),
	              phase != nullptr ? phase->load(std::memory_order_relaxed) : 0, now_ms);
	if (held) {
		if (!was) {
			Log::WriteToConsoleAndLog(fmt::format("RT load hold: a level is loading; software-RT "
			                                      "dispatches are skipped until {} ms after "
			                                      "the level opened and {} ms after the reads calm "
			                                      "down (hold {}; KYTY_RT_LOAD_HOLD=0 "
			                                      "disables)\n",
			                                      hold.phase_quiet_ms, hold.quiet_ms,
			                                      hold.Holds()));
		}
		if (rt) skipped++;
	} else if (was) {
		Log::WriteToConsoleAndLog(
		    fmt::format("RT load hold: over, {} software-RT dispatches skipped so far\n", skipped));
	}
	was = held;
	return held;
}

static bool RtBudgetHold(RenderContext& context) {
	namespace Recompiler = Libs::Graphics::ShaderRecompiler;
	const auto& options  = Recompiler::GetCodegenOptions();
	if (!options.rt_software || options.rt_node_budget == 0) return false;
	static const auto env = [](const char* name, uint64_t fallback) {
		const char* v = std::getenv(name);
		return v != nullptr && v[0] != '\0' ? std::strtoull(v, nullptr, 0) : fallback;
	};
	static const bool enabled = env("KYTY_RT_BUDGET_GUARD", 1) != 0;
	if (!enabled) return false;
	static RtSession::BudgetGuard guard = [] {
		RtSession::BudgetGuard g;
		g.trip_lanes = static_cast<uint32_t>(env("KYTY_RT_BUDGET_GUARD_LANES", g.trip_lanes));
		g.hold_ms    = env("KYTY_RT_BUDGET_GUARD_HOLD_MS", g.hold_ms);
		return g;
	}();
	const auto mapped = context.GetBufferCache().GetGdsBuffer()->Mapped();
	if (mapped.size() < Recompiler::RtNodeBudgetGdsFromEnd * sizeof(uint32_t)) return false;
	uint32_t exhausted = 0;
	std::memcpy(&exhausted,
	            mapped.data() + mapped.size() -
	                Recompiler::RtNodeBudgetGdsFromEnd * sizeof(uint32_t),
	            sizeof(exhausted));
	const auto now_ms =
	    static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
	                              std::chrono::steady_clock::now().time_since_epoch())
	                              .count());
	const auto trips = guard.Trips();
	const bool allow = guard.Allow(exhausted, now_ms);
	if (guard.Trips() != trips) {
		Log::WriteToConsoleAndLog(fmt::format(
		    "RT budget guard: {} lanes have exhausted the BVH node budget (invalid BVH?); "
		    "software-RT "
		    "dispatches are skipped for {} ms (trip {}; KYTY_RT_BUDGET_GUARD=0 disables)\n",
		    exhausted, guard.CurrentHoldMs(), guard.Trips()));
	}
	return !allow;
}
} // namespace Libs::Graphics

namespace Libs::Graphics {
// KYTY_DISPATCH_OVERLAP accounting, printed every 10 s once a direct dispatch ran:
//   DispatchOverlap 10s (on|off): <direct> direct dispatches, <pairs> unordered pairs (<pct>%),
//   <overlapped> overlapped, <indirect> indirect (<empty> empty skipped, KYTY_SKIP_EMPTY_INDIRECT)
// An unordered pair is a direct dispatch that followed another with nothing in between (it would
// overlap with the switch on); counted with the switch off too, for sizing.
static std::atomic<uint64_t> g_overlap_direct {0}, g_overlap_pairs {0}, g_overlap_taken {0},
    g_overlap_indirect {0}, g_overlap_empty_indirect {0};
static Live::Switch g_skip_empty_indirect("KYTY_SKIP_EMPTY_INDIRECT", Live::ParseDefaultOn);
static std::atomic<int64_t> g_overlap_last_print_ns {0};

static void PrintDispatchOverlap(bool enabled) {
	const auto now = static_cast<int64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                          std::chrono::steady_clock::now().time_since_epoch())
	                                          .count());
	auto last = g_overlap_last_print_ns.load(std::memory_order_relaxed);
	if (last == 0) {
		g_overlap_last_print_ns.store(now, std::memory_order_relaxed);
		return;
	}
	if (now - last < 10'000'000'000ll ||
	    !g_overlap_last_print_ns.compare_exchange_strong(last, now, std::memory_order_relaxed)) {
		return;
	}
	const auto direct   = g_overlap_direct.exchange(0, std::memory_order_relaxed);
	const auto pairs    = g_overlap_pairs.exchange(0, std::memory_order_relaxed);
	const auto taken    = g_overlap_taken.exchange(0, std::memory_order_relaxed);
	const auto indirect = g_overlap_indirect.exchange(0, std::memory_order_relaxed);
	const auto empty    = g_overlap_empty_indirect.exchange(0, std::memory_order_relaxed);
	std::printf("DispatchOverlap 10s (%s): %llu direct dispatches, %llu unordered pairs (%.1f%%), "
	            "%llu overlapped, %llu indirect (%llu empty skipped)\n",
	            enabled ? "on" : "off", static_cast<unsigned long long>(direct),
	            static_cast<unsigned long long>(pairs),
	            direct != 0 ? 100.0 * static_cast<double>(pairs) / static_cast<double>(direct) : 0.0,
	            static_cast<unsigned long long>(taken), static_cast<unsigned long long>(indirect),
	            static_cast<unsigned long long>(empty));
}

static void CountDispatchOverlap(bool unordered, bool enabled) {
	g_overlap_direct.fetch_add(1, std::memory_order_relaxed);
	if (unordered) {
		g_overlap_pairs.fetch_add(1, std::memory_order_relaxed);
		if (enabled) {
			g_overlap_taken.fetch_add(1, std::memory_order_relaxed);
		}
	}
	PrintDispatchOverlap(enabled);
}

static bool FillSourcesDisjoint(std::span<const ShaderRecompiler::IR::DescriptorValue> sources,
                                 GuestRange destination, uint32_t output_buffer = UINT32_MAX) {
	for (uint32_t i = 0; i < sources.size(); ++i) {
		if (i == output_buffer) continue;
		const auto source = DecodeNativeDescriptor<ShaderBufferResource>(sources[i]);
		const auto bytes  = source.GetSize();
		if (source.Base48() < destination.End() && destination.address < source.Base48() + bytes)
			return false;
	}
	return true;
}

// KYTY_COMPUTE_META_CLEAR_DISPATCH=1 (default off; upstream 01ce97388's behaviour): a dispatch
// whose buffer writes recorded a metadata clear (UpdateComputeMetaClears) still runs, so its guest
// writes to the metadata memory are kept for readers that reuse it as an ordinary buffer. Off, the
// recorded clear consumes the dispatch, as before.
static bool ComputeMetaClearDispatchEnabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_COMPUTE_META_CLEAR_DISPATCH");
		return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
	}();
	return enabled;
}

// Whether a metadata clear was recorded.
bool RenderExecutor::UpdateComputeMetaClears(const ShaderComputeInputInfo& input,
                                            const CommandBuffer&          buffer) {
	const auto& program   = *input.stage.program;
	const auto& resources = *input.stage.resources;
	if (resources.buffers.size() != program.info.buffers.size()) {
		EXIT("compute runtime buffer count does not match shader metadata\n");
	}
	auto& cache = buffer.GetContext().GetTextureCache();
	for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
		const auto& resource   = program.info.buffers[i];
		const auto  descriptor = DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[i]);
		// Reads and transforms of metadata do not establish a clear state.
		if ((!resource.written || resource.read) && cache.IsMeta(descriptor.Base48())) {
			return false;
		}
	}

	// memtrk: KYTY_META_EXPORT_NOT_CLEAR=1 (default off). A dispatch that writes metadata AND other
	// outputs (a storage image, or a buffer that is not metadata) is a surface export, not a clear:
	// UE5 Nanite writes scene depth and its HTILE from one compute pass. Recording it as a clear
	// skipped the dispatch and made the next depth-target use load with CLEAR, which wiped the
	// exported depth (SAROS: only the rasterised character keeps depth; the world is fogged out).
	// On, such a dispatch runs and its metadata stops being clear. KYTY_META_CLEAR_LOG=1 prints
	// each compute shader's classification once (clear / export / ignored-xor).
	static const bool export_not_clear = [] {
		const auto* value = std::getenv("KYTY_META_EXPORT_NOT_CLEAR");
		return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
	}();
	static const bool meta_log = [] {
		const auto* value = std::getenv("KYTY_META_CLEAR_LOG");
		return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
	}();
	uint32_t meta_writes = 0, other_buffer_writes = 0, image_writes = 0;
	for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
		if (!program.info.buffers[i].written) continue;
		const auto descriptor = DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[i]);
		if (cache.IsMeta(descriptor.Base48())) {
			meta_writes++;
		} else {
			other_buffer_writes++;
		}
	}
	for (const auto& image: program.info.images) {
		image_writes += image.written ? 1u : 0u;
	}
	const bool is_export = meta_writes != 0 && (other_buffer_writes != 0 || image_writes != 0);
	if (meta_log && meta_writes != 0) {
		static std::mutex                   log_mutex;
		static std::unordered_set<uint64_t> logged;
		bool                                first = false;
		{
			std::scoped_lock lock(log_mutex);
			first = logged.insert(program.shader_hash).second;
		}
		if (first) {
			std::fprintf(stderr,
			             "[memtrk] meta-writing CS 0x%016" PRIx64
			             ": %u metadata buffers, %u other buffers, %u images "
			             "written, xor %d -> %s\n",
			             program.shader_hash, meta_writes, other_buffer_writes, image_writes,
			             program.info.has_bitwise_xor ? 1 : 0,
			             program.info.has_bitwise_xor ? "ignored (xor)"
			             : is_export ? (export_not_clear ? "export (runs, metadata expanded)"
			                                             : "export recorded as CLEAR (switch off)")
			                         : "clear");
		}
	}
	// memtrk: KYTY_HTILE_CS_EXPAND=1 (default off). A compute pass that writes only HTILE is not
	// necessarily a fast clear: it may initialise HTILE to "expanded" before a raw depth export
	// (UE5 Nanite writes scene depth from compute afterwards). Kyty does not decode the HTILE
	// values, so recording it as a clear made every following depth bind load CLEAR and wipe the
	// exported depth (SAROS r93mx2: the scene depth 0x3079090000 loads CLEAR at each bind). On,
	// such a dispatch runs and leaves the metadata expanded; draws that clear depth themselves
	// still do.
	static const bool cs_expand = [] {
		const auto* value = std::getenv("KYTY_HTILE_CS_EXPAND");
		return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
	}();
	if (cs_expand && meta_writes != 0 && !is_export && !program.info.has_bitwise_xor) {
		for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
			if (!program.info.buffers[i].written) continue;
			const auto descriptor =
			    DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[i]);
			(void)cache.ExpandMeta(descriptor.Base48());
		}
		return false;
	}
	if (export_not_clear && is_export) {
		for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
			if (!program.info.buffers[i].written) continue;
			const auto descriptor =
			    DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[i]);
			(void)cache.ExpandMeta(descriptor.Base48());
		}
		return false;
	}

	bool cleared = false;
	if (!program.info.has_bitwise_xor) {
		for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
			const auto& resource = program.info.buffers[i];
			if (resource.written) {
				const auto descriptor =
				    DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[i]);
				cleared |= cache.ClearMeta(descriptor.Base48());
			}
		}
	}
	return cleared;
}

bool ResolveComputeBufferFill(const ShaderComputeInputInfo& input, uint32_t group_x,
                              uint32_t group_y, uint32_t group_z, uint32_t mode,
                              ShaderBufferResource& resolved_descriptor, uint32_t& resolved_clear,
                              uint64_t& resolved_size) {
	const auto& resources = *input.stage.resources;
	const auto& fill      = resources.uniform_fill;
	if (fill.kind != ShaderRecompiler::IR::UniformFillKind::Buffer) {
		return false;
	}
	const auto element_size = fill.words * sizeof(uint32_t);
	const auto descriptor =
	    DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[fill.resource]);
	constexpr std::array formats {
	    Prospero::BufferFormat::k32UInt, Prospero::BufferFormat::k32_32UInt,
	    Prospero::BufferFormat::k32_32_32UInt, Prospero::BufferFormat::k32_32_32_32UInt};
	if (descriptor.Stride() != element_size || descriptor.Format() != formats[fill.words - 1] ||
	    descriptor.SwizzleEnabled() || descriptor.IndexStride() != 0 || descriptor.AddTid() ||
	    descriptor.Base48() == 0) {
		return false;
	}
	if (input.threads_num[0] == 0 || input.threads_num[0] != fill.group_stride[0] ||
	    input.threads_num[1] != 1 || input.threads_num[2] != 1 || group_x == 0 || group_y != 1 ||
	    group_z != 1 || mode != (input.dispatch_thread_dimensions ? 0x61u : 0x41u)) {
		return false;
	}
	const uint64_t invocations = input.dispatch_thread_dimensions
	                                 ? group_x
	                                 : static_cast<uint64_t>(group_x) * input.threads_num[0];
	const auto     size        = descriptor.GetSize();
	if (invocations != descriptor.NumRecords() || size == 0 || size > UINT32_MAX ||
	    (input.dispatch_thread_dimensions &&
	     (group_x % input.threads_num[0] != 0 || input.dispatch_threads_num[0] != group_x ||
	      input.dispatch_threads_num[1] != 1 || input.dispatch_threads_num[2] != 1))) {
		return false;
	}
	if (!FillSourcesDisjoint(resources.buffers, {descriptor.Base48(), size}, fill.resource))
		return false;
	resolved_descriptor = descriptor;
	resolved_clear      = fill.value;
	resolved_size       = size;
	return true;
}

// rtimage KYTY_COMPUTE_IMAGE_COPY (computeImageCopy.h): the dispatch of a proven copy kernel
// (IR::AnalyzeUniformCopy: destination element i = source element i for every invocation) copies
// the bytes [source, source + size) to [destination, destination + size) exactly when
//  - both descriptors are 32UInt elements of 4 bytes without swizzle, index stride or thread-id
//    addressing: the formatted load returns the element's dword and the formatted store writes it;
//  - the dispatch runs one invocation per element of both buffers (complete workgroups, as
//    ResolveComputeBufferFill requires), so no load is out of range and no element is left out;
//  - the two ranges do not overlap.
static bool ResolveComputeBufferCopy(const ShaderComputeInputInfo& input, uint32_t group_x,
                                     uint32_t group_y, uint32_t group_z, uint32_t mode,
                                     const ShaderRecompiler::IR::UniformCopy& copy,
                                     uint64_t& source, uint64_t& destination, uint64_t& size) {
	const auto& resources = *input.stage.resources;
	if (!copy.valid || resources.buffers.size() != 2 || copy.source >= 2 || copy.destination >= 2 ||
	    copy.source == copy.destination || !resources.images.empty()) {
		return false;
	}
	const auto from = DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[copy.source]);
	const auto to =
	    DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[copy.destination]);
	for (const auto* descriptor: {&from, &to}) {
		if (descriptor->Format() != Prospero::BufferFormat::k32UInt || descriptor->Stride() != 4 ||
		    descriptor->SwizzleEnabled() || descriptor->IndexStride() != 0 ||
		    descriptor->AddTid() || descriptor->Base48() == 0) {
			return false;
		}
	}
	if (input.threads_num[0] == 0 || input.threads_num[0] != copy.group_stride ||
	    input.threads_num[1] != 1 || input.threads_num[2] != 1 || group_x == 0 || group_y != 1 ||
	    group_z != 1 || mode != (input.dispatch_thread_dimensions ? 0x61u : 0x41u)) {
		return false;
	}
	const uint64_t invocations = input.dispatch_thread_dimensions
	                                 ? group_x
	                                 : static_cast<uint64_t>(group_x) * input.threads_num[0];
	const auto     bytes       = from.GetSize();
	if (invocations != from.NumRecords() || invocations != to.NumRecords() || bytes == 0 ||
	    bytes != to.GetSize() || bytes != invocations * 4u || bytes > UINT32_MAX ||
	    (input.dispatch_thread_dimensions &&
	     (group_x % input.threads_num[0] != 0 || input.dispatch_threads_num[0] != group_x ||
	      input.dispatch_threads_num[1] != 1 || input.dispatch_threads_num[2] != 1))) {
		return false;
	}
	if (from.Base48() < to.Base48() + bytes && to.Base48() < from.Base48() + bytes) {
		return false;
	}
	source      = from.Base48();
	destination = to.Base48();
	size        = bytes;
	return true;
}

bool RenderExecutor::TryConsumeComputeImageClear(const ShaderComputeInputInfo& input,
                                                CommandBuffer& command, uint32_t group_x,
                                                uint32_t group_y, uint32_t group_z, uint32_t mode) {
	const auto& program   = *input.stage.program;
	const auto& resources = *input.stage.resources;
	const auto& fill      = resources.uniform_fill;
	auto&       cache     = command.GetContext().GetTextureCache();
	if (fill.kind == ShaderRecompiler::IR::UniformFillKind::Image) {
		if (mode != 0x41u || input.dispatch_thread_dimensions || fill.value > 255 ||
		    input.threads_num[2] != 1)
			return false;
		const auto  descriptor = DecodeNativeDescriptor<ShaderTextureResource>(resources.images[0]);
		const auto& resource   = program.info.images[0];
		if (descriptor.IsNull() || descriptor.Format() != Prospero::BufferFormat::k8UInt ||
		    descriptor.Type() != Prospero::ImageType::kColor2DArray || descriptor.MetaCompress() ||
		    descriptor.WriteCompress() || descriptor.BaseLevel() > descriptor.LastLevel() ||
		    descriptor.BaseLevel() > descriptor.MaxMip() ||
		    descriptor.BaseArray5() > descriptor.Depth() || descriptor.DstSelX() != 4)
			return false;
		const std::array extents {
		    std::max(1u, (descriptor.Width5() + 1u) >> descriptor.BaseLevel()),
		    std::max(1u, (descriptor.Height5() + 1u) >> descriptor.BaseLevel()),
		    descriptor.Depth() - descriptor.BaseArray5() + 1u};
		const std::array groups {group_x, group_y, group_z};
		for (uint32_t axis = 0; axis < 3; ++axis) {
			const uint64_t threads = input.threads_num[axis];
			// Guest image writes outside the descriptor dimensions are discarded. Only the
			// final workgroup may extend beyond the selected image view.
			if (threads == 0 || threads != fill.group_stride[axis] ||
			    groups[axis] != (extents[axis] + threads - 1) / threads ||
			    groups[axis] * threads > UINT32_MAX) return false;
		}
		const auto  binding     = ResolveTexture(resource, resources.images[0]);
		const auto& destination = binding.desc.info.data;
		if (!FillSourcesDisjoint(resources.buffers, destination)) return false;
		std::scoped_lock lock {cache.m_lock};
		const auto&      image = cache.GetImage(binding.image_id);
		const auto&      view  = binding.desc.view_info;
		if (image.backing.format != vk::Format::eD32SfloatS8Uint || image.info.samples != 1 ||
		    image.info.stencil != destination || view.base_level >= image.backing.mip_levels ||
		    view.base_layer >= image.backing.layers || view.layer_count != extents[2] ||
		    view.layer_count > image.backing.layers - view.base_layer ||
		    std::max(1u, image.info.extent.width >> view.base_level) != extents[0] ||
		    std::max(1u, image.info.extent.height >> view.base_level) != extents[1]) return false;
		const vk::ImageSubresourceRange range {vk::ImageAspectFlagBits::eStencil, view.base_level,
		                                       1, view.base_layer, view.layer_count};
		vk::ClearValue clear {};
		clear.depthStencil = vk::ClearDepthStencilValue {0.0f, fill.value};
		// A rejected clear (reported) leaves the guest shader to run.
		return cache.ClearImage(command, binding.image_id, image.backing.format, range, clear,
		                        "compute-stencil-fill");
	}
	ShaderBufferResource descriptor;
	uint32_t             packed_clear = 0;
	uint64_t             size         = 0;
	if (!ResolveComputeBufferFill(input, group_x, group_y, group_z, mode, descriptor, packed_clear,
	                              size)) {
		return false;
	}
	if (!cache.ClearImageFromBuffer(command, descriptor.Base48(), size, packed_clear)) {
		return false;
	}
	static std::atomic<uint32_t> logged_clears {0};
	if (logged_clears.fetch_add(1, std::memory_order_relaxed) < 32) {
		LOGF("GraphicsRenderDispatchDirect: compute image clear shader=0x%016" PRIx64
		     " addr=0x%016" PRIx64 " size=0x%016" PRIx64 " value=0x%08" PRIx32 "\n",
		     input.stage.program->shader_hash, descriptor.Base48(), size, packed_clear);
	}
	return true;
}

// nps5 nanite: KYTY_CS_PROBE / KYTY_DRAW_PROBE / KYTY_PROBE_IDS (nanProbe.h). One-shot host copies
// of bound images around a dispatch, written once their submission completed.
static bool IndirectLogEnabled();
namespace NanProbeImpl {
struct Entry {
	char             name[160] {};
	vk::Buffer       buffer = nullptr;
	vk::DeviceMemory memory = nullptr;
	void*            mapped = nullptr;
	uint32_t         width = 0, height = 0, texel = 0, layers = 1;
	uint32_t         copy_w = 0, copy_h = 0; // sarosclay: texel extent of the copy (block formats:
	                                         // width/height are block counts)
	uint64_t tick = 0;
};
static std::vector<Entry>& Pending() {
	static std::vector<Entry> pending;
	return pending;
}
static uint32_t TexelBytes(vk::Format format) {
	switch (format) {
		case vk::Format::eR8Unorm:
		case vk::Format::eR8Uint: return 1;
		case vk::Format::eR16Uint:
		case vk::Format::eR16Sfloat:
		case vk::Format::eR16Unorm:
		case vk::Format::eR8G8Unorm: return 2;
		case vk::Format::eR32Uint:
		case vk::Format::eR32Sint:
		case vk::Format::eR32Sfloat:
		case vk::Format::eR8G8B8A8Unorm:
		case vk::Format::eR8G8B8A8Uint:
		case vk::Format::eB8G8R8A8Unorm:
		case vk::Format::eB8G8R8A8Srgb: // sarosclay: probe-only allowance (the albedo target is
		                                // sRGB)
		case vk::Format::eR8G8B8A8Srgb:
		case vk::Format::eA2B10G10R10UnormPack32:
		case vk::Format::eB10G11R11UfloatPack32:
		case vk::Format::eR16G16Sfloat:
		case vk::Format::eR16G16Unorm:
		case vk::Format::eR16G16Uint: return 4;
		case vk::Format::eR32G32Uint:
		case vk::Format::eR32G32Sfloat:
		case vk::Format::eR64Uint:
		case vk::Format::eR16G16B16A16Sfloat:
		case vk::Format::eR16G16B16A16Unorm:
		case vk::Format::eR16G16B16A16Uint: return 8;
		case vk::Format::eR32G32B32A32Uint:
		case vk::Format::eR32G32B32A32Sfloat: return 16;
		default: return 0;
	}
}
// sarosclay: BC formats are dumped as raw blocks (width/height in blocks, texel = block bytes);
// probe only.
static uint32_t BlockBytes(vk::Format format) {
	switch (format) {
		case vk::Format::eBc1RgbUnormBlock:
		case vk::Format::eBc1RgbSrgbBlock:
		case vk::Format::eBc1RgbaUnormBlock:
		case vk::Format::eBc1RgbaSrgbBlock:
		case vk::Format::eBc4UnormBlock:
		case vk::Format::eBc4SnormBlock: return 8;
		case vk::Format::eBc2UnormBlock:
		case vk::Format::eBc2SrgbBlock:
		case vk::Format::eBc3UnormBlock:
		case vk::Format::eBc3SrgbBlock:
		case vk::Format::eBc5UnormBlock:
		case vk::Format::eBc5SnormBlock:
		case vk::Format::eBc6HUfloatBlock:
		case vk::Format::eBc6HSfloatBlock:
		case vk::Format::eBc7UnormBlock:
		case vk::Format::eBc7SrgbBlock: return 16;
		default: return 0;
	}
}
static void LogIdentity(const char* what, uint64_t hash, uint32_t slot, TextureCache& cache,
                        ImageId id, const TextureCache::ImageDesc* desc) {
	auto* image = cache.TryGetImage(id);
	if (image == nullptr) {
		std::fprintf(stderr, "[nanprobe] id %s 0x%016" PRIx64 " img%u slot %u:%u FREED\n", what,
		             hash, slot, id.index, id.generation);
		return;
	}
	std::fprintf(
	    stderr,
	    "[nanprobe] id %s 0x%016" PRIx64 " img%u%s slot %u:%u vk %p guest 0x%010" PRIx64
	    "+0x%" PRIx64 " %ux%u %s gpu-mod %d buf-mod %d cpu-dirty %d owner %d writes %llu reg %d\n",
	    what, hash, slot,
	    desc != nullptr ? (desc->type == TextureCache::BindingType::Storage ? " st" : " tx") : "",
	    id.index, id.generation, static_cast<void*>(static_cast<VkImage>(image->backing.image)),
	    image->info.data.address, image->info.data.size, image->backing.extent.width,
	    image->backing.extent.height, vk::to_string(image->backing.format).c_str(),
	    image->IsGpuModified() ? 1 : 0, image->IsBufferModified() ? 1 : 0,
	    image->IsCpuDirty() ? 1 : 0, image->alias_owner ? 1 : 0,
	    static_cast<unsigned long long>(image->DefiniteWrites()), image->registered ? 1 : 0);
}
// Records a level-0/layer-0 copy of `id` into the current command buffer.
static void Capture(RenderContext& context, CommandBuffer& buffer, ImageId id, const char* label,
                    uint64_t hash, uint32_t slot) {
	auto& cache = context.GetTextureCache();
	LogIdentity(label, hash, slot, cache, id, nullptr);
	auto* image = cache.TryGetImage(id);
	if (image == nullptr || image->backing.image == nullptr || image->info.IsDepth() ||
	    image->backing.samples != 1) {
		return;
	}
	uint32_t       texel = TexelBytes(image->backing.format);
	const uint32_t block = texel == 0 ? BlockBytes(image->backing.format) : 0u;
	if (texel == 0 && block == 0) {
		std::fprintf(stderr, "[nanprobe] %s img%u: format %s not dumped\n", label, slot,
		             vk::to_string(image->backing.format).c_str());
		return;
	}
	auto& graphics = context.GetGraphics();
	auto  device   = graphics.device;
	Entry e {};
	e.width  = image->backing.extent.width;
	e.height = image->backing.extent.height;
	e.texel  = texel;
	e.copy_w = e.width;
	e.copy_h = e.height;
	if (block != 0) {
		e.width  = (e.width + 3u) / 4u;
		e.height = (e.height + 3u) / 4u;
		e.texel  = block;
		texel    = block;
	}
	// saroslight: KYTY_PROBE_LAYERS=<n> copies up to n array layers (default 1) into one raw dump.
	static const uint32_t max_layers = [] {
		const auto* v = std::getenv("KYTY_PROBE_LAYERS");
		return v != nullptr && v[0] != '\0'
		           ? static_cast<uint32_t>(std::max<long>(std::strtol(v, nullptr, 10), 1))
		           : 1u;
	}();
	e.layers = image->info.IsVolume()
	               ? 1u
	               : std::max(1u, std::min(max_layers, image->info.resources.layers));
	std::snprintf(e.name, sizeof(e.name),
	              "nanprobe_%s_%016" PRIx64 "_img%u_%" PRIx64 "_%ux%u_%ub%s_L%u", label, hash, slot,
	              image->info.data.address, e.width, e.height, texel, block != 0 ? "blk" : "",
	              e.layers);
	vk::BufferCreateInfo create {};
	create.size  = static_cast<vk::DeviceSize>(e.width) * e.height * texel * e.layers;
	create.usage = vk::BufferUsageFlagBits::eTransferDst;
	if (device.createBuffer(&create, nullptr, &e.buffer) != vk::Result::eSuccess) return;
	vk::MemoryRequirements requirements {};
	device.getBufferMemoryRequirements(e.buffer, &requirements);
	const auto& memory = graphics.physical_device_memory_properties;
	const auto  wanted =
	    vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;
	uint32_t type = UINT32_MAX;
	for (uint32_t i = 0; i < memory.memoryTypeCount; i++) {
		if ((requirements.memoryTypeBits & (1u << i)) != 0 &&
		    (memory.memoryTypes[i].propertyFlags & wanted) == wanted) {
			type = i;
			break;
		}
	}
	vk::MemoryAllocateInfo allocate {};
	allocate.allocationSize  = requirements.size;
	allocate.memoryTypeIndex = type;
	if (type == UINT32_MAX ||
	    device.allocateMemory(&allocate, nullptr, &e.memory) != vk::Result::eSuccess) {
		device.destroyBuffer(e.buffer, nullptr);
		return;
	}
	device.bindBufferMemory(e.buffer, e.memory, 0);
	if (device.mapMemory(e.memory, 0, VK_WHOLE_SIZE, {}, &e.mapped) != vk::Result::eSuccess) return;
	auto command = buffer.Handle();
	// Every earlier write (fragment storage writes, compute, transfers) before the copy.
	vk::MemoryBarrier full {};
	full.srcAccessMask = vk::AccessFlagBits::eMemoryWrite;
	full.dstAccessMask = vk::AccessFlagBits::eTransferRead;
	command.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                        vk::PipelineStageFlagBits::eTransfer, {}, 1, &full, 0, nullptr, 0,
	                        nullptr);
	image->Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, {},
	               command);
	vk::BufferImageCopy copy {};
	copy.imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, e.layers};
	copy.imageExtent      = vk::Extent3D {e.copy_w, e.copy_h, 1};
	command.copyImageToBuffer(image->backing.image, vk::ImageLayout::eTransferSrcOptimal, e.buffer,
	                          1, &copy);
	vk::MemoryBarrier after {};
	after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask = vk::AccessFlagBits::eHostRead;
	command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost,
	                        {}, 1, &after, 0, nullptr, 0, nullptr);
	e.tick = context.GetCommandScheduler().CurrentTick();
	Pending().push_back(e);
}
static void Write(vk::Device device, Entry& e) {
	const uint64_t       n       = static_cast<uint64_t>(e.width) * e.height;
	const auto*          src     = static_cast<const uint8_t*>(e.mapped);
	uint64_t             nonzero = 0;
	uint32_t             hi_min = UINT32_MAX, hi_max = 0, lo_min = UINT32_MAX, lo_max = 0;
	std::vector<uint8_t> bmp(n * 4);
	for (uint64_t i = 0; i < n; i++) {
		const uint8_t* t   = src + i * e.texel;
		uint32_t       low = 0;
		std::memcpy(&low, t, std::min<uint32_t>(e.texel, 4u));
		bool any = false;
		for (uint32_t b = 0; b < e.texel; b++)
			any |= t[b] != 0;
		uint8_t r = 0, g = 0, bl = 0;
		if (any) {
			nonzero++;
			lo_min = std::min(lo_min, low);
			lo_max = std::max(lo_max, low);
			if (e.texel >= 8) {
				uint32_t hi = 0;
				std::memcpy(&hi, t + 4, 4);
				hi_min = std::min(hi_min, hi);
				hi_max = std::max(hi_max, hi);
			}
			uint32_t h = low * 2654435761u;
			h ^= h >> 15;
			r  = static_cast<uint8_t>(64 + (h & 0xbf));
			g  = static_cast<uint8_t>(64 + ((h >> 8) & 0xbf));
			bl = static_cast<uint8_t>(64 + ((h >> 16) & 0xbf));
		}
		bmp[i * 4 + 0] = bl;
		bmp[i * 4 + 1] = g;
		bmp[i * 4 + 2] = r;
		bmp[i * 4 + 3] = 0xff;
	}
	const char* dir = std::getenv("KYTY_DUMP_DIR");
	std::string base =
	    dir != nullptr && dir[0] != 0 ? std::string(dir) + "/" + e.name : std::string(e.name);
	if (auto* f = std::fopen((base + ".raw").c_str(), "wb"); f != nullptr) {
		std::fwrite(src, 1, n * e.texel * e.layers, f);
		std::fclose(f);
	}
	uint8_t    header[54] {};
	const auto bytes = static_cast<uint32_t>(n * 4);
	const auto put   = [&](size_t at, uint32_t value) { std::memcpy(header + at, &value, 4); };
	header[0]        = 'B';
	header[1]        = 'M';
	put(2, 54u + bytes);
	put(10, 54u);
	put(14, 40u);
	put(18, e.width);
	put(22, static_cast<uint32_t>(-static_cast<int32_t>(e.height)));
	header[26] = 1;
	header[28] = 32;
	put(34, bytes);
	if (auto* f = std::fopen((base + ".bmp").c_str(), "wb"); f != nullptr) {
		std::fwrite(header, 1, sizeof(header), f);
		std::fwrite(bmp.data(), 1, bytes, f);
		std::fclose(f);
	}
	std::fprintf(
	    stderr, "[nanprobe] wrote %s: nonzero %.2f%% low 0x%08x..0x%08x high 0x%08x..0x%08x\n",
	    e.name,
	    100.0 * static_cast<double>(nonzero) / static_cast<double>(std::max<uint64_t>(n, 1)),
	    nonzero ? lo_min : 0u, lo_max, nonzero && e.texel >= 8 ? hi_min : 0u, hi_max);
	device.unmapMemory(e.memory);
	device.destroyBuffer(e.buffer, nullptr);
	device.freeMemory(e.memory, nullptr);
}
static void Poll(RenderContext& context) {
	auto& pending = Pending();
	if (pending.empty()) return;
	auto& scheduler = context.GetCommandScheduler();
	for (size_t i = 0; i < pending.size();) {
		if (pending[i].tick < scheduler.CurrentTick() && scheduler.IsFree(pending[i].tick)) {
			Write(context.GetGraphics().device, pending[i]);
			pending.erase(pending.begin() + static_cast<std::ptrdiff_t>(i));
		} else {
			i++;
		}
	}
}
} // namespace NanProbeImpl
// saroslights: for a KYTY_CS_PROBE hash, nonzero-dword statistics of every bound buffer (the bytes
// the previous dispatches left) before its 1st, 2nd, 3rd, 5th, 9th, 17th, ... (2^k+1) and 2^k-th
// dispatch.
template <class Program, class Resources>
static void SarosBufferStats(const Program& program, const Resources& resources,
                             const std::vector<uint64_t>& probe_hashes, uint64_t args_addr = 0) {
	if (std::find(probe_hashes.begin(), probe_hashes.end(), program.shader_hash) ==
	    probe_hashes.end())
		return;
	static std::unordered_map<uint64_t, uint32_t> buf_counts;
	const uint32_t                                n = ++buf_counts[program.shader_hash];
	if ((n & (n - 1u)) != 0u && (n - 1u & (n - 2u)) != 0u && n != 3u)
		return; // 1, 2, 3, 5, 9, 17, 33, 65, ... and powers of two
	if (args_addr != 0) {
		uint32_t args[3] {};
		if (LibKernel::Memory::SynchronizeGpuBackingForRead(args_addr, sizeof(args)) &&
		    LibKernel::Memory::TryReadGpuCleanBacking(args_addr, args, sizeof(args))) {
			std::fprintf(stderr, "[saroslights] cs 0x%016" PRIx64 " #%u indirect args %ux%ux%u\n",
			             program.shader_hash, n, args[0], args[1], args[2]);
		}
	}
	for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
		const auto     r    = DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[i]);
		const uint64_t base = r.Base48();
		const uint64_t size = std::min<uint64_t>(static_cast<uint64_t>(r.GetSize()), 4ull << 20u);
		if (base == 0 || size < 4) continue;
		std::vector<uint32_t> data(size / 4u);
		if (!LibKernel::Memory::SynchronizeGpuBackingForRead(base, size) ||
		    !LibKernel::Memory::TryReadGpuCleanBacking(base, data.data(), data.size() * 4u)) {
			std::fprintf(stderr,
			             "[saroslights] cs 0x%016" PRIx64 " #%u buf%u 0x%010" PRIx64 "+0x%" PRIx64
			             " %s unreadable\n",
			             program.shader_hash, n, i, base, size,
			             program.info.buffers[i].written ? "w" : "r");
			continue;
		}
		uint64_t nonzero = 0, first = UINT64_MAX, last = 0, sum = 0;
		for (uint64_t k = 0; k < data.size(); k++) {
			if (data[k] != 0) {
				nonzero++;
				if (first == UINT64_MAX) first = k * 4;
				last = k * 4;
			}
			sum += data[k];
		}
		std::fprintf(stderr,
		             "[saroslights] cs 0x%016" PRIx64 " #%u buf%u 0x%010" PRIx64 "+0x%" PRIx64
		             " %s nonzero %" PRIu64 "/%zu first 0x%" PRIx64 " last 0x%" PRIx64
		             " sum 0x%" PRIx64 "\n",
		             program.shader_hash, n, i, base, size,
		             program.info.buffers[i].written ? "w" : "r", nonzero, data.size(),
		             first == UINT64_MAX ? 0ull : first, last, sum);
	}
}

// KYTY_LDS_DEVICE_BUFFER_MB (default 1024): the most memory the compute LDS device buffer may
// take; a direct dispatch with more workgroups than it holds runs its excess workgroups without
// LDS (their accesses read zero and drop writes), and says so once per shader.
static uint64_t ComputeLdsMaxBytes(const GraphicContext& graphics) {
	static const uint64_t configured = [] {
		const auto* value = std::getenv("KYTY_LDS_DEVICE_BUFFER_MB");
		const auto  mib   = value != nullptr && value[0] != '\0' ? std::strtoull(value, nullptr, 0) : 1024u;
		return std::max<uint64_t>(mib, 1u) << 20u;
	}();
	return std::min<uint64_t>(configured,
	                          graphics.GetPhysicalDeviceProperties().limits.maxStorageBufferRange);
}

// An indirect dispatch's workgroup count is GPU data: it gets at least this much (1,365 regions
// of 48 KiB), or the buffer as a direct dispatch grew it.
static constexpr uint64_t LdsIndirectBytes = 64ull << 20u;

bool RenderExecutor::BindComputeLds(const ShaderComputeInputInfo& input, PreparedBindings& bindings,
                                    const uint32_t* groups) {
	const auto& program = *input.stage.program;
	if (ShaderRecompiler::IR::FindBinding(
	        program.bindings, ShaderRecompiler::IR::DescriptorBindingKind::SharedMemory) == nullptr) {
		return false;
	}
	auto&          graphics = m_context.GetGraphics();
	const uint64_t region =
	    uint64_t {ShaderRecompiler::LdsStorageRegionDwords(input.lds_size_dwords)} * sizeof(uint32_t);
	EXIT_IF(region == 0);
	const uint64_t max_regions = std::max<uint64_t>(ComputeLdsMaxBytes(graphics) / region, 1u);
	uint64_t       regions     = 0;
	if (groups != nullptr) {
		const uint64_t count = uint64_t {groups[0]} * groups[1] * groups[2];
		regions              = std::min(count, max_regions);
		if (count > max_regions) {
			static std::atomic<uint32_t> logged {0};
			if (logged.fetch_add(1, std::memory_order_relaxed) < 8) {
				char text[256];
				std::snprintf(text, sizeof(text),
				              "Kyty LDS device buffer: compute shader 0x%016" PRIx64
				              " dispatches %" PRIu64 " workgroups of %u bytes of LDS; the buffer holds "
				              "%" PRIu64 " (KYTY_LDS_DEVICE_BUFFER_MB), the others run without LDS\n",
				              program.shader_hash, count, input.lds_size_dwords * 4u, max_regions);
				Log::WriteToConsoleAndLog(text);
			}
		}
	} else {
		const auto current = m_compute_lds != nullptr ? m_compute_lds->Size() / region : 0u;
		regions = std::min(std::max<uint64_t>(current, LdsIndirectBytes / region), max_regions);
	}
	regions            = std::max<uint64_t>(regions, 1u);
	const auto needed = regions * region;
	if (m_compute_lds == nullptr || m_compute_lds->Size() < needed) {
		auto& scheduler = m_context.GetCommandScheduler();
		const auto old_size = m_compute_lds != nullptr ? m_compute_lds->Size() : 0u;
		// Grows by at least half, so a growing series of dispatches reallocates a few times only.
		const auto size =
		    std::min(std::max(needed, old_size + old_size / 2u), max_regions * region);
		if (m_compute_lds != nullptr) {
			// Recorded commands of the current tick may still use the old buffer.
			scheduler.DeferOperation([old = std::move(m_compute_lds)]() mutable { old.reset(); });
		}
		m_compute_lds = std::make_unique<Buffer>(graphics, scheduler, MemoryUsage::DeviceLocal, 0,
		                                         vk::BufferUsageFlagBits::eStorageBuffer, size);
		static std::atomic<uint32_t> logged {0};
		if (logged.fetch_add(1, std::memory_order_relaxed) < 8) {
			char text[256];
			std::snprintf(text, sizeof(text),
			              "Kyty LDS device buffer (KYTY_LDS_DEVICE_BUFFER): %.1f MiB for compute shader "
			              "0x%016" PRIx64 " (%u bytes of LDS per workgroup, %" PRIu64 " workgroups%s)\n",
			              static_cast<double>(size) / (1024.0 * 1024.0), program.shader_hash,
			              input.lds_size_dwords * 4u, regions, groups == nullptr ? ", indirect" : "");
			Log::WriteToConsoleAndLog(text);
		}
	}
	// A direct dispatch binds its own regions; an indirect one every region the buffer holds.
	const auto range =
	    groups != nullptr ? needed : m_compute_lds->Size() / region * region;
	bindings.shared_memory = {m_compute_lds->Handle(), 0, range};
	return true;
}

// KYTY_ORDERED_APPEND=1 (default off; upstream d4465faa7): a dispatch with
// DISPATCH_INITIATOR.ORDERED_APPEND_ENABLE whose program uses GDS and whose workgroups are one
// guest wave without emulated LDS runs one workgroup per vkCmdDispatchBase, in launch order with a
// barrier between them, so ordered-append counters see the workgroups in order. Logged once.
bool OrderedAppendSplitEnabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_ORDERED_APPEND");
		return value != nullptr && std::strcmp(value, "1") == 0;
	}();
	return enabled;
}

// sarosfable: KYTY_BUNDLE_LOG=1 (diagnostic, default off). After KYTY_BUNDLE_LOG_AFTER_S (140)
// seconds, the first two dispatches of every compute shader that has no program print their user
// data; one whose user data has the layout of the command-compaction kernel (SAROS cs
// d872ee3c0e77b1ff: s0 records, s1 record bytes, s2 trailer bytes, s3 args offset|stride<<16, V#
// args s4, V# input s8, V# output s12) also prints its record indices, their dispatch arguments,
// records, and the output region.
static const auto g_bundle_t0 = std::chrono::steady_clock::now();
static bool       BundleLogEnabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_BUNDLE_LOG");
		return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
	}();
	return enabled;
}
static void BundleHex(const char* what, uint64_t addr, uint32_t bytes) {
	std::vector<uint32_t> words((bytes + 3u) / 4u);
	if (words.empty() ||
	    !LibKernel::Memory::TryReadBacking(addr, words.data(), words.size() * 4u)) {
		std::fprintf(stderr, "[bundle]   %s 0x%010" PRIx64 " +%u: unreadable\n", what, addr, bytes);
		return;
	}
	for (size_t i = 0; i < words.size(); i += 8) {
		std::string line;
		char        part[16];
		for (size_t k = i; k < std::min(words.size(), i + 8); k++) {
			std::snprintf(part, sizeof(part), " %08x", words[k]);
			line += part;
		}
		std::fprintf(stderr, "[bundle]   %s 0x%010" PRIx64 ":%s\n", what, addr + i * 4u,
		             line.c_str());
	}
}
template <class Cs, class Cache>
static void BundleLogNoProgram(const char* path, const Cs& cs, Cache& cache, uint32_t gx,
                               uint32_t gy, uint32_t gz, uint64_t args_addr) {
	static const double after = [] {
		const auto* value = std::getenv("KYTY_BUNDLE_LOG_AFTER_S");
		return value != nullptr && value[0] != '\0' ? std::atof(value) : 140.0;
	}();
	if (std::chrono::duration<double>(std::chrono::steady_clock::now() - g_bundle_t0).count() <
	    after)
		return;
	const uint64_t va = cs.cs_regs.data_addr;
	{
		static std::mutex                             mutex;
		static std::unordered_map<uint64_t, uint32_t> seen;
		std::scoped_lock                              lock(mutex);
		if (++seen[va] > 2) return;
	}
	const uint32_t* s = cs.cs_user_sgpr.value;
	std::fprintf(stderr,
	             "[bundle] no-program %s va 0x%010" PRIx64 " grid %ux%ux%u args 0x%010" PRIx64
	             " ud(%u): %08x %08x %08x %08x | %08x %08x %08x %08x | %08x %08x %08x %08x | %08x "
	             "%08x %08x %08x\n",
	             path, va, gx, gy, gz, args_addr, cs.cs_user_sgpr.count, s[0], s[1], s[2], s[3],
	             s[4], s[5], s[6], s[7], s[8], s[9], s[10], s[11], s[12], s[13], s[14], s[15]);
	const auto base = [&](int i) {
		return s[i] | (static_cast<uint64_t>(s[i + 1] & 0xffffu) << 32u);
	};
	const auto size = [&](int i) {
		const uint64_t stride = (s[i + 1] >> 16u) & 0x3fffu;
		return stride == 0 ? uint64_t {s[i + 2]} : stride * s[i + 2];
	};
	const uint32_t n = s[0], r = s[1] & 0xffffu, tail = s[2];
	const uint64_t args = base(4), in = base(8), out = base(12);
	if (n == 0 || n > 8192u || r == 0 || r > 4096u || (r & 3u) != 0 || tail > 4096u || args == 0 ||
	    in == 0 || out == 0) {
		return;
	}
	const uint32_t arg_off = s[3] & 0xffffu, arg_stride = s[3] >> 16u;
	std::fprintf(
	    stderr,
	    "[bundle] compactor layout: records %u x %u bytes, trailer %u bytes, args 0x%010" PRIx64
	    "+0x%" PRIx64 " (offset %u stride %u), input 0x%010" PRIx64 "+0x%" PRIx64
	    ", output 0x%010" PRIx64 "+0x%" PRIx64 " (commands at +8, trailer at +0x%x)\n",
	    n, r, tail, args, size(4), arg_off, arg_stride, in, size(8), out, size(12), 8u + n * r);
	std::vector<uint32_t> index(n);
	if (!LibKernel::Memory::TryReadBacking(in, index.data(), n * 4u)) {
		std::fprintf(stderr, "[bundle]   index table unreadable\n");
		return;
	}
	uint32_t live = 0, nonempty = 0;
	for (uint32_t i = 0; i < n; i++) {
		if (index[i] == 0xffffffffu) continue;
		live++;
		uint32_t       a[3]  = {0, 0, 0};
		const uint64_t at    = args + arg_off + uint64_t {arg_stride} * index[i];
		const bool     ok    = LibKernel::Memory::TryReadBacking(at, a, sizeof(a));
		const bool     dirty = cache.HasGpuDirtyBytes(at, sizeof(a));
		if (ok && a[0] != 0 && a[1] != 0 && a[2] != 0) nonempty++;
		if (i < 96u) {
			std::fprintf(
			    stderr, "[bundle]   record %u index %u args 0x%010" PRIx64 " = %ux%ux%u%s%s\n", i,
			    index[i], at, a[0], a[1], a[2], dirty ? " gpu-dirty" : "", ok ? "" : " unreadable");
		}
	}
	std::fprintf(
	    stderr,
	    "[bundle]   %u of %u records have an index, %u with non-empty arguments in guest memory\n",
	    live, n, nonempty);
	for (uint32_t i = 0; i < std::min(n, 2u); i++) {
		BundleHex(i == 0 ? "record0" : "record1", in + n * 4u + uint64_t {r} * i, r);
	}
	BundleHex("out-head", out, 8u + std::min(n * r, 192u));
	BundleHex("out-trailer", out + 8u + n * r, std::min(tail, 256u));
}

void RenderExecutor::DispatchDirect(uint64_t submit_id, CommandBuffer& buffer,
                                    uint32_t thread_group_x, uint32_t thread_group_y,
                                    uint32_t thread_group_z, uint32_t mode, bool async_compute) {
	KYTY_GPU_OP_SITE("dispatch");
	// KYTY_CP_GAP_STATS: this scope's phases (drawPrep/cpGaps.h).
	const CpGaps::Scope gap(CpGaps::Cat::DispatchOp);
	EXIT_IF(buffer.IsInvalid());
	m_context.GetCommandScheduler().PopReadyOperations();
	auto& ctx    = buffer.GetRegisters();
	auto& sh_ctx = buffer.GetShaders();

	buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::DispatchDirect), submit_id,
	                    thread_group_x, thread_group_y, thread_group_z, mode,
	                    sh_ctx.GetCs().cs_regs.data_addr);

	if (thread_group_x == 0 || thread_group_y == 0 || thread_group_z == 0) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 32) {
			LOGF("GraphicsRenderDispatchDirect: skipping zero-sized dispatch groups=%ux%ux%u "
			     "mode=0x%08" PRIx32 " shader=0x%016" PRIx64 "\n",
			     thread_group_x, thread_group_y, thread_group_z, mode,
			     sh_ctx.GetCs().cs_regs.data_addr);
		}
		return;
	}

	RenderLockGuard lock(m_context.GetMutex());
	// KYTY_DRAW_RUN: a dispatch ends the run of the draws before it (drawPrep/drawRun.h).
	BeginDrawRun();
	if (sh_ctx.GetCs().cs_regs.data_addr == 0) {
		LOGF("GraphicsRenderDispatchDirect: temporary: ignoring dispatch with null CS shader, "
		     "groups=%ux%ux%u mode=%u\n",
		     thread_group_x, thread_group_y, thread_group_z, mode);
		return;
	}

	if (sh_ctx.GetCs().cs_regs.data_addr == 0) {
		return;
	}

	constexpr uint32_t DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS = 1u << 5u;
	constexpr uint32_t DISPATCH_INITIATOR_BASE_BITS             = 0x41u;
	constexpr uint32_t DISPATCH_INITIATOR_MODIFIER_BITS         = 0xa038u;
	constexpr uint32_t DISPATCH_INITIATOR_KNOWN_MASK =
	    DISPATCH_INITIATOR_BASE_BITS | DISPATCH_INITIATOR_MODIFIER_BITS;

	const uint32_t unknown_mode_bits = mode & ~DISPATCH_INITIATOR_KNOWN_MASK;
	if (unknown_mode_bits != 0) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 32) {
			LOGF("GraphicsRenderDispatchDirect: unknown dispatch initiator bits "
			     "mode=0x%08" PRIx32 " unknown=0x%08" PRIx32 " shader=0x%016" PRIx64
			     " groups=%ux%ux%u\n",
			     mode, unknown_mode_bits, sh_ctx.GetCs().cs_regs.data_addr, thread_group_x,
			     thread_group_y, thread_group_z);
		}
	}

	const auto& cs_regs = sh_ctx.GetCs();
	const auto& sh_regs = ctx.GetShaderRegisters();

	ShaderComputeInputInfo input_info {};
	const bool use_thread_dimensions = (mode & DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) != 0;
	DrawPrep::MakeComputeInputInfo(cs_regs, {thread_group_x, thread_group_y, thread_group_z, mode},
	                               input_info);
	// After MakeComputeInputInfo (it resets the info). Part of the program key (GETREG HW_ID2).
	input_info.async_compute = async_compute;
	CpGaps::Phase(CpGaps::Cat::DispatchProgram);
	ShaderProgram compute_program;
	// KYTY_DRAW_PREP_DISPATCH (drawPrep.h): the program a worker prepared from this dispatch's
	// register snapshot, when its certificate holds; otherwise the serial lookup. Workers prepare
	// graphics-queue dispatches only (MakeComputeInputInfo leaves async_compute false).
	auto* const prepared = std::exchange(m_prepared_dispatch, nullptr);
	if (prepared != nullptr && !async_compute && DrawPrep::ValidateDispatch(*prepared, input_info)) {
		if (DrawPrep::VerifyMode() != 0) {
			// The serial lookup on copies; it only reads memory the certificate proved clean.
			auto serial_prep = std::make_unique<PipelineCache::StagePrep>();
			auto serial_info = input_info;
			const auto serial_program = m_context.GetPipelineCache().GetComputeProgram(
			    cs_regs, sh_regs, serial_info, *serial_prep);
			(void)DrawPrep::VerifyDispatch(*prepared, serial_program, serial_info, *serial_prep);
		}
		std::swap(m_compute_prep, prepared->compute_prep);
		input_info                 = prepared->compute_info;
		input_info.stage.resources = &m_compute_prep.resources;
		compute_program            = prepared->compute_program;
	} else {
		compute_program = m_context.GetPipelineCache().GetComputeProgram(cs_regs, sh_regs, input_info,
		                                                                 m_compute_prep);
	}
	CpGaps::Phase(CpGaps::Cat::DispatchOp);
	if (!compute_program) {
		if (BundleLogEnabled()) [[unlikely]] {
			BundleLogNoProgram("direct", cs_regs, m_context.GetBufferCache(), thread_group_x,
			                   thread_group_y, thread_group_z, 0);
		}
		// Temporary until RT is implemented.
		if (IndirectLogEnabled()) [[unlikely]] {
			static std::mutex                   drop_mutex;
			static std::unordered_set<uint64_t> drop_logged;
			const uint64_t                      va         = cs_regs.cs_regs.data_addr;
			bool                                first_drop = false;
			{
				std::scoped_lock drop_lock(drop_mutex);
				first_drop = drop_logged.insert(va).second;
			}
			if (first_drop) {
				std::fprintf(
				    stderr, "[nanite] CS dropped (%s): no program for shader va 0x%010" PRIx64 "\n",
				    "indirect", va);
			}
		}
		if (IndirectLogEnabled()) [[unlikely]] {
			static std::mutex                   drop_mutex;
			static std::unordered_set<uint64_t> drop_logged;
			const uint64_t                      va         = cs_regs.cs_regs.data_addr;
			bool                                first_drop = false;
			{
				std::scoped_lock drop_lock(drop_mutex);
				first_drop = drop_logged.insert(va).second;
			}
			if (first_drop) {
				std::fprintf(
				    stderr, "[nanite] CS dropped (%s): no program for shader va 0x%010" PRIx64 "\n",
				    "direct", va);
			}
		}
		return;
	}
	if (use_thread_dimensions) {
		input_info.dispatch_threads_num[0]    = thread_group_x;
		input_info.dispatch_threads_num[1]    = thread_group_y;
		input_info.dispatch_threads_num[2]    = thread_group_z;
	}

	const auto& program   = *input_info.stage.program;
	const auto& resources = *input_info.stage.resources;
	// A recorded metadata clear consumes the dispatch unless KYTY_COMPUTE_META_CLEAR_DISPATCH keeps
	// its guest writes (upstream 01ce97388).
	if (UpdateComputeMetaClears(input_info, buffer) && !ComputeMetaClearDispatchEnabled()) {
		ResetBindings();
		return;
	}
	if (TryConsumeComputeImageClear(input_info, buffer, thread_group_x, thread_group_y,
	                                thread_group_z, mode)) {
		ResetBindings();
		return;
	}
	// mfxlayer KYTY_CS_KO (metalfx.h; diagnostic, default off): knocked-out compute shaders.
	if (MetalFx::CsKnockoutEnabled() && MetalFx::CsKnockout(program.shader_hash)) [[unlikely]] {
		ResetBindings();
		return;
	}
	// rtimage KYTY_COMPUTE_IMAGE_COPY: a surface copy through two buffers as a native image copy.
	bool     copy_verify_kernel = false;
	uint32_t copy_slots[2]      = {0, 0};
	uint64_t copy_source = 0, copy_destination = 0, copy_size = 0;
	if (const auto copy_mode = ComputeImageCopy::GetMode();
	    copy_mode != ComputeImageCopy::Mode::Off) [[unlikely]] {
		if (const auto copy = ComputeImageCopy::Find(program.shader_hash);
		    copy &&
		    ResolveComputeBufferCopy(input_info, thread_group_x, thread_group_y, thread_group_z,
		                             mode, *copy, copy_source, copy_destination, copy_size)) {
			TilerTrace::Count("imgcopy.kernel-dispatch", 1, copy_size);
			if (copy_mode == ComputeImageCopy::Mode::VerifyKernel) {
				copy_verify_kernel = true;
				copy_slots[0]      = copy->source;
				copy_slots[1]      = copy->destination;
			} else if (m_context.GetTextureCache().CopyImageFromBuffers(
			               buffer, copy_source, copy_destination, copy_size,
			               copy_mode == ComputeImageCopy::Mode::VerifyImage)) {
				ResetBindings();
				return;
			}
		}
	}

	if (use_thread_dimensions) {
		const uint32_t old_x = thread_group_x;
		const uint32_t old_y = thread_group_y;
		const uint32_t old_z = thread_group_z;
		thread_group_x       = input_info.workgroup_counts[0];
		thread_group_y       = input_info.workgroup_counts[1];
		thread_group_z       = input_info.workgroup_counts[2];

		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 32) {
			LOGF("GraphicsRenderDispatchDirect: use-thread-dimensions %ux%ux%u / %ux%ux%u -> "
			     "groups %ux%ux%u\n",
			     old_x, old_y, old_z, std::max(cs_regs.cs_regs.num_thread_x, 1u),
			     std::max(cs_regs.cs_regs.num_thread_y, 1u),
			     std::max(cs_regs.cs_regs.num_thread_z, 1u), thread_group_x, thread_group_y,
			     thread_group_z);
		}
	}

	buffer.EndRendering();
	CpGaps::Phase(CpGaps::Cat::DispatchPipeline);
	// A software ray-tracing kernel whose pipeline is still being built, or any RT kernel after one
	// failed to build (rtSession.h), is skipped: the lighting it feeds lags or stays as it was.
	const bool rt_kernel    = input_info.stage.program->info.uses_bvh;
	const bool rt_load_hold = RtLoadHold(rt_kernel);
	auto*      pipeline_or_null =
	    rt_kernel && (RtBudgetHold(m_context) || rt_load_hold)
	        ? nullptr
	        : m_context.GetPipelineCache().TryGetComputePipeline(input_info, compute_program);
	if (pipeline_or_null == nullptr) {
		ResetBindings();
		return;
	}
	auto& pipeline = *pipeline_or_null;
	auto& bindings = m_compute_bindings;
	CpGaps::Phase(CpGaps::Cat::DispatchBind);
	BufferCache::SetWatchProgram(
	    input_info.stage.program != nullptr ? input_info.stage.program->shader_hash : 0,
	    input_info.async_compute ? 1 : 0);
	if (NanProbe::Get().any) [[unlikely]] {
		NanProbe::LastCs().store(program.shader_hash, std::memory_order_relaxed);
	}
	PrepareBindings(input_info.stage, bindings);
	if (program.bindings.dispatch_thread_dword != ShaderRecompiler::IR::PushData::NoStart) {
		// The requested thread extents initialize EXEC (upstream e4aca7a60).
		std::copy(std::begin(input_info.dispatch_threads_num),
		          std::end(input_info.dispatch_threads_num),
		          bindings.shader_data.begin() + program.bindings.dispatch_thread_dword);
	}
	// The native dispatch size bounds workgroup ids for write-range proofs.
	bindings.dispatch_groups     = {thread_group_x, thread_group_y, thread_group_z};
	bindings.has_dispatch_groups = true;
	FindBuffers(bindings);
	if (MetalFx::FindWide()) [[unlikely]] {
		MetalFx::FindLog("dispatch", bindings, m_context.GetTextureCache(), ImageId {}, 0);
	}
	// mfxlayer KYTY_METALFX_UE (metalfx.h, default off): UE's FSR3 upscaler slot.
	if (MetalFx::UeEnabled()) [[unlikely]] {
		if (const int ue = MetalFx::UeDispatch(buffer, bindings, m_context.GetTextureCache());
		    ue != 0) {
			if (ue == 2) {
				auto& ue_cache = m_context.GetTextureCache();
				ue_cache.CommitGpuWrite(ue_cache.GetImage(MetalFx::UeOutput()));
			}
			ResetBindings();
			return;
		}
	}
	// mfxlayer KYTY_METALFX_COMPUTE_REPLACE (metalfx.h, default off): MetalFX instead of the
	// dispatch.
	if (MetalFx::Requested() && MetalFx::ComputeReplace(input_info.stage.program->shader_hash))
	    [[unlikely]] {
		if (TryMetalFxCompute(buffer, bindings)) {
			ResetBindings();
			return;
		}
	}
	if (MetalFx::ResAuditEnabled()) [[unlikely]] {
		const uint32_t groups[3]  = {thread_group_x, thread_group_y, thread_group_z};
		const uint32_t threads[3] = {cs_regs.cs_regs.num_thread_x, cs_regs.cs_regs.num_thread_y,
		                             cs_regs.cs_regs.num_thread_z};
		MetalFx::ResAudit("dispatch", bindings, m_context.GetTextureCache(), ImageId {}, 0, groups,
		                  threads, cs_regs.cs_regs.data_addr);
	}
	CpGaps::Phase(CpGaps::Cat::DispatchBda);
	if (program.info.uses_dma) {
		NoteBdaUser("dispatch", static_cast<uint32_t>(program.stage), program.shader_hash,
		            program.info.bda_writes, 0, 0);
		m_context.PrepareBda();
	}
	if (program.info.uses_bvh) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::RtStubDispatches);
	}
	if (program.info.bda_writes) {
		// KYTY_BDA_WRITES: settled synchronously right after the dispatch below. The settle relies
		// on PrepareBda's pass above having uploaded every CPU-dirty page first.
		EXIT_IF(!program.info.uses_dma);
		m_context.GetBufferCache().PrepareBdaWrites();
	}
	CpGaps::Phase(CpGaps::Cat::DispatchRebind);
	RebindImages(bindings);
	if (PassTrace::Active()) [[unlikely]] {
		PreparedBindings* traced = &bindings;
		TracePass(m_context.GetTextureCache(), "dispatch", std::span<PreparedBindings* const>(&traced, 1), {});
	}
	BindComputeLds(input_info, bindings, input_info.workgroup_counts);
	PreparedBindings* flight_stage = &bindings;
	FlightRecDraw::Begin('X', std::span<PreparedBindings* const>(&flight_stage, 1),
	                     m_context.GetCommandScheduler().CurrentTick());
	m_rebind_epoch = m_context.GetBufferCache().UnregisterEpoch();
	RebindBuffers(bindings);
	// KYTY_REBIND_AFTER_JOIN: a later binding's lookup joined a buffer an earlier one resolved.
	if (RebindAfterJoinEnabled()) {
		auto& cache = m_context.GetBufferCache();
		for (uint32_t pass = 0; pass < 4 && cache.UnregisterEpoch() != m_rebind_epoch; pass++) {
			NoteRebindAfterJoin("dispatch", nullptr);
			m_rebind_epoch = cache.UnregisterEpoch();
			RebindBuffers(bindings);
		}
	}
	FlightRecDraw::End(m_context.GetTextureCache(),
	                   std::span<PreparedBindings* const>(&flight_stage, 1), {});
	InputProfileDraw::Note(m_context.GetBufferCache(),
	                       std::span<PreparedBindings* const>(&flight_stage, 1));
	CpGaps::Phase(CpGaps::Cat::DispatchEmit);

	// nps5 nanite probes (nanProbe.h): identity log, after-draw and pre-dispatch image copies.
	bool nan_probe_post = false;
	if (NanProbe::Get().any) [[unlikely]] {
		NanProbeImpl::Poll(m_context);
		if (NanProbe::TickArm()) {
			auto& cache = m_context.GetTextureCache();
			if (NanProbe::IdHash(program.shader_hash)) {
				for (uint32_t i = 0; i < bindings.images.size(); i++) {
					if (NanProbe::TakeLine()) {
						NanProbeImpl::LogIdentity(
						    program.info.images[i].written ? "cs-w" : "cs-r", program.shader_hash,
						    i, cache, bindings.images[i].image_id, &bindings.images[i].desc);
					}
				}
			}
			auto& draw = NanProbe::Draw();
			if (draw.pending && !draw.done) {
				draw.done = true;
				std::fprintf(stderr,
				             "[nanprobe] after-draw probe of 0x%016" PRIx64
				             " at dispatch of CS 0x%016" PRIx64 "\n",
				             draw.hash, program.shader_hash);
				for (size_t i = 0; i < draw.index.size(); i++) {
					NanProbeImpl::Capture(m_context, buffer, ImageId {draw.index[i], draw.gen[i]},
					                      "afterdraw", draw.hash, static_cast<uint32_t>(i));
				}
				// sarosclay: the draw's sampled textures too (level 0), as "drawtex" (BC raw
				// blocks).
				for (size_t i = 0; i < draw.tex_index.size(); i++) {
					NanProbeImpl::Capture(m_context, buffer,
					                      ImageId {draw.tex_index[i], draw.tex_gen[i]}, "drawtex",
					                      draw.hash, static_cast<uint32_t>(i));
				}
			}
			static std::vector<uint64_t> cs_done_hashes;
			const auto&                  probe_hashes = NanProbe::Get().cs_hashes;
			if (std::find(probe_hashes.begin(), probe_hashes.end(), program.shader_hash) !=
			        probe_hashes.end() &&
			    std::find(cs_done_hashes.begin(), cs_done_hashes.end(), program.shader_hash) ==
			        cs_done_hashes.end()) {
				cs_done_hashes.push_back(program.shader_hash);
				nan_probe_post = true;
				for (uint32_t i = 0; i < bindings.images.size(); i++) {
					NanProbeImpl::Capture(m_context, buffer, bindings.images[i].image_id, "pre",
					                      program.shader_hash, i);
				}
			}
			SarosBufferStats(program, resources, probe_hashes);
		}
	}
	PreparedBindings* descriptor_stage = &bindings;
	// Emission safe point (KYTY_CP_RECORDER, render.h): no native handle from the preparation above
	// is alive. Binding commits and the barrier requests below record state commands only (image
	// transitions and dependencies are batched); the dispatch goes through Sink(), which records
	// them first as one barrier, exactly as Handle() did.
	buffer.BeginEmission();
	CommitBindings(buffer, vk::PipelineBindPoint::eCompute, pipeline,
	               std::span {&descriptor_stage, 1u});
	bool has_storage_writes = bindings.shared_memory.buffer != nullptr ||
	                          HasShaderBufferWrites(input_info.stage) || program.info.bda_writes;
	has_storage_writes =
	    std::any_of(program.info.images.begin(), program.info.images.end(),
	                [](const auto& image) {
		                return image.written &&
		                       image.resource_class ==
		                           ShaderRecompiler::IR::ImageResourceClass::Storage;
	                }) ||
	    has_storage_writes;
	// KYTY_DISPATCH_OVERLAP (render.h): nothing happened since the previous dispatch, so the guest
	// did not order the two; its pending ShaderAccess request is dropped and this dispatch skips
	// its own hazard request. Emulated LDS (one device buffer) and BDA-write settles never overlap.
	const bool no_overlap      = bindings.shared_memory.buffer != nullptr || program.info.bda_writes;
	const bool overlap_enabled = DispatchOverlapEnabled();
	const bool overlap         = buffer.TakeDispatchOverlap(no_overlap, overlap_enabled);
	CountDispatchOverlap(overlap, overlap_enabled);
	if (has_storage_writes && !(overlap && overlap_enabled)) {
		// A host fence used to serialize every dispatch. Preserve its read-before-write ordering
		// while allowing the queue to execute asynchronously.
		ShaderWriteHazardBarrier(buffer, vk::PipelineStageFlagBits::eComputeShader);
	}
	// memtrk: KYTY_CS_BIND_LOG=1 prints, once per compute shader, its grid and every image
	// (address, guest format, extent, r/w) and every written buffer (address, bytes): which pass
	// reads or writes which surface (SAROS: who reads the Nanite visibility buffer, who writes
	// scene depth).
	static const bool cs_bind_log = [] {
		const auto* value = std::getenv("KYTY_CS_BIND_LOG");
		return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
	}();
	if (cs_bind_log) {
		static std::mutex                   bind_mutex;
		static std::unordered_set<uint64_t> bind_logged;
		bool                                first = false;
		{
			std::scoped_lock lock(bind_mutex);
			first = bind_logged.insert(program.shader_hash).second;
		}
		if (first) {
			std::string line;
			char        part[160];
			for (uint32_t i = 0; i < program.info.images.size(); i++) {
				const auto& image = program.info.images[i];
				const auto  r = DecodeNativeDescriptor<ShaderTextureResource>(resources.images[i]);
				std::snprintf(part, sizeof(part), " img%u=0x%010" PRIx64 ":f%u:%ux%u:%s%s", i,
				              r.Base40(), static_cast<uint32_t>(r.Format()),
				              static_cast<uint32_t>(r.Width5()) + 1u,
				              static_cast<uint32_t>(r.Height5()) + 1u, image.read ? "r" : "",
				              image.written ? "w" : "");
				line += part;
				// saroscons: with KYTY_SAROS_HANDSHADERS set, also the view fields of each T#
				// (type, base array+depth, base level, max mip / last level) to tell which layer a
				// consumer reads.
				static const bool saros_view = [] {
					const auto* v = std::getenv("KYTY_SAROS_HANDSHADERS");
					return v != nullptr && v[0] != '\0' && std::strcmp(v, "0") != 0;
				}();
				if (saros_view) {
					std::snprintf(
					    part, sizeof(part), "{t%u a%u+%u l%u-%u}", static_cast<uint32_t>(r.Type()),
					    static_cast<uint32_t>(r.BaseArray5()), static_cast<uint32_t>(r.Depth()),
					    static_cast<uint32_t>(r.BaseLevel()),
					    static_cast<uint32_t>(image.r128 ? r.LastLevel() : r.MaxMip()));
					line += part;
				}
			}
			for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
				if (!program.info.buffers[i].written) continue;
				const auto r = DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[i]);
				std::snprintf(part, sizeof(part), " wbuf%u=0x%010" PRIx64 "+0x%" PRIx64, i,
				              r.Base48(), static_cast<uint64_t>(r.GetSize()));
				line += part;
			}
			std::fprintf(stderr, "[memtrk] CS bind 0x%016" PRIx64 " grid %ux%ux%u mode 0x%x:%s\n",
			             program.shader_hash, thread_group_x, thread_group_y, thread_group_z, mode,
			             line.c_str());
		}
	}
	if (Config::GraphicsDebugDumpEnabled()) {
		const auto sampled_images = std::count_if(
		    program.info.images.begin(), program.info.images.end(), [](const auto& image) {
			    return image.resource_class == ShaderRecompiler::IR::ImageResourceClass::Sampled;
		    });
		const uint32_t frame_num = static_cast<uint32_t>(m_context.GetGpu().GetFrameNum());
		LOGF("GraphicsRenderDispatchDirect: frame=%u shader=0x%016" PRIx64
		     " hash=0x%016" PRIx64 " tick=%" PRIu64 " completed_tick=%" PRIu64
		     " groups=%ux%ux%u mode=0x%08" PRIx32 " local=%ux%ux%u "
		     "buffers=%zu textures=%zu sampled=%zu storage=%zu samplers=%zu push=%u\n",
		     frame_num, sh_ctx.GetCs().cs_regs.data_addr, program.shader_hash,
		     m_context.GetCommandScheduler().CurrentTick(),
		     m_context.GetCommandScheduler().GetMasterSemaphore().KnownGpuTick(),
		     thread_group_x, thread_group_y, thread_group_z, mode,
		     input_info.threads_num[0], input_info.threads_num[1],
		     input_info.threads_num[2], program.info.buffers.size(), program.info.images.size(),
		     sampled_images, program.info.images.size() - sampled_images,
		     program.info.samplers.size(),
		     program.bindings.UsesPushData()
		         ? static_cast<uint32_t>(sizeof(ShaderRecompiler::IR::PushData))
		         : 0u);
		for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
			const auto& buffer = program.info.buffers[i];
			const auto  r      = DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[i]);
			LOGF("  CS buffer[%u]: source=%u usage=%s addr=0x%012" PRIx64
			     " stride=%u records=%u format=%u\n",
			     i, buffer.source, buffer.written ? "read-write" : "read-only", r.Base48(),
			     r.Stride(), r.NumRecords(), r.RawFormat());
		}
		for (uint32_t i = 0; i < program.info.images.size(); i++) {
			const auto& image = program.info.images[i];
			const auto  r     = DecodeNativeDescriptor<ShaderTextureResource>(resources.images[i]);
			LOGF("  CS texture[%u]: source=%u usage=%s sampled=%s addr=0x%010" PRIx64
			     " type=%u fmt=%u extent=%ux%u depth=%u levels=%u tile=%u\n",
			     i, image.source, image.written ? "read-write" : "read-only",
			     image.resource_class == ShaderRecompiler::IR::ImageResourceClass::Sampled
			         ? "true"
			         : "false",
			     r.Base40(), static_cast<uint32_t>(r.Type()), static_cast<uint32_t>(r.Format()),
			     static_cast<uint32_t>(r.Width5()) + 1u, static_cast<uint32_t>(r.Height5()) + 1u,
			     static_cast<uint32_t>(r.Depth()) + 1u,
			     r.Type() == Prospero::ImageType::kColor2DMsaa ||
			             r.Type() == Prospero::ImageType::kColor2DMsaaArray
			         ? 1u
			         : static_cast<uint32_t>(image.r128 ? r.LastLevel() : r.MaxMip()) + 1u,
			     static_cast<uint32_t>(r.TileMode()));
		}
		for (uint32_t i = 0; i < program.info.samplers.size(); i++) {
			const auto r = DecodeNativeDescriptor<ShaderSamplerResource>(
			    resources.samplers[program.info.samplers[i].snapshot_index]);
			LOGF("  CS sampler[%u]: source=%u clamp=%u/%u/%u filter=%u/%u/%u mip=%u "
			     "lod=%u-%u bias=%d\n",
			     i, program.info.samplers[i].source, static_cast<uint32_t>(r.ClampX()),
			     static_cast<uint32_t>(r.ClampY()), static_cast<uint32_t>(r.ClampZ()),
			     static_cast<uint32_t>(r.XyMagFilter()), static_cast<uint32_t>(r.XyMinFilter()),
			     static_cast<uint32_t>(r.ZFilter()), static_cast<uint32_t>(r.MipFilter()),
			     static_cast<uint32_t>(r.MinLod()), static_cast<uint32_t>(r.MaxLod()),
			     static_cast<int32_t>(r.LodBias()));
		}
	}
	{ // rtimage KYTY_RTIMAGE_PROBE=1 (diagnostic): dispatch shape of shaders with a formatted or
	  // written buffer, first 3 distinct shapes per shader.
		static const bool probe = [] {
			const char* value = std::getenv("KYTY_RTIMAGE_PROBE");
			return value != nullptr && value[0] == '1';
		}();
		if (probe) [[unlikely]] {
			bool any = false;
			for (const auto& resource: program.info.buffers) {
				any = any || resource.formatted || resource.written;
			}
			if (any) {
				static std::mutex                                          mutex;
				static std::unordered_map<uint64_t, std::set<std::string>> seen;
				auto                                                       text = fmt::format(
				    "groups={}x{}x{} threads={}x{}x{} mode={:#x} tdim={} fill={} nbuf={} nimg={}",
				    thread_group_x, thread_group_y, thread_group_z, input_info.threads_num[0],
				    input_info.threads_num[1], input_info.threads_num[2], mode,
				    input_info.dispatch_thread_dimensions ? 1 : 0,
				    static_cast<uint32_t>(input_info.stage.resources->uniform_fill.kind),
				    program.info.buffers.size(), program.info.images.size());
				for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
					const auto d = DecodeNativeDescriptor<ShaderBufferResource>(
					    input_info.stage.resources->buffers[i]);
					text += fmt::format(
					    " b{}[{}{}{} f{} s{} n{}]", i, program.info.buffers[i].read ? "R" : "",
					    program.info.buffers[i].written ? "W" : "",
					    program.info.buffers[i].formatted ? "F" : "",
					    static_cast<uint32_t>(d.Format()), d.Stride(), d.NumRecords());
				}
				std::scoped_lock lock {mutex};
				auto&            shapes = seen[program.shader_hash];
				if (shapes.size() < 3 && shapes.insert(text).second) {
					std::fprintf(stderr, "[rtimage] dispatch sh=%016" PRIx64 " %s\n",
					             program.shader_hash, text.c_str());
				}
			}
		}
	}
	buffer.BindPipeline(vk::PipelineBindPoint::eCompute, pipeline.pipeline);
	constexpr uint32_t DispatchInitiatorOrderedAppendEnable = 1u << 3u;
	const uint64_t     group_threads = uint64_t {std::max(input_info.threads_num[0], 1u)} *
	                               std::max(input_info.threads_num[1], 1u) *
	                               std::max(input_info.threads_num[2], 1u);
	if (OrderedAppendSplitEnabled() && (mode & DispatchInitiatorOrderedAppendEnable) != 0 &&
	    (ShaderRecompiler::IR::FindBinding(program.bindings, ShaderRecompiler::IR::DescriptorBindingKind::Gds) != nullptr) && group_threads <= input_info.wave_size && !input_info.lds_storage) {
		static std::atomic_bool logged = false;
		if (!logged.exchange(true, std::memory_order_relaxed)) {
			std::printf("KYTY_ORDERED_APPEND: shader 0x%016" PRIx64
			            " runs %ux%ux%u ordered-append workgroups one by one\n",
			            program.shader_hash, thread_group_x, thread_group_y, thread_group_z);
			std::fflush(stdout);
		}
		for (uint32_t z = 0; z < thread_group_z; ++z) {
			for (uint32_t y = 0; y < thread_group_y; ++y) {
				for (uint32_t x = 0; x < thread_group_x; ++x) {
					if ((x | y | z) != 0u) {
						ShaderAccessBarrier(buffer, vk::PipelineStageFlagBits::eComputeShader);
					}
					buffer.Sink().dispatchBase(x, y, z, 1, 1, 1);
				}
			}
		}
	} else {
		// nps5 memtrk: KYTY_CS_MAX_GROUPS=<hex hash>:<n>[,<hex hash>:<n>...] (opt-in, SAROS). A
		// persistent-threads kernel (UE5 culling: workgroups pull work through a global atomic and
		// spin on flags other workgroups publish) assumes all its workgroups run at once; on Metal
		// workgroups that are not yet resident never publish, and the resident ones spin to the
		// loop cap. The listed programs' 1D grids are capped at n workgroups (enough to be
		// co-resident). The first dispatch of each listed program is logged with its grid.
		struct CsLimit {
			uint64_t hash;
			uint32_t groups;
		};
		static const std::vector<CsLimit> limits = [] {
			std::vector<CsLimit> list;
			const char*          v = std::getenv("KYTY_CS_MAX_GROUPS");
			while (v != nullptr && *v != '\0') {
				char*          end  = nullptr;
				const uint64_t hash = std::strtoull(v, &end, 16);
				if (end == nullptr || *end != ':') break;
				const auto n = static_cast<uint32_t>(std::strtoul(end + 1, &end, 10));
				list.push_back({hash, n});
				v = (end != nullptr && *end == ',') ? end + 1 : nullptr;
			}
			return list;
		}();
		uint32_t gx = thread_group_x;
		for (const auto& limit: limits) {
			if (limit.hash == program.shader_hash) {
				static std::mutex                   mutex;
				static std::unordered_set<uint64_t> logged;
				bool                                first = false;
				{
					std::scoped_lock lock(mutex);
					first = logged.insert(program.shader_hash).second;
				}
				if (first) {
					std::fprintf(stderr,
					             "[memtrk] KYTY_CS_MAX_GROUPS: CS 0x%016" PRIx64
					             " grid %ux%ux%u (group %llu threads)%s%u\n",
					             program.shader_hash, thread_group_x, thread_group_y,
					             thread_group_z, static_cast<unsigned long long>(group_threads),
					             thread_group_y == 1 && thread_group_z == 1
					                 ? " -> x capped at "
					                 : ", not 1D: left at x=",
					             thread_group_y == 1 && thread_group_z == 1
					                 ? std::min(thread_group_x, limit.groups)
					                 : thread_group_x);
				}
				if (thread_group_y == 1 && thread_group_z == 1 && limit.groups != 0) {
					gx = std::min(gx, limit.groups);
				}
			}
		}
		if (static_cast<uint64_t>(gx) * thread_group_y * thread_group_z > 4194304u) [[unlikely]] {
			// nps5 memtrk: name the program behind KK's dispatch-clamp line (SAROS: 66977792x1x1
			// since r55).
			static std::mutex                   clamp_mutex;
			static std::unordered_set<uint64_t> clamp_logged;
			bool                                first = false;
			{
				std::scoped_lock lock(clamp_mutex);
				first = clamp_logged.insert(program.shader_hash).second;
			}
			if (first) {
				std::fprintf(stderr,
				             "[memtrk] oversized direct dispatch: CS 0x%016" PRIx64
				             " grid %ux%ux%u (group %llu threads, mode 0x%x, "
				             "async %d)\n",
				             program.shader_hash, gx, thread_group_y, thread_group_z,
				             static_cast<unsigned long long>(group_threads), mode,
				             async_compute ? 1 : 0);
			}
		}
		buffer.Sink().dispatch(gx, thread_group_y, thread_group_z);
		m_context.GetBufferCache().FlagReadbackAfterDispatch(program.shader_hash, gx,
		                                                     thread_group_y, thread_group_z);
	}
	Common::DebugCounters::Add(Common::DebugCounters::Counter::Dispatches);

	// The removed host fence also ordered read-only dispatches before later writers.
	ShaderAccessBarrier(buffer, vk::PipelineStageFlagBits::eComputeShader);
	buffer.NoteGuestDispatch(no_overlap);
	if (copy_verify_kernel) [[unlikely]] {
		// rtimage KYTY_COMPUTE_IMAGE_COPY=verify-kernel: the guest shader ran; the destination
		// range must now hold the source range's bytes.
		static std::atomic<uint32_t> ordinal {0};
		const auto                   n = ordinal.fetch_add(1, std::memory_order_relaxed);
		if ((n < 64 || n % 16 == 0) && bindings.buffers.size() == 2) {
			const auto& layout = program.bindings;
			const auto  start  = [&](uint32_t slot) {
				const auto dword = layout.memory_offset_dword + slot / 4u;
				return bindings.buffers[slot].offset +
				       ((bindings.shader_data[dword] >> ((slot % 4u) * 8u)) & 0xffu);
			};
			m_context.GetTextureCache().VerifyBufferRanges(
			    "imgcopy-kernel", copy_destination, bindings.buffers[copy_slots[0]].buffer,
			    start(copy_slots[0]), bindings.buffers[copy_slots[1]].buffer, start(copy_slots[1]),
			    copy_size);
		}
	}
	if (nan_probe_post) [[unlikely]] {
		for (uint32_t i = 0; i < bindings.images.size(); i++) {
			if (program.info.images[i].written) {
				NanProbeImpl::Capture(m_context, buffer, bindings.images[i].image_id, "post",
				                      program.shader_hash, i);
			}
		}
	}
	if (program.info.bda_writes) {
		m_context.GetBufferCache().SettleBdaWrites(program.shader_hash);
	}
	// A proven uniform buffer fill leaves a known value (e.g. DCC fast-clear codes). Recording
	// it lets consumers skip reading the range back while nothing else writes it.
	{
		ShaderBufferResource known_descriptor;
		uint32_t             known_value = 0;
		uint64_t             known_size  = 0;
		const auto groups_x = use_thread_dimensions ? input_info.dispatch_threads_num[0]
		                                           : thread_group_x;
		if (ResolveComputeBufferFill(input_info, groups_x,
		                             use_thread_dimensions ? input_info.dispatch_threads_num[1]
		                                                   : thread_group_y,
		                             use_thread_dimensions ? input_info.dispatch_threads_num[2]
		                                                   : thread_group_z,
		                             mode, known_descriptor, known_value, known_size)) {
			m_context.GetBufferCache().RecordKnownFill(known_descriptor.Base48(), known_size,
			                                           known_value);
		}
	}
	// Observe only dispatches that actually reached the native path. This reuses the
	// already materialized descriptor/value proof and performs no guest-memory reads.
	if (Profiler::DetailedEnabled() && tracy::ProfilerAvailable() && TracyIsConnected) {
		ShaderBufferResource fill_descriptor;
		uint32_t fill_value = 0;
		uint64_t fill_size = 0;
		const auto groups_x = use_thread_dimensions ? input_info.dispatch_threads_num[0]
		                                           : thread_group_x;
		const auto groups_y = use_thread_dimensions ? input_info.dispatch_threads_num[1]
		                                           : thread_group_y;
		const auto groups_z = use_thread_dimensions ? input_info.dispatch_threads_num[2]
		                                           : thread_group_z;
		if (ResolveComputeBufferFill(input_info, groups_x, groups_y, groups_z, mode,
		                             fill_descriptor, fill_value, fill_size)) {
			KYTY_PROFILER_DETAIL_BLOCK("Compute::UniformFillEligible");
			static std::atomic<uint32_t> fill_count {0};
			constexpr uint32_t MaxFillMessages = 4096;
			const auto ordinal = fill_count.fetch_add(1, std::memory_order_relaxed);
			if (ordinal < MaxFillMessages) {
				char text[512];
				const uint32_t byte = fill_value & 0xffu;
				const auto written = std::snprintf(
				    text, sizeof(text),
				    "COMPUTE_UNIFORM_FILL id=%u shader=0x%" PRIx64 " address=0x%" PRIx64
				    " bytes=%" PRIu64 " value=0x%08x byte_uniform=%u tick=%" PRIu64,
				    ordinal + 1u, program.shader_hash, fill_descriptor.Base48(), fill_size,
				    fill_value, fill_value == byte * 0x01010101u ? 1u : 0u,
				    m_context.GetCommandScheduler().CurrentTick());
				if (written > 0 && static_cast<size_t>(written) < sizeof(text)) {
					TracyMessageS(text, static_cast<size_t>(written), 0);
				}
			} else if (ordinal == MaxFillMessages) {
				TracyMessageLS("COMPUTE_UNIFORM_FILL_LIMIT messages=4096", 0);
			}
		}
	}
	// nps5 KYTY_METALFX (metalfx.h): a compute pass that scales the scene to the output size.
	if (MetalFx::Requested() && !MetalFx::ComputeReplace(program.shader_hash)) [[unlikely]] {
		TryMetalFxCompute(buffer, bindings);
	}
	ResetBindings();
}

// nps5 nanite: KYTY_INDIRECT_LOG=1 prints, once per compute shader dispatched indirectly, its
// images (address, guest format, extent, r/w), shader address, argument address and the arguments
// as guest memory holds them (and whether a GPU write to them is pending); and the first 64
// indirect dispatches KYTY_SKIP_EMPTY_INDIRECT drops. SAROS: who writes the G-buffer for Nanite
// pixels.
static bool IndirectLogEnabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_INDIRECT_LOG");
		return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
	}();
	return enabled;
}

void RenderExecutor::DispatchIndirect(uint64_t submit_id, CommandBuffer& buffer,
                                      uint64_t args_addr, uint32_t mode, bool async_compute) {
	KYTY_GPU_OP_SITE("dispatch.indirect");
	g_overlap_indirect.fetch_add(1, std::memory_order_relaxed);
	const CpGaps::Scope gap(CpGaps::Cat::DispatchOp);
	EXIT_IF(buffer.IsInvalid() || args_addr == 0 || (args_addr & 3u) != 0 ||
	        (mode & Pm4::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) != 0);
	m_context.GetCommandScheduler().PopReadyOperations();
	buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::DispatchIndirect), submit_id,
	                    static_cast<uint32_t>(args_addr), static_cast<uint32_t>(args_addr >> 32u),
	                    0, mode, buffer.GetShaders().GetCs().cs_regs.data_addr);
	RenderLockGuard lock(m_context.GetMutex());
	// KYTY_DRAW_RUN: a dispatch ends the run of the draws before it (drawPrep/drawRun.h).
	BeginDrawRun();
	const auto& cs_regs = buffer.GetShaders().GetCs();
	if (cs_regs.cs_regs.data_addr == 0) {
		NaniteShadeStats::NoteIndirect(args_addr, NaniteShadeStats::Outcome::NoShader, 0);
		return;
	}
	ShaderComputeInputInfo input_info {};
	input_info.async_compute = async_compute;
	CpGaps::Phase(CpGaps::Cat::DispatchProgram);
	ShaderProgram compute_program;
	// KYTY_DRAW_PREP_DISPATCH (drawPrep.h): as in DispatchDirect, from no workgroup counts.
	auto* const prepared = std::exchange(m_prepared_dispatch, nullptr);
	// KYTY_SKIP_EMPTY_INDIRECT (live, default on; upstream be5915c24): arguments no GPU write is
	// pending for, read from guest memory, with a zero group count: the dispatch does nothing.
	if (g_skip_empty_indirect.Get() != 0) {
		vk::DispatchIndirectCommand args {};
		if (!m_context.GetBufferCache().HasGpuDirtyBytes(args_addr, sizeof(args)) &&
		    LibKernel::Memory::TryReadBacking(args_addr, &args, sizeof(args)) &&
		    (args.x == 0 || args.y == 0 || args.z == 0)) {
			g_overlap_empty_indirect.fetch_add(1, std::memory_order_relaxed);
			if (IndirectLogEnabled()) [[unlikely]] {
				static std::atomic<uint32_t> skips {0};
				if (skips.fetch_add(1, std::memory_order_relaxed) < 64) {
					std::fprintf(stderr,
					             "[nanite] CS-ind skip empty: shader va 0x%010" PRIx64
					             " args 0x%010" PRIx64 " = %ux%ux%u\n",
					             cs_regs.cs_regs.data_addr, args_addr, args.x, args.y, args.z);
				}
			}
			NaniteShadeStats::NoteIndirect(args_addr, NaniteShadeStats::Outcome::SkipEmpty, 0);
			return;
		}
	}
	if (prepared != nullptr && !async_compute && DrawPrep::ValidateDispatch(*prepared, input_info)) {
		if (DrawPrep::VerifyMode() != 0) {
			auto serial_prep = std::make_unique<PipelineCache::StagePrep>();
			auto serial_info = input_info;
			const auto serial_program = m_context.GetPipelineCache().GetComputeProgram(
			    cs_regs, buffer.GetRegisters().GetShaderRegisters(), serial_info, *serial_prep);
			(void)DrawPrep::VerifyDispatch(*prepared, serial_program, serial_info, *serial_prep);
		}
		std::swap(m_compute_prep, prepared->compute_prep);
		input_info                 = prepared->compute_info;
		input_info.stage.resources = &m_compute_prep.resources;
		compute_program            = prepared->compute_program;
	} else {
		compute_program = m_context.GetPipelineCache().GetComputeProgram(
		    cs_regs, buffer.GetRegisters().GetShaderRegisters(), input_info, m_compute_prep);
	}
	CpGaps::Phase(CpGaps::Cat::DispatchPipeline);
	if (!compute_program) {
		if (BundleLogEnabled()) [[unlikely]] {
			BundleLogNoProgram("indirect", cs_regs, m_context.GetBufferCache(), 0, 0, 0, args_addr);
		}
		// sarosbundle: this return was silent (the "CS dropped" prints live in DispatchDirect
		// only).
		if (IndirectLogEnabled()) [[unlikely]] {
			static std::mutex                   drop_mutex;
			static std::unordered_set<uint64_t> drop_logged;
			const uint64_t                      va         = cs_regs.cs_regs.data_addr;
			bool                                first_drop = false;
			{
				std::scoped_lock drop_lock(drop_mutex);
				first_drop = drop_logged.insert(va).second;
			}
			if (first_drop) {
				std::fprintf(
				    stderr, "[nanite] CS dropped (%s): no program for shader va 0x%010" PRIx64 "\n",
				    "indirect-dispatch", va);
			}
		}
		NaniteShadeStats::NoteIndirect(args_addr, NaniteShadeStats::Outcome::NoProgram, 0);
		// Temporary until RT is implemented.
		return;
	}
	if (MetalFx::CsKnockoutEnabled() && MetalFx::CsKnockout(input_info.stage.program->shader_hash))
	    [[unlikely]] {
		ResetBindings();
		return;
	}
	buffer.EndRendering();
	const bool rt_kernel    = input_info.stage.program->info.uses_bvh;
	const bool rt_load_hold = RtLoadHold(rt_kernel);
	auto*      pipeline_or_null =
	    rt_kernel && (RtBudgetHold(m_context) || rt_load_hold)
	        ? nullptr
	        : m_context.GetPipelineCache().TryGetComputePipeline(input_info, compute_program);
	if (pipeline_or_null == nullptr) {
		NaniteShadeStats::NoteIndirect(args_addr, NaniteShadeStats::Outcome::NotReady, 0);
		ResetBindings();
		return;
	}
	auto& pipeline = *pipeline_or_null;
	auto& bindings = m_compute_bindings;
	CpGaps::Phase(CpGaps::Cat::DispatchBind);
	BufferCache::SetWatchProgram(
	    input_info.stage.program != nullptr ? input_info.stage.program->shader_hash : 0,
	    input_info.async_compute ? 1 : 0);
	PrepareBindings(input_info.stage, bindings);
	// GPU-produced dispatch arguments: workgroup ids stay unbounded for write-range proofs.
	bindings.has_dispatch_groups = false;
	FindBuffers(bindings);
	if (MetalFx::FindWide()) [[unlikely]] {
		MetalFx::FindLog("dispatch", bindings, m_context.GetTextureCache(), ImageId {}, 0);
	}
	// mfxlayer KYTY_METALFX_UE (metalfx.h, default off): UE's FSR3 upscaler slot.
	if (MetalFx::UeEnabled()) [[unlikely]] {
		if (const int ue = MetalFx::UeDispatch(buffer, bindings, m_context.GetTextureCache());
		    ue != 0) {
			if (ue == 2) {
				auto& ue_cache = m_context.GetTextureCache();
				ue_cache.CommitGpuWrite(ue_cache.GetImage(MetalFx::UeOutput()));
			}
			ResetBindings();
			return;
		}
	}
	// mfxlayer KYTY_METALFX_COMPUTE_REPLACE (metalfx.h, default off): MetalFX instead of the
	// dispatch.
	if (MetalFx::Requested() && MetalFx::ComputeReplace(input_info.stage.program->shader_hash))
	    [[unlikely]] {
		if (TryMetalFxCompute(buffer, bindings)) {
			ResetBindings();
			return;
		}
	}
	if (MetalFx::ResAuditEnabled()) [[unlikely]] {
		MetalFx::ResAudit("dispatch-indirect", bindings, m_context.GetTextureCache(), ImageId {}, 0,
		                  nullptr, nullptr, cs_regs.cs_regs.data_addr);
	}
	CpGaps::Phase(CpGaps::Cat::DispatchBda);
	const auto& program = *input_info.stage.program;
	if (program.info.uses_dma) {
		NoteBdaUser("dispatch", static_cast<uint32_t>(program.stage), program.shader_hash,
		            program.info.bda_writes, 0, 0);
		m_context.PrepareBda();
	}
	if (program.info.uses_bvh) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::RtStubDispatches);
	}
	if (program.info.bda_writes) {
		// KYTY_BDA_WRITES: settled synchronously right after the dispatch below. The settle relies
		// on PrepareBda's pass above having uploaded every CPU-dirty page first.
		EXIT_IF(!program.info.uses_dma);
		m_context.GetBufferCache().PrepareBdaWrites();
	}
	CpGaps::Phase(CpGaps::Cat::DispatchRebind);
	BindComputeLds(input_info, bindings, nullptr);
	RebindImages(bindings);
	if (IndirectLogEnabled()) [[unlikely]] {
		const auto&                         iprogram   = *input_info.stage.program;
		const auto&                         iresources = *input_info.stage.resources;
		static std::mutex                   ind_mutex;
		static std::unordered_set<uint64_t> ind_logged;
		bool                                first = false;
		{
			std::scoped_lock ind_lock(ind_mutex);
			// sarosmip: KYTY_CS_BIND_LOG=2 logs each shader again every 30 s (textures stream in
			// and out).
			static const auto start = std::chrono::steady_clock::now();
			const uint64_t    bucket =
			    mip_view_samplers()
			        ? static_cast<uint64_t>(
			              std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
			                  .count() /
			              30.0) +
			              1u
			        : 0u;
			first =
			    ind_logged.insert(iprogram.shader_hash ^ (bucket * 0x9e3779b97f4a7c15ull)).second;
		}
		if (first) {
			std::string line;
			char        part[160];
			for (uint32_t k = 0; k < iprogram.info.images.size(); k++) {
				const auto& image = iprogram.info.images[k];
				const auto  r = DecodeNativeDescriptor<ShaderTextureResource>(iresources.images[k]);
				std::snprintf(part, sizeof(part), " img%u=0x%010" PRIx64 ":f%u:%ux%u:%s%s", k,
				              r.Base40(), static_cast<uint32_t>(r.Format()),
				              static_cast<uint32_t>(r.Width5()) + 1u,
				              static_cast<uint32_t>(r.Height5()) + 1u, image.read ? "r" : "",
				              image.written ? "w" : "");
				line += part;
				// sarosmip: KYTY_CS_BIND_LOG=2 adds the T# view fields: type, base-last level,
				// MAX_MIP, MIN_LOD (U4.8).
				static const bool mip_view = [] {
					const auto* v = std::getenv("KYTY_CS_BIND_LOG");
					return v != nullptr && std::strcmp(v, "2") == 0;
				}();
				if (mip_view) {
					std::snprintf(
					    part, sizeof(part), "{t%u l%u-%u mm%u minlod%u}",
					    static_cast<uint32_t>(r.Type()), static_cast<uint32_t>(r.BaseLevel()),
					    static_cast<uint32_t>(r.LastLevel()), static_cast<uint32_t>(r.MaxMip()),
					    static_cast<uint32_t>(r.MinLod()));
					line += part;
				}
			}
			for (uint32_t k = 0; mip_view_samplers() && k < iprogram.info.samplers.size(); k++) {
				const auto s = DecodeNativeDescriptor<ShaderSamplerResource>(
				    iresources.samplers[iprogram.info.samplers[k].snapshot_index]);
				std::snprintf(part, sizeof(part), " smp%u={mip%u lod%u-%u bias%d}", k,
				              static_cast<uint32_t>(s.MipFilter()),
				              static_cast<uint32_t>(s.MinLod()), static_cast<uint32_t>(s.MaxLod()),
				              static_cast<int32_t>(s.LodBias()));
				line += part;
			}
			vk::DispatchIndirectCommand guest_args {};
			const bool                  dirty =
			    m_context.GetBufferCache().HasGpuDirtyBytes(args_addr, sizeof(guest_args));
			const bool read =
			    LibKernel::Memory::TryReadBacking(args_addr, &guest_args, sizeof(guest_args));
			std::fprintf(stderr,
			             "[nanite] CS-ind bind 0x%016" PRIx64 " va 0x%010" PRIx64
			             " args 0x%010" PRIx64 " guest-args %ux%ux%u%s%s:%s\n",
			             iprogram.shader_hash, cs_regs.cs_regs.data_addr, args_addr, guest_args.x,
			             guest_args.y, guest_args.z, dirty ? " gpu-dirty" : "",
			             read ? "" : " unreadable", line.c_str());
		}
	}
	if (NanProbe::Get().any) [[unlikely]] {
		SarosBufferStats(*input_info.stage.program, *input_info.stage.resources,
		                 NanProbe::Get().cs_hashes, args_addr);
	}
	if (PassTrace::Active()) [[unlikely]] {
		PreparedBindings* traced = &bindings;
		TracePass(m_context.GetTextureCache(), "dispatch", std::span<PreparedBindings* const>(&traced, 1), {});
	}
	// Acquiring arguments can merge cache buffers; finalize shader bindings afterward.
	PreparedBindings* flight_stage = &bindings;
	FlightRecDraw::Begin('X', std::span<PreparedBindings* const>(&flight_stage, 1),
	                     m_context.GetCommandScheduler().CurrentTick());
	m_rebind_epoch = m_context.GetBufferCache().UnregisterEpoch();
	auto [args_buffer, args_offset] = m_context.GetBufferCache().ObtainBuffer(
	    args_addr, sizeof(vk::DispatchIndirectCommand), false);
	EXIT_IF(args_buffer == nullptr || (args_offset & 3u) != 0);
	RebindBuffers(bindings);
	// KYTY_REBIND_AFTER_JOIN: the bindings' lookups joined the argument buffer (or one another's).
	if (RebindAfterJoinEnabled()) {
		auto& cache = m_context.GetBufferCache();
		for (uint32_t pass = 0; pass < 4 && cache.UnregisterEpoch() != m_rebind_epoch; pass++) {
			NoteRebindAfterJoin("indirect dispatch", nullptr);
			m_rebind_epoch               = cache.UnregisterEpoch();
			std::tie(args_buffer, args_offset) =
			    cache.ObtainBuffer(args_addr, sizeof(vk::DispatchIndirectCommand), false);
			EXIT_IF(args_buffer == nullptr || (args_offset & 3u) != 0);
			RebindBuffers(bindings);
		}
	}
	FlightRecDraw::End(m_context.GetTextureCache(),
	                   std::span<PreparedBindings* const>(&flight_stage, 1), {});
	InputProfileDraw::Note(m_context.GetBufferCache(),
	                       std::span<PreparedBindings* const>(&flight_stage, 1));
	CpGaps::Phase(CpGaps::Cat::DispatchEmit);
	PreparedBindings* descriptor_stage = &bindings;
	// Emission safe point (see DispatchDirect).
	buffer.BeginEmission();
	CommitBindings(buffer, vk::PipelineBindPoint::eCompute, pipeline,
	               std::span {&descriptor_stage, 1u});
	const bool has_storage_writes = bindings.shared_memory.buffer != nullptr ||
	    HasShaderBufferWrites(input_info.stage) || program.info.bda_writes ||
	    std::any_of(program.info.images.begin(), program.info.images.end(), [](const auto& image) {
		    return image.written && image.resource_class ==
		                                ShaderRecompiler::IR::ImageResourceClass::Storage;
	    });
	if (has_storage_writes) {
		ShaderWriteHazardBarrier(buffer, vk::PipelineStageFlagBits::eComputeShader);
	}
	if (BarrierBatchEnabled()) {
		buffer.RequestMemoryBarrier(vk::PipelineStageFlagBits2::eAllGraphics |
		                                vk::PipelineStageFlagBits2::eComputeShader |
		                                vk::PipelineStageFlagBits2::eTransfer,
		                            vk::AccessFlagBits2::eShaderWrite |
		                                vk::AccessFlagBits2::eTransferWrite,
		                            vk::PipelineStageFlagBits2::eDrawIndirect,
		                            vk::AccessFlagBits2::eIndirectCommandRead,
		                            BarrierOrigin::IndirectArgs);
	} else {
		vk::MemoryBarrier barrier {};
		barrier.srcAccessMask =
		    vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eTransferWrite;
		barrier.dstAccessMask = vk::AccessFlagBits::eIndirectCommandRead;
		buffer.Handle().pipelineBarrier(vk::PipelineStageFlagBits::eAllGraphics |
		                                    vk::PipelineStageFlagBits::eComputeShader |
		                                    vk::PipelineStageFlagBits::eTransfer,
		                                vk::PipelineStageFlagBits::eDrawIndirect, {}, 1, &barrier,
		                                0, nullptr, 0, nullptr);
	}
	buffer.BindPipeline(vk::PipelineBindPoint::eCompute, pipeline.pipeline);
	buffer.Sink().dispatchIndirect(args_buffer->Handle(), args_offset);
	NaniteShadeStats::NoteIndirect(args_addr, NaniteShadeStats::Outcome::Executed,
	                               program.shader_hash);
	Common::DebugCounters::Add(Common::DebugCounters::Counter::Dispatches);
	ShaderAccessBarrier(buffer, vk::PipelineStageFlagBits::eComputeShader);
	if (program.info.bda_writes) {
		m_context.GetBufferCache().SettleBdaWrites(program.shader_hash);
	}
	if (MetalFx::Requested() && !MetalFx::ComputeReplace(program.shader_hash)) [[unlikely]] {
		TryMetalFxCompute(buffer, bindings);
	}
	ResetBindings();
}

} // namespace Libs::Graphics
