#include "graphics/presentation/window/lowLatency.h"

#include "graphics/host_gpu/renderer/frameGen.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

namespace Libs::Graphics::LowLatency {

namespace {

struct State {
	std::atomic<bool>          enabled {false};
	VkDevice                   device = VK_NULL_HANDLE;
	PFN_vkSetLatencySleepModeNV set_sleep_mode = nullptr;
	PFN_vkLatencySleepNV        sleep          = nullptr;
	PFN_vkSetLatencyMarkerNV    marker         = nullptr;
	PFN_vkGetLatencyTimingsNV   timings        = nullptr;
	VkSemaphore                 semaphore      = VK_NULL_HANDLE;
	std::atomic<uint64_t>       semaphore_value {0};
	std::mutex                  mutex; // the swapchain's calls (externally synchronized)
	VkSwapchainKHR              swapchain = VK_NULL_HANDLE;
	bool                        sleep_on  = false; // mode 1: the driver's sleep is called
	VkSwapchainLatencyCreateInfoNV create_info {};
	// Guest frames: flips set since the last accounted submission; frames ended.
	std::atomic<uint64_t> flips_set {0};
	std::atomic<uint64_t> flips_accounted {0};
	std::atomic<uint64_t> guest_frame {0};
	std::atomic<uint64_t> gpu_frame {0};
	// Presenter.
	uint64_t next_present_frame = 0;
	uint64_t last_present_id    = 0;
	int64_t  report_ns          = 0;
	// Statistics (any thread).
	std::atomic<uint64_t> sleeps {0};
	std::atomic<uint64_t> sleep_us {0};
	std::atomic<uint64_t> sleep_timeouts {0};
	std::atomic<uint64_t> presents_with_id {0};
};

State& Get() {
	static State state;
	return state;
}

int64_t NowNs() {
	return std::chrono::duration_cast<std::chrono::nanoseconds>(
	           std::chrono::steady_clock::now().time_since_epoch())
	    .count();
}

void MarkLocked(State& s, uint64_t frame, VkLatencyMarkerNV marker) {
	if (s.swapchain == VK_NULL_HANDLE || frame == 0) {
		return;
	}
	VkSetLatencyMarkerInfoNV info {};
	info.sType     = VK_STRUCTURE_TYPE_SET_LATENCY_MARKER_INFO_NV;
	info.presentID = frame;
	info.marker    = marker;
	s.marker(s.device, s.swapchain, &info);
}

void Mark(uint64_t frame, VkLatencyMarkerNV marker) {
	auto&                       s = Get();
	std::lock_guard<std::mutex> lock(s.mutex);
	MarkLocked(s, frame, marker);
}

// The end of guest frame `frame`: its marker, the driver's sleep, the next frame's start.
void EndGuestFrame(uint64_t frame) {
	auto& s = Get();
	if (FrameGen::Active()) {
		FrameGen::GuestFrameEnd(frame);
		return;
	}
	Mark(frame, VK_LATENCY_MARKER_SIMULATION_END_NV);
	uint64_t value = 0;
	{
		std::lock_guard<std::mutex> lock(s.mutex);
		if (s.swapchain != VK_NULL_HANDLE && s.sleep_on) {
			value = s.semaphore_value.fetch_add(1, std::memory_order_relaxed) + 1;
			VkLatencySleepInfoNV info {};
			info.sType           = VK_STRUCTURE_TYPE_LATENCY_SLEEP_INFO_NV;
			info.signalSemaphore = s.semaphore;
			info.value           = value;
			if (s.sleep(s.device, s.swapchain, &info) != VK_SUCCESS) {
				value = 0;
			}
		}
	}
	if (value != 0) {
		const auto      start = NowNs();
		VkSemaphoreWaitInfo wait {};
		wait.sType          = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
		wait.semaphoreCount = 1;
		wait.pSemaphores    = &s.semaphore;
		wait.pValues        = &value;
		// Bounded: a frame is never held longer than 100 ms by the sleep.
		const auto result =
		    VULKAN_HPP_DEFAULT_DISPATCHER.vkWaitSemaphores(s.device, &wait, 100'000'000ull);
		if (result == VK_TIMEOUT) {
			s.sleep_timeouts.fetch_add(1, std::memory_order_relaxed);
		}
		s.sleeps.fetch_add(1, std::memory_order_relaxed);
		s.sleep_us.fetch_add(static_cast<uint64_t>((NowNs() - start) / 1000),
		                     std::memory_order_relaxed);
	}
	Mark(frame + 1, VK_LATENCY_MARKER_SIMULATION_START_NV);
}

void Report(State& s) {
	// Called with s.mutex held, from the presenter.
	const auto now = NowNs();
	if (s.report_ns == 0) {
		s.report_ns = now;
		return;
	}
	if (now - s.report_ns < 10'000'000'000LL) {
		return;
	}
	s.report_ns = now;
	VkGetLatencyMarkerInfoNV query {};
	query.sType = VK_STRUCTURE_TYPE_GET_LATENCY_MARKER_INFO_NV;
	s.timings(s.device, s.swapchain, &query);
	std::vector<VkLatencyTimingsFrameReportNV> reports(query.timingCount);
	for (auto& report: reports) {
		report       = {};
		report.sType = VK_STRUCTURE_TYPE_LATENCY_TIMINGS_FRAME_REPORT_NV;
	}
	query.pTimings = reports.data();
	if (!reports.empty()) {
		s.timings(s.device, s.swapchain, &query);
	}
	uint64_t filled = 0, total = 0, sim = 0, queue = 0, gpu = 0;
	uint64_t with_present = 0, with_gpu = 0;
	for (uint32_t i = 0; i < query.timingCount; i++) {
		const auto& r = reports[i];
		with_present += r.presentEndTimeUs != 0 ? 1u : 0u;
		with_gpu += r.gpuRenderEndTimeUs != 0 ? 1u : 0u;
		if (r.simStartTimeUs == 0 || r.gpuRenderEndTimeUs < r.simStartTimeUs) {
			continue;
		}
		filled++;
		total += r.gpuRenderEndTimeUs - r.simStartTimeUs;
		if (r.simEndTimeUs >= r.simStartTimeUs) {
			sim += r.simEndTimeUs - r.simStartTimeUs;
		}
		if (r.gpuRenderStartTimeUs >= r.presentEndTimeUs && r.presentEndTimeUs != 0) {
			queue += r.gpuRenderStartTimeUs - r.presentEndTimeUs;
		}
		if (r.gpuRenderEndTimeUs >= r.gpuRenderStartTimeUs && r.gpuRenderStartTimeUs != 0) {
			gpu += r.gpuRenderEndTimeUs - r.gpuRenderStartTimeUs;
		}
	}
	const auto sleeps = s.sleeps.exchange(0, std::memory_order_relaxed);
	const auto sleep_us = s.sleep_us.exchange(0, std::memory_order_relaxed);
	const double d = static_cast<double>(filled == 0 ? 1 : filled);
	std::printf("LowLatency 10s: driver reports %u (present times %llu, GPU times %llu, full %llu), "
	            "sim start -> GPU end "
	            "%.2f ms, guest frame %.2f ms, present -> GPU start %.2f ms, GPU %.2f ms | sleeps "
	            "%llu avg %.2f ms, %llu timeouts | presents with ID %llu | flip packets %llu, frames guest %llu CP %llu\n",
	            query.timingCount, static_cast<unsigned long long>(with_present),
	            static_cast<unsigned long long>(with_gpu), static_cast<unsigned long long>(filled),
	            static_cast<double>(total) / d / 1000.0, static_cast<double>(sim) / d / 1000.0,
	            static_cast<double>(queue) / d / 1000.0, static_cast<double>(gpu) / d / 1000.0,
	            static_cast<unsigned long long>(sleeps),
	            sleeps == 0 ? 0.0 : static_cast<double>(sleep_us) / static_cast<double>(sleeps) / 1000.0,
	            static_cast<unsigned long long>(s.sleep_timeouts.exchange(0)),
	            static_cast<unsigned long long>(s.presents_with_id.exchange(0)),
	            static_cast<unsigned long long>(s.flips_set.load()),
	            static_cast<unsigned long long>(s.guest_frame.load()),
	            static_cast<unsigned long long>(s.gpu_frame.load()));
	std::fflush(stdout);
}

} // namespace

// 0 off, 1 on, 2 measure (markers and driver reports, low-latency mode off).
static int Mode() {
	static const int mode = [] {
		const char* value = std::getenv("ASTROHYBRID_LOW_LATENCY");
		if (value == nullptr) {
			return 0;
		}
		if (std::strcmp(value, "1") == 0) {
			return 1;
		}
		return std::strcmp(value, "measure") == 0 ? 2 : 0;
	}();
	return mode;
}

bool Requested() {
	return Mode() != 0;
}

bool WantExtensions(bool has_low_latency2, bool has_present_id) {
	if (!Requested()) {
		return false;
	}
	if (FrameGen::Requested()) {
		std::printf("LowLatency: DLSS-G takes over the markers through Streamline's Reflex\n");
		return false;
	}
	if (!has_low_latency2 || !has_present_id) {
		std::printf("LowLatency: requested but unavailable (VK_NV_low_latency2 %d, VK_KHR_present_id "
		            "%d): off\n",
		            has_low_latency2 ? 1 : 0, has_present_id ? 1 : 0);
		return false;
	}
	return true;
}

void OnDeviceCreated(VkDevice device) {
	auto& s = Get();
	const auto get = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr;
	s.device         = device;
	s.set_sleep_mode = reinterpret_cast<PFN_vkSetLatencySleepModeNV>(get(device, "vkSetLatencySleepModeNV"));
	s.sleep          = reinterpret_cast<PFN_vkLatencySleepNV>(get(device, "vkLatencySleepNV"));
	s.marker         = reinterpret_cast<PFN_vkSetLatencyMarkerNV>(get(device, "vkSetLatencyMarkerNV"));
	s.timings        = reinterpret_cast<PFN_vkGetLatencyTimingsNV>(get(device, "vkGetLatencyTimingsNV"));
	if (s.set_sleep_mode == nullptr || s.sleep == nullptr || s.marker == nullptr ||
	    s.timings == nullptr) {
		std::printf("LowLatency: VK_NV_low_latency2 entry points missing: off\n");
		return;
	}
	VkSemaphoreTypeCreateInfo type {};
	type.sType         = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
	type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
	type.initialValue  = 0;
	VkSemaphoreCreateInfo create {};
	create.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
	create.pNext = &type;
	if (VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateSemaphore(device, &create, nullptr, &s.semaphore) !=
	    VK_SUCCESS) {
		std::printf("LowLatency: timeline semaphore creation failed: off\n");
		return;
	}
	s.create_info.sType             = VK_STRUCTURE_TYPE_SWAPCHAIN_LATENCY_CREATE_INFO_NV;
	s.create_info.latencyModeEnable = VK_TRUE;
	s.enabled.store(true, std::memory_order_release);
	std::printf("LowLatency: VK_NV_low_latency2 enabled on the device\n");
	std::fflush(stdout);
}

bool Enabled() {
	return Get().enabled.load(std::memory_order_acquire);
}

const void* SwapchainCreateNext(const void* next) {
	auto& s = Get();
	if (!Enabled()) {
		return next;
	}
	s.create_info.pNext = next;
	return &s.create_info;
}

void OnSwapchainCreated(VkSwapchainKHR swapchain) {
	auto& s = Get();
	if (!Enabled()) {
		return;
	}
	std::lock_guard<std::mutex> lock(s.mutex);
	VkLatencySleepModeInfoNV mode {};
	mode.sType             = VK_STRUCTURE_TYPE_LATENCY_SLEEP_MODE_INFO_NV;
	mode.lowLatencyMode    = Mode() == 1 ? VK_TRUE : VK_FALSE;
	mode.lowLatencyBoost   = Mode() == 1 ? VK_TRUE : VK_FALSE;
	mode.minimumIntervalUs = 0;
	const auto result = s.set_sleep_mode(s.device, swapchain, &mode);
	std::printf("LowLatency: swapchain sleep mode %s -> %s\n",
	            Mode() == 1 ? "on (boost)" : "off (measure only)",
	            result == VK_SUCCESS ? "ok" : "failed");
	std::fflush(stdout);
	s.swapchain       = result == VK_SUCCESS ? swapchain : VK_NULL_HANDLE;
	s.sleep_on        = Mode() == 1;
	s.last_present_id = 0;
}

void OnSwapchainDestroyed(VkSwapchainKHR swapchain) {
	auto& s = Get();
	if (!Enabled()) {
		return;
	}
	std::lock_guard<std::mutex> lock(s.mutex);
	if (s.swapchain == swapchain) {
		s.swapchain = VK_NULL_HANDLE;
	}
}

namespace {
bool Counting() {
	return Enabled() || FrameGen::Active();
}
} // namespace

void NoteFlipPacket() {
	if (Counting()) {
		Get().flips_set.fetch_add(1, std::memory_order_acq_rel);
	}
}

void AfterDcbSubmit() {
	if (!Counting()) {
		return;
	}
	auto&      s   = Get();
	const auto set = s.flips_set.load(std::memory_order_acquire);
	auto       old = s.flips_accounted.load(std::memory_order_acquire);
	if (set == old || !s.flips_accounted.compare_exchange_strong(old, set)) {
		return;
	}
	// One frame per flip packet submitted (a submission normally carries one).
	uint64_t frame = 0;
	for (uint64_t i = old; i < set; i++) {
		frame = s.guest_frame.fetch_add(1, std::memory_order_acq_rel) + 1;
	}
	EndGuestFrame(frame);
}

namespace {
thread_local uint64_t t_cpu_flip_frame = 0;
} // namespace

uint64_t BeginCpuFlip() {
	if (!Counting()) {
		return 0;
	}
	t_cpu_flip_frame = Get().guest_frame.fetch_add(1, std::memory_order_acq_rel) + 1;
	return t_cpu_flip_frame;
}

uint64_t TakeCpuFlipFrame() {
	const auto frame = t_cpu_flip_frame;
	t_cpu_flip_frame = 0;
	return frame;
}

void EndCpuFlip(uint64_t frame) {
	if (Counting() && frame != 0) {
		EndGuestFrame(frame);
	}
}

uint64_t OnGpuFlipReserved() {
	if (!Counting()) {
		return 0;
	}
	auto&      s     = Get();
	const auto frame = s.gpu_frame.fetch_add(1, std::memory_order_acq_rel) + 1;
	if (FrameGen::Active()) {
		FrameGen::RenderSubmitted(frame);
		return frame;
	}
	std::lock_guard<std::mutex> lock(s.mutex);
	MarkLocked(s, frame, VK_LATENCY_MARKER_RENDERSUBMIT_END_NV);
	MarkLocked(s, frame + 1, VK_LATENCY_MARKER_RENDERSUBMIT_START_NV);
	return frame;
}

namespace {
thread_local uint64_t t_frame_gen_present = 0;
} // namespace

void SetNextPresentFrame(uint64_t frame) {
	if (Counting()) {
		Get().next_present_frame = frame;
	}
}

uint64_t BeginPresent(VkSwapchainKHR swapchain) {
	if (FrameGen::Active()) {
		auto& s              = Get();
		t_frame_gen_present  = s.next_present_frame;
		s.next_present_frame = 0;
		FrameGen::PresentStart(t_frame_gen_present);
		return 0;
	}
	if (!Enabled()) {
		return 0;
	}
	auto& s = Get();
	s.mutex.lock();
	if (s.swapchain != swapchain) {
		return 0;
	}
	const auto frame     = s.next_present_frame;
	s.next_present_frame = 0;
	if (frame == 0 || frame <= s.last_present_id) {
		return 0;
	}
	s.last_present_id = frame;
	MarkLocked(s, frame, VK_LATENCY_MARKER_PRESENT_START_NV);
	s.presents_with_id.fetch_add(1, std::memory_order_relaxed);
	return frame;
}

void EndPresent(VkSwapchainKHR swapchain, uint64_t present_id) {
	if (FrameGen::Active()) {
		FrameGen::PresentEnd(t_frame_gen_present);
		t_frame_gen_present = 0;
		return;
	}
	if (!Enabled()) {
		return;
	}
	auto& s = Get();
	if (s.swapchain == swapchain) {
		MarkLocked(s, present_id, VK_LATENCY_MARKER_PRESENT_END_NV);
		Report(s);
	}
	s.mutex.unlock();
}

} // namespace Libs::Graphics::LowLatency
