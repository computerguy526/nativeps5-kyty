#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_WINDOW_LOWLATENCY_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_WINDOW_LOWLATENCY_H_

#include <cstdint>

#include <vulkan/vulkan_core.h>

// ASTROHYBRID_LOW_LATENCY=1 (default 0; "measure": markers and driver reports with the
// low-latency mode and its sleep off, for comparison): NVIDIA low-latency mode through VK_NV_low_latency2 (the
// Vulkan side of Reflex), driven by Kyty itself. Enabled only when the device reports both
// VK_NV_low_latency2 and VK_KHR_present_id with its presentId feature; otherwise the switch is
// ignored and logged as unavailable.
//
// Frames are numbered by guest flips, in the order the guest issues them:
//  - guest render thread: a DCB submission that carries a flip packet (AgcDcbSetFlip since the
//    previous submission), or a CPU flip (sceVideoOutSubmitFlip), ends frame N
//    (SIMULATION_END N); the thread then calls vkLatencySleepNV and waits for its semaphore
//    (bounded), which is where the driver holds back the start of the next frame, and marks
//    SIMULATION_START N+1;
//  - command processor: reserving the N-th GPU flip marks RENDERSUBMIT_END N and
//    RENDERSUBMIT_START N+1 (the CP translates and submits the frame's work up to its flip);
//  - presenter: the presented flip's frame number is the present ID (VkPresentIdKHR) and
//    PRESENT_START / PRESENT_END frame the vkQueuePresentKHR call. A present without a new frame
//    (a repeat at vblank) carries no ID.
// Every 10 s the presenter reads the driver's own per-frame reports (vkGetLatencyTimingsNV) and
// prints "LowLatency 10s: ..."; reports with filled timestamps are the proof the driver is acting
// on the markers. Calls naming the swapchain are serialized by one mutex (also held around
// vkQueuePresentKHR), as the swapchain is externally synchronized.
namespace Libs::Graphics::LowLatency {

// The environment switch (read once).
[[nodiscard]] bool Requested();
// Device creation: whether to add the two extensions (both present in `available`).
[[nodiscard]] bool WantExtensions(bool has_low_latency2, bool has_present_id);
// After vkCreateDevice with them and presentId enabled.
void OnDeviceCreated(VkDevice device);
[[nodiscard]] bool Enabled();
// Swapchain creation: the pNext to chain into VkSwapchainCreateInfoKHR (nullptr when disabled).
[[nodiscard]] const void* SwapchainCreateNext(const void* next);
void OnSwapchainCreated(VkSwapchainKHR swapchain);
void OnSwapchainDestroyed(VkSwapchainKHR swapchain);

// Guest side.
void NoteFlipPacket();      // AgcDcbSetFlip (any thread)
void AfterDcbSubmit();      // the bridge's DCB submission returned (guest thread)
// sceVideoOutSubmitFlip (guest thread): Begin numbers the frame before its reservation (the
// reservation takes the number with TakeCpuFlipFrame), End ends it once the flip is queued.
[[nodiscard]] uint64_t BeginCpuFlip();
[[nodiscard]] uint64_t TakeCpuFlipFrame();
void                   EndCpuFlip(uint64_t frame);
// Command processor: a GPU flip was reserved; returns its frame number (0 when disabled).
[[nodiscard]] uint64_t OnGpuFlipReserved();

// Presenter.
void SetNextPresentFrame(uint64_t frame);
// Around vkQueuePresentKHR (present thread): Begin returns the present ID to chain (0: none) and
// takes the swapchain mutex; End marks PRESENT_END and releases it.
[[nodiscard]] uint64_t BeginPresent(VkSwapchainKHR swapchain);
void EndPresent(VkSwapchainKHR swapchain, uint64_t present_id);

} // namespace Libs::Graphics::LowLatency

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_WINDOW_LOWLATENCY_H_
