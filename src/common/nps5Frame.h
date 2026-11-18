// nps5 cachesave: frame-time probe shared by the present path (writer) and background savers
// (readers).
#pragma once
#include <atomic>
#include <cstdint>

namespace Nps5Frame {
// Wall-clock ns of the last Present() and the ms length of the last frame (0 until two frames were
// seen).
inline std::atomic<uint64_t> g_last_present_ns {0};
inline std::atomic<uint32_t> g_last_frame_ms {0};
// Steady ns at which the frame time was last above 100 ms (0 = never).
inline std::atomic<uint64_t> g_slow_frame_ns {0};
} // namespace Nps5Frame
