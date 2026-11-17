#include "common/gpuSafety.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#if !defined(_WIN32)
#include <csignal>
#include <unistd.h>
#endif

namespace GpuSafety {

std::atomic<int>      g_state {0};
std::atomic<uint64_t> g_head {0};
Entry                 g_ring[RingSize];

uint64_t NowNs() noexcept {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}

bool WatchdogEnabledFromEnv(const char* value) noexcept {
	if (value == nullptr || *value == '\0') {
		return true; // default 4 s
	}
	char*        end = nullptr;
	const double v   = std::strtod(value, &end);
	if (end == value) {
		return true;
	}
	return v != 0.0;
}

size_t FormatRing(char* buf, size_t cap, size_t max) noexcept {
	if (cap == 0) {
		return 0;
	}
	size_t         len  = 0;
	buf[0]              = '\0';
	const uint64_t head = g_head.load(std::memory_order_acquire);
	const uint64_t now  = NowNs();
	size_t         shown = 0;
	for (uint64_t n = head; n > 0 && shown < max && shown < RingSize; --n, ++shown) {
		const Entry& e = g_ring[n & (RingSize - 1)];
		if (e.seq.load(std::memory_order_acquire) != n) {
			continue;
		}
		const uint64_t args = e.args.load(std::memory_order_relaxed);
		const int      w    = std::snprintf(
            buf + len, cap - len,
            "[kyty-safe]  submit #%llu tick=%llu guest_submit=%llu op=%u arg0=%u age=%llu ms\n",
            static_cast<unsigned long long>(n),
            static_cast<unsigned long long>(e.tick.load(std::memory_order_relaxed)),
            static_cast<unsigned long long>(e.submit.load(std::memory_order_relaxed)),
            static_cast<unsigned>(args >> 32), static_cast<unsigned>(args & 0xffffffffu),
            static_cast<unsigned long long>((now - e.ns.load(std::memory_order_relaxed)) / 1000000ull));
		if (w < 0 || static_cast<size_t>(w) >= cap - len) {
			break;
		}
		len += static_cast<size_t>(w);
	}
	return len;
}

#if !defined(_WIN32)
static void Sigusr2(int) {
	static char buf[8192];
	const char  head[] = "[kyty-safe] GPU watchdog fired in the driver; last native submissions (newest first):\n";
	(void)!write(2, head, sizeof(head) - 1);
	const size_t n = FormatRing(buf, sizeof(buf), 24);
	(void)!write(2, buf, n);
}
#endif

void InitSlow() noexcept {
	const bool on = WatchdogEnabledFromEnv(std::getenv("KYTY_GPU_WATCHDOG_S"));
#if !defined(_WIN32)
	if (on) {
		struct sigaction sa;
		std::memset(&sa, 0, sizeof(sa));
		sa.sa_handler = Sigusr2;
		sigemptyset(&sa.sa_mask);
		sigaction(SIGUSR2, &sa, nullptr);
	}
#endif
	int expected = 0;
	g_state.compare_exchange_strong(expected, on ? 1 : 2);
}

} // namespace GpuSafety
