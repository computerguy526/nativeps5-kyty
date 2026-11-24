#include "common/gpuSafety.h"

#include <cerrno>
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
	size_t len           = 0;
	buf[0]               = '\0';
	const uint64_t head  = g_head.load(std::memory_order_acquire);
	const uint64_t now   = NowNs();
	size_t         shown = 0;
	for (uint64_t n = head; n > 0 && shown < max && shown < RingSize; --n, ++shown) {
		const Entry& e = g_ring[n & (RingSize - 1)];
		if (e.seq.load(std::memory_order_acquire) != n) {
			continue;
		}
		const uint64_t tick   = e.tick.load(std::memory_order_acquire);
		const uint64_t submit = e.submit.load(std::memory_order_acquire);
		const uint64_t args   = e.args.load(std::memory_order_acquire);
		const uint64_t ns     = e.ns.load(std::memory_order_acquire);
		// A newer submission took the slot while it was read (the ring was lapped): its fields are
		// not n's.
		if (e.seq.load(std::memory_order_acquire) != n) {
			continue;
		}
		const int w = std::snprintf(
		    buf + len, cap - len,
		    "[kyty-safe]  submit #%llu tick=%llu guest_submit=%llu op=%u arg0=%u age=%llu ms\n",
		    static_cast<unsigned long long>(n), static_cast<unsigned long long>(tick),
		    static_cast<unsigned long long>(submit), static_cast<unsigned>(args >> 32),
		    static_cast<unsigned>(args & 0xffffffffu),
		    static_cast<unsigned long long>((now > ns ? now - ns : 0) / 1000000ull));
		if (w < 0 || static_cast<size_t>(w) >= cap - len) {
			break;
		}
		len += static_cast<size_t>(w);
	}
	return len;
}

#if !defined(_WIN32)
// The SIGUSR2 action found at InitSlow: the host process's own handler, called after the ring dump.
static struct sigaction  g_previous_usr2;
static std::atomic<bool> g_previous_usr2_valid {false};

static void Sigusr2(int sig, siginfo_t* info, void* context) {
	const int   saved_errno = errno;
	static char buf[8192];
	const char  head[] =
	    "[kyty-safe] GPU watchdog fired in the driver; last native submissions (newest first):\n";
	(void)!write(2, head, sizeof(head) - 1);
	const size_t n = FormatRing(buf, sizeof(buf), 24);
	(void)!write(2, buf, n);
	if (g_previous_usr2_valid.load(std::memory_order_acquire)) {
		if ((g_previous_usr2.sa_flags & SA_SIGINFO) != 0) {
			if (g_previous_usr2.sa_sigaction != nullptr) {
				g_previous_usr2.sa_sigaction(sig, info, context);
			}
		} else if (g_previous_usr2.sa_handler != SIG_DFL && g_previous_usr2.sa_handler != SIG_IGN) {
			g_previous_usr2.sa_handler(sig);
		}
	}
	errno = saved_errno;
}
#endif

void InitSlow() noexcept {
	int expected = 0;
	if (!g_state.compare_exchange_strong(expected, 3)) {
		return; // initialised, or another thread is at it (a second install would chain to
		        // ourselves)
	}
	const bool on = WatchdogEnabledFromEnv(std::getenv("KYTY_GPU_WATCHDOG_S"));
#if !defined(_WIN32)
	if (on) {
		struct sigaction old;
		std::memset(&old, 0, sizeof(old));
		if (sigaction(SIGUSR2, nullptr, &old) == 0 &&
		    ((old.sa_flags & SA_SIGINFO) != 0
		         ? old.sa_sigaction != nullptr
		         : (old.sa_handler != SIG_DFL && old.sa_handler != SIG_IGN))) {
			g_previous_usr2 = old;
			g_previous_usr2_valid.store(true, std::memory_order_release);
		}
		struct sigaction sa;
		std::memset(&sa, 0, sizeof(sa));
		sa.sa_sigaction = Sigusr2;
		sa.sa_flags     = SA_SIGINFO | SA_RESTART;
		sigemptyset(&sa.sa_mask);
		sigaction(SIGUSR2, &sa, nullptr);
	}
#endif
	g_state.store(on ? 1 : 2, std::memory_order_release);
}

} // namespace GpuSafety
