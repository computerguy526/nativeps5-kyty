#include "common/gpuSafety.h"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

static int g_checks = 0;
#define CHECK(c)                                                                                   \
	do {                                                                                           \
		++g_checks;                                                                                \
		if (!(c)) {                                                                                \
			std::fprintf(stderr, "GpuSafetyTests: FAIL line %d: %s\n", __LINE__, #c);              \
			std::abort();                                                                          \
		}                                                                                          \
	} while (0)

// nps5: the library lives in a host process: the host's own SIGUSR2 handler must keep working.
static std::atomic<int> g_host_handler_calls {0};
static void HostUsr2(int, siginfo_t*, void*) {
	g_host_handler_calls.fetch_add(1);
}

int main() {
	// The host installed a SIGUSR2 handler before the library's first submission.
	struct sigaction host {};
	host.sa_sigaction = HostUsr2;
	host.sa_flags     = SA_SIGINFO;
	sigemptyset(&host.sa_mask);
	CHECK(sigaction(SIGUSR2, &host, nullptr) == 0);
	unsetenv("KYTY_GPU_WATCHDOG_S");
	{
		// First submissions race on several threads: one initialisation.
		std::vector<std::thread> init;
		for (int t = 0; t < 8; ++t) {
			init.emplace_back([] { GpuSafety::InitSlow(); });
		}
		for (auto& t : init) {
			t.join();
		}
	}
	CHECK(GpuSafety::g_state.load() == 1);
	struct sigaction now {};
	CHECK(sigaction(SIGUSR2, nullptr, &now) == 0);
	CHECK(now.sa_sigaction != HostUsr2);             // the ring dump is installed
	CHECK((now.sa_flags & SA_RESTART) != 0);         // a stray SIGUSR2 does not fail the host's system calls with EINTR
	CHECK(raise(SIGUSR2) == 0);
	CHECK(g_host_handler_calls.load() == 1);         // and the host's handler still runs, once

	CHECK(GpuSafety::WatchdogEnabledFromEnv(nullptr));
	CHECK(GpuSafety::WatchdogEnabledFromEnv(""));
	CHECK(GpuSafety::WatchdogEnabledFromEnv("4"));
	CHECK(GpuSafety::WatchdogEnabledFromEnv("0.5"));
	CHECK(GpuSafety::WatchdogEnabledFromEnv("garbage"));
	CHECK(!GpuSafety::WatchdogEnabledFromEnv("0"));
	CHECK(!GpuSafety::WatchdogEnabledFromEnv("0.0"));

	// Order, wrap and formatting.
	unsetenv("KYTY_GPU_WATCHDOG_S");
	for (uint64_t i = 1; i <= 200; ++i) {
		GpuSafety::NoteSubmit(i, 1000 + i, 7, static_cast<uint32_t>(i));
	}
	CHECK(GpuSafety::g_state.load() == 1);
	CHECK(GpuSafety::g_head.load() == 200);
	char   buf[16384];
	size_t n = GpuSafety::FormatRing(buf, sizeof(buf), 5);
	CHECK(n > 0);
	CHECK(std::strstr(buf, "submit #200 tick=200 guest_submit=1200 op=7 arg0=200") == buf + 0 ||
	      std::strstr(buf, "submit #200 tick=200 guest_submit=1200 op=7 arg0=200") != nullptr);
	CHECK(std::strstr(buf, "submit #196 ") != nullptr);
	CHECK(std::strstr(buf, "submit #195 ") == nullptr); // max 5
	n = GpuSafety::FormatRing(buf, sizeof(buf), 1000);
	CHECK(std::strstr(buf, "submit #137 ") != nullptr); // 64 newest: 200..137
	CHECK(std::strstr(buf, "submit #136 ") == nullptr);
	// tiny buffer never overflows
	char small[100];
	n = GpuSafety::FormatRing(small, sizeof(small), 64);
	CHECK(n < sizeof(small));
	CHECK(GpuSafety::FormatRing(small, 0, 64) == 0);

	// Concurrent writers + reader: no crash, head counts exactly.
	std::vector<std::thread> th;
	for (int t = 0; t < 4; ++t) {
		th.emplace_back([t] {
			for (uint32_t i = 0; i < 50000; ++i) {
				GpuSafety::NoteSubmit(i, t, 1, i);
			}
		});
	}
	std::thread reader([&] {
		char b[8192];
		for (int i = 0; i < 2000; ++i) {
			GpuSafety::FormatRing(b, sizeof(b), 32);
		}
	});
	for (auto& t : th) {
		t.join();
	}
	reader.join();
	CHECK(GpuSafety::g_head.load() == 200 + 4 * 50000);

	// nps5: a line never mixes two submissions. One writer laps the ring while readers format it: every
	// line that is printed has tick == guest_submit == arg0 (the writer's invariant).
	std::atomic<bool>     stop {false};
	std::atomic<uint64_t> torn {0}, lines {0};
	std::thread           writer([&] {
        for (uint32_t i = 1; !stop.load(std::memory_order_relaxed); ++i) {
            GpuSafety::NoteSubmit(i, i, 3, i);
        }
    });
	std::vector<std::thread> readers;
	for (int r = 0; r < 6; ++r) {
		readers.emplace_back([&] {
			char b[8192];
			while (!stop.load(std::memory_order_relaxed)) {
				GpuSafety::FormatRing(b, sizeof(b), 64);
				for (const char* line = b; (line = std::strstr(line, "tick=")) != nullptr; line += 5) {
					unsigned long long tick = 0, submit = 0;
					unsigned           op = 0, arg0 = 0;
					if (std::sscanf(line, "tick=%llu guest_submit=%llu op=%u arg0=%u", &tick, &submit, &op, &arg0) == 4) {
						lines.fetch_add(1, std::memory_order_relaxed);
						if (op == 3 && (tick != submit || static_cast<uint32_t>(tick) != arg0)) {
							torn.fetch_add(1, std::memory_order_relaxed);
						}
					}
				}
			}
		});
	}
	std::this_thread::sleep_for(std::chrono::milliseconds(1500));
	stop.store(true);
	writer.join();
	for (auto& t : readers) {
		t.join();
	}
	std::printf("GpuSafetyTests: %llu lines read while the ring was lapped, %llu torn\n",
	            static_cast<unsigned long long>(lines.load()), static_cast<unsigned long long>(torn.load()));
	CHECK(lines.load() > 1000);
	CHECK(torn.load() == 0);
	std::printf("GpuSafetyTests: %d checks OK\n", g_checks);
	return 0;
}
