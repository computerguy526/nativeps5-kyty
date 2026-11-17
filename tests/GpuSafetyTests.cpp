#include "common/gpuSafety.h"

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

int main() {
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
	std::printf("GpuSafetyTests: %d checks OK\n", g_checks);
	return 0;
}
