#pragma once

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>

namespace Common {
inline void HostInputTrace(const char* event, int a, int b, int c = 0) {
	static FILE* output = [] {
		const char* path = std::getenv("KYTY_HOST_INPUT_TRACE");
		return path && *path ? std::fopen(path, "a") : nullptr;
	}();
	if (!output) { return; }
	// archfix E1: never destroyed (threads outlive exit()'s static destructors: 'mutex lock failed'
	// -> terminate)
	static std::mutex& mutex = *new std::mutex;
	std::lock_guard lock(mutex);
	const auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
	    std::chrono::steady_clock::now().time_since_epoch()).count();
	std::fprintf(output, "%lld,%s,%d,%d,%d\n", static_cast<long long>(now), event, a, b, c);
	std::fflush(output);
}
} // namespace Common
