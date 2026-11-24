#ifndef EMULATOR_SRC_LIBS_GUESTCALLTRACE_H_
#define EMULATOR_SRC_LIBS_GUESTCALLTRACE_H_

// nps5 (kyty_gpu.dylib): KYTY_GPU_CALL_TRACE=1 logs every export call the game makes (name, the
// first six arguments as the game passed them, the result as the game receives it) to stderr.
// Included after libs.h by the files that register guest-pointer exports; replaces their LIB_FUNC.

#if defined(KYTY_GPU_DYLIB) && defined(KYTY_LIB_GUEST_POINTERS)

#include "libs/libs.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <string>

namespace Libs::GuestCall {

inline bool TraceEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_GPU_CALL_TRACE");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

// KYTY_GPU_CALL_TRACE_WATCH=<guest address>: the 8 bytes there, printed after every traced call.
inline void TraceWatch() {
	static const unsigned long long address = [] {
		const char* value = std::getenv("KYTY_GPU_CALL_TRACE_WATCH");
		return value != nullptr ? std::strtoull(value, nullptr, 0) : 0ull;
	}();
	// KYTY_GPU_CALL_TRACE_DUMP=<guest address>: 0x180 bytes there after every 8th traced call past
	// 1100.
	static const unsigned long long dump = [] {
		const char* value = std::getenv("KYTY_GPU_CALL_TRACE_DUMP");
		return value != nullptr ? std::strtoull(value, nullptr, 0) : 0ull;
	}();
	static unsigned long long calls = 0;
	if (dump != 0 && ++calls > 1100 && (calls & 7u) == 0) {
		const auto* words = Guest::Ptr<const volatile unsigned long long>(dump);
		for (int i = 0; i < 48; i += 4) {
			std::fprintf(stderr, "[kyty dump] %llx: %016llx %016llx %016llx %016llx\n",
			             dump + i * 8ull, words[i], words[i + 1], words[i + 2], words[i + 3]);
		}
	}
	if (address != 0) {
		static unsigned long long last = ~0ull;
		const auto                now  = *Guest::Ptr<const volatile unsigned long long>(address);
		if (now != last) {
			std::fprintf(stderr, "[kyty watch] %llx: %llx -> %llx\n", address, last, now);
			last = now;
		}
	}
}

template <typename T>
inline unsigned long long TraceWord(const T& value) {
	if constexpr (std::is_pointer_v<T>) {
		return reinterpret_cast<unsigned long long>(value);
	} else if constexpr (std::is_integral_v<T> || std::is_enum_v<T>) {
		return static_cast<unsigned long long>(value);
	} else {
		return 0xeeeeeeeeull;
	}
}

template <auto Function>
struct Traced;

template <typename R, typename... Args, R (*Function)(Args...)>
struct Traced<Function> {
	static inline const char* name = "?";
	// A void export still leaves a defined 0 in the result register: the host's thunk hands it to
	// the game as RAX, and the host's prototype of the same export may return a status.
	using Result = std::conditional_t<std::is_void_v<R>, uint64_t, R>;
	// KYTY_GPU_HOST_IMPL=<names,...> (diagnostic A/B): the named exports run the HOST process's own
	// implementation (dlsym "sce<Name>", same C prototype, host pointers) instead of Kyty's.
	static void* HostImpl() {
		static void* const impl = []() -> void* {
			const char* list  = std::getenv("KYTY_GPU_HOST_IMPL");
			const char* colon = std::strrchr(name, ':');
			const char* base  = colon != nullptr ? colon + 1 : name;
			if (list == nullptr) {
				return nullptr;
			}
			const std::string wanted = std::string(",") + list + ",";
			if (wanted.find(std::string(",") + base + ",") == std::string::npos) {
				return nullptr;
			}
			void* found = dlsym(RTLD_DEFAULT, (std::string("sce") + base).c_str());
			std::fprintf(stderr, "[kyty call] %s -> host implementation %p\n", name, found);
			return found;
		}();
		return impl;
	}
	static Result Invoke(Args... args) {
		if (void* impl = HostImpl(); impl != nullptr) [[unlikely]] {
			if constexpr (std::is_void_v<R>) {
				Call<R>(reinterpret_cast<R (*)(Args...)>(impl), args...);
				return 0;
			} else {
				return Call<R>(reinterpret_cast<R (*)(Args...)>(impl), args...);
			}
		}
		if (!TraceEnabled()) [[likely]] {
			if constexpr (std::is_void_v<R>) {
				Call<R>(Function, args...);
				return 0;
			} else {
				return Call<R>(Function, args...);
			}
		}
		unsigned long long words[8] = {};
		unsigned           count    = 0;
		((count < 8 ? (void)(words[count++] = TraceWord(args)) : (void)0), ...);
		if constexpr (std::is_void_v<R>) {
			Call<R>(Function, args...);
			std::fprintf(stderr, "[kyty call] %s(%llx, %llx, %llx, %llx, %llx, %llx)\n", name,
			             words[0], words[1], words[2], words[3], words[4], words[5]);
			TraceWatch();
			return 0;
		} else {
			const R result = Call<R>(Function, args...);
			std::fprintf(stderr, "[kyty call] %s(%llx, %llx, %llx, %llx, %llx, %llx) = %llx\n",
			             name, words[0], words[1], words[2], words[3], words[4], words[5],
			             TraceWord(result));
			TraceWatch();
			return result;
		}
	}
};

} // namespace Libs::GuestCall

#undef LIB_FUNC
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define LIB_FUNC(n, f)                                                                             \
	Libs::GuestCall::Traced<&f>::name = #f;                                                        \
	LIB_ADD(n, &Libs::GuestCall::Traced<&f>::Invoke, Loader::SymbolType::Func)

#endif

#endif /* EMULATOR_SRC_LIBS_GUESTCALLTRACE_H_ */
