#ifndef EMULATOR_SRC_COMMON_GUESTADDRESS_H_
#define EMULATOR_SRC_COMMON_GUESTADDRESS_H_

// nps5: guest addresses in a host process whose guest memory is not at its guest address
// (kyty_gpu.dylib in the nativeps5 runtime on arm64 macOS: host = guest + bias, one constant for
// the whole guest space, 0 in kyty_emulator and AstroHybrid on Windows where both are the same).
//
// Rule: every address Kyty keeps, compares, hashes or writes back to the guest is a GUEST address
// (PM4 words, resource bases, labels, tracker pages). The bias is added only where bytes are
// touched, through these helpers. Null stays null.

#include <cstdint>

namespace Guest {

extern uint64_t
    g_bias; // set once by the hybrid bridge before the renderer starts (guestAddress.cpp)

// Calls the guest function at guest address `function` (up to six integer arguments, System V
// order) and returns its RAX. Null where guest code is called directly (kyty_emulator,
// AstroHybrid).
extern uint64_t (*g_call_guest)(uint64_t function, const uint64_t* args6);

[[nodiscard]] inline uint64_t CallGuest(uint64_t function, uint64_t a0 = 0, uint64_t a1 = 0,
                                        uint64_t a2 = 0, uint64_t a3 = 0, uint64_t a4 = 0,
                                        uint64_t a5 = 0) {
	const uint64_t args[6] = {a0, a1, a2, a3, a4, a5};
	return g_call_guest(function, args);
}

[[nodiscard]] inline uint64_t Bias() noexcept {
	return g_bias;
}

// Host pointer to the bytes at guest address `guest`.
template <typename T = void>
[[nodiscard]] inline T* Ptr(uint64_t guest) noexcept {
	return guest != 0 ? reinterpret_cast<T*>(guest + g_bias) : nullptr;
}

// Host pointer for a pointer-typed value that holds a guest address (a pointer read from guest
// data or passed by guest code).
template <typename T>
[[nodiscard]] inline T* Host(T* guest_valued) noexcept {
	return guest_valued != nullptr
	           ? reinterpret_cast<T*>(reinterpret_cast<uint64_t>(guest_valued) + g_bias)
	           : nullptr;
}

// The guest address of host pointer `host` (into guest memory).
[[nodiscard]] inline uint64_t Addr(const volatile void* host) noexcept {
	return host != nullptr ? reinterpret_cast<uint64_t>(host) - g_bias : 0;
}

// The guest address of `host` when it points into the host view of the guest space
// [bias, 2 * bias) (the nativeps5 arena: one contiguous view, bias 2^40); any other pointer (a
// host-side copy) is returned as it is. For values that upstream takes from a pointer that is
// usually, but not always, guest memory (shader code spans).
[[nodiscard]] inline uint64_t AddrIfGuest(const volatile void* host) noexcept {
	const auto value = reinterpret_cast<uint64_t>(host);
	return g_bias != 0 && value >= g_bias && value - g_bias < g_bias ? value - g_bias : value;
}

// The pointer-typed guest value for host pointer `host` (to store into guest data or return to
// guest code).
template <typename T>
[[nodiscard]] inline T* Value(T* host) noexcept {
	return host != nullptr ? reinterpret_cast<T*>(reinterpret_cast<uint64_t>(host) - g_bias)
	                       : nullptr;
}

} // namespace Guest

#endif /* EMULATOR_SRC_COMMON_GUESTADDRESS_H_ */
