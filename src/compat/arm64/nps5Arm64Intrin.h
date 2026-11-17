#ifndef NPS5_ARM64_INTRIN_H_
#define NPS5_ARM64_INTRIN_H_

// nps5: native arm64 build of Kyty's GPU stack (kyty_gpu.dylib). The x86 intrinsics the graphics
// code uses outside of real SIMD work (spin-wait hints, time stamps, prefetch, CPUID) mapped to
// their arm64 counterparts. Reached through the immintrin.h/x86intrin.h/intrin.h/xmmintrin.h/
// emmintrin.h wrappers in this directory, which precede the compiler's own on arm64 only.

#if defined(__aarch64__)

#include <stdint.h>

static inline __attribute__((always_inline)) void _mm_pause(void) {
	__asm__ volatile("isb");
}

// The virtual counter (24 MHz on Apple silicon): monotonic, cheap, coarser than the x86 TSC.
static inline __attribute__((always_inline)) uint64_t nps5_rdtsc(void) {
	uint64_t value;
	__asm__ volatile("mrs %0, cntvct_el0" : "=r"(value));
	return value;
}
#define __rdtsc() nps5_rdtsc()
static inline __attribute__((always_inline)) uint64_t __rdtscp(unsigned int* aux) {
	if (aux != 0) {
		*aux = 0;
	}
	return nps5_rdtsc();
}

#define _MM_HINT_T0 3
#define _MM_HINT_T1 2
#define _MM_HINT_T2 1
#define _MM_HINT_NTA 0
#define _mm_prefetch(address, hint) __builtin_prefetch((address), 0, (hint))

static inline void nps5_cpuid(int regs[4], int leaf) {
	(void)leaf;
	regs[0] = regs[1] = regs[2] = regs[3] = 0;
}
#define __cpuid(regs, leaf) nps5_cpuid((regs), (leaf))
#define __cpuidex(regs, leaf, subleaf) nps5_cpuid((regs), (leaf))

static inline __attribute__((always_inline)) void _mm_lfence(void) {}
static inline __attribute__((always_inline)) void _mm_mfence(void) {
	__sync_synchronize();
}
static inline __attribute__((always_inline)) void _mm_sfence(void) {}

#endif // __aarch64__

#endif /* NPS5_ARM64_INTRIN_H_ */
