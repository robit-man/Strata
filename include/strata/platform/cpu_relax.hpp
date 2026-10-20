#pragma once

#include <atomic>

#if defined(_MSC_VER) || defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

namespace strata::platform {

inline void cpu_relax() noexcept {
#if defined(_MSC_VER) || defined(__x86_64__) || defined(__i386__)
    _mm_pause();
#elif defined(__aarch64__)
    __asm__ volatile("yield" ::: "memory");
#else
    std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
}

// Publish host writes before a CUDA kernel observes a mapped doorbell.  x86 uses
// SFENCE for its write-combining mappings; AArch64 needs an explicit store
// barrier because its memory model is weaker.
inline void store_fence() noexcept {
#if defined(_MSC_VER) || defined(__x86_64__) || defined(__i386__)
    _mm_sfence();
#elif defined(__aarch64__)
    __asm__ volatile("dmb ishst" ::: "memory");
#else
    std::atomic_thread_fence(std::memory_order_seq_cst);
#endif
}

}  // namespace strata::platform
