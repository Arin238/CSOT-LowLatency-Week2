// ============================================================================
//  cache_shift_reg.cpp
//
//  "Shift Register" Architecture
//  - 1 Line = 64 bits (Valid: bit 63, Dirty: bit 62, Tag/Block Addr: 0-61)
//  - 1 Set = 8 Lines (Exactly 64 bytes = 1 physical cache line)
//  - MRU is ALWAYS lines[0]. LRU is ALWAYS lines[7].
//  - Shifts maintain implicit LRU.
// ============================================================================

#include "cache_sim.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <immintrin.h>

#ifdef CSOT_CHECK_ALLOCS
#include <new>
#include <iostream>
#include <cstdlib>
#ifdef __linux__
#include <sys/prctl.h>
#endif

thread_local bool g_hot_path_active = false;

namespace {
    alignas(4096) char g_pool[1024 * 1024 * 2];
    std::size_t g_pool_offset = 0;
}

[[gnu::malloc, gnu::alloc_size(1), gnu::assume_aligned(64), gnu::hot]]
void* operator new(std::size_t size) {
    if (g_hot_path_active) {
        std::cerr << "\n[ALLOCATION DETECTED ON HOT PATH] Size: " << size << " bytes\n";
        std::abort();
    }
    std::size_t aligned_size = (size + 63) & ~63;
    std::size_t offset = g_pool_offset;
    g_pool_offset += aligned_size;
    return __builtin_assume_aligned(g_pool + offset, 64);
}
[[gnu::malloc, gnu::alloc_size(1), gnu::assume_aligned(64), gnu::hot]]
void* operator new(std::size_t size, std::align_val_t) { return operator new(size); }
void operator delete(void*) noexcept {}
void operator delete(void*, std::size_t) noexcept {}
void operator delete(void*, std::align_val_t) noexcept {}
void operator delete(void*, std::size_t, std::align_val_t) noexcept {}
void* operator new[](std::size_t size) { return operator new(size); }
void* operator new[](std::size_t size, std::align_val_t) { return operator new(size); }
void operator delete[](void*) noexcept {}
void operator delete[](void*, std::size_t) noexcept {}
void operator delete[](void*, std::align_val_t) noexcept {}
void operator delete[](void*, std::size_t, std::align_val_t) noexcept {}
#endif

namespace {

constexpr std::uint64_t VALID_BIT = 1ULL << 63;
constexpr std::uint64_t DIRTY_BIT = 1ULL << 62;
constexpr std::uint64_t ADDR_MASK = ~(VALID_BIT | DIRTY_BIT);

template <int SETS, int WAYS>
struct Level {
    static constexpr int NUM_SETS = SETS;
    static constexpr int NUM_WAYS = WAYS;
    static constexpr int INDEX_MASK = SETS - 1;

    // lines[s][0] is MRU, lines[s][WAYS-1] is LRU
    // 8x uint64_t = 64 bytes. Perfectly cache aligned per set.
    alignas(64) std::uint64_t lines[SETS][WAYS];

    void init() {
        std::memset(lines, 0, sizeof(lines));
    }

    [[gnu::always_inline]] static constexpr int set_of(std::uint64_t blk) {
        return static_cast<int>(blk & INDEX_MASK);
    }

    // Find the way index. Returns -1 if miss.
    [[gnu::always_inline]] int find_way(int si, std::uint64_t target) const {
#if defined(__AVX2__)
        __m256i t = _mm256_set1_epi64x(target);
        __m256i d_mask = _mm256_set1_epi64x(~DIRTY_BIT);
        
        __m256i a = _mm256_load_si256((const __m256i*)&lines[si][0]);
        __m256i a_cmp = _mm256_cmpeq_epi64(_mm256_and_si256(a, d_mask), t);
        unsigned m1 = static_cast<unsigned>(_mm256_movemask_pd(_mm256_castsi256_pd(a_cmp)));
        
        // MRU Short-Circuit: Cache hits are heavily skewed toward the most recently used lines.
        // By branching here, we avoid the 2nd AVX2 load & compare for >80% of hits.
        if (__builtin_expect(m1 != 0, 1)) {
            return __builtin_ctz(m1);
        }
        
        __m256i c = _mm256_load_si256((const __m256i*)&lines[si][4]);
        __m256i c_cmp = _mm256_cmpeq_epi64(_mm256_and_si256(c, d_mask), t);
        unsigned m2 = static_cast<unsigned>(_mm256_movemask_pd(_mm256_castsi256_pd(c_cmp)));
                    
        return m2 ? 4 + __builtin_ctz(m2) : -1;
#else
        for (int w = 0; w < WAYS; ++w) {
            if ((lines[si][w] & ~DIRTY_BIT) == target) return w;
        }
        return -1;
#endif
    }

    // Shifts elements 0..(k-1) right by 1, and inserts 'line' at 0.
    [[gnu::always_inline]] void promote(int si, int k, std::uint64_t line) {
        // A switch perfectly forces a jump table / conditional moves without loops
        switch (k) {
            case 7: lines[si][7] = lines[si][6]; [[fallthrough]];
            case 6: lines[si][6] = lines[si][5]; [[fallthrough]];
            case 5: lines[si][5] = lines[si][4]; [[fallthrough]];
            case 4: lines[si][4] = lines[si][3]; [[fallthrough]];
            case 3: lines[si][3] = lines[si][2]; [[fallthrough]];
            case 2: lines[si][2] = lines[si][1]; [[fallthrough]];
            case 1: lines[si][1] = lines[si][0]; [[fallthrough]];
            case 0: lines[si][0] = line;
        }
    }
};

using L1 = Level<64, 8>;
using L2 = Level<512, 8>;

class ShiftRegCacheSim final : public csot::CacheSim {
public:
    void on_init() override {
        l1_.init();
        l2_.init();
    }

    [[gnu::always_inline, gnu::hot]]
    void process_one(const csot::MemAccess* __restrict acc,
                     std::uint64_t& c_writes,
                     std::uint64_t& c_l1_misses,
                     std::uint64_t& c_l2_hits,
                     std::uint64_t& c_dirty_writebacks) {
        
        // Prefetch the raw trace data
        __builtin_prefetch(acc + 8, 0, 0);

        // Look-ahead Prefetching: Prefetch the actual L1 cache line for a future access
        // This hides the 30-50 cycle main memory latency of the simulated cache fetch
        const std::uint64_t future_b = (acc + 4)->address >> 6;
        __builtin_prefetch(&l1_.lines[L1::set_of(future_b)][0], 0, 3);

        const std::uint32_t wr = acc->is_write;
        c_writes += wr;

        const std::uint64_t b = acc->address >> 6;
        const std::uint64_t target = b | VALID_BIT;
        const int s1 = L1::set_of(b);

        const int w1 = l1_.find_way(s1, target);

        if (__builtin_expect(w1 >= 0, 1)) {
            // L1 HIT
            std::uint64_t line = l1_.lines[s1][w1] | (static_cast<std::uint64_t>(wr) << 62);
            l1_.promote(s1, w1, line);
            return;
        }

        // L1 MISS
        ++c_l1_misses;
        __asm__ volatile("" : "+r"(c_l1_misses));

        const int s2 = L2::set_of(b);
        const int w2 = l2_.find_way(s2, target);
        
        if (w2 >= 0) {
            // L2 HIT
            ++c_l2_hits;
            l2_.promote(s2, w2, l2_.lines[s2][w2]);
        } else {
            // L2 MISS
            std::uint64_t v2 = l2_.lines[s2][7];
            c_dirty_writebacks += (v2 >> 63) & (v2 >> 62); // valid & dirty
            l2_.promote(s2, 7, target); // clean line from memory
        }

        // L1 Fill
        std::uint64_t v1 = l1_.lines[s1][7];
        if ((v1 >> 63) & (v1 >> 62)) {
            // L1 Writeback
            std::uint64_t vb = v1 & ADDR_MASK;
            std::uint64_t v_target = vb | VALID_BIT;
            const int s2v = L2::set_of(vb);
            const int w2v = l2_.find_way(s2v, v_target);
            
            if (w2v >= 0) {
                // L2 Hit for Writeback
                l2_.lines[s2v][w2v] |= DIRTY_BIT;
            } else {
                // L2 Miss for Writeback
                std::uint64_t v2v = l2_.lines[s2v][7];
                c_dirty_writebacks += (v2v >> 63) & (v2v >> 62);
                l2_.promote(s2v, 7, v_target | DIRTY_BIT);
            }
        }

        // Complete L1 Fill
        l1_.promote(s1, 7, target | (static_cast<std::uint64_t>(wr) << 62));
    }

    csot::CacheStats run(const csot::MemAccess* __restrict acc, std::size_t n) override {
#ifdef CSOT_CHECK_ALLOCS
        g_hot_path_active = true;
#ifdef __linux__
        prctl(PR_TASK_PERF_EVENTS_ENABLE);
#endif
#endif
        std::uint64_t c_writes = 0;
        std::uint64_t c_l1_misses = 0;
        std::uint64_t c_l2_hits = 0;
        std::uint64_t c_dirty_writebacks = 0;

        const csot::MemAccess* const end = acc + n;
        for (; acc < end; ++acc) {
            process_one(acc, c_writes, c_l1_misses, c_l2_hits, c_dirty_writebacks);
        }

#ifdef CSOT_CHECK_ALLOCS
#ifdef __linux__
        prctl(PR_TASK_PERF_EVENTS_DISABLE);
#endif
        g_hot_path_active = false;
#endif

        csot::CacheStats st{};
        st.writes = c_writes;
        st.reads = n - c_writes;
        st.l1_misses = c_l1_misses;
        st.l1_hits = n - c_l1_misses;
        st.l2_hits = c_l2_hits;
        st.l2_misses = c_l1_misses - c_l2_hits;
        st.dirty_writebacks = c_dirty_writebacks;
        return st;
    }

private:
    L1 l1_;
    L2 l2_;
};

} // namespace

extern "C" csot::CacheSim* create_cache_sim() {
    return new ShiftRegCacheSim();
}
