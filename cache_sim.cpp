// ============================================================================
//  cache_sim.cpp — packed-nibble LRU, SIMD tag scan, zero alloc
// ============================================================================

#include "cache_sim.hpp"

#include <cstddef>
#include <cstdint>
#include <array>
#include <cstring>
#include <immintrin.h>
#ifdef __linux__
#include <sys/mman.h>
#endif

#ifdef CSOT_CHECK_ALLOCS
#include <new>
#include <iostream>
#include <cstdlib>
#ifdef __linux__
#include <sys/prctl.h>
#endif

thread_local bool g_hot_path_active = false;

namespace {
    alignas(4096) char g_pool[1024 * 1024 * 2]; // 2MB memory pool
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

void operator delete(void*) noexcept {}
void operator delete(void*, std::size_t) noexcept {}
void* operator new[](std::size_t size) { return operator new(size); }
void operator delete[](void*) noexcept {}
void operator delete[](void*, std::size_t) noexcept {}
#endif

// ============================================================================
// Packed-Nibble LRU — eliminates the 630 KiB lookup table entirely.
//
// Encoding: 8 way-indices packed as 4-bit nibbles in a uint32_t.
//   Nibble 0 (bits  3: 0) = MRU way
//   Nibble 7 (bits 31:28) = LRU way
//   Initial state: 0x76543210 (identity permutation)
//
// touch():  ~5 ALU ops, zero memory accesses, fully branchless.
// victim(): single shift + mask.
// ============================================================================
namespace PackedLRU {

    // Move `way` to MRU position. Branchless, no memory accesses.
    [[gnu::always_inline]]
    static inline std::uint32_t touch(std::uint32_t state, int way) {
        std::uint32_t w = static_cast<std::uint32_t>(way);

        // Broadcast `way` to all nibbles, XOR with state → zero nibble at match
        std::uint32_t cmp = state ^ (w * 0x11111111u);

        // Detect the zero nibble (borrow-safe: OR in high bit prevents propagation)
        std::uint32_t zero = ~((cmp | 0x88888888u) - 0x11111111u) & 0x88888888u;
        int shift = __builtin_ctz(zero) & ~3; // nibble position × 4

        // Split state around the matched nibble and reconstruct:
        //   [upper nibbles unchanged] [lower nibbles shifted up by 1] [way at MRU]
        std::uint32_t lower_mask = (1u << shift) - 1;
        std::uint32_t lower = (state & lower_mask) << 4;
        std::uint32_t full_mask = static_cast<std::uint32_t>((1ULL << (shift + 4)) - 1);
        std::uint32_t upper = state & ~full_mask;

        return upper | lower | w;
    }

    // Return the LRU way (highest nibble).
    [[gnu::always_inline]]
    static inline int victim(std::uint32_t state) {
        return static_cast<int>((state >> 28) & 0xF);
    }

} // namespace PackedLRU


namespace {

constexpr int log2_of(int v) { int r = 0; while ((1 << r) < v) ++r; return r; }

// ============================================================================
// Level template – SoA layout, per‑set LRU state as packed-nibble uint32_t
// ============================================================================
template <int SETS, int WAYS>
struct Level {
    static constexpr int NUM_SETS = SETS;
    static constexpr int NUM_WAYS = WAYS;
    static constexpr int INDEX_BITS = log2_of(SETS);
    static constexpr std::uint64_t INDEX_MASK = SETS - 1;

    alignas(64) std::array<std::uint64_t, SETS * WAYS> tag{};
    alignas(64) std::array<std::uint8_t, SETS> valid{};
    alignas(64) std::array<std::uint8_t, SETS> dirty{};
    alignas(64) std::array<std::uint32_t, SETS> lru{};

    void init() {
        tag.fill(~0ULL); // Invalid tags: all 1s, physically impossible, never matches
        valid.fill(0);
        dirty.fill(0);
        lru.fill(0x76543210u); // Identity permutation: way 0=MRU, way 7=LRU
    }

    static constexpr int set_of(std::uint64_t blk) {
        return static_cast<int>(blk & INDEX_MASK);
    }

    // -------------------------------------------------------------------
    // SIMD branchless tag scan — AVX2 → SSE4.1 → scalar fallback
    // -------------------------------------------------------------------
    template <int W>
    [[gnu::always_inline]] static inline void check_way(const std::uint64_t* tag, std::uint64_t t, unsigned& m) {
        m |= static_cast<unsigned>(tag[W] == t) << W;
    }

    template <int... Ws>
    [[gnu::always_inline]] static inline unsigned check_all_ways(const std::uint64_t* tag, std::uint64_t t, std::integer_sequence<int, Ws...>) {
        unsigned m = 0;
        (check_way<Ws>(tag, t, m), ...);
        return m;
    }

    [[gnu::always_inline]] int find_way(int si, std::uint64_t t) const {
        std::size_t base = static_cast<std::size_t>(si) * WAYS;
        
#if defined(__AVX2__)
        __m256i key;
        __asm__ ("vmovq %1, %%xmm0\n\t"
                 "vpbroadcastq %%xmm0, %0"
                 : "=x"(key) : "r"(t) : "xmm0");
        
        __m256i a = _mm256_load_si256(reinterpret_cast<const __m256i*>(&tag[base]));
        __m256i b = _mm256_load_si256(reinterpret_cast<const __m256i*>(&tag[base + 4]));
        
        unsigned m = (unsigned)_mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(a, key)))
                   | ((unsigned)_mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(b, key))) << 4);
#elif defined(__SSE4_1__)
        __m128i key = _mm_set1_epi64x(t);
        __m128i a = _mm_load_si128(reinterpret_cast<const __m128i*>(&tag[base + 0]));
        __m128i b = _mm_load_si128(reinterpret_cast<const __m128i*>(&tag[base + 2]));
        __m128i c = _mm_load_si128(reinterpret_cast<const __m128i*>(&tag[base + 4]));
        __m128i d = _mm_load_si128(reinterpret_cast<const __m128i*>(&tag[base + 6]));
        unsigned m = (unsigned)_mm_movemask_pd(_mm_castsi128_pd(_mm_cmpeq_epi64(a, key)))
                   | ((unsigned)_mm_movemask_pd(_mm_castsi128_pd(_mm_cmpeq_epi64(b, key))) << 2)
                   | ((unsigned)_mm_movemask_pd(_mm_castsi128_pd(_mm_cmpeq_epi64(c, key))) << 4)
                   | ((unsigned)_mm_movemask_pd(_mm_castsi128_pd(_mm_cmpeq_epi64(d, key))) << 6);
#else
        unsigned m = check_all_ways(tag.data() + base, t, std::make_integer_sequence<int, WAYS>{});
#endif
        return m ? __builtin_ctz(m) : -1;
    }

    [[gnu::always_inline]] void touch_mru(int si, int way) {
        lru[si] = PackedLRU::touch(lru[si], way);
    }

    [[gnu::always_inline]] int victim_way(int si) const {
        unsigned invalid = static_cast<unsigned>(~valid[si]) & 0xFF;
        if (__builtin_expect(invalid, 0)) {
            return __builtin_ctz(invalid);
        }
        return PackedLRU::victim(lru[si]);
    }

    [[gnu::always_inline]] void set_line(int si, int way, bool v, bool d, std::uint64_t t) {
        tag[static_cast<std::size_t>(si) * WAYS + way] = t;
        
        std::uint8_t mask = 1 << way;
        valid[si] = (valid[si] & ~mask) | (-static_cast<int>(v) & mask);
        dirty[si] = (dirty[si] & ~mask) | (-static_cast<int>(d) & mask);
        
        touch_mru(si, way);
    }
};

using L1 = Level<64, 8>;
using L2 = Level<512, 8>;

// ============================================================================
class BaselineCacheSim final : public csot::CacheSim {
public:
    void on_init() override {
        l1_.init();
        l2_.init();

#ifdef __linux__
        // Request transparent huge pages to reduce TLB pressure.
        // Advisory: silently ignored if THP is disabled or alignment is wrong.
        auto try_hugepage = [](void* addr, std::size_t len) {
            auto page = reinterpret_cast<std::uintptr_t>(addr) & ~4095ULL;
            auto adj_len = len + (reinterpret_cast<std::uintptr_t>(addr) - page);
            madvise(reinterpret_cast<void*>(page), adj_len, MADV_HUGEPAGE);
        };
        try_hugepage(&l1_, sizeof(l1_));
        try_hugepage(&l2_, sizeof(l2_));
#endif
    }

    [[gnu::always_inline]] void process_one(const csot::MemAccess* __restrict acc, 
                                            std::uint64_t& c_writes,
                                            std::uint64_t& c_l1_misses,
                                            std::uint64_t& c_l2_hits,
                                            std::uint64_t& c_dirty_writebacks) {
        // Prefetch 8 accesses ahead in the input stream
        __builtin_prefetch(acc + 8, 0, 0);

        const std::uint32_t wr = acc->is_write;
        c_writes += wr;

        const std::uint64_t b = acc->address >> 6;
        const int s1 = L1::set_of(b);

        int w1 = l1_.find_way(s1, b);
        bool l1_hit = (w1 >= 0);

        if (__builtin_expect(l1_hit, 1)) {
            l1_.touch_mru(s1, w1);
            l1_.dirty[s1] |= static_cast<std::uint8_t>(wr << w1);
            return;
        }

        ++c_l1_misses;
        __asm__ volatile("" : "+r"(c_l1_misses));

        const int s2 = L2::set_of(b);

        int w2 = l2_.find_way(s2, b);
        bool l2_hit = (w2 >= 0);
        c_l2_hits += l2_hit;

        if (l2_hit) {
            l2_.touch_mru(s2, w2);
        } else {
            int victim = l2_.victim_way(s2);
            c_dirty_writebacks += ((l2_.valid[s2] & l2_.dirty[s2]) >> victim) & 1;
            l2_.set_line(s2, victim, true, false, b);
        }

        int v1 = l1_.victim_way(s1);
        if (((l1_.valid[s1] & l1_.dirty[s1]) >> v1) & 1) {
            std::uint64_t bv = l1_.tag[static_cast<std::size_t>(s1) * L1::NUM_WAYS + v1];
            int s2v = L2::set_of(bv);

            int wv = l2_.find_way(s2v, bv);
            if (wv >= 0) {
                l2_.dirty[s2v] |= (1 << wv);
            } else {
                int vv = l2_.victim_way(s2v);
                c_dirty_writebacks += ((l2_.valid[s2v] & l2_.dirty[s2v]) >> vv) & 1;
                l2_.set_line(s2v, vv, true, true, bv);
            }
        }

        l1_.set_line(s1, v1, true, wr, b);
    }

    template <size_t... Is>
    [[gnu::always_inline]] inline void process_batch_impl(const csot::MemAccess* __restrict acc,
                                                          std::uint64_t& c_writes,
                                                          std::uint64_t& c_l1_misses,
                                                          std::uint64_t& c_l2_hits,
                                                          std::uint64_t& c_dirty_writebacks,
                                                          std::index_sequence<Is...>) {
        (process_one(acc + Is, c_writes, c_l1_misses, c_l2_hits, c_dirty_writebacks), ...);
    }
    
    template <size_t N>
    [[gnu::always_inline]] inline void process_batch(const csot::MemAccess* __restrict acc,
                                                     std::uint64_t& c_writes,
                                                     std::uint64_t& c_l1_misses,
                                                     std::uint64_t& c_l2_hits,
                                                     std::uint64_t& c_dirty_writebacks) {
        process_batch_impl(acc, c_writes, c_l1_misses, c_l2_hits, c_dirty_writebacks, std::make_index_sequence<N>{});
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

        constexpr std::size_t BATCH_SIZE = 8;
        const csot::MemAccess* const end_batch = acc + (n / BATCH_SIZE) * BATCH_SIZE;
        for (; acc < end_batch; acc += BATCH_SIZE) {
            process_batch<BATCH_SIZE>(acc, c_writes, c_l1_misses, c_l2_hits, c_dirty_writebacks);
        }

        const csot::MemAccess* const end = acc + (n % BATCH_SIZE);
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
    return new BaselineCacheSim();
}