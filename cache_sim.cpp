// ============================================================================
//  cache_sim.cpp — final branch‑free LRU, SIMD tag scan, zero alloc
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

#include <algorithm>

// =======================================================================
// Hybrid TableLRU Configuration
// =======================================================================

namespace TableLRU {
    constexpr int c_popcount(std::uint32_t x) {
        int c = 0;
        while (x) { c += x & 1; x >>= 1; }
        return c;
    }

    constexpr int c_ctz(std::uint32_t x) {
        if (!x) return 32;
        int c = 0;
        while ((x & 1) == 0) { x >>= 1; ++c; }
        return c;
    }

    // Fast Lehmer encoding
    constexpr std::uint16_t encode_fast(std::uint64_t v) {
        std::uint32_t seen = 0;
        std::uint32_t code = 0;
        std::uint32_t p;

        p = v & 0xFF; seen |= (1u << p); code += p * 5040; v >>= 8;
        p = v & 0xFF; code += (p - c_popcount(seen & ((1u << p) - 1u))) * 720; seen |= (1u << p); v >>= 8;
        p = v & 0xFF; code += (p - c_popcount(seen & ((1u << p) - 1u))) * 120; seen |= (1u << p); v >>= 8;
        p = v & 0xFF; code += (p - c_popcount(seen & ((1u << p) - 1u))) * 24; seen |= (1u << p); v >>= 8;
        p = v & 0xFF; code += (p - c_popcount(seen & ((1u << p) - 1u))) * 6; seen |= (1u << p); v >>= 8;
        p = v & 0xFF; code += (p - c_popcount(seen & ((1u << p) - 1u))) * 2; seen |= (1u << p); v >>= 8;
        p = v & 0xFF; code += (p - c_popcount(seen & ((1u << p) - 1u)));
        return static_cast<std::uint16_t>(code);
    }

    // Fast Lehmer decoding completely unrolled
    // Use uint32_t to force fast imul instead of slow 16-bit idiv!
    constexpr void decode_fast(std::uint32_t code, std::uint8_t p[8]) {
        std::uint32_t avail = 0xFF; std::uint32_t idx; std::uint32_t tmp;
        idx = code / 5040; code %= 5040; tmp = avail; for (std::uint32_t k = 0; k < idx; ++k) tmp &= tmp - 1; p[0] = static_cast<std::uint8_t>(c_ctz(tmp)); avail &= ~(1u << p[0]);
        idx = code / 720; code %= 720; tmp = avail; for (std::uint32_t k = 0; k < idx; ++k) tmp &= tmp - 1; p[1] = static_cast<std::uint8_t>(c_ctz(tmp)); avail &= ~(1u << p[1]);
        idx = code / 120; code %= 120; tmp = avail; for (std::uint32_t k = 0; k < idx; ++k) tmp &= tmp - 1; p[2] = static_cast<std::uint8_t>(c_ctz(tmp)); avail &= ~(1u << p[2]);
        idx = code / 24; code %= 24; tmp = avail; for (std::uint32_t k = 0; k < idx; ++k) tmp &= tmp - 1; p[3] = static_cast<std::uint8_t>(c_ctz(tmp)); avail &= ~(1u << p[3]);
        idx = code / 6; code %= 6; tmp = avail; for (std::uint32_t k = 0; k < idx; ++k) tmp &= tmp - 1; p[4] = static_cast<std::uint8_t>(c_ctz(tmp)); avail &= ~(1u << p[4]);
        idx = code / 2; code %= 2; tmp = avail; for (std::uint32_t k = 0; k < idx; ++k) tmp &= tmp - 1; p[5] = static_cast<std::uint8_t>(c_ctz(tmp)); avail &= ~(1u << p[5]);
        idx = code; tmp = avail; for (std::uint32_t k = 0; k < idx; ++k) tmp &= tmp - 1; p[6] = static_cast<std::uint8_t>(c_ctz(tmp)); avail &= ~(1u << p[6]);
        p[7] = static_cast<std::uint8_t>(c_ctz(avail));
    }

    constexpr int CONSTEXPR_STATES = 5040; // 1/8th of the table (compile-time)

    struct PartialTable {
        std::uint16_t next[CONSTEXPR_STATES][8];
        std::uint8_t victim[CONSTEXPR_STATES];
    };

    constexpr PartialTable compute_partial() {
        PartialTable t{};
        for (std::uint32_t state = 0; state < CONSTEXPR_STATES; ++state) {
            std::uint8_t perm[8];
            decode_fast(state, perm);
            t.victim[state] = perm[7];

            std::uint64_t v = 0;
            for (int i = 0; i < 8; ++i) v |= static_cast<std::uint64_t>(perm[i]) << (i * 8);

            std::uint8_t inv[8];
            for (int i = 0; i < 8; ++i) inv[perm[i]] = i;

            for (int w = 0; w < 8; ++w) {
                int pos = inv[w];
                std::uint64_t lower_mask = (1ULL << (pos * 8)) - 1;
                std::uint64_t upper_mask = (pos == 7) ? 0 : (~0ULL << ((pos + 1) * 8));
                std::uint64_t new_v = static_cast<std::uint64_t>(w) | ((v & lower_mask) << 8) | (v & upper_mask);
                t.next[state][w] = encode_fast(new_v);
            }
        }
        return t;
    }

    constexpr PartialTable PARTIAL = compute_partial();

    // ── Runtime-only encode/decode using hardware POPCNT/TZCNT ──────────
    // These replace the constexpr scalar loops that were ALL the perf hotspots.
    // __builtin_popcount → single POPCNT instruction (1 cycle vs ~10-cycle loop)
    // __builtin_ctz      → single TZCNT instruction  (3 cycles vs ~8-cycle loop)

    [[gnu::always_inline]] inline std::uint16_t encode_rt(std::uint64_t v) {
        std::uint32_t seen = 0, code = 0, p;
        p = v & 0xFF; seen |= (1u << p); code += p * 5040; v >>= 8;
        p = v & 0xFF; code += (p - __builtin_popcount(seen & ((1u << p) - 1u))) * 720; seen |= (1u << p); v >>= 8;
        p = v & 0xFF; code += (p - __builtin_popcount(seen & ((1u << p) - 1u))) * 120; seen |= (1u << p); v >>= 8;
        p = v & 0xFF; code += (p - __builtin_popcount(seen & ((1u << p) - 1u))) * 24;  seen |= (1u << p); v >>= 8;
        p = v & 0xFF; code += (p - __builtin_popcount(seen & ((1u << p) - 1u))) * 6;   seen |= (1u << p); v >>= 8;
        p = v & 0xFF; code += (p - __builtin_popcount(seen & ((1u << p) - 1u))) * 2;   seen |= (1u << p); v >>= 8;
        p = v & 0xFF; code += (p - __builtin_popcount(seen & ((1u << p) - 1u)));
        return static_cast<std::uint16_t>(code);
    }

    [[gnu::always_inline]] inline void decode_rt(std::uint32_t code, std::uint8_t p[8]) {
        std::uint32_t avail = 0xFF, idx, tmp;
        idx = code / 5040; code %= 5040; tmp = avail; for (std::uint32_t k = 0; k < idx; ++k) tmp &= tmp - 1; p[0] = static_cast<std::uint8_t>(__builtin_ctz(tmp)); avail &= ~(1u << p[0]);
        idx = code / 720;  code %= 720;  tmp = avail; for (std::uint32_t k = 0; k < idx; ++k) tmp &= tmp - 1; p[1] = static_cast<std::uint8_t>(__builtin_ctz(tmp)); avail &= ~(1u << p[1]);
        idx = code / 120;  code %= 120;  tmp = avail; for (std::uint32_t k = 0; k < idx; ++k) tmp &= tmp - 1; p[2] = static_cast<std::uint8_t>(__builtin_ctz(tmp)); avail &= ~(1u << p[2]);
        idx = code / 24;   code %= 24;   tmp = avail; for (std::uint32_t k = 0; k < idx; ++k) tmp &= tmp - 1; p[3] = static_cast<std::uint8_t>(__builtin_ctz(tmp)); avail &= ~(1u << p[3]);
        idx = code / 6;    code %= 6;    tmp = avail; for (std::uint32_t k = 0; k < idx; ++k) tmp &= tmp - 1; p[4] = static_cast<std::uint8_t>(__builtin_ctz(tmp)); avail &= ~(1u << p[4]);
        idx = code / 2;    code %= 2;    tmp = avail; for (std::uint32_t k = 0; k < idx; ++k) tmp &= tmp - 1; p[5] = static_cast<std::uint8_t>(__builtin_ctz(tmp)); avail &= ~(1u << p[5]);
        idx = code;                      tmp = avail; for (std::uint32_t k = 0; k < idx; ++k) tmp &= tmp - 1; p[6] = static_cast<std::uint8_t>(__builtin_ctz(tmp)); avail &= ~(1u << p[6]);
        p[7] = static_cast<std::uint8_t>(__builtin_ctz(avail));
    }

    alignas(4096) std::uint16_t next_state[40320][8];
    alignas(4096) std::uint8_t victim_way[40320];
    bool initialized = false;

    void init() {
        if (initialized) return;

        // Copy compile-time portion from .rodata
        for (int s = 0; s < CONSTEXPR_STATES; ++s) {
            for (int w = 0; w < 8; ++w) next_state[s][w] = PARTIAL.next[s][w];
            victim_way[s] = PARTIAL.victim[s];
        }

        // Compute remainder using hardware POPCNT/TZCNT (not the scalar loops)
        for (std::uint32_t state = CONSTEXPR_STATES; state < 40320; ++state) {
            std::uint8_t perm[8];
            decode_rt(state, perm);
            victim_way[state] = perm[7];

            std::uint64_t v = 0;
            for (int i = 0; i < 8; ++i) v |= static_cast<std::uint64_t>(perm[i]) << (i * 8);

            std::uint8_t inv[8];
            for (int i = 0; i < 8; ++i) inv[perm[i]] = i;

            for (int w = 0; w < 8; ++w) {
                int pos = inv[w];
                std::uint64_t lower_mask = (1ULL << (pos * 8)) - 1;
                std::uint64_t upper_mask = (pos == 7) ? 0 : (~0ULL << ((pos + 1) * 8));
                std::uint64_t new_v = static_cast<std::uint64_t>(w) | ((v & lower_mask) << 8) | (v & upper_mask);
                next_state[state][w] = encode_rt(new_v);
            }
        }
        initialized = true;
    }
}

namespace {

constexpr int log2_of(int v) { int r = 0; while ((1 << r) < v) ++r; return r; }

// ============================================================================
// Level template – SoA layout, per‑set LRU state stored as uint32_t (age‑based)
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
    alignas(64) std::array<std::uint16_t, SETS> lru{};

    void init() {
        tag.fill(~0ULL); // Initialize with all 1s (physically impossible tag, acts as invalid)
        valid.fill(0);
        dirty.fill(0);
        lru.fill(0);
    }

    static constexpr int set_of(std::uint64_t blk) {
        return static_cast<int>(blk & INDEX_MASK);
    }
    static constexpr std::uint64_t tag_of(std::uint64_t blk) {
        return blk >> INDEX_BITS;
    }

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
        
        // Inline asm: vmovq GP->XMM then vpbroadcastq XMM->YMM
        // Avoids the compiler's stack spill (mov %rax,(%rsp); vpbroadcastq (%rsp),%ymm0)
#if defined(__AVX2__)
        __m256i key;
        __asm__ ("vmovq %1, %%xmm0\n\t"
                 "vpbroadcastq %%xmm0, %0"
                 : "=x"(key) : "r"(t) : "xmm0");
        
        // The tag array is alignas(64), so base is naturally 32-byte aligned. Use _mm256_load_si256!
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
        // No need for 'm &= meta[si].valid' because invalid tags are ~0ULL, which will never match 't'
        return m ? __builtin_ctz(m) : -1;
    }

    [[gnu::always_inline]] void touch_mru(int si, int way) {
        lru[si] = TableLRU::next_state[lru[si]][way];
    }

    [[gnu::always_inline]] int victim_way(int si) const {
        unsigned invalid = static_cast<unsigned>(~valid[si]) & 0xFF;
        if (__builtin_expect(invalid, 0)) {
            return __builtin_ctz(invalid);
        }
        return TableLRU::victim_way[lru[si]];
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
        TableLRU::init();
        l1_.init();
        l2_.init();
    }

    [[gnu::always_inline]] void process_one(const csot::MemAccess* __restrict acc, 
                                            std::uint64_t& c_writes,
                                            std::uint64_t& c_l1_misses,
                                            std::uint64_t& c_l2_hits,
                                            std::uint64_t& c_dirty_writebacks) {
        __builtin_prefetch(acc + 8, 0, 0);

        const std::uint32_t wr = acc->is_write;
        c_writes += wr;

        const std::uint64_t b = acc->address >> 6;
        const int s1 = L1::set_of(b);

        // Prefetch this access's LRU table row BEFORE the tag comparison.
        // find_way takes ~10 cycles (SIMD), giving the prefetch time to land.
        __builtin_prefetch(&TableLRU::next_state[l1_.lru[s1]], 0, 0);

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

        // Prefetch L2 LRU table row before the tag comparison
        __builtin_prefetch(&TableLRU::next_state[l2_.lru[s2]], 0, 0);

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
        std::uint64_t c_l1_misses = 0;  // Track MISSES, not hits — removes incq from L1 hot path
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