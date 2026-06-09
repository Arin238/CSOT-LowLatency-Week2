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

namespace TableLRU {
    // Precomputed factorials for Lehmer code
    constexpr int fact[8] = {1, 1, 2, 6, 24, 120, 720, 5040};
    constexpr int kFact[7] = {5040, 720, 120, 24, 6, 2, 1}; // for encode_fast

    // Fast Lehmer encoding (same as earlier encode_fast)
    static std::uint16_t encode_fast(const std::uint8_t p[8]) {
        std::uint8_t seen = 0;
        std::uint16_t code = 0;
        for (int i = 0; i < 7; ++i) {
            const int rank = p[i] - __builtin_popcount(seen & ((1u << p[i]) - 1u));
            seen |= (1u << p[i]);
            code += static_cast<std::uint16_t>(rank * kFact[i]);
        }
        return code;
    }

    // Fast Lehmer decoding
    static void decode_fast(std::uint16_t code, std::uint8_t p[8]) {
        std::uint8_t avail = 0xFF;
        for (int i = 0; i < 8; ++i) {
            const int idx = code / fact[7 - i];
            code %= fact[7 - i];
            std::uint8_t tmp = avail;
            for (int k = 0; k < idx; ++k) tmp &= tmp - 1;
            p[i] = static_cast<std::uint8_t>(__builtin_ctz(tmp));
            avail &= ~(1u << p[i]);
        }
    }

    alignas(4096) std::uint16_t next_state[40320][8];
    alignas(4096) std::uint8_t victim_way[40320];
    bool initialized = false;

    void init() {
        if (initialized) return;
#ifdef __linux__
        // Hint the kernel to use 2MB transparent huge pages if available
        madvise(next_state, sizeof(next_state), MADV_HUGEPAGE);
        madvise(victim_way, sizeof(victim_way), MADV_HUGEPAGE);
        // Lock the tables so the kernel never swaps them
        mlock(next_state, sizeof(next_state));
        mlock(victim_way, sizeof(victim_way));
#endif
        for (std::uint16_t state = 0; state < 40320; ++state) {
            std::uint8_t perm[8];
            decode_fast(state, perm);
            victim_way[state] = perm[7];

            // Build inverse mapping for O(1) position lookup
            std::uint8_t inv[8];
            for (int i = 0; i < 8; ++i) inv[perm[i]] = i;

            for (int w = 0; w < 8; ++w) {
                const int pos = inv[w];
                // Actually the permutation encodes MRU...LRU order: index 0 is MRU, index 7 is LRU.
                // On access to way w, move w to front (MRU) and shift the prefix.
                // So we need to place w at position 0, and shift 0..pos-1 to 1..pos.
                // The code below does that correctly.
                std::uint8_t tmp[8];
                tmp[0] = w;
                for (int i = 0; i < pos; ++i) tmp[i + 1] = perm[i];
                for (int i = pos + 1; i < 8; ++i) tmp[i] = perm[i];
                next_state[state][w] = encode_fast(tmp);
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

    struct Meta {
        std::uint8_t valid;   // bitmask: 1 = valid
        std::uint8_t dirty;   // bitmask: 1 = dirty
        std::uint16_t lru;    // permutation ID 0..40319
    };

    alignas(64) std::array<std::uint64_t, SETS * WAYS> tag{};
    std::array<Meta, SETS> meta{};

    void init() {
        tag.fill(0);
        for (int i = 0; i < SETS; ++i) {
            meta[i].valid = 0;
            meta[i].dirty = 0;
            meta[i].lru = 0;
        }
    }

    static constexpr int set_of(std::uint64_t blk) {
        return static_cast<int>(blk & INDEX_MASK);
    }
    static constexpr std::uint64_t tag_of(std::uint64_t blk) {
        return blk >> INDEX_BITS;
    }

    [[gnu::always_inline]] int find_way(int si, std::uint64_t t) const {
        std::size_t base = static_cast<std::size_t>(si) * WAYS;
        
        // Force register-to-register broadcast to avoid store-to-load forwarding stall
#if defined(__AVX2__)
        __m256i key = _mm256_broadcastq_epi64(_mm_cvtsi64_si128(t));
        
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
        unsigned m = 0;
        for (int w = 0; w < WAYS; ++w) {
            m |= static_cast<unsigned>(tag[base + w] == t) << w;
        }
#endif
        m &= meta[si].valid;
        return m ? __builtin_ctz(m) : -1;
    }

    void touch_mru(int si, int way) {
        meta[si].lru = TableLRU::next_state[meta[si].lru][way];
    }

    int victim_way(int si) const {
        unsigned invalid = static_cast<unsigned>(~meta[si].valid) & 0xFF;
        if (__builtin_expect(invalid, 0)) {
            return __builtin_ctz(invalid);
        }
        return TableLRU::victim_way[meta[si].lru];
    }

    [[gnu::always_inline]] void set_line(int si, int way, bool v, bool d, std::uint64_t t) {
        tag[static_cast<std::size_t>(si) * WAYS + way] = t;
        
        // Branchless bitwise valid update (v is always true in practice)
        meta[si].valid |= static_cast<std::uint8_t>(1 << way);
        
        // Branchless dirty bit manipulation using two's complement negation (-d)
        // If d=1, -d = 0xFFFFFFFF, (-d & mask) sets the bit
        // If d=0, -d = 0x00000000, (-d & mask) clears the bit
        std::uint8_t mask = 1 << way;
        meta[si].dirty = (meta[si].dirty & ~mask) | (-static_cast<int>(d) & mask);
        
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

    csot::CacheStats run(const csot::MemAccess* __restrict acc, std::size_t n) override {
#ifdef CSOT_CHECK_ALLOCS
        g_hot_path_active = true;
#ifdef __linux__
        prctl(PR_TASK_PERF_EVENTS_ENABLE);
#endif
#endif
        std::uint64_t c_writes = 0;
        std::uint64_t c_l1_hits = 0;
        std::uint64_t c_l2_hits = 0;
        std::uint64_t c_dirty_writebacks = 0;

        for (std::size_t i = 0; i < n; ++i) {
            // Optional prefetch – test with your trace; often redundant on modern CPUs
            // __builtin_prefetch(&acc[i + 16], 0, 0);

            const csot::MemAccess& a = acc[i];
            const std::uint32_t wr = a.is_write; // strictly 0 or 1
            c_writes += wr;

            const std::uint64_t b = a.address >> 6;
            const int s1 = L1::set_of(b);
            const std::uint64_t t1 = L1::tag_of(b);
            const int s2 = L2::set_of(b);
            const std::uint64_t t2 = L2::tag_of(b);

            // L1 lookup
            int w1 = l1_.find_way(s1, t1);
            bool l1_hit = (w1 >= 0);
            c_l1_hits += l1_hit;

            if (__builtin_expect(l1_hit, 1)) {
                l1_.touch_mru(s1, w1);
                // Branchless dirty update: if wr is 0, (0 << w1) is 0, dirty is unchanged.
                // If wr is 1, (1 << w1) sets the dirty bit.
                l1_.meta[s1].dirty |= static_cast<std::uint8_t>(wr << w1);
                continue;
            }

            // L2 lookup
            int w2 = l2_.find_way(s2, t2);
            bool l2_hit = (w2 >= 0);
            c_l2_hits += l2_hit;

            if (l2_hit) {
                l2_.touch_mru(s2, w2);
            } else {
                int victim = l2_.victim_way(s2);
                c_dirty_writebacks += ((l2_.meta[s2].valid & l2_.meta[s2].dirty) >> victim) & 1;
                l2_.set_line(s2, victim, true, false, t2);
            }

            // Fill L1, possibly write back dirty L1 victim
            int v1 = l1_.victim_way(s1);
            if (((l1_.meta[s1].valid & l1_.meta[s1].dirty) >> v1) & 1) {
                std::uint64_t victim_tag = l1_.tag[static_cast<std::size_t>(s1) * L1::NUM_WAYS + v1];
                std::uint64_t bv = (victim_tag << L1::INDEX_BITS) | static_cast<std::uint64_t>(s1);
                int s2v = L2::set_of(bv);
                std::uint64_t t2v = L2::tag_of(bv);

                int wv = l2_.find_way(s2v, t2v);
                if (wv >= 0) {
                    l2_.meta[s2v].dirty |= (1 << wv);
                } else {
                    int vv = l2_.victim_way(s2v);
                    c_dirty_writebacks += ((l2_.meta[s2v].valid & l2_.meta[s2v].dirty) >> vv) & 1;
                    l2_.set_line(s2v, vv, true, true, t2v);
                }
            }

            l1_.set_line(s1, v1, true, wr, t1);
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
        st.l1_hits = c_l1_hits;
        st.l1_misses = n - c_l1_hits;
        st.l2_hits = c_l2_hits;
        st.l2_misses = st.l1_misses - c_l2_hits;
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