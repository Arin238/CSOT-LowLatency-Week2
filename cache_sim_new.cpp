#include "cache_sim.h"
#include <immintrin.h>
#include <cstdint>
#include <cstring>

namespace {

// =======================================================================
// Hybrid Bit-Matrix LRU Configuration
//
// The "Binary/Lookup Dimension": Hardcoded mask arrays for State Updates.
// Fits inside L1 cache / registers.
// =======================================================================

alignas(64) const std::uint64_t ROW_ONES[8] = {
    0x00000000000000FFULL,
    0x000000000000FF00ULL,
    0x0000000000FF0000ULL,
    0x00000000FF000000ULL,
    0x000000FF00000000ULL,
    0x0000FF0000000000ULL,
    0x00FF000000000000ULL,
    0xFF00000000000000ULL
};

alignas(64) const std::uint64_t COL_ZEROS[8] = {
    ~0x0101010101010101ULL,
    ~0x0202020202020202ULL,
    ~0x0404040404040404ULL,
    ~0x0808080808080808ULL,
    ~0x1010101010101010ULL,
    ~0x2020202020202020ULL,
    ~0x4040404040404040ULL,
    ~0x8080808080808080ULL
};

// Initial perfect MRU ordering: 0 is MRU, 7 is LRU
const std::uint64_t LRU_INIT_MATRIX = 0x0080C0E0F0F8FCFEULL;

template <int SETS, int WAYS>
struct Level {
    static constexpr int NUM_SETS = SETS;
    static constexpr int NUM_WAYS = WAYS;
    static constexpr int INDEX_MASK = SETS - 1;

    struct Meta {
        std::uint8_t valid; 
        std::uint8_t dirty;
        std::uint64_t lru_matrix;
    };

    alignas(64) std::uint64_t tag[SETS * WAYS];
    alignas(64) Meta meta[SETS];

    void init() {
        // Initialize all tags to an invalid physical address (~0ULL)
        // This eliminates checking `valid` on the critical hit path!
        std::memset(tag, 0xFF, sizeof(tag));
        for (int i = 0; i < SETS; ++i) {
            meta[i].valid = 0;
            meta[i].dirty = 0;
            meta[i].lru_matrix = LRU_INIT_MATRIX;
        }
    }

    static constexpr int set_of(std::uint64_t blk) {
        return static_cast<int>(blk & INDEX_MASK);
    }

    // Lookup compares the full block address (b) instead of just the tag
    [[gnu::always_inline]] int find_way(int si, std::uint64_t b) const {
        std::size_t base = static_cast<std::size_t>(si) * WAYS;
        
#if defined(__AVX2__)
        __m256i key;
        __asm__ ("vmovq %1, %%xmm0\n\t"
                 "vpbroadcastq %%xmm0, %0"
                 : "=x"(key) : "r"(b) : "xmm0");
        
        __m256i a = _mm256_load_si256(reinterpret_cast<const __m256i*>(&tag[base]));
        __m256i c = _mm256_load_si256(reinterpret_cast<const __m256i*>(&tag[base + 4]));
        
        unsigned m = (unsigned)_mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(a, key)))
                   | ((unsigned)_mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(c, key))) << 4);
        
        if (m) {
            return __builtin_ctz(m);
        }
        return -1;
#else
        for (int i = 0; i < WAYS; ++i) {
            if (tag[base + i] == b) return i;
        }
        return -1;
#endif
    }

    [[gnu::always_inline]] void touch_mru(int si, int way) {
        // The "Runtime Dimension": Branchless bitwise matrix update!
        // 2 Instructions: OR and AND. Zero memory pressure.
        meta[si].lru_matrix = (meta[si].lru_matrix | ROW_ONES[way]) & COL_ZEROS[way];
    }

    [[gnu::always_inline]] int victim_way(int si) const {
        std::uint64_t m = meta[si].lru_matrix;
        // SWAR technique to locate the single "zero byte" inside the matrix.
        // A row with all zeros is mathematically guaranteed to be the exact true-LRU.
        std::uint64_t has_zero = (m - 0x0101010101010101ULL) & ~m & 0x8080808080808080ULL;
        return __builtin_ctzll(has_zero) >> 3; // Shift by 3 is divide by 8
    }

    [[gnu::always_inline]] void set_line(int si, int way, bool valid, bool dirty, std::uint64_t b) {
        tag[si * WAYS + way] = b;
        
        if (valid) {
            meta[si].valid |= (1 << way);
        } else {
            meta[si].valid &= ~(1 << way);
        }
        
        if (dirty) {
            meta[si].dirty |= (1 << way);
        } else {
            meta[si].dirty &= ~(1 << way);
        }
        
        touch_mru(si, way);
    }
};

using L1 = Level<256, 8>;
using L2 = Level<8192, 8>;

class HybridCacheSim : public csot::CacheSim {
public:
    HybridCacheSim() {
        l1_.init();
        l2_.init();
    }

    void on_init() override {
        // Nothing special needed! Bit-Matrix LRU is totally compact!
    }

    csot::CacheStats run(const csot::MemAccess* __restrict acc, std::size_t n) override {
        std::uint64_t c_writes = 0;
        std::uint64_t c_l1_misses = 0;
        std::uint64_t c_l2_hits = 0;
        std::uint64_t c_dirty_writebacks = 0;

        const csot::MemAccess* const end = acc + n;
        for (; acc < end; ++acc) {
            // Software prefetching
            __builtin_prefetch(acc + 8, 0, 0);

            const std::uint32_t wr = acc->is_write;
            c_writes += wr;

            // Compute b: The full block address. Eliminates "tag shifting"!
            const std::uint64_t b = acc->address >> 6;
            const int s1 = L1::set_of(b);

            int w1 = l1_.find_way(s1, b);
            bool l1_hit = (w1 >= 0);

            if (__builtin_expect(l1_hit, 1)) {
                l1_.touch_mru(s1, w1);
                l1_.meta[s1].dirty |= static_cast<std::uint8_t>(wr << w1);
                continue;
            }

            ++c_l1_misses;
            // Compiler barrier to prevent transforming misses into a hit counter
            __asm__ volatile("" : "+r"(c_l1_misses));

            const int s2 = L2::set_of(b);
            int w2 = l2_.find_way(s2, b);
            bool l2_hit = (w2 >= 0);
            c_l2_hits += l2_hit;

            if (l2_hit) {
                l2_.touch_mru(s2, w2);
            } else {
                int victim = l2_.victim_way(s2);
                c_dirty_writebacks += ((l2_.meta[s2].valid & l2_.meta[s2].dirty) >> victim) & 1;
                l2_.set_line(s2, victim, true, false, b);
            }

            // Writeback dirty L1 victim
            int v1 = l1_.victim_way(s1);
            if (((l1_.meta[s1].valid & l1_.meta[s1].dirty) >> v1) & 1) {
                // b is stored directly! No tag shifting/reconstruction required!
                std::uint64_t bv = l1_.tag[static_cast<std::size_t>(s1) * L1::NUM_WAYS + v1];
                int s2v = L2::set_of(bv);

                int wv = l2_.find_way(s2v, bv);
                if (wv >= 0) {
                    l2_.meta[s2v].dirty |= (1 << wv);
                    // FIXED: A victim written to L2 counts as an L2 access!
                    l2_.touch_mru(s2v, wv);
                } else {
                    int vv = l2_.victim_way(s2v);
                    c_dirty_writebacks += ((l2_.meta[s2v].valid & l2_.meta[s2v].dirty) >> vv) & 1;
                    l2_.set_line(s2v, vv, true, true, bv);
                }
            }

            l1_.set_line(s1, v1, true, wr, b);
        }

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
    return new HybridCacheSim();
}
