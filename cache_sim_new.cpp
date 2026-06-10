#include "cache_sim.hpp"
#include <immintrin.h>
#include <cstdint>
#include <cstring>

#include <utility>

namespace {

// =======================================================================
// Hybrid Bit-Matrix LRU Configuration
//
// The "Binary/Lookup Dimension" is now fully computed in the ALU!
// No arrays needed, meaning ZERO L1 Data Cache pressure for masks!
// =======================================================================

// Initial perfect MRU ordering: 0 is MRU, 7 is LRU
const std::uint64_t LRU_INIT_MATRIX = 0x0080C0E0F0F8FCFEULL;

// =======================================================================
// Compile-Time Meta-Programming: Fold Expressions for Scalar Loop Unrolling
// =======================================================================
template <int W>
[[gnu::always_inline]] static inline void check_way(const std::uint64_t* tag, std::uint64_t b, unsigned& m) {
    m |= static_cast<unsigned>(tag[W] == b) << W;
}

template <int... Ws>
[[gnu::always_inline]] static inline unsigned check_all_ways(const std::uint64_t* tag, std::uint64_t b, std::integer_sequence<int, Ws...>) {
    unsigned m = 0;
    (check_way<Ws>(tag, b, m), ...);
    return m;
}

template <int SETS, int WAYS>
struct Level {
    static constexpr int NUM_SETS = SETS;
    static constexpr int NUM_WAYS = WAYS;
    static constexpr int INDEX_MASK = SETS - 1;

    alignas(64) std::uint64_t tag[SETS * WAYS];
    // Pure Struct-of-Arrays (SoA) layout eliminates padding and reduces address math
    alignas(64) std::uint8_t valid[SETS];
    alignas(64) std::uint8_t dirty[SETS];
    alignas(64) std::uint64_t lru_matrix[SETS];

    void init() {
        // Initialize all tags to an invalid physical address (~0ULL)
        // This eliminates checking `valid` on the critical hit path!
        std::memset(tag, 0xFF, sizeof(tag));
        for (int i = 0; i < SETS; ++i) {
            valid[i] = 0;
            dirty[i] = 0;
            lru_matrix[i] = LRU_INIT_MATRIX;
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
        
        return m ? __builtin_ctz(m) : -1;
#elif defined(__SSE4_1__)
        __m128i key = _mm_set1_epi64x(b);
        __m128i a_val = _mm_load_si128(reinterpret_cast<const __m128i*>(&tag[base + 0]));
        __m128i b_val = _mm_load_si128(reinterpret_cast<const __m128i*>(&tag[base + 2]));
        __m128i c_val = _mm_load_si128(reinterpret_cast<const __m128i*>(&tag[base + 4]));
        __m128i d_val = _mm_load_si128(reinterpret_cast<const __m128i*>(&tag[base + 6]));
        unsigned m = (unsigned)_mm_movemask_pd(_mm_castsi128_pd(_mm_cmpeq_epi64(a_val, key)))
                   | ((unsigned)_mm_movemask_pd(_mm_castsi128_pd(_mm_cmpeq_epi64(b_val, key))) << 2)
                   | ((unsigned)_mm_movemask_pd(_mm_castsi128_pd(_mm_cmpeq_epi64(c_val, key))) << 4)
                   | ((unsigned)_mm_movemask_pd(_mm_castsi128_pd(_mm_cmpeq_epi64(d_val, key))) << 6);
        return m ? __builtin_ctz(m) : -1;
#else
        unsigned m = check_all_ways(tag + base, b, std::make_integer_sequence<int, WAYS>{});
        return m ? __builtin_ctz(m) : -1;
#endif
    }

    [[gnu::always_inline]] void touch_mru(int si, int way) {
        // The "Runtime Dimension": Branchless ALU matrix update!
        // We compute masks via ALU to avoid 2 array loads entirely!
        std::uint64_t row = 0xFFULL << (way * 8);
        std::uint64_t col = ~(0x0101010101010101ULL << way);
        lru_matrix[si] = (lru_matrix[si] | row) & col;
    }

    [[gnu::always_inline]] int victim_way(int si) const {
        unsigned invalid = static_cast<unsigned>(~valid[si]) & 0xFF;
        if (__builtin_expect(invalid, 0)) {
            return __builtin_ctz(invalid);
        }
        std::uint64_t m = lru_matrix[si];
        // SWAR technique to locate the single "zero byte" inside the matrix.
        // A row with all zeros is mathematically guaranteed to be the exact true-LRU.
        std::uint64_t has_zero = (m - 0x0101010101010101ULL) & ~m & 0x8080808080808080ULL;
        return __builtin_ctzll(has_zero) >> 3; // Shift by 3 is divide by 8
    }

    [[gnu::always_inline]] void set_line(int si, int way, bool v, bool d, std::uint64_t b) {
        tag[si * WAYS + way] = b;
        
        std::uint8_t mask = 1 << way;
        valid[si] = (valid[si] & ~mask) | (-static_cast<int>(v) & mask);
        dirty[si] = (dirty[si] & ~mask) | (-static_cast<int>(d) & mask);
        
        touch_mru(si, way);
    }
};

using L1 = Level<64, 8>;
using L2 = Level<512, 8>;

class HybridCacheSim : public csot::CacheSim {
public:
    HybridCacheSim() {
    }

    void on_init() override {
        l1_.init();
        l2_.init();
    }

    [[gnu::always_inline]] void process_one(const csot::MemAccess* __restrict acc, 
                                            std::uint64_t& c_writes,
                                            std::uint64_t& c_l1_misses,
                                            std::uint64_t& c_l2_hits,
                                            std::uint64_t& c_dirty_writebacks) {
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
            l1_.dirty[s1] |= static_cast<std::uint8_t>(wr << w1);
            return;
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
            c_dirty_writebacks += ((l2_.valid[s2] & l2_.dirty[s2]) >> victim) & 1;
            l2_.set_line(s2, victim, true, false, b);
        }

        // Writeback dirty L1 victim
        int v1 = l1_.victim_way(s1);
        if (((l1_.valid[s1] & l1_.dirty[s1]) >> v1) & 1) {
            // b is stored directly! No tag shifting/reconstruction required!
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
        // Fold expression expands process_one sequentially for all N elements
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
        std::uint64_t c_writes = 0;
        std::uint64_t c_l1_misses = 0;
        std::uint64_t c_l2_hits = 0;
        std::uint64_t c_dirty_writebacks = 0;

        constexpr std::size_t BATCH_SIZE = 4;
        const csot::MemAccess* const end_batch = acc + (n / BATCH_SIZE) * BATCH_SIZE;
        for (; acc < end_batch; acc += BATCH_SIZE) {
            process_batch<BATCH_SIZE>(acc, c_writes, c_l1_misses, c_l2_hits, c_dirty_writebacks);
        }

        // Tail processing for remaining accesses
        const csot::MemAccess* const end = acc + (n % BATCH_SIZE);
        for (; acc < end; ++acc) {
            process_one(acc, c_writes, c_l1_misses, c_l2_hits, c_dirty_writebacks);
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
