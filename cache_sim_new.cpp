#include "cache_sim.hpp"

#include <array>
#include <cstdint>
#include <cstddef>

#if defined(__SSE4_1__)
#include <smmintrin.h> // For SSE4.1 intrinsics
#endif

namespace csot {

template <std::size_t Sets, std::size_t Ways>
struct CacheLevel {
    // A single set formatted as Struct-of-Arrays (SoA) and aligned to a 64-byte 
    // cache line boundary for optimal hardware prefetching and zero false-sharing.
    struct alignas(64) Set {
        std::uint64_t tags[Ways];
        std::uint8_t valid_mask = 0;
        std::uint8_t dirty_mask = 0;
        std::uint8_t lru[Ways]; // Simple LRU array: lru[0] is MRU way, lru[Ways-1] is LRU way
    };

    std::array<Set, Sets> sets;

    // Zero-initialization: populate the LRU array initially.
    CacheLevel() {
        for (std::size_t s = 0; s < Sets; ++s) {
            for (std::size_t w = 0; w < Ways; ++w) {
                sets[s].lru[w] = static_cast<std::uint8_t>(w);
                sets[s].tags[w] = 0;
            }
        }
    }

    // Branchless SIMD tag scan (SSE4.1)
    // Returns way index on hit, -1 on miss.
    int find(std::size_t set_idx, std::uint64_t target_block_addr) const {
        const auto& set = sets[set_idx];
#if defined(__SSE4_1__)
        __m128i key = _mm_set1_epi64x(target_block_addr);
        __m128i a = _mm_load_si128(reinterpret_cast<const __m128i*>(&set.tags[0]));
        __m128i b = _mm_load_si128(reinterpret_cast<const __m128i*>(&set.tags[2]));
        __m128i c = _mm_load_si128(reinterpret_cast<const __m128i*>(&set.tags[4]));
        __m128i d = _mm_load_si128(reinterpret_cast<const __m128i*>(&set.tags[6]));

        unsigned mask = (unsigned)_mm_movemask_pd(_mm_castsi128_pd(_mm_cmpeq_epi64(a, key)))
                      | ((unsigned)_mm_movemask_pd(_mm_castsi128_pd(_mm_cmpeq_epi64(b, key))) << 2)
                      | ((unsigned)_mm_movemask_pd(_mm_castsi128_pd(_mm_cmpeq_epi64(c, key))) << 4)
                      | ((unsigned)_mm_movemask_pd(_mm_castsi128_pd(_mm_cmpeq_epi64(d, key))) << 6);
        
        mask &= set.valid_mask;
        if (mask) {
            return __builtin_ctz(mask);
        }
#else
        for (std::size_t i = 0; i < Ways; ++i) {
            if ((set.valid_mask & (1 << i)) && set.tags[i] == target_block_addr) {
                return static_cast<int>(i);
            }
        }
#endif
        return -1;
    }

    // Move `way` to MRU position (lru[0]). 
    // Shifts elements down branchlessly for the touched subset.
    void touch(std::size_t set_idx, int way) {
        auto& set = sets[set_idx];
        // Find where this way is currently positioned in the LRU array
        int pos = 0;
        for (int i = 0; i < Ways; ++i) {
            if (set.lru[i] == way) {
                pos = i;
                break;
            }
        }
        
        if (pos == 0) return; // Already MRU
        
        // Shift everything before `pos` down by 1
        for (int i = pos; i > 0; --i) {
            set.lru[i] = set.lru[i - 1];
        }
        set.lru[0] = static_cast<std::uint8_t>(way);
    }

    // Pick an invalid way if any, otherwise return LRU way.
    int victim_way(std::size_t set_idx) const {
        const auto& set = sets[set_idx];
        unsigned invalid_mask = ~set.valid_mask & ((1 << Ways) - 1);
        if (invalid_mask) {
            return __builtin_ctz(invalid_mask);
        }
        return set.lru[Ways - 1];
    }
};

class BaselineCacheSim : public CacheSim {
private:
    CacheLevel<64, 8> l1_;
    CacheLevel<512, 8> l2_;

public:
    void on_init() override {}

    CacheStats run(const MemAccess* accesses, std::size_t n) override {
        CacheStats st{};

        for (std::size_t i = 0; i < n; ++i) {
            const std::uint64_t b = accesses[i].address >> 6;
            const bool is_write = accesses[i].is_write != 0;
            
            if (is_write) st.writes++;
            else st.reads++;

            // Prefetch next access for software pipelining
            if (i + 1 < n) {
                const std::size_t next_s1 = (accesses[i + 1].address >> 6) & 63;
                __builtin_prefetch(&l1_.sets[next_s1]);
            }

            const std::size_t s1 = b & 63;

            // Probe L1 using full block address
            int w1 = l1_.find(s1, b);
            if (w1 >= 0) [[likely]] {
                st.l1_hits++;
                l1_.touch(s1, w1);
                if (is_write) l1_.sets[s1].dirty_mask |= (1 << w1);
                continue;
            }
            
            st.l1_misses++;
            
            // L2 Geometry parameters
            const std::size_t s2 = b & 511;

            int w2 = l2_.find(s2, b);
            if (w2 >= 0) {
                st.l2_hits++;
                l2_.touch(s2, w2);
            } else {
                st.l2_misses++;
                int v = l2_.victim_way(s2);
                
                if ((l2_.sets[s2].valid_mask & (1 << v)) && (l2_.sets[s2].dirty_mask & (1 << v))) {
                    st.dirty_writebacks++;
                }
                
                l2_.sets[s2].tags[v] = b;
                l2_.sets[s2].valid_mask |= (1 << v);
                l2_.sets[s2].dirty_mask &= ~(1 << v);
                l2_.touch(s2, v);
            }

            // Fill L1
            int v1 = l1_.victim_way(s1);
            
            if ((l1_.sets[s1].valid_mask & (1 << v1)) && (l1_.sets[s1].dirty_mask & (1 << v1))) {
                // Since we store the full block address, no bit-shifting reconstruction needed!
                const std::uint64_t bv = l1_.sets[s1].tags[v1];
                const std::size_t s2v = bv & 511;
                
                int wv = l2_.find(s2v, bv);
                if (wv >= 0) {
                    l2_.sets[s2v].dirty_mask |= (1 << wv);
                } else {
                    int vv = l2_.victim_way(s2v);
                    if ((l2_.sets[s2v].valid_mask & (1 << vv)) && (l2_.sets[s2v].dirty_mask & (1 << vv))) {
                        st.dirty_writebacks++;
                    }
                    
                    l2_.sets[s2v].tags[vv] = bv;
                    l2_.sets[s2v].valid_mask |= (1 << vv);
                    l2_.sets[s2v].dirty_mask |= (1 << vv);
                    l2_.touch(s2v, vv);
                }
            }
            
            // Overwrite L1 victim
            l1_.sets[s1].tags[v1] = b;
            l1_.sets[s1].valid_mask |= (1 << v1);
            if (is_write) {
                l1_.sets[s1].dirty_mask |= (1 << v1);
            } else {
                l1_.sets[s1].dirty_mask &= ~(1 << v1);
            }
            l1_.touch(s1, v1);
        }

        return st;
    }
};

} // namespace csot

extern "C" csot::CacheSim* create_cache_sim() {
    return new csot::BaselineCacheSim();
}
