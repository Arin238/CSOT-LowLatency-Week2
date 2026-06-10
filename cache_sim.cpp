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

// TableLRU completely removed
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
        std::uint64_t lru;    // 64-bit packed byte array of way ordering [MRU...LRU]
    };

    alignas(64) std::array<std::uint64_t, SETS * WAYS> tag{};
    std::array<Meta, SETS> meta{};

    void init() {
        tag.fill(0);
        for (int i = 0; i < SETS; ++i) {
            meta[i].valid = 0;
            meta[i].dirty = 0;
            // Initialize with way 0 at MRU, way 7 at LRU
            meta[i].lru = 0x0706050403020100ULL;
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

    [[gnu::always_inline]] void touch_mru(int si, int way) {
        std::uint64_t v = meta[si].lru;
        
        // Find the byte position of 'way' inside the 64-bit integer
        std::uint64_t search = static_cast<std::uint64_t>(way) * 0x0101010101010101ULL;
        std::uint64_t eq = v ^ search;
        // zero_bytes sets the MSB (0x80) for any byte that equals 'way'
        std::uint64_t zero_bytes = (eq - 0x0101010101010101ULL) & ~eq & 0x8080808080808080ULL;
        int pos = __builtin_ctzll(zero_bytes) / 8;
        
        // Construct the new state by bringing 'way' to the front (MRU)
        std::uint64_t lower_mask = (1ULL << (pos * 8)) - 1;
        std::uint64_t upper_mask = (pos == 7) ? 0 : (~0ULL << ((pos + 1) * 8));
        meta[si].lru = static_cast<std::uint64_t>(way) | ((v & lower_mask) << 8) | (v & upper_mask);
    }

    [[gnu::always_inline]] int victim_way(int si) const {
        unsigned invalid = static_cast<unsigned>(~meta[si].valid) & 0xFF;
        if (__builtin_expect(invalid, 0)) {
            return __builtin_ctz(invalid);
        }
        // The LRU way is stored at the very end of the 64-bit integer (bits 56-63)
        return (meta[si].lru >> 56) & 0xFF;
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