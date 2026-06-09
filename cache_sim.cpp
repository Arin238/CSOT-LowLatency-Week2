// ============================================================================
//  cache_sim.cpp — final branch‑free LRU, SIMD tag scan, zero alloc
// ============================================================================

// UNCOMMENT THIS LINE to benchmark the 685KB True LRU lookup tables:
// #define USE_TABLE_LRU

#include "cache_sim.hpp"

#include <cstddef>
#include <cstdint>
#include <array>
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

void* operator new(std::size_t size) {
    if (g_hot_path_active) {
        std::cerr << "\n[ALLOCATION DETECTED ON HOT PATH] Size: " << size << " bytes\n";
        std::abort();
    }
    void* p = std::malloc(size);
    if (!p) throw std::bad_alloc();
    return p;
}

void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
#endif

#include <algorithm>

#ifdef USE_TABLE_LRU
namespace TableLRU {
    std::uint16_t next_state[40320][8];
    std::uint8_t victim_way[40320];
    bool initialized = false;

    int encode(int p[8]) {
        int code = 0;
        int fact[8] = {1, 1, 2, 6, 24, 120, 720, 5040};
        for (int i = 0; i < 8; ++i) {
            int less = 0;
            for (int j = i + 1; j < 8; ++j) {
                if (p[j] < p[i]) less++;
            }
            code += less * fact[7 - i];
        }
        return code;
    }

    void init() {
        if (initialized) return;
        initialized = true;
        int p[8] = {0, 1, 2, 3, 4, 5, 6, 7};
        do {
            int id = encode(p);
            for (int w = 0; w < 8; ++w) {
                if (p[w] == 7) victim_way[id] = w;
            }
            for (int w = 0; w < 8; ++w) {
                int target = p[w];
                int next_p[8];
                for (int i = 0; i < 8; ++i) {
                    if (p[i] < target) next_p[i] = p[i] + 1;
                    else if (p[i] > target) next_p[i] = p[i];
                    else next_p[i] = 0;
                }
                next_state[id][w] = encode(next_p);
            }
        } while (std::next_permutation(p, p + 8));
    }
}
#endif

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
#ifdef USE_TABLE_LRU
        std::uint16_t lru;    // permutation ID 0..40319
#else
        std::uint32_t lru;    // 4 bits per way, 0 = most recent, 7 = least recent
#endif
    };

    alignas(64) std::array<std::uint64_t, SETS * WAYS> tag{};
    std::array<Meta, SETS> meta{};

    void init() {
        tag.fill(0);
        for (int i = 0; i < SETS; ++i) {
            meta[i].valid = 0;
            meta[i].dirty = 0;
#ifdef USE_TABLE_LRU
            meta[i].lru = 0;
#else
            // Initial LRU: each way has its index as age (0 = most recent, 7 = least recent)
            std::uint32_t lru = 0;
            for (int w = 0; w < WAYS; ++w)
                lru |= static_cast<std::uint32_t>(w) << (w * 4);
            meta[i].lru = lru;
#endif
        }
    }

    static constexpr int set_of(std::uint64_t blk) {
        return static_cast<int>(blk & INDEX_MASK);
    }
    static constexpr std::uint64_t tag_of(std::uint64_t blk) {
        return blk >> INDEX_BITS;
    }

    int find_way(int si, std::uint64_t t) const {
        const std::size_t base = static_cast<std::size_t>(si) * WAYS;
#if defined(__AVX2__)
        __m256i key = _mm256_set1_epi64x(t);
        __m256i a = _mm256_load_si256(reinterpret_cast<const __m256i*>(&tag[base]));
        __m256i b = _mm256_load_si256(reinterpret_cast<const __m256i*>(&tag[base + 4]));
        unsigned m = (unsigned)_mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(a, key)))
                   | ((unsigned)_mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(b, key))) << 4);
        _mm256_zeroupper();   // Prevent AVX‑SSE transition penalty
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

    // Branch‑free, table‑free LRU update using SWAR bitwise ops
    void touch_mru(int si, int way) {
#ifdef USE_TABLE_LRU
        meta[si].lru = TableLRU::next_state[meta[si].lru][way];
#else
        std::uint32_t lru_val = meta[si].lru;
        std::uint32_t target_age = (lru_val >> (way * 4)) & 0xF;
        std::uint32_t targets = target_age * 0x11111111;
        
        // Find fields where age < target_age
        // (8 + age - target) has MSB=1 if age >= target, MSB=0 if age < target
        std::uint32_t geq_mask = ((lru_val | 0x88888888) - targets) & 0x88888888;
        std::uint32_t less_mask = (~geq_mask) & 0x88888888;
        
        // ONLY increment ages that were younger than the touched way!
        lru_val += (less_mask >> 3);
        lru_val -= (target_age << (way * 4)); // set touched way to exactly 0
        meta[si].lru = lru_val;
#endif
    }

    int victim_way(int si) const {
        unsigned invalid = static_cast<unsigned>(~meta[si].valid) & 0xFF;
        if (__builtin_expect(invalid, 0)) {
            return __builtin_ctz(invalid);
        }
#ifdef USE_TABLE_LRU
        return TableLRU::victim_way[meta[si].lru];
#else
        // Find the way with age 7
        std::uint32_t lru = meta[si].lru;
        unsigned victim_mask = (lru + 0x11111111) & 0x88888888;
        return __builtin_ctz(victim_mask) >> 2;
#endif
    }

    void set_line(int si, int way, bool v, bool d, std::uint64_t t) {
        const std::size_t base = static_cast<std::size_t>(si) * WAYS;
        if (v) meta[si].valid |= (1 << way); else meta[si].valid &= ~(1 << way);
        if (d) meta[si].dirty |= (1 << way); else meta[si].dirty &= ~(1 << way);
        tag[base + way] = t;
        touch_mru(si, way);
    }
};

using L1 = Level<64, 8>;
using L2 = Level<512, 8>;

// ============================================================================
class BaselineCacheSim final : public csot::CacheSim {
public:
    void on_init() override {
#ifdef USE_TABLE_LRU
        TableLRU::init();
#endif
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