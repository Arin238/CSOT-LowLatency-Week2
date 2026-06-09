// ============================================================================
//  cache_sim.cpp
// ============================================================================

#include "cache_sim.hpp"

#include <cstddef>
#include <cstdint>
#include <array>
#include <cstring>

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

namespace {

constexpr int log2_of(int v) { int r = 0; while ((1 << r) < v) ++r; return r; }

// ============================================================================
//  Level<SETS, WAYS>
//
//  Memory layout (SoA, all arrays cache-line-aligned):
//
//    tag[]   — uint64_t, one full tag per slot      → 8 bytes/entry
//    valid[] — uint8_t,  0 or 1                     → 1 byte/entry
//    dirty[] — uint8_t,  0 or 1                     → 1 byte/entry
//    lru[]   — uint8_t,  way index [0, WAYS)         → 1 byte/entry
//
//  For L1 (64×8 = 512 entries):
//    tag: 4 KiB   valid/dirty/lru: 512 B each   → ~5.5 KiB total
//  For L2 (512×8 = 4096 entries):
//    tag: 32 KiB  valid/dirty/lru: 4 KiB each   → ~44 KiB total
//  Grand total: ~50 KiB — fits in CPU L2/L3; previously 144 KiB with uint64_t
//  everywhere, which blew out the CPU's L1D on every access.
// ============================================================================
template <int SETS, int WAYS>
struct Level {
    static constexpr int           NUM_SETS   = SETS;
    static constexpr int           NUM_WAYS   = WAYS;
    static constexpr int           INDEX_BITS = log2_of(SETS);
    static constexpr std::uint64_t INDEX_MASK = SETS - 1;

    struct Way {
        std::uint64_t tag   : 59;
        std::uint64_t lru   : 3;
        std::uint64_t valid : 1;
        std::uint64_t dirty : 1;
    };

    struct alignas(64) CacheSet {
        std::array<Way, WAYS> ways;
    };

    std::array<CacheSet, SETS> sets{};

    void init() {
        for (int si = 0; si < SETS; ++si) {
            for (int w = 0; w < WAYS; ++w) {
                sets[si].ways[w].tag   = 0;
                sets[si].ways[w].valid = 0;
                sets[si].ways[w].dirty = 0;
                sets[si].ways[w].lru   = static_cast<std::uint64_t>(w);
            }
        }
    }

    static constexpr int           set_of(std::uint64_t blk) { return static_cast<int>(blk & INDEX_MASK); }
    static constexpr std::uint64_t tag_of(std::uint64_t blk) { return blk >> INDEX_BITS; }

    int find_way(int si, std::uint64_t t) const {
        int match = -1;
        for (int w = 0; w < WAYS; ++w) {
            bool hit = sets[si].ways[w].valid & (sets[si].ways[w].tag == t);
            match = hit ? w : match;
        }
        return match;
    }

    void touch_mru(int si, int way) {
        std::uint64_t target_lru = sets[si].ways[way].lru;
        for (int w = 0; w < WAYS; ++w) {
            sets[si].ways[w].lru += (sets[si].ways[w].lru < target_lru);
        }
        sets[si].ways[way].lru = 0;
    }

    int victim_way(int si) const {
        int match = 0;
        int max_score = -1;
        for (int w = 0; w < WAYS; ++w) {
            int score = (!sets[si].ways[w].valid) * 10 + sets[si].ways[w].lru;
            bool is_better = score > max_score;
            match = is_better ? w : match;
            max_score = is_better ? score : max_score;
        }
        return match;
    }

    void set_line(int si, int way, bool v, bool d, std::uint64_t t) {
        sets[si].ways[way].valid = v;
        sets[si].ways[way].dirty = d;
        sets[si].ways[way].tag   = t;
        touch_mru(si, way);
    }
};

using L1 = Level<64,  8>;   // 32 KiB,  8-way, 64  sets, 64-byte lines
using L2 = Level<512, 8>;   // 256 KiB, 8-way, 512 sets, 64-byte lines

// ============================================================================
class BaselineCacheSim final : public csot::CacheSim {
public:
    void on_init() override { l1_.init(); l2_.init(); }

    csot::CacheStats run(const csot::MemAccess* acc, std::size_t n) override {
#ifdef CSOT_CHECK_ALLOCS
        g_hot_path_active = true;
#ifdef __linux__
        prctl(PR_TASK_PERF_EVENTS_ENABLE);
#endif
#endif
        csot::CacheStats st{};

        for (std::size_t i = 0; i < n; ++i) {
            // Prefetch ~16 elements ahead. 
            // 0 = Read intention, 0 = No temporal locality (don't pollute L1/L2)
            __builtin_prefetch(&acc[i + 16], 0, 0);

            const csot::MemAccess& a  = acc[i];
            const bool             wr = (a.is_write != 0);

            st.writes += wr;
            st.reads  += !wr;

            const std::uint64_t b  = a.address >> 6;   // 64-byte block address
            const int           s1 = L1::set_of(b);
            const std::uint64_t t1 = L1::tag_of(b);
            const int           s2 = L2::set_of(b);
            const std::uint64_t t2 = L2::tag_of(b);

            // ----------------------------------------------------------------
            // §5.2  L1 probe
            // ----------------------------------------------------------------
            const int w1 = l1_.find_way(s1, t1);
            bool l1_hit = (w1 >= 0);
            st.l1_hits += l1_hit;
            st.l1_misses += !l1_hit;

            if (l1_hit) {
                l1_.touch_mru(s1, w1);
                l1_.sets[s1].ways[w1].dirty |= wr;
                continue;
            }

            // ----------------------------------------------------------------
            // §5.3  L2 probe (only on L1 miss)
            // ----------------------------------------------------------------
            const int w2 = l2_.find_way(s2, t2);
            bool l2_hit = (w2 >= 0);
            st.l2_hits += l2_hit;
            st.l2_misses += !l2_hit;

            const int vl2 = l2_hit ? w2 : l2_.victim_way(s2);
            st.dirty_writebacks += (!l2_hit) & l2_.sets[s2].ways[vl2].valid & l2_.sets[s2].ways[vl2].dirty;
            
            l2_.sets[s2].ways[vl2].valid = 1;
            l2_.sets[s2].ways[vl2].dirty &= l2_hit;
            l2_.sets[s2].ways[vl2].tag = t2;
            l2_.touch_mru(s2, vl2);

            // ----------------------------------------------------------------
            // §5.4  Fill into L1 (write-allocate).
            // If the L1 victim is dirty, write it back to L2 first (§5.5).
            // ----------------------------------------------------------------
            const int vl1 = l1_.victim_way(s1);
            bool writeback = l1_.sets[s1].ways[vl1].valid & l1_.sets[s1].ways[vl1].dirty;

            if (writeback) {
                // Reconstruct block address of dirty L1 victim
                const std::uint64_t vtag = l1_.sets[s1].ways[vl1].tag;
                const std::uint64_t bv   = (vtag << L1::INDEX_BITS) | static_cast<std::uint64_t>(s1);
                const int           s2v  = L2::set_of(bv);
                const std::uint64_t t2v  = L2::tag_of(bv);

                const int wv = l2_.find_way(s2v, t2v);
                bool wv_hit = (wv >= 0);
                
                int safe_wv = wv_hit ? wv : 0;
                l2_.sets[s2v].ways[safe_wv].dirty |= wv_hit;

                if (!wv_hit) {
                    const int vv = l2_.victim_way(s2v);
                    st.dirty_writebacks += l2_.sets[s2v].ways[vv].valid & l2_.sets[s2v].ways[vv].dirty;
                    l2_.set_line(s2v, vv, true, true, t2v);
                }
            }

            l1_.set_line(s1, vl1, true, wr, t1);
        }

#ifdef CSOT_CHECK_ALLOCS
#ifdef __linux__
        prctl(PR_TASK_PERF_EVENTS_DISABLE);
#endif
        g_hot_path_active = false;
#endif
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