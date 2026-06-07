// ============================================================================
//  cache_sim.stub.cpp — STARTING POINT for the Week-2 cache-sim challenge.
//
//  Copy this to `cache_sim.cpp`, then make it correct, then make it fast:
//      cp samples/cache_sim.stub.cpp cache_sim.cpp
//      cmake -B build -DCSOT_CACHE_SIM_SRC=cache_sim.cpp && cmake --build build -j
//      ./build/cache_sim_runner data/tiny.trace      # compare to data/tiny.stats.json
//
//  This stub COMPILES and RUNS but is INTENTIONALLY NOT A CORRECT SIMULATOR.
//  It counts reads/writes and treats every access as an L1 miss + L2 miss so
//  you can see the harness work end-to-end. Your job is to implement the real
//  two-level hierarchy from CACHE_SPEC.md so the seven counters match the
//  reference exactly — and then to make run() as fast as you can.
//
//  Everything must live in this ONE translation unit. The judge builds exactly
//  this file against its own main(); no extra .cpp, no custom CMake.
// ============================================================================

#include "cache_sim.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>
#include <cstring>

namespace {

// Geometry constants from CACHE_SPEC.md
constexpr int LINE_SIZE = 64;
constexpr int L1_SIZE = 32 * 1024;
constexpr int L1_WAYS = 8;
constexpr int L1_SETS = 64;
constexpr int L1_INDEX_BITS = 6;

constexpr int L2_SIZE = 256 * 1024;
constexpr int L2_WAYS = 8;
constexpr int L2_SETS = 512;
constexpr int L2_INDEX_BITS = 9;

struct Line {
    bool valid;
    bool dirty;
    std::uint64_t tag;
};

struct Set {
    Line ways[L1_WAYS];        // max ways; we'll use appropriate WAYS per cache
    int lru_order[L1_WAYS];    // MRU..LRU
};

// We reuse Set but interpret ways count appropriately for L1/L2.
class BaselineCacheSim final : public csot::CacheSim {
public:
    void on_init() override {
        // allocate and initialize L1 and L2 sets
        l1_sets.resize(L1_SETS);
        l2_sets.resize(L2_SETS);

        // initialize each set: mark lines invalid and set canonical LRU order
        for (int si = 0; si < L1_SETS; ++si) {
            for (int w = 0; w < L1_WAYS; ++w) {
                l1_sets[si].ways[w].valid = false;
                l1_sets[si].ways[w].dirty = false;
                l1_sets[si].ways[w].tag = 0;
                l1_sets[si].lru_order[w] = w;
            }
        }
        for (int si = 0; si < L2_SETS; ++si) {
            for (int w = 0; w < L2_WAYS; ++w) {
                l2_sets[si].ways[w].valid = false;
                l2_sets[si].ways[w].dirty = false;
                l2_sets[si].ways[w].tag = 0;
                l2_sets[si].lru_order[w] = w;
            }
        }
    }

    csot::CacheStats run(const csot::MemAccess* acc, std::size_t n) override {
        csot::CacheStats st{};

        for (std::size_t i = 0; i < n; ++i) {
            const csot::MemAccess &a = acc[i];
            const uint64_t addr = a.address;
            const bool wr = (a.is_write != 0);

            // §5.1 count the access
            if (wr) ++st.writes; else ++st.reads;

            // block address and indices/tags
            const uint64_t b = addr >> 6; // block address

            const int s1 = static_cast<int>(b & (L1_SETS - 1));
            const uint64_t t1 = b >> L1_INDEX_BITS; // b >> 6

            const int s2 = static_cast<int>(b & (L2_SETS - 1));
            const uint64_t t2 = b >> L2_INDEX_BITS; // b >> 9

            // §5.2 probe L1
            int w1 = find_way(l1_sets, L1_WAYS, s1, t1);
            if (w1 >= 0) {
                ++st.l1_hits;
                touch_mru(l1_sets, L1_WAYS, s1, w1);
                if (wr) l1_sets[s1].ways[w1].dirty = true;
                continue; // done with this access
            }
            ++st.l1_misses;

            // §5.3 probe L2 (only on L1 miss)
            int w2 = find_way(l2_sets, L2_WAYS, s2, t2);
            if (w2 >= 0) {
                ++st.l2_hits;
                touch_mru(l2_sets, L2_WAYS, s2, w2);
            } else {
                ++st.l2_misses;
                int victim = victim_way(l2_sets, L2_WAYS, s2);
                if (l2_sets[s2].ways[victim].valid && l2_sets[s2].ways[victim].dirty) {
                    ++st.dirty_writebacks; // L2 -> main memory
                }
                // install clean line
                set_line(l2_sets, L2_WAYS, s2, victim, true, false, t2);
            }

            // §5.4 fill into L1 (write-allocate)
            int v1 = victim_way(l1_sets, L1_WAYS, s1);
            if (l1_sets[s1].ways[v1].valid && l1_sets[s1].ways[v1].dirty) {
                // Writing dirty L1 victim back to L2 (§5.5)
                uint64_t victim_tag = l1_sets[s1].ways[v1].tag;
                uint64_t bv = (victim_tag << L1_INDEX_BITS) | static_cast<uint64_t>(s1);
                int s2v = static_cast<int>(bv & (L2_SETS - 1));
                uint64_t t2v = bv >> L2_INDEX_BITS;

                int wv = find_way(l2_sets, L2_WAYS, s2v, t2v);
                if (wv >= 0) {
                    // set dirty on existing L2 line; do NOT touch LRU or counts
                    l2_sets[s2v].ways[wv].dirty = true;
                } else {
                    // install dirty into L2; may evict L2 victim -> memory
                    int vv = victim_way(l2_sets, L2_WAYS, s2v);
                    if (l2_sets[s2v].ways[vv].valid && l2_sets[s2v].ways[vv].dirty) {
                        ++st.dirty_writebacks;
                    }
                    set_line(l2_sets, L2_WAYS, s2v, vv, true, true, t2v);
                }
            }

            // place the requested line into L1; dirty iff write
            set_line(l1_sets, L1_WAYS, s1, v1, true, wr, t1);
        }

        return st;
    }

private:
    // storage for sets: note we use the same Set type sized for L1_WAYS
    std::vector<Set> l1_sets;
    std::vector<Set> l2_sets;

    // find the way index with matching tag in given sets array; return -1 if not found
    static int find_way(const std::vector<Set> &sets, int ways, int set_idx, uint64_t tag) {
        const Set &s = sets[set_idx];
        for (int w = 0; w < ways; ++w) {
            if (s.ways[w].valid && s.ways[w].tag == tag) return w;
        }
        return -1;
    }

    // touch: move way to MRU in lru_order
    static void touch_mru(std::vector<Set> &sets, int ways, int set_idx, int way) {
        Set &s = sets[set_idx];
        int pos = -1;
        for (int i = 0; i < ways; ++i) if (s.lru_order[i] == way) { pos = i; break; }
        if (pos <= 0) return; // already MRU or not found
        for (int i = pos; i > 0; --i) s.lru_order[i] = s.lru_order[i-1];
        s.lru_order[0] = way;
    }

    // pick invalid way if any, else LRU way index
    static int victim_way(const std::vector<Set> &sets, int ways, int set_idx) {
        const Set &s = sets[set_idx];
        for (int w = 0; w < ways; ++w) if (!s.ways[w].valid) return w;
        return s.lru_order[ways - 1];
    }

    // set a line and mark MRU
    static void set_line(std::vector<Set> &sets, int ways, int set_idx, int way, bool valid, bool dirty, uint64_t tag) {
        Set &s = sets[set_idx];
        s.ways[way].valid = valid;
        s.ways[way].dirty = dirty;
        s.ways[way].tag = tag;
        // update lru order: move 'way' to MRU
        int pos = -1;
        for (int i = 0; i < ways; ++i) if (s.lru_order[i] == way) { pos = i; break; }
        if (pos > 0) {
            for (int i = pos; i > 0; --i) s.lru_order[i] = s.lru_order[i-1];
            s.lru_order[0] = way;
        } else if (pos == -1) {
            // way not present in order (shouldn't happen), rotate right and set at MRU
            for (int i = ways - 1; i > 0; --i) s.lru_order[i] = s.lru_order[i-1];
            s.lru_order[0] = way;
        }
    }
};

} // namespace

extern "C" csot::CacheSim* create_cache_sim() {
    return new BaselineCacheSim();
}