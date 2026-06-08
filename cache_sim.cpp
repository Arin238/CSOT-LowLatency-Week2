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

void operator delete(void* p) noexcept {
    std::free(p);
}

void operator delete(void* p, std::size_t) noexcept {
    std::free(p);
}
#endif

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

// SoA Level layout: flat arrays per field, plus per-set LRU order (MRU..LRU)
struct Level {
    static constexpr std::size_t WAYS = 8;
    std::size_t sets = 0;
    std::vector<std::uint64_t> tag;   // size = sets * WAYS
    std::vector<uint8_t> valid;       // 0/1, size = sets * WAYS
    std::vector<uint8_t> dirty;       // 0/1, size = sets * WAYS
    std::vector<uint8_t> lru;         // per-set MRU..LRU order, size = sets * WAYS

    void init(std::size_t s) {
        sets = s;
        tag.assign(s * WAYS, 0);
        valid.assign(s * WAYS, 0);
        dirty.assign(s * WAYS, 0);
        lru.assign(s * WAYS, 0);
        for (std::size_t si = 0; si < s; ++si) {
            for (std::size_t w = 0; w < WAYS; ++w) {
                lru[si * WAYS + w] = static_cast<uint8_t>(w); // MRU..LRU initial
            }
        }
    }
};

class BaselineCacheSim final : public csot::CacheSim {
public:
    void on_init() override {
        // allocate and initialize L1 and L2 levels (SoA)
        l1.init(L1_SETS);
        l2.init(L2_SETS);
    }

    csot::CacheStats run(const csot::MemAccess* acc, std::size_t n) override {
#ifdef CSOT_CHECK_ALLOCS
        g_hot_path_active = true;
#ifdef __linux__
        prctl(PR_TASK_PERF_EVENTS_ENABLE);
#endif
#endif

        csot::CacheStats st{};

        for (std::size_t i = 0; i < n; ++i) {
            const csot::MemAccess &a = acc[i]; //ARIN- cache
            const uint64_t addr = a.address; // ARIN- Gain Intuition
            const bool wr = (a.is_write != 0); // ARIN- Gain Intuition

            // §5.1 count the access (branchless)
            st.writes += wr;
            st.reads += !wr;

            // block address and indices/tags
            const uint64_t b = addr >> 6; // block address

            const int s1 = static_cast<int>(b & (L1_SETS - 1));
            const uint64_t t1 = b >> L1_INDEX_BITS; // b >> 6

            const int s2 = static_cast<int>(b & (L2_SETS - 1));
            const uint64_t t2 = b >> L2_INDEX_BITS; // b >> 9

            // §5.2 probe L1
            int w1 = find_way(l1, s1, t1);
            bool l1_hit = (w1 >= 0);
            st.l1_hits += l1_hit;
            st.l1_misses += !l1_hit;

            if (l1_hit) { // Let the CPU predict this! (Very fast)
                touch_mru(l1, s1, w1);
                l1.dirty[static_cast<std::size_t>(s1) * Level::WAYS + w1] |= wr;
                continue; 
            }

            // §5.3 probe L2 (only on L1 miss)
            int w2 = find_way(l2, s2, t2);
            bool l2_hit = (w2 >= 0);
            st.l2_hits += l2_hit;
            st.l2_misses += !l2_hit;

            if (l2_hit) {
                touch_mru(l2, s2, w2);
            } else {
                int victim = victim_way(l2, s2);
                // Branchless dirty writeback check
                st.dirty_writebacks += (l2.valid[static_cast<std::size_t>(s2) * Level::WAYS + victim] & 
                                        l2.dirty[static_cast<std::size_t>(s2) * Level::WAYS + victim]);
                // install clean line
                set_line(l2, s2, victim, true, false, t2);
            }

            // §5.4 fill into L1 (write-allocate)
            int v1 = victim_way(l1, s1);
            if (l1.valid[static_cast<std::size_t>(s1) * Level::WAYS + v1] &&
                l1.dirty[static_cast<std::size_t>(s1) * Level::WAYS + v1]) {
                // Writing dirty L1 victim back to L2 (§5.5)
                uint64_t victim_tag = l1.tag[static_cast<std::size_t>(s1) * Level::WAYS + v1];
                uint64_t bv = (victim_tag << L1_INDEX_BITS) | static_cast<uint64_t>(s1);
                int s2v = static_cast<int>(bv & (L2_SETS - 1));
                uint64_t t2v = bv >> L2_INDEX_BITS;

                int wv = find_way(l2, s2v, t2v);
                if (wv >= 0) {
                    // set dirty on existing L2 line; do NOT touch LRU or counts
                    l2.dirty[static_cast<std::size_t>(s2v) * Level::WAYS + wv] = 1;
                } else {
                    // install dirty into L2; may evict L2 victim -> memory
                    int vv = victim_way(l2, s2v);
                    // Branchless dirty writeback check
                    st.dirty_writebacks += (l2.valid[static_cast<std::size_t>(s2v) * Level::WAYS + vv] &
                                            l2.dirty[static_cast<std::size_t>(s2v) * Level::WAYS + vv]);
                    set_line(l2, s2v, vv, true, true, t2v);
                }
            }

            // place the requested line into L1; dirty iff write
            set_line(l1, s1, v1, true, wr ? 1 : 0, t1);
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
    Level l1;
    Level l2;

    // find the way index with matching tag in a Level; return -1 if not found
    [[gnu::always_inline]] inline static int find_way(const Level &L, int set_idx, uint64_t tag) {
        const std::size_t base = static_cast<std::size_t>(set_idx) * Level::WAYS;
        for (std::size_t w = 0; w < Level::WAYS; ++w) {
            if (L.valid[base + w] && L.tag[base + w] == tag) return static_cast<int>(w);
        }
        return -1;
    }

    // touch: move way to MRU in per-set lru array
    [[gnu::always_inline]] inline static void touch_mru(Level &L, int set_idx, int way) {
        std::size_t base = static_cast<std::size_t>(set_idx) * Level::WAYS;
        int pos = -1;
        for (std::size_t i = 0; i < Level::WAYS; ++i) if (L.lru[base + i] == static_cast<uint8_t>(way)) { pos = static_cast<int>(i); break; }
        if (pos <= 0) return;
        for (int i = pos; i > 0; --i) L.lru[base + i] = L.lru[base + i - 1];
        L.lru[base + 0] = static_cast<uint8_t>(way);
    }

    // pick invalid way if any, else LRU way index
    [[gnu::always_inline]] inline static int victim_way(const Level &L, int set_idx) {
        std::size_t base = static_cast<std::size_t>(set_idx) * Level::WAYS;
        for (std::size_t w = 0; w < Level::WAYS; ++w) if (!L.valid[base + w]) return static_cast<int>(w);
        return static_cast<int>(L.lru[base + Level::WAYS - 1]);
    }

    // set a line (valid/dirty/tag) and mark MRU for that way
    [[gnu::noinline]] inline static void set_line(Level &L, int set_idx, int way, bool valid, uint8_t dirty, uint64_t tag) {
        std::size_t base = static_cast<std::size_t>(set_idx) * Level::WAYS;
        L.valid[base + way] = valid ? 1 : 0;
        L.dirty[base + way] = dirty ? 1 : 0;
        L.tag[base + way] = tag;
        // update MRU
        int pos = -1;
        for (std::size_t i = 0; i < Level::WAYS; ++i) if (L.lru[base + i] == static_cast<uint8_t>(way)) { pos = static_cast<int>(i); break; }
        if (pos > 0) {
            for (int i = pos; i > 0; --i) L.lru[base + i] = L.lru[base + i - 1];
            L.lru[base + 0] = static_cast<uint8_t>(way);
        } else if (pos == -1) {
            for (int i = static_cast<int>(Level::WAYS) - 1; i > 0; --i) L.lru[base + i] = L.lru[base + i - 1];
            L.lru[base + 0] = static_cast<uint8_t>(way);
        }
    }
};

} // namespace

extern "C" csot::CacheSim* create_cache_sim() {
    return new BaselineCacheSim();
}