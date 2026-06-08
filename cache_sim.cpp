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

void operator delete(void* p) noexcept {
    std::free(p);
}

void operator delete(void* p, std::size_t) noexcept {
    std::free(p);
}
#endif

namespace {

// Compile-time helper: number of bits needed to index SETS
constexpr int log2_of(int v) { int r = 0; while ((1 << r) < v) ++r; return r; }

// ============================================================================
// WayEntry — per-way data packed together for cache-line locality.
// When find_way scans 8 ways, all tag+valid+dirty data for one set lives
// in 8 × 16 = 128 bytes = 2 cache lines, instead of bouncing between
// separate arrays that are tens of KiB apart.
// ============================================================================
struct alignas(16) WayEntry {
    std::uint64_t tag;      // 8 bytes
    std::uint8_t  valid;    // 1 byte
    std::uint8_t  dirty;    // 1 byte
    std::uint8_t  _pad[6];  // pad to 16 bytes for alignment
};
static_assert(sizeof(WayEntry) == 16, "WayEntry must be exactly 16 bytes");

// ============================================================================
// Templatized Level — AoS layout for tag/valid/dirty, separate LRU array.
// SETS and WAYS are compile-time constants for unrolling and constant folding.
// ============================================================================
template <int SETS, int WAYS>
struct Level {
    static constexpr int NUM_SETS = SETS;
    static constexpr int NUM_WAYS = WAYS;
    static constexpr int INDEX_BITS = log2_of(SETS);
    static constexpr std::uint64_t INDEX_MASK = SETS - 1;

    // AoS: per-way data packed together (tag + valid + dirty in one struct)
    // 8 ways × 16 bytes = 128 bytes per set = 2 cache lines per set
    std::array<WayEntry, SETS * WAYS> ways{};

    // LRU order kept separate — only touched on hits/fills, not during tag scan
    std::array<std::uint8_t, SETS * WAYS> lru{};

    void init() {
        for (auto& w : ways) { w.tag = 0; w.valid = 0; w.dirty = 0; }
        // Initialize LRU: way 0 = MRU, way WAYS-1 = LRU
        for (int si = 0; si < SETS; ++si) {
            for (int w = 0; w < WAYS; ++w) {
                lru[si * WAYS + w] = static_cast<std::uint8_t>(w);
            }
        }
    }

    // Extract set index from a block address
    static constexpr int set_of(std::uint64_t block_addr) {
        return static_cast<int>(block_addr & INDEX_MASK);
    }

    // Extract tag from a block address
    static constexpr std::uint64_t tag_of(std::uint64_t block_addr) {
        return block_addr >> INDEX_BITS;
    }

    // Find the way with matching tag; return -1 if not found.
    // All 8 WayEntry structs for one set are in 2 adjacent cache lines.
    int find_way(int set_idx, std::uint64_t t) const {
        const std::size_t base = static_cast<std::size_t>(set_idx) * WAYS;
        for (int w = 0; w < WAYS; ++w) {
            if (ways[base + w].valid && ways[base + w].tag == t)
                return w;
        }
        return -1;
    }

    // Move way to MRU position in the per-set LRU array
    void touch_mru(int set_idx, int way) {
        std::size_t base = static_cast<std::size_t>(set_idx) * WAYS;
        int pos = -1;
        for (int i = 0; i < WAYS; ++i) {
            if (lru[base + i] == static_cast<std::uint8_t>(way)) { pos = i; break; }
        }
        if (pos <= 0) return;
        for (int i = pos; i > 0; --i) lru[base + i] = lru[base + i - 1];
        lru[base] = static_cast<std::uint8_t>(way);
    }

    // Pick an invalid way if any, else return the LRU way
    int victim_way(int set_idx) const {
        std::size_t base = static_cast<std::size_t>(set_idx) * WAYS;
        for (int w = 0; w < WAYS; ++w) {
            if (!ways[base + w].valid) return w;
        }
        return static_cast<int>(lru[base + WAYS - 1]);
    }

    // Install a line (valid/dirty/tag) and mark MRU
    void set_line(int set_idx, int way, bool v, std::uint8_t d, std::uint64_t t) {
        std::size_t base = static_cast<std::size_t>(set_idx) * WAYS;
        ways[base + way].valid = v ? 1 : 0;
        ways[base + way].dirty = d ? 1 : 0;
        ways[base + way].tag = t;
        // update MRU
        int pos = -1;
        for (int i = 0; i < WAYS; ++i) {
            if (lru[base + i] == static_cast<std::uint8_t>(way)) { pos = i; break; }
        }
        if (pos > 0) {
            for (int i = pos; i > 0; --i) lru[base + i] = lru[base + i - 1];
            lru[base] = static_cast<std::uint8_t>(way);
        } else if (pos == -1) {
            for (int i = WAYS - 1; i > 0; --i) lru[base + i] = lru[base + i - 1];
            lru[base] = static_cast<std::uint8_t>(way);
        }
    }
};

// Instantiate the two levels with their geometry from CACHE_SPEC.md
using L1 = Level<64, 8>;    // 32 KiB, 8-way, 64 sets
using L2 = Level<512, 8>;   // 256 KiB, 8-way, 512 sets

class BaselineCacheSim final : public csot::CacheSim {
public:
    void on_init() override {
        l1_.init();
        l2_.init();
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
            const csot::MemAccess &a = acc[i];
            const std::uint64_t addr = a.address;
            const bool wr = (a.is_write != 0);

            // §5.1 count the access (branchless)
            st.writes += wr;
            st.reads += !wr;

            // block address
            const std::uint64_t b = addr >> 6;

            // L1 set/tag — masks and shifts are compile-time constants
            const int s1 = L1::set_of(b);
            const std::uint64_t t1 = L1::tag_of(b);

            // L2 set/tag
            const int s2 = L2::set_of(b);
            const std::uint64_t t2 = L2::tag_of(b);

            // §5.2 probe L1
            int w1 = l1_.find_way(s1, t1);
            bool l1_hit = (w1 >= 0);
            st.l1_hits += l1_hit;
            st.l1_misses += !l1_hit;

            if (l1_hit) {
                l1_.touch_mru(s1, w1);
                l1_.ways[static_cast<std::size_t>(s1) * L1::NUM_WAYS + w1].dirty |= wr;
                continue;
            }

            // §5.3 probe L2 (only on L1 miss)
            int w2 = l2_.find_way(s2, t2);
            bool l2_hit = (w2 >= 0);
            st.l2_hits += l2_hit;
            st.l2_misses += !l2_hit;

            if (l2_hit) {
                l2_.touch_mru(s2, w2);
            } else {
                int victim = l2_.victim_way(s2);
                // Branchless dirty writeback check
                const auto& vw = l2_.ways[static_cast<std::size_t>(s2) * L2::NUM_WAYS + victim];
                st.dirty_writebacks += (vw.valid & vw.dirty);
                // install clean line
                l2_.set_line(s2, victim, true, false, t2);
            }

            // §5.4 fill into L1 (write-allocate)
            int v1 = l1_.victim_way(s1);
            const auto& l1_victim = l1_.ways[static_cast<std::size_t>(s1) * L1::NUM_WAYS + v1];
            if (l1_victim.valid && l1_victim.dirty) {
                // Writing dirty L1 victim back to L2 (§5.5)
                std::uint64_t bv = (l1_victim.tag << L1::INDEX_BITS) | static_cast<std::uint64_t>(s1);
                int s2v = L2::set_of(bv);
                std::uint64_t t2v = L2::tag_of(bv);

                int wv = l2_.find_way(s2v, t2v);
                if (wv >= 0) {
                    // set dirty on existing L2 line; do NOT touch LRU or counts
                    l2_.ways[static_cast<std::size_t>(s2v) * L2::NUM_WAYS + wv].dirty = 1;
                } else {
                    // install dirty into L2; may evict L2 victim -> memory
                    int vv = l2_.victim_way(s2v);
                    // Branchless dirty writeback check
                    const auto& l2v = l2_.ways[static_cast<std::size_t>(s2v) * L2::NUM_WAYS + vv];
                    st.dirty_writebacks += (l2v.valid & l2v.dirty);
                    l2_.set_line(s2v, vv, true, true, t2v);
                }
            }

            // place the requested line into L1; dirty iff write
            l1_.set_line(s1, v1, true, wr ? 1 : 0, t1);
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