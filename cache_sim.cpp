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
#include <cstdlib>
#include <cstring>

#ifdef CSOT_CHECK_ALLOCS
#include <new>
#include <iostream>
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
// WayEntry — AoS per-way data, packed for cache locality.
//
// 16 bytes per entry → 8 ways per set = 128 bytes = exactly 2 cache lines.
// When find_way probes a set, it touches at most 2 cache lines instead of
// bouncing between tag[] and valid[] arrays that are 32 KiB apart (SoA).
//
// Bit flags in vd:
//   bit 0 (0x01) = valid
//   bit 1 (0x02) = dirty
// ============================================================================
struct WayEntry {
    std::uint64_t tag;     // 8 bytes
    std::uint8_t  vd;      // 1 byte: bit 0 = valid, bit 1 = dirty
    std::uint8_t  pad[7];  // pad to 16 bytes total
};
static_assert(sizeof(WayEntry) == 16, "WayEntry must be 16 bytes for cache-line packing");

static constexpr std::uint8_t VD_VALID = 0x01;
static constexpr std::uint8_t VD_DIRTY = 0x02;
static constexpr std::uint8_t VD_BOTH  = VD_VALID | VD_DIRTY;

// ============================================================================
// Templatized Level — geometry is compile-time, data is heap-allocated.
//
// The Level object itself is just 2 pointers (16 bytes). The actual cache
// data lives on the heap via aligned_alloc, so the BaselineCacheSim object
// stays small and hot-path loads go straight to the data through the pointer.
// ============================================================================
template <int SETS, int WAYS>
struct Level {
    static constexpr int NUM_SETS = SETS;
    static constexpr int NUM_WAYS = WAYS;
    static constexpr int INDEX_BITS = log2_of(SETS);
    static constexpr std::uint64_t INDEX_MASK = SETS - 1;

    // The Level object = just 2 pointers (16 bytes on the stack/object)
    WayEntry*     ways = nullptr;   // SETS * WAYS entries, AoS layout
    std::uint8_t* lru  = nullptr;   // SETS * WAYS bytes for LRU ordering

    void init() {
        // 64-byte aligned allocation for cache-line friendliness
        ways = static_cast<WayEntry*>(
            std::aligned_alloc(64, sizeof(WayEntry) * SETS * WAYS));
        lru = static_cast<std::uint8_t*>(
            std::aligned_alloc(64, sizeof(std::uint8_t) * SETS * WAYS));

        std::memset(ways, 0, sizeof(WayEntry) * SETS * WAYS);
        for (int si = 0; si < SETS; ++si) {
            for (int w = 0; w < WAYS; ++w) {
                lru[si * WAYS + w] = static_cast<std::uint8_t>(w);
            }
        }
    }

    void destroy() {
        std::free(ways);
        std::free(lru);
        ways = nullptr;
        lru = nullptr;
    }

    // Extract set index from a block address (constexpr mask)
    static constexpr int set_of(std::uint64_t block_addr) {
        return static_cast<int>(block_addr & INDEX_MASK);
    }

    // Extract tag from a block address (constexpr shift)
    static constexpr std::uint64_t tag_of(std::uint64_t block_addr) {
        return block_addr >> INDEX_BITS;
    }

    // Find the way with matching tag; return -1 if not found.
    // 8 ways = 128 bytes = 2 cache lines. Compiler unrolls fully.
    int find_way(int set_idx, std::uint64_t t) const {
        const WayEntry* base = ways + set_idx * WAYS;
        for (int w = 0; w < WAYS; ++w) {
            if ((base[w].vd & VD_VALID) && base[w].tag == t)
                return w;
        }
        return -1;
    }

    // Move way to MRU position in the per-set LRU array
    void touch_mru(int set_idx, int way) {
        std::uint8_t* base = lru + set_idx * WAYS;
        int pos = -1;
        for (int i = 0; i < WAYS; ++i) {
            if (base[i] == static_cast<std::uint8_t>(way)) { pos = i; break; }
        }
        if (pos <= 0) return;
        for (int i = pos; i > 0; --i) base[i] = base[i - 1];
        base[0] = static_cast<std::uint8_t>(way);
    }

    // Pick an invalid way if any, else return the LRU way
    int victim_way(int set_idx) const {
        const WayEntry* wbase = ways + set_idx * WAYS;
        for (int w = 0; w < WAYS; ++w) {
            if (!(wbase[w].vd & VD_VALID)) return w;
        }
        return static_cast<int>(lru[set_idx * WAYS + WAYS - 1]);
    }

    // Install a line and mark MRU.
    // vd_flags: use VD_VALID, VD_DIRTY, VD_BOTH constants.
    void set_line(int set_idx, int way, std::uint8_t vd_flags, std::uint64_t t) {
        WayEntry& e = ways[set_idx * WAYS + way];
        e.tag = t;
        e.vd  = vd_flags;

        // update MRU
        std::uint8_t* base = lru + set_idx * WAYS;
        int pos = -1;
        for (int i = 0; i < WAYS; ++i) {
            if (base[i] == static_cast<std::uint8_t>(way)) { pos = i; break; }
        }
        if (pos > 0) {
            for (int i = pos; i > 0; --i) base[i] = base[i - 1];
            base[0] = static_cast<std::uint8_t>(way);
        } else if (pos == -1) {
            for (int i = WAYS - 1; i > 0; --i) base[i] = base[i - 1];
            base[0] = static_cast<std::uint8_t>(way);
        }
    }
};

// Instantiate the two levels with their geometry from CACHE_SPEC.md
using L1 = Level<64, 8>;    // 32 KiB, 8-way, 64 sets
using L2 = Level<512, 8>;   // 256 KiB, 8-way, 512 sets

class BaselineCacheSim final : public csot::CacheSim {
public:
    ~BaselineCacheSim() override {
        l1_.destroy();
        l2_.destroy();
    }

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

        // Cache heap pointers locally so the compiler keeps them in registers
        WayEntry* const     l1w   = l1_.ways;
        std::uint8_t* const l1lru = l1_.lru;
        WayEntry* const     l2w   = l2_.ways;
        std::uint8_t* const l2lru = l2_.lru;

        for (std::size_t i = 0; i < n; ++i) {
            const std::uint64_t addr = acc[i].address;
            const std::uint8_t  wr   = acc[i].is_write;  // 0 or 1

            // §5.1 count the access (branchless)
            st.writes += wr;
            st.reads  += (wr ^ 1);

            // block address
            const std::uint64_t b = addr >> 6;

            // L1 set/tag — masks and shifts are compile-time constants
            const int s1 = L1::set_of(b);
            const std::uint64_t t1 = L1::tag_of(b);

            // §5.2 probe L1 (AoS: tag+vd on same cache line)
            int w1 = l1_.find_way(s1, t1);
            bool l1_hit = (w1 >= 0);
            st.l1_hits   += l1_hit;
            st.l1_misses += !l1_hit;

            if (l1_hit) {
                l1_.touch_mru(s1, w1);
                // set dirty bit if write (branchless: wr << 1 = DIRTY_BIT when wr=1)
                l1w[s1 * L1::NUM_WAYS + w1].vd |= (wr << 1);
                continue;
            }

            // L2 set/tag — only computed on L1 miss (saves work on ~90% of accesses)
            const int s2 = L2::set_of(b);
            const std::uint64_t t2 = L2::tag_of(b);

            // §5.3 probe L2 (only on L1 miss)
            int w2 = l2_.find_way(s2, t2);
            bool l2_hit = (w2 >= 0);
            st.l2_hits   += l2_hit;
            st.l2_misses += !l2_hit;

            if (l2_hit) {
                l2_.touch_mru(s2, w2);
            } else {
                int victim = l2_.victim_way(s2);
                // Branchless dirty writeback: (vd & VALID) & (vd >> 1) = 1 iff both valid+dirty
                std::uint8_t vd = l2w[s2 * L2::NUM_WAYS + victim].vd;
                st.dirty_writebacks += ((vd & 1) & (vd >> 1));
                // install clean line
                l2_.set_line(s2, victim, VD_VALID, t2);
            }

            // §5.4 fill into L1 (write-allocate)
            int v1 = l1_.victim_way(s1);
            std::uint8_t l1_vd = l1w[s1 * L1::NUM_WAYS + v1].vd;
            if ((l1_vd & VD_BOTH) == VD_BOTH) {
                // Writing dirty L1 victim back to L2 (§5.5)
                std::uint64_t victim_tag = l1w[s1 * L1::NUM_WAYS + v1].tag;
                std::uint64_t bv = (victim_tag << L1::INDEX_BITS) | static_cast<std::uint64_t>(s1);
                int s2v = L2::set_of(bv);
                std::uint64_t t2v = L2::tag_of(bv);

                int wv = l2_.find_way(s2v, t2v);
                if (wv >= 0) {
                    // set dirty on existing L2 line; do NOT touch LRU or counts
                    l2w[s2v * L2::NUM_WAYS + wv].vd |= VD_DIRTY;
                } else {
                    // install dirty into L2; may evict L2 victim -> memory
                    int vv = l2_.victim_way(s2v);
                    std::uint8_t vd = l2w[s2v * L2::NUM_WAYS + vv].vd;
                    st.dirty_writebacks += ((vd & 1) & (vd >> 1));
                    l2_.set_line(s2v, vv, VD_BOTH, t2v);
                }
            }

            // place the requested line into L1; dirty iff write
            l1_.set_line(s1, v1, VD_VALID | (wr << 1), t1);
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