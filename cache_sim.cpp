// ============================================================================
//  cache_sim.cpp — Optimized Week-2 cache-sim
//
//  Compile-time: consteval geometry, packed LRU identity in .rodata
//  Runtime:      branchless math, force-inlined helpers, packed uint64_t LRU
//
//      cmake -B build -DCSOT_CACHE_SIM_SRC=cache_sim.cpp && cmake --build build -j
//      ./build/cache_sim_runner data/tiny.trace
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

// ============================================================================
// Compile-time geometry — consteval guarantees zero runtime cost
// ============================================================================
namespace geo {
    // consteval: MUST be evaluated at compile time, never sneaks into runtime
    consteval std::uint64_t mask(std::uint64_t sets) { return sets - 1; }

    inline constexpr int LINE_BITS   = 6;   // log2(64-byte cache line)

    inline constexpr int L1_SETS     = 64;
    inline constexpr int L1_WAYS     = 8;
    inline constexpr int L1_IDX_BITS = 6;   // log2(L1_SETS)
    inline constexpr std::uint64_t L1_MASK = mask(L1_SETS);   // = 63, forced compile-time

    inline constexpr int L2_SETS     = 512;
    inline constexpr int L2_WAYS     = 8;
    inline constexpr int L2_IDX_BITS = 9;   // log2(L2_SETS)
    inline constexpr std::uint64_t L2_MASK = mask(L2_SETS);   // = 511, forced compile-time
} // namespace geo

// ============================================================================
// Compile-time packed LRU identity value — lives in .rodata, zero runtime cost
//
// Packing (little-endian): byte 0 = MRU way, byte 7 = LRU way
// Identity: way 0 at MRU, way 7 at LRU → 0x07'06'05'04'03'02'01'00
// ============================================================================
consteval std::uint64_t make_identity_lru() {
    std::uint64_t v = 0;
    for (int w = 0; w < 8; ++w)
        v |= static_cast<std::uint64_t>(w) << (w * 8);
    return v;
}
inline constexpr std::uint64_t IDENTITY_LRU = make_identity_lru();

// ============================================================================
// SoA Level layout: flat arrays for tag/valid/dirty, packed uint64_t LRU
// ============================================================================
struct Level {
    static constexpr std::size_t WAYS = 8;
    std::size_t sets = 0;
    std::vector<std::uint64_t> tag;     // size = sets * WAYS
    std::vector<std::uint8_t>  valid;   // 0/1, size = sets * WAYS
    std::vector<std::uint8_t>  dirty;   // 0/1, size = sets * WAYS
    std::vector<std::uint64_t> lru;     // PACKED: 1 uint64_t per set (8 way indices in bytes)

    void init(std::size_t s) {
        sets = s;
        tag.assign(s * WAYS, 0);
        valid.assign(s * WAYS, 0);
        dirty.assign(s * WAYS, 0);
        lru.assign(s, IDENTITY_LRU);   // compile-time constant, each set = identity permutation
    }
};

// ============================================================================
// Optimized cache simulator
// ============================================================================
class BaselineCacheSim final : public csot::CacheSim {
public:
    void on_init() override {
        l1.init(geo::L1_SETS);
        l2.init(geo::L2_SETS);
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
            const uint64_t addr = a.address;
            const uint8_t wr = a.is_write;

            // §5.1 count the access (branchless)
            st.writes += wr;
            st.reads  += (wr ^ 1);

            // block address and indices/tags (consteval masks — guaranteed compile-time)
            const uint64_t b = addr >> geo::LINE_BITS;

            const int s1 = static_cast<int>(b & geo::L1_MASK);
            const uint64_t t1 = b >> geo::L1_IDX_BITS;

            const int s2 = static_cast<int>(b & geo::L2_MASK);
            const uint64_t t2 = b >> geo::L2_IDX_BITS;

            // §5.2 probe L1
            int w1 = find_way(l1, s1, t1);
            bool l1_hit = (w1 >= 0);
            st.l1_hits   += l1_hit;
            st.l1_misses += !l1_hit;

            if (l1_hit) {   // predictable branch — let the CPU speculate
                touch_mru(l1, s1, w1);
                l1.dirty[static_cast<std::size_t>(s1) * Level::WAYS + w1] |= wr;
                continue;
            }

            // §5.3 probe L2 (only on L1 miss)
            int w2 = find_way(l2, s2, t2);
            bool l2_hit = (w2 >= 0);
            st.l2_hits   += l2_hit;
            st.l2_misses += !l2_hit;

            if (l2_hit) {
                touch_mru(l2, s2, w2);
            } else {
                int victim = victim_way(l2, s2);
                // Branchless dirty writeback check
                st.dirty_writebacks += (l2.valid[static_cast<std::size_t>(s2) * Level::WAYS + victim] &
                                        l2.dirty[static_cast<std::size_t>(s2) * Level::WAYS + victim]);
                // install clean line
                set_line(l2, s2, victim, 1, 0, t2);
            }

            // §5.4 fill into L1 (write-allocate)
            int v1 = victim_way(l1, s1);
            if (l1.valid[static_cast<std::size_t>(s1) * Level::WAYS + v1] &&
                l1.dirty[static_cast<std::size_t>(s1) * Level::WAYS + v1]) {
                // Writing dirty L1 victim back to L2 (§5.5)
                uint64_t victim_tag = l1.tag[static_cast<std::size_t>(s1) * Level::WAYS + v1];
                uint64_t bv = (victim_tag << geo::L1_IDX_BITS) | static_cast<uint64_t>(s1);
                int s2v = static_cast<int>(bv & geo::L2_MASK);
                uint64_t t2v = bv >> geo::L2_IDX_BITS;

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
                    set_line(l2, s2v, vv, 1, 1, t2v);
                }
            }

            // place the requested line into L1; dirty iff write
            set_line(l1, s1, v1, 1, wr, t1);
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

    // ========================================================================
    // Force-inlined hot helpers — zero call overhead on the hot path
    // ========================================================================

    // find the way index with matching tag; return -1 if not found (branchless scan)
    __attribute__((always_inline))
    static inline int find_way(const Level &L, int set_idx, uint64_t tag) {
        const std::size_t base = static_cast<std::size_t>(set_idx) * Level::WAYS;
        int match = -1;
        for (int w = 0; w < static_cast<int>(Level::WAYS); ++w) {
            match = (L.valid[base + w] && L.tag[base + w] == tag) ? w : match;
        }
        return match;
    }

    // touch: move way to MRU in packed LRU uint64_t (branchless shift, single load/store)
    __attribute__((always_inline))
    static inline void touch_mru(Level &L, int set_idx, int way) {
        uint64_t packed = L.lru[set_idx];       // single 8-byte load
        uint8_t arr[8];
        std::memcpy(arr, &packed, 8);           // compiler turns this into register ops

        const uint8_t target = static_cast<uint8_t>(way);
        bool found = (arr[0] == target);
        uint8_t prev = arr[0];
        arr[0] = target;

        for (int i = 1; i < 8; ++i) {
            uint8_t curr = arr[i];
            bool is_target = (curr == target);
            arr[i] = found ? curr : prev;       // branchless cmov
            found |= is_target;
            prev = curr;
        }

        std::memcpy(&packed, arr, 8);           // compiler turns this into register ops
        L.lru[set_idx] = packed;                // single 8-byte store
    }

    // pick invalid way if any, else LRU victim (byte 7 of packed uint64_t — one shift+mask)
    __attribute__((always_inline))
    static inline int victim_way(const Level &L, int set_idx) {
        const std::size_t base = static_cast<std::size_t>(set_idx) * Level::WAYS;
        int invalid_way = -1;
        // Scan backwards so first invalid way (w=0) wins
        for (int w = static_cast<int>(Level::WAYS) - 1; w >= 0; --w) {
            invalid_way = (L.valid[base + w] == 0) ? w : invalid_way;
        }
        // LRU victim = byte 7 (MSB) of packed uint64_t — single shift + mask
        return (invalid_way != -1) ? invalid_way
                                   : static_cast<int>((L.lru[set_idx] >> 56) & 0xFF);
    }

    // set a line (valid/dirty/tag) and mark MRU
    __attribute__((always_inline))
    static inline void set_line(Level &L, int set_idx, int way,
                                uint8_t valid, uint8_t dirty, uint64_t tag) {
        const std::size_t base = static_cast<std::size_t>(set_idx) * Level::WAYS;
        L.valid[base + way] = valid;
        L.dirty[base + way] = dirty;
        L.tag[base + way]   = tag;
        touch_mru(L, set_idx, way);
    }
};

} // namespace

extern "C" csot::CacheSim* create_cache_sim() {
    return new BaselineCacheSim();
}