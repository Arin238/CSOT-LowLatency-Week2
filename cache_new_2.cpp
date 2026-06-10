// ============================================================================
//  cache_new_2.cpp — Ground-up rewrite: bit-matrix LRU, zero table pressure
//
//  Key design decisions:
//  1. Bit-matrix LRU: 3 ALU ops per touch, 0 bytes of lookup table.
//     Eliminates the 630KB next_state table that polluted L1d.
//  2. Store full block address in tag[] (not shifted tag). Eliminates
//     tag reconstruction on L1 dirty writeback (was: bv = (tag << 6) | s1).
//  3. Interleaved metadata: valid/dirty/lru packed per-set for spatial locality.
//  4. SIMD tag comparison (AVX2 with inline asm broadcast, SSE4.1 fallback).
//  5. Software prefetching: next MemAccess + L1 tag line.
// ============================================================================

#include "cache_sim.hpp"

#include <cstddef>
#include <cstdint>
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

namespace {
    alignas(4096) char g_pool[1024 * 1024 * 2];
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

namespace {

// ============================================================================
// Bit-Matrix LRU
//
// State: a single uint64_t per set, encoding an 8x8 triangular bit-matrix.
// Each byte i represents row i. Bit j in row i means "way i was used more
// recently than way j". The LRU victim is the way whose row is all-zeros.
//
// touch_mru(way): set row[way] = 0xFF, clear column[way] across all rows.
//   3 ALU ops: OR, AND, (constant computation of row/col masks).
//   No memory access. No table. Zero L1d pressure.
//
// victim_way: find the zero-byte using the SWAR "has zero byte" trick,
//   or use an invalid way if one exists.
// ============================================================================

// Initial matrix state: way 0 is MRU, way 7 is LRU
// Row i has bits set for all ways j < i (i.e., way 0 was touched most recently)
static constexpr std::uint64_t LRU_INIT = 0x0080'C0E0'F0F8'FCFEULL;

// ============================================================================
// Level template: SoA layout with bit-matrix LRU
// ============================================================================
template <int SETS, int WAYS>
struct Level {
    static constexpr int NUM_SETS  = SETS;
    static constexpr int NUM_WAYS  = WAYS;
    static constexpr int INDEX_BITS = []{ int r = 0; while ((1 << r) < SETS) ++r; return r; }();
    static constexpr std::uint64_t INDEX_MASK = SETS - 1;

    // Tags: store full block address (b = addr >> 6), not shifted tag.
    // This means find_way compares against b directly, and writeback
    // can read bv straight from tag[] without reconstruction.
    alignas(64) std::uint64_t tag[SETS * WAYS];

    // Per-set metadata, packed for spatial locality
    alignas(64) std::uint8_t  valid[SETS];
    alignas(64) std::uint8_t  dirty[SETS];
    alignas(64) std::uint64_t lru[SETS];  // bit-matrix state

    void init() {
        // Fill tags with impossible value (~0ULL) so invalid slots never match
        std::memset(tag, 0xFF, sizeof(tag));
        std::memset(valid, 0, sizeof(valid));
        std::memset(dirty, 0, sizeof(dirty));
        for (int i = 0; i < SETS; ++i) lru[i] = LRU_INIT;
    }

    [[gnu::always_inline]] static constexpr int set_of(std::uint64_t blk) {
        return static_cast<int>(blk & INDEX_MASK);
    }

    // ── Tag comparison: AVX2 > SSE4.1 > scalar ──────────────────────────
    [[gnu::always_inline]] int find_way(int si, std::uint64_t b) const {
        const std::size_t base = static_cast<std::size_t>(si) * WAYS;

#if defined(__AVX2__)
        __m256i key;
        // Inline asm avoids the compiler's stack spill for vpbroadcastq
        __asm__ ("vmovq %1, %%xmm0\n\t"
                 "vpbroadcastq %%xmm0, %0"
                 : "=x"(key) : "r"(b) : "xmm0");

        __m256i a = _mm256_load_si256(reinterpret_cast<const __m256i*>(&tag[base]));
        __m256i c = _mm256_load_si256(reinterpret_cast<const __m256i*>(&tag[base + 4]));

        unsigned m = static_cast<unsigned>(_mm256_movemask_pd(
                        _mm256_castsi256_pd(_mm256_cmpeq_epi64(a, key))))
                   | (static_cast<unsigned>(_mm256_movemask_pd(
                        _mm256_castsi256_pd(_mm256_cmpeq_epi64(c, key)))) << 4);
#elif defined(__SSE4_1__)
        __m128i key = _mm_set1_epi64x(b);
        __m128i a = _mm_load_si128(reinterpret_cast<const __m128i*>(&tag[base + 0]));
        __m128i b_ = _mm_load_si128(reinterpret_cast<const __m128i*>(&tag[base + 2]));
        __m128i c = _mm_load_si128(reinterpret_cast<const __m128i*>(&tag[base + 4]));
        __m128i d = _mm_load_si128(reinterpret_cast<const __m128i*>(&tag[base + 6]));
        unsigned m = static_cast<unsigned>(_mm_movemask_pd(_mm_castsi128_pd(_mm_cmpeq_epi64(a, key))))
                   | (static_cast<unsigned>(_mm_movemask_pd(_mm_castsi128_pd(_mm_cmpeq_epi64(b_, key)))) << 2)
                   | (static_cast<unsigned>(_mm_movemask_pd(_mm_castsi128_pd(_mm_cmpeq_epi64(c, key)))) << 4)
                   | (static_cast<unsigned>(_mm_movemask_pd(_mm_castsi128_pd(_mm_cmpeq_epi64(d, key)))) << 6);
#else
        unsigned m = 0;
        for (int w = 0; w < WAYS; ++w)
            m |= static_cast<unsigned>(tag[base + w] == b) << w;
#endif
        return m ? __builtin_ctz(m) : -1;
    }

    // ── Bit-matrix LRU: touch way to MRU position ───────────────────────
    [[gnu::always_inline]] void touch_mru(int si, int way) {
        // Set all bits in row[way] (this way beats everyone)
        const std::uint64_t row_mask = 0xFFULL << (way * 8);
        // Clear bit `way` from every row (nobody beats this way)
        const std::uint64_t col_mask = ~(0x0101010101010101ULL << way);
        lru[si] = (lru[si] | row_mask) & col_mask;
    }

    // ── Find victim: invalid way first, then true LRU ───────────────────
    [[gnu::always_inline]] int victim_way(int si) const {
        const unsigned inv = static_cast<unsigned>(~valid[si]) & 0xFF;
        if (__builtin_expect(inv, 0)) {
            return __builtin_ctz(inv);
        }
        // SWAR zero-byte detection: the LRU way has an all-zero row
        const std::uint64_t m = lru[si];
        const std::uint64_t has_zero = (m - 0x0101010101010101ULL) & ~m & 0x8080808080808080ULL;
        return __builtin_ctzll(has_zero) >> 3;
    }

    // ── Install a line: set tag, valid, dirty, touch MRU ────────────────
    [[gnu::always_inline]] void set_line(int si, int way, bool v, bool d, std::uint64_t b) {
        tag[static_cast<std::size_t>(si) * WAYS + way] = b;

        const std::uint8_t mask = static_cast<std::uint8_t>(1u << way);
        valid[si] = (valid[si] & ~mask) | (static_cast<std::uint8_t>(-static_cast<int>(v)) & mask);
        dirty[si] = (dirty[si] & ~mask) | (static_cast<std::uint8_t>(-static_cast<int>(d)) & mask);

        touch_mru(si, way);
    }
};

using L1 = Level<64, 8>;
using L2 = Level<512, 8>;

// ============================================================================
// Cache Simulator
// ============================================================================
class FastCacheSim final : public csot::CacheSim {
public:
    void on_init() override {
        l1_.init();
        l2_.init();
    }

    // ── Process a single memory access ──────────────────────────────────
    [[gnu::always_inline, gnu::hot]]
    void process_one(const csot::MemAccess* __restrict acc,
                     std::uint64_t& c_writes,
                     std::uint64_t& c_l1_misses,
                     std::uint64_t& c_l2_hits,
                     std::uint64_t& c_dirty_writebacks) {
        // Prefetch next access's data (stream prefetch, low temporal locality)
        __builtin_prefetch(acc + 8, 0, 0);

        const std::uint32_t wr = acc->is_write;
        c_writes += wr;

        const std::uint64_t b = acc->address >> 6;
        const int s1 = L1::set_of(b);

        // ── L1 probe ────────────────────────────────────────────────────
        const int w1 = l1_.find_way(s1, b);

        if (__builtin_expect(w1 >= 0, 1)) {
            // L1 HIT — fast path
            l1_.touch_mru(s1, w1);
            l1_.dirty[s1] |= static_cast<std::uint8_t>(wr << w1);
            return;
        }

        // L1 MISS
        ++c_l1_misses;
        // Compiler barrier: prevent the compiler from transforming miss counting
        __asm__ volatile("" : "+r"(c_l1_misses));

        // ── L2 probe ────────────────────────────────────────────────────
        const int s2 = L2::set_of(b);
        const int w2 = l2_.find_way(s2, b);
        const bool l2_hit = (w2 >= 0);
        c_l2_hits += l2_hit;

        if (l2_hit) {
            l2_.touch_mru(s2, w2);
        } else {
            // L2 MISS — demand-install clean line from memory
            const int victim = l2_.victim_way(s2);
            c_dirty_writebacks += ((l2_.valid[s2] & l2_.dirty[s2]) >> victim) & 1;
            l2_.set_line(s2, victim, true, false, b);
        }

        // ── L1 fill (write-allocate) ────────────────────────────────────
        const int v1 = l1_.victim_way(s1);

        // Check if L1 victim is dirty and needs writeback to L2
        if (((l1_.valid[s1] & l1_.dirty[s1]) >> v1) & 1) {
            // Block address stored directly — no reconstruction needed
            const std::uint64_t bv = l1_.tag[static_cast<std::size_t>(s1) * L1::NUM_WAYS + v1];
            const int s2v = L2::set_of(bv);

            const int wv = l2_.find_way(s2v, bv);
            if (wv >= 0) {
                // L2 has the line — just mark dirty (NO LRU touch per spec §5.5)
                l2_.dirty[s2v] |= static_cast<std::uint8_t>(1u << wv);
            } else {
                // L2 doesn't have it — install dirty
                const int vv = l2_.victim_way(s2v);
                c_dirty_writebacks += ((l2_.valid[s2v] & l2_.dirty[s2v]) >> vv) & 1;
                l2_.set_line(s2v, vv, true, true, bv);
            }
        }

        // Install the new line in L1
        l1_.set_line(s1, v1, true, wr, b);
    }

    csot::CacheStats run(const csot::MemAccess* __restrict acc, std::size_t n) override {
#ifdef CSOT_CHECK_ALLOCS
        g_hot_path_active = true;
#ifdef __linux__
        prctl(PR_TASK_PERF_EVENTS_ENABLE);
#endif
#endif
        std::uint64_t c_writes = 0;
        std::uint64_t c_l1_misses = 0;
        std::uint64_t c_l2_hits = 0;
        std::uint64_t c_dirty_writebacks = 0;

        const csot::MemAccess* const end = acc + n;
        for (; acc < end; ++acc) {
            process_one(acc, c_writes, c_l1_misses, c_l2_hits, c_dirty_writebacks);
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
    return new FastCacheSim();
}