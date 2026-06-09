// ============================================================================
//  cache_sim.cpp — final portable version (no consteval, runtime tables)
// ============================================================================

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

namespace {

constexpr int log2_of(int v) { int r = 0; while ((1 << r) < v) ++r; return r; }

// ============================================================================
// LRU state machine – Lehmer code (0…40319) for 8‑way permutations.
// Tables are built once at runtime; hot path uses O(1) array lookups.
// ============================================================================
using Perm = std::array<std::uint8_t, 8>;

static std::uint16_t encode_fast(const Perm& p) {
    constexpr int kFact[7] = {5040, 720, 120, 24, 6, 2, 1};
    std::uint8_t seen = 0;
    std::uint16_t code = 0;
    for (int i = 0; i < 7; ++i) {
        // rank among remaining = (# values < p[i]) − (# of those already placed)
        const int rank = p[i] - __builtin_popcount(seen & ((1u << p[i]) - 1u));
        seen |= (1u << p[i]);
        code += static_cast<std::uint16_t>(rank * kFact[i]);
    }
    return code;
}

static Perm decode_fast(std::uint16_t code) {
    constexpr int kFact[8] = {5040, 720, 120, 24, 6, 2, 1, 1};
    Perm p{};
    std::uint8_t avail = 0xFF;          // bits 0-7 = elements {0..7} still available
    for (int i = 0; i < 8; ++i) {
        const int idx  = code / kFact[i];
        code           %= kFact[i];
        // idx-th set bit: clear idx lowest set bits, then ctz
        std::uint8_t tmp = avail;
        for (int k = 0; k < idx; ++k) tmp &= tmp - 1;
        p[i]  = static_cast<std::uint8_t>(__builtin_ctz(tmp));
        avail &= ~(1u << p[i]);
    }
    return p;
}

struct LruTables {
    std::array<std::uint8_t, 40320> victim{};
    std::array<std::array<std::uint16_t, 8>, 40320> next_state{};
};

static const LruTables* build_lru_tables() {
    static LruTables tables;
    static bool init = false;
    if (!init) {
        for (std::uint16_t state = 0; state < 40320; ++state) {
            const Perm p = decode_fast(state);
            tables.victim[state] = p[7];

            // build inverse once per state — O(1) position lookup below
            Perm inv{};
            for (int i = 0; i < 8; ++i)
                inv[p[i]] = static_cast<std::uint8_t>(i);

            for (int w = 0; w < 8; ++w) {
                Perm np = p;
                const int pos = inv[w];                      // O(1), was O(8) scan
                for (int i = pos; i > 0; --i) np[i] = np[i - 1];
                np[0] = static_cast<std::uint8_t>(w);
                tables.next_state[state][w] = encode_fast(np); // O(n), was O(n²)
            }
        }
        init = true;
    }
    return &tables;
}

inline const LruTables* kLru = build_lru_tables();

const std::uint16_t kInitialLruState = []() {
    Perm id;
    for (int i = 0; i < 8; ++i) id[i] = static_cast<std::uint8_t>(i);
    return encode_fast(id);
}();

// ============================================================================
// Level template – SoA layout, per‑set LRU state stored as single uint16_t
// ============================================================================
template <int SETS, int WAYS>
struct Level {
    static constexpr int NUM_SETS = SETS;
    static constexpr int NUM_WAYS = WAYS;
    static constexpr int INDEX_BITS = log2_of(SETS);
    static constexpr std::uint64_t INDEX_MASK = SETS - 1;

    struct Meta {
        std::uint8_t valid;
        std::uint8_t dirty;
        std::uint16_t lru;
    };

    alignas(64) std::array<std::uint64_t, SETS * WAYS> tag{};
    std::array<Meta, SETS> meta{};

    void init() {
        tag.fill(0);
        for (int i = 0; i < SETS; ++i) {
            meta[i].valid = 0;
            meta[i].dirty = 0;
            meta[i].lru = kInitialLruState;
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
        __m256i a   = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&tag[base]));
        __m256i b   = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&tag[base + 4]));
        unsigned m  = unsigned(_mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(a, key))))
                    | (unsigned(_mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(b, key)))) << 4);
#elif defined(__SSE4_1__)
        __m128i key = _mm_set1_epi64x(t);
        __m128i a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&tag[base + 0]));
        __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&tag[base + 2]));
        __m128i c = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&tag[base + 4]));
        __m128i d = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&tag[base + 6]));
        unsigned m = unsigned(_mm_movemask_pd(_mm_castsi128_pd(_mm_cmpeq_epi64(a, key))))
                   | (unsigned(_mm_movemask_pd(_mm_castsi128_pd(_mm_cmpeq_epi64(b, key)))) << 2)
                   | (unsigned(_mm_movemask_pd(_mm_castsi128_pd(_mm_cmpeq_epi64(c, key)))) << 4)
                   | (unsigned(_mm_movemask_pd(_mm_castsi128_pd(_mm_cmpeq_epi64(d, key)))) << 6);
#else
        unsigned m = 0;
        for (int w = 0; w < WAYS; ++w) {
            m |= unsigned(tag[base + w] == t) << w;
        }
#endif
        m &= meta[si].valid;
        return m ? __builtin_ctz(m) : -1;
    }

    void touch_mru(int si, int way) {
        meta[si].lru = kLru->next_state[meta[si].lru][way];
    }

    int victim_way(int si) const {
        unsigned invalid = static_cast<unsigned>(static_cast<std::uint8_t>(~meta[si].valid));
        return invalid ? __builtin_ctz(invalid) : kLru->victim[meta[si].lru];
    }

    void set_line(int si, int way, bool v, bool d, std::uint64_t t) {
        const std::size_t base = static_cast<std::size_t>(si) * WAYS;
        if (v) meta[si].valid |= (1 << way); else meta[si].valid &= ~(1 << way);
        if (d) meta[si].dirty |= (1 << way); else meta[si].dirty &= ~(1 << way);
        tag[base + way] = t;
        meta[si].lru = kLru->next_state[meta[si].lru][way];
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

            const csot::MemAccess& a = acc[i];
            const bool wr = (a.is_write != 0);
            st.writes += wr;
            st.reads  += !wr;

            const std::uint64_t b = a.address >> 6;
            const int s1 = L1::set_of(b);
            const std::uint64_t t1 = L1::tag_of(b);
            const int s2 = L2::set_of(b);
            const std::uint64_t t2 = L2::tag_of(b);

            // L1 lookup
            int w1 = l1_.find_way(s1, t1);
            bool l1_hit = (w1 >= 0);
            st.l1_hits += l1_hit;
            st.l1_misses += !l1_hit;

            if (l1_hit) {
                l1_.touch_mru(s1, w1);
                if (wr) l1_.meta[s1].dirty |= (1 << w1);
                continue;
            }

            // L2 lookup
            int w2 = l2_.find_way(s2, t2);
            bool l2_hit = (w2 >= 0);
            st.l2_hits += l2_hit;
            st.l2_misses += !l2_hit;

            if (l2_hit) {
                l2_.touch_mru(s2, w2);
            } else {
                int victim = l2_.victim_way(s2);
                st.dirty_writebacks += ((l2_.meta[s2].valid & l2_.meta[s2].dirty) >> victim) & 1;
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
                    st.dirty_writebacks += ((l2_.meta[s2v].valid & l2_.meta[s2v].dirty) >> vv) & 1;
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