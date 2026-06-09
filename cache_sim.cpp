// ============================================================================
//  cache_sim.cpp — final portable version (no consteval, runtime tables)
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
// LRU state machine – Lehmer code (0…40319) for 8‑way permutations.
// Tables are built once at runtime; hot path uses O(1) array lookups.
// ============================================================================
using Perm = std::array<std::uint8_t, 8>;

constexpr std::uint16_t encode_perm(const Perm& p) {
    std::uint16_t code = 0;
    const int fact[8] = {1, 1, 2, 6, 24, 120, 720, 5040};
    for (int i = 0; i < 7; ++i) {
        int count = 0;
        for (int j = i + 1; j < 8; ++j)
            if (p[j] < p[i]) ++count;
        code += static_cast<std::uint16_t>(count * fact[7 - i]);
    }
    return code;
}

constexpr Perm decode_perm(std::uint16_t code) {
    Perm p{};
    const int fact[8] = {1, 1, 2, 6, 24, 120, 720, 5040};
    std::uint8_t available[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    for (int i = 0; i < 8; ++i) {
        int idx = code / fact[7 - i];
        code %= fact[7 - i];
        p[i] = available[idx];
        for (int j = idx; j < 7 - i; ++j)
            available[j] = available[j + 1];
    }
    return p;
}

constexpr Perm next_perm(const Perm& p, int way) {
    Perm np = p;
    int pos = -1;
    for (int i = 0; i < 8; ++i)
        if (np[i] == way) { pos = i; break; }
    if (pos == -1) pos = 7;   // shouldn't happen
    for (int i = pos; i > 0; --i)
        np[i] = np[i - 1];
    np[0] = static_cast<std::uint8_t>(way);
    return np;
}

struct LruTables {
    std::array<std::uint8_t, 40320> victim{};
    std::array<std::array<std::uint16_t, 8>, 40320> next_state{};
};

constexpr LruTables build_lru_tables() {
    LruTables tables{};
    for (std::uint16_t state = 0; state < 40320; ++state) {
        Perm p = decode_perm(state);
        tables.victim[state] = p[7];
        for (int w = 0; w < 8; ++w)
            tables.next_state[state][w] = encode_perm(next_perm(p, w));
    }
    return tables;
}

constexpr LruTables kLru = build_lru_tables();

constexpr std::uint16_t get_initial_state() {
    Perm id{};
    for (int i = 0; i < 8; ++i) id[i] = static_cast<std::uint8_t>(i);
    return encode_perm(id);
}
constexpr std::uint16_t kInitialLruState = get_initial_state();

// ============================================================================
// Level template – SoA layout, per‑set LRU state stored as single uint16_t
// ============================================================================
template <int SETS, int WAYS>
struct Level {
    static constexpr int NUM_SETS = SETS;
    static constexpr int NUM_WAYS = WAYS;
    static constexpr int INDEX_BITS = log2_of(SETS);
    static constexpr std::uint64_t INDEX_MASK = SETS - 1;

    alignas(64) std::array<std::uint64_t, SETS * WAYS> tag{};
    alignas(64) std::array<std::uint8_t,  SETS * WAYS> valid{};
    alignas(64) std::array<std::uint8_t,  SETS * WAYS> dirty{};
    std::array<std::uint16_t, SETS> lru{};   // one LRU state per set

    void init() {
        tag.fill(0);
        valid.fill(0);
        dirty.fill(0);
        lru.fill(kInitialLruState);
    }

    static constexpr int set_of(std::uint64_t blk) {
        return static_cast<int>(blk & INDEX_MASK);
    }
    static constexpr std::uint64_t tag_of(std::uint64_t blk) {
        return blk >> INDEX_BITS;
    }

    int find_way(int si, std::uint64_t t) const {
        const std::size_t base = static_cast<std::size_t>(si) * WAYS;
        for (int w = 0; w < WAYS; ++w)
            if (valid[base + w] && tag[base + w] == t) return w;
        return -1;
    }

    void touch_mru(int si, int way) {
        lru[si] = kLru.next_state[lru[si]][way];
    }

    int victim_way(int si) const {
        const std::size_t base = static_cast<std::size_t>(si) * WAYS;
        for (int w = 0; w < WAYS; ++w)
            if (!valid[base + w]) return w;
        return kLru.victim[lru[si]];
    }

    void set_line(int si, int way, bool v, bool d, std::uint64_t t) {
        const std::size_t base = static_cast<std::size_t>(si) * WAYS;
        valid[base + way] = v ? 1 : 0;
        dirty[base + way] = d ? 1 : 0;
        tag[base + way] = t;
        lru[si] = kLru.next_state[lru[si]][way];
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
                l1_.dirty[static_cast<std::size_t>(s1) * L1::NUM_WAYS + w1] |= wr;
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
                const std::size_t idx = static_cast<std::size_t>(s2) * L2::NUM_WAYS + victim;
                st.dirty_writebacks += (l2_.valid[idx] & l2_.dirty[idx]);
                l2_.set_line(s2, victim, true, false, t2);
            }

            // Fill L1, possibly write back dirty L1 victim
            int v1 = l1_.victim_way(s1);
            const std::size_t v1_idx = static_cast<std::size_t>(s1) * L1::NUM_WAYS + v1;
            if (l1_.valid[v1_idx] && l1_.dirty[v1_idx]) {
                std::uint64_t victim_tag = l1_.tag[v1_idx];
                std::uint64_t bv = (victim_tag << L1::INDEX_BITS) | static_cast<std::uint64_t>(s1);
                int s2v = L2::set_of(bv);
                std::uint64_t t2v = L2::tag_of(bv);

                int wv = l2_.find_way(s2v, t2v);
                if (wv >= 0) {
                    l2_.dirty[static_cast<std::size_t>(s2v) * L2::NUM_WAYS + wv] = 1;
                } else {
                    int vv = l2_.victim_way(s2v);
                    const std::size_t vv_idx = static_cast<std::size_t>(s2v) * L2::NUM_WAYS + vv;
                    st.dirty_writebacks += (l2_.valid[vv_idx] & l2_.dirty[vv_idx]);
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