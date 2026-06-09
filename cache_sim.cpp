// ============================================================================
//  cache_sim.cpp – final optimized version
// ============================================================================

#include "cache_sim.hpp"
#include <cstddef>
#include <cstdint>
#include <array>
#include <cstring>

namespace {

constexpr int log2_of(int v) { int r = 0; while ((1 << r) < v) ++r; return r; }

// ============================================================================
//  Level – Structure of Arrays (SoA), all arrays cache‑line aligned.
//  Each set has 8 ways.  LRU is stored as an array of way indices:
//    lru[0] = most recent, lru[WAYS-1] = least recent.
//  This layout minimises memory traffic and allows memmove for LRU updates.
// ============================================================================
template <int SETS, int WAYS>
struct Level {
    static constexpr int           NUM_SETS   = SETS;
    static constexpr int           NUM_WAYS   = WAYS;
    static constexpr int           INDEX_BITS = log2_of(SETS);
    static constexpr std::uint64_t INDEX_MASK = SETS - 1;

    alignas(64) std::array<std::uint64_t, SETS * WAYS> tag{};
    alignas(64) std::array<std::uint8_t,  SETS * WAYS> valid{};
    alignas(64) std::array<std::uint8_t,  SETS * WAYS> dirty{};
    alignas(64) std::array<std::uint8_t,  SETS * WAYS> lru{};

    void init() {
        tag.fill(0); valid.fill(0); dirty.fill(0);
        for (int si = 0; si < SETS; ++si) {
            const std::size_t base = si * WAYS;
            for (int w = 0; w < WAYS; ++w)
                lru[base + w] = static_cast<std::uint8_t>(w);
        }
    }

    static constexpr int set_of(std::uint64_t blk) {
        return static_cast<int>(blk & INDEX_MASK);
    }
    static constexpr std::uint64_t tag_of(std::uint64_t blk) {
        return blk >> INDEX_BITS;
    }

    // Linear search – fully unrolled for WAYS=8, only 8 comparisons.
    int find_way(int si, std::uint64_t t) const {
        const std::size_t base = si * WAYS;
        for (int w = 0; w < WAYS; ++w)
            if (valid[base + w] && tag[base + w] == t) return w;
        return -1;
    }

    // Promote ‘way’ to MRU.  The LRU array stores way indices.
    // memmove of at most 7 bytes compiles to 4-5 mov instructions.
    void touch_mru(int si, int way) {
        const std::size_t  base = si * WAYS;
        const std::uint8_t w8   = static_cast<std::uint8_t>(way);
        // Find current position of this way in the LRU list.
        int pos = 0;
        for (; pos < WAYS; ++pos)
            if (lru[base + pos] == w8) break;
        if (pos == 0) return;          // already MRU
        // Shift entries [0, pos) right by one, then place w8 at front.
        std::memmove(&lru[base + 1], &lru[base], pos);
        lru[base] = w8;
    }

    // Victim selection: first invalid way, else the last (LRU) way.
    int victim_way(int si) const {
        const std::size_t base = si * WAYS;
        for (int w = 0; w < WAYS; ++w)
            if (!valid[base + w]) return w;
        return static_cast<int>(lru[base + WAYS - 1]);
    }

    // Install a line and mark it MRU.
    void set_line(int si, int way, bool v, bool d, std::uint64_t t) {
        const std::size_t base = si * WAYS;
        valid[base + way] = v;
        dirty[base + way] = d;
        tag[base + way]   = t;
        touch_mru(si, way);   // promotes the way to MRU
    }
};

using L1 = Level<64,  8>;   // 32 KiB L1
using L2 = Level<512, 8>;   // 256 KiB L2

// ============================================================================
class BaselineCacheSim final : public csot::CacheSim {
public:
    void on_init() override { l1_.init(); l2_.init(); }

    csot::CacheStats run(const csot::MemAccess* acc, std::size_t n) override {
        csot::CacheStats st{};

        for (std::size_t i = 0; i < n; ++i) {
            // Optional prefetch – only every 8 iterations to reduce overhead.
            // Remove this line entirely if you don't need it.
            if ((i & 7) == 0)
                __builtin_prefetch(&acc[i + 16], 0, 0);

            const csot::MemAccess& a  = acc[i];
            const bool             wr = (a.is_write != 0);

            st.writes += wr;
            st.reads  += !wr;

            const std::uint64_t b  = a.address >> 6;
            const int           s1 = L1::set_of(b);
            const std::uint64_t t1 = L1::tag_of(b);
            const int           s2 = L2::set_of(b);
            const std::uint64_t t2 = L2::tag_of(b);

            // ---------- L1 lookup ----------
            const int w1 = l1_.find_way(s1, t1);
            if (w1 >= 0) {
                ++st.l1_hits;
                l1_.touch_mru(s1, w1);
                l1_.dirty[s1 * L1::NUM_WAYS + w1] |= wr;
                continue;
            }
            ++st.l1_misses;

            // ---------- L2 lookup (only on L1 miss) ----------
            const int w2 = l2_.find_way(s2, t2);
            if (w2 >= 0) {
                ++st.l2_hits;
                l2_.touch_mru(s2, w2);
                // No need to rewrite tag/valid – they are already correct.
                // Dirty bit is irrelevant for L2 in this design.
            } else {
                ++st.l2_misses;
                const int         vl2  = l2_.victim_way(s2);
                const std::size_t vidx = s2 * L2::NUM_WAYS + vl2;
                st.dirty_writebacks += l2_.valid[vidx] & l2_.dirty[vidx];
                l2_.set_line(s2, vl2, true, false, t2);
            }

            // ---------- Fill L1 (write‑allocate) ----------
            const int         vl1  = l1_.victim_way(s1);
            const std::size_t v1x  = s1 * L1::NUM_WAYS + vl1;

            // If the victim line in L1 is dirty, write it back to L2.
            if (l1_.valid[v1x] & l1_.dirty[v1x]) {
                const std::uint64_t vtag = l1_.tag[v1x];
                const std::uint64_t bv   = (vtag << L1::INDEX_BITS) | s1;
                const int           s2v  = L2::set_of(bv);
                const std::uint64_t t2v  = L2::tag_of(bv);

                const int wv = l2_.find_way(s2v, t2v);
                if (wv >= 0) {
                    // Already in L2 – just mark dirty.
                    l2_.dirty[s2v * L2::NUM_WAYS + wv] = 1;
                } else {
                    // Need to evict from L2.
                    const int         vv  = l2_.victim_way(s2v);
                    const std::size_t vvx = s2v * L2::NUM_WAYS + vv;
                    st.dirty_writebacks += l2_.valid[vvx] & l2_.dirty[vvx];
                    l2_.set_line(s2v, vv, true, true, t2v);
                }
            }

            l1_.set_line(s1, vl1, true, wr, t1);
        }
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