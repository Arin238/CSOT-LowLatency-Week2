#include "cache_sim.hpp"
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

namespace geo {
    consteval uint64_t mask(uint64_t s) { return s - 1; }
    inline constexpr int LINE_BITS   = 6;
    inline constexpr int L1_SETS     = 64,  L1_IDX_BITS = 6;
    inline constexpr int L2_SETS     = 512, L2_IDX_BITS = 9;
    inline constexpr uint64_t L1_MASK = mask(L1_SETS);
    inline constexpr uint64_t L2_MASK = mask(L2_SETS);
}

consteval uint64_t make_identity_lru() {
    uint64_t v = 0;
    for (int w = 0; w < 8; ++w) v |= uint64_t(w) << (w * 8);
    return v;
}
inline constexpr uint64_t IDENTITY_LRU = make_identity_lru();

struct Level {
    static constexpr int WAYS = 8;
    std::vector<uint64_t> tag;
    std::vector<uint8_t>  valid;
    std::vector<uint8_t>  dirty;
    std::vector<uint64_t> lru;

    void init(int sets) {
        tag.assign(sets * WAYS, 0);
        valid.assign(sets * WAYS, 0);
        dirty.assign(sets * WAYS, 0);
        lru.assign(sets, IDENTITY_LRU);
    }
};

class Sim final : public csot::CacheSim {
public:
    void on_init() override {
        l1.init(geo::L1_SETS);
        l2.init(geo::L2_SETS);
    }

    csot::CacheStats run(const csot::MemAccess* acc, std::size_t n) override {
        csot::CacheStats st{};

        for (std::size_t i = 0; i < n; ++i) {
            const uint64_t addr = acc[i].address;
            const uint8_t wr    = acc[i].is_write;

            st.writes += wr;
            st.reads  += (wr ^ 1);

            const uint64_t b  = addr >> geo::LINE_BITS;
            const int      s1 = int(b & geo::L1_MASK);
            const uint64_t t1 = b >> geo::L1_IDX_BITS;
            const int      s2 = int(b & geo::L2_MASK);
            const uint64_t t2 = b >> geo::L2_IDX_BITS;
            const int      b1 = s1 * Level::WAYS;

            int w1 = find_way(l1, b1, t1);
            bool l1_hit = (w1 >= 0);
            st.l1_hits   += l1_hit;
            st.l1_misses += !l1_hit;

            if (l1_hit) {
                touch_mru(l1, s1, w1);
                l1.dirty[b1 + w1] |= wr;
                continue;
            }

            const int b2 = s2 * Level::WAYS;
            int w2 = find_way(l2, b2, t2);
            bool l2_hit = (w2 >= 0);
            st.l2_hits   += l2_hit;
            st.l2_misses += !l2_hit;

            if (l2_hit) {
                touch_mru(l2, s2, w2);
            } else {
                int victim = victim_way(l2, b2, s2);
                st.dirty_writebacks += (l2.valid[b2 + victim] & l2.dirty[b2 + victim]);
                set_line(l2, b2, s2, victim, 1, 0, t2);
            }

            int v1 = victim_way(l1, b1, s1);
            if (l1.valid[b1 + v1] && l1.dirty[b1 + v1]) {
                uint64_t bv  = (l1.tag[b1 + v1] << geo::L1_IDX_BITS) | uint64_t(s1);
                int      s2v = int(bv & geo::L2_MASK);
                uint64_t t2v = bv >> geo::L2_IDX_BITS;
                int      b2v = s2v * Level::WAYS;

                int wv = find_way(l2, b2v, t2v);
                if (wv >= 0) {
                    l2.dirty[b2v + wv] = 1;
                } else {
                    int vv = victim_way(l2, b2v, s2v);
                    st.dirty_writebacks += (l2.valid[b2v + vv] & l2.dirty[b2v + vv]);
                    set_line(l2, b2v, s2v, vv, 1, 1, t2v);
                }
            }

            set_line(l1, b1, s1, v1, 1, wr, t1);
        }
        return st;
    }

private:
    Level l1, l2;

    __attribute__((always_inline))
    static inline int find_way(const Level& L, int base, uint64_t tag) {
        int m = -1;
        for (int w = 0; w < Level::WAYS; ++w)
            m = (L.valid[base + w] && L.tag[base + w] == tag) ? w : m;
        return m;
    }

    __attribute__((always_inline))
    static inline void touch_mru(Level& L, int set_idx, int way) {
        uint64_t packed = L.lru[set_idx];
        uint8_t a[8];
        std::memcpy(a, &packed, 8);
        uint8_t t = uint8_t(way);
        bool found = (a[0] == t);
        uint8_t prev = a[0];
        a[0] = t;
        for (int i = 1; i < 8; ++i) {
            uint8_t c = a[i];
            a[i] = found ? c : prev;
            found |= (c == t);
            prev = c;
        }
        std::memcpy(&packed, a, 8);
        L.lru[set_idx] = packed;
    }

    __attribute__((always_inline))
    static inline int victim_way(const Level& L, int base, int set_idx) {
        int inv = -1;
        for (int w = Level::WAYS - 1; w >= 0; --w)
            inv = (L.valid[base + w] == 0) ? w : inv;
        return (inv != -1) ? inv : int((L.lru[set_idx] >> 56) & 0xFF);
    }

    __attribute__((always_inline))
    static inline void set_line(Level& L, int base, int set_idx, int way,
                                uint8_t valid, uint8_t dirty, uint64_t tag) {
        L.valid[base + way] = valid;
        L.dirty[base + way] = dirty;
        L.tag[base + way]   = tag;
        touch_mru(L, set_idx, way);
    }
};

} // namespace

extern "C" csot::CacheSim* create_cache_sim() {
    return new Sim();
}