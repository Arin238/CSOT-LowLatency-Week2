#include "cache_sim.hpp"

#include <array>
#include <cstdint>
#include <cstddef>

namespace csot {

// A simple Cache Line struct.
// Explicitly avoiding bit-packing or clever tricks for maximum readability.
struct CacheLine {
    bool valid = false;
    bool dirty = false;
    std::uint64_t tag = 0;
};

// A generalized N-way Set Associative Cache Level.
// Stores the ways in an Array-of-Structs format where ways[0] is strictly MRU
// and ways[Ways - 1] is strictly LRU.
template <std::size_t Sets, std::size_t Ways>
struct CacheLevel {
    struct Set {
        std::array<CacheLine, Ways> ways;
    };
    
    std::array<Set, Sets> sets;

    // Returns the index of the matching way if the tag is resident, else -1.
    int find(std::size_t set_idx, std::uint64_t tag) const {
        const auto& set = sets[set_idx];
        for (std::size_t i = 0; i < Ways; ++i) {
            if (set.ways[i].valid && set.ways[i].tag == tag) {
                return static_cast<int>(i);
            }
        }
        return -1;
    }

    // Moves the given way to the most-recently-used position (index 0).
    // Shifts the remaining elements down to maintain true LRU order.
    void touch(std::size_t set_idx, int way) {
        if (way == 0) return; // Already MRU
        
        auto& set = sets[set_idx];
        CacheLine mru_line = set.ways[way];
        
        // Shift elements down
        for (int i = way; i > 0; --i) {
            set.ways[i] = set.ways[i - 1];
        }
        
        set.ways[0] = mru_line;
    }

    // Picks an invalid way if any exist. 
    // If all are valid, returns the LRU way (the last element).
    int victim_way(std::size_t set_idx) const {
        const auto& set = sets[set_idx];
        for (std::size_t i = 0; i < Ways; ++i) {
            if (!set.ways[i].valid) {
                return static_cast<int>(i);
            }
        }
        return static_cast<int>(Ways - 1);
    }
};

// The Baseline Cache Simulator.
// Instantiates the hierarchy and orchestrates the inclusion/eviction logic.
class BaselineCacheSim : public CacheSim {
private:
    // Fixed arrays as class members to prevent heap allocation inside run().
    // The zero-initialization of these structs automatically sets `valid = false`.
    CacheLevel<64, 8> l1_;
    CacheLevel<512, 8> l2_;

public:
    void on_init() override {
        // Nothing needed here. The standard library default-initializes the 
        // std::array structures during object construction, perfectly satisfying 
        // the "zero allocation on hot path" and "warm up pages" constraints.
    }

    CacheStats run(const MemAccess* accesses, std::size_t n) override {
        CacheStats st{};

        for (std::size_t i = 0; i < n; ++i) {
            // 1. Decode memory access geometry
            const std::uint64_t b = accesses[i].address >> 6;
            const bool is_write = accesses[i].is_write != 0;
            
            if (is_write) {
                st.writes++;
            } else {
                st.reads++;
            }

            // L1 Geometry parameters (Masking instead of modulo)
            const std::size_t s1 = b & 63;
            const std::uint64_t t1 = b >> 6;

            // 2. Probe L1 Cache
            int w1 = l1_.find(s1, t1);
            if (w1 >= 0) {
                // ---- L1 hit ----
                st.l1_hits++;
                l1_.touch(s1, w1);
                if (is_write) {
                    l1_.sets[s1].ways[0].dirty = true;
                }
                continue;
            }
            
            // ---- L1 miss ----
            st.l1_misses++;

            // L2 Geometry parameters
            const std::size_t s2 = b & 511;
            const std::uint64_t t2 = b >> 9;
            
            // 3. Probe L2 Cache
            int w2 = l2_.find(s2, t2);
            if (w2 >= 0) {
                // ---- L2 hit ----
                st.l2_hits++;
                l2_.touch(s2, w2);
            } else {
                // ---- L2 miss: demand-install CLEAN ----
                st.l2_misses++;
                int v = l2_.victim_way(s2);
                
                // Track dirty writebacks leaving L2 for main memory
                if (l2_.sets[s2].ways[v].valid && l2_.sets[s2].ways[v].dirty) {
                    st.dirty_writebacks++;
                }
                
                l2_.sets[s2].ways[v] = {/*valid=*/true, /*dirty=*/false, t2};
                l2_.touch(s2, v);
            }

            // 4. Fill into L1 (write-allocate)
            int v1 = l1_.victim_way(s1);
            
            // Check if we need to evict a dirty L1 line
            if (l1_.sets[s1].ways[v1].valid && l1_.sets[s1].ways[v1].dirty) {
                // Reconstruct victim block address
                const std::uint64_t bv = (l1_.sets[s1].ways[v1].tag << 6) | s1;
                
                // Determine victim L2 mapping
                const std::size_t s2v = bv & 511;
                const std::uint64_t t2v = bv >> 9;
                
                int wv = l2_.find(s2v, t2v);
                if (wv >= 0) {
                    // Dirty L1 line found in L2. Mark dirty, DO NOT touch LRU.
                    l2_.sets[s2v].ways[wv].dirty = true;
                } else {
                    // Dirty L1 line not in L2. Install dirty into L2.
                    int vv = l2_.victim_way(s2v);
                    
                    if (l2_.sets[s2v].ways[vv].valid && l2_.sets[s2v].ways[vv].dirty) {
                        st.dirty_writebacks++;
                    }
                    
                    l2_.sets[s2v].ways[vv] = {/*valid=*/true, /*dirty=*/true, t2v};
                    l2_.touch(s2v, vv);
                }
            }
            
            // Overwrite L1 victim with the new data
            l1_.sets[s1].ways[v1] = {/*valid=*/true, /*dirty=*/is_write, t1};
            l1_.touch(s1, v1);
        }

        return st;
    }
};

} // namespace csot

// Factory entry point for the judge
extern "C" csot::CacheSim* create_cache_sim() {
    return new csot::BaselineCacheSim();
}
