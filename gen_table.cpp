#include <iostream>
#include <fstream>
#include <cstdint>

int c_popcount(std::uint32_t x) {
    int c = 0;
    while (x) { c += x & 1; x >>= 1; }
    return c;
}

int c_ctz(std::uint32_t x) {
    if (!x) return 32;
    int c = 0;
    while ((x & 1) == 0) { x >>= 1; ++c; }
    return c;
}

std::uint16_t encode_fast(std::uint64_t v) {
    std::uint32_t seen = 0;
    std::uint32_t code = 0;
    std::uint32_t p;
    p = v & 0xFF; seen |= (1u << p); code += p * 5040; v >>= 8;
    p = v & 0xFF; code += (p - c_popcount(seen & ((1u << p) - 1u))) * 720; seen |= (1u << p); v >>= 8;
    p = v & 0xFF; code += (p - c_popcount(seen & ((1u << p) - 1u))) * 120; seen |= (1u << p); v >>= 8;
    p = v & 0xFF; code += (p - c_popcount(seen & ((1u << p) - 1u))) * 24; seen |= (1u << p); v >>= 8;
    p = v & 0xFF; code += (p - c_popcount(seen & ((1u << p) - 1u))) * 6; seen |= (1u << p); v >>= 8;
    p = v & 0xFF; code += (p - c_popcount(seen & ((1u << p) - 1u))) * 2; seen |= (1u << p); v >>= 8;
    p = v & 0xFF; code += (p - c_popcount(seen & ((1u << p) - 1u)));
    return static_cast<std::uint16_t>(code);
}

void decode_fast(std::uint32_t code, std::uint8_t p[8]) {
    std::uint32_t avail = 0xFF; std::uint32_t idx; std::uint32_t tmp;
    idx = code / 5040; code %= 5040; tmp = avail; for (std::uint32_t k = 0; k < idx; ++k) tmp &= tmp - 1; p[0] = static_cast<std::uint8_t>(c_ctz(tmp)); avail &= ~(1u << p[0]);
    idx = code / 720; code %= 720; tmp = avail; for (std::uint32_t k = 0; k < idx; ++k) tmp &= tmp - 1; p[1] = static_cast<std::uint8_t>(c_ctz(tmp)); avail &= ~(1u << p[1]);
    idx = code / 120; code %= 120; tmp = avail; for (std::uint32_t k = 0; k < idx; ++k) tmp &= tmp - 1; p[2] = static_cast<std::uint8_t>(c_ctz(tmp)); avail &= ~(1u << p[2]);
    idx = code / 24; code %= 24; tmp = avail; for (std::uint32_t k = 0; k < idx; ++k) tmp &= tmp - 1; p[3] = static_cast<std::uint8_t>(c_ctz(tmp)); avail &= ~(1u << p[3]);
    idx = code / 6; code %= 6; tmp = avail; for (std::uint32_t k = 0; k < idx; ++k) tmp &= tmp - 1; p[4] = static_cast<std::uint8_t>(c_ctz(tmp)); avail &= ~(1u << p[4]);
    idx = code / 2; code %= 2; tmp = avail; for (std::uint32_t k = 0; k < idx; ++k) tmp &= tmp - 1; p[5] = static_cast<std::uint8_t>(c_ctz(tmp)); avail &= ~(1u << p[5]);
    idx = code; tmp = avail; for (std::uint32_t k = 0; k < idx; ++k) tmp &= tmp - 1; p[6] = static_cast<std::uint8_t>(c_ctz(tmp)); avail &= ~(1u << p[6]);
    p[7] = static_cast<std::uint8_t>(c_ctz(avail));
}

int main() {
    std::ofstream out("hardcoded_table.hpp");
    out << "    constexpr int HARDCODED_START = 10080;\n";
    out << "    constexpr int HARDCODED_COUNT = 10080;\n";
    out << "    const std::uint16_t HARDCODED_NEXT[HARDCODED_COUNT][8] = {\n";
    
    for (int state = 10080; state < 20160; ++state) {
        std::uint8_t perm[8];
        decode_fast(state, perm);
        std::uint64_t v = 0;
        for (int i = 0; i < 8; ++i) v |= static_cast<std::uint64_t>(perm[i]) << (i * 8);
        std::uint8_t inv[8];
        for (int i = 0; i < 8; ++i) inv[perm[i]] = i;

        out << "        {";
        for (int w = 0; w < 8; ++w) {
            int pos = inv[w];
            std::uint64_t lower_mask = (1ULL << (pos * 8)) - 1;
            std::uint64_t upper_mask = (pos == 7) ? 0 : (~0ULL << ((pos + 1) * 8));
            std::uint64_t new_v = static_cast<std::uint64_t>(w) | ((v & lower_mask) << 8) | (v & upper_mask);
            out << encode_fast(new_v) << (w == 7 ? "" : ",");
        }
        out << "},\n";
    }
    out << "    };\n";
    
    out << "    const std::uint8_t HARDCODED_VICTIM[HARDCODED_COUNT] = {\n";
    out << "        ";
    for (int state = 10080; state < 20160; ++state) {
        std::uint8_t perm[8];
        decode_fast(state, perm);
        out << static_cast<int>(perm[7]) << ",";
        if ((state + 1) % 64 == 0) out << "\n        ";
    }
    out << "\n    };\n";
    return 0;
}
