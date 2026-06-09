#include <immintrin.h>
#include <cstdint>

__m256i test(std::uint64_t t) {
    __m256i key;
    asm("vmovq %1, %x0\n\tvpbroadcastq %x0, %0"
        : "=y"(key)
        : "r"(t));
    return key;
}
