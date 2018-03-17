#include <simd/bit_vector.h>

#include <iostream>

namespace simd {

std::ostream& operator<<(std::ostream& os, simd::f32x8 v) {
    alignas(32) float out[8];
    v.store(as_aligned_view<32>(out));
    os << "[";
    for (size_t i = 0; i < 7; ++i) {
        os << out[i] << ", ";
    }
    os << out[7] << "]";
    return os;
}

}  // namespace simd
