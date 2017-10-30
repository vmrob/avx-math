#pragma once

#include <immintrin.h>

#include <cstdint>

namespace math {

template <size_t width>
struct avx;

template <>
struct avx<256> {
    struct m256 {
        __m256 data;

        friend m256 operator*(m256 lhs, m256 rhs) {
            return {_mm256_mul_ps(lhs.data, rhs.data)};
        }
        m256& operator*=(m256 rhs) {
            data = _mm256_mul_ps(data, rhs.data);
            return *this;
        }
    };
    static m256 load_aligned(float* ptr) { return {_mm256_load_ps(ptr)}; }
};

}  // namespace math
