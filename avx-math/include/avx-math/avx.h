#pragma once

#include <immintrin.h>

#include <cstdint>

namespace avx {

template <size_t width>
struct f32;

template <>
struct f32<4> {
    __m128 data;

    static constexpr size_t width = 32;
    static constexpr size_t size = 4;

    static f32<4> from(float f3, float f2, float f1, float f0) {
        return {_mm_set_ps(f0, f1, f2, f3)};
    }

    friend f32<4> operator*(f32<4> lhs, f32<4> rhs) {
        return {_mm_mul_ps(lhs.data, rhs.data)};
    }

    f32<4>& operator*=(f32<4> rhs) {
        data = _mm_mul_ps(data, rhs.data);
        return *this;
    }

    bool operator==(f32<4> rhs) const {
        // compare not equal, unordered, non-signaling
        __m128 result_neq = _mm_cmp_ps(data, rhs.data, _CMP_NEQ_UQ);
        return _mm_test_all_zeros(result_neq, result_neq);
    }
};

template <>
struct f32<8> {
    __m256 data;

    static constexpr size_t width = 32;
    static constexpr size_t size = 8;

    static f32<8>
    from(float f7,
         float f6,
         float f5,
         float f4,
         float f3,
         float f2,
         float f1,
         float f0) {
        return {_mm256_set_ps(f0, f1, f2, f3, f4, f5, f6, f7)};
    }

    friend f32<8> operator*(f32<8> lhs, f32<8> rhs) {
        return {_mm256_mul_ps(lhs.data, rhs.data)};
    }
    f32<8>& operator*=(f32<8> rhs) {
        data = _mm256_mul_ps(data, rhs.data);
        return *this;
    }

    bool operator==(f32<8> rhs) const {
        // compare not equal, unordered, non-signaling
        __m256 result_neq    = _mm256_cmp_ps(data, rhs.data, _CMP_NEQ_UQ);
        __m128 result_neq_lo = _mm256_castps256_ps128(result_neq);
        __m128 result_neq_hi = _mm256_extractf128_ps(result_neq, 1);
        return _mm_test_all_zeros(result_neq_hi, result_neq_lo);
    }

    f32<4> low_bits() const { return {_mm256_castps256_ps128(data)}; }
    f32<4> high_bits() const { return {_mm256_extractf128_ps(data, 1)}; }
};

using f32x8 = f32<8>;
using f32x4 = f32<4>;

///// load /////

template <typename T>
inline T load_aligned(__attribute((aligned(32))) float* ptr);

template <>
inline f32x8 load_aligned<f32x8>(__attribute((aligned(32))) float* ptr) {
    return {_mm256_load_ps(ptr)};
}

template <>
inline f32x4 load_aligned<f32x4>(__attribute((aligned(16))) float* ptr) {
    return {_mm_load_ps(ptr)};
}

template <typename T>
inline T load_unaligned(float* ptr);

template <>
inline f32x8 load_unaligned<f32x8>(float* ptr) {
    return {_mm256_loadu_ps(ptr)};
}

template <>
inline f32x4 load_unaligned<f32x4>(float* ptr) {
    return {_mm_loadu_ps(ptr)};
}

///// store /////

inline void store_aligned(__attribute((aligned(16))) float* ptr, f32x4 v) {
    _mm_store_ps(ptr, v.data);
}

inline void store_unaligned(float* ptr, f32x4 v) {
    _mm_storeu_ps(ptr, v.data);
}

inline void store_aligned(__attribute((aligned(32))) float* ptr, f32x8 v) {
    _mm256_store_ps(ptr, v.data);
}

inline void store_unaligned(float* ptr, f32x8 v) {
    _mm256_storeu_ps(ptr, v.data);
}

///// hadd /////

inline f32x8 hadd(f32x8 v1, f32x8 v2) {
    return {_mm256_hadd_ps(v1.data, v2.data)};
}

inline f32x4 hadd(f32x4 v1, f32x4 v2) {
    return {_mm_hadd_ps(v1.data, v2.data)};
}

}  // namespace avx

