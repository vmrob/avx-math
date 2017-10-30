#pragma once

#include <immintrin.h>

#include <type_traits>
#include <cstdint>

namespace avx {

template <unsigned i0, unsigned i1, unsigned i2, unsigned i3>
struct control4 : std::integral_constant<
                          unsigned,
                          (i3 << 6 | i2 << 4 | i1 << 2 | i0 << 0)> {
    static_assert(i0 < 4, "out of bounds");
    static_assert(i1 < 4, "out of bounds");
    static_assert(i2 < 4, "out of bounds");
    static_assert(i3 < 4, "out of bounds");
};

template <size_t width>
struct i32;
template <>
struct i32<4>;
template <>
struct i32<8>;

template <size_t width>
struct f32;
template <>
struct f32<4>;
template <>
struct f32<8>;

template <>
struct i32<4> {
    __m128i data;

    static constexpr size_t width = 32;
    static constexpr size_t size  = 4;

    static i32<4> from(int32_t i3, int32_t i2, int32_t i1, int32_t i0) {
        return {_mm_set_epi32(i0, i1, i2, i3)};
    }

    explicit operator f32<4>() const;

    bool operator==(i32<4> rhs) const {
        __m128i result = _mm_xor_si128(data, rhs.data);
        return _mm_test_all_zeros(result, result);
    }
};

template <>
struct i32<8> {
    __m256i data;

    static constexpr size_t width = 32;
    static constexpr size_t size  = 8;

    static i32<8>
    from(int32_t i7,
         int32_t i6,
         int32_t i5,
         int32_t i4,
         int32_t i3,
         int32_t i2,
         int32_t i1,
         int32_t i0) {
        return {_mm256_set_epi32(i0, i1, i2, i3, i4, i5, i6, i7)};
    }

    explicit operator f32<8>() const;

    bool operator==(i32<8> rhs) const {
        __m256i result    = _mm256_xor_si256(data, rhs.data);
        __m128  result_lo = _mm256_castps256_ps128(result);
        __m128  result_hi = _mm256_extractf128_ps(result, 1);
        return _mm_test_all_zeros(result_hi, result_lo);
    }

    i32<4> low_bits() const { return {_mm256_castsi256_si128(data)}; }
    i32<4> high_bits() const { return {_mm256_extracti128_si256(data, 1)}; }
};

template <>
struct f32<4> {
    __m128 data;

    static constexpr size_t width = 32;
    static constexpr size_t size  = 4;

    static f32<4> from(float f3, float f2, float f1, float f0) {
        return {_mm_set_ps(f0, f1, f2, f3)};
    }

    explicit operator i32<4>() const;

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
    static constexpr size_t size  = 8;

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

    explicit operator i32<8>() const;

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

using i32x8 = i32<8>;
using i32x4 = i32<4>;
using f32x8 = f32<8>;
using f32x4 = f32<4>;

inline i32<4>::operator f32<4>() const {
    return {_mm_castsi128_ps(data)};
}

inline i32<8>::operator f32<8>() const {
    return {_mm256_castsi256_ps(data)};
}

inline f32<4>::operator i32<4>() const {
    return {_mm_castps_si128(data)};
}

inline f32<8>::operator i32<8>() const {
    return {_mm256_castps_si256(data)};
}

///// load aligned /////

template <typename T>
inline T load_aligned(__attribute((aligned(32))) int32_t* ptr);

template <>
inline i32x4 load_aligned<i32x4>(__attribute((aligned(16))) int32_t* ptr) {
    return {_mm_load_si128(reinterpret_cast<__m128i*>(ptr))};
}

template <>
inline i32x8 load_aligned<i32x8>(__attribute((aligned(32))) int32_t* ptr) {
    return {_mm256_load_si256(reinterpret_cast<__m256i*>(ptr))};
}

template <typename T>
inline T load_aligned(__attribute((aligned(32))) float* ptr);

template <>
inline f32x4 load_aligned<f32x4>(__attribute((aligned(16))) float* ptr) {
    return {_mm_load_ps(ptr)};
}

template <>
inline f32x8 load_aligned<f32x8>(__attribute((aligned(32))) float* ptr) {
    return {_mm256_load_ps(ptr)};
}

///// load unaligned /////

template <typename T>
inline T load_unaligned(int32_t* ptr);

template <>
inline i32x4 load_unaligned<i32x4>(int32_t* ptr) {
    return {_mm_loadu_si128(reinterpret_cast<__m128i*>(ptr))};
}

template <>
inline i32x8 load_unaligned<i32x8>(int32_t* ptr) {
    return {_mm256_loadu_si256(reinterpret_cast<__m256i*>(ptr))};
}

template <typename T>
inline T load_unaligned(float* ptr);

template <>
inline f32x4 load_unaligned<f32x4>(float* ptr) {
    return {_mm_loadu_ps(ptr)};
}

template <>
inline f32x8 load_unaligned<f32x8>(float* ptr) {
    return {_mm256_loadu_ps(ptr)};
}

///// store aligned /////

inline void store_aligned(__attribute((aligned(16))) int32_t* ptr, i32x4 v) {
    _mm_store_si128(reinterpret_cast<__m128i*>(ptr), v.data);
}

inline void store_aligned(__attribute((aligned(32))) int32_t* ptr, i32x8 v) {
    _mm256_store_si256(reinterpret_cast<__m256i*>(ptr), v.data);
}

inline void store_aligned(__attribute((aligned(16))) float* ptr, f32x4 v) {
    _mm_store_ps(ptr, v.data);
}

inline void store_aligned(__attribute((aligned(32))) float* ptr, f32x8 v) {
    _mm256_store_ps(ptr, v.data);
}

///// store unaligned /////

inline void store_unaligned(int32_t* ptr, i32x4 v) {
    _mm_storeu_si128(reinterpret_cast<__m128i*>(ptr), v.data);
}

inline void store_unaligned(int32_t* ptr, i32x8 v) {
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), v.data);
}

inline void store_unaligned(float* ptr, f32x4 v) {
    _mm_storeu_ps(ptr, v.data);
}

inline void store_unaligned(float* ptr, f32x8 v) {
    _mm256_storeu_ps(ptr, v.data);
}

///// hadd /////

inline i32x8 hadd(i32x8 v1, i32x8 v2) {
    return {_mm256_hadd_epi32(v1.data, v2.data)};
}

inline i32x4 hadd(i32x4 v1, i32x4 v2) {
    return {_mm_hadd_epi32(v1.data, v2.data)};
}

inline f32x8 hadd(f32x8 v1, f32x8 v2) {
    return {_mm256_hadd_ps(v1.data, v2.data)};
}

inline f32x4 hadd(f32x4 v1, f32x4 v2) {
    return {_mm_hadd_ps(v1.data, v2.data)};
}

///// permute /////

template <unsigned... flags>
inline i32x8 permute4x64(i32x8 v, control4<flags...>) {
    return {_mm256_permute4x64_epi64(v.data, control4<flags...>::value)};
}

template <unsigned... flags>
inline f32x8 permute4x64(f32x8 v, control4<flags...>) {
    return {_mm256_permute4x64_epi64(static_cast<i32x8>(v).data, control4<flags...>::value)};
}

}  // namespace avx

