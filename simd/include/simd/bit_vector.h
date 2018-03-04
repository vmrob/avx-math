#pragma once

#include <simd/view.h>

#include <immintrin.h>

#include <cstdint>
#include <type_traits>

namespace simd {

template <unsigned i0, unsigned i1, unsigned i2, unsigned i3>
struct control4 : std::integral_constant<
                          unsigned,
                          (i3 << 6 | i2 << 4 | i1 << 2 | i0 << 0)> {
    static_assert(i0 < 4);
    static_assert(i1 < 4);
    static_assert(i2 < 4);
    static_assert(i3 < 4);
};

template <typename Rep, size_t Bits>
struct bit_vector;

template <>
struct bit_vector<float, 128>;
template <>
struct bit_vector<float, 256>;
template <>
struct bit_vector<int32_t, 128>;
template <>
struct bit_vector<int32_t, 256>;

template <>
struct bit_vector<int32_t, 128> {
    __m128i data;

    static constexpr size_t width_bytes = 32;
    static constexpr size_t size        = 4;

    static bit_vector<int32_t, 128>
    from(int32_t i3, int32_t i2, int32_t i1, int32_t i0) {
        return {_mm_set_epi32(i0, i1, i2, i3)};
    }

    static bit_vector<int32_t, 128> load(aligned_view<int32_t, 16> ptr) {
        return {_mm_load_si128(reinterpret_cast<__m128i*>(ptr.get()))};
    }

    static bit_vector<int32_t, 128> load(unaligned_view<int32_t> ptr) {
        return {_mm_loadu_si128(reinterpret_cast<__m128i*>(ptr.get()))};
    }

    void store(aligned_view<int32_t, 16> ptr) const {
        _mm_store_si128(reinterpret_cast<__m128i*>(ptr.get()), data);
    }

    void store(unaligned_view<int32_t> ptr) const {
        _mm_storeu_si128(reinterpret_cast<__m128i*>(ptr.get()), data);
    }

    explicit operator bit_vector<float, 128>() const;

    bool operator==(bit_vector<int32_t, 128> rhs) const {
        __m128i result = _mm_xor_si128(data, rhs.data);
        return _mm_test_all_zeros(result, result);
    }
};

template <>
struct bit_vector<int32_t, 256> {
    __m256i data;

    static constexpr size_t width_bytes = 32;
    static constexpr size_t size        = 8;

    static bit_vector<int32_t, 256>
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

    static bit_vector<int32_t, 256> load(aligned_view<int32_t, 32> ptr) {
        return {_mm256_load_si256(reinterpret_cast<__m256i*>(ptr.get()))};
    }

    static bit_vector<int32_t, 256> load(unaligned_view<int32_t> ptr) {
        return {_mm256_loadu_si256(reinterpret_cast<__m256i*>(ptr.get()))};
    }

    void store(aligned_view<int32_t, 32> ptr) const {
        _mm256_store_si256(reinterpret_cast<__m256i*>(ptr.get()), data);
    }

    void store(unaligned_view<int32_t> ptr) const {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr.get()), data);
    }

    explicit operator bit_vector<float, 256>() const;

    bool operator==(bit_vector<int32_t, 256> rhs) const {
        __m256i result    = _mm256_xor_si256(data, rhs.data);
        __m128  result_lo = _mm256_castps256_ps128(result);
        __m128  result_hi = _mm256_extractf128_ps(result, 1);
        return _mm_test_all_zeros(result_hi, result_lo);
    }

    bit_vector<int32_t, 128> low_bits() const {
        return {_mm256_castsi256_si128(data)};
    }
    bit_vector<int32_t, 128> high_bits() const {
        return {_mm256_extracti128_si256(data, 1)};
    }
};

template <>
struct bit_vector<float, 128> {
    __m128 data;

    static constexpr size_t width_bytes = 32;
    static constexpr size_t size        = 4;

    static bit_vector<float, 128> from(float f3, float f2, float f1, float f0) {
        return {_mm_set_ps(f0, f1, f2, f3)};
    }

    static bit_vector<float, 128> load(aligned_view<float, 16> ptr) {
        return {_mm_load_ps(ptr.get())};
    }

    static bit_vector<float, 128> load(unaligned_view<float> ptr) {
        return {_mm_loadu_ps(ptr.get())};
    }

    void store(aligned_view<float, 16> ptr) { _mm_store_ps(ptr.get(), data); }
    void store(unaligned_view<float> ptr) { _mm_storeu_ps(ptr.get(), data); }

    explicit operator bit_vector<int32_t, 128>() const;

    friend bit_vector<float, 128>
    operator*(bit_vector<float, 128> lhs, bit_vector<float, 128> rhs) {
        return {_mm_mul_ps(lhs.data, rhs.data)};
    }

    bit_vector<float, 128>& operator*=(bit_vector<float, 128> rhs) {
        data = _mm_mul_ps(data, rhs.data);
        return *this;
    }

    bool operator==(bit_vector<float, 128> rhs) const {
        // compare not equal, unordered, non-signaling
        __m128 result_neq = _mm_cmp_ps(data, rhs.data, _CMP_NEQ_UQ);
        return _mm_test_all_zeros(result_neq, result_neq);
    }
};

template <>
struct bit_vector<float, 256> {
    __m256 data;

    static constexpr size_t width_bytes = 32;
    static constexpr size_t size        = 8;

    static bit_vector<float, 256>
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

    static bit_vector<float, 256> load(aligned_view<float, 32> ptr) {
        return {_mm256_load_ps(ptr.get())};
    }

    static bit_vector<float, 256> load(unaligned_view<float> ptr) {
        return {_mm256_loadu_ps(ptr.get())};
    }

    void store(aligned_view<float, 32> ptr) const {
        _mm256_store_ps(ptr.get(), data);
    }

    void store(unaligned_view<float> ptr) const {
        _mm256_storeu_ps(ptr.get(), data);
    }

    explicit operator bit_vector<int32_t, 256>() const;

    friend bit_vector<float, 256>
    operator*(bit_vector<float, 256> lhs, bit_vector<float, 256> rhs) {
        return {_mm256_mul_ps(lhs.data, rhs.data)};
    }

    bit_vector<float, 256>& operator*=(bit_vector<float, 256> rhs) {
        data = _mm256_mul_ps(data, rhs.data);
        return *this;
    }

    bool operator==(bit_vector<float, 256> rhs) const {
        // compare not equal, unordered, non-signaling
        __m256 result_neq    = _mm256_cmp_ps(data, rhs.data, _CMP_NEQ_UQ);
        __m128 result_neq_lo = _mm256_castps256_ps128(result_neq);
        __m128 result_neq_hi = _mm256_extractf128_ps(result_neq, 1);
        return _mm_test_all_zeros(result_neq_hi, result_neq_lo);
    }

    bit_vector<float, 128> low_bits() const {
        return {_mm256_castps256_ps128(data)};
    }
    bit_vector<float, 128> high_bits() const {
        return {_mm256_extractf128_ps(data, 1)};
    }
};

using i32x8 = bit_vector<int32_t, 256>;
using i32x4 = bit_vector<int32_t, 128>;
using f32x8 = bit_vector<float, 256>;
using f32x4 = bit_vector<float, 128>;

inline bit_vector<int32_t, 128>::operator bit_vector<float, 128>() const {
    return {_mm_castsi128_ps(data)};
}

inline bit_vector<int32_t, 256>::operator bit_vector<float, 256>() const {
    return {_mm256_castsi256_ps(data)};
}

inline bit_vector<float, 128>::operator bit_vector<int32_t, 128>() const {
    return {_mm_castps_si128(data)};
}

inline bit_vector<float, 256>::operator bit_vector<int32_t, 256>() const {
    return {_mm256_castps_si256(data)};
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
    return {_mm256_permute4x64_epi64(
            static_cast<i32x8>(v).data, control4<flags...>::value)};
}

}  // namespace simd
