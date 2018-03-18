#pragma once

#include <simd/bit_vector.h>
#include <simd/math/vector.h>
#include <simd/view.h>

namespace simd::math {

template <typename ComponentType, typename IterationCountType, size_t Alignment>
void dot_product_n(
        aligned_view<vector2<ComponentType>, Alignment> a,
        aligned_view<vector2<ComponentType>, Alignment> b,
        aligned_view<ComponentType, Alignment> out,
        IterationCountType n) {
    // TODO: it currently only supports floating point types because integrals
    // would require twice the number of output bits
    static_assert(
            std::is_floating_point_v<ComponentType>,
            "dot_product_n only supports floating point types");
    auto af  = a.template as<ComponentType>();
    auto bf  = b.template as<ComponentType>();
    size_t i = 0;
#ifdef __AVX__
    if constexpr (Alignment >= 32) {
        using SimdVector = simd::bit_vector<ComponentType, 256>;
        using ByteViewType
                = aligned_view<ComponentType, SimdVector::width_bytes>;
        ByteViewType af_view = af;
        ByteViewType bf_view = bf;
        const size_t simd_iterations
                = n / ByteViewType::size;  // intentionally truncates
#pragma unroll 4
        for (; i + 2 <= simd_iterations; ++i) {
            // [a1x*b1x, a1y*b1y, a2x*b2x, a2y*b2y, ...]
            auto prod_0_3 = SimdVector::load(af_view + i * 2)
                            * SimdVector::load(bf_view + i * 2);

            // [a4x*b4x, a4y*b4y, a5x*b5x, a5y*b5y, ...]
            auto prod_4_7 = SimdVector::load(af_view + 1 + i * 2)
                            * SimdVector::load(bf_view + 1 + i * 2);

            // [r0, r1, r4, r5, r2, r3, r6, r7] (if 32x8)
            // [r0, r2, r1, r3] (if 64x4)
            auto interleaved = simd::hadd(prod_0_3, prod_4_7);
            auto result      = simd::permute4x64(
                    interleaved, simd::control4<0, 2, 1, 3>());

            result.store(out + i);
        }
        i *= ByteViewType::size;
    }
#endif
#pragma unroll 4
    for (; i < n; ++i) {
        out[i] = a[i].dot(b[i]);
    }
}

template <typename T, typename IterationType>
inline void dot_product_n(
        unaligned_view<vector2<T>> a,
        unaligned_view<vector2<T>> b,
        unaligned_view<T> out,
        IterationType n) {
#pragma unroll 4
    for (size_t i = 0; i < n; ++i) {
        out[i] = a[i].dot(b[i]);
    }
}

}  // namespace simd::math
