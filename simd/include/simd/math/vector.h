#pragma once

#include <simd/bit_vector.h>
#include <simd/byte_view.h>

#include <cstdint>
#include <type_traits>

namespace math {

#pragma pack(push, 0)
template <typename T>
struct vector2 {
    T x;
    T y;
    template <typename U>
    U as() {
        static_assert(std::is_pointer_v<U>);
        static_assert(std::is_trivial_v<std::remove_pointer_t<U>>);
        return reinterpret_cast<U>(this);
    }
};
#pragma pack(pop)

using vector2i = vector2<int32_t>;
using vector2f = vector2<float>;
using vector2l = vector2<int64_t>;
using vector2d = vector2<double>;

template <typename T>
__attribute((always_inline)) inline T dot_product(vector2<T> a, vector2<T> b) {
    return a.x * b.x + a.y * b.y;
}

template <typename ComponentType, typename IterationCountType, size_t Alignment>
auto dot_product_n(
        aligned_byte_view<vector2<ComponentType>, Alignment> a,
        aligned_byte_view<vector2<ComponentType>, Alignment> b,
        aligned_byte_view<ComponentType, Alignment>          out,
        IterationCountType n) -> decltype(n(), void()) {
    // TODO: it currently only supports floating point types because integrals
    // would require twice the number of output bits
    static_assert(
            std::is_floating_point<ComponentType>::value,
            "dot_product_n only supports floating point types");
    auto af = aligned_byte_view<ComponentType, Alignment>{
            a.get()->template as<ComponentType*>()};
    auto bf = aligned_byte_view<ComponentType, Alignment>{
            b.get()->template as<ComponentType*>()};
    size_t i = 0;
#ifdef __AVX__
    if constexpr (Alignment >= 32) {
        using AVXVectorType = simd::bit_vector<ComponentType, 256>;
        using ByteViewType
                = aligned_byte_view<ComponentType, AVXVectorType::width_bytes>;
        ByteViewType af_view = af;
        ByteViewType bf_view = bf;
        const size_t simd_iterations
                = n() / ByteViewType::size;  // intentionally truncates
#pragma unroll 4
        for (; i + 2 <= simd_iterations; ++i) {
            // [a1x*b1x, a1y*b1y, a2x*b2x, a2y*b2y, ...]
            auto prod_0_3 = AVXVectorType::load(af_view + i * 2)
                            * AVXVectorType::load(bf_view + i * 2);

            // [a4x*b4x, a4y*b4y, a5x*b5x, a5y*b5y, ...]
            auto prod_4_7 = AVXVectorType::load(af_view + 1 + i * 2)
                            * AVXVectorType::load(bf_view + 1 + i * 2);

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
#pragma unroll 8
    for (; i < n(); ++i) {
        out[i] = dot_product(a[i], b[i]);
    }
}

template <typename T, size_t Alignment>
inline void dot_product_n(
        aligned_byte_view<vector2<T>, Alignment> a,
        aligned_byte_view<vector2<T>, Alignment> b,
        aligned_byte_view<T, Alignment>          out,
        unsigned long long int                   n) {
    auto iteration_count = [&] {
        struct anon {
            anon(size_t n) : _n{n} {}
            size_t operator()() const { return _n; }
            size_t _n;
        };
        return anon{n};
    }();
    dot_product_n(a, b, out, iteration_count);
}

template <typename T>
inline void dot_product_n(
        unaligned_byte_view<vector2<T>> a,
        unaligned_byte_view<vector2<T>> b,
        unaligned_byte_view<T>          out,
        unsigned long long int          n) {
    dot_product_n(
            aligned_byte_view<vector2<T>, 1>{a.get()},
            aligned_byte_view<vector2<T>, 1>{b.get()},
            aligned_byte_view<T, 1>{out.get()},
            n);
}

}  // namespace math
