#pragma once

#include <avx-math/avx.h>

#include <immintrin.h>

#include <cstdint>

namespace math {

#pragma pack(push, 0)
template <typename T>
struct vector {
    T x;
    T y;
};
#pragma pack(pop)

using vector32i = vector<int32_t>;
using vector32f = vector<float>;
using vector64i = vector<int64_t>;
using vector64f = vector<double>;

template <typename T>
using aligned_ptr = __attribute((aligne_value(64))) T*;

template <typename ComponentType, typename IterationCountType>
auto dot_product_n_aligned(
        aligned_ptr<vector<ComponentType>> a,
        aligned_ptr<vector<ComponentType>> b,
        aligned_ptr<ComponentType>         out,
        IterationCountType                 n) -> decltype(n(), void()) {
    size_t         i  = 0;
    ComponentType* af = reinterpret_cast<ComponentType*>(a);
    ComponentType* bf = reinterpret_cast<ComponentType*>(b);
#ifdef __AVX__
    using AVXVectorType = typename avx::widest_avx_vector<ComponentType>::type;
#pragma unroll 4
    for (; i + AVXVectorType::size <= n(); i += AVXVectorType::size) {
        // [a1x*b1x, a1y*b1y, a2x*b2x, a2y*b2y, ...]
        auto prod_0_3 = avx::load_aligned<AVXVectorType>(af + i)
                        * avx::load_aligned<AVXVectorType>(bf + i);

        // [a4x*b4x, a4y*b4y, a5x*b5x, a5y*b5y, ...]
        auto prod_4_7
                = avx::load_aligned<AVXVectorType>(af + i + AVXVectorType::size)
                  * avx::load_aligned<AVXVectorType>(
                            bf + i + AVXVectorType::size);

        // [r0, r1, r4, r5, r2, r3, r6, r7] (if 32x8)
        // [r0, r2, r1, r3] (if 64x4)
        auto interleaved = avx::hadd(prod_0_3, prod_4_7);
        auto result
                = avx::permute4x64(interleaved, avx::control4<0, 2, 1, 3>());

        avx::store_aligned(out + i, result);
    }
#endif
#pragma unroll 8
    for (; i < n(); ++i) {
        out[i] = a[i].x * b[i].x + a[i].y * b[i].y;
    }
}

template <typename T>
__attribute((always_inline)) inline T dot_product(vector<T> a, vector<T> b) {
    return a.x * b.x + a.y * b.y;
}

template <typename T>
inline void dot_product_n_unaligned(
        vector<T>* a, vector<T>* b, T* out, unsigned long long int n) {
    for (unsigned long long int i = 0; i < n; ++i) {
        out[i] = dot_product(a[i], b[i]);
    }
}

template <typename T>
inline void dot_product_n_aligned(
        vector<T>* a, vector<T>* b, T* out, unsigned long long int n) {
    auto iteration_count = [&] {
        struct anon {
            anon(size_t n) : _n{n} {}
            size_t operator()() const { return _n; }
            size_t _n;
        };
        return anon{n};
    }();
    dot_product_n_aligned(a, b, out, iteration_count);
}

}  // namespace math
