#include <avx-math/avx.h>
#include <avx-math/vector.h>
#include <cmath>
#include <cstdio>

#include <immintrin.h>

namespace math {
namespace {

template <typename T>
void dot_product_n_aligned_impl(
        __attribute((aligned(64))) vector<T>* a,
        __attribute((aligned(64))) vector<T>* b,
        __attribute((aligned(64))) T*         out,
        unsigned long long int                n) {}

}  // namespace

void dot_product_n_aligned(
        __attribute((aligned(64))) vector32f* a,
        __attribute((aligned(64))) vector32f* b,
        __attribute((aligned(64))) float*     out,
        unsigned long long int                n) {
    size_t i = 0;
#ifdef __AVX__
    float* af = reinterpret_cast<float*>(a);
    float* bf = reinterpret_cast<float*>(b);
    for (; i + 8 < n; i += 8) {
        // [a1x*b1x, a1y*b1y, a2x*b2x, a2y*b2y, ...]
        auto prod_0_3
                = avx::load_aligned<8>(af + i) * avx::load_aligned<8>(bf + i);

        // [a4x*b4x, a4y*b4y, a5x*b5x, a5y*b5y, ...]
        auto prod_4_7 = avx::load_aligned<8>(af + i + 8)
                        * avx::load_aligned<8>(bf + i + 8);

        // [r0, r1, r4, r5, r2, r3, r6, r7]
        auto result_interleaved = avx::hadd(prod_0_3, prod_4_7);

        __m256i result_i = _mm256_permute4x64_epi64(
                _mm256_castps_si256(result_interleaved.data), 0b11011000);

        __m256 result = _mm256_castsi256_ps(result_i);

        avx::store_aligned(out + i, {result});
    }
#endif
    for (; i < n; ++i) {
        out[i] = a[i].x * b[i].x + a[i].y * b[i].y;
    }
}

}  // namespace math
