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
    for (; i + 8 < n; i += 8) {
        auto a_0_3 = avx<256>::load_aligned(af + i);
        auto b_0_3 = avx<256>::load_aligned(bf + i);
        // [a1x*b1x, a1y*b1y, a2x*b2x, a2y*b2y, ...]
        auto prod_0_3 = a_0_3 * b_0_3;

        auto a_4_7 = avx<256>::load_aligned(af + i + 4);
        auto b_4_7 = avx<256>::load_aligned(bf + i + 4);
        // [a4x*b4x, a4y*b4y, a5x*b5x, a5y*b5y, ...]
        auto prod_4_7 = a_4_7 * b_4_7;

        // results is interleaved [r0, r1, r4, r5, r2, r3, r6, r7]
        __m256 result_0_3 = _mm256_hadd_ps(prod_0_3.data, prod_0_3.data);
        __m256 result_4_7 = _mm256_hadd_ps(prod_4_7.data, prod_4_7.data);

        _mm256_store_ps(out + i, result);
        printf("i: %zu -> %zu\n", i, i + 8);
        for (size_t j = 0; j < 8; ++j) {
            printf("\t%f * %f + %f * %f -> %f\n",
                   a[j].x,
                   b[j].x,
                   a[j].y,
                   b[j].y,
                   out[j]);
            if (std::fabs(dot_product(a[j], b[j]) - result[j]) > 0.0001) {
                printf("error\n");
            }
        }
    }
#endif
    for (; i < n; ++i) {
        out[i] = a[i].x * b[i].x + a[i].y * b[i].y;
        printf("i: %zu\n", i);
        printf("\t%f * %f + %f * %f -> %f\n",
               a[i].x,
               b[i].x,
               a[i].y,
               b[i].y,
               out[i]);
    }
}

}  // namespace math
