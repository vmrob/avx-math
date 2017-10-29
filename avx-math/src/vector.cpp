#include <avx-math/vector.h>

#include <immintrin.h>

namespace {

__attribute((always_inline)) inline void dot_product_aligned_n_individual(
        __attribute((aligned(32))) point32f* a,
        __attribute((aligned(32))) point32f* b,
        __attribute((aligned(32))) float*    out,
        size_t                               n) {
    for (size_t i = 0; i < n; ++i) {
        out[i] = a[i].x * b[i].x + a[i].y * b[i].y;
    }
}

}  // namespace

void dot_product_aligned_n(
        __attribute((aligned(32))) point32f* a,
        __attribute((aligned(32))) point32f* b,
        __attribute((aligned(32))) float*    out,
        unsigned long long                   n) {
    size_t i = 0;
#ifdef __AVX512F__
    for (; i + 16 < n; i += 16) {
        __m512 a_0_7    = _mm512_load_ps(reinterpret_cast<float*>(a + i));
        __m512 b_0_7    = _mm512_load_ps(reinterpret_cast<float*>(b + i));
        __m512 prod_0_7 = _mm512_mul_ps(a_0_7, b_0_7);

        __m512 a_8_15    = _mm512_load_ps(reinterpret_cast<float*>(a + i + 8));
        __m512 b_8_15    = _mm512_load_ps(reinterpret_cast<float*>(b + i + 8));
        __m512 prod_8_15 = _mm512_mul_ps(a_8_15, b_8_15);

        __m512 result = _mm512_hadd_ps(prod_0_7, prod_8_15);

        _mm512_store_ps(out, result);
    }
#endif
#ifdef __AVX2__
    for (; i + 8 < n; i += 8) {
        __m256 a_1_4    = _mm256_load_ps(reinterpret_cast<float*>(a + i));
        __m256 b_1_4    = _mm256_load_ps(reinterpret_cast<float*>(b + i));
        __m256 prod_1_4 = _mm256_mul_ps(a_1_4, b_1_4);

        __m256 a_5_8    = _mm256_load_ps(reinterpret_cast<float*>(a + i + 4));
        __m256 b_5_8    = _mm256_load_ps(reinterpret_cast<float*>(b + i + 4));
        __m256 prod_5_8 = _mm256_mul_ps(a_5_8, b_5_8);

        __m256 result = _mm256_hadd_ps(prod_1_4, prod_5_8);

        _mm256_store_ps(out, result);
    }
#endif
    dot_product_aligned_n_individual(a + i, b + i, out + i, n - i);
}

void dot_product_aligned_n_slow(
        __attribute((aligned(32))) point32f* a,
        __attribute((aligned(32))) point32f* b,
        __attribute((aligned(32))) float*    out,
        unsigned long long                   n) {
    dot_product_aligned_n_individual(a, b, out, n);
}
