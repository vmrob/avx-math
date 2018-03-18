#include <simd/math/dot_product.h>
#include <simd/math/vector2.h>
#include <simd/memory.h>

#include <benchmark/benchmark.h>

#include <cstdlib>
#include <random>

void gen_vectors(simd::math::vector2f* vectors, size_t n) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-10000.0f, 10000.0f);
    for (size_t i = 0; i < n; ++i) {
        vectors[i].x = dis(gen);
        vectors[i].y = dis(gen);
    }
}

struct test_data {
    simd::math::vector2f* a = nullptr;
    simd::math::vector2f* b = nullptr;
    float* out              = nullptr;

    test_data(size_t n) {
        a = simd::aligned_alloc<simd::math::vector2f>(
                32, sizeof(simd::math::vector2f) * n);
        b = simd::aligned_alloc<simd::math::vector2f>(
                32, sizeof(simd::math::vector2f) * n);
        out = simd::aligned_alloc<float>(32, sizeof(float) * n);
        gen_vectors(a, n);
        gen_vectors(b, n);
    }

    ~test_data() {
        free(a);
        free(b);
        free(out);
    }
};

static void BM_dot_product_n_aligned(benchmark::State& state) {
    const size_t n = state.range(0);

    test_data data{n};

    while (state.KeepRunning()) {
        dot_product_n(
                simd::as_aligned_view<32>(data.a),
                simd::as_aligned_view<32>(data.b),
                simd::as_aligned_view<32>(data.out),
                n);

        benchmark::DoNotOptimize(data.a);
        benchmark::DoNotOptimize(data.b);
        benchmark::DoNotOptimize(data.out);
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * n);
}

BENCHMARK(BM_dot_product_n_aligned)->Range(2, 16192);

template <size_t I>
static void BM_dot_product_impl(benchmark::State& state) {
    test_data data{I};

    while (state.KeepRunning()) {
        dot_product_n(
                simd::as_aligned_view<32>(data.a),
                simd::as_aligned_view<32>(data.b),
                simd::as_aligned_view<32>(data.out),
                std::integral_constant<size_t, I>{});

        benchmark::DoNotOptimize(data.a);
        benchmark::DoNotOptimize(data.b);
        benchmark::DoNotOptimize(data.out);
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * I);
}

static void BM_dot_product_n_aligned_static_2(benchmark::State& state) {
    BM_dot_product_impl<2>(state);
}
BENCHMARK(BM_dot_product_n_aligned_static_2);

static void BM_dot_product_n_aligned_static_8(benchmark::State& state) {
    BM_dot_product_impl<8>(state);
}
BENCHMARK(BM_dot_product_n_aligned_static_8);

static void BM_dot_product_n_aligned_static_64(benchmark::State& state) {
    BM_dot_product_impl<64>(state);
}
BENCHMARK(BM_dot_product_n_aligned_static_64);

static void BM_dot_product_n_aligned_static_512(benchmark::State& state) {
    BM_dot_product_impl<512>(state);
}
BENCHMARK(BM_dot_product_n_aligned_static_512);

static void BM_dot_product_n_aligned_static_4096(benchmark::State& state) {
    BM_dot_product_impl<4096>(state);
}
BENCHMARK(BM_dot_product_n_aligned_static_4096);

static void BM_dot_product_n_aligned_static_16192(benchmark::State& state) {
    BM_dot_product_impl<16192>(state);
}
BENCHMARK(BM_dot_product_n_aligned_static_16192);

static void BM_dot_product_n_unaligned(benchmark::State& state) {
    const size_t n = state.range(0);

    test_data data{n};

    while (state.KeepRunning()) {
        dot_product_n(
                simd::as_unaligned_view(data.a),
                simd::as_unaligned_view(data.b),
                simd::as_unaligned_view(data.out),
                n);

        benchmark::DoNotOptimize(data.a);
        benchmark::DoNotOptimize(data.b);
        benchmark::DoNotOptimize(data.out);
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * n);
}

BENCHMARK(BM_dot_product_n_unaligned)->Range(2, 16192);

static void BM_dot_product_naive(benchmark::State& state) {
    const size_t n = state.range(0);

    test_data data{n};

    while (state.KeepRunning()) {
        for (size_t i = 0; i < n; ++i) {
            data.out[i] = data.a[i].dot(data.b[i]);
        }

        benchmark::DoNotOptimize(data.a);
        benchmark::DoNotOptimize(data.b);
        benchmark::DoNotOptimize(data.out);
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * n);
}

BENCHMARK(BM_dot_product_naive)->Range(2, 16192);

BENCHMARK_MAIN();
