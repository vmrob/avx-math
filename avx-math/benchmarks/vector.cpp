#include <avx-math/vector.h>

#include <benchmark/benchmark.h>

#include <cstdlib>
#include <random>

void gen_vectors(point32f* points, size_t n) {
    std::random_device                    rd;
    std::mt19937                          gen(rd());
    std::uniform_real_distribution<float> dis(-10000.0f, 10000.0f);
    for (size_t i = 0; i < n; ++i) {
        points[i].x = dis(gen);
        points[i].y = dis(gen);
    }
}

struct test_data {
    point32f* a      = nullptr;
    point32f* b      = nullptr;
    float*    result = nullptr;

    test_data(size_t n) {
        posix_memalign(reinterpret_cast<void**>(&a), 32, sizeof(point32f) * n);
        posix_memalign(reinterpret_cast<void**>(&b), 32, sizeof(point32f) * n);
        posix_memalign(
                reinterpret_cast<void**>(&result), 32, sizeof(float) * n);
        gen_vectors(a, n);
        gen_vectors(b, n);
    }

    ~test_data() {
        free(a);
        free(b);
        free(result);
    }
};

static void
BM_dot_product_aligned_n(benchmark::State& state) {
    const size_t n = state.range(0);

    test_data data{n};

    while (state.KeepRunning()) {
        dot_product_aligned_n(data.a, data.b, data.result, n);

        benchmark::DoNotOptimize(data.a);
        benchmark::DoNotOptimize(data.b);
        benchmark::DoNotOptimize(data.result);
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * 8);
}

BENCHMARK(BM_dot_product_aligned_n)->Range(2, 16192);

static void
BM_dot_product_aligned_n_slow(benchmark::State& state) {
    const size_t n = state.range(0);

    test_data data{n};

    while (state.KeepRunning()) {
        dot_product_aligned_n_slow(data.a, data.b, data.result, n);

        benchmark::DoNotOptimize(data.a);
        benchmark::DoNotOptimize(data.b);
        benchmark::DoNotOptimize(data.result);
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * 8);
}

BENCHMARK(BM_dot_product_aligned_n_slow)->Range(2, 16192);


BENCHMARK_MAIN();
