#include <avx-math/vector.h>

#include <gtest/gtest.h>

#include <random>

using namespace math;

class vector_fixture : public ::testing::Test {
public:
    vector32f* a        = nullptr;
    vector32f* b        = nullptr;
    float*     result   = nullptr;
    float*     expected = nullptr;

    void TearDown() { free_vectors(); }

    void free_vectors() {
        free(a);
        free(b);
        free(result);
        free(expected);
        a        = nullptr;
        b        = nullptr;
        result   = nullptr;
        expected = nullptr;
    }

    void allocate_vectors() {
        posix_memalign(
                reinterpret_cast<void**>(&a), 32, sizeof(vector32f) * _size);
        posix_memalign(
                reinterpret_cast<void**>(&b), 32, sizeof(vector32f) * _size);
        posix_memalign(
                reinterpret_cast<void**>(&result), 32, sizeof(float) * _size);
        posix_memalign(
                reinterpret_cast<void**>(&expected), 32, sizeof(float) * _size);
    }

    void regenerate(size_t n) {
        _size = n;
        free_vectors();
        allocate_vectors();
        randomize_vectors();
    }

    void randomize_vectors() {
        std::random_device                    rd;
        std::mt19937                          gen(rd());
        std::uniform_real_distribution<float> dis(-10000.0f, 10000.0f);
        for (size_t i = 0; i < _size; ++i) {
            a[i].x = dis(gen);
            a[i].y = dis(gen);
            b[i].x = dis(gen);
            b[i].y = dis(gen);
        }
    }

    size_t size() const { return _size; }

    bool results_are_near() {
        for (size_t i = 0; i < _size; ++i) {
            if (std::fabs(expected[i] - result[i]) > 0.0001) {
                printf("unexpected: %f * %f + %f * %f -> %f vs %f\n",
                       a[i].x,
                       b[i].x,
                       a[i].y,
                       b[i].y,
                       result[i],
                       expected[i]);
                return false;
            }
        }
        return true;
    }

    void calculate() {
        dot_product_n(a, b, expected, _size);
        dot_product_n_aligned(a, b, result, _size);
    }

private:
    size_t _size = 0;
};

TEST(vector, basics) {
    {
        vector32f v1{0.0, 0.0};
        vector32f v2{0.0, 0.0};
        EXPECT_EQ(dot_product(v1, v2), 0.0);
    }
    {
        vector32f v1{2.0, 3.0};
        vector32f v2{4.0, 5.0};
        EXPECT_EQ(dot_product(v1, v2), 23.0);
    }
    {
        vector32f v1{-2.0, 3.0};
        vector32f v2{4.0, 5.0};
        EXPECT_EQ(dot_product(v1, v2), 7.0);
    }
}

TEST_F(vector_fixture, vectorized_implementation) {
    regenerate(1);
    calculate();
    EXPECT_TRUE(results_are_near());

    regenerate(2);
    calculate();
    EXPECT_TRUE(results_are_near());

    regenerate(8);
    calculate();
    EXPECT_TRUE(results_are_near());

    regenerate(9);
    calculate();
    EXPECT_TRUE(results_are_near());

    regenerate(16);
    calculate();
    EXPECT_TRUE(results_are_near());

    // regenerate(18);
    // calculate();
    // EXPECT_TRUE(results_are_near());

    // regenerate(100);
    // calculate();
    // EXPECT_TRUE(results_are_near());

    // regenerate(1000);
    // calculate();
    // EXPECT_TRUE(results_are_near());
}
