#include <simd/math/dot_product.h>
#include <simd/math/vector2.h>
#include <simd/memory.h>

#include <gtest/gtest.h>

#include <cstdlib>
#include <random>

using namespace simd::math;

class dot_product_fixture : public ::testing::Test {
public:
    vector2f* a     = nullptr;
    vector2f* b     = nullptr;
    float* result   = nullptr;
    float* expected = nullptr;

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
        a        = simd::aligned_alloc<vector2f>(32, sizeof(vector2f) * _size);
        b        = simd::aligned_alloc<vector2f>(32, sizeof(vector2f) * _size);
        result   = simd::aligned_alloc<float>(32, sizeof(float) * _size);
        expected = simd::aligned_alloc<float>(32, sizeof(float) * _size);
    }

    void regenerate(size_t n) {
        _size = n;
        free_vectors();
        allocate_vectors();
        fill_vectors();
    }

    void fill_vectors() {
        for (size_t i = 0; i < _size; ++i) {
            a[i].x = i;
            a[i].y = i;
            b[i].x = i;
            b[i].y = i;
        }
    }

    size_t size() const { return _size; }

    bool results_are_near() {
        for (size_t i = 0; i < _size; ++i) {
            if (std::fabs(expected[i] - result[i]) > 0.0001) {
                return false;
            }
        }
        return true;
    }

    void calculate() {
        dot_product_n(
                simd::as_unaligned_view(a),
                simd::as_unaligned_view(b),
                simd::as_unaligned_view(expected),
                _size);
        dot_product_n(
                simd::as_aligned_view<32>(a),
                simd::as_aligned_view<32>(b),
                simd::as_aligned_view<32>(result),
                _size);
    }

private:
    size_t _size = 0;
};

TEST(vector, basics) {
    {
        vector2f v1{0.0, 0.0};
        vector2f v2{0.0, 0.0};
        EXPECT_EQ(v1.dot(v2), 0.0);
    }
    {
        vector2f v1{2.0, 3.0};
        vector2f v2{4.0, 5.0};
        EXPECT_EQ(v1.dot(v2), 23.0);
    }
    {
        vector2f v1{-2.0, 3.0};
        vector2f v2{4.0, 5.0};
        EXPECT_EQ(v1.dot(v2), 7.0);
    }
}

TEST_F(dot_product_fixture, vectorized_implementation_1) {
    regenerate(1);
    calculate();
    EXPECT_TRUE(results_are_near());
}

TEST_F(dot_product_fixture, vectorized_implementation_2) {
    regenerate(2);
    calculate();
    EXPECT_TRUE(results_are_near());
}

TEST_F(dot_product_fixture, vectorized_implementation_8) {
    regenerate(8);
    calculate();
    EXPECT_TRUE(results_are_near());
}

TEST_F(dot_product_fixture, vectorized_implementation_9) {
    regenerate(9);
    calculate();
    EXPECT_TRUE(results_are_near());
}

TEST_F(dot_product_fixture, vectorized_implementation_16) {
    regenerate(16);
    calculate();
    EXPECT_TRUE(results_are_near());
}

TEST_F(dot_product_fixture, vectorized_implementation_18) {
    regenerate(18);
    calculate();
    EXPECT_TRUE(results_are_near());
}

TEST_F(dot_product_fixture, vectorized_implementation_31) {
    regenerate(31);
    calculate();
    EXPECT_TRUE(results_are_near());
}

TEST_F(dot_product_fixture, vectorized_implementation_32) {
    regenerate(32);
    calculate();
    EXPECT_TRUE(results_are_near());
}

TEST_F(dot_product_fixture, vectorized_implementation_100) {
    regenerate(100);
    calculate();
    EXPECT_TRUE(results_are_near());
}

TEST_F(dot_product_fixture, vectorized_implementation_200) {
    regenerate(200);
    calculate();
    EXPECT_TRUE(results_are_near());
}

TEST_F(dot_product_fixture, vectorized_implementation_1000) {
    regenerate(1000);
    calculate();
    EXPECT_TRUE(results_are_near());
}
