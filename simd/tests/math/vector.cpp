#include <simd/math/vector.h>

#include <gtest/gtest.h>

#include <random>

using namespace math;

namespace {

std::ostream& operator<<(std::ostream& os, simd::f32x8 v) {
    __attribute((aligned(32))) float out[8];
    v.store(as_aligned_view<32>(out));
    // clang-format off
    os << "["
        << out[0] << ", "
        << out[1] << ", "
        << out[2] << ", "
        << out[3] << ", "
        << out[4] << ", "
        << out[5] << ", "
        << out[6] << ", "
        << out[7] <<
    "]";
    // clang-format on
    return os;
}

}  // namespace

class vector_fixture : public ::testing::Test {
public:
    vector2f* a        = nullptr;
    vector2f* b        = nullptr;
    float*    result   = nullptr;
    float*    expected = nullptr;

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
                reinterpret_cast<void**>(&a), 32, sizeof(vector2f) * _size);
        posix_memalign(
                reinterpret_cast<void**>(&b), 32, sizeof(vector2f) * _size);
        posix_memalign(
                reinterpret_cast<void**>(&result), 32, sizeof(float) * _size);
        posix_memalign(
                reinterpret_cast<void**>(&expected), 32, sizeof(float) * _size);
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
                as_unaligned_view(a),
                as_unaligned_view(b),
                as_unaligned_view(expected),
                _size);
        dot_product_n(
                as_aligned_view<32>(a),
                as_aligned_view<32>(b),
                as_aligned_view<32>(result),
                _size);
    }

private:
    size_t _size = 0;
};

TEST(vector, basics) {
    {
        vector2f v1{0.0, 0.0};
        vector2f v2{0.0, 0.0};
        EXPECT_EQ(dot_product(v1, v2), 0.0);
    }
    {
        vector2f v1{2.0, 3.0};
        vector2f v2{4.0, 5.0};
        EXPECT_EQ(dot_product(v1, v2), 23.0);
    }
    {
        vector2f v1{-2.0, 3.0};
        vector2f v2{4.0, 5.0};
        EXPECT_EQ(dot_product(v1, v2), 7.0);
    }
}

TEST_F(vector_fixture, vectorized_implementation_1) {
    regenerate(1);
    calculate();
    EXPECT_TRUE(results_are_near());
}

TEST_F(vector_fixture, vectorized_implementation_2) {
    regenerate(2);
    calculate();
    EXPECT_TRUE(results_are_near());
}

TEST_F(vector_fixture, vectorized_implementation_8) {
    regenerate(8);
    calculate();
    EXPECT_TRUE(results_are_near());
}

TEST_F(vector_fixture, vectorized_implementation_9) {
    regenerate(9);
    calculate();
    EXPECT_TRUE(results_are_near());
}

TEST_F(vector_fixture, vectorized_implementation_16) {
    regenerate(16);
    calculate();
    EXPECT_TRUE(results_are_near());
}

TEST_F(vector_fixture, vectorized_implementation_18) {
    regenerate(18);
    calculate();
    EXPECT_TRUE(results_are_near());
}

TEST_F(vector_fixture, vectorized_implementation_31) {
    regenerate(31);
    calculate();
    EXPECT_TRUE(results_are_near());
}

TEST_F(vector_fixture, vectorized_implementation_32) {
    regenerate(32);
    calculate();
    EXPECT_TRUE(results_are_near());
}

TEST_F(vector_fixture, vectorized_implementation_100) {
    regenerate(100);
    calculate();
    EXPECT_TRUE(results_are_near());
}

TEST_F(vector_fixture, vectorized_implementation_200) {
    regenerate(200);
    calculate();
    EXPECT_TRUE(results_are_near());
}

TEST_F(vector_fixture, vectorized_implementation_1000) {
    regenerate(1000);
    calculate();
    EXPECT_TRUE(results_are_near());
}
