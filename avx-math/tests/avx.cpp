#include <avx-math/avx.h>

#include <gtest/gtest.h>

#include <cstdlib>
#include <numeric>

namespace {

std::ostream& operator<<(std::ostream& os, avx::f32x8 v) {
    __attribute((aligned(32))) float out[8];
    avx::store_aligned(out, v);
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

TEST(f32x4, from) {
    float expected[4] = {0.0, 1.0, 2.0, 3.0};
    float actual[4];
    avx::store_unaligned(actual, avx::f32x4::from(0.0f, 1.0f, 2.0f, 3.0f));
    EXPECT_TRUE(std::equal(expected, expected + 4, actual));
}

TEST(f32x8, from) {
    float expected[8] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
    float actual[8];
    avx::store_unaligned(
            actual,
            avx::f32x8::from(0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f));
    EXPECT_TRUE(std::equal(expected, expected + 8, actual));
}

template <typename T>
void test_f32_load_store_aligned() {
    __attribute((aligned(32))) float expected[]
            = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
    __attribute((aligned(32))) float actual[T::size];

    auto v = avx::load_aligned<T>(expected);
    avx::store_aligned(actual, v);

    EXPECT_TRUE(std::equal(expected, expected + T::size, actual));
}

TEST(f32x4, load_store_aligned) {
    test_f32_load_store_aligned<avx::f32x4>();
}

TEST(f32x8, load_store_aligned) {
    test_f32_load_store_aligned<avx::f32x4>();
}

template <typename T>
void test_f32_load_store_unaligned() {
    // -1.0 is only used for padding
    __attribute((aligned(32))) float expected[]
            = {-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
    __attribute((aligned(32))) float actual[1 + T::size] = {-1.0};

    ASSERT_TRUE(*reinterpret_cast<uint64_t*>(&expected) % 32 == 0);
    ASSERT_TRUE(*reinterpret_cast<uint64_t*>(&actual) % 32 == 0);

    auto v = avx::load_unaligned<T>(expected + 1);
    avx::store_unaligned(actual + 1, v);

    EXPECT_TRUE(std::equal(expected, expected + 1 + T::size, actual));
}

TEST(f32x4, load_store_unaligned) {
    test_f32_load_store_unaligned<avx::f32x4>();
}

TEST(f32x8, load_store_unaligned) {
    test_f32_load_store_unaligned<avx::f32x8>();
}

template <typename T>
void test_f32_multiply() {
    float input[]    = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
    float expected[] = {0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0};
    float actual[T::size];

    auto v = avx::load_unaligned<T>(input);
    avx::store_unaligned(actual, v * v);

    EXPECT_TRUE(std::equal(expected, expected + T::size, actual));
}

TEST(f32x4, multiply) {
    test_f32_multiply<avx::f32x4>();
}

TEST(f32x8, multiply) {
    test_f32_multiply<avx::f32x8>();
}

template <typename T>
void test_f32_equality() {
    float a[] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
    float b[] = {-0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};  // -0.0 == 0.0

    auto v1 = avx::load_unaligned<T>(a);
    auto v2 = v1;
    auto v3 = avx::load_unaligned<T>(b);

    EXPECT_EQ(v1, v2);
    EXPECT_EQ(v1, v3);
}

TEST(f32x4, equality) {
    test_f32_equality<avx::f32x4>();
}

TEST(f32x8, equality) {
    test_f32_equality<avx::f32x8>();
}

template <typename T>
void test_f32_hadd() {
    float a[] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
    float b[] = {8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0};
    float c[] = {1.0, 5.0, 17.0, 21.0, 9.0, 13.0, 25.0, 29.0};

    auto v1       = avx::load_unaligned<T>(a);
    auto v2       = avx::load_unaligned<T>(b);
    auto expected = avx::load_unaligned<T>(c);

    auto actual = avx::hadd(v1, v2);

    EXPECT_EQ(expected, actual);
}

TEST(f32x4, hadd) {
    test_f32_hadd<avx::f32x4>();
}

TEST(f32x8, hadd) {
    test_f32_hadd<avx::f32x8>();
}
