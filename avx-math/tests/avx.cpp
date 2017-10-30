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

TEST(avx256, load_store_aligned) {
    __attribute((aligned(32))) float expected[8]
            = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
    __attribute((aligned(32))) float actual[8];

    auto v = avx::load_aligned<8>(expected);
    avx::store_aligned(actual, v);

    EXPECT_TRUE(std::equal(expected, expected + 8, actual));
}

TEST(avx, load_store_unaligned_8) {
    // -1.0 is only used for padding
    __attribute((aligned(32))) float expected[9]
            = {-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
    __attribute((aligned(32))) float actual[9] = {-1.0};

    ASSERT_TRUE(*reinterpret_cast<uint64_t*>(&expected) % 32 == 0);
    ASSERT_TRUE(*reinterpret_cast<uint64_t*>(&actual) % 32 == 0);

    auto v = avx::load_unaligned<8>(expected + 1);
    avx::store_unaligned(actual + 1, v);

    EXPECT_TRUE(std::equal(expected, expected + 9, actual));
}

TEST(f32x8, from) {
    float expected[8] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
    float actual[8];

    avx::store_unaligned(
            actual,
            avx::f32x8::from(0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f));

    EXPECT_TRUE(std::equal(expected, expected + 8, actual));
}

TEST(f32x8, multiply) {
    auto expected = avx::f32x8::from(
            0.0f, 1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f, 49.0f);

    auto v = avx::f32x8::from(0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f);

    EXPECT_EQ(v * v, expected);
}

TEST(f32x8, equality) {
    EXPECT_EQ(
            avx::f32x8::from(0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f),
            avx::f32x8::from(0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f));
    EXPECT_EQ(  // -0.0 == 0.0
            avx::f32x8::from(-0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f),
            avx::f32x8::from(0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f));
}

TEST(avx256, hadd) {
    auto expected = avx::f32x8::from(
            1.0f, 5.0f, 17.0f, 21.0f, 9.0f, 13.0f, 25.0f, 29.0f);

    auto v1 = avx::f32x8::from(0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f);
    auto v2 = avx::f32x8::from(
            8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f);

    EXPECT_EQ(expected, avx::hadd(v1, v2));
}
