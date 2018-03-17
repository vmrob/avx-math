#include <simd/math/vector.h>

#include <gtest/gtest.h>

template <typename T>
void TestAddition() {
    using X = decltype(T::x);
    using Y = decltype(T::y);

    constexpr T v1 = {.x = X(1.0), .y = Y(2.0)};
    constexpr T v2 = {.x = X(3.0), .y = Y(4.0)};
    {
        T vec = v1;
        vec += v2;
        EXPECT_EQ(vec.x, X(4.0));
        EXPECT_EQ(vec.y, Y(6.0));
    }
    {
        const T vec = v1 + v2;
        EXPECT_EQ(vec.x, X(4.0));
        EXPECT_EQ(vec.y, Y(6.0));
    }
}

TEST(vector2, addition) {
    TestAddition<simd::math::vector2i>();
    TestAddition<simd::math::vector2l>();
    TestAddition<simd::math::vector2f>();
    TestAddition<simd::math::vector2d>();
}

template <typename T>
void TestSubtraction() {
    using X = decltype(T::x);
    using Y = decltype(T::y);

    constexpr T v1 = {.x = X(1.0), .y = Y(2.0)};
    constexpr T v2 = {.x = X(3.0), .y = Y(4.0)};
    {
        T vec = v2;
        vec -= v1;
        EXPECT_EQ(vec.x, 2.0);
        EXPECT_EQ(vec.y, 2.0);
    }
    {
        const T vec = v2 - v1;
        EXPECT_EQ(vec.x, 2.0);
        EXPECT_EQ(vec.y, 2.0);
    }
}

TEST(vector2, subtraction) {
    TestSubtraction<simd::math::vector2i>();
    TestSubtraction<simd::math::vector2l>();
    TestSubtraction<simd::math::vector2f>();
    TestSubtraction<simd::math::vector2d>();
}

template <typename T>
void TestMultiplicationByScalar() {
    using X = decltype(T::x);
    using Y = decltype(T::y);

    constexpr T v = {.x = X(1.0), .y = Y(2.0)};
    {
        T vec = v;
        vec *= 2.0;
        EXPECT_EQ(vec.x, 2.0);
        EXPECT_EQ(vec.y, 4.0);
    }
    {
        const T vec = v * 2.0;
        EXPECT_EQ(vec.x, 2.0);
        EXPECT_EQ(vec.y, 4.0);
    }
    {
        const T vec = 2.0 * v;
        EXPECT_EQ(vec.x, 2.0);
        EXPECT_EQ(vec.y, 4.0);
    }
}

TEST(vector2, multiplication) {
    TestMultiplicationByScalar<simd::math::vector2i>();
    TestMultiplicationByScalar<simd::math::vector2l>();
    TestMultiplicationByScalar<simd::math::vector2f>();
    TestMultiplicationByScalar<simd::math::vector2d>();
}

template <typename T>
void TestDivisionByScalar() {
    using X = decltype(T::x);
    using Y = decltype(T::y);

    constexpr T v = {.x = X(2.0), .y = Y(4.0)};
    {
        T vec = v;
        vec /= 2.0;
        EXPECT_EQ(vec.x, 1.0);
        EXPECT_EQ(vec.y, 2.0);
    }
    {
        const T vec = v / 2.0;
        EXPECT_EQ(vec.x, 1.0);
        EXPECT_EQ(vec.y, 2.0);
    }
}

TEST(vector2, division) {
    TestDivisionByScalar<simd::math::vector2i>();
    TestDivisionByScalar<simd::math::vector2l>();
    TestDivisionByScalar<simd::math::vector2f>();
    TestDivisionByScalar<simd::math::vector2d>();
}
