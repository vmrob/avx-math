#include <simd/math/vector.h>

#include <gtest/gtest.h>

template <typename T, typename X, typename Y>
void TestConstruction(X x, Y y) {
    T vec = {.x = x, .y = y};
    EXPECT_EQ(vec.x, x);
    EXPECT_EQ(vec.y, y);
}

TEST(vector2, construction) {
    TestConstruction<simd::math::vector2i>(1, 2);
    TestConstruction<simd::math::vector2l>(1l, 2l);
    TestConstruction<simd::math::vector2f>(1.0f, 2.0f);
    TestConstruction<simd::math::vector2d>(1.0, 2.0);
}

template <typename T, typename X, typename Y>
void TestAddition(X x, Y y) {
    {
        T vec = {.x = x, .y = y};
        vec += T{.x = x, .y = y};
        EXPECT_EQ(vec.x, x + x);
        EXPECT_EQ(vec.y, y + y);
    }
    {
        const T vec = T{.x = x, .y = y} + T{.x = x, .y = y};
        EXPECT_EQ(vec.x, x + x);
        EXPECT_EQ(vec.y, y + y);
    }
}

TEST(vector2, addition) {
    TestAddition<simd::math::vector2i>(1, 2);
    TestAddition<simd::math::vector2l>(1l, 2l);
    TestAddition<simd::math::vector2f>(1.0f, 2.0f);
    TestAddition<simd::math::vector2d>(1.0, 2.0);
}

template <typename T, typename X, typename Y>
void TestSubtraction(X x, Y y) {
    {
        T vec = {.x = 2 * x, .y = 2 * y};
        vec -= T{.x = x, .y = y};
        EXPECT_EQ(vec.x, x);
        EXPECT_EQ(vec.y, y);
    }
    {
        const T vec = T{.x = 2 * x, .y = 2 * y} - T{.x = x, .y = y};
        EXPECT_EQ(vec.x, x);
        EXPECT_EQ(vec.y, y);
    }
}

TEST(vector2, subtraction) {
    TestSubtraction<simd::math::vector2i>(1, 2);
    TestSubtraction<simd::math::vector2l>(1l, 2l);
    TestSubtraction<simd::math::vector2f>(1.0f, 2.0f);
    TestSubtraction<simd::math::vector2d>(1.0, 2.0);
}

template <typename T, typename V, typename S>
void TestMultiplicationByScalar(V v, S s) {
    {
        T vec = {.x = 2 * v, .y = v};
        vec *= s;
        EXPECT_EQ(vec.x, 2 * v * s);
        EXPECT_EQ(vec.y, v * s);
    }
    {
        const T vec = T{.x = 2 * v, .y = v} * s;
        EXPECT_EQ(vec.x, 2 * v * s);
        EXPECT_EQ(vec.y, v * s);
    }
    {
        const T vec = s * T{.x = 2 * v, .y = v};
        EXPECT_EQ(vec.x, 2 * v * s);
        EXPECT_EQ(vec.y, v * s);
    }
}

TEST(vector2, multiplication) {
    TestMultiplicationByScalar<simd::math::vector2i>(1, 2);
    TestMultiplicationByScalar<simd::math::vector2l>(1l, 2l);
    TestMultiplicationByScalar<simd::math::vector2f>(1.0f, 2.0f);
    TestMultiplicationByScalar<simd::math::vector2d>(1.0, 2.0);
}

template <typename T, typename V, typename S>
void TestDivisionByScalar(V v, S s) {
    {
        T vec = {.x = 2 * v, .y = v};
        vec /= s;
        EXPECT_EQ(vec.x, 2 * v / s);
        EXPECT_EQ(vec.y, v / s);
    }
    {
        const T vec = T{.x = 2 * v, .y = v} / s;
        EXPECT_EQ(vec.x, 2 * v / s);
        EXPECT_EQ(vec.y, v / s);
    }
}

TEST(vector2, division) {
    TestDivisionByScalar<simd::math::vector2i>(1, 2);
    TestDivisionByScalar<simd::math::vector2l>(1l, 2l);
    TestDivisionByScalar<simd::math::vector2f>(1.0f, 2.0f);
    TestDivisionByScalar<simd::math::vector2d>(1.0, 2.0);
}
