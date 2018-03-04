#include <simd/bit_vector.h>

#include <gtest/gtest.h>

#include <cstdlib>
#include <numeric>

TEST(i32x4, from) {
    int32_t expected[4] = {0, 1, 2, 3};
    int32_t actual[4];
    simd::i32x4::from(0, 1, 2, 3).store(make_unaligned_view(actual));
    EXPECT_TRUE(std::equal(expected, expected + 4, actual));
}

TEST(i32x8, from) {
    int32_t expected[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    int32_t actual[8];
    simd::i32x8::from(0, 1, 2, 3, 4, 5, 6, 7)
            .store(make_unaligned_view(actual));
    EXPECT_TRUE(std::equal(expected, expected + 8, actual));
}

TEST(f32x4, from) {
    float expected[4] = {0.0, 1.0, 2.0, 3.0};
    float actual[4];
    simd::f32x4::from(0.0f, 1.0f, 2.0f, 3.0f)
            .store(make_unaligned_view(actual));
    EXPECT_TRUE(std::equal(expected, expected + 4, actual));
}

TEST(f32x8, from) {
    float expected[8] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
    float actual[8];
    simd::f32x8::from(0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f)
            .store(make_unaligned_view(actual));
    EXPECT_TRUE(std::equal(expected, expected + 8, actual));
}

template <typename SourceT, typename T>
void test_32_load_store_aligned() {
    __attribute((aligned(32))) SourceT expected[] = {0, 1, 2, 3, 4, 5, 6, 7};
    __attribute((aligned(32))) SourceT actual[T::size];

    auto v = T::load(make_aligned_view<32>(expected));
    v.store(make_unaligned_view(actual));

    EXPECT_TRUE(std::equal(expected, expected + T::size, actual));
}

TEST(i32x4, load_store_aligned) {
    test_32_load_store_aligned<int32_t, simd::i32x4>();
}

TEST(i32x8, load_store_aligned) {
    test_32_load_store_aligned<int32_t, simd::i32x4>();
}

TEST(f32x4, load_store_aligned) {
    test_32_load_store_aligned<float, simd::f32x4>();
}

TEST(f32x8, load_store_aligned) {
    test_32_load_store_aligned<float, simd::f32x4>();
}

template <typename SourceT, typename T>
void test_32_load_store_unaligned() {
    // -1.0 is only used for padding
    __attribute((aligned(32))) SourceT expected[]
            = {-1, 0, 1, 2, 3, 4, 5, 6, 7};
    __attribute((aligned(32))) SourceT actual[1 + T::size] = {-1};

    // do the test twice because one of these will have to be unaligned
    {
        auto v = T::load(make_unaligned_view(expected));
        v.store(make_unaligned_view(actual));

        EXPECT_TRUE(std::equal(expected, expected + T::size, actual));
    }
    {
        auto v = T::load(make_unaligned_view(expected + 1));
        v.store(make_unaligned_view(actual + 1));

        EXPECT_TRUE(std::equal(expected, expected + 1 + T::size, actual));
    }
}

TEST(i32x4, load_store_unaligned) {
    test_32_load_store_unaligned<int32_t, simd::i32x4>();
}

TEST(i32x8, load_store_unaligned) {
    test_32_load_store_unaligned<int32_t, simd::i32x8>();
}

TEST(f32x4, load_store_unaligned) {
    test_32_load_store_unaligned<float, simd::f32x4>();
}

TEST(f32x8, load_store_unaligned) {
    test_32_load_store_unaligned<float, simd::f32x8>();
}

template <typename T>
void test_f32_multiply() {
    float input[]    = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
    float expected[] = {0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0};
    float actual[T::size];

    auto v = T::load(make_unaligned_view(input));
    (v * v).store(make_unaligned_view(actual));

    EXPECT_TRUE(std::equal(expected, expected + T::size, actual));
}

TEST(f32x4, multiply) {
    test_f32_multiply<simd::f32x4>();
}

TEST(f32x8, multiply) {
    test_f32_multiply<simd::f32x8>();
}

template <typename T>
void test_f32_equality() {
    float a[] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
    float b[] = {-0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};  // -0.0 == 0.0

    auto v1 = T::load(make_unaligned_view(a));
    auto v2 = v1;
    auto v3 = T::load(make_unaligned_view(b));

    EXPECT_EQ(v1, v2);
    EXPECT_EQ(v1, v3);
}

TEST(f32x4, equality) {
    test_f32_equality<simd::f32x4>();
}

TEST(f32x8, equality) {
    test_f32_equality<simd::f32x8>();
}

template <typename SourceT, typename T>
void test_32_hadd() {
    SourceT a[] = {0, 1, 2, 3, 4, 5, 6, 7};
    SourceT b[] = {8, 9, 10, 11, 12, 13, 14, 15};
    SourceT c[] = {1, 5, 17, 21, 9, 13, 25, 29};

    auto v1       = T::load(make_unaligned_view(a));
    auto v2       = T::load(make_unaligned_view(b));
    auto expected = T::load(make_unaligned_view(c));

    auto actual = simd::hadd(v1, v2);

    EXPECT_EQ(expected, actual);
}

TEST(i32x4, hadd) {
    test_32_hadd<int32_t, simd::i32x4>();
}

TEST(i32x8, hadd) {
    test_32_hadd<int32_t, simd::i32x8>();
}

TEST(f32x4, hadd) {
    test_32_hadd<float, simd::f32x4>();
}

TEST(f32x8, hadd) {
    test_32_hadd<float, simd::f32x8>();
}

template <typename T>
void test_permute4x64() {
    auto a        = T::from(0, 1, 2, 3, 4, 5, 6, 7);
    auto expected = T::from(0, 1, 4, 5, 6, 7, 2, 3);  // ordered [0, 2, 3, 1]

    auto actual = simd::permute4x64(a, simd::control4<0, 2, 3, 1>());

    EXPECT_EQ(expected, actual);
}

TEST(i32x8, permute4x64) {
    test_permute4x64<simd::i32x8>();
}

TEST(f32x8, permute4x64) {
    test_permute4x64<simd::f32x8>();
}
