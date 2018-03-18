#include <simd/view.h>

#include <gtest/gtest.h>

#include <type_traits>

TEST(aligned_view, basics) {
    static_assert(!std::is_same<
                  simd::aligned_view<int, 4>,
                  simd::aligned_view<int, 8>>::value);

    alignas(32) int32_t arr[8];
    simd::aligned_view<int, 4> ptr{arr};

    EXPECT_EQ(reinterpret_cast<size_t>(ptr.get()) % 4, 0);
    EXPECT_EQ(ptr.get(), arr);
}

TEST(unaligned_view, basics) {
    static_assert(!std::is_same<
                  simd::unaligned_view<int>,
                  simd::unaligned_view<float>>::value);
    static_assert(!std::is_same<
                  simd::unaligned_view<int>,
                  simd::aligned_view<int, 1>>::value);

    int32_t arr[8];
    simd::unaligned_view<int> ptr{arr};

    EXPECT_EQ(ptr.get(), arr);
}

TEST(aligned_view, conversion) {
    alignas(64) int i1;
    alignas(64) int i2;

    simd::aligned_view<int, 32> b{&i1};
    simd::aligned_view<int, 64> c{&i2};

    EXPECT_NE(c.get(), b.get());
    b = c;
    EXPECT_EQ(c.get(), b.get());
}

TEST(unaligned_view, assignment) {
    int i1;
    int i2;

    simd::unaligned_view<int> b{&i1};
    simd::unaligned_view<int> c{&i2};

    EXPECT_NE(c.get(), b.get());
    b = c;
    EXPECT_EQ(c.get(), b.get());
}

TEST(aligned_view, make) {
    alignas(64) int i;

    auto ptr = simd::as_aligned_view<64>(&i);
    EXPECT_EQ(&i, ptr.get());
}

TEST(unaligned_view, make) {
    int i;
    auto ptr = simd::as_unaligned_view(&i);
    EXPECT_EQ(&i, ptr.get());
}

TEST(aligned_view, dereference) {
    struct foo {
        int bar() const { return 0; }
    };

    {
        alignas(64) foo f;

        auto ptr = simd::as_aligned_view<64>(&f);
        EXPECT_EQ(0, ptr->bar());
    }
    {
        alignas(64) const foo f;

        auto ptr = simd::as_aligned_view<64>(&f);
        EXPECT_EQ(0, ptr->bar());
    }
    {
        alignas(64) int i = 0;

        auto ptr = simd::as_aligned_view<64>(&i);
        EXPECT_EQ(0, *ptr);
    }
    {
        alignas(64) const int i = 0;

        auto ptr = simd::as_aligned_view<64>(&i);
        EXPECT_EQ(0, *ptr);
    }
}

TEST(unaligned_view, dereference) {
    struct foo {
        int bar() const { return 0; }
    };

    {
        foo f;
        auto ptr = simd::as_unaligned_view(&f);
        EXPECT_EQ(0, ptr->bar());
    }
    {
        const foo f;
        auto ptr = simd::as_unaligned_view(&f);
        EXPECT_EQ(0, ptr->bar());
    }
    {
        int i    = 0;
        auto ptr = simd::as_unaligned_view(&i);
        EXPECT_EQ(0, *ptr);
    }
    {
        const int i = 0;
        auto ptr    = simd::as_unaligned_view(&i);
        EXPECT_EQ(0, *ptr);
    }
}

TEST(aligned_view, addition_subtraction) {
    // TODO: tests with types larger than alignment
    {
        alignas(32) int32_t i;

        auto ptr = simd::as_aligned_view<32>(&i);
        EXPECT_EQ(simd::as_aligned_view<32>(&i + 8).get(), (ptr + 1).get());
    }
    {
        alignas(32) int32_t i;

        const auto ptr = simd::as_aligned_view<32>(&i);
        EXPECT_EQ(simd::as_aligned_view<32>(&i + 8).get(), (ptr + 1).get());
    }
    {
        alignas(64) int32_t i;

        auto ptr = simd::as_aligned_view<64>(&i);
        EXPECT_EQ(simd::as_aligned_view<64>(&i + 16).get(), (ptr + 1).get());
    }
    {
        alignas(64) int32_t i;

        const auto ptr = simd::as_aligned_view<64>(&i);
        EXPECT_EQ(simd::as_aligned_view<64>(&i + 16).get(), (ptr + 1).get());
    }
}

TEST(unaligned_view, addition_subtraction) {
    {
        int32_t i;
        auto ptr = simd::as_unaligned_view(&i);
        EXPECT_EQ(simd::as_unaligned_view(&i + 8).get(), (ptr + 8).get());
    }
    {
        int32_t i;
        const auto ptr = simd::as_unaligned_view(&i);
        EXPECT_EQ(simd::as_unaligned_view(&i + 8).get(), (ptr + 8).get());
    }
}
