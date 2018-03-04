#include <simd/view.h>

#include <gtest/gtest.h>

#include <type_traits>

TEST(aligned_view, basics) {
    static_assert(
            !std::is_same<aligned_view<int, 4>, aligned_view<int, 8>>::value);

    __attribute((aligned(32))) int32_t arr[8];
    aligned_view<int, 4>               ptr{arr};

    EXPECT_EQ(reinterpret_cast<size_t>(ptr.get()) % 4, 0);
    EXPECT_EQ(ptr.get(), arr);
}

TEST(unaligned_view, basics) {
    static_assert(
            !std::is_same<unaligned_view<int>, unaligned_view<float>>::value);
    static_assert(
            !std::is_same<unaligned_view<int>, aligned_view<int, 1>>::value);

    int32_t             arr[8];
    unaligned_view<int> ptr{arr};

    EXPECT_EQ(ptr.get(), arr);
}

TEST(aligned_view, conversion) {
    __attribute((aligned(64))) int i1;
    __attribute((aligned(64))) int i2;

    aligned_view<int, 32> b{&i1};
    aligned_view<int, 64> c{&i2};

    EXPECT_NE(c.get(), b.get());
    b = c;
    EXPECT_EQ(c.get(), b.get());
}

TEST(unaligned_view, assignment) {
    int i1;
    int i2;

    unaligned_view<int> b{&i1};
    unaligned_view<int> c{&i2};

    EXPECT_NE(c.get(), b.get());
    b = c;
    EXPECT_EQ(c.get(), b.get());
}

TEST(aligned_view, make) {
    __attribute((aligned(64))) int i;

    auto ptr = as_aligned_view<64>(&i);
    EXPECT_EQ(&i, ptr.get());
}

TEST(unaligned_view, make) {
    int  i;
    auto ptr = as_unaligned_view(&i);
    EXPECT_EQ(&i, ptr.get());
}

TEST(aligned_view, dereference) {
    struct foo {
        int bar() const { return 0; }
    };

    {
        __attribute((aligned(64))) foo f;

        auto ptr = as_aligned_view<64>(&f);
        EXPECT_EQ(0, ptr->bar());
    }
    {
        __attribute((aligned(64))) const foo f;

        auto ptr = as_aligned_view<64>(&f);
        EXPECT_EQ(0, ptr->bar());
    }
    {
        __attribute((aligned(64))) int i = 0;

        auto ptr = as_aligned_view<64>(&i);
        EXPECT_EQ(0, *ptr);
    }
    {
        __attribute((aligned(64))) const int i = 0;

        auto ptr = as_aligned_view<64>(&i);
        EXPECT_EQ(0, *ptr);
    }
}

TEST(unaligned_view, dereference) {
    struct foo {
        int bar() const { return 0; }
    };

    {
        foo  f;
        auto ptr = as_unaligned_view(&f);
        EXPECT_EQ(0, ptr->bar());
    }
    {
        const foo f;
        auto      ptr = as_unaligned_view(&f);
        EXPECT_EQ(0, ptr->bar());
    }
    {
        int  i   = 0;
        auto ptr = as_unaligned_view(&i);
        EXPECT_EQ(0, *ptr);
    }
    {
        const int i   = 0;
        auto      ptr = as_unaligned_view(&i);
        EXPECT_EQ(0, *ptr);
    }
}

TEST(aligned_view, addition_subtraction) {
    // TODO: tests with types larger than alignment
    {
        __attribute((aligned(32))) int32_t i;

        auto ptr = as_aligned_view<32>(&i);
        EXPECT_EQ(as_aligned_view<32>(&i + 8).get(), (ptr + 1).get());
    }
    {
        __attribute((aligned(32))) int32_t i;

        const auto ptr = as_aligned_view<32>(&i);
        EXPECT_EQ(as_aligned_view<32>(&i + 8).get(), (ptr + 1).get());
    }
    {
        __attribute((aligned(64))) int32_t i;

        auto ptr = as_aligned_view<64>(&i);
        EXPECT_EQ(as_aligned_view<64>(&i + 16).get(), (ptr + 1).get());
    }
    {
        __attribute((aligned(64))) int32_t i;

        const auto ptr = as_aligned_view<64>(&i);
        EXPECT_EQ(as_aligned_view<64>(&i + 16).get(), (ptr + 1).get());
    }
}

TEST(unaligned_view, addition_subtraction) {
    {
        int32_t i;
        auto    ptr = as_unaligned_view(&i);
        EXPECT_EQ(as_unaligned_view(&i + 8).get(), (ptr + 8).get());
    }
    {
        int32_t    i;
        const auto ptr = as_unaligned_view(&i);
        EXPECT_EQ(as_unaligned_view(&i + 8).get(), (ptr + 8).get());
    }
}
