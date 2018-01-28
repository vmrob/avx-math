#include <simd/byte_view.h>

#include <gtest/gtest.h>

#include <type_traits>

TEST(aligned_byte_view, basics) {
    static_assert(!std::is_same<
                  aligned_byte_view<int, 4>,
                  aligned_byte_view<int, 8>>::value);

    __attribute((aligned(32))) int32_t arr[8];
    aligned_byte_view<int, 4>          ptr{arr};

    EXPECT_EQ(reinterpret_cast<size_t>(ptr.get()) % 4, 0);
    EXPECT_EQ(ptr.get(), arr);
}

TEST(unaligned_byte_view, basics) {
    static_assert(!std::is_same<
                  unaligned_byte_view<int>,
                  unaligned_byte_view<float>>::value);
    static_assert(
            !std::is_same<unaligned_byte_view<int>, aligned_byte_view<int, 1>>::
                    value);

    int32_t                  arr[8];
    unaligned_byte_view<int> ptr{arr};

    EXPECT_EQ(ptr.get(), arr);
}

TEST(aligned_byte_view, conversion) {
    __attribute((aligned(64))) int i1;
    __attribute((aligned(64))) int i2;

    aligned_byte_view<int, 32> b{&i1};
    aligned_byte_view<int, 64> c{&i2};

    EXPECT_NE(c.get(), b.get());
    b = c;
    EXPECT_EQ(c.get(), b.get());
}

TEST(unaligned_byte_view, assignment) {
    int i1;
    int i2;

    unaligned_byte_view<int> b{&i1};
    unaligned_byte_view<int> c{&i2};

    EXPECT_NE(c.get(), b.get());
    b = c;
    EXPECT_EQ(c.get(), b.get());
}

TEST(aligned_byte_view, make) {
    __attribute((aligned(64))) int i;

    auto ptr = make_aligned_byte_view<64>(&i);
    EXPECT_EQ(&i, ptr.get());
}

TEST(unaligned_byte_view, make) {
    int  i;
    auto ptr = make_unaligned_byte_view(&i);
    EXPECT_EQ(&i, ptr.get());
}

TEST(aligned_byte_view, dereference) {
    struct foo {
        int bar() const { return 0; }
    };

    {
        __attribute((aligned(64))) foo f;

        auto ptr = make_aligned_byte_view<64>(&f);
        EXPECT_EQ(0, ptr->bar());
    }
    {
        __attribute((aligned(64))) const foo f;

        auto ptr = make_aligned_byte_view<64>(&f);
        EXPECT_EQ(0, ptr->bar());
    }
    {
        __attribute((aligned(64))) int i = 0;

        auto ptr = make_aligned_byte_view<64>(&i);
        EXPECT_EQ(0, *ptr);
    }
    {
        __attribute((aligned(64))) const int i = 0;

        auto ptr = make_aligned_byte_view<64>(&i);
        EXPECT_EQ(0, *ptr);
    }
}

TEST(unaligned_byte_view, dereference) {
    struct foo {
        int bar() const { return 0; }
    };

    {
        foo  f;
        auto ptr = make_unaligned_byte_view(&f);
        EXPECT_EQ(0, ptr->bar());
    }
    {
        const foo f;
        auto      ptr = make_unaligned_byte_view(&f);
        EXPECT_EQ(0, ptr->bar());
    }
    {
        int  i   = 0;
        auto ptr = make_unaligned_byte_view(&i);
        EXPECT_EQ(0, *ptr);
    }
    {
        const int i   = 0;
        auto      ptr = make_unaligned_byte_view(&i);
        EXPECT_EQ(0, *ptr);
    }
}

TEST(aligned_byte_view, addition_subtraction) {
    // TODO: tests with types larger than alignment
    {
        __attribute((aligned(32))) int32_t i;

        auto ptr = make_aligned_byte_view<32>(&i);
        EXPECT_EQ(make_aligned_byte_view<32>(&i + 8).get(), (ptr + 1).get());
    }
    {
        __attribute((aligned(32))) int32_t i;

        const auto ptr = make_aligned_byte_view<32>(&i);
        EXPECT_EQ(make_aligned_byte_view<32>(&i + 8).get(), (ptr + 1).get());
    }
    {
        __attribute((aligned(64))) int32_t i;

        auto ptr = make_aligned_byte_view<64>(&i);
        EXPECT_EQ(make_aligned_byte_view<64>(&i + 16).get(), (ptr + 1).get());
    }
    {
        __attribute((aligned(64))) int32_t i;

        const auto ptr = make_aligned_byte_view<64>(&i);
        EXPECT_EQ(make_aligned_byte_view<64>(&i + 16).get(), (ptr + 1).get());
    }
}

TEST(unaligned_byte_view, addition_subtraction) {
    {
        int32_t i;
        auto    ptr = make_unaligned_byte_view(&i);
        EXPECT_EQ(make_unaligned_byte_view(&i + 8).get(), (ptr + 8).get());
    }
    {
        int32_t    i;
        const auto ptr = make_unaligned_byte_view(&i);
        EXPECT_EQ(make_unaligned_byte_view(&i + 8).get(), (ptr + 8).get());
    }
}
