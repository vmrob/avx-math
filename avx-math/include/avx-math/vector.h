#pragma once

#pragma pack(push, 0)
template <typename T>
struct point {
    T x;
    T y;
};
#pragma pack(pop)

using point32f = point<float>;

// computes a.b and stores the output to out.
// dot(a, b) -> a.x * b.x + a.y * b.y
void dot_product_aligned_n(
        __attribute((aligned(32))) point32f* a,
        __attribute((aligned(32))) point32f* b,
        __attribute((aligned(32))) float*    out,
        unsigned long long                   n);

void dot_product_aligned_n_slow(
        __attribute((aligned(32))) point32f* a,
        __attribute((aligned(32))) point32f* b,
        __attribute((aligned(32))) float*    out,
        unsigned long long                   n);

