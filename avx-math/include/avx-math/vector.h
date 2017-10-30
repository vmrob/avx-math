#pragma once

#include <cstdint>

namespace math {

#pragma pack(push, 0)
template <typename T>
struct vector {
    T x;
    T y;
};
#pragma pack(pop)

using vector32i = vector<int32_t>;
using vector32f = vector<float>;
using vector64i = vector<int64_t>;
using vector64f = vector<double>;

template <typename T>
__attribute((always_inline)) inline T dot_product(vector<T> a, vector<T> b) {
    return a.x * b.x + a.y * b.y;
}

template <typename T>
inline void
dot_product_n(vector<T>* a, vector<T>* b, T* out, unsigned long long int n) {
    for (unsigned long long int i = 0; i < n; ++i) {
        out[i] = dot_product(a[i], b[i]);
    }
}

void dot_product_n_aligned(
        __attribute((aligned(64))) vector32i* a,
        __attribute((aligned(64))) vector32i* b,
        __attribute((aligned(64))) int32_t*   out,
        unsigned long long int                n);

void dot_product_n_aligned(
        __attribute((aligned(64))) vector64i* a,
        __attribute((aligned(64))) vector64i* b,
        __attribute((aligned(64))) int64_t*   out,
        unsigned long long int                n);

void dot_product_n_aligned(
        __attribute((aligned(64))) vector32f* a,
        __attribute((aligned(64))) vector32f* b,
        __attribute((aligned(64))) float*     out,
        unsigned long long int                n);

void dot_product_n_aligned(
        __attribute((aligned(64))) vector64f* a,
        __attribute((aligned(64))) vector64f* b,
        __attribute((aligned(64))) double*    out,
        unsigned long long int                n);

}  // namespace math
