#pragma once

#include <cstdint>
#include <type_traits>

namespace simd::math {

#pragma pack(push, 0)
template <typename T>
struct vector2 {
    T x;
    T y;

    T dot(const vector2<T> other) { return x * other.x + y * other.y; }

    template <typename U>
    U as() {
        static_assert(std::is_pointer_v<U>);
        static_assert(std::is_trivial_v<std::remove_pointer_t<U>>);
        return reinterpret_cast<U>(this);
    }
};
#pragma pack(pop)

using vector2i = vector2<int32_t>;
using vector2f = vector2<float>;
using vector2l = vector2<int64_t>;
using vector2d = vector2<double>;

}  // namespace simd::math
