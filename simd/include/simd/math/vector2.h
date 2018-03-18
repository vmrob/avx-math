#pragma once

#include <cstdint>
#include <type_traits>

namespace simd::math {

#pragma pack(push, 0)
template <typename T>
struct vector2 {
    T x;
    T y;

    constexpr vector2<T>& operator+=(const vector2<T> rhs) {
        x += rhs.x;
        y += rhs.y;
        return *this;
    }

    constexpr vector2<T> operator+(const vector2<T> rhs) const {
        return {.x = x + rhs.x, .y = y + rhs.y};
    }

    constexpr vector2<T>& operator-=(const vector2<T> rhs) {
        x -= rhs.x;
        y -= rhs.y;
        return *this;
    }

    constexpr vector2<T> operator-(const vector2<T> rhs) const {
        return {.x = x - rhs.x, .y = y - rhs.y};
    }

    template <typename Scalar>
    constexpr vector2<T>& operator*=(Scalar s) {
        x *= s;
        y *= s;
        return *this;
    }

    template <typename Scalar>
    constexpr vector2<T>& operator/=(Scalar s) {
        x /= s;
        y /= s;
        return *this;
    }

    constexpr T dot(const vector2<T> other) const {
        return x * other.x + y * other.y;
    }
};
#pragma pack(pop)

using vector2i = vector2<int32_t>;
using vector2f = vector2<float>;
using vector2l = vector2<int64_t>;
using vector2d = vector2<double>;

template <typename T, typename Scalar>
constexpr vector2<T> operator*(const vector2<T> vec, Scalar s) {
    return {.x = T(vec.x * s), .y = T(vec.y * s)};
}

template <typename T, typename Scalar>
constexpr vector2<T> operator*(Scalar s, const vector2<T> vec) {
    return {.x = T(s * vec.x), .y = T(s * vec.y)};
}

template <typename T, typename Scalar>
constexpr vector2<T> operator/(const vector2<T> vec, Scalar s) {
    return {.x = T(vec.x / s), .y = T(vec.y / s)};
}

}  // namespace simd::math
