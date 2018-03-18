#pragma once

#include <cassert>
#include <type_traits>

namespace simd {

template <typename T, size_t Alignment>
struct aligned_view;

template <typename T>
struct unaligned_view {
    T* data;

    using value_type                  = T;
    static constexpr size_t alignment = 1;

    explicit operator T*() { return data; }
    explicit operator const T*() const { return data; }

    T& operator*() { return *data; }
    const T& operator*() const { return *data; }

    T* operator->() { return data; }
    const T* operator->() const { return data; }

    T& operator[](size_t i) { return data[i]; }
    const T& operator[](size_t i) const { return data[i]; }

    unaligned_view<T> operator+(long long int n) { return {data + n}; }
    const unaligned_view<T> operator+(long long int n) const {
        return {data + n};
    }
    unaligned_view<T>& operator+=(long long int n) {
        data += n;
        return *this;
    }
    unaligned_view<T>& operator-(long long int n) { return {data - n}; }
    const unaligned_view<T>& operator-(long long int n) const {
        return {data - n};
    }
    unaligned_view<T>& operator-=(long long int n) {
        data -= n;
        return *this;
    }

    T* get() { return data; }
    const T* get() const { return data; }

    template <typename U>
    unaligned_view<U> as() {
        static_assert(std::is_trivial_v<T>);
        static_assert(std::is_trivial_v<U>);
        return unaligned_view<U>{reinterpret_cast<U*>(data)};
    }

    template <typename U>
    unaligned_view<U> as() const {
        static_assert(std::is_trivial_v<U>);
        static_assert(std::is_trivial_v<T>);
        return unaligned_view<const U>{reinterpret_cast<U*>(data)};
    }

    explicit operator aligned_view<T, 1>() { return {data}; }
    explicit operator aligned_view<const T, 1>() const { return {data}; }
};

template <typename T, size_t Alignment>
struct aligned_view {
    using aligned_type = __attribute((align_value(Alignment))) T*;
    aligned_type data;

    static_assert(
            Alignment > 0 && (Alignment & (Alignment - 1)) == 0,
            "Alignment must be a positive power of 2");
    static_assert(
            sizeof(T) > 0 && (sizeof(T) & (sizeof(T) - 1)) == 0,
            "sizeof(T) must be a positive power of 2");

    using value_type                  = T;
    static constexpr size_t alignment = Alignment;
    static constexpr size_t size
            = Alignment > sizeof(T) ? Alignment / sizeof(T) : 1;

    explicit operator T*() { return data; }
    explicit operator const T*() const { return data; }

    T& operator*() { return *data; }
    const T& operator*() const { return *data; }

    T* operator->() { return data; }
    const T* operator->() const { return data; }

    T& operator[](size_t i) { return data[i]; }
    const T& operator[](size_t i) const { return data[i]; }

    aligned_view<T, Alignment> operator+(long long int n) {
        return {data + n * size};
    }
    const aligned_view<T, Alignment> operator+(long long int n) const {
        return {data + n * size};
    }
    aligned_view<T, Alignment>& operator+=(long long int n) {
        data += n * size;
        return *this;
    }
    aligned_view<T, Alignment>& operator-(long long int n) {
        return {data - n * size};
    }
    const aligned_view<T, Alignment>& operator-(long long int n) const {
        return {data - n * size};
    }
    aligned_view<T, Alignment>& operator-=(long long int n) {
        data -= n * size;
        return *this;
    }
    template <typename RhsT, size_t RhsAlignment>
    bool operator==(const aligned_view<RhsT, RhsAlignment>& rhs) const {
        return data == rhs.data;
    }
    template <typename RhsT, size_t RhsAlignment>
    bool operator!=(const aligned_view<RhsT, RhsAlignment>& rhs) const {
        return data != rhs.data;
    }

    T* get() { return data; }
    const T* get() const { return data; }

    template <size_t NewAlignment>
    operator aligned_view<T, NewAlignment>() {
        static_assert(
                Alignment > NewAlignment,
                "invalid conversion to more restrictive alignment");
        return aligned_view<T, NewAlignment>{data};
    }

    template <size_t NewAlignment>
    operator aligned_view<const T, NewAlignment>() const {
        static_assert(
                Alignment > NewAlignment,
                "invalid conversion to more restrictive alignment");
        return aligned_view<const T, NewAlignment>{data};
    }

    explicit operator unaligned_view<T>() { return {data}; }
    explicit operator unaligned_view<const T>() const { return {data}; }

    template <typename U>
    aligned_view<U, Alignment> as() {
        static_assert(std::is_trivial_v<T>);
        static_assert(std::is_trivial_v<U>);
        return aligned_view<U, Alignment>{reinterpret_cast<U*>(data)};
    }

    template <typename U>
    aligned_view<U, Alignment> as() const {
        static_assert(std::is_trivial_v<U>);
        static_assert(std::is_trivial_v<T>);
        return aligned_view<const U, Alignment>{reinterpret_cast<U*>(data)};
    }
};

template <size_t Alignment, typename T>
aligned_view<T, Alignment> as_aligned_view(T* ptr) {
    assert(*reinterpret_cast<size_t*>(&ptr) % Alignment == 0);
    return {ptr};
}

template <typename T>
unaligned_view<T> as_unaligned_view(T* ptr) {
    return {ptr};
}

}  // namespace simd
