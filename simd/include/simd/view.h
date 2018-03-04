#pragma once

#include <cassert>
#include <type_traits>

/// \file byte_view.h
/// This file provides classes and functions for zero-overhead pointer wrappers
/// used to convey alignment requirements in a type-safe manner.
///
/// No checking on input pointers is performed, but once in this form,
/// compile-time assertions for type conversions are possible. Pointer
/// arithmatic for unaligned classes occurs as you would expect, but for aligned
/// classes, Alignment must be a multiple of the size or the size a multiple of
/// the alignment.
///
/// ~~~cpp
/// aligned_view<int32_t, 32> view{ptr};
/// assert((view + 1).get() == ptr + 8);
/// static_assert(decltype(view)::size == 8);
/// ~~~
///
/// Without this behavior, view + 1 would result in an unaligned pointer.
///
/// For all intents and purposes, unaligned_view<T> is identical to
/// aligned_view<T, 1>, but exists for overload resolution.

template <typename T>
struct unaligned_view {
    T* data;

    using value_type                  = T;
    static constexpr size_t alignment = 1;

    explicit operator T*() { return data; }
    explicit operator const T*() const { return data; }

    T&       operator*() { return *data; }
    const T& operator*() const { return *data; }
    T*       operator->() { return data; }
    const T* operator->() const { return data; }
    T&       operator[](size_t i) { return data[i]; }
    const T& operator[](size_t i) const { return data[i]; }

    unaligned_view<T>       operator+(long long int n) { return {data + n}; }
    const unaligned_view<T> operator+(long long int n) const {
        return {data + n};
    }
    unaligned_view<T>& operator+=(long long int n) {
        data += n;
        return *this;
    }
    unaligned_view<T>&       operator-(long long int n) { return {data - n}; }
    const unaligned_view<T>& operator-(long long int n) const {
        return {data - n};
    }
    unaligned_view<T>& operator-=(long long int n) {
        data -= n;
        return *this;
    }

    T*       get() { return data; }
    const T* get() const { return data; }
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

    T&       operator*() { return *data; }
    const T& operator*() const { return *data; }
    T*       operator->() { return data; }
    const T* operator->() const { return data; }
    T&       operator[](size_t i) { return data[i]; }
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

    T*       get() { return data; }
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
};

template <size_t Alignment, typename T>
aligned_view<T, Alignment> make_aligned_view(T* ptr) {
    assert(*reinterpret_cast<size_t*>(&ptr) % Alignment == 0);
    return {ptr};
}

template <typename T>
unaligned_view<T> make_unaligned_view(T* ptr) {
    return {ptr};
}
