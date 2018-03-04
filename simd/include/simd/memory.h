#pragma once

#include <cassert>
#include <memory>

namespace simd {

template <typename T>
T* aligned_alloc(size_t alignment, size_t size) {
    T* ptr;
    [[maybe_unused]] int result
            = posix_memalign(reinterpret_cast<void**>(&ptr), alignment, size);
    assert(result == 0);
    return ptr;
}

}  // namespace simd
