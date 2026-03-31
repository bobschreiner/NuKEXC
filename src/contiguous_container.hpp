#pragma once
#include <array>

namespace NuKEXC {
namespace detail {

template <typename T, size_t N>
inline T *contiguous_data(const std::array<T, N> &arr) {
  return reinterpret_cast<T *>(&const_cast<std::array<T, N> &>(arr));
}

} // namespace detail
} // namespace NuKEXC
