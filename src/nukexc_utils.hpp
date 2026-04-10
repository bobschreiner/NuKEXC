#pragma once

#include <Kokkos_Core.hpp>

namespace NuKEXC {
namespace utils {
KOKKOS_INLINE_FUNCTION
double rad_dist(const Kokkos::View<double *, Kokkos::LayoutStride> &a,
                const Kokkos::View<double *, Kokkos::LayoutStride> &b) {
  double dist = 0;
  for (int i = 0; i < a.extent(0); ++i) {
    dist += std::pow(a(i) - b(i), 2);
  }
  dist = std::sqrt(dist);
  return dist;
}

} // namespace utils
} // namespace NuKEXC
