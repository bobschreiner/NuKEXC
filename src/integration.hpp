#pragma once

#include "Kokkos_config.hpp"

namespace NuKEXC {

template <typename F>
void quadrature(exec_space stream, double &energy,
                Kokkos::View<double *> quadrature_points,
                Kokkos::View<double *> weights, F functional) {
  size_t N = weights.extent(0);
  Kokkos::parallel_reduce(
      stream, "Quadrature", N,
      KOKKOS_LAMBDA(const int i, double &tmp) {
        tmp += weights[i] * functional(quadrature_points[i]);
      },
      energy);
}

} // namespace NuKEXC
