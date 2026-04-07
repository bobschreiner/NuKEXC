#pragma once

#include "kokkos_config.hpp"
#include "molecule.hpp"
#include <iostream>

namespace NuKEXC {

KOKKOS_INLINE_FUNCTION
double dist(const Kokkos::View<double[3], exec_space> &a,
            const Kokkos::View<double[3], exec_space> &b) {
  double dist = 0;
  for (int i = 0; i < 3; ++i) {
    dist += std::pow(a[i] - b[i], 2);
  }
  dist = std::sqrt(dist);
  return dist;
}

void partition_becke(exec_space stream,
                     const Kokkos::View<double *[3], exec_space> &atom_centers,
                     const Kokkos::View<double **[3], exec_space> &quadrature_points,
                     Kokkos::View<double **, exec_space> &weights) {

  size_t natoms = atom_centers.extent(0);
  assert(natoms = quadrature_points.extent(0));
  size_t nquad_points_per_atom = quadrature_points.extent(1);

  Kokkos::View<double **, exec_space> R("Distance between atoms (R)", natoms, natoms);
  Kokkos::View<double ****, exec_space> mu("mu", natoms, nquad_points_per_atom, natoms,
                               natoms);
  Kokkos::View<double ****, exec_space> partition_polynomials(
      "partition polynomials", natoms, nquad_points_per_atom, natoms, natoms);

  Kokkos::View<double ***, exec_space> partition_weights("partition weights", natoms,
                                             nquad_points_per_atom, natoms);

  // Range policy for atomic distance calculations
  Kokkos::MDRangePolicy range_p2(stream, {0, 0}, {natoms, natoms});
  // Range policy for Voronoi polynomials
  Kokkos::MDRangePolicy range_p4(stream,
      {0, 0, 0, 0}, {natoms, nquad_points_per_atom, natoms, natoms});

  // Range policy for reduction to compute atomic weights
  Kokkos::MDRangePolicy range_p3(stream, {0, 0, 0},
                                 {natoms, nquad_points_per_atom, natoms});

  Kokkos::MDRangePolicy range_quad_points(stream, {0, 0},
                                          {natoms, nquad_points_per_atom});

  // Computes the atomic distances and stroes them in R_ij
  Kokkos::parallel_for(
      "Compute atomic distances", range_p2,
      KOKKOS_LAMBDA(const int &i, const int &j) {
        auto subView_i = Kokkos::subview(atom_centers, i, Kokkos::ALL);
        auto subView_j = Kokkos::subview(atom_centers, j, Kokkos::ALL);
        R(i, j) = dist(subView_i, subView_j);
      });

  // Computes the partition polynomials and stores them in partition_polynomials
  Kokkos::parallel_for(
      "Compute polynomials", range_p4,
      KOKKOS_LAMBDA(const int &p, const int &g, const int &i, const int &j) {
        if (i != j) {
          auto subView_g =
              Kokkos::subview(quadrature_points, p, g, Kokkos::ALL);
          auto subView_i = Kokkos::subview(atom_centers, i, Kokkos::ALL);
          auto subView_j = Kokkos::subview(atom_centers, j, Kokkos::ALL);

          double r_i = dist(subView_g, subView_i);
          double r_j = dist(subView_g, subView_j);

          mu(p, g, i, j) = (r_i - r_j) / R(i, j);

          if (mu(p, g, i, j) < -1.0)
            mu(p, g, i, j) = -1.0;
          else if (mu(p, g, i, j) > 1.0)
            mu(p, g, i, j) = 1.0;

          // Compute f_3(mu) = p(p(p(mu)))
          partition_polynomials(p, g, i, j) =
              1.5 * mu(p, g, i, j) - 0.5 * std::pow(mu(p, g, i, j), 3);
          partition_polynomials(p, g, i, j) =
              1.5 * partition_polynomials(p, g, i, j) -
              0.5 * std::pow(partition_polynomials(p, g, i, j), 3);
          partition_polynomials(p, g, i, j) =
              1.5 * partition_polynomials(p, g, i, j) -
              0.5 * std::pow(partition_polynomials(p, g, i, j), 3);

          // Compute polynomials s(mu) = 1/2 - (1-f_3(mu))
          partition_polynomials(p, g, i, j) =
              0.5 * (1.0 - partition_polynomials(p, g, i, j));
        }
      });

  // Computes the partition weights
  Kokkos::parallel_for(
      "Compute weights", range_p3,
      KOKKOS_LAMBDA(const int p, const int g, const int i) {
        partition_weights(p, g, i) = 1.0;
        for (int j = 0; j < natoms; ++j)
          if (i != j)
            partition_weights(p, g, i) *= partition_polynomials(p, g, i, j);
      });

  // Normalize the weights and update the weights paramter
  Kokkos::parallel_for(
      "Normalized weights", range_quad_points,
      KOKKOS_LAMBDA(const int &p, const int &g) {
        double w = 0;
        for (int i = 0; i < natoms; ++i) {
          w += partition_weights(p, g, i);
        }
        w = partition_weights(p, g, p) / w;
        weights(p, g) *= w;
      });
}
} // namespace NuKEXC
