#pragma once

#include "kokkos_config.hpp"

// #include <iostream>

namespace NuKEXC {

KOKKOS_INLINE_FUNCTION
double dist(const Kokkos::View<double *, Kokkos::LayoutStride> &a,
            const Kokkos::View<double *, Kokkos::LayoutStride> &b) {
  double dist = 0;
  for (int i = 0; i < a.extent(0); ++i) {
    dist += std::pow(a(i) - b(i), 2);
  }
  dist = std::sqrt(dist);
  return dist;
}

KOKKOS_INLINE_FUNCTION
inline double compute_mu(const double r_i, const double r_j,
                         const double R_ij) {
  double mu = (r_i - r_j) / R_ij;

  if (mu < -1.0)
    mu = -1.0;
  else if (mu > 1.0)
    mu = 1.0;

  return mu;
}

KOKKOS_INLINE_FUNCTION
inline double compute_p(const double x) {
  return (1.5 * x) - (0.5 * std::pow(x, 3));
}

inline double compute_f(const double x) { return 0.5 * (1.0 - x); }

void partition_becke(
    ExecSpace stream,
    const Kokkos::View<double *[3], Layout, ExecSpace> &atom_centers,
    const Kokkos::View<double **[3], Layout, ExecSpace> &quadrature_points,
    Kokkos::View<double **, Layout, ExecSpace> &weights) {

  size_t natoms = atom_centers.extent(0);
  assert(natoms = quadrature_points.extent(0));
  size_t nquad_points_per_atom = quadrature_points.extent(1);

  Kokkos::View<double **, Layout, ExecSpace> R("Distance between atoms (R)",
                                               natoms, natoms);

  Kokkos::View<double ****, Layout, ExecSpace> partition_polynomials(
      "partition polynomials", natoms, nquad_points_per_atom, natoms, natoms);

  Kokkos::View<double ***, Layout, ExecSpace> partition_weights(
      "partition weights", natoms, nquad_points_per_atom, natoms);

  // Range policy for atomic distance calculations
  Kokkos::MDRangePolicy range_p2(stream, {0, 0}, {natoms, natoms});
  // Range policy for Voronoi polynomials
  Kokkos::MDRangePolicy range_p4(
      stream, {0, 0, 0, 0}, {natoms, nquad_points_per_atom, natoms, natoms});

  // Range policy for reduction to compute atomic weights
  Kokkos::MDRangePolicy range_p3(stream, {0, 0, 0},
                                 {natoms, nquad_points_per_atom, natoms});

  Kokkos::MDRangePolicy range_quad_points(stream, {0, 0},
                                          {natoms, nquad_points_per_atom});

  // Computes the atomic distances and stroes them in R_ij
  Kokkos::parallel_for(
      "Compute atomic distances", range_p2,
      KOKKOS_LAMBDA(const int &i, const int &j) {
        auto subView_i = Kokkos::subview(atom_centers, i, Kokkos::ALL());
        auto subView_j = Kokkos::subview(atom_centers, j, Kokkos::ALL());
        R(i, j) = dist(subView_i, subView_j);
      });

  // Computes the partition polynomials and stores them in partition_polynomials

  Kokkos::parallel_for(
      "Compute polynomials", range_p4,
      KOKKOS_LAMBDA(const int &p, const int &g, const int &i, const int &j) {
        if (i != j) {
          auto subView_g =
              Kokkos::subview(quadrature_points, p, g, Kokkos::ALL());
          auto subView_i = Kokkos::subview(atom_centers, i, Kokkos::ALL());
          auto subView_j = Kokkos::subview(atom_centers, j, Kokkos::ALL());

          double r_i = dist(subView_g, subView_i);
          double r_j = dist(subView_g, subView_j);

          double mu = (r_i - r_j) / R(i, j);

          if (mu < -1.0)
            mu = -1.0;
          else if (mu > 1.0)
            mu = 1.0;

          // Compute f_3(mu) = p(p(p(mu)))
          partition_polynomials(p, g, i, j) =
              compute_p(compute_p(compute_p(mu)));
          partition_polynomials(p, g, i, j) =
              compute_f(partition_polynomials(p, g, i, j));
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

void partition_becke_alt(
    ExecSpace stream,
    const Kokkos::View<double *[3], Layout, ExecSpace> &atom_centers,
    const Kokkos::View<double **[3], Layout, ExecSpace> &quadrature_points,
    Kokkos::View<double **, Layout, ExecSpace> &weights) {

  size_t natoms = atom_centers.extent(0);
  assert(natoms = quadrature_points.extent(0));
  size_t nquad_points_per_atom = quadrature_points.extent(1);

  Kokkos::View<double **> R("R", natoms, natoms);

  Kokkos::View<double ***> r("r", natoms, nquad_points_per_atom, natoms);

  Kokkos::MDRangePolicy range_quad_points(stream, {0, 0},
                                          {natoms, nquad_points_per_atom});
  Kokkos::MDRangePolicy range_quad_points_natoms(
      stream, {0, 0, 0}, {natoms, nquad_points_per_atom, natoms});

  Kokkos::MDRangePolicy range_natoms_natoms(stream, {0, 0}, {natoms, natoms});

  // Computes the atomic distances and stroes them in R_ij
  Kokkos::parallel_for(
      "Compute inter-atomic distances", range_natoms_natoms,
      KOKKOS_LAMBDA(const int &i, const int &j) {
        auto subView_i = Kokkos::subview(atom_centers, i, Kokkos::ALL());
        auto subView_j = Kokkos::subview(atom_centers, j, Kokkos::ALL());
        R(i, j) = dist(subView_i, subView_j);
      });

  // Computes the atomic distances and stroes them in R_ij
  Kokkos::parallel_for(
      "Compute atomic distances to quad_points", range_quad_points_natoms,
      KOKKOS_LAMBDA(const int &p, const int &g, const int &i) {
        auto subView_pg =
            Kokkos::subview(quadrature_points, p, g, Kokkos::ALL());
        auto subView_i = Kokkos::subview(atom_centers, i, Kokkos::ALL());
        r(p, g, i) = dist(subView_pg, subView_i);
      });

  Kokkos::parallel_for(
      "Compute weights", range_quad_points,
      KOKKOS_LAMBDA(const size_t p, const size_t g) {
        double w;
        double normalization = 0;
        for (size_t i = 0; i < natoms; ++i) {
          double part_weight = 1.;
          for (size_t j = 0; j < natoms; ++j) {
            if (i != j) {
              double mu = (r(p, g, i) - r(p, g, j)) / R(i, j);
              double poly = compute_p(compute_p(compute_p(mu)));
              double s = compute_f(poly);
              part_weight *= s;
            }
          }
          if (i == p)
            w = part_weight;
          normalization += part_weight;
        }
        weights(p, g) *= w / normalization;
      });
}
} // namespace NuKEXC
