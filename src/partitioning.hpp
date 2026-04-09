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
double compute_mu(const double r_i, const double r_j, const double R_ij) {
  double mu = (r_i - r_j) / R_ij;

  if (mu < -1.0)
    mu = -1.0;
  else if (mu > 1.0)
    mu = 1.0;

  return mu;
}

KOKKOS_INLINE_FUNCTION
double compute_p(const double x) { return (1.5 * x) - (0.5 * std::pow(x, 3)); }

KOKKOS_INLINE_FUNCTION
double compute_f(const double x) { return 0.5 * (1.0 - x); }

void partition_becke(
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
      "Compute weights batched", range_quad_points,
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

// Define a shorthand for the TeamPolicy
using TeamPolicy = Kokkos::TeamPolicy<ExecSpace>;
using MemberType = typename TeamPolicy::member_type;

void partition_becke_team(
    ExecSpace stream,
    const Kokkos::View<double *[3], Layout, ExecSpace> &atom_centers,
    const Kokkos::View<double **[3], Layout, ExecSpace> &quadrature_points,
    Kokkos::View<double **, Layout, ExecSpace> &weights) {

  size_t natoms = atom_centers.extent(0);
  size_t nquad_points_per_atom = quadrature_points.extent(1);

  Kokkos::View<double **, Layout, ExecSpace> R_ij("R_ij", natoms, natoms);
  Kokkos::parallel_for(
      "Precompute R_ij",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>(stream, {0, 0},
                                                        {natoms, natoms}),
      KOKKOS_LAMBDA(const int i, const int j) {
        if (i != j) {
          double d2 = 0;
          for (int k = 0; k < 3; ++k) {
            double d = atom_centers(i, k) - atom_centers(j, k);
            d2 += d * d;
          }
          R_ij(i, j) = sqrt(d2);
        }
      });

  size_t bytes_per_thread =
      Kokkos::View<double *, ExecSpace::scratch_memory_space>::shmem_size(
          natoms);

  // Set the policy to use that amount "PerThread"
  auto policy = TeamPolicy(stream, natoms, Kokkos::AUTO)
                    .set_scratch_size(0, Kokkos::PerThread(bytes_per_thread));

  Kokkos::parallel_for(
      "Becke Team Parallel", policy,
      KOKKOS_LAMBDA(const MemberType &team_member) {
        size_t p = team_member.league_rank(); // Each team handles one atom p

        // Scratch memory for distance caching per thread
        Kokkos::View<double *, ExecSpace::scratch_memory_space,
                     Kokkos::MemoryUnmanaged>
            r_cache(team_member.thread_scratch(0), natoms);

        // Parallelize over the quadrature points 'g' within the team
        Kokkos::parallel_for(
            Kokkos::TeamVectorRange(team_member, nquad_points_per_atom),
            [&](const size_t g) {
              // Cache distances for this point g to all atoms i
              for (size_t i = 0; i < natoms; ++i) {
                double d2 = 0;
                for (int k = 0; k < 3; ++k) {
                  double d = quadrature_points(p, g, k) - atom_centers(i, k);
                  d2 += d * d;
                }
                r_cache(i) = sqrt(d2);
              }

              double w_p = 0.0;
              double normalization = 0.0;

              for (size_t i = 0; i < natoms; ++i) {
                double w_i = 1.0;
                for (size_t j = 0; j < natoms; ++j) {
                  if (i == j)
                    continue;

                  double mu = (r_cache(i) - r_cache(j)) / R_ij(i, j);
                  double poly = compute_p(compute_p(compute_p(mu)));
                  w_i *= 0.5 * (1.0 - compute_f(poly));
                }

                if (i == p)
                  w_p = w_i;
                normalization += w_i;
              }

              weights(p, g) *= (w_p / normalization);
            });
      });
}
} // namespace NuKEXC
