/*
 *    NuKEXC -- Numerical Kokkos Enhanced Exchange Correlation Integrator
 *    Copyright (C) 2026 Bob Schreiner
 *
 *    This program is free software: you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation, either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *
 *
 */

#pragma once

#include "kokkos_config.hpp"
#include "nukexc_utils.hpp"
// #include <iostream>

namespace NuKEXC {

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
    const Kokkos::View<double *[3], Layout, ExecSpace> &atom_centers,
    const Kokkos::View<double **[3], Layout, ExecSpace> &quadrature_points,
    Kokkos::View<double **, Layout, ExecSpace> &weights) {

  size_t natoms = atom_centers.extent(0);
  assert(natoms = quadrature_points.extent(0));
  size_t nquad_points_per_atom = quadrature_points.extent(1);

  Kokkos::View<double **> R("R", natoms, natoms);

  Kokkos::View<double ***> r("r", natoms, nquad_points_per_atom, natoms);

  Kokkos::MDRangePolicy range_quad_points({0, 0},
                                          {natoms, nquad_points_per_atom});
  Kokkos::MDRangePolicy range_quad_points_natoms(
      {0, 0, 0}, {natoms, nquad_points_per_atom, natoms});

  Kokkos::MDRangePolicy range_natoms_natoms({0, 0}, {natoms, natoms});

  // Computes the atomic distances and stroes them in R_ij
  Kokkos::parallel_for(
      "Compute inter-atomic distances", range_natoms_natoms,
      KOKKOS_LAMBDA(const int &i, const int &j) {
        auto subView_i = Kokkos::subview(atom_centers, i, Kokkos::ALL());
        auto subView_j = Kokkos::subview(atom_centers, j, Kokkos::ALL());
        R(i, j) = rad_dist(subView_i, subView_j);
      });

  Kokkos::fence();
  // Computes the distances from atom centers to quadrature points and stroes
  // them in r_pij
  Kokkos::parallel_for(
      "Compute atomic distances to quad_points", range_quad_points_natoms,
      KOKKOS_LAMBDA(const int &p, const int &g, const int &i) {
        auto subView_pg =
            Kokkos::subview(quadrature_points, p, g, Kokkos::ALL());
        auto subView_i = Kokkos::subview(atom_centers, i, Kokkos::ALL());
        r(p, g, i) = rad_dist(subView_pg, subView_i);
      });

  Kokkos::fence();
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
    const Kokkos::View<double *[3], Layout, ExecSpace> &atom_centers,
    const Kokkos::View<double **[3], Layout, ExecSpace> &quadrature_points,
    Kokkos::View<double **, Layout, ExecSpace> &weights) {

  size_t natoms = atom_centers.extent(0);
  size_t nquad_points_per_atom = quadrature_points.extent(1);

  Kokkos::View<double **, Layout, ExecSpace,
               Kokkos::MemoryTraits<Kokkos::RandomAccess>>
      R_ij("R_ij", natoms, natoms);

  Kokkos::parallel_for(
      "Precompute R_ij",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0},
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

  // Use L1 cache for large molecules, L0 cache of small molecules
  int cache_level = natoms > 100 ? 1 : 0;

  // Set the policy to use that amount "PerThread"
  auto policy =
      TeamPolicy(natoms, Kokkos::AUTO)
          .set_scratch_size(cache_level, Kokkos::PerThread(bytes_per_thread));

  Kokkos::parallel_for(
      "Becke Team Parallel", policy,
      KOKKOS_LAMBDA(const MemberType &team_member) {
        size_t p = team_member.league_rank(); // Each team handles one atom p

        // Scratch memory for distance caching per thread
        Kokkos::View<double *, ExecSpace::scratch_memory_space,
                     Kokkos::MemoryUnmanaged>
            r_cache(team_member.thread_scratch(cache_level), natoms);

        // Parallelize over the quadrature points 'g' within the team
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team_member, nquad_points_per_atom),
            [&](const size_t g) {
              // Cache distances for quadrature point g to all atoms i
              for (size_t i = 0; i < natoms; ++i) {
                auto subView_pg =
                    Kokkos::subview(quadrature_points, p, g, Kokkos::ALL());
                auto subView_i =
                    Kokkos::subview(atom_centers, i, Kokkos::ALL());
                r_cache(i) = rad_dist(subView_pg, subView_i);
              }

              double w_p;
              double normalization = 0.0;
              for (size_t i = 0; i < natoms; ++i) {
                double w_i = 1.0;
                for (size_t j = 0; j < natoms; ++j) {
                  if (i == j)
                    continue;

                  double mu = (r_cache(i) - r_cache(j)) / R_ij(i, j);
                  double poly = compute_p(compute_p(compute_p(mu)));
                  w_i *= compute_f(poly);
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
