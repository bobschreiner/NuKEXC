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

#include "nukexc_config.hpp"
#include "nukexc_utils.hpp"
#include <iostream>

namespace Nukexc {

KOKKOS_INLINE_FUNCTION
double compute_mu(const double r_i, const double r_j, const double R_ij) {
  double mu = (r_i - r_j) / R_ij;
  mu = mu > 1.0 ? 1.0 : mu < -1.0 ? -1.0 : mu;
  return mu;
}

KOKKOS_INLINE_FUNCTION
double compute_mu_laqua(const double r_i, const double r_j, const double R_ij,
                        const double R_cutoff = 5.) {
  double R = Kokkos::min(R_ij, R_cutoff);
  double mu = (r_i - r_j) / R;
  mu = mu > 1.0 ? 1.0 : mu < -1.0 ? -1.0 : mu;
  return mu;
}
KOKKOS_INLINE_FUNCTION
double compute_p(const double x) { return (1.5 * x) - (0.5 * std::pow(x, 3)); }

KOKKOS_INLINE_FUNCTION
double compute_s(const double f) { return 0.5 * (1.0 - f); }

void partition_becke(const Kokkos::View<Point *> &atom_centers,
                     const Kokkos::View<Point **> &quadrature_points,
                     Kokkos::View<double **> &weights) {

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
        R(i, j) = dist(atom_centers(i), atom_centers(j)) + epsilon_shift;
      });

  Kokkos::fence();
  // Computes the distances from atom centers to quadrature points and stroes
  // them in r_pij
  Kokkos::parallel_for(
      "Compute atomic distances to quad_points", range_quad_points_natoms,
      KOKKOS_LAMBDA(const int &p, const int &g, const int &i) {
        auto subView_pg = r(p, g, i) =
            dist(quadrature_points(p, g), atom_centers(i));
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
              double mu = compute_mu_laqua(r(p, g, i), r(p, g, j), R(i, j));
              double poly = compute_p(compute_p(compute_p(mu)));

              double s = compute_s(poly);
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

void partition_becke_team(const Kokkos::View<Point *> &atom_centers,
                          const Kokkos::View<Point **> &quadrature_points,
                          Kokkos::View<double **> &weights) {

  size_t natoms = atom_centers.extent(0);
  size_t nquad_points_per_atom = quadrature_points.extent(1);

  Kokkos::View<double **, Layout, ExecSpace,
               Kokkos::MemoryTraits<Kokkos::RandomAccess>>
      R_ij("R_ij", natoms, natoms);

  // Precompute: for each atom i, which atoms j are "close enough" to matter?
  // Threshold: if R_ij > R_screen, s(p(p(p(mu)))) is indistinguishable from 1
  // A value of ~8-10 bohr is typically sufficient for Becke + Laqua screening
  //
  // neighbor_list(i, k) = index of k-th neighbor of atom i
  // n_neighbors(i)      = number of neighbors of atom i
  Kokkos::View<int *> n_neighbors("n_neighbors", natoms);

  const double R_screen = 10.0;

  // --- Pass 1: count neighbors only ---
  Kokkos::parallel_for(
      "Precompute R_ij", natoms, KOKKOS_LAMBDA(const int i) {
        for (int j = 0; j < natoms; ++j) {
          if (i == j)
            continue;
          double d = dist(atom_centers(i), atom_centers(j));
          R_ij(i, j) = d + epsilon_shift;
          if (R_ij(i, j) < R_screen) {
            n_neighbors(i) += 1;
          }
        }
      });

  // --- Reduce to find max_n, then allocate with correct size ---
  int max_n = 0;
  Kokkos::parallel_reduce(
      "Find maximum number of neighbors", natoms,
      KOKKOS_LAMBDA(const int &i, int &lmax) {
        if (lmax < n_neighbors(i))
          lmax = n_neighbors(i);
      },
      Kokkos::Max<int>(max_n));

  Kokkos::View<int **> neighbor_list("neighbors", natoms, max_n);
  Kokkos::deep_copy(neighbor_list, -1);

  // Reset counters to reuse as fill indices in pass 2
  Kokkos::deep_copy(n_neighbors, 0);

  // --- Pass 2: fill neighbor list ---
  Kokkos::parallel_for(
      "Fill neighbor list", natoms, KOKKOS_LAMBDA(const int i) {
        for (int j = 0; j < natoms; ++j) {
          if (i == j)
            continue;
          if (R_ij(i, j) < R_screen) {
            int idx = n_neighbors(i);
            neighbor_list(i, idx) = j;
            n_neighbors(i) += 1;
          }
        }
      });

  using scratch_view_double =
      Kokkos::View<double *, ExecSpace::scratch_memory_space>;

  const size_t bytes_per_team = 2 * scratch_view_double::shmem_size(natoms);

  // Use L1 cache for large molecules, L0 cache of small molecules
  const int cache_level = 0;

  // Set the policy to use that amount "PerThread"
  auto policy =
      TeamPolicy(natoms * nquad_points_per_atom, Kokkos::AUTO)
          .set_scratch_size(cache_level, Kokkos::PerTeam(bytes_per_team));

  Kokkos::parallel_for(
      "Becke Team Parallel", policy,
      KOKKOS_LAMBDA(const MemberType &team_member) {
        int pg = team_member.league_rank();
        int p = pg / nquad_points_per_atom;
        int g = pg % nquad_points_per_atom;

        // PerTeam shared scratch — one r_cache and w_cache per quadrature point
        scratch_view_double r_cache(team_member.team_scratch(cache_level),
                                    natoms);

        scratch_view_double w_cache(team_member.team_scratch(cache_level),
                                    natoms);

        Point pt = quadrature_points(p, g);

        // Phase 1: parallel distance cache over atoms
        Kokkos::parallel_for(
            Kokkos::TeamVectorRange(team_member, natoms),
            [=](const int i) { r_cache(i) = dist(pt, atom_centers(i)); });
        team_member.team_barrier();

        // Phase 2: parallel w_i computation
        Kokkos::parallel_for(
            Kokkos::TeamVectorRange(team_member, natoms), [=](const int i) {
              double w_i = 1.0;
              for (int k = 0; k < n_neighbors(i); ++k) {
                int j = neighbor_list(i, k);
                double mu =
                    compute_mu_laqua(r_cache(i), r_cache(j), R_ij(i, j));
                w_i *= compute_s(compute_p(compute_p(compute_p(mu))));
              }
              w_cache(i) = w_i;
            });
        team_member.team_barrier();

        // Phase 3: parallel reduction for normalization
        double normalization = 0.0;
        Kokkos::parallel_reduce(
            Kokkos::TeamVectorRange(team_member, natoms),
            [=](const int i, double &lsum) { lsum += w_cache(i); },
            normalization);

        // Single thread applies final weight
        Kokkos::single(Kokkos::PerTeam(team_member),
                       [=]() { weights(p, g) *= w_cache(p) / normalization; });
      });
}
} // namespace Nukexc
