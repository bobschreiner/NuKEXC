/*
 *    NuKEXC -- Numerical Kokkos Enhanced Exchange Correlation Integrator
 *    Copyright (c) 2026, Bob Schreiner
 *    All rights reserved.
 *
 *    SPDX-License-Identifier: BSD-3-Clause
 *
 *    Redistribution and use in source and binary forms, with or without
 *    modification, are permitted provided that the following conditions are
 *    met:
 *
 *    1. Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *
 *    2. Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *
 *    3. Neither the name of the copyright holder nor the names of its
 *       contributors may be used to endorse or promote products derived from
 *       this software without specific prior written permission.
 *
 *    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 *    A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 *    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 *    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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
double compute_p(const double x) { return (1.5 * x) - (0.5 * std::pow(x, 3)); }

KOKKOS_INLINE_FUNCTION
double compute_s(const double f) { return 0.5 * (1.0 - f); }

// Becke partitioning on a flat (possibly irregular) grid.
//
// The grid is stored as flat arrays of length `total_points`. `point_owner(g)`
// is the index of the atom that owns quadrature point `g` -- this replaces the
// old convention where the point's owning atom was implied by a 2D row index,
// and is what lets each center carry a different number of quadrature points.
void partition_becke(const Kokkos::View<Point *> &atom_centers,
                     const Kokkos::View<Point *> &quadrature_points,
                     const Kokkos::View<int *> &point_owner,
                     Kokkos::View<double *> &weights) {

  size_t natoms = atom_centers.extent(0);
  size_t total_points = quadrature_points.extent(0);

  Kokkos::View<double **> R("R", natoms, natoms);

  Kokkos::View<double **> r("r", total_points, natoms);

  Kokkos::MDRangePolicy range_points_natoms({0, 0}, {total_points, natoms});

  Kokkos::MDRangePolicy range_natoms_natoms({0, 0}, {natoms, natoms});

  // Computes the atomic distances and stroes them in R_ij
  Kokkos::parallel_for(
      "Compute inter-atomic distances", range_natoms_natoms,
      KOKKOS_LAMBDA(const int &i, const int &j) {
        R(i, j) = dist(atom_centers(i), atom_centers(j));
      });

  Kokkos::fence();
  // Computes the distances from each quadrature point to every atom center and
  // stores them in r
  Kokkos::parallel_for(
      "Compute atomic distances to quad_points", range_points_natoms,
      KOKKOS_LAMBDA(const int &g, const int &i) {
        r(g, i) = dist(quadrature_points(g), atom_centers(i));
      });

  Kokkos::fence();

  Kokkos::parallel_for(
      "Compute weights batched", Kokkos::RangePolicy<ExecSpace>(0, total_points),
      KOKKOS_LAMBDA(const size_t g) {
        const int owner = point_owner(g);
        double w = 0;
        double normalization = 0;
        for (size_t i = 0; i < natoms; ++i) {
          double part_weight = 1.;
          for (size_t j = 0; j < natoms; ++j) {
            if (i != j) {
              double mu = compute_mu(r(g, i), r(g, j), R(i, j));
              double poly = compute_p(compute_p(compute_p(mu)));

              double s = compute_s(poly);
              part_weight *= s;
            }
          }
          if (i == owner)
            w = part_weight;
          normalization += part_weight;
        }
        weights(g) *= w / normalization;
      });
}

// Define a shorthand for the TeamPolicy
using TeamPolicy = Kokkos::TeamPolicy<ExecSpace>;
using MemberType = typename TeamPolicy::member_type;

void partition_becke_team(const Kokkos::View<Point *> &atom_centers,
                          const Kokkos::View<Point *> &quadrature_points,
                          const Kokkos::View<int *> &point_owner,
                          Kokkos::View<double *> &weights) {

  size_t natoms = atom_centers.extent(0);
  size_t total_points = quadrature_points.extent(0);

  const double R_cutoff = 5.0;

  Kokkos::View<double **, Layout, ExecSpace,
               Kokkos::MemoryTraits<Kokkos::RandomAccess>>
      R_ij("R_ij", natoms, natoms);

  Kokkos::MDRangePolicy range_natoms_natoms({0, 0}, {natoms, natoms});

  // Computes the atomic distances and stroes them in R_ij
  Kokkos::parallel_for(
      "Compute inter-atomic distances", range_natoms_natoms,
      KOKKOS_LAMBDA(const int &i, const int &j) {
        R_ij(i, j) =
            Kokkos::min(dist(atom_centers(i), atom_centers(j)), R_cutoff);
      });

  // Two PerTeam scratch caches (distances and partial weights) sized to natoms.
  using scratch_view_double =
      Kokkos::View<double *, ExecSpace::scratch_memory_space>;

  const size_t bytes_per_team = 2 * scratch_view_double::shmem_size(natoms);

  // Use L1 cache for large molecules, L0 cache of small molecules
  const int cache_level = 0;

  // Set the policy to use that amount "PerThread" -- one team per quadrature
  // point, so the league spans all points across all (possibly unequally sized)
  // atomic grids.
  auto policy =
      TeamPolicy(total_points, Kokkos::AUTO)
          .set_scratch_size(cache_level, Kokkos::PerTeam(bytes_per_team));

  Kokkos::parallel_for(
      "Becke Team Parallel", policy,
      KOKKOS_LAMBDA(const MemberType &team_member) {
        int g = team_member.league_rank(); // flat quadrature-point index
        int owner = point_owner(g);        // atom that owns this point

        // PerTeam shared scratch — one r_cache and w_cache per quadrature point
        scratch_view_double r_cache(team_member.team_scratch(cache_level),
                                    natoms);

        scratch_view_double w_cache(team_member.team_scratch(cache_level),
                                    natoms);

        Point pt = quadrature_points(g);

        // Phase 1: parallel distance cache over atoms
        Kokkos::parallel_for(
            Kokkos::TeamVectorRange(team_member, natoms),
            [=](const int i) { r_cache(i) = dist(pt, atom_centers(i)); });
        team_member.team_barrier();

        // Phase 2: parallel w_i computation
        Kokkos::parallel_for(
            Kokkos::TeamVectorRange(team_member, natoms), [=](const int i) {
              double w_i = 1.0;
              for (int j = 0; j < natoms; ++j) {
                if (i == j)
                  continue;
                double mu = compute_mu(r_cache(i), r_cache(j), R_ij(i, j));
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
        Kokkos::single(
            Kokkos::PerTeam(team_member),
            [=]() { weights(g) *= w_cache(owner) / normalization; });
      });
}
} // namespace Nukexc
