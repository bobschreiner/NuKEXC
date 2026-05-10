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
 */

#pragma once

#include <ArborX.hpp>

#include "grid.hpp"
#include "molecule.hpp"
#include "nukexc_config.hpp"
#include "nukexc_utils.hpp"
#include "stobasis.hpp"

#include <detail/ArborX_SpaceFillingCurves.hpp>
#include <detail/ArborX_TreeVisualization.hpp>
#include <fstream>
#include <integratorxx/quadratures/s2/lebedev_laikov.hpp>
#include <iostream>

namespace NuKEXC {

using Point = ArborX::Point<3, double>;
using Box = ArborX::Box<3, double>;

Kokkos::View<Box *, ExecSpace>
create_bounding_boxes(FlatGrid grid, const int max_points_per_bb) {

  const int num_points = grid.quad_points.extent(0);

  // ── Copy grid coordinates into ArborX points ─────────────────────────
  // ── Wrap each point as a degenerate box for BVH construction ─────────
  Kokkos::View<Box *, ExecSpace> boxes("boxes", num_points);
  Kokkos::parallel_for(
      "copy_points_to_boxes", Kokkos::RangePolicy<ExecSpace>(0, num_points),
      KOKKOS_LAMBDA(int i) {
        boxes(i) = {grid.quad_points(i), grid.quad_points(i)};
      });

  // ── Build BVH over per-point boxes ───────────────────────────────────
  ArborX::BoundingVolumeHierarchy bvh{
      ExecSpace{}, ArborX::Experimental::attach_indices(boxes)};

  // ── Compute Morton-order permutation of the grid points ───────────────
  // We construct Nearest queries only to give ArborX a view with the
  // right geometry type; the queries are never actually executed.
  // n_neighbors = 1 is sufficient — only the query geometry is used.
  const int n_queries = bvh.size();
  Kokkos::View<ArborX::Nearest<Point> *, ExecSpace> queries("queries",
                                                            n_queries);
  Kokkos::parallel_for(
      "initialize_queries", Kokkos::RangePolicy(ExecSpace{}, 0, n_queries),
      KOKKOS_LAMBDA(int i) {
        queries(i) = ArborX::nearest(grid.quad_points(i), 1);
      });

  auto permute = ArborX::Details::computeSpaceFillingCurvePermutation(
      ExecSpace{},
      ArborX::Details::PredicateIndexables<decltype(queries)>{queries},
      ArborX::Experimental::Morton64{}, bvh.bounds());

  // ── Apply permutation to all three arrays in lock-step ───────────────
  // NOTE: `boxes` and `bvh` are now stale (they reflect the old ordering).
  //       Do not use `bvh` for spatial queries after this point.
  ArborX::Details::applyPermutation(ExecSpace{}, permute, grid.quad_points);
  ArborX::Details::applyPermutation(ExecSpace{}, permute, grid.weights);

  // ── Build tiling: group the Morton-sorted points into blocks of N ────
  const int n_bounding_boxes =
      (num_points + max_points_per_bb - 1) / max_points_per_bb;

  Kokkos::View<Box *, ExecSpace> bounding_boxes("bounding_boxes",
                                                n_bounding_boxes);

  Kokkos::parallel_for(
      "compute_bounding_boxes",
      Kokkos::RangePolicy<ExecSpace>(0, n_bounding_boxes),
      KOKKOS_LAMBDA(int i) {
        const int start = i * max_points_per_bb;

        const int count = Kokkos::min(max_points_per_bb, num_points - start);

        Box tile_box{grid.quad_points(start),
                     grid.quad_points(start)}; // init: min==max
        for (int k = 1; k < count; ++k) {
          const auto &p = grid.quad_points(start + k);
          for (int d = 0; d < 3; ++d) {
            tile_box.minCorner()._coords[d] =
                Kokkos::min(tile_box.minCorner()._coords[d], p._coords[d]);
            tile_box.maxCorner()._coords[d] =
                Kokkos::max(tile_box.maxCorner()._coords[d], p._coords[d]);
          }
        }
        bounding_boxes(i) = tile_box;
      });
  ExecSpace{}.fence();
  return bounding_boxes;
}

} // namespace NuKEXC
