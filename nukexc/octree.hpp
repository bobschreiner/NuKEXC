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
#include "nukexc_config.hpp"

#include <detail/ArborX_Predicates.hpp>
#include <detail/ArborX_SpaceFillingCurves.hpp>
#include <detail/ArborX_TreeVisualization.hpp>
#include <impl/Kokkos_Profiling.hpp>
#include <integratorxx/quadratures/s2/lebedev_laikov.hpp>
#include <sorting/Kokkos_SortByKeyPublicAPI.hpp>

namespace NuKEXC {

using Point = ArborX::Point<3, double>;
using Box = ArborX::Box<3, double>;

Kokkos::View<Box *, ExecSpace>
create_bounding_boxes(FlatGrid &grid, const int max_points_per_bb) {

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
            tile_box.minCorner()[d] =
                Kokkos::min(tile_box.minCorner()[d], p[d]);
            tile_box.maxCorner()[d] =
                Kokkos::max(tile_box.maxCorner()[d], p[d]);
          }
        }
        bounding_boxes(i) = tile_box;
      });
  ExecSpace{}.fence();
  return bounding_boxes;
}

// Data structure that will later be used to compute batched Dgemm and scalings
struct NeighborList {
  Kokkos::View<int *> neighbors; // flat list of basis function indices
  Kokkos::View<int *>
      offsets; // offsets(i)..offsets(i+1) gives neighbors of box i
  int max_points_per_box;
  int total_points;
};

Kokkos::View<Box *, ExecSpace>
create_shell_bounding_boxes(const FlatGrid &grid) {

  const int natoms = grid.atom_centers.extent(0);
  const int nrad = grid.nrad;
  const int nang = grid.nang;
  const int num_boxes = natoms * nrad; // one box per (atom, shell)

  Kokkos::View<Box *, ExecSpace> boxes("shell_boxes", num_boxes);

  Kokkos::parallel_for(
      "compute_shell_bounding_boxes",
      Kokkos::RangePolicy<ExecSpace>(0, num_boxes), KOKKOS_LAMBDA(int idx) {
        // idx == iatom * nrad + irad
        const int start = idx * nang; // contiguous in the flat array

        Box b{grid.quad_points(start), grid.quad_points(start)};
        for (int k = 1; k < nang; ++k) {
          const auto &p = grid.quad_points(start + k);
          for (int d = 0; d < 3; ++d) {
            b.minCorner()[d] = Kokkos::min(b.minCorner()[d], p[d]);
            b.maxCorner()[d] = Kokkos::max(b.maxCorner()[d], p[d]);
          }
        }
        boxes(idx) = b;
      });

  ExecSpace{}.fence();
  return boxes;
}
template <typename BASIS>
void build_neighbor_list(const BASIS basis,
                         const Kokkos::View<Box *, ExecSpace> &bounding_boxes,
                         const int max_points_per_box, const int total_points,
                         NeighborList &neighbor_list) {

  auto basis_origins = basis.O;
  auto cutoff_radii = basis.cutoff_radii;

  const int N = basis_origins.extent(0);
  const int num_boxes = bounding_boxes.extent(0);

  // ── Build a sphere for each basis function ────────────────────────────
  // The sphere has the center at the basis origin and radius = cutoff radius.
  // A grid tile needs this basis function iff its bounding box intersects
  // the sphere.
  using Sphere = ArborX::Sphere<3, double>;
  Kokkos::View<Sphere *, ExecSpace> spheres("spheres", N);
  Kokkos::parallel_for(
      "build_basis_spheres", Kokkos::RangePolicy<ExecSpace>(0, N),
      KOKKOS_LAMBDA(int i) {
        spheres(i) = Sphere{basis_origins(i), cutoff_radii(i)};
      });

  // ── BVH over bounding boxes ──────────────────────────────────
  // attach_indices so the query results carry the original bounding_box index.
  ArborX::BoundingVolumeHierarchy bvh{
      ExecSpace{}, ArborX::Experimental::attach_indices(spheres)};

  // ── One intersects(sphere) query per basis function ───────────────────
  // ArborX will return all spheres whose bounding volume overlaps the box.
  Kokkos::View<ArborX::Intersects<Box> *, ExecSpace> queries("queries",
                                                             num_boxes);
  Kokkos::parallel_for(
      "build_box_queries", Kokkos::RangePolicy<ExecSpace>(0, num_boxes),
      KOKKOS_LAMBDA(int i) {
        queries(i) = ArborX::intersects(bounding_boxes(i));
      });

  // ── Execute queries ───────────────────────────────────────────────────
  // offsets is length num_boxes+1 (CSR row pointers).
  // values contains PairValueIndex{sphere_value, basis_index}.

  Kokkos::View<int *, ExecSpace> offsets("offsets", 0);
  Kokkos::View<int *, ExecSpace> neighbors("neighbors", 0);

  // Use a custom callback to extract the index from PairValueIndex
  bvh.query(
      ExecSpace{}, queries,
      KOKKOS_LAMBDA(auto const &query, auto const &value, auto const &out) {
        // 'value' is the PairValueIndex{Speh, Index}
        // 'out' is the internal mechanism that fills your 'neighbors' view
        out(value.index);
      },
      neighbors, offsets);

  ExecSpace{}.fence();
  using policy = Kokkos::TeamPolicy<ExecSpace>;
  using member_type = Kokkos::TeamPolicy<ExecSpace>::member_type;

  Kokkos::parallel_for(
      "Sort neighborlist", policy(num_boxes, Kokkos::AUTO()),
      KOKKOS_LAMBDA(member_type team_member) {
        const int box_idx = team_member.league_rank();
        const int start = offsets(box_idx);
        const int end = offsets(box_idx + 1);
        auto segment =
            Kokkos::subview(neighbors, Kokkos::make_pair(start, end));
        Kokkos::Experimental::sort_team(team_member, segment);
      });
  ExecSpace{}.fence();

  neighbor_list.neighbors = neighbors;
  neighbor_list.offsets = offsets;
  neighbor_list.max_points_per_box = max_points_per_box;
  neighbor_list.total_points = total_points;
}

} // namespace NuKEXC
