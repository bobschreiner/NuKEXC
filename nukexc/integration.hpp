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
#include "molecule.hpp"
#include "nukexc_config.hpp"
#include "octree.hpp"
#include "partitioning.hpp"
#include "stobasis.hpp"

#include <KokkosBatched_Dot.hpp>
#include <KokkosBatched_Gemm_Decl.hpp>
#include <KokkosBatched_Gemm_Team_Impl.hpp>
#include <KokkosBatched_Util.hpp>
#include <KokkosBlas2_team_gemv.hpp>
#include <KokkosBlas3_gemm.hpp>
#include <Kokkos_Pair.hpp>
#include <decl/Kokkos_Declare_OPENMP.hpp>
#include <impl/Kokkos_HostThreadTeam.hpp>
#include <stdexcept>
#include <traits/Kokkos_IterationPatternTrait.hpp>

namespace NuKEXC {
// A helper to determine batch size based on available memory or a fixed
// constant
const size_t CHUNK_SIZE = 50000;

DeviceView2DLeft
overlap_integral(STOBasisSet &basis, Kokkos::View<Point *> quadrature_points,
                 Kokkos::View<double *, ExecSpace> quadrature_weights) {

  int N = basis.nbf();
  int total_quad_points = quadrature_points.extent(0);

  DeviceView2DLeft overlap_matrix("Overlap matrix", N, N);
  // Pre-allocate ALL buffers — two sets, one per stream
  DeviceView2DLeft col_a("col_a", N, CHUNK_SIZE);
  DeviceView2DLeft col_b("col_b", N, CHUNK_SIZE);
  DeviceView2DLeft wt_a("wt_a", N, CHUNK_SIZE);
  DeviceView2DLeft wt_b("wt_b", N, CHUNK_SIZE);

  ExecSpace space_a, space_b;

  for (size_t start = 0; start < total_quad_points; start += CHUNK_SIZE) {
    int current_batch_size = std::min(CHUNK_SIZE, total_quad_points - start);
    bool even = (start / CHUNK_SIZE) % 2 == 0;

    auto &col_cur = even ? col_a : col_b;
    auto &wt_cur = even ? wt_a : wt_b;
    auto &space_cur = even ? space_a : space_b;
    auto &space_prev = even ? space_b : space_a;

    auto batch_pts = Kokkos::subview(
        quadrature_points, std::make_pair(start, start + current_batch_size));
    auto batch_wts = Kokkos::subview(
        quadrature_weights, std::make_pair(start, start + current_batch_size));
    auto col_view = Kokkos::subview(col_cur, Kokkos::ALL,
                                    std::make_pair(0, current_batch_size));
    auto wt_view = Kokkos::subview(wt_cur, Kokkos::ALL,
                                   std::make_pair(0, current_batch_size));

    // Collocation on space_cur — can overlap with GEMM still running
    // on space_prev from the last iteration
    fill_collocation(space_cur, basis, batch_pts, col_view); // no alloc

    Kokkos::parallel_for(
        "Scale",
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>(
            space_cur, {0, 0}, {N, current_batch_size}),
        KOKKOS_LAMBDA(int i, int g) {
          wt_view(i, g) = batch_wts(g) * col_view(i, g);
        });

    // CRITICAL: wait for previous GEMM to finish before starting next one
    // They cannot run concurrently — both write to overlap_matrix
    space_prev.fence();

    KokkosBlas::gemm(space_cur, "N", "T", 1.0, wt_view, col_view, 1.0,
                     overlap_matrix);
  }

  space_a.fence();
  space_b.fence();
  return overlap_matrix;
}

DeviceView2DLeft
diag_overlap_integral(STOBasisSet &basis,
                      Kokkos::View<Point *> quadrature_points,
                      Kokkos::View<double *> quadrature_weights) {

  size_t N = basis.nbf();
  size_t nquad_points = quadrature_points.extent(0);

  DeviceView2DLeft diag_overlap_matrix("Overlap matrix", N, N);
  DeviceView2DLeft collocation_values("Collocation values", N, nquad_points);
  ExecSpace space;

  fill_collocation(space, basis, quadrature_points, collocation_values);
  // Can be replaced by Kokkos kernel later

  Kokkos::parallel_for(
      "Compute diag{S}", N, KOKKOS_LAMBDA(const int &i) {
        for (int g = 0; g < nquad_points; ++g) {
          diag_overlap_matrix(i, i) += quadrature_weights(g) *
                                       collocation_values(i, g) *
                                       collocation_values(i, g);
        }
      });
  return diag_overlap_matrix;
}

DeviceView2DLeft nuclear_potential_integral(
    STOBasisSet &basis, Kokkos::View<Point *> quadrature_points,
    Kokkos::View<double *> quadrature_weights,
    Kokkos::View<Point *> atom_centers, Kokkos::View<unsigned *> Z) {

  int N = basis.nbf();
  int total_quad_points = quadrature_points.extent(0);

  DeviceView2DLeft V_n("Nuclear potential matrix", N, N);

  // Pre-allocate ALL buffers — two sets, one per stream
  DeviceView2DLeft col_a("col_a", N, CHUNK_SIZE);
  DeviceView2DLeft col_b("col_b", N, CHUNK_SIZE);
  DeviceView2DLeft wt_a("wt_a", N, CHUNK_SIZE);
  DeviceView2DLeft wt_b("wt_b", N, CHUNK_SIZE);

  ExecSpace space_a, space_b;

  for (size_t start = 0; start < total_quad_points; start += CHUNK_SIZE) {
    int current_batch_size = std::min(CHUNK_SIZE, total_quad_points - start);
    bool even = (start / CHUNK_SIZE) % 2 == 0;

    auto &col_cur = even ? col_a : col_b;
    auto &wt_cur = even ? wt_a : wt_b;
    auto &space_cur = even ? space_a : space_b;
    auto &space_prev = even ? space_b : space_a;

    auto batch_pts = Kokkos::subview(
        quadrature_points, std::make_pair(start, start + current_batch_size));
    auto batch_wts = Kokkos::subview(
        quadrature_weights, std::make_pair(start, start + current_batch_size));
    auto col_view = Kokkos::subview(col_cur, Kokkos::ALL,
                                    std::make_pair(0, current_batch_size));
    auto wt_view = Kokkos::subview(wt_cur, Kokkos::ALL,
                                   std::make_pair(0, current_batch_size));
    // Collocation on space_cur — can overlap with GEMM still running
    // on space_prev from the last iteration
    fill_collocation(space_cur, basis, batch_pts, col_view); // no alloc
    Kokkos::deep_copy(space_cur, wt_view, 0.0); // zero out previous batch
    Kokkos::parallel_for(
        "Scale (nuclear potential)",
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>(
            space_cur, {0, 0}, {N, current_batch_size}),
        KOKKOS_LAMBDA(int i, int g) {
          for (unsigned int k = 0; k < atom_centers.extent(0); ++k) {
            double r = dist(batch_pts(g), atom_centers(k)) + epsilon_shift;

            wt_view(i, g) -= (Z(k) / r) * batch_wts(g) * col_view(i, g);
          }
        });

    // CRITICAL: wait for previous GEMM to finish before starting next one
    // They cannot run concurrently — both write to overlap_matrix
    space_prev.fence();

    KokkosBlas::gemm(space_cur, "N", "T", 1.0, wt_view, col_view, 1.0, V_n);
  }
  space_a.fence();
  space_b.fence();
  return V_n;
}

DeviceView2DLeft
kinetic_integral(STOBasisSet &basis,
                 Kokkos::View<Point *, ExecSpace> quadrature_points,
                 Kokkos::View<double *, ExecSpace> quadrature_weights) {

  int N = basis.nbf();
  int total_quad_points = quadrature_points.extent(0);

  DeviceView2DLeft kinetic_matrix("Kinetic matrix", N, N);

  // Pre-allocate double buffers — grad collocation has 3 components
  Kokkos::View<double **[3], ExecSpace> grad_a("grad_a", N, CHUNK_SIZE);
  Kokkos::View<double **[3], ExecSpace> grad_b("grad_b", N, CHUNK_SIZE);

  // Weighted gradient buffers — one per spatial direction per stream
  DeviceView2DLeft Gx_a("Gx_a", N, CHUNK_SIZE), Gx_b("Gx_b", N, CHUNK_SIZE);
  DeviceView2DLeft Gy_a("Gy_a", N, CHUNK_SIZE), Gy_b("Gy_b", N, CHUNK_SIZE);
  DeviceView2DLeft Gz_a("Gz_a", N, CHUNK_SIZE), Gz_b("Gz_b", N, CHUNK_SIZE);

  ExecSpace space_a, space_b;

  for (size_t start = 0; start < total_quad_points; start += CHUNK_SIZE) {
    int cur = std::min(CHUNK_SIZE, total_quad_points - start);
    bool even = (start / CHUNK_SIZE) % 2 == 0;

    auto &grad_cur = even ? grad_a : grad_b;
    auto &Gx_cur = even ? Gx_a : Gx_b;
    auto &Gy_cur = even ? Gy_a : Gy_b;
    auto &Gz_cur = even ? Gz_a : Gz_b;
    auto &space_cur = even ? space_a : space_b;
    auto &space_prev = even ? space_b : space_a;

    auto batch_pts =
        Kokkos::subview(quadrature_points, std::make_pair(start, start + cur));
    auto batch_wts =
        Kokkos::subview(quadrature_weights, std::make_pair(start, start + cur));

    // Subviews into current chunk size
    auto grad_view = Kokkos::subview(grad_cur, Kokkos::ALL,
                                     std::make_pair(0, cur), Kokkos::ALL);
    auto Gx_view = Kokkos::subview(Gx_cur, Kokkos::ALL, std::make_pair(0, cur));
    auto Gy_view = Kokkos::subview(Gy_cur, Kokkos::ALL, std::make_pair(0, cur));
    auto Gz_view = Kokkos::subview(Gz_cur, Kokkos::ALL, std::make_pair(0, cur));

    // Fill grad collocation on space_cur — overlaps with GEMM on space_prev
    fill_grad_collocation(space_cur, basis, batch_pts, grad_view);

    // Weight the gradients: G{xyz}(i,g) = grad(i,g,d) * sqrt(w(g))
    Kokkos::parallel_for(
        "Weight gradients",
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>(space_cur, {0, 0},
                                                          {N, cur}),
        KOKKOS_LAMBDA(int i, int g) {
          double wf = Kokkos::sqrt(batch_wts(g));
          Gx_view(i, g) = grad_view(i, g, 0) * wf;
          Gy_view(i, g) = grad_view(i, g, 1) * wf;
          Gz_view(i, g) = grad_view(i, g, 2) * wf;
        });

    // Wait for previous iteration's GEMMs before writing to kinetic_matrix
    space_prev.fence();

    // Three GEMMs accumulate into kinetic_matrix sequentially on space_cur
    // First batch uses beta=1.0 to accumulate across chunks
    KokkosBlas::gemm(space_cur, "N", "T", 0.5, Gx_view, Gx_view, 1.0,
                     kinetic_matrix);
    KokkosBlas::gemm(space_cur, "N", "T", 0.5, Gy_view, Gy_view, 1.0,
                     kinetic_matrix);
    KokkosBlas::gemm(space_cur, "N", "T", 0.5, Gz_view, Gz_view, 1.0,
                     kinetic_matrix);
  }

  space_a.fence();
  space_b.fence();
  return kinetic_matrix;
}

struct CoreHamiltonianResult {
  DeviceView2DLeft overlap;
  DeviceView2DLeft kinetic;
  DeviceView2DLeft nuclear;
  DeviceView2DLeft hamiltonian; // T + V_n
};

CoreHamiltonianResult compute_core_hamiltonian(const STOBasisSet &basis,
                                               const FlatGrid &grid) {

  int N = basis.nbf();
  auto quadrature_points = grid.quad_points;
  auto quadrature_weights = grid.weights;
  auto Z = grid.Z;
  auto atom_centers = grid.atom_centers;

  int total_quad_points = quadrature_points.extent(0);

  CoreHamiltonianResult result;
  result.overlap = DeviceView2DLeft("Overlap matrix", N, N);
  result.kinetic = DeviceView2DLeft("Kinetic matrix", N, N);
  result.nuclear = DeviceView2DLeft("Nuclear potential matrix", N, N);
  result.hamiltonian = DeviceView2DLeft("Core Hamiltonian", N, N);

  // Single set of double buffers shared across all three integrals
  DeviceView2DLeft col_a("col_a", N, CHUNK_SIZE);
  DeviceView2DLeft col_b("col_b", N, CHUNK_SIZE);
  DeviceView2DLeft wt_overlap_a("wt_overlap_a", N, CHUNK_SIZE);
  DeviceView2DLeft wt_overlap_b("wt_overlap_b", N, CHUNK_SIZE);
  DeviceView2DLeft wt_nuclear_a("wt_nuclear_a", N, CHUNK_SIZE);
  DeviceView2DLeft wt_nuclear_b("wt_nuclear_b", N, CHUNK_SIZE);

  Kokkos::View<double **[3], ExecSpace> grad_a("grad_a", N, CHUNK_SIZE);
  Kokkos::View<double **[3], ExecSpace> grad_b("grad_b", N, CHUNK_SIZE);

  DeviceView2DLeft Gx_a("Gx_a", N, CHUNK_SIZE), Gx_b("Gx_b", N, CHUNK_SIZE);
  DeviceView2DLeft Gy_a("Gy_a", N, CHUNK_SIZE), Gy_b("Gy_b", N, CHUNK_SIZE);
  DeviceView2DLeft Gz_a("Gz_a", N, CHUNK_SIZE), Gz_b("Gz_b", N, CHUNK_SIZE);

  ExecSpace space_a, space_b;

  for (size_t start = 0; start < total_quad_points; start += CHUNK_SIZE) {
    int current_batch_size = std::min(CHUNK_SIZE, total_quad_points - start);
    bool even = (start / CHUNK_SIZE) % 2 == 0;

    auto &col_cur = even ? col_a : col_b;
    auto &wt_ov_cur = even ? wt_overlap_a : wt_overlap_b;
    auto &wt_nuc_cur = even ? wt_nuclear_a : wt_nuclear_b;
    auto &grad_cur = even ? grad_a : grad_b;
    auto &Gx_cur = even ? Gx_a : Gx_b;
    auto &Gy_cur = even ? Gy_a : Gy_b;
    auto &Gz_cur = even ? Gz_a : Gz_b;
    auto &space_cur = even ? space_a : space_b;
    auto &space_prev = even ? space_b : space_a;

    auto batch_pts = Kokkos::subview(
        quadrature_points, std::make_pair(start, start + current_batch_size));
    auto batch_wts = Kokkos::subview(
        quadrature_weights, std::make_pair(start, start + current_batch_size));

    auto col_view = Kokkos::subview(col_cur, Kokkos::ALL,
                                    std::make_pair(0, current_batch_size));
    auto wt_ov_view = Kokkos::subview(wt_ov_cur, Kokkos::ALL,
                                      std::make_pair(0, current_batch_size));
    auto wt_nuc_view = Kokkos::subview(wt_nuc_cur, Kokkos::ALL,
                                       std::make_pair(0, current_batch_size));
    auto grad_view =
        Kokkos::subview(grad_cur, Kokkos::ALL,
                        std::make_pair(0, current_batch_size), Kokkos::ALL);
    auto Gx_view = Kokkos::subview(Gx_cur, Kokkos::ALL,
                                   std::make_pair(0, current_batch_size));
    auto Gy_view = Kokkos::subview(Gy_cur, Kokkos::ALL,
                                   std::make_pair(0, current_batch_size));
    auto Gz_view = Kokkos::subview(Gz_cur, Kokkos::ALL,
                                   std::make_pair(0, current_batch_size));

    // Single collocation evaluation — used by overlap AND nuclear
    fill_collocation(space_cur, basis, batch_pts, col_view);

    // Single grad collocation evaluation — used by kinetic only
    fill_grad_collocation(space_cur, basis, batch_pts, grad_view);

    // Zero nuclear weight buffer before accumulating
    Kokkos::deep_copy(space_cur, wt_nuc_view, 0.0);

    // Single fused kernel: compute overlap weights, nuclear weights,
    // and gradient weights all in one pass over (i, g)
    Kokkos::parallel_for(
        "Fused scale",
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>(
            space_cur, {0, 0}, {N, current_batch_size}),
        KOKKOS_LAMBDA(int i, int g) {
          double phi_ig = col_view(i, g);
          double w_g = batch_wts(g);

          // Overlap weight
          wt_ov_view(i, g) = w_g * phi_ig;

          // Nuclear weight — accumulate over atoms
          double v_nuc = 0.0;
          for (unsigned k = 0; k < atom_centers.extent(0); ++k) {
            double r = dist(batch_pts(g), atom_centers(k)) + epsilon_shift;
            v_nuc -= double(Z(k)) / r;
          }
          wt_nuc_view(i, g) = v_nuc * w_g * phi_ig;

          // Gradient weights
          double wf = Kokkos::sqrt(w_g);
          Gx_view(i, g) = grad_view(i, g, 0) * wf;
          Gy_view(i, g) = grad_view(i, g, 1) * wf;
          Gz_view(i, g) = grad_view(i, g, 2) * wf;
        });

    // Serialise GEMMs against previous iteration
    space_prev.fence();

    // Overlap: S += wt_ov * col^T
    KokkosBlas::gemm(space_cur, "N", "T", 1.0, wt_ov_view, col_view, 1.0,
                     result.overlap);

    // Nuclear: V += wt_nuc * col^T
    KokkosBlas::gemm(space_cur, "N", "T", 1.0, wt_nuc_view, col_view, 1.0,
                     result.nuclear);

    // Kinetic: T += 0.5 * G{xyz} * G{xyz}^T
    KokkosBlas::gemm(space_cur, "N", "T", 0.5, Gx_view, Gx_view, 1.0,
                     result.kinetic);
    KokkosBlas::gemm(space_cur, "N", "T", 0.5, Gy_view, Gy_view, 1.0,
                     result.kinetic);
    KokkosBlas::gemm(space_cur, "N", "T", 0.5, Gz_view, Gz_view, 1.0,
                     result.kinetic);
  }

  space_a.fence();
  space_b.fence();

  // Form H = T + V_n
  int total = N * N;
  auto T = result.kinetic;
  auto V = result.nuclear;
  auto H = result.hamiltonian;
  Kokkos::parallel_for(
      "Form core Hamiltonian",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {N, N}),
      KOKKOS_LAMBDA(int i, int j) { H(i, j) = T(i, j) + V(i, j); });

  return result;
}

struct CoreHamiltonianReducer {
  double s = 0.0, v = 0.0, t = 0.0;

  KOKKOS_INLINE_FUNCTION
  CoreHamiltonianReducer &operator+=(const CoreHamiltonianReducer &rhs) {
    s += rhs.s;
    v += rhs.v;
    t += rhs.t;
    return *this;
  }

  KOKKOS_INLINE_FUNCTION
  CoreHamiltonianReducer operator+(const CoreHamiltonianReducer &rhs) const {
    return {s + rhs.s, v + rhs.v, t + rhs.t};
  }
  KOKKOS_INLINE_FUNCTION static void
  join(volatile CoreHamiltonianReducer &dst,
       const volatile CoreHamiltonianReducer &src) {
    dst.s += src.s;
    dst.v += src.v;
    dst.t += src.t;
  }

  KOKKOS_INLINE_FUNCTION static void init(CoreHamiltonianReducer &val) {
    val.s = 0.0;
    val.v = 0.0;
    val.t = 0.0;
  }
};

CoreHamiltonianResult compute_core_hamiltonian_screened(
    const STOBasisSet &basis, const FlatGrid &grid, const NeighborList &nl) {

  int N = basis.nbf();
  auto Z = grid.Z;
  auto atom_centers = grid.atom_centers;

  const int max_points_per_box = nl.max_points_per_box;
  const int total_points = nl.total_points;
  const int num_boxes = nl.offsets.extent(0) - 1;

  CoreHamiltonianResult result;
  result.overlap = DeviceView2DLeft("Overlap matrix", N, N);
  result.kinetic = DeviceView2DLeft("Kinetic matrix", N, N);
  result.nuclear = DeviceView2DLeft("Nuclear potential matrix", N, N);
  result.hamiltonian = DeviceView2DLeft("Core Hamiltonian", N, N);

  // --- Reduce to find max_n, then allocate with correct size ---
  int max_neighbors = 0;
  Kokkos::parallel_reduce(
      "Find maximum number of neighbors", nl.offsets.extent(0) - 1,
      KOKKOS_LAMBDA(const int &i, int &lmax) {
        const int num_neighbors = nl.offsets(i + 1) - nl.offsets(i);
        if (lmax < num_neighbors)
          lmax = num_neighbors;
      },
      Kokkos::Max<int>(max_neighbors));

  // Define helpers for scratch space access
  typedef ExecSpace::scratch_memory_space ScratchSpace;

  typedef Kokkos::View<double *, ScratchSpace,
                       Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      shared_view_double;

  typedef Kokkos::View<Point *, ScratchSpace,
                       Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      shared_view_points;

  Kokkos::TeamPolicy<ExecSpace> policy(num_boxes, Kokkos::AUTO());
  using member_type = Kokkos::TeamPolicy<ExecSpace>::member_type;

  // Compute required cache sizes
  size_t scratch_grid_size =
      (shared_view_double::shmem_size(max_points_per_box)     // weights
       + shared_view_double::shmem_size(max_points_per_box)   // v_nuc
       + shared_view_points::shmem_size(max_points_per_box)); // quad_points

  size_t scratch_basis_size = 3 * shared_view_double::shmem_size(max_neighbors);

  // Check how much memory is available per cache
  int scratch_grid_level = 0;
  int scratch_basis_level = 1;
  if (scratch_grid_size > policy.scratch_size_max(0))
    scratch_grid_level = 1;

#if 1
  std::cout << "---------- Memory Allocations ------------\n";

  std::cout << "Max Neighbors         : " << max_neighbors << "\n";
  std::cout << "Available L0 size     : " << policy.scratch_size_max(0) << "\n";
  std::cout << "Available L1 size     : " << policy.scratch_size_max(1) << "\n";
  std::cout << "Scratch size needed for grid: " << scratch_grid_size << "\n";
  std::cout << "Scratch size needed for : " << scratch_basis_size << "\n";
#endif

  if (scratch_basis_level == scratch_grid_level) {
    policy.set_scratch_size(
        scratch_grid_level,
        Kokkos::PerTeam(scratch_grid_size + scratch_basis_size));
  } else {
    policy.set_scratch_size(scratch_grid_level,
                            Kokkos::PerTeam(scratch_grid_size));
    policy.set_scratch_size(scratch_basis_level,
                            Kokkos::PerTeam(scratch_basis_size));
  }

  if (scratch_grid_size > policy.scratch_size_max(1))
    throw std::runtime_error("Could not allocate engouh memory on scratch\n");

  if (scratch_basis_size > policy.scratch_size_max(1))
    throw std::runtime_error("Could not allocate engouh memory on scratch\n");

  if (scratch_grid_size + scratch_basis_size > policy.scratch_size_max(1))
    throw std::runtime_error("Could not allocate engouh memory on scratch\n");

  Kokkos::View<double ***, Kokkos::LayoutLeft> basis_val(
      "Basis functions", num_boxes, max_neighbors, nl.max_points_per_box);

  Kokkos::View<double ***, Kokkos::LayoutLeft> basis_gx(
      "Basis gradient x", num_boxes, max_neighbors, nl.max_points_per_box);

  Kokkos::View<double ***, Kokkos::LayoutLeft> basis_gy(
      "Basis gradient y", num_boxes, max_neighbors, nl.max_points_per_box);

  Kokkos::View<double ***, Kokkos::LayoutLeft> basis_gz(
      "Basis gradient z", num_boxes, max_neighbors, nl.max_points_per_box);

  Kokkos::parallel_for(
      "Compute all basis functions",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
          {0, 0}, {num_boxes, nl.max_points_per_box}),
      KOKKOS_LAMBDA(const int box_idx, const int local_g) {
        const int start_points = box_idx * max_points_per_box;
        const int end_points =
            Kokkos::min(start_points + max_points_per_box, total_points);
        const int num_points = end_points - start_points;

        if (local_g >= num_points)
          return;

        const int start_neighbors = nl.offsets(box_idx);
        const int end_neighbors = nl.offsets(box_idx + 1);
        const int num_neighbors = end_neighbors - start_neighbors;

        const int global_g = max_points_per_box * box_idx + local_g;
        for (int local_i = 0; local_i < num_neighbors; ++local_i) {

          const int global_i = nl.neighbors(nl.offsets(box_idx) + local_i);
          ScratchBasisParams local_basis = {
              basis.zeta(global_i), basis.norm(global_i), basis.O(global_i),
              basis.n(global_i),    basis.l(global_i),    basis.m(global_i)};

          basis_eval_with_grad(local_basis, grid.quad_points(global_g),
                               basis_val(box_idx, local_i, local_g),
                               basis_gx(box_idx, local_i, local_g),
                               basis_gy(box_idx, local_i, local_g),
                               basis_gz(box_idx, local_i, local_g));
        }
      });

  Kokkos::parallel_for(
      "Compute Core Hamiltonian Screened", policy,
      KOKKOS_LAMBDA(const member_type &team_member) {
        const int box_idx = team_member.league_rank();

        // Compute number of points per box
        const int start_points = box_idx * max_points_per_box;
        const int end_points =
            Kokkos::min(start_points + max_points_per_box, total_points);
        const int num_points = end_points - start_points;

        // Compute number of neighbors per box
        const int start_neighbors = nl.offsets(box_idx);
        const int end_neighbors = nl.offsets(box_idx + 1);
        const int num_neighbors = end_neighbors - start_neighbors;

        // Initialize shared memory
        shared_view_double weights(team_member.team_scratch(scratch_grid_level),
                                   num_points);
        shared_view_points quad_points(
            team_member.team_scratch(scratch_grid_level), num_points);
        shared_view_double v_nuc(team_member.team_scratch(scratch_grid_level),
                                 num_points);
        shared_view_double s_local(
            team_member.team_scratch(scratch_basis_level), max_neighbors);
        shared_view_double v_local(
            team_member.team_scratch(scratch_basis_level), max_neighbors);

        shared_view_double t_local(
            team_member.team_scratch(scratch_basis_level), max_neighbors);

        // Fill weights
        Kokkos::parallel_for(
            Kokkos::TeamVectorRange(team_member, num_points),
            [=](int &local_g) {
              const int global_g = max_points_per_box * box_idx + local_g;
              weights(local_g) = Kokkos::sqrt(grid.weights(global_g));
              quad_points(local_g) = grid.quad_points(global_g);

              v_nuc(local_g) = 0;
              double r_sum = 0.0;
              for (int k = 0; k < grid.atom_centers.extent(0); ++k) {
                double r = dist(grid.atom_centers(k), quad_points(local_g));
                r_sum += grid.Z(k) / r;
              }
              v_nuc(local_g) = Kokkos::sqrt(r_sum);
            });

        auto team_basis =
            Kokkos::subview(basis_val, box_idx, Kokkos::ALL(), Kokkos::ALL());

        auto team_gx =
            Kokkos::subview(basis_gx, box_idx, Kokkos::ALL(), Kokkos::ALL());
        auto team_gy =
            Kokkos::subview(basis_gy, box_idx, Kokkos::ALL(), Kokkos::ALL());
        auto team_gz =
            Kokkos::subview(basis_gz, box_idx, Kokkos::ALL(), Kokkos::ALL());

        team_member.team_barrier();
        Kokkos::parallel_for(
            Kokkos::TeamVectorMDRange(team_member, num_neighbors, num_points),
            [=](int &local_i, int &local_g) {
              team_basis(local_i, local_g) *= weights(local_g);
              team_gx(local_i, local_g) *= weights(local_g);
              team_gy(local_i, local_g) *= weights(local_g);
              team_gz(local_i, local_g) *= weights(local_g);
            });

        for (int local_j = 0; local_j < num_neighbors; ++local_j) {

          int global_j = nl.neighbors(nl.offsets(box_idx) + local_j);

          auto local_subview_basis =
              Kokkos::subview(basis_val, box_idx, local_j, Kokkos::ALL());

          auto local_subview_gx =
              Kokkos::subview(basis_gx, box_idx, local_j, Kokkos::ALL());
          auto local_subview_gy =
              Kokkos::subview(basis_gy, box_idx, local_j, Kokkos::ALL());

          auto local_subview_gz =
              Kokkos::subview(basis_gz, box_idx, local_j, Kokkos::ALL());

          team_member.team_barrier();
          KokkosBlas::TeamGemv<
              member_type, KokkosBlas::Trans::NoTranspose,
              KokkosBlas::Algo::Gemv::Unblocked>::invoke(team_member, 1.0,
                                                         team_basis,
                                                         local_subview_basis,
                                                         0.0, s_local);

          KokkosBlas::TeamGemv<
              member_type, KokkosBlas::Trans::NoTranspose,
              KokkosBlas::Algo::Gemv::Unblocked>::invoke(team_member, 0.5,
                                                         team_gx,
                                                         local_subview_gx, 0.0,
                                                         t_local);
          KokkosBlas::TeamGemv<
              member_type, KokkosBlas::Trans::NoTranspose,
              KokkosBlas::Algo::Gemv::Unblocked>::invoke(team_member, 0.5,
                                                         team_gy,
                                                         local_subview_gy, 1.0,
                                                         t_local);

          KokkosBlas::TeamGemv<
              member_type, KokkosBlas::Trans::NoTranspose,
              KokkosBlas::Algo::Gemv::Blocked>::invoke(team_member, 0.5,
                                                       team_gz,
                                                       local_subview_gz, 1.0,
                                                       t_local);

          team_member.team_barrier();
          Kokkos::single(Kokkos::PerThread(team_member), [=]() {
            for (int local_i = 0; local_i < num_neighbors; ++local_i) {
              int global_i = nl.neighbors(nl.offsets(box_idx) + local_i);

              Kokkos::atomic_fetch_add(&result.overlap(global_i, global_j),
                                       s_local(local_i));
              Kokkos::atomic_fetch_add(&result.kinetic(global_i, global_j),
                                       t_local(local_i));
            }
          });
        }

        team_member.team_barrier();
        Kokkos::parallel_for(
            Kokkos::TeamVectorMDRange(team_member, num_neighbors, num_points),
            [=](int &local_i, int &local_g) {
              team_basis(local_i, local_g) *= v_nuc(local_g);
            });

        team_member.team_barrier();

        for (int local_j = 0; local_j < num_neighbors; ++local_j) {

          int global_j = nl.neighbors(nl.offsets(box_idx) + local_j);
          auto local_subview_basis =
              Kokkos::subview(basis_val, box_idx, local_j, Kokkos::ALL());

          team_member.team_barrier();
          KokkosBlas::TeamGemv<
              member_type, KokkosBlas::Trans::NoTranspose,
              KokkosBlas::Algo::Gemv::Blocked>::invoke(team_member, -1.0,
                                                       team_basis,
                                                       local_subview_basis, 0.0,
                                                       v_local);

          team_member.team_barrier();
          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(team_member, num_neighbors),
              [=](int local_i) {
                int global_i = nl.neighbors(nl.offsets(box_idx) + local_i);
                Kokkos::atomic_fetch_add(&result.nuclear(global_i, global_j),
                                         v_local(local_i));
              });
        }
      });

  ExecSpace().fence();
  Kokkos::parallel_for(
      "Compute Core Hamiltonian Matrix",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {N, N}),
      KOKKOS_LAMBDA(const int i, const int j) {
        result.hamiltonian(i, j) = result.kinetic(i, j) + result.nuclear(i, j);
      });

  Kokkos::fence();
  return result;
}

CoreHamiltonianResult compute_core_hamiltonian_screened_scratch(
    const STOBasisSet &basis, const FlatGrid &grid, const NeighborList &nl) {
  int N = basis.nbf();
  auto Z = grid.Z;
  auto atom_centers = grid.atom_centers;

  const int max_points_per_box = nl.max_points_per_box;
  const int total_points = nl.total_points;
  const int num_boxes = nl.offsets.extent(0) - 1;

  CoreHamiltonianResult result;
  result.overlap = DeviceView2DLeft("Overlap matrix", N, N);
  result.kinetic = DeviceView2DLeft("Kinetic matrix", N, N);
  result.nuclear = DeviceView2DLeft("Nuclear potential matrix", N, N);
  result.hamiltonian = DeviceView2DLeft("Core Hamiltonian", N, N);

  // --- Reduce to find max_n, then allocate with correct size ---
  int max_neighbors = 0;
  Kokkos::parallel_reduce(
      "Find maximum number of neighbors", nl.offsets.extent(0) - 1,
      KOKKOS_LAMBDA(const int &i, int &lmax) {
        const int num_neighbors = nl.offsets(i + 1) - nl.offsets(i);
        if (lmax < num_neighbors)
          lmax = num_neighbors;
      },
      Kokkos::Max<int>(max_neighbors));

  // Define helpers for scratch space access
  typedef ExecSpace::scratch_memory_space ScratchSpace;

  typedef Kokkos::View<ScratchBasisParams *, ScratchSpace,
                       Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      shared_view_basis;

  typedef Kokkos::View<double *, ScratchSpace,
                       Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      shared_view_double;

  typedef Kokkos::View<Point *, ScratchSpace,
                       Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      shared_view_points;

  Kokkos::TeamPolicy<ExecSpace> policy(num_boxes, Kokkos::AUTO());
  using member_type = Kokkos::TeamPolicy<ExecSpace>::member_type;

  // Compute required cache sizes
  size_t scratch_size_grid =
      (shared_view_double::shmem_size(max_points_per_box)     // weights
       + shared_view_double::shmem_size(max_points_per_box)   // v_nuc
       + shared_view_points::shmem_size(max_points_per_box)); // quad_points
  size_t scratch_size_basis =
      shared_view_basis::shmem_size(max_neighbors); // basis

  // Check how much memory is available per cache
  int scratch_level_grid = 0;
  int scratch_level_basis = 0;

  // Grid will almost always be bigger than basis
  // So we put both in level 1 if grid exceeds level 0 scratch
  if (scratch_size_grid > policy.scratch_size_max(0)) {
    scratch_level_grid = 1;
    scratch_level_basis = 1;
  }
  // In case scratch_level_grid is 0 we need to check if both fit in 0
  if (scratch_level_grid == 0) {
    if (scratch_size_basis + scratch_size_grid > policy.scratch_size_max(0))
      scratch_level_basis = 1;
  }
#if 1
  std::cout << "---------- Memory Allocations ------------\n";
  std::cout << "Available L0 size             : " << policy.scratch_size_max(0)
            << "\n";
  std::cout << "Available L1 size             : " << policy.scratch_size_max(1)
            << "\n";
  std::cout << "Scratch size needed for grid  : " << scratch_size_grid << "\n";
  std::cout << "Scratch size needed for basis : " << scratch_size_basis << "\n";
  std::cout << "Scratch level for grid        : " << scratch_level_grid << "\n";
  std::cout << "Scratch level for basis       : " << scratch_level_basis
            << "\n";
#endif

  // In case scratch_level_grid is 1 we need to check if both fit in 1
  if (scratch_level_basis == scratch_level_grid)
    policy.set_scratch_size(
        scratch_level_grid,
        Kokkos::PerTeam(scratch_size_grid + scratch_size_basis));
  else {
    policy.set_scratch_size(scratch_level_grid,
                            Kokkos::PerTeam(scratch_size_grid));
    policy.set_scratch_size(scratch_level_basis,
                            Kokkos::PerTeam(scratch_size_basis));
  }

  if (scratch_size_basis + scratch_size_grid > policy.scratch_size_max(1))
    throw std::runtime_error("Could not allocate engouh memory on scratch\n");

  if (scratch_size_basis > policy.scratch_size_max(1))
    throw std::runtime_error(
        "Could not allocate engouh memory on scratch for basis\n");

  Kokkos::parallel_for(
      "Compute Core Hamiltonian Screened", policy,
      KOKKOS_LAMBDA(const member_type &team_member) {
        const int box_idx = team_member.league_rank();

        // Compute number of points per box
        const int start_points = box_idx * max_points_per_box;
        const int end_points =
            Kokkos::min(start_points + max_points_per_box, total_points);
        const int num_points = end_points - start_points;

        // Compute number of neighbors per box
        const int start_neighbors = nl.offsets(box_idx);
        const int end_neighbors = nl.offsets(box_idx + 1);
        const int num_neighbors = end_neighbors - start_neighbors;

        // Initialize shared memory
        shared_view_double weights(team_member.team_scratch(scratch_level_grid),
                                   num_points);
        shared_view_points quad_points(
            team_member.team_scratch(scratch_level_grid), num_points);
        shared_view_double v_nuc(team_member.team_scratch(scratch_level_grid),
                                 num_points);
        shared_view_basis scratch_basis(
            team_member.team_scratch(scratch_level_basis), num_neighbors);

        // Fill weights
        Kokkos::parallel_for(
            Kokkos::TeamVectorRange(team_member, num_points),
            [=](int &local_g) {
              const int global_g = max_points_per_box * box_idx + local_g;
              weights(local_g) = grid.weights(global_g);
              quad_points(local_g) = grid.quad_points(global_g);

              v_nuc(local_g) = 0;
              double r_sum = 0.0;
              for (int k = 0; k < grid.atom_centers.extent(0); ++k) {
                double r = dist(grid.atom_centers(k), quad_points(local_g));
                r_sum -= grid.Z(k) / r;
              }
              v_nuc(local_g) = r_sum;
            });

        Kokkos::parallel_for(
            Kokkos::TeamVectorRange(team_member, num_neighbors),
            [=](int local_i) {
              int global_i = nl.neighbors(nl.offsets(box_idx) + local_i);
              scratch_basis(local_i).zeta = basis.zeta(global_i);
              scratch_basis(local_i).O = basis.O(global_i);
              scratch_basis(local_i).n = basis.n(global_i);
              scratch_basis(local_i).l = basis.l(global_i);
              scratch_basis(local_i).m = basis.m(global_i);
              scratch_basis(local_i).norm = basis.norm(global_i);
            });

        team_member.team_barrier();
        //  Evaluate basis functions at each quadraure point
        Kokkos::parallel_for(
            Kokkos::TeamThreadMDRange(team_member, num_neighbors,
                                      num_neighbors),
            [=](int local_i, int local_j) {
              CoreHamiltonianReducer total_contributions{0.0, 0.0, 0.0};

              const ScratchBasisParams &basis_i = scratch_basis(local_i);
              const ScratchBasisParams &basis_j = scratch_basis(local_j);
              // Loop over all quadrature_points in the box
              Kokkos::parallel_reduce(
                  Kokkos::ThreadVectorRange(team_member, num_points),
                  [=](int &local_g, CoreHamiltonianReducer &update) {
                    double basis_value_i;
                    double basis_gradx_i;
                    double basis_grady_i;
                    double basis_gradz_i;

                    double basis_value_j;
                    double basis_gradx_j;
                    double basis_grady_j;
                    double basis_gradz_j;

                    const Point quad_point = quad_points(local_g);
                    const double w = weights(local_g);

                    basis_eval_with_grad(basis_i, quad_point, basis_value_i,
                                         basis_gradx_i, basis_grady_i,
                                         basis_gradz_i);

                    basis_eval_with_grad(basis_j, quad_point, basis_value_j,
                                         basis_gradx_j, basis_grady_j,
                                         basis_gradz_j);

                    update.s += w * basis_value_i * basis_value_j;
                    update.v +=
                        v_nuc(local_g) * w * basis_value_i * basis_value_j;
                    update.t += 0.5 * w *
                                (basis_gradx_i * basis_gradx_j +
                                 basis_grady_i * basis_grady_j +
                                 basis_gradz_i * basis_gradz_j);
                  },
                  total_contributions);

              int global_i = nl.neighbors(nl.offsets(box_idx) + local_i);
              int global_j = nl.neighbors(nl.offsets(box_idx) + local_j);

              // Scatter contributions
              Kokkos::single(Kokkos::PerThread(team_member), [=]() {
                Kokkos::atomic_fetch_add(&result.overlap(global_i, global_j),
                                         total_contributions.s);
                Kokkos::atomic_fetch_add(&result.kinetic(global_i, global_j),
                                         total_contributions.t);
                Kokkos::atomic_fetch_add(&result.nuclear(global_i, global_j),
                                         total_contributions.v);
              });
            }); // thread parallel md loop
      });       // team parallel loop
  ExecSpace().fence();
  Kokkos::parallel_for(
      "Compute Core Hamiltonian Matrix",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {N, N}),
      KOKKOS_LAMBDA(const int i, const int j) {
        result.hamiltonian(i, j) = result.kinetic(i, j) + result.nuclear(i, j);
      });

  Kokkos::fence();
  return result;
}
} // namespace NuKEXC
  //
