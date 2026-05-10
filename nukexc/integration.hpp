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
#include <KokkosBlas3_gemm.hpp>

namespace NuKEXC {
// A helper to determine batch size based on available memory or a fixed
// constant
const size_t CHUNK_SIZE = 50000;

DeviceView2DLeft
overlap_integral(STOBasisSet &basis,
                 Kokkos::View<double *[3], ExecSpace> quadrature_points,
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
        quadrature_points, std::make_pair(start, start + current_batch_size),
        Kokkos::ALL);
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
                      Kokkos::View<double *[3]> quadrature_points,
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
    STOBasisSet &basis, Kokkos::View<double *[3]> quadrature_points,
    Kokkos::View<double *> quadrature_weights,
    Kokkos::View<double *[3]> atom_centers, Kokkos::View<unsigned *> Z) {

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
        quadrature_points, std::make_pair(start, start + current_batch_size),
        Kokkos::ALL);
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
            double dx = batch_pts(g, 0) - atom_centers(k, 0);
            double dy = batch_pts(g, 1) - atom_centers(k, 1);
            double dz = batch_pts(g, 2) - atom_centers(k, 2);
            double r =
                Kokkos::sqrt(dx * dx + dy * dy + dz * dz) + epsilon_shift;

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
                 Kokkos::View<double *[3], ExecSpace> quadrature_points,
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

    auto batch_pts = Kokkos::subview(
        quadrature_points, std::make_pair(start, start + cur), Kokkos::ALL);
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

CoreHamiltonianResult
compute_core_hamiltonian(STOBasisSet &basis,
                         Kokkos::View<double *[3], ExecSpace> quadrature_points,
                         Kokkos::View<double *, ExecSpace> quadrature_weights,
                         Kokkos::View<double *[3], ExecSpace> atom_centers,
                         Kokkos::View<unsigned *, ExecSpace> Z) {

  int N = basis.nbf();
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
        quadrature_points, std::make_pair(start, start + current_batch_size),
        Kokkos::ALL);
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
            double dx = batch_pts(g, 0) - atom_centers(k, 0);
            double dy = batch_pts(g, 1) - atom_centers(k, 1);
            double dz = batch_pts(g, 2) - atom_centers(k, 2);
            double r =
                Kokkos::sqrt(dx * dx + dy * dy + dz * dz) + epsilon_shift;
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
} // namespace NuKEXC
