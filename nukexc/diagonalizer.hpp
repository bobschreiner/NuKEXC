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
#include "nukexc/nukexc_utils.hpp"
#include "nukexc_config.hpp"

#include <KokkosBlas3_gemm.hpp>
#include <KokkosBlas3_trsm.hpp>

#include <KokkosBatched_Dot.hpp>
#include <KokkosLapack_svd.hpp>

#include <Kokkos_Core.hpp>

// The Kokkos Cholesky implementation is a work in progress as of May 2026
// We use svd instead
// #include <KokkosLapack_potrf.hpp>

namespace Nukexc {

class Diagonalizer {
public:
  Diagonalizer(int N) : _N(N) {}

  // Call this only when the overlap matrix S changes (e.g., new geometry)
  DeviceView2DLeft
  compute_transformation(const DeviceView2DLeft &overlap_matrix,
                         const double lin_dep_threshold = 1e-5) {
    _X = compute_half_inverse(overlap_matrix, lin_dep_threshold);
    return _X;
  }

  // Repeatedly call this with updated Fock matrices
  void solve(const DeviceView2DLeft &fock_matrix, DeviceView2DLeft &mo_orbitals,
             DeviceView1D &mo_coeff) {

    // 1. Transform Fock Matrix: F' = X^T * F * X
    const int K = _X.extent(1);

    DeviceView2DLeft XT_F("XT_F", K, _N);
    DeviceView2DLeft F("Fock Reduced", K, K);
    DeviceView2DLeft U("U", K, K);
    DeviceView2DLeft VT("VT", K, K);

    KokkosBlas::gemm("T", "N", 1.0, _X, fock_matrix, 0.0, XT_F);
    KokkosBlas::gemm("N", "N", 1.0, XT_F, _X, 0.0, F);

    // 2. Diagonalize the transformed F
    KokkosLapack::svd("S", "S", F, mo_coeff, U, VT);

    Kokkos::parallel_for(
        "SwitchSigns", K, KOKKOS_LAMBDA(const int j) {
          double dot = 0.0;
          for (int k = 0; k < K; ++k) {
            dot += U(k, j) * VT(j, k);
          }
          if (dot < 0) {
            mo_coeff(j) = -mo_coeff(j);
          }
        });

    // 4. Back-transform: C = X * U
    KokkosBlas::gemm("N", "N", 1.0, _X, U, 0.0, mo_orbitals);

    // 5. Final Sorting
    sort(mo_orbitals, mo_coeff, K);
  }

  // Sort K orbitals (columns of mo_orbitals, entries of mo_coeff) by
  // ascending energy.  mo_orbitals has _N rows and K columns.
  void sort(DeviceView2DLeft &mo_orbitals, DeviceView1D &mo_coeff,
            const int K) {
    const int N = _N; // row count — captured separately from K (column count)
    Kokkos::parallel_for(
        "SerialSort", 1, KOKKOS_LAMBDA(const int) {
          for (int i = 0; i < K - 1; i++) { // Bug 1 fixed: was N-1
            int min_idx = i;
            for (int j = i + 1; j < K; j++) { // Bug 1 fixed: was N
              if (mo_coeff(j) < mo_coeff(min_idx))
                min_idx = j;
            }
            if (min_idx != i) {
              // Swap energies
              double e_tmp = mo_coeff(i);
              mo_coeff(i) = mo_coeff(min_idx);
              mo_coeff(min_idx) = e_tmp;
              // Swap orbital columns — loop over N rows
              for (int k = 0; k < N; k++) { // Bug 2 clarified: N rows
                double c_tmp = mo_orbitals(k, i);
                mo_orbitals(k, i) = mo_orbitals(k, min_idx);
                mo_orbitals(k, min_idx) = c_tmp;
              }
            }
          }
        });
  }

private:
  int _N;
  DeviceView2DLeft _X;
};

} // namespace Nukexc
