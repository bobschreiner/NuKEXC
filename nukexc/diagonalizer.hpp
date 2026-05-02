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

#include <KokkosBlas3_gemm.hpp>
#include <KokkosBlas3_trsm.hpp>

#include <KokkosBatched_Dot.hpp>
#include <KokkosLapack_svd.hpp>

#include <Kokkos_Core.hpp>

// The Kokkos Cholesky implementation is a work in progress as of May 2026
// We use svd instead
// #include <KokkosLapack_potrf.hpp>

namespace NuKEXC {

class Diagonalizer {
public:
  Diagonalizer(int N) : _N(N) {
    _X = DeviceView2DLeft("TransformationMatrix_X", _N, _N);
    _XT_F = DeviceView2DLeft("XT_F_temp", _N, _N);
    _U = DeviceView2DLeft("U_temp", _N, _N);
    _VT = DeviceView2DLeft("VT_temp", _N, _N);
  }

  // Call this only when the overlap matrix S changes (e.g., new geometry)
  void compute_transformation(const DeviceView2DLeft &overlap_matrix) {
    DeviceView2DLeft S("TempS", _N, _N);
    Kokkos::deep_copy(S, overlap_matrix);

    DeviceView2DLeft Us("Us", _N, _N);
    DeviceView2DLeft VTs("VTs", _N, _N);
    DeviceView1D sigma("sigma", _N);

    // SVD of S to handle potential singularity
    KokkosLapack::svd("S", "S", S, sigma, Us, VTs);

    // Build X = Us * sigma^-1/2 (Canonical Orthogonalization)
    auto X_local = _X;
    Kokkos::parallel_for(
        "BuildX", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {_N, _N}),
        KOKKOS_LAMBDA(const int i, const int k) {
          if (sigma(k) > 1e-7) {
            X_local(i, k) = Us(i, k) / Kokkos::sqrt(sigma(k));
          } else {
            X_local(i, k) = 0.0;
          }
        });
  }

  // Repeatedly call this with updated Fock matrices
  void solve(const DeviceView2DLeft &fock_matrix, DeviceView2DLeft &mo_coeff,
             DeviceView1D &mo_energies) {

    DeviceView2DLeft F("LocalF", _N, _N);
    Kokkos::deep_copy(F, fock_matrix);

    // 1. Transform Fock Matrix: F' = X^T * F * X
    KokkosBlas::gemm("T", "N", 1.0, _X, F, 0.0, _XT_F);
    KokkosBlas::gemm("N", "N", 1.0, _XT_F, _X, 0.0, F);

    // 2. Diagonalize the transformed F
    KokkosLapack::svd("S", "S", F, mo_energies, _U, _VT);

    // 3. Restore signs for symmetric singular values
    auto U_local = _U;
    auto VT_local = _VT;
    int N = _N;
    Kokkos::parallel_for(
        "SwitchSigns", N, KOKKOS_LAMBDA(const int j) {
          double dot = 0.0;
          for (int k = 0; k < N; ++k) {
            dot += U_local(k, j) * VT_local(j, k);
          }
          if (dot < 0)
            mo_energies(j) = -mo_energies(j);
        });

    // 4. Back-transform: C = X * U
    KokkosBlas::gemm("N", "N", 1.0, _X, _U, 0.0, mo_coeff);

    // 5. Final Sorting
    sort(mo_coeff, mo_energies);
  }

private:
  int _N;
  DeviceView2DLeft _X, _XT_F, _U, _VT;

  void sort(DeviceView2DLeft &mo_coeff, DeviceView1D &mo_energies) {
    int N = _N;
    Kokkos::parallel_for(
        "SerialSort", 1, KOKKOS_LAMBDA(const int) {
          for (int i = 0; i < N - 1; i++) {
            int min_idx = i;
            for (int j = i + 1; j < N; j++) {
              if (mo_energies(j) < mo_energies(min_idx))
                min_idx = j;
            }
            if (min_idx != i) {
              double e_tmp = mo_energies(i);
              mo_energies(i) = mo_energies(min_idx);
              mo_energies(min_idx) = e_tmp;
              for (int k = 0; k < N; k++) {
                double c_tmp = mo_coeff(k, i);
                mo_coeff(k, i) = mo_coeff(k, min_idx);
                mo_coeff(k, min_idx) = c_tmp;
              }
            }
          }
        });
  }
};

} // namespace NuKEXC
