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
#include "nukexc/grid.hpp"
#include "nukexc_config.hpp"
#include "octree.hpp"
#include "stobasis.hpp"

#include <KokkosBatched_Copy_Decl.hpp>
#include <KokkosBatched_Copy_Impl.hpp>
#include <KokkosBatched_Dot.hpp>
#include <KokkosBatched_Gemm_Decl.hpp>
#include <KokkosBatched_Gemm_Team_Impl.hpp>
#include <KokkosBatched_Util.hpp>

#include <KokkosBlas3_gemm.hpp>
#include <Kokkos_Core_fwd.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <Kokkos_Pair.hpp>
#include <decl/Kokkos_Declare_OPENMP.hpp>
#include <impl/Kokkos_Profiling.hpp>

namespace NuKEXC {
DeviceView1D
compute_density(const STOBasisSet basis, const FlatGrid grid,
                Kokkos::View<double **, ExecSpace> density_matrix) {

  ExecSpace space;
  Kokkos::View<Point *, ExecSpace> collocation_points = grid.quad_points;
  const int N_quad = collocation_points.extent(0);
  const int N_bf = basis.nbf();

  DeviceView1D density("density", N_quad);
  DeviceView2D collocation_values("collocation", N_quad, N_bf);
  DeviceView2D intermediate_matrix("intermediate", N_quad, N_bf);

  fill_collocation_transpose(space, basis, collocation_points,
                             collocation_values);
  KokkosBlas::gemm(space, "N", "N", 1.0, collocation_values, density_matrix,
                   0.0, intermediate_matrix);

  Kokkos::parallel_for(
      "Contract Basis", N_quad, KOKKOS_LAMBDA(const int g) {
        double sum = 0;
        for (int i = 0; i < N_bf; ++i) {
          sum += collocation_values(g, i) * intermediate_matrix(g, i);
        }
        density(g) = sum;
      });

  return density;
};
}; // namespace NuKEXC
