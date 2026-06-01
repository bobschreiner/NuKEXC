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

#include "density.hpp"
#include "grid.hpp"
#include "nukexc/partitioning.hpp"
#include "nukexc_config.hpp"
#include "nukexc_utils.hpp"
#include "stobasis.hpp"

#include <KokkosBlas2_gemv.hpp>
#include <KokkosBlas2_gemv_impl.hpp>
#include <KokkosBlas3_gemm.hpp>
#include <KokkosBlas3_gemm_impl.hpp>

#include <KokkosLapack_gesv.hpp>

#include <Kokkos_Macros.hpp>
#include <Kokkos_MathematicalFunctions.hpp>

namespace NuKEXC {

DeviceView2DLeft compute_lda(const STOBasisSet basis, const FlatGrid grid,
                             const DeviceView2D density_matrix) {

  DeviceView2DLeft result;

  // TODO: Implement LDA
  return result;
}

DeviceView2DLeft compute_gga(const STOBasisSet basis, const FlatGrid grid,
                             const DeviceView2D density_matrix) {

  DeviceView2DLeft result;

  // TODO: Implement GGA

  return result;
}

DeviceView2DLeft compute_mgga(const STOBasisSet basis, const FlatGrid grid,
                              const DeviceView2D density_matrix) {

  DeviceView2DLeft result;

  // TODO: Implement META GGA

  return result;
}

} // namespace NuKEXC
