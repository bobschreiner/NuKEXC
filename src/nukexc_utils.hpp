/*
 *    NuKEXC Numerical Kokkos Enhanced Exchange Correlation Integrator 
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

#include <Kokkos_Core.hpp>

namespace NuKEXC {
namespace utils {
KOKKOS_INLINE_FUNCTION
double rad_dist(const Kokkos::View<double *, Kokkos::LayoutStride> &a,
                const Kokkos::View<double *, Kokkos::LayoutStride> &b) {
  double dist = 0;
  for (int i = 0; i < a.extent(0); ++i) {
    dist += std::pow(a(i) - b(i), 2);
  }
  dist = std::sqrt(dist);
  return dist;
}

} // namespace utils
} // namespace NuKEXC
