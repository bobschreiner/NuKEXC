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

#include <ArborX.hpp>
#include <Kokkos_Core.hpp>

namespace NuKEXC {
// using ExecSpace = Kokkos::Serial;
// using ExecSpace = Kokkos::Threads;
// using ExecSpace = Kokkos::OpenMP;
// using ExecSpace = Kokkos::Cuda;
// using ExecSpace = Kokkos::HIP;
using ExecSpace = Kokkos::DefaultExecutionSpace;

using HostSpace = Kokkos::HostSpace;
// using MemSpace = Kokkos::OpenMP;
// using MemSpace = Kokkos::CudaSpace;
// using MemSpace = Kokkos::CudaUVMSpace;
// using MemSpace = Kokkos::HIPSpace;
using MemSpace = ExecSpace::memory_space;

// using Layout = Kokkos::LayoutLeft;
using Layout = Kokkos::LayoutRight;
const double epsilon_shift = 1e-30;

// Standard Views
using View1D = Kokkos::View<double *>;
using DeviceView2D = Kokkos::View<double **, ExecSpace>;
using HostView2D = Kokkos::View<double **, HostSpace>;

// Device Views with specific layout
using DeviceView2DLeft = Kokkos::View<double **, Kokkos::LayoutLeft,
                                      ExecSpace>; // LAPACK requires LeftLayout
using DeviceView2DRight =
    Kokkos::View<double **, Kokkos::LayoutRight, ExecSpace>;
using DeviceView1D = Kokkos::View<double *, ExecSpace>;

// Host Views with specific layout
using HostView2DLeft = Kokkos::View<double **, Kokkos::LayoutLeft, HostSpace>;
using HostView2DRight = Kokkos::View<double **, Kokkos::LayoutRight, HostSpace>;
using HostView1D = Kokkos::View<double *, HostSpace>;

// Geometry definintions
using Point = ArborX::Point<3, double>;
using Box = ArborX::Box<3, double>;

} // namespace NuKEXC
