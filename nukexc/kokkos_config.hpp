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

//using Layout = Kokkos::LayoutLeft;
using Layout = Kokkos::LayoutRight;
const double epsilon_shift = 1e-30;
} // namespace NuKEXC
