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

namespace Nukexc {

// Maximal angular momentum

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

// Function qualifier for host+device functions that must NOT be inlined into
// their caller -- use it exactly like KOKKOS_INLINE_FUNCTION. Forcing a real
// call boundary keeps a register-heavy callee's temporaries confined to its own
// frame instead of inflating the caller kernel's simultaneously-live register
// set, capping the kernel's register count at max-over-call-graph rather than
// the inlined sum (this is what buys occupancy on GPU).
//
// The `inline` inherited from KOKKOS_INLINE_FUNCTION is a *linkage* specifier
// (it makes this header definition ODR-safe across translation units); the
// noinline attribute is a *codegen* directive. They act on different axes and
// do not conflict.
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
#define NUKEXC_NOINLINE_FUNCTION KOKKOS_INLINE_FUNCTION __noinline__
#else
#define NUKEXC_NOINLINE_FUNCTION                                                \
  KOKKOS_INLINE_FUNCTION __attribute__((noinline))
#endif

// Min blocks/SM target for the tiled integral kernels' Kokkos::LaunchBounds.
// The kernels launch 128-thread blocks; on sm_90 (65536 regs/SM) this caps the
// per-thread register count: 8 -> <=64 regs (50% occupancy), 7 -> <=73 (43.75%),
// 6 -> <=85 (37.5%). Raising it forces ptxas to spill toward the target -- cheap
// here because these kernels leave the memory subsystem idle. Tune + rebuild to
// sweep the occupancy/spill trade-off. (No-op on CPU backends.)
#ifndef NUKEXC_TILED_MIN_BLOCKS
#define NUKEXC_TILED_MIN_BLOCKS 8
#endif

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
using DeviceView1DLeft = Kokkos::View<double *, Kokkos::LayoutLeft, ExecSpace>;
using DeviceView1DRight =
    Kokkos::View<double *, Kokkos::LayoutRight, ExecSpace>;
// Host Views with specific layout
using HostView2DLeft = Kokkos::View<double **, Kokkos::LayoutLeft, HostSpace>;
using HostView2DRight = Kokkos::View<double **, Kokkos::LayoutRight, HostSpace>;
using HostView1D = Kokkos::View<double *, HostSpace>;

// Geometry definintions
using Point = ArborX::Point<3, double>;
using Box = ArborX::Box<3, double>;

} // namespace Nukexc
