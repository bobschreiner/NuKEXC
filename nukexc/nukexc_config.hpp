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
