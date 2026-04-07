#pragma once

#include <Kokkos_Core.hpp>

namespace NuKEXC {
// using ExecSpace = Kokkos::Serial;
// using ExecSpace = Kokkos::Threads;
using ExecSpace = Kokkos::OpenMP;
// using ExecSpace = Kokkos::Cuda;
// using ExecSpace = Kokkos::HIP;

// using MemSpace = Kokkos::HostSpace;
using MemSpace = Kokkos::OpenMP;
// using MemSpace = Kokkos::CudaSpace;
// using MemSpace = Kokkos::CudaUVMSpace;
// using MemSpace = Kokkos::HIPSpace;

using Layout = Kokkos::LayoutLeft;
//using Layout = Kokkos::LayoutRight;
} // namespace NuKEXC
