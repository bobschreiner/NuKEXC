# NuKEXC — Numerical Kokkos Enhanced Exchange Correlation Integrator

NuKEXC is a header-only C++ library for numerically integrating atomic orbital integrals (overlap, kinetic, nuclear potential, exchange-correlation) on GPUs using Kokkos. It targets Numerical Atomic Orbital (NAO) and Slater-type orbital (STO) basis sets and is designed for use in GPU-accelerated self-consistent field (SCF) calculations.

---

## Requirements

| Requirement | Minimum Version | Notes |
|---|---|---|
| CMake | 3.20 | |
| C++ compiler | C++20 | Clang 14+, GCC 12+, or Cray CC |
| Kokkos | 4.0 | GPU backend required for production use |
| KokkosKernels | 4.0 | Must match Kokkos version exactly |
| IntegratorXX | main branch | Header-only, fetched automatically |
| ROCm (AMD) | 5.6+ | For HIP/AMD GPU backend |
| CUDA (NVIDIA) | 11.8+ | For CUDA/NVIDIA GPU backend |

A CPU-only build is supported for testing but is not recommended for production — the library is designed around GPU execution.

---

## Dependencies

### 1. Kokkos

Kokkos provides the portable parallel programming model. NuKEXC requires a Kokkos installation with either the HIP (AMD) or CUDA (NVIDIA) backend enabled.

**Installing Kokkos with HIP (AMD GPU)**

```bash
git clone https://github.com/kokkos/kokkos.git
cd kokkos
mkdir build && cd build

cmake .. \
  -DCMAKE_INSTALL_PREFIX=/path/to/kokkos/install \
  -DCMAKE_CXX_COMPILER=hipcc \
  -DKokkos_ENABLE_HIP=ON \
  -DKokkos_ARCH_VEGA90A=ON   # replace with your GPU arch: VEGA906, VEGA908, etc.

make -j$(nproc) install
```

**Installing Kokkos with CUDA (NVIDIA GPU)**

```bash
cmake .. \
  -DCMAKE_INSTALL_PREFIX=/path/to/kokkos/install \
  -DCMAKE_CXX_COMPILER=nvcc_wrapper \
  -DKokkos_ENABLE_CUDA=ON \
  -DKokkos_ENABLE_CUDA_LAMBDA=ON \
  -DKokkos_ARCH_AMPERE80=ON   # replace with your arch: VOLTA70, TURING75, HOPPER90, etc.

make -j$(nproc) install
```

**Installing Kokkos for CPU only (testing)**

```bash
cmake .. \
  -DCMAKE_INSTALL_PREFIX=/path/to/kokkos/install \
  -DKokkos_ENABLE_OPENMP=ON

make -j$(nproc) install
```

Common GPU architecture flags:

| GPU | Flag |
|---|---|
| MI250X (LUMI, Frontier) | `-DKokkos_ARCH_VEGA90A=ON` |
| MI100 | `-DKokkos_ARCH_VEGA908=ON` |
| A100 | `-DKokkos_ARCH_AMPERE80=ON` |
| H100 | `-DKokkos_ARCH_HOPPER90=ON` |
| V100 | `-DKokkos_ARCH_VOLTA70=ON` |

---

### 2. KokkosKernels

KokkosKernels provides the BLAS and LAPACK routines (GEMM, SVD) used by NuKEXC. It must be built against the same Kokkos installation.

**Installing KokkosKernels**

```bash
git clone https://github.com/kokkos/kokkos-kernels.git
cd kokkos-kernels
mkdir build && cd build

# HIP example
cmake .. \
  -DCMAKE_INSTALL_PREFIX=/path/to/kokkos-kernels/install \
  -DCMAKE_CXX_COMPILER=hipcc \
  -DKokkos_ROOT=/path/to/kokkos/install \
  -DKokkosKernels_ENABLE_TPL_ROCBLAS=ON \
  -DKokkosKernels_ENABLE_TPL_ROCSOLVER=ON

# CUDA example
cmake .. \
  -DCMAKE_INSTALL_PREFIX=/path/to/kokkos-kernels/install \
  -DCMAKE_CXX_COMPILER=nvcc_wrapper \
  -DKokkos_ROOT=/path/to/kokkos/install \
  -DKokkosKernels_ENABLE_TPL_CUBLAS=ON \
  -DKokkosKernels_ENABLE_TPL_CUSOLVER=ON

make -j$(nproc) install
```

The TPL (third-party library) flags enable vendor BLAS/LAPACK backends. Without them KokkosKernels falls back to a reference implementation that is orders of magnitude slower — always enable the appropriate TPL for your hardware.

---

### 3. IntegratorXX

IntegratorXX provides the radial and angular quadrature grids (Treutler-Ahlrichs radial, Lebedev-Laikov angular) used to build the numerical integration grid. It is a header-only library and is fetched automatically by CMake via `FetchContent` — no manual installation is required.

If you are working offline or on a system without internet access, clone it manually and point CMake to it:

```bash
git clone https://github.com/wavefunction91/IntegratorXX.git /path/to/integratorxx
```

Then pass `-DIntegratorXX_ROOT=/path/to/integratorxx` when configuring NuKEXC.

---

## Building NuKEXC

```bash
git clone https://github.com/bobschreiner/NuKEXC.git
cd NuKEXC
mkdir build && cd build

# HIP / AMD GPU example
cmake .. \
  -DCMAKE_CXX_COMPILER=hipcc \
  -DKokkos_ROOT=/path/to/kokkos/install \
  -DKokkosKernels_ROOT=/path/to/kokkos-kernels/install \
  -DCMAKE_BUILD_TYPE=Release \
  -DNuKEXC_BUILD_TESTING=ON

make -j$(nproc)
```

**CMake options**

| Option | Default | Description |
|---|---|---|
| `NuKEXC_BUILD_TESTING` | ON (top-level), OFF (subproject) | Build the test suite |
| `CMAKE_BUILD_TYPE` | RelWithDebInfo | Debug / Release / RelWithDebInfo |
| `NuKEXC_INSTALL_CMAKEDIR` | `lib/cmake/NuKEXC` | CMake config install location |

---

## Running the Tests

```bash
cd build
ctest --output-on-failure
```

To run the SCF benchmark specifically:

```bash
./tests/benchmark_scf
```

Profiling with the Kokkos simple kernel timer:

```bash
export KOKKOS_TOOLS_LIBS=/path/to/kokkos-tools/build/profiling/simple-kernel-timer/libkp_kernel_timer.so
./tests/benchmark_scf
kp_reader *.dat
```

---

## Using NuKEXC as a Subproject

Add NuKEXC to your project via `FetchContent`:

```cmake
include(FetchContent)
FetchContent_Declare(
  NuKEXC
  GIT_REPOSITORY https://github.com/bobschreiner/NuKEXC.git
  GIT_TAG        main
)
FetchContent_MakeAvailable(NuKEXC)

target_link_libraries(your_target PRIVATE NuKEXC::NuKEXC)
```

Or via a local installation:

```cmake
find_package(NuKEXC REQUIRED)
target_link_libraries(your_target PRIVATE NuKEXC::NuKEXC)
```

---

## Platform Notes

**LUMI (AMD MI250X)**

Use the Cray compiler wrappers and load the appropriate modules before configuring:

```bash
module load PrgEnv-cray craype-accel-amd-gfx90a rocm
export CXX=CC
```

Set `-DKokkos_ARCH_VEGA90A=ON` and enable ROCBlas/ROCSolver TPLs.

**HPC clusters without internet access**

Set `FETCHCONTENT_UPDATES_DISCONNECTED=ON` (already the default in NuKEXC's CMake) and pre-clone IntegratorXX into your source tree or a shared directory. Pass its path via `-DIntegratorXX_ROOT`.

---

## License

NuKEXC is free software distributed under the GNU General Public License v3 or later. See `LICENSE` for details.
