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
| ArborX | 2.1 | main branch | Header-only, fetched automatically |
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
  -DKokkosKernels_ENABLE_TPL_ROCSPARSE=ON \
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

The build tree mirrors the three source directories:

| Source | Build output | What lives there |
|---|---|---|
| `tests/` | `build/tests/` | Correctness tests (run by `ctest`) and the `standalone` SCF driver |
| `benchmarking/` | `build/benchmarking/` | Performance benchmarks and the W4-11 accuracy/timing sweep |
| `convergence_studies/` | `build/convergence_studies/` | Grid-convergence sweeps and the scripts that plot them |

Each of the two latter directories has its own README describing how to
reproduce the figures it feeds.

To run an end-to-end SCF with its per-section timing breakdown, use the
`standalone` driver (run it from `build/`, so it finds the copied `input/`):

```bash
./tests/standalone
```

Profiling with the Kokkos simple kernel timer:

```bash
export KOKKOS_TOOLS_LIBS=/path/to/kokkos-tools/build/profiling/simple-kernel-timer/libkp_kernel_timer.so
./tests/standalone
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
 
### LUMI (AMD MI250X)
 
On LUMI, Kokkos and KokkosKernels must be installed via EasyBuild using the provided easyconfig files. Do **not** attempt a manual CMake install — the Cray PE toolchain requires specific flags that are encoded in the easyconfigs.
 
**Prerequisites**
 
EasyBuild 5.1.2 or later must be available. The builds use the `cpeAMD/25.03` toolchain and depend on the `rocm` external module, both of which are available on LUMI-G nodes.
 
**Step 1 — Copy the easyconfig files**
 
The two easyconfig files are provided in the `lumi/` directory of this repository:
 
```
lumi/Kokkos-4.6.02-cpeAMD-25.03-rocm.eb
lumi/Kokkos-kernels-4.6.02-cpeAMD-25.03-rocm.eb
```
 
**Step 2 — Set up EasyBuild for your user installation**
 
```bash
module load LUMI/25.03 partition/G
module load EasyBuild-user
```
 
**Step 3 — Install Kokkos**
 
Kokkos must be installed before KokkosKernels. Run from a login node (`uan`):
 
```bash
eb Kokkos-4.6.02-cpeAMD-25.03-rocm.eb
```
 
This builds Kokkos 4.6.02 with the following configuration:
- Compiler: `${ROCM_PATH}/bin/hipcc`
- GPU arch: `AMD_GFX90A` (MI250X)
- CPU arch: `ZEN3`
- HIP + OpenMP backends enabled
- `HIP_MULTIPLE_KERNEL_INSTANTIATIONS` enabled for better GPU occupancy
- Build time: approximately 72 seconds
**Step 4 — Install KokkosKernels**
 
```bash
eb Kokkos-kernels-4.6.02-cpeAMD-25.03-rocm.eb
```
 
This builds KokkosKernels 4.6.02 with:
- `Kokkos_ROOT` set automatically from the installed Kokkos module
- ROCBlas, ROCSparse, and ROCSolver TPLs enabled — these are required for GPU-accelerated GEMM and SVD
- Host BLAS and LAPACK also enabled
- Build time: approximately 7 minutes
**Step 5 — Load the modules**
 
After installation, load the modules before configuring NuKEXC:
 
```bash
module load Kokkos/4.6.02-cpeAMD-25.03-rocm
module load Kokkos-kernels/4.6.02-cpeAMD-25.03-rocm
```
 
**Step 6 — Configure and build NuKEXC**
 
```bash
cd NuKEXC
mkdir build && cd build
 
cmake .. \
  -DCMAKE_CXX_COMPILER=CC \
  -DKokkos_ROOT=${EBROOTKOKKOS} \
  -DKokkosKernels_ROOT=${EBROOTKOKKOSKERNELS} \
  -DCMAKE_BUILD_TYPE=Release \
  -DNuKEXC_BUILD_TESTING=ON \
  -DAMDGPU_TARGETS="gfx90a" \
  -DOpenMP_CXX_FLAGS="-fopenmp" \
  -DOpenMP_CXX_LIB_NAMES="libomp" \
  -DOpenMP_libomp_LIBRARY="/opt/rocm-6.3.4/llvm/lib/libomp.so"
 
make -j16
```
 
Note that `craype-accel-amd-gfx90a` is explicitly unloaded during the Kokkos and KokkosKernels builds (as specified in the easyconfigs) to avoid conflicts with `hipcc`. NuKEXC itself uses the Cray `CC` wrapper which handles this automatically.
 
**HPC clusters without internet access**
 
`FETCHCONTENT_UPDATES_DISCONNECTED=ON` is already set in NuKEXC's CMake, so FetchContent will not attempt network access if the source is already present. On LUMI, pre-clone IntegratorXX into your source tree or a shared project directory before configuring:
 
```bash
git clone https://github.com/wavefunction91/IntegratorXX.git /path/to/integratorxx
```
 
Then pass `-DIntegratorXX_ROOT=/path/to/integratorxx` to CMake.

---

## License

NuKEXC is free software distributed under the GNU General Public License v3 or later. See `LICENSE` for details.
