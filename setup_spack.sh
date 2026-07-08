#!/usr/bin/env bash
set -euo pipefail
# Run this from inside your NuKEXC repo root.
rm -rf spack
mkdir -p spack/common spack/cpu spack/cuda spack/rocm spack/sycl
mkdir -p spack/repo/packages/integratorxx spack/repo/packages/open-orbital-optimizer

mkdir -p "$(dirname 'spack/README.md')"
cat > 'spack/README.md' << 'NUKEXC_EOF'
# NuKEXC spack environments

## Layout

```
spack/
      common/packages.yaml   site config: externals (CUDA/ROCm/oneAPI toolkits, MPI, target arch)
      repo/                  custom spack package repo (IntegratorXX, OpenOrbitalOptimizer
                              are not in the spack builtin repo)
      cpu/spack.yaml         library-only deps, Kokkos::Serial+OpenMP, LAPACK/OpenBLAS
      cpu/spack.dev.yaml     same + test deps (Catch2, Armadillo, OpenOrbitalOptimizer)
      cuda/spack.yaml        library-only, Kokkos::Cuda, cuBLAS/cuSOLVER
      cuda/spack.dev.yaml    + test deps
      rocm/spack.yaml        library-only, Kokkos::HIP, rocBLAS/rocSOLVER
      rocm/spack.dev.yaml    + test deps
      sycl/spack.yaml         library-only, Kokkos::SYCL, oneMKL
      sycl/spack.dev.yaml     + test deps

```

One directory = one target backend = one compiler toolchain. That's the
mechanism for "depends on the compiler being used": you don't pick the
backend dynamically inside a single spack.yaml (spack specs are static,
concretized ahead of time) - you pick which *environment* to activate, and
each environment pins the matching compiler (`%gcc`, `%rocmcc`, `%oneapi`)
together with the matching Kokkos/Kokkos-Kernels/ArborX variants. This also
matches how HPC clusters normally work: the GPU-backend module you load
(cuda vs rocm vs oneapi) already tells you which compiler you're using.

The `.dev.yaml` files are identical to `spack.yaml` plus the packages that
only `tests/CMakeLists.txt` needs (Catch2, Armadillo, OpenOrbitalOptimizer).
Production/library-only installs should never pull those in.

## First-time setup (per cluster)

```bash
# 1. Point spack at compilers already on the system (or module-load them first)
spack compiler find

# 2. Fill in common/packages.yaml with real externals:
spack external find cuda hip mpi     # on a node with the relevant module loaded
#    then paste results into common/packages.yaml, plus set `target:` to your
#    cluster's microarchitecture (see `spack arch`).

# 3. Edit cuda/*.yaml (cuda_arch=) and rocm/*.yaml (amdgpu_target=) for your GPU.

# 4. Pin real versions/tags for integratorxx and open-orbital-optimizer in
#    repo/packages/*/package.py once you know which commit/tag NuKEXC needs.
```

## Building the library-only dependency set (e.g. on a CUDA cluster)

```bash
spack env create nukexc-cuda cuda/spack.yaml
spack env activate nukexc-cuda
spack concretize -f
spack install
```

This creates a filesystem view (`view: true`) inside the environment, so
`spack env activate nukexc-cuda` also sets `CMAKE_PREFIX_PATH` for you.
Then just:

```bash
cd nukexc
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DNuKEXC_BUILD_TESTING=OFF
cmake --build build -j
```

`NuKEXC_BUILD_TESTING=OFF` skips `add_subdirectory(tests)` entirely, so
Catch2/Armadillo/OpenOrbitalOptimizer are never even searched for - matching
the fact that the `cuda/spack.yaml` environment never installed them.

## Building with tests (dev workflow)

```bash
spack env create nukexc-cuda-dev cuda/spack.dev.yaml
spack env activate nukexc-cuda-dev
spack install

cd nukexc
cmake -S . -B build -DNuKEXC_BUILD_TESTING=ON   # default at top level anyway
cmake --build build -j
ctest --test-dir build
```

Repeat with `cpu/`, `rocm/`, `sycl/` on the respective partitions/login
nodes. If your cluster has all backends visible from one login node, you can
have all four environments installed side by side; just `spack env
activate` the one matching the partition you're about to build/run on.

## Testing this locally on a laptop first

You don't need HPC access or a GPU to validate that these environments are
well-formed. Spack itself is just a Python-based dependency resolver, so
concretization (does the dependency graph resolve at all, are variant names
valid, do version constraints conflict) works anywhere:

```bash
git clone -c feature.manyFiles=true https://github.com/spack/spack.git ~/spack
. ~/spack/share/spack/setup-env.sh   # add to your shell rc file
spack compiler find
spack external find                  # picks up cmake/python already installed
```

Then, per environment:

```bash
cd cpu   # or cuda/, rocm/, sycl/
spack env create nukexc-cpu spack.yaml
spack env activate nukexc-cpu
spack repo add ../repo
spack concretize -f
```

- `cpu/spack.yaml` will fully **build** on a laptop (`spack install`) - this
  is the best end-to-end check, including the two custom recipes in `repo/`.
- `cuda/`, `rocm/`, `sycl/` will **concretize** fine without the actual
  hardware/toolchain, which is enough to catch bad variant names or version
  conflicts, but `spack install` will fail without `nvcc`/`hipcc`/`icpx`.
- `common/packages.yaml`'s externals (CUDA/ROCm/oneAPI prefixes) point at
  cluster paths that won't exist locally - comment those blocks out (or
  just don't `install` those environments) when testing on a laptop.

## Things to double-check before relying on this

- **Variant names**: `kokkos-kernels` TPL variant names (`cublas`,
  `cusolver`, `rocblas`, `rocsolver`, `mkl`, `blas`, `lapack`) and `arborx`'s
  backend variants (`cuda`, `rocm`, `sycl`) can change between spack
  releases - run `spack info kokkos-kernels` / `spack info arborx` on your
  spack version and adjust the specs if a variant has been renamed.
- **libxc CUDA/HIP offload**: current libxc is CPU-only in spack; it's
  listed identically in all four environments on purpose. If you later need
  GPU-resident libxc evaluations, that's a separate, non-spack concern.
- **SYCL + Kokkos-Kernels**: there's no native SYCL "solver" TPL analogous
  to cuSOLVER/rocSOLVER; the sycl env falls back to oneMKL. Confirm that's
  sufficient for what `nukexc-kokkos-kernels.cmake` actually calls into.
- **`integratorxx` / `open-orbital-optimizer` package.py**: these are
  placeholder `CMakePackage` recipes pointed at the upstream `master`
  branches with guessed CMake option names
  (`INTEGRATORXX_ENABLE_TESTS`, `OPENORBITALOPTIMIZER_ENABLE_TESTS`).
  Check each project's actual `CMakeLists.txt` and pin a real
  version/tag before using this in production.
- **MPI**: none of the specs above request `+mpi` on Kokkos/ArborX/Kokkos-
  Kernels. Add it (and route it through `common/packages.yaml`'s `mpi:`
  provider) if NuKEXC needs distributed-memory builds.
NUKEXC_EOF

mkdir -p "$(dirname 'spack/common/packages.yaml')"
cat > 'spack/common/packages.yaml' << 'NUKEXC_EOF'
packages:
  all:
    target: [x86_64_v3]        # set to your cluster's actual microarch, e.g. zen3, icelake
    providers:
      mpi: [cray-mpich, openmpi, mpich]

  # --- Point spack at the cluster-provided CUDA toolkit instead of building
  # one from source. Run `spack external find cuda` on a GPU node and paste
  # the result here, or edit the prefix manually.
  cuda:
    externals:
    - spec: cuda@12.4
      prefix: /usr/local/cuda-12.4
    buildable: false

  # --- Same idea for ROCm.
  hip:
    externals:
    - spec: hip@6.0
      prefix: /opt/rocm-6.0.0
    buildable: false
  hsa-rocr-dev:
    externals:
    - spec: hsa-rocr-dev@6.0
      prefix: /opt/rocm-6.0.0
    buildable: false
  llvm-amdgpu:
    externals:
    - spec: llvm-amdgpu@6.0
      prefix: /opt/rocm-6.0.0
    buildable: false

  # --- Intel oneAPI compilers/MKL for the SYCL backend (icpx).
  intel-oneapi-compilers:
    externals:
    - spec: intel-oneapi-compilers@2024.1
      prefix: /opt/intel/oneapi
    buildable: false
  intel-oneapi-mkl:
    externals:
    - spec: intel-oneapi-mkl@2024.1
      prefix: /opt/intel/oneapi
    buildable: false

  # --- Cray/vendor MPI, if applicable
  # mpich:
  #   externals:
  #   - spec: mpich@8.1.28
  #     prefix: /opt/cray/pe/mpich/8.1.28/ofi/gnu/9.1
  #   buildable: false
NUKEXC_EOF

mkdir -p "$(dirname 'spack/cpu/spack.yaml')"
cat > 'spack/cpu/spack.yaml' << 'NUKEXC_EOF'
# NuKEXC library dependencies - CPU backend (Kokkos::Serial/OpenMP + LAPACK)
spack:
  include:
  - ../common/packages.yaml

  repos:
  - ../repo

  concretizer:
    unify: true

  compilers: []   # rely on `spack compiler find`; pin below if you need a
                  # specific toolchain, e.g. append %gcc@12.3.0 to every spec

  specs:
  - kokkos@4 +openmp ~cuda ~rocm ~sycl cxxstd=20 %gcc
  - kokkos-kernels@4 +openmp +blas +lapack
    ^kokkos +openmp ~cuda ~rocm ~sycl
    ^openblas threads=openmp
  - arborx@1 +openmp ~cuda ~rocm ~sycl
    ^kokkos +openmp ~cuda ~rocm ~sycl
  - libxc@6 +cxx
  - integratorxx@master

  view: true
NUKEXC_EOF

mkdir -p "$(dirname 'spack/cpu/spack.dev.yaml')"
cat > 'spack/cpu/spack.dev.yaml' << 'NUKEXC_EOF'
# Same as spack.yaml, plus the packages that are ONLY needed to build
# nukexc/tests. Use this env on a dev/login node when you want to run
# `ctest`; use the plain spack.yaml for production/library-only installs.
spack:
  include:
  - ../common/packages.yaml

  repos:
  - ../repo

  concretizer:
    unify: true

  specs:
  - kokkos@4 +openmp ~cuda ~rocm ~sycl cxxstd=20 %gcc
  - kokkos-kernels@4 +openmp +blas +lapack
    ^kokkos +openmp ~cuda ~rocm ~sycl
    ^openblas threads=openmp
  - arborx@1 +openmp ~cuda ~rocm ~sycl
    ^kokkos +openmp ~cuda ~rocm ~sycl
  - libxc@6 +cxx
  - integratorxx@master
  # --- test-only, from here down ---
  - catch2@3
  - armadillo ^openblas threads=openmp
  - open-orbital-optimizer

  view: true
NUKEXC_EOF

mkdir -p "$(dirname 'spack/cuda/spack.yaml')"
cat > 'spack/cuda/spack.yaml' << 'NUKEXC_EOF'
# NuKEXC library dependencies - NVIDIA GPU backend (Kokkos::Cuda + cuSOLVER)
#
# Set CUDA_ARCH to match your cluster's GPU:
#   70 = V100, 80 = A100, 90 = H100/H200, 90a = GH200
spack:
  include:
  - ../common/packages.yaml

  repos:
  - ../repo

  concretizer:
    unify: true

  # EDIT ME: set to your GPU's cuda_arch (70/80/90/90a/...). It must be
  # identical on every line below - spack.yaml has no variable
  # interpolation, so this is a plain find-and-replace.
  specs:
  - kokkos@4 +cuda +cuda_lambda +wrapper ~rocm ~sycl cxxstd=20
    cuda_arch=80
    %gcc
    ^cuda@12
  - kokkos-kernels@4 +cublas +cusolver +cusparse
    ^kokkos +cuda +wrapper cuda_arch=80 ~rocm ~sycl
    ^cuda@12
  - arborx@1 +cuda ~rocm ~sycl
    ^kokkos +cuda +wrapper cuda_arch=80 ~rocm ~sycl
    ^cuda@12
  - libxc@6 +cxx
  - integratorxx@master

  view: true
NUKEXC_EOF

mkdir -p "$(dirname 'spack/cuda/spack.dev.yaml')"
cat > 'spack/cuda/spack.dev.yaml' << 'NUKEXC_EOF'
# Same as spack.yaml, plus test-only packages. cuda_arch must match spack.yaml.
spack:
  include:
  - ../common/packages.yaml

  repos:
  - ../repo

  concretizer:
    unify: true

  specs:
  - kokkos@4 +cuda +cuda_lambda +wrapper ~rocm ~sycl cxxstd=20
    cuda_arch=80
    %gcc
    ^cuda@12
  - kokkos-kernels@4 +cublas +cusolver +cusparse
    ^kokkos +cuda +wrapper cuda_arch=80 ~rocm ~sycl
    ^cuda@12
  - arborx@1 +cuda ~rocm ~sycl
    ^kokkos +cuda +wrapper cuda_arch=80 ~rocm ~sycl
    ^cuda@12
  - libxc@6 +cxx
  - integratorxx@master
  # --- test-only, from here down ---
  - catch2@3
  - armadillo ^openblas threads=openmp
  - open-orbital-optimizer

  view: true
NUKEXC_EOF

mkdir -p "$(dirname 'spack/rocm/spack.yaml')"
cat > 'spack/rocm/spack.yaml' << 'NUKEXC_EOF'
# NuKEXC library dependencies - AMD GPU backend (Kokkos::HIP + rocSOLVER)
#
# EDIT amdgpu_target to match your cluster's GPU, e.g.:
#   gfx90a = MI210/MI250X, gfx940/942 = MI300A/X
spack:
  include:
  - ../common/packages.yaml

  repos:
  - ../repo

  concretizer:
    unify: true

  specs:
  - kokkos@4 +rocm ~cuda ~sycl cxxstd=20
    amdgpu_target=gfx90a
    %rocmcc
    ^hip@6
  - kokkos-kernels@4 +rocblas +rocsolver +rocsparse
    ^kokkos +rocm amdgpu_target=gfx90a ~cuda ~sycl
    ^hip@6
  - arborx@1 +rocm ~cuda ~sycl
    ^kokkos +rocm amdgpu_target=gfx90a ~cuda ~sycl
    ^hip@6
  - libxc@6 +cxx %rocmcc
  - integratorxx@master %rocmcc

  view: true
NUKEXC_EOF

mkdir -p "$(dirname 'spack/rocm/spack.dev.yaml')"
cat > 'spack/rocm/spack.dev.yaml' << 'NUKEXC_EOF'
# Same as spack.yaml, plus test-only packages. amdgpu_target must match spack.yaml.
spack:
  include:
  - ../common/packages.yaml

  repos:
  - ../repo

  concretizer:
    unify: true

  specs:
  - kokkos@4 +rocm ~cuda ~sycl cxxstd=20
    amdgpu_target=gfx90a
    %rocmcc
    ^hip@6
  - kokkos-kernels@4 +rocblas +rocsolver +rocsparse
    ^kokkos +rocm amdgpu_target=gfx90a ~cuda ~sycl
    ^hip@6
  - arborx@1 +rocm ~cuda ~sycl
    ^kokkos +rocm amdgpu_target=gfx90a ~cuda ~sycl
    ^hip@6
  - libxc@6 +cxx %rocmcc
  - integratorxx@master %rocmcc
  # --- test-only, from here down ---
  - catch2@3 %rocmcc
  - armadillo %rocmcc
  - open-orbital-optimizer %rocmcc

  view: true
NUKEXC_EOF

mkdir -p "$(dirname 'spack/sycl/spack.yaml')"
cat > 'spack/sycl/spack.yaml' << 'NUKEXC_EOF'
# NuKEXC library dependencies - Intel GPU backend (Kokkos::SYCL + MKL/LAPACK)
#
# Requires Intel oneAPI (icpx) registered as an external compiler - see
# ../common/packages.yaml. Kokkos-Kernels does not have a "cusolver-style"
# native SYCL solver TPL, so it falls back to oneMKL for LAPACK-equivalent
# functionality on Intel GPUs.
spack:
  include:
  - ../common/packages.yaml

  repos:
  - ../repo

  concretizer:
    unify: true

  specs:
  - kokkos@4 +sycl ~cuda ~rocm cxxstd=20 %oneapi
  - kokkos-kernels@4 +mkl
    ^kokkos +sycl ~cuda ~rocm %oneapi
    ^intel-oneapi-mkl
  - arborx@1 +sycl ~cuda ~rocm
    ^kokkos +sycl ~cuda ~rocm %oneapi
  - libxc@6 +cxx %oneapi
  - integratorxx@master %oneapi

  view: true
NUKEXC_EOF

mkdir -p "$(dirname 'spack/sycl/spack.dev.yaml')"
cat > 'spack/sycl/spack.dev.yaml' << 'NUKEXC_EOF'
spack:
  include:
  - ../common/packages.yaml

  repos:
  - ../repo

  concretizer:
    unify: true

  specs:
  - kokkos@4 +sycl ~cuda ~rocm cxxstd=20 %oneapi
  - kokkos-kernels@4 +mkl
    ^kokkos +sycl ~cuda ~rocm %oneapi
    ^intel-oneapi-mkl
  - arborx@1 +sycl ~cuda ~rocm
    ^kokkos +sycl ~cuda ~rocm %oneapi
  - libxc@6 +cxx %oneapi
  - integratorxx@master %oneapi
  # --- test-only, from here down ---
  - catch2@3 %oneapi
  - armadillo %oneapi
  - open-orbital-optimizer %oneapi

  view: true
NUKEXC_EOF

mkdir -p "$(dirname 'spack/repo/repo.yaml')"
cat > 'spack/repo/repo.yaml' << 'NUKEXC_EOF'
repo:
  namespace: nukexc
NUKEXC_EOF

mkdir -p "$(dirname 'spack/repo/packages/integratorxx/package.py')"
cat > 'spack/repo/packages/integratorxx/package.py' << 'NUKEXC_EOF'
from spack.package import *


class Integratorxx(CMakePackage):
    """IntegratorXX: quadrature rules for numerical integration of atomic
    orbitals / molecular grids (radial + angular quadratures)."""

    homepage = "https://github.com/wavefunction91/IntegratorXX"
    git = "https://github.com/wavefunction91/IntegratorXX.git"

    maintainers("YOUR_GITHUB_HANDLE")

    # Pin to a real tag/commit once you know which one NuKEXC needs.
    version("master", branch="master")
    # version("0.1.0", tag="v0.1.0")

    variant("shared", default=True, description="Build shared libraries")

    depends_on("cxx", type="build")
    depends_on("cmake@3.20:", type="build")

    def cmake_args(self):
        args = [
            self.define_from_variant("BUILD_SHARED_LIBS", "shared"),
            # IntegratorXX has its own test suite driven by a bundled
            # FetchContent'd Catch2 - disable it when building as a
            # dependency to avoid pulling in unrelated network fetches.
            self.define("INTEGRATORXX_ENABLE_TESTS", False),
        ]
        return args
NUKEXC_EOF

mkdir -p "$(dirname 'spack/repo/packages/open-orbital-optimizer/package.py')"
cat > 'spack/repo/packages/open-orbital-optimizer/package.py' << 'NUKEXC_EOF'
from spack.package import *


class OpenOrbitalOptimizer(CMakePackage):
    """OpenOrbitalOptimizer: SCF orbital optimization library used by
    NuKEXC's standalone_hf test/benchmark executable."""

    homepage = "https://github.com/susilehtola/OpenOrbitalOptimizer"
    git = "https://github.com/susilehtola/OpenOrbitalOptimizer.git"

    maintainers("YOUR_GITHUB_HANDLE")

    version("master", branch="master")
    # version("0.1.0", tag="v0.1.0")

    depends_on("cxx", type="build")
    depends_on("cmake@3.20:", type="build")
    depends_on("armadillo")

    def cmake_args(self):
        return [
            self.define("OPENORBITALOPTIMIZER_ENABLE_TESTS", False),
        ]
NUKEXC_EOF

echo "Done. spack/ and README.md created."


