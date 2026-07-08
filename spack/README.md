# NuKEXC spack environments

## Layout

```
spack-envs/
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
