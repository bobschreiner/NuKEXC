// #include <iostream>
#include "nukexc_config.hpp"
#include <Kokkos_Core.hpp>

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  Kokkos::Timer timer;

  using namespace NuKEXC;

  size_t N = Kokkos::exp2(30);
  // Allocate a 1-dimensional view of integers
  Kokkos::View<int *, Layout, ExecSpace> v("v", N);
  // Fill view with sequentially increasing values v=[0,1,2,3,4]
  timer.reset();
  Kokkos::parallel_for("fill", N, KOKKOS_LAMBDA(int i) { v(i) = 1; });
  // Compute accumulated sum of v's elements r=0+1+2+3+4
  int r;
  Kokkos::parallel_reduce(
      "accumulate", N,
      KOKKOS_LAMBDA(int i, int &partial_r) { partial_r += v(i); }, r);
  // Check the result
  KOKKOS_ASSERT(r == N);
  double time = timer.seconds();
  Kokkos::printf("Program took %f seoncds\n", time);
  Kokkos::finalize();

  return 0;
}
