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

#include "standards.hpp"
#include <catch2/catch_all.hpp>
#include <catch2/catch_test_macros.hpp>
#include <nukexc/nukexc_config.hpp>
#include <nukexc/octree.hpp>
#include <nukexc/stobasis.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

using namespace Nukexc;

TEST_CASE("NeighborList: benzene regression",
          "[octree][benzene][regression]") {
  // Screening must actually discard basis functions: the neighbor list has to
  // come out strictly smaller than the dense num_boxes * N product. Benzene at
  // a 1e-6 cutoff currently drops ~18% of the pairs, which is too close to the
  // noise to pin to a fixed percentage -- so the assertion is just "screening
  // removed something" and the measured figure is printed for tracking. The
  // taxol case below carries the hard sparsity floor.
  using ta_type = IntegratorXX::TreutlerAhlrichs<double, double>;
  using ll_type = IntegratorXX::LebedevLaikov<double>;

  Molecule mol = make_benzene();
  STOBasisSet basis = load_adf_basis(mol, "input/zorabasis/QZ4P", 1e-6);
  FlatGrid grid = make_flat_grid<ta_type, ll_type>(mol, 30, 30);

  const int points_per_box = 8;

  auto bb = create_bounding_boxes(grid, points_per_box);

  NeighborList nl;
  build_neighbor_list(basis, bb, points_per_box, grid.quad_points.extent(0),
                      nl, true, "benzene_histogram");

  auto offsets_h =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, nl.offsets);

  const int64_t num_boxes = static_cast<int64_t>(bb.extent(0));
  const int64_t N = static_cast<int64_t>(basis.nbf());
  const int64_t dense = num_boxes * N;
  const int64_t screened = offsets_h(num_boxes);
  const double sparsity = 1.0 - static_cast<double>(screened) / dense;

  std::cout << "[benzene regression]\n"
            << "  boxes    : " << num_boxes << '\n'
            << "  basis    : " << N << '\n'
            << "  dense    : " << dense << '\n'
            << "  screened : " << screened << '\n'
            << "  sparsity : " << std::fixed << std::setprecision(1)
            << sparsity * 100 << "%\n";

  // Screening removed work -- the property this regression test exists for.
  CHECK(screened < dense);

  // nl.offsets is a CSR prefix array, so it must be non-decreasing.
  for (int b = 0; b < num_boxes; ++b)
    CHECK(offsets_h(b + 1) >= offsets_h(b));
}

TEST_CASE("NeighborList: taxol regression",
          "[octree][neighborlist][regression]") {
  // Same screening regression as the benzene case above, on a molecule large
  // enough for a hard sparsity floor to be meaningful.
  //
  // NOTE: GPU-only. On a CPU build the whole body is preprocessed away and this
  // test case passes without exercising anything.
#if (defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_CUDA))
  using ta_type = IntegratorXX::TreutlerAhlrichs<double, double>;
  using ll_type = IntegratorXX::LebedevLaikov<double>;

  Molecule mol = make_taxol();
  STOBasisSet basis = load_adf_basis(mol, "input/zorabasis/TZP", 1e-6);
  FlatGrid grid = make_flat_grid<ta_type, ll_type>(mol, 30, 30);

  const int points_per_box = 8;

  auto bb = create_bounding_boxes(grid, points_per_box);
  NeighborList nl;
  build_neighbor_list(basis, bb, points_per_box, grid.quad_points.extent(0),
                      nl, true, "taxol_histogram");

  auto offsets_h =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, nl.offsets);

  const int64_t num_boxes = static_cast<int64_t>(bb.extent(0));
  const int64_t N = static_cast<int64_t>(basis.nbf());
  const int64_t dense = num_boxes * N;
  const int64_t screened = offsets_h(num_boxes);
  const double sparsity = 1.0 - static_cast<double>(screened) / dense;

  std::cout << "[taxol regression]\n"
            << "  boxes    : " << num_boxes << '\n'
            << "  basis    : " << N << '\n'
            << "  dense    : " << dense << '\n'
            << "  screened : " << screened << '\n'
            << "  sparsity : " << std::fixed << std::setprecision(1)
            << sparsity * 100 << "%\n";

  // Taxol on TZP at a 1e-6 cutoff: expect at least 30% of the dense pairs gone.
  CHECK(sparsity > 0.3);

  // nl.offsets is a CSR prefix array, so it must be non-decreasing.
  for (int b = 0; b < num_boxes; ++b)
    CHECK(offsets_h(b + 1) >= offsets_h(b));
#endif
}

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  {
    int result = Catch::Session().run(argc, argv);
  }
  Kokkos::finalize();
  return 0;
}
