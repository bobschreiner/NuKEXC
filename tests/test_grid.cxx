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

#include <iomanip>
#include <iostream>
#include <string_view>

#include <catch2/catch_all.hpp>
#include <integratorxx/composite_quadratures/pruned_spherical_quadrature.hpp>

#include <integratorxx/composite_quadratures/pruned_spherical_quadrature.hpp>
#include <integratorxx/composite_quadratures/spherical_quadrature.hpp>
#include <integratorxx/generators/radial_factory.hpp>
#include <integratorxx/generators/spherical_factory.hpp>
#include <integratorxx/quadratures/radial.hpp>
#include <integratorxx/quadratures/s2.hpp>

#include "nukexc/grid.hpp"
#include "standards.hpp"

using namespace NuKEXC;

using bk_type = IntegratorXX::Becke<double, double>;
using mk_type = IntegratorXX::MuraKnowles<double, double>;
using mhl_type = IntegratorXX::MurrayHandyLaming<double, double>;
using ta_type = IntegratorXX::TreutlerAhlrichs<double, double>;

using ah_type = IntegratorXX::AhrensBeylkin<double>;
using de_type = IntegratorXX::Delley<double>;
using ll_type = IntegratorXX::LebedevLaikov<double>;
using wo_type = IntegratorXX::Womersley<double>;

using sph_test_types = std::tuple<
    std::tuple<bk_type, ah_type>, std::tuple<bk_type, de_type>,
    //               std::tuple<bk_type, ll_type>, std::tuple<bk_type, wo_type>,

    //               std::tuple<mk_type, ah_type>, std::tuple<mk_type, de_type>,
    //               std::tuple<mk_type, ll_type>, std::tuple<mk_type, wo_type>,

    //               std::tuple<mhl_type, ah_type>, std::tuple<mhl_type,
    //               de_type>, std::tuple<mhl_type, ll_type>,
    //               std::tuple<mhl_type, wo_type>,

    //               std::tuple<ta_type, ah_type>, std::tuple<ta_type, de_type>,
    std::tuple<ta_type, ll_type>, std::tuple<ta_type, wo_type>>;

TEMPLATE_LIST_TEST_CASE("Unpruned", "[sph-gen]", sph_test_types) {

  using namespace IntegratorXX;
  using radial_type =
      std::decay_t<decltype(std::get<0>(std::declval<TestType>()))>;
  using angular_type =
      std::decay_t<decltype(std::get<1>(std::declval<TestType>()))>;
  using angular_traits = quadrature_traits<angular_type>;

  using spherical_type = SphericalQuadrature<radial_type, angular_type>;

  size_t nrad = 10;
  size_t nang = angular_traits::npts_by_algebraic_order(
      angular_traits::next_algebraic_order(
          3)); // Smallest possible angular grid

  // Generate the quadrature manually
  radial_type rq(nrad, 1.0);
  angular_type aq(nang);
  spherical_type sph_ref(rq, aq);

  // Generate via runtime API
  auto rad_spec = radial_from_type<radial_type>();
  auto rad_traits = make_radial_traits(rad_spec, nrad, 1.0);
  UnprunedSphericalGridSpecification unp(
      rad_spec, *rad_traits, angular_from_type<angular_type>(), nang);

  auto sph = SphericalGridFactory::generate_grid(unp);

  // Check that they're the same
  REQUIRE(sph->npts() == sph_ref.npts());

  const auto npts = sph->npts();
  for (auto i = 0; i < npts; ++i) {
    auto pt = sph->points()[i];
    auto pt_ref = sph_ref.points()[i];
    REQUIRE_THAT(pt[0], Catch::Matchers::WithinAbs(pt_ref[0], 1e-15));
    REQUIRE_THAT(pt[1], Catch::Matchers::WithinAbs(pt_ref[1], 1e-15));
    REQUIRE_THAT(pt[2], Catch::Matchers::WithinAbs(pt_ref[2], 1e-15));

    auto w = sph->weights()[i];
    auto w_ref = sph_ref.weights()[i];
    REQUIRE_THAT(w, Catch::Matchers::WithinAbs(w_ref, 1e-15));
  }
}

void visualize_points_with_tiles(const FlatGrid &grid) {

  Kokkos::View<Point *, ExecSpace> points_dev = grid.quad_points;
  auto pts =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, points_dev);
  const int total = static_cast<int>(pts.extent(0));

  const std::string path = "grid/grid_points.vtk";

  std::filesystem::path p(path);
  std::filesystem::create_directories(p.parent_path());

  std::ofstream out(path);
  if (!out) {
    std::cerr << "[visualize_points_with_tiles] ERROR: cannot open " << path
              << '\n';
    return;
  }

  out << "# vtk DataFile Version 3.0\n"
      << "Grid Points\n"
      << "ASCII\n"
      << "DATASET UNSTRUCTURED_GRID\n\n";

  out << "POINTS " << total << " double\n";
  for (int i = 0; i < total; ++i)
    out << pts(i)[0] << ' ' << pts(i)[1] << ' ' << pts(i)[2] << '\n';

  // Each point is its own VTK_VERTEX cell (type 1)
  out << "\nCELLS " << total << ' ' << total * 2 << '\n';
  for (int i = 0; i < total; ++i)
    out << "1 " << i << '\n';

  out << "\nCELL_TYPES " << total << '\n';
  for (int i = 0; i < total; ++i)
    out << "1\n";

  out << "\nCELL_DATA " << total << '\n'
      << "SCALARS tile_id int 1\n"
      << "LOOKUP_TABLE default\n";

  for (int j = 0; j < grid.nang; ++j)
    for (int i = 0; i < grid.nrad; ++i)
      out << i << '\n';

  out.flush();
  std::cout << "[visualize_points_with_tiles]\n"
            << "  Points : " << total << '\n'
            << "  Output : " << path << '\n'
            << "  Tip    : load alongside bounding_boxes.vtk in ParaView;\n"
            << "           matching tile_id colours confirm points are\n"
            << "           correctly contained within their boxes.\n\n";
}

int main() {

  Kokkos::initialize();
  
  int result = Catch::Session().run();
  {
    Molecule mol(std::vector<std::vector<double>>{{0., 0., 0.}},
                 std::vector<unsigned>{1u});
    auto grid = make_flat_grid<ta_type, ll_type>(mol, 20, 40);
    visualize_points_with_tiles(grid);
  } // grid and mol destroyed here, while Kokkos is still alive
 Kokkos::finalize();
  return result;
}
