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

#include "standards.hpp"

using namespace NuKEXC;

// Source - https://stackoverflow.com/a/20170989
// Posted by Howard Hinnant, modified by community. See post 'Timeline' for
// change history Retrieved 2026-04-02, License - CC BY-SA 4.0

template <class T> constexpr std::string_view type_name() {
  using namespace std;
#ifdef __clang__
  string_view p = __PRETTY_FUNCTION__;
  return string_view(p.data() + 34, p.size() - 34 - 1);
#elif defined(__GNUC__)
  string_view p = __PRETTY_FUNCTION__;
#if __cplusplus < 201402
  return string_view(p.data() + 36, p.size() - 36 - 1);
#else
  return string_view(p.data() + 49, p.find(';', 49) - 49);
#endif
#elif defined(_MSC_VER)
  string_view p = __FUNCSIG__;
  return string_view(p.data() + 84, p.size() - 84 - 7);
#endif
}
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

  std::cout << type_name<spherical_type>() << std::endl;
  std::cout << "---------------------------------------------------------------"
               "----------------------------"
            << std::endl;
  std::cout << std::setw(20) << "x" << std::setw(20) << "y" << std::setw(20)
            << "z" << std::setw(20) << "w" << std::endl;
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
    std::cout << std::setw(20) << pt[0] << std::setw(20) << pt[1]
              << std::setw(20) << pt[2] << std::setw(20) << w << std::endl;
  }
}

int main() {
  int result = Catch::Session().run();
  return result;
}
