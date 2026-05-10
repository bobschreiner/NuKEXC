/**NuKEXC-- Numerical Kokkos Enhanced Exchange Correlation Integrator *
            Copyright(C) 2026 Bob Schreiner **This program is free software
    : you can redistribute it and /
    or modify *it under the terms of the GNU General Public License as published
    by *the Free Software Foundation,
    either version 3 of the License,
    or *(at your option)any later version.**This program is
        distributed in the hope that it will be useful,
    *but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *
 *
 */

#include <Kokkos_Core.hpp>
// #include <iostream>
#include <catch2/matchers/catch_matchers.hpp>
#include <string_view>

#include <catch2/catch_all.hpp>

#include <integratorxx/composite_quadratures/pruned_spherical_quadrature.hpp>
#include <integratorxx/composite_quadratures/spherical_quadrature.hpp>
#include <integratorxx/generators/radial_factory.hpp>
#include <integratorxx/generators/spherical_factory.hpp>
#include <integratorxx/quadratures/radial.hpp>
#include <integratorxx/quadratures/s2.hpp>

#include <nukexc/integration.hpp>
#include <nukexc/molecule.hpp>
#include <nukexc/partitioning.hpp>
#include <nukexc/stobasis.hpp>

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

template <typename radial_type, typename angular_type, typename REC>
double convergence_analysis(size_t nrad, size_t nang, REC recorder) {

  // Generate via runtime API
  auto rad_spec = IntegratorXX::radial_from_type<radial_type>();
  auto ang_spec = IntegratorXX::angular_from_type<angular_type>();
  auto rad_traits = IntegratorXX::make_radial_traits(rad_spec, nrad, 1.0);

  IntegratorXX::UnprunedSphericalGridSpecification unp(rad_spec, *rad_traits,
                                                       ang_spec, nang);

  auto pruning_spec =
      create_pruned_spec(IntegratorXX::PruningScheme::Robust, unp);
  auto sph = IntegratorXX::SphericalGridFactory::generate_grid(pruning_spec);

  const auto npts = sph->npts();

  // Generate water
  Molecule mol = make_water();
  unsigned int natoms = mol.natoms;
  STOBasisSet stobasis = load_adf_basis(mol, "input/zorabasis/QZ4P");
  FlatGrid grid = make_flat_grid<bk_type, ll_type>(mol);

  // Create all the Kokkos Views on host device

  DeviceView2DLeft S =
      overlap_integral(stobasis, grid.quad_points, grid.weights);

  auto S_h = Kokkos::create_mirror_view(S);
  Kokkos::deep_copy(S_h, S);

  double max_error = 0.0;
  for (unsigned i = 0; i < S_h.extent(0); ++i) {
    max_error = std::max(max_error, std::abs(S_h(i, i) - 1.0));
  }
  return max_error;
}

int main() {
  Kokkos::initialize();

  using namespace IntegratorXX;
  using radial_type = ta_type;
  using angular_type = ll_type;
  using angular_traits = quadrature_traits<angular_type>;
  using spherical_type = SphericalQuadrature<radial_type, angular_type>;

  // Gater data in std::vectors
  std::vector<double> errors;
  std::vector<size_t> rad_npts;
  std::vector<size_t> ang_npts;
  std::vector<size_t> atom_npts;
  std::vector<size_t> total_npts;

  auto recorder = [&rad_npts, &ang_npts, &atom_npts, &total_npts](
                      size_t rad, size_t ang, size_t atom, size_t total) {
    rad_npts.push_back(rad);
    ang_npts.push_back(ang);
    atom_npts.push_back(atom);
    total_npts.push_back(total);
  };

  for (unsigned m = 1; m < 10; ++m) {
    for (unsigned n = 3; n < 10; ++n) {
      size_t nrad = std::pow(2, n);
      size_t ang_deg = m * 10;
      size_t nang = angular_traits::npts_by_algebraic_order(
          angular_traits::next_algebraic_order(
              ang_deg)); // Smallest possible angular grid
      errors.push_back(convergence_analysis<radial_type, angular_type>(
          nrad, nang, recorder));
    }
  }

  std::cout << std::setw(15) << "rad_pts" << std::setw(15) << "ang_pts"
            << std::setw(15) << "pts_per_atom" << std::setw(15) << "pts_total"
            << std::setw(15) << "error" << std::endl;
  for (int i = 0; i < errors.size(); ++i) {
    std::cout << std::setw(15) << rad_npts[i] << std::setw(15) << ang_npts[i]
              << std::setw(15) << atom_npts[i] << std::setw(15) << total_npts[i]
              << std::setw(15) << errors[i] << std::endl;
  }

  Kokkos::finalize();

  return 0;
}
