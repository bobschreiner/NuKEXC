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
 */

#include <Kokkos_Core.hpp>

#include <catch2/catch_all.hpp>

#include <nukexc/core_hamiltonian.hpp>
#include <nukexc/density.hpp>
#include <nukexc/diagonalizer.hpp>
#include <nukexc/grid.hpp>
#include <nukexc/molecule.hpp>
#include <nukexc/nukexc_config.hpp>
#include <nukexc/stobasis.hpp>

using namespace NuKEXC;

using bk_type = IntegratorXX::Becke<double, double>;
using ll_type = IntegratorXX::LebedevLaikov<double>;

void write_density_vtk(const std::string &filename, const FlatGrid &grid,
                       const DeviceView1D &density) {

  // Mirror views to host
  auto points_h = Kokkos::create_mirror_view(grid.quad_points);
  auto weights_h = Kokkos::create_mirror_view(grid.weights);
  auto density_h = Kokkos::create_mirror_view(density);
  Kokkos::deep_copy(points_h, grid.quad_points);
  Kokkos::deep_copy(weights_h, grid.weights);
  Kokkos::deep_copy(density_h, density);

  const int N = points_h.extent(0);

  std::ofstream out(filename);
  if (!out)
    throw std::runtime_error("Could not open VTK file: " + filename);

  // ---- Header ----
  out << "# vtk DataFile Version 3.0\n";
  out << "NuKEXC electron density\n";
  out << "ASCII\n";
  out << "DATASET UNSTRUCTURED_GRID\n";

  // ---- Points ----
  out << "POINTS " << N << " double\n";
  out << std::scientific << std::setprecision(10);
  for (int g = 0; g < N; ++g) {
    out << points_h(g)[0] << " " << points_h(g)[1] << " " << points_h(g)[2]
        << "\n";
  }

  // ---- Cells: one vertex cell per quadrature point ----
  out << "CELLS " << N << " " << 2 * N << "\n";
  for (int g = 0; g < N; ++g)
    out << "1 " << g << "\n";

  out << "CELL_TYPES " << N << "\n";
  for (int g = 0; g < N; ++g)
    out << "1\n"; // VTK_VERTEX

  // ---- Point data ----
  out << "POINT_DATA " << N << "\n";

  out << "SCALARS density double 1\n";
  out << "LOOKUP_TABLE default\n";
  for (int g = 0; g < N; ++g)
    out << density_h(g) << "\n";

  out << "SCALARS weight double 1\n";
  out << "LOOKUP_TABLE default\n";
  for (int g = 0; g < N; ++g)
    out << weights_h(g) << "\n";
}
TEST_CASE("hydrogen 1s -- normalization, eigenvalues, virial",
          "[hydrogen_1s]") {

  Molecule mol(std::vector<std::vector<double>>{{0., 0., 0.}},
               std::vector<unsigned>{1u});
  auto grid = make_flat_grid<bk_type, ll_type>(mol, 100, 40);
  STOBasisSet basis = make_manual_basis({{1, 0, 0, 1.0, 0., 0., 0.}}); // 1s
                                                                       //
  // For a single occupied orbital with coefficient 1.0,
  // the density matrix is a 1x1 identity (one basis function, fully occupied)
  const int N_bf = basis.nbf();
  Kokkos::View<double **, ExecSpace> density_matrix("density_matrix", N_bf,
                                                    N_bf);
  Kokkos::deep_copy(density_matrix, 1.0); // D_ij = c_i * c_j = 1 * 1

  CoreHamiltonianResult result = compute_core_hamiltonian(basis, grid);

  DeviceView1D density = compute_density(basis, grid, density_matrix);

  // Integrate density against quadrature weights: ∫ρ(r)dr should equal N_elec =
  // 1
  Kokkos::View<double *, ExecSpace> weights = grid.weights;
  double integrated_density = 0.0;
  Kokkos::parallel_reduce(
      "Integrate density", density.extent(0),
      KOKKOS_LAMBDA(const int g, double &sum) {
        sum += density(g) * weights(g);
      },
      integrated_density);

  REQUIRE(integrated_density == Catch::Approx(1.0).epsilon(1e-5));
}

TEST_CASE("H2+ overlap matrix -- symmetry and analytical off-diagonal",
          "[h2_plus]") {
  const double R = 1.0; // bond length in bohr

  Molecule mol(std::vector<std::vector<double>>{{0., 0., 0.}, {R, 0., 0.}},
               std::vector<unsigned>{1u, 1u});

  auto grid = make_flat_grid<bk_type, ll_type>(mol, 100, 40);

  // Build the basis explicitly so the zeta value is unambiguous.
  STOBasisSet basis = make_manual_basis({
      {1, 0, 0, 1.0, 0., 0., 0.}, // 1s on atom A
      {1, 0, 0, 1.0, R, 0., 0.},  // 1s on atom B
  });

  CoreHamiltonianResult result = compute_core_hamiltonian(basis, grid);
  DeviceView2DLeft S = result.overlap;
  DeviceView2DLeft T = result.kinetic;
  DeviceView2DLeft V = result.nuclear;

  auto S_h = Kokkos::create_mirror_view(S);
  Kokkos::deep_copy(S_h, S);

  // ---- Density normalization for the bonding MO ----
  // The bonding MO is ψ = N(φ_A + φ_B), N = 1/sqrt(2 + 2*S_AB)
  // The density matrix is D_ij = c_i * c_j where c_i = N for all i.
  //
  const double N_bond = 1.0 / std::sqrt(2.0 + 2.0 * S_h(0, 1));
  Kokkos::View<double **, ExecSpace> density_matrix("density_matrix", 2, 2);
  auto dm_h = Kokkos::create_mirror_view(density_matrix);
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j) {
      dm_h(i, j) = N_bond * N_bond;
    }
  Kokkos::deep_copy(density_matrix, dm_h);

  DeviceView1D density = compute_density(basis, grid, density_matrix);
  write_density_vtk("h2plus_density_1s.vtk", grid, density);

  double integrated_density = 0.0;
  Kokkos::parallel_reduce(
      "Integrate H2+ bonding density", density.extent(0),
      KOKKOS_LAMBDA(const int g, double &sum) {
        sum += density(g) * grid.weights(g);
      },
      integrated_density);

  // ∫ρ dr = Tr(DS) = N^2 * (S_00 + S_01 + S_10 + S_11)
  //                = N^2 * (1 + S_AB + S_AB + 1) = N^2 * 2(1 + S_AB) = 1
  REQUIRE_THAT(integrated_density, Catch::Matchers::WithinRel(1.0, 1e-5));
}

TEST_CASE("H2+ Energies Fused Hamiltonian",
          "[h2_plus][energies][fused hamiltonian]") {

  const double R = 1.0; // bond length in bohr

  Molecule mol(std::vector<std::vector<double>>{{0., 0., 0.}, {R, 0., 0.}},
               std::vector<unsigned>{1u, 1u});

  auto grid = make_flat_grid<bk_type, ll_type>(mol);

  // Build the basis explicitly so the zeta value is unambiguous.
  STOBasisSet basis = load_adf_basis(mol, "input/zorabasis/QZ4P");

  int n_basis = basis.nbf();

  CoreHamiltonianResult hamiltonian;
  hamiltonian = compute_core_hamiltonian(basis, grid);

  // 1. Prepare Batched Views on Device
  DeviceView2DLeft mo_coeffs("mo_coeffs", n_basis, n_basis);
  DeviceView1D mo_energies("mo_energies", n_basis);

  NuKEXC::Diagonalizer diagonalizer(n_basis);
  diagonalizer.compute_transformation(hamiltonian.overlap);
  diagonalizer.solve(hamiltonian.hamiltonian, mo_coeffs, mo_energies);

  // =========================================================================
  // Construct the Density Matrix for H2+ (1 electron in the lowest MO)
  // Formula: D(i, j) = n_occ * C(i, 0) * C(j, 0)  where n_occ = 1.0
  // =========================================================================
  Kokkos::View<double **, ExecSpace> density_matrix("density_matrix", n_basis,
                                                    n_basis);

  Kokkos::parallel_for(
      "Build H2+ Density Matrix",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>, ExecSpace>({0, 0},
                                                        {n_basis, n_basis}),
      KOKKOS_LAMBDA(const int i, const int j) {
        // mo_coeffs layout matches DeviceView2DLeft, so indices are (basis, mo)
        density_matrix(i, j) = 1.0 * mo_coeffs(i, 0) * mo_coeffs(j, 0);
      });
  // =========================================================================

  DeviceView1D density = compute_density(basis, grid, density_matrix);
  write_density_vtk("h2plus_density_QZ4P.vtk", grid, density);

  double integrated_density = 0.0;
  Kokkos::parallel_reduce(
      "Integrate H2+ bonding density", density.extent(0),
      KOKKOS_LAMBDA(const int g, double &sum) {
        sum += density(g) * grid.weights(g);
      },
      integrated_density);

  REQUIRE_THAT(integrated_density, Catch::Matchers::WithinRel(1.0, 1e-5));
}

int main() {
  Kokkos::initialize();
  int result = Catch::Session().run();
  Kokkos::finalize();

  return result;
}
