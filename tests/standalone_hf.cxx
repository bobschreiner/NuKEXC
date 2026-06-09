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

#include <nukexc/core_hamiltonian.hpp>
#include <nukexc/coulomb.hpp>
#include <nukexc/diagonalizer.hpp>
#include <nukexc/exact_exchange.hpp>
#include <nukexc/grid.hpp>
#include <nukexc/molecule.hpp>
#include <nukexc/nuclear_repulsion.hpp>
#include <nukexc/nukexc_config.hpp>
#include <nukexc/partitioning.hpp>
#include <nukexc/stobasis.hpp>
#include <nukexc/xc_integrals.hpp>

#include <openorbitaloptimizer/scfsolver.hpp>

#include <cmath>
#include <tuple>
#include <vector>

using namespace Nukexc;
using bk_type = IntegratorXX::Becke<double, double>;
using ta_type = IntegratorXX::TreutlerAhlrichs<double, double>;
using ll_type = IntegratorXX::LebedevLaikov<double>;

// ---------------------------------------------------------------------------
// Helper: Kokkos DeviceView2DLeft → arma::mat (column-major copy)
// OOO expects arma::mat where columns are MOs; NuKEXC stores mo_coeff(nbf, nmo)
// ---------------------------------------------------------------------------
arma::mat kokkos_to_arma(const DeviceView2DLeft &v) {
  auto h = Kokkos::create_mirror_view(v);
  Kokkos::deep_copy(h, v);
  // h(i,j): row i = basis function, col j = MO  →  arma column-major matches
  arma::mat out(h.extent(0), h.extent(1));
  for (std::size_t i = 0; i < h.extent(0); ++i)
    for (std::size_t j = 0; j < h.extent(1); ++j)
      out(i, j) = h(i, j);
  return out;
}

// ---------------------------------------------------------------------------
// Helper: arma::mat → Kokkos DeviceView2DLeft
// ---------------------------------------------------------------------------
DeviceView2DLeft arma_to_kokkos(const arma::mat &m, const std::string &label) {
  DeviceView2DLeft v(label, m.n_rows, m.n_cols);
  auto h = Kokkos::create_mirror_view(v);
  for (std::size_t i = 0; i < m.n_rows; ++i)
    for (std::size_t j = 0; j < m.n_cols; ++j)
      h(i, j) = m(i, j);
  Kokkos::deep_copy(v, h);
  return v;
}

// ---------------------------------------------------------------------------
// Helper: arma::vec → Kokkos DeviceView1D
// ---------------------------------------------------------------------------
DeviceView1D arma_to_kokkos1d(const arma::vec &v, const std::string &label) {
  DeviceView1D kv(label, v.n_elem);
  auto h = Kokkos::create_mirror_view(kv);
  for (std::size_t i = 0; i < v.n_elem; ++i)
    h(i) = v(i);
  Kokkos::deep_copy(kv, h);
  return kv;
}

int main() {

  Kokkos::initialize();
  {
    // ---- Geometry & grid --------------------------------------------------
    Molecule mol;
    read_xyz("input/water.xyz", mol);

    const int nrad = 100;
    const int nang = 30;
    const double screening_tol = 1e-10;
    const double conv_thr = 1e-8;

    auto grid = make_flat_grid<bk_type, ll_type>(mol, nrad, nang);

    // ---- Basis sets -------------------------------------------------------
    // Primary basis
    STOBasisSet basis = load_adf_basis(
        mol, "input/zorabasis_cholesky/TZP.cholesky", screening_tol);
    // Auxiliary basis for density fitting (Coulomb + exchange)
    STOBasisSet basis_aux =
        load_adf_basis(mol, "input/zorabasis_cholesky/TZP.cholesky",
                       screening_tol, /*fit=*/true);

    const int nbf = basis.nbf();

    // ---- Core Hamiltonian (overlap + H_core in Kokkos views) -------------
    auto hcore = compute_core_hamiltonian(basis, grid);
    // hcore.hamiltonian : DeviceView2DLeft (nbf x nbf)
    // hcore.overlap     : DeviceView2DLeft (nbf x nbf)

    // ---- Orthogonalisation matrix X via NuKEXC diagonalizer ---------------
    // Diagonalizer already computes X = S^{-1/2} internally;
    // we reuse it each SCF cycle to solve F C = S C ε  in the AO basis.
    Diagonalizer diag(nbf);
    DeviceView2DLeft X = diag.compute_transformation(hcore.overlap);

    arma::mat h_core = kokkos_to_arma(hcore.hamiltonian);
    arma::mat X_arma = kokkos_to_arma(X);
    arma::mat S_arma = kokkos_to_arma(hcore.overlap); // need S on host
    arma::mat h_core_orth = X_arma.t() * h_core * X_arma;

    // ---- OOO setup
    // -------------------------------------------------------- For a
    // molecule we have NO angular-momentum symmetry blocks: one particle
    // type, one block containing all nbf basis functions.
    arma::uvec blocks_per_type = {1, 1};    // 2 spin blocks (α and β)
    arma::vec max_occupations = {1.0, 1.0}; // max 1 electron per spin channel
    arma::vec number_of_particles = {5.0, 5.0}; // 5α + 5β for water singlet
    std::vector<std::string> block_descriptions = {"alpha", "beta"};

    // Compute the nuclear repulsion energy once and pass it to the fock_builder
    double E_nuc = compute_nuclear_repulsion(mol);
    // ---- Fock builder -----------------------------------------------------
    // Captures by value everything that doesn't change between iterations.
    // OOO calls this every SCF iteration with the current DensityMatrix.
    auto fock_builder =
        [basis, basis_aux, grid, h_core, X_arma, nbf, E_nuc,
         S_arma](const OpenOrbitalOptimizer::DensityMatrix<double, double> &dm)
        -> std::pair<double, OpenOrbitalOptimizer::FockMatrix<double>> {
      const auto &orbitals = dm.first;     // vector<arma::mat>, one per block
      const auto &occupations = dm.second; // vector<arma::vec>, one per block

      // orbitals[0] : (nbf x nbf) MO coefficient matrix (columns = MOs)
      // occupations[0] : (nbf) occupation numbers
      // α and β orbitals and occupations from OOO
      const arma::mat C_alpha = X_arma * orbitals[0];
      const arma::mat C_beta = X_arma * orbitals[1];
      const arma::vec occ_alpha = occupations[0]; // 0 or 1
      const arma::vec occ_beta = occupations[1];  // 0 or 1

      // Build total density for Coulomb
      arma::mat D_alpha = C_alpha * arma::diagmat(occ_alpha) * C_alpha.t();
      arma::mat D_beta = C_beta * arma::diagmat(occ_beta) * C_beta.t();
      arma::mat D_tot = D_alpha + D_beta;

      // Diagnostics
#if NDEBUG
      std::cout << "Tr[D_alpha * S] = " << arma::trace(D_alpha * S_arma)
                << "\n"; // expect 5
      std::cout << "Tr[D_beta  * S] = " << arma::trace(D_beta * S_arma)
                << "\n"; // expect 5
      std::cout << "Tr[D_tot   * S] = " << arma::trace(D_tot * S_arma)
                << "\n"; // expect 10

#endif
      // Need to pass D_tot into compute_coulomb somehow
      // Easiest: pass combined orbitals [C_alpha | C_beta] with combined
      // occupations
      arma::mat C_combined = arma::join_horiz(C_alpha, C_beta);
      arma::vec occ_combined = arma::join_vert(occ_alpha, occ_beta);

      DeviceView2DLeft k_C_tot = arma_to_kokkos(C_combined, "C_combined");
      DeviceView1D k_occ_tot = arma_to_kokkos1d(occ_combined, "occ_combined");

      // J built from total density — same as RHF
      DeviceView2DLeft J =
          compute_coulomb(basis, basis_aux, grid, k_C_tot, k_occ_tot);

      // K built separately per spin — pass 0/1 occupations (no occ prefactor)
      DeviceView2DLeft k_C_alpha = arma_to_kokkos(C_alpha, "C_alpha");
      DeviceView2DLeft k_C_beta = arma_to_kokkos(C_beta, "C_beta");
      DeviceView1D k_occ_alpha = arma_to_kokkos1d(occ_alpha, "occ_alpha");
      DeviceView1D k_occ_beta = arma_to_kokkos1d(occ_beta, "occ_beta");

      DeviceView2DLeft K_alpha = compute_exact_exchange(basis, basis_aux, grid,
                                                        k_C_alpha, k_occ_alpha);
      DeviceView2DLeft K_beta =
          compute_exact_exchange(basis, basis_aux, grid, k_C_beta, k_occ_beta);
      // Convert to Kokkos for NuKEXC compute_coulomb / compute_exact_exchange

      auto J_arma = kokkos_to_arma(J);
      auto K_alpha_arma = kokkos_to_arma(K_alpha);
      auto K_beta_arma = kokkos_to_arma(K_beta);

      // No factors of 2 anywhere — D_α and D_β have occupation 0/1
      double E_core = arma::trace(D_tot * h_core); // Tr[(Dα+Dβ)*H]
      double E_coulomb =
          0.5 * arma::trace(D_tot * J_arma); // 0.5 * Tr[D_tot * J]
      double E_exchange =
          -0.5 * arma::trace(D_alpha * K_alpha_arma) -
          0.5 * arma::trace(D_beta * K_beta_arma); // one per spin
                                                   //
      double Etot = E_nuc + E_core + E_coulomb + E_exchange;
      std::cout << "E_core = " << E_core << "\n";
      std::cout << "E_coulomb = " << E_coulomb << "\n";
      std::cout << "E_exchange = " << E_exchange << "\n";
      std::cout << "E_nuc_repulsion = " << E_nuc << "\n";

      arma::mat F_alpha = h_core + J_arma - K_alpha_arma;
      arma::mat F_beta = h_core + J_arma - K_beta_arma;

      arma::mat F_alpha_orth = X_arma.t() * F_alpha * X_arma;
      arma::mat F_beta_orth = X_arma.t() * F_beta * X_arma;

      std::vector<arma::mat> fock_arma;
      fock_arma.push_back(F_alpha_orth);
      fock_arma.push_back(F_beta_orth);

      return std::make_pair(Etot, fock_arma); // two Fock matrices for OOO
    };

    // ---- Construct and run SCF solver -------------------------------------
    OpenOrbitalOptimizer::SCFSolver<double, double> solver(
        blocks_per_type, max_occupations, number_of_particles, fock_builder,
        block_descriptions);
    solver.convergence_threshold(conv_thr);
    solver.verbosity(5);
    solver.initialize_with_fock({h_core_orth, h_core_orth});
    solver.run();

    // ---- Check SCF converged to a sensible energy -------------------------
    double E_scf = solver.get_fock_build().first;
    std::cout << "SCF total energy: " << E_scf << " Eh\n";
  }
  Kokkos::finalize();
  return 0;
}
