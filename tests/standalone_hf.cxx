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

struct Config {
  std::string xyz_file = "input/water.xyz";
  std::string basis_file = "input/zorabasis_cholesky/TZ2P.cholesky";
  int nrad = 100;
  int nang = 30;
  double lin_dep_threshold = 1e-6;
  double conv_thr = 1e-8;
  int charge = 0;
  int multiplicity = 1;
};

Config parse_args(int argc, char *argv[]) {
  Config cfg;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];

    auto parse_string = [&](const std::string &prefix, std::string &out) {
      if (arg.rfind(prefix, 0) == 0) {
        out = arg.substr(prefix.size());
        return true;
      }
      return false;
    };
    auto parse_int = [&](const std::string &prefix, int &out) {
      if (arg.rfind(prefix, 0) == 0) {
        out = std::stoi(arg.substr(prefix.size()));
        return true;
      }
      return false;
    };
    auto parse_double = [&](const std::string &prefix, double &out) {
      if (arg.rfind(prefix, 0) == 0) {
        out = std::stod(arg.substr(prefix.size()));
        return true;
      }
      return false;
    };

    if (arg == "--help" || arg == "-h") {
      std::cout
          << "Usage: " << argv[0] << " [options]\n"
          << "  --xyz=<file>          Molecule XYZ file        (default: "
          << cfg.xyz_file << ")\n"
          << "  --basis=<file>        Basis set file           (default: "
          << cfg.basis_file << ")\n"
          << "  --nrad=<int>          Radial grid points       (default: "
          << cfg.nrad << ")\n"
          << "  --nang=<int>          Angular grid points      (default: "
          << cfg.nang << ")\n"
          << "  --lin-dep=<float>     Linear dep. threshold    (default: "
          << "  --conv-thr=<float>    SCF convergence threshold (default: "

          << cfg.conv_thr << ")\n"
          << cfg.lin_dep_threshold << ")\n";

      std::exit(0);
    } else if (!parse_string("--xyz=", cfg.xyz_file) &&
               !parse_string("--basis=", cfg.basis_file) &&
               !parse_int("--nrad=", cfg.nrad) &&
               !parse_int("--nang=", cfg.nang) &&
               !parse_double("--lin-dep=", cfg.lin_dep_threshold) &&
               !parse_double("--conv-thr=", cfg.conv_thr) &&
               !parse_int("--charge=", cfg.charge) &&
               !parse_int("--multiplicity=", cfg.multiplicity)) {
      throw std::runtime_error("Unknown argument: " + arg + " (try --help)");
    }
  }
  if (cfg.nrad <= 0 || cfg.nang <= 0)
    throw std::runtime_error("--nrad and --nang must be positive");
  return cfg;
}

auto repeat(const std::string &s, int n) {
  std::string r;
  for (int i = 0; i < n; ++i)
    r += s;
  return r;
}
void print_config(const Config &cfg) {
  int width =
      std::max({cfg.xyz_file.size(), cfg.basis_file.size(), size_t(20)});
  std::string h = repeat("─", width + 2);

  std::cout << "\n";
  std::cout << "┌───────────────────────" << h << "┐\n";
  std::cout << "│    HF Configuration" << repeat(" ", width + 5) << "│\n";
  std::cout << "├───────────────────────" << h << "┤\n";
  std::cout << "│ Molecule file        │ " << std::setw(width) << std::left
            << cfg.xyz_file << " │\n";
  std::cout << "│ Basis file           │ " << std::setw(width) << std::left
            << cfg.basis_file << " │\n";
  std::cout << "│ Radial points        │ " << std::setw(width) << cfg.nrad
            << " │\n";
  std::cout << "│ Angular order        │ " << std::setw(width) << cfg.nang
            << " │\n";
  std::cout << "│ Lin. dep. threshold  │ " << std::setw(width)
            << cfg.lin_dep_threshold << " │\n";
  std::cout << "│ Conv. threshold      │ " << std::setw(width) << cfg.conv_thr
            << " │\n";
  std::cout << "│ Charge               │ " << std::setw(width) << cfg.charge
            << " │\n";
  std::cout << "│ Multiplicity         │ " << std::setw(width)
            << cfg.multiplicity << " │\n";
  std::cout << "└──────────────────────┴" << h << "┘\n\n";
  std::cout << std::flush;
}
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

int main(int argc, char *argv[]) {
  Config cfg;
  try {
    cfg = parse_args(argc, argv);
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
  print_config(cfg);
  Kokkos::initialize();
  {
    Molecule mol;
    read_xyz(cfg.xyz_file, mol);

    auto grid = make_flat_grid<bk_type, ll_type>(mol, cfg.nrad, cfg.nang);

    const double screening_tol = 1e-10;
    STOBasisSet basis = load_adf_basis(mol, cfg.basis_file, screening_tol);
    STOBasisSet basis_aux =
        load_adf_basis(mol, cfg.basis_file, screening_tol, /*fit=*/true);

    // ---- Compute collocations --------------------------------------------

    const int N_bf = basis.nbf();
    const int N_bf_aux = basis_aux.nbf();
    const int N_quad = grid.quad_points.extent(0);

    DeviceView2DLeft basis_collocation("Basis collocation", N_bf, N_quad);
    DeviceView2DLeft basis_aux_collocation("Auxillary Basis collocation",
                                           N_bf_aux, N_quad);
    DeviceView2DLeft potential_collocation_scaled("Potential collocation",
                                                  N_bf_aux, N_quad);

    ExecSpace space;
    fill_collocation(space, basis, grid.quad_points, basis_collocation);
    fill_collocation(space, basis_aux, grid.quad_points, basis_aux_collocation);
    sto_potential_collocation(space, basis_aux, grid,
                              potential_collocation_scaled);

    Kokkos::TeamPolicy<ExecSpace> policy(space, N_quad, Kokkos::AUTO());
    using member_type = Kokkos::TeamPolicy<ExecSpace>::member_type;

    Kokkos::parallel_for(
        "Scale potential", policy,
        KOKKOS_LAMBDA(const member_type &team_member) {
          const int g = team_member.league_rank();
          const double w_g = grid.weights(g);
          Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, N_bf_aux),
                               [=](const int alpha) {
                                 potential_collocation_scaled(alpha, g) *= w_g;
                               });
        });

    // ---- Core Hamiltonian (overlap + H_core in Kokkos views) -------------
    auto hcore = compute_core_hamiltonian(basis, grid);
    // hcore.hamiltonian : DeviceView2DLeft (nbf x nbf)
    // hcore.overlap     : DeviceView2DLeft (nbf x nbf)

    // ---- Orthogonalisation matrix X via NuKEXC diagonalizer ---------------
    // Diagonalizer already computes X = S^{-1/2} internally;
    // we reuse it each SCF cycle to solve F C = S C ε  in the AO basis.
    Diagonalizer diag(N_bf);
    DeviceView2DLeft X =
        diag.compute_transformation(hcore.overlap, cfg.lin_dep_threshold);

    arma::mat h_core = kokkos_to_arma(hcore.hamiltonian);
    arma::mat X_arma = kokkos_to_arma(X);
    arma::mat S_arma = kokkos_to_arma(hcore.overlap); // need S on host
    arma::mat h_core_orth = X_arma.t() * h_core * X_arma;

    // Derive electron counts from geometry + charge + multiplicity
    int n_elec = mol.Z_total - cfg.charge;
    std::cout << "Number of Electrons: " << n_elec << std::endl;
    std::cout << "Z_total: " << mol.Z_total << std::endl;
    if ((n_elec + cfg.multiplicity - 1) % 2 != 0)
      throw std::runtime_error("Charge and multiplicity are inconsistent with "
                               "the number of electrons");
    double n_alpha = (n_elec + (cfg.multiplicity - 1)) / 2.0;
    double n_beta = (n_elec - (cfg.multiplicity - 1)) / 2.0;

    // ---- OOO setup
    // -------------------------------------------------------- For a
    // molecule we have NO angular-momentum symmetry blocks: one particle
    // type, one block containing all nbf basis functions.
    arma::uvec blocks_per_type = {1, 1};    // 2 spin blocks (α and β)
    arma::vec max_occupations = {1.0, 1.0}; // max 1 electron per spin channel
    arma::vec number_of_particles = {n_alpha, n_beta};
    std::vector<std::string> block_descriptions = {"alpha", "beta"};

    // Compute the nuclear repulsion energy once and pass it to the fock_builder
    double E_nuc = compute_nuclear_repulsion(mol);
    // ---- Fock builder -----------------------------------------------------
    // Captures by value everything that doesn't change between iterations.
    // OOO calls this every SCF iteration with the current DensityMatrix.
    auto fock_builder =
        [space, basis_collocation, basis_aux_collocation,
         potential_collocation_scaled, h_core, X_arma, N_bf, E_nuc, S_arma,
         cfg](const OpenOrbitalOptimizer::DensityMatrix<double, double> &dm)
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
      DeviceView2DLeft J = compute_coulomb(
          space, k_C_tot, k_occ_tot, basis_collocation, basis_aux_collocation,
          potential_collocation_scaled, cfg.lin_dep_threshold);

      // K built separately per spin — pass 0/1 occupations (no occ prefactor)
      DeviceView2DLeft k_C_alpha = arma_to_kokkos(C_alpha, "C_alpha");
      DeviceView2DLeft k_C_beta = arma_to_kokkos(C_beta, "C_beta");
      DeviceView1D k_occ_alpha = arma_to_kokkos1d(occ_alpha, "occ_alpha");
      DeviceView1D k_occ_beta = arma_to_kokkos1d(occ_beta, "occ_beta");

      DeviceView2DLeft K_alpha = compute_exact_exchange(
          space, k_C_alpha, k_occ_alpha, basis_collocation,
          basis_aux_collocation, potential_collocation_scaled,
          cfg.lin_dep_threshold);

      DeviceView2DLeft K_beta = compute_exact_exchange(
          space, k_C_beta, k_occ_beta, basis_collocation, basis_aux_collocation,
          potential_collocation_scaled, cfg.lin_dep_threshold);

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

    // ---- GWH initial guess (replaces h_core_orth as the starting Fock) ----
    arma::mat F_gwh(N_bf, N_bf, arma::fill::zeros);
    const double K_gwh = 1.75;
    for (int i = 0; i < N_bf; ++i) {
      for (int j = 0; j < N_bf; ++j) {
        if (i == j) {
          F_gwh(i, j) = h_core(i, i);
        } else {
          F_gwh(i, j) =
              0.5 * K_gwh * (h_core(i, i) + h_core(j, j)) * S_arma(i, j);
        }
      }
    }
    arma::mat F_gwh_orth = X_arma.t() * F_gwh * X_arma;
    // ---- Construct and run SCF solver -------------------------------------
    OpenOrbitalOptimizer::SCFSolver<double, double> solver(
        blocks_per_type, max_occupations, number_of_particles, fock_builder,
        block_descriptions);
    solver.convergence_threshold(cfg.conv_thr);
    solver.verbosity(5);
    solver.initialize_with_fock({F_gwh_orth, F_gwh_orth});
    solver.run();

    // ---- Check SCF converged to a sensible energy -------------------------
    double E_scf = solver.get_fock_build().first;
    std::cout << "SCF total energy: " << E_scf << " Eh\n";
  }
  Kokkos::finalize();
  return 0;
}
