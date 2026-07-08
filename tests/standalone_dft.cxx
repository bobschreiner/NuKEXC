/*
 *    NuKEXC -- Numerical Kokkos Enhanced Exchange Correlation Integrator
 *    Copyright (C) 2026 Bob Schreiner
 *
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

#include <iomanip>
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
#include <xc.h>
#include <xc_funcs.h>

#include <cmath>
#include <cstdint>
#include <tuple>
#include <vector>

using namespace Nukexc;
using bk_type = IntegratorXX::Becke<double, double>;
using ta_type = IntegratorXX::TreutlerAhlrichs<double, double>;
using ll_type = IntegratorXX::LebedevLaikov<double>;

// ---------------------------------------------------------------------------
// Compile-time string hashing so we can "switch" on a runtime std::string.
// C++ can't switch on std::string directly, but case labels only need to be
// constant expressions -- so we hash the *literal* case labels at compile
// time via this constexpr function, and hash the *runtime* input string with
// the exact same function at runtime, then switch on the resulting uint32_t.
// ---------------------------------------------------------------------------
constexpr std::uint32_t fnv1a(const char *s, std::uint32_t h = 2166136261u) {
  return (*s == '\0') ? h
                      : fnv1a(s + 1, (h ^ static_cast<std::uint32_t>(
                                              static_cast<unsigned char>(*s))) *
                                         16777619u);
}
inline std::uint32_t fnv1a_rt(const std::string &s) { return fnv1a(s.c_str()); }

// ---------------------------------------------------------------------------
// Functional family + libxc id, resolved from a runtime name via switch.
// ---------------------------------------------------------------------------
enum class XCFamily { LDA, GGA };

struct FunctionalInfo {
  int xc_id;
  XCFamily family;
  std::string canonical_name;
};

FunctionalInfo lookup_functional(const std::string &name) {
  switch (fnv1a_rt(name)) {
  // ---- LDA exchange -----------------------------------------------------
  case fnv1a("lda_x"):
    return {XC_LDA_X, XCFamily::LDA, "lda_x"};

  // ---- LDA correlation ---------------------------------------------------
  case fnv1a("lda_c_pw"):
    return {XC_LDA_C_PW, XCFamily::LDA, "lda_c_pw"};
  case fnv1a("lda_c_vwn"):
    return {XC_LDA_C_VWN, XCFamily::LDA, "lda_c_vwn"};

  // ---- GGA exchange -------------------------------------------------------
  case fnv1a("gga_x_pbe"):
    return {XC_GGA_X_PBE, XCFamily::GGA, "gga_x_pbe"};
  case fnv1a("gga_x_b88"):
    return {XC_GGA_X_B88, XCFamily::GGA, "gga_x_b88"};
  case fnv1a("gga_x_pw91"):
    return {XC_GGA_X_PW91, XCFamily::GGA, "gga_x_pw91"};
  case fnv1a("gga_xc_b3lyp3"):
    // 394 is the standard Libxc ID for B3LYP with VWN3
    return {394, XCFamily::GGA, "gga_xc_b3lyp3"};

  // ---- GGA correlation ------------------------------------------------
  case fnv1a("gga_c_pbe"):
    return {XC_GGA_C_PBE, XCFamily::GGA, "gga_c_pbe"};
  case fnv1a("gga_c_lyp"):
    return {XC_GGA_C_LYP, XCFamily::GGA, "gga_c_lyp"};
  case fnv1a("gga_c_pw91"):
    return {XC_GGA_C_PW91, XCFamily::GGA, "gga_c_pw91"};

  default:
    throw std::runtime_error(
        "Unknown functional name: '" + name +
        "' (see --help or lookup_functional() for supported names)");
  }
}

struct Config {
  std::string xyz_file = "input/water.xyz";
  std::string basis_file = "input/zorabasis_cholesky/TZ2P.cholesky";
  int nrad = 100;
  int nang = 30;
  double lin_dep_threshold = 1e-6;
  double conv_thr = 1e-8;
  int charge = 0;
  int multiplicity = 1;
  std::string xfunc = "lda_x";
  std::string cfunc = "lda_c_pw";
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
          << cfg.lin_dep_threshold << ")\n"
          << "  --conv-thr=<float>    SCF convergence threshold (default: "
          << cfg.conv_thr << ")\n"
          << "  --charge=<int>        Molecular charge          (default: "
          << cfg.charge << ")\n"
          << "  --multiplicity=<int>  Spin multiplicity         (default: "
          << cfg.multiplicity << ")\n"
          << "  --xfunc=<name>        Exchange functional       (default: "
          << cfg.xfunc << ")\n"
          << "  --cfunc=<name>        Correlation functional    (default: "
          << cfg.cfunc << ")\n"
          << "\n"
          << "  Supported functional names:\n"
          << "    LDA exchange:     lda_x\n"
          << "    LDA correlation:  lda_c_pw, lda_c_vwn\n"
          << "    GGA exchange:     gga_x_pbe, gga_x_b88, gga_x_pw91\n"
          << "    GGA correlation:  gga_c_pbe, gga_c_lyp, gga_c_pw91\n";
      std::exit(0);
    } else if (!parse_string("--xyz=", cfg.xyz_file) &&
               !parse_string("--basis=", cfg.basis_file) &&
               !parse_int("--nrad=", cfg.nrad) &&
               !parse_int("--nang=", cfg.nang) &&
               !parse_double("--lin-dep=", cfg.lin_dep_threshold) &&
               !parse_double("--conv-thr=", cfg.conv_thr) &&
               !parse_int("--charge=", cfg.charge) &&
               !parse_int("--multiplicity=", cfg.multiplicity) &&
               !parse_string("--xfunc=", cfg.xfunc) &&
               !parse_string("--cfunc=", cfg.cfunc)) {
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
  int width = std::max({cfg.xyz_file.size(), cfg.basis_file.size(),
                        cfg.xfunc.size() + cfg.cfunc.size() + 3, size_t(20)});
  std::string h = repeat("─", width + 2);

  std::cout << "\n";
  std::cout << "┌───────────────────────" << h << "┐\n";
  std::cout << "│    DFT Configuration" << repeat(" ", width + 5) << "│\n";
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
  std::cout << "│ Exchange functional  │ " << std::setw(width) << std::left
            << cfg.xfunc << " │\n";
  std::cout << "│ Correlation function │ " << std::setw(width) << std::left
            << cfg.cfunc << " │\n";
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

// ---------------------------------------------------------------------------
// Dispatch a single functional (X or C) to the correct polarized evaluator
// based on its family. Both compute_lsda and compute_gga_lsda must return
// XC_result_polarized { energy, potential_alpha, potential_beta }.
// ---------------------------------------------------------------------------
XC_result_polarized evaluate_functional(
    const FunctionalInfo &info, const xc_func_type &func,
    const DeviceView2DLeft &basis_collocation,
    const DeviceView2DLeft &basis_collocation_gx,
    const DeviceView2DLeft &basis_collocation_gy,
    const DeviceView2DLeft &basis_collocation_gz, const DeviceView1D &weights,
    const DeviceView2DLeft &k_C_alpha, const DeviceView1D &k_occ_alpha,
    const DeviceView2DLeft &k_C_beta, const DeviceView1D &k_occ_beta) {


  switch (func.info->family) {
  case (XC_FAMILY_LDA):
    return compute_lsda(basis_collocation, weights, k_C_alpha, k_occ_alpha,
                        k_C_beta, k_occ_beta, func);
  case XC_FAMILY_HYB_LDA:
    return compute_lsda(basis_collocation, weights, k_C_alpha, k_occ_alpha,
                        k_C_beta, k_occ_beta, func);
  case XC_FAMILY_GGA:
    return compute_gga_lsda(basis_collocation, basis_collocation_gx,
                            basis_collocation_gy, basis_collocation_gz, weights,
                            k_C_alpha, k_occ_alpha, k_C_beta, k_occ_beta, func);
  case XC_FAMILY_HYB_GGA:
    return compute_gga_lsda(basis_collocation, basis_collocation_gx,
                            basis_collocation_gy, basis_collocation_gz, weights,
                            k_C_alpha, k_occ_alpha, k_C_beta, k_occ_beta, func);
  default:
    throw std::runtime_error("Unhandled XCFamily in evaluate_functional");
  }
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

    const double screening_tol = 1e-10;
    auto grid = make_flat_grid<bk_type, ll_type>(mol, cfg.nrad, cfg.nang,
                                                 screening_tol * 1e-5);

    STOBasisSet basis = load_adf_basis(mol, cfg.basis_file, screening_tol);
    STOBasisSet basis_aux =
        load_adf_basis(mol, cfg.basis_file, screening_tol, /*fit=*/true);

    // ---- Resolve functionals from runtime strings via switch dispatch ----
    FunctionalInfo x_info = lookup_functional(cfg.xfunc);

    xc_func_type func_x;
    if (xc_func_init(&func_x, x_info.xc_id, XC_POLARIZED) != 0) {
      throw std::runtime_error(
          "Failed to initialize Libxc exchange functional '" +
          x_info.canonical_name + "'");
    }

    // ---- Determine if the functional handles BOTH Exchange and Correlation
    // ---- Libxc defines: XC_EXCHANGE=0, XC_CORRELATION=1,
    // XC_EXCHANGE_CORRELATION=2
    bool is_combined_xc = (func_x.info->kind == XC_EXCHANGE_CORRELATION);

    xc_func_type func_c;
    FunctionalInfo c_info = lookup_functional(cfg.cfunc);

    bool has_separate_c = false;

    if (is_combined_xc) {
      // For combined functionals (like B3LYP or M06), cfunc should either be
      // ignored or match the exact same combined functional name to prevent
      // double-counting.
      has_separate_c = false;
    } else {
      // Fallback: This is a pure exchange component, we MUST initialize a
      // correlation functional
      if (xc_func_init(&func_c, c_info.xc_id, XC_POLARIZED) != 0) {
        xc_func_end(&func_x);
        throw std::runtime_error(
            "Failed to initialize Libxc correlation functional '" +
            c_info.canonical_name + "'");
      }
      has_separate_c = true;
    }

    // ---- Extract exact exchange fraction for Hybrids ----
    double a_exx = xc_hyb_exx_coef(&func_x);

    // ---- Compute collocations
    // --------------------------------------------

    const int N_bf = basis.nbf();
    const int N_bf_aux = basis_aux.nbf();
    const int N_quad = grid.quad_points.extent(0);

    DeviceView2DLeft basis_collocation("Basis collocation", N_bf, N_quad);
    DeviceView2DLeft basis_aux_collocation("Auxillary Basis collocation",
                                           N_bf_aux, N_quad);
    DeviceView2DLeft potential_collocation_scaled("Potential collocation",
                                                  N_bf_aux, N_quad);

    DeviceView2DLeft aux_overlap("Aux overlap", N_bf_aux, N_bf_aux);
    DeviceView2DLeft aux_overlap_sym("Aux overlap sym", N_bf_aux, N_bf_aux);

    ExecSpace space;
    fill_collocation(space, basis, grid.quad_points, basis_collocation);
    fill_collocation(space, basis_aux, grid.quad_points, basis_aux_collocation);
    sto_potential_collocation_scaled(space, basis_aux, grid,
                                     potential_collocation_scaled);

    // Compute (A|B)
    KokkosBlas::gemm(space, "N", "T", 1.0, basis_aux_collocation,
                     potential_collocation_scaled, 0.0, aux_overlap);

    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {N_bf_aux, N_bf_aux}),
        KOKKOS_LAMBDA(int i, int j) {
          aux_overlap_sym(i, j) = 0.5 * (aux_overlap(i, j) + aux_overlap(j, i));
        });

    DeviceView2DLeft half_inverse_X =
        compute_half_inverse(aux_overlap_sym, cfg.lin_dep_threshold);

    // ---- GGA basis-function gradient collocations (only if needed) -------

    DeviceView2DLeft basis_collocation_gx("Basis collocation dX", N_bf, N_quad);
    DeviceView2DLeft basis_collocation_gy("Basis collocation dY", N_bf, N_quad);
    DeviceView2DLeft basis_collocation_gz("Basis collocation dZ", N_bf, N_quad);

    fill_grad_collocation(space, basis, grid.quad_points, basis_collocation_gx,
                          basis_collocation_gy, basis_collocation_gz);

    // ---- Core Hamiltonian (overlap + H_core in Kokkos views)
    // -------------
    auto hcore = compute_core_hamiltonian(basis, grid);

    // ---- Orthogonalisation matrix X via NuKEXC diagonalizer
    // ---------------
    Diagonalizer diag(N_bf);
    DeviceView2DLeft X =
        diag.compute_transformation(hcore.overlap, cfg.lin_dep_threshold);

    arma::mat h_core = kokkos_to_arma(hcore.hamiltonian);
    arma::mat X_arma = kokkos_to_arma(X);
    arma::mat S_arma = kokkos_to_arma(hcore.overlap);
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
    // --------------------------------------------------------
    arma::uvec blocks_per_type = {1, 1};
    arma::vec max_occupations = {1.0, 1.0};
    arma::vec number_of_particles = {n_alpha, n_beta};
    std::vector<std::string> block_descriptions = {"alpha", "beta"};

    double E_nuc = compute_nuclear_repulsion(mol);

    // ---- Fock builder
    // -----------------------------------------------------
    auto fock_builder =
        [space, grid, basis_collocation, basis_collocation_gx,
         basis_collocation_gy, basis_collocation_gz, basis_aux_collocation,
         potential_collocation_scaled, h_core, X_arma, half_inverse_X, N_bf,
         E_nuc, S_arma, cfg, func_c, func_x, x_info, c_info, has_separate_c,
         a_exx](const OpenOrbitalOptimizer::DensityMatrix<double, double> &dm)
        -> std::pair<double, OpenOrbitalOptimizer::FockMatrix<double>> {
      const auto &orbitals = dm.first;
      const auto &occupations = dm.second;

      const arma::mat C_alpha = X_arma * orbitals[0];
      const arma::mat C_beta = X_arma * orbitals[1];
      const arma::vec occ_alpha = occupations[0];
      const arma::vec occ_beta = occupations[1];

      arma::mat D_alpha = C_alpha * arma::diagmat(occ_alpha) * C_alpha.t();
      arma::mat D_beta = C_beta * arma::diagmat(occ_beta) * C_beta.t();
      arma::mat D_tot = D_alpha + D_beta;

#ifndef NDEBUG
      std::cout << "Tr[D_alpha * S] = " << arma::trace(D_alpha * S_arma)
                << "\n";
      std::cout << "Tr[D_beta  * S] = " << arma::trace(D_beta * S_arma) << "\n";
      std::cout << "Tr[D_tot   * S] = " << arma::trace(D_tot * S_arma) << "\n";
#endif

      arma::mat C_combined = arma::join_horiz(C_alpha, C_beta);
      arma::vec occ_combined = arma::join_vert(occ_alpha, occ_beta);

      DeviceView2DLeft k_C_tot = arma_to_kokkos(C_combined, "C_combined");
      DeviceView1D k_occ_tot = arma_to_kokkos1d(occ_combined, "occ_combined");

      DeviceView2DLeft J = compute_coulomb(
          space, k_C_tot, k_occ_tot, basis_collocation, basis_aux_collocation,
          potential_collocation_scaled, half_inverse_X);

      auto J_arma = kokkos_to_arma(J);

      DeviceView2DLeft k_C_alpha = arma_to_kokkos(C_alpha, "C_alpha");
      DeviceView2DLeft k_C_beta = arma_to_kokkos(C_beta, "C_beta");
      DeviceView1D k_occ_alpha = arma_to_kokkos1d(occ_alpha, "occ_alpha");
      DeviceView1D k_occ_beta = arma_to_kokkos1d(occ_beta, "occ_beta");

      // Always evaluate the exchange or combined XC functional

      XC_result_polarized E_x = evaluate_functional(
          x_info, func_x, basis_collocation, basis_collocation_gx,
          basis_collocation_gy, basis_collocation_gz, grid.weights, k_C_alpha,
          k_occ_alpha, k_C_beta, k_occ_beta);

      double E_exchange =
          E_x.energy; // This holds total E_xc if is_combined_xc == true
      double E_correlation = 0.0;

      auto Vx_alpha = kokkos_to_arma(E_x.potential_alpha);
      auto Vx_beta = kokkos_to_arma(E_x.potential_beta);
      arma::mat Vc_alpha =
          arma::zeros<arma::mat>(Vx_alpha.n_rows, Vx_alpha.n_cols);
      arma::mat Vc_beta =
          arma::zeros<arma::mat>(Vx_beta.n_rows, Vx_beta.n_cols);

      // Only evaluate a second functional if it was explicitly split out
      if (has_separate_c) {
        XC_result_polarized E_c = evaluate_functional(
            c_info, func_c, basis_collocation, basis_collocation_gx,
            basis_collocation_gy, basis_collocation_gz, grid.weights, k_C_alpha,
            k_occ_alpha, k_C_beta, k_occ_beta);

        E_correlation = E_c.energy;
        Vc_alpha = kokkos_to_arma(E_c.potential_alpha);
        Vc_beta = kokkos_to_arma(E_c.potential_beta);
      }

      arma::mat K_alpha_arma =
          arma::zeros<arma::mat>(Vx_alpha.n_rows, Vx_alpha.n_cols);
      arma::mat K_beta_arma =
          arma::zeros<arma::mat>(Vx_beta.n_rows, Vx_beta.n_cols);

      // Compute Exact Exchange in case of hybrid functionals
      double E_exact_exchange = 0.0;
      if (a_exx > 0) {

        DeviceView2DLeft K_alpha = compute_exact_exchange(
            space, k_C_alpha, k_occ_alpha, basis_collocation,
            basis_aux_collocation, potential_collocation_scaled,
            half_inverse_X);

        DeviceView2DLeft K_beta = compute_exact_exchange(
            space, k_C_beta, k_occ_beta, basis_collocation,
            basis_aux_collocation, potential_collocation_scaled,
            half_inverse_X);
        K_alpha_arma = kokkos_to_arma(K_alpha);
        K_beta_arma = kokkos_to_arma(K_beta);

        E_exact_exchange = -0.5 * a_exx *
                           (arma::trace(D_alpha * K_alpha_arma) +
                            arma::trace(D_beta * K_beta_arma));
      }

      double E_core = arma::trace(D_tot * h_core);
      double E_coulomb = 0.5 * arma::trace(D_tot * J_arma);

      double Etot = E_nuc + E_core + E_coulomb + E_exchange + E_correlation +
                    E_exact_exchange;

      std::cout << "Total energy        : " << std::setprecision(10) << Etot
                << "\n";
      std::cout << "--------------------------------------------- \n";
      std::cout << "Nuclear Repulsion   : " << std::setprecision(10) << E_nuc
                << "\n";
      std::cout << "Electronic   Energy : " << std::setprecision(10)
                << E_core + E_coulomb + E_exchange << "\n";
      std::cout << "One Electron Energy : " << std::setprecision(10) << E_core
                << "\n";
      std::cout << "Two Electron Energy : " << std::setprecision(10)
                << E_coulomb + E_exchange << "\n\n";

      arma::mat F_alpha =
          h_core + J_arma - a_exx * K_alpha_arma + Vx_alpha + Vc_alpha;
      arma::mat F_beta =
          h_core + J_arma - a_exx * K_beta_arma + Vx_beta + Vc_beta;

      arma::mat F_alpha_orth = X_arma.t() * F_alpha * X_arma;
      arma::mat F_beta_orth = X_arma.t() * F_beta * X_arma;

      std::vector<arma::mat> fock_arma;
      fock_arma.push_back(F_alpha_orth);
      fock_arma.push_back(F_beta_orth);

      return std::make_pair(Etot, fock_arma);
    };

    // ---- GWH initial guess
    // ----
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

    // ---- Construct and run SCF solver
    // -------------------------------------
    OpenOrbitalOptimizer::SCFSolver<double, double> solver(
        blocks_per_type, max_occupations, number_of_particles, fock_builder,
        block_descriptions);
    solver.convergence_threshold(cfg.conv_thr);
    solver.verbosity(5);
    solver.initialize_with_fock({F_gwh_orth, F_gwh_orth});
    solver.run();

    xc_func_end(&func_x);
    if (has_separate_c)
      xc_func_end(&func_c);

    double E_scf = solver.get_fock_build().first;
    std::cout << "SCF total energy: " << E_scf << " Eh\n";

    // ---- HOMO-LUMO gap ---------------------------------------------------

    auto final_fock =
        solver.get_fock_matrix(); // {F_alpha, F_beta}, orthonormal basis
    auto diagonalized =
        solver.compute_orbitals(final_fock); // {orbitals, orbital_energies}
    const auto &orbital_energies =
        diagonalized.second; // vector<arma::vec>, one per block
    const auto &occupations =
        solver.get_orbital_occupations(); // vector<arma::vec>, matching order

    auto find_homo_lumo = [](const arma::vec &energies, const arma::vec &occ,
                             double occ_thresh = 0.5) {
      double homo = -std::numeric_limits<double>::infinity();
      double lumo = std::numeric_limits<double>::infinity();
      for (arma::uword i = 0; i < energies.n_elem; ++i) {
        if (occ(i) > occ_thresh)
          homo = std::max(homo, energies(i));
        else
          lumo = std::min(lumo, energies(i));
      }
      return std::make_pair(homo, lumo);
    };

    auto [homo_a, lumo_a] = find_homo_lumo(orbital_energies[0], occupations[0]);
    auto [homo_b, lumo_b] = find_homo_lumo(orbital_energies[1], occupations[1]);

    double homo = std::max(homo_a, homo_b);
    double lumo = std::min(lumo_a, lumo_b);
    constexpr double Eh_to_eV = 27.211386245988;

    std::cout << "\nHOMO energy : " << std::setprecision(10) << homo << " Eh  ("
              << homo * Eh_to_eV << " eV)\n";
    std::cout << "LUMO energy : " << std::setprecision(10) << lumo << " Eh  ("
              << lumo * Eh_to_eV << " eV)\n";
    std::cout << "HOMO-LUMO gap: " << (lumo - homo) << " Eh  ("
              << (lumo - homo) * Eh_to_eV << " eV)\n";
  }
  Kokkos::finalize();
  return 0;
}
