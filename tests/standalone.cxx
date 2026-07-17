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

#include <iomanip>
#include <nukexc/core_hamiltonian.hpp>
#include <nukexc/coulomb.hpp>
#include <nukexc/diagonalizer.hpp>
#include <nukexc/exact_exchange.hpp>
#include <nukexc/grid.hpp>
#include <nukexc/molecule.hpp>
#include <nukexc/nuclear_repulsion.hpp>
#include <nukexc/nukexc_config.hpp>
#include <nukexc/octree.hpp>
#include <nukexc/partitioning.hpp>
#include <nukexc/stobasis.hpp>
#include <nukexc/xc_integrals.hpp>

#include <openorbitaloptimizer/scfsolver.hpp>
#include <xc.h>
#include <xc_funcs.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <tuple>
#include <vector>

using namespace Nukexc;
using bk_type = IntegratorXX::Becke<double, double>;
using ta_type = IntegratorXX::TreutlerAhlrichs<double, double>;
using ll_type = IntegratorXX::LebedevLaikov<double>;

// ===========================================================================
// Timing infrastructure
// ===========================================================================
struct StackEntry {
  std::string name;
  std::unique_ptr<Kokkos::Timer> timer;
  int depth;
};

class TimingRegistry {
public:
  struct Entry {
    std::string name;
    double seconds;
    int depth;
  };

  void start(const std::string &name) {
    stack_.push_back(
        StackEntry{name, std::make_unique<Kokkos::Timer>(), depth_});
    ++depth_;
  }

  // Record a pre-measured duration directly (e.g. a total merged from
  // another registry) without going through start()/stop().
  void record(const std::string &name, double seconds) {
    entries_.push_back(Entry{name, seconds, depth_});
  }

  void stop() {
    --depth_;
    StackEntry top = std::move(stack_.back());
    stack_.pop_back();
    entries_.push_back(Entry{top.name, top.timer->seconds(), top.depth});
  }

  // Total time of all depth-0 (top level) blocks recorded so far.
  double total_top_level_seconds() const {
    double total = 0.0;
    for (const auto &e : entries_)
      if (e.depth == 0)
        total += e.seconds;
    return total;
  }

  void report(const std::string &title) const {
    std::cout << "\n"
              << title << "\n"
              << std::string(title.size(), '=') << "\n";
    if (entries_.empty()) {
      std::cout << "  (no timed sections recorded)\n\n";
      return;
    }
    constexpr int name_col = 42;
    for (const auto &e : entries_) {
      std::string indented = std::string(e.depth * 2, ' ') + e.name;
      std::cout << "  " << std::left << std::setw(name_col) << indented << ": "
                << std::right << std::fixed << std::setprecision(4)
                << std::setw(10) << e.seconds << " s\n";
    }
    std::cout << "  " << std::string(name_col + 15, '-') << "\n";
    std::cout << "  " << std::left << std::setw(name_col) << "Total (top level)"
              << ": " << std::right << std::fixed << std::setprecision(4)
              << std::setw(10) << total_top_level_seconds() << " s\n\n"
              << std::flush;
  }

  void clear() {
    entries_.clear();
    stack_.clear();
    depth_ = 0;
  }

private:
  std::vector<StackEntry> stack_;
  std::vector<Entry> entries_;
  int depth_ = 0;
};

// RAII scope: starts a named timer on construction, stops it on destruction.
struct ScopedTiming {
  ScopedTiming(TimingRegistry &reg, std::string name) : reg_(reg) {
    reg_.start(name);
  }
  ~ScopedTiming() { reg_.stop(); }
  ScopedTiming(const ScopedTiming &) = delete;
  ScopedTiming &operator=(const ScopedTiming &) = delete;
  TimingRegistry &reg_;
};

#define NUKEXC_CONCAT_(a, b) a##b
#define NUKEXC_CONCAT(a, b) NUKEXC_CONCAT_(a, b)
// Usage: { TIME_SCOPE(registry, "Some step"); do_work(); }
#define TIME_SCOPE(reg, name)                                                  \
  ScopedTiming NUKEXC_CONCAT(_scoped_timer_, __LINE__)((reg), (name))

// ===========================================================================
// Compile-time string hashing so we can "switch" on a runtime std::string.
// C++ can't switch on std::string directly, but case labels only need to be
// constant expressions -- so we hash the *literal* case labels at compile
// time via this constexpr function, and hash the *runtime* input string with
// the exact same function at runtime, then switch on the resulting uint32_t.
// ===========================================================================
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
  int xc_id = -1;
  XCFamily family = XCFamily::LDA;
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
  (void)info;
  switch (func.info->family) {
  case XC_FAMILY_LDA:
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

// ===========================================================================
// Config
// ===========================================================================
struct Config {
  std::string xyz_file = "input/water.xyz";
  std::string basis_file = "input/zorabasis_cholesky/TZ2P.cholesky";
  std::string method = "hf";       // "hf" or "dft"
  std::string algorithm = "dense"; // "dense" or "sparse"
  int nrad = 100;
  int nang = 30;
  double lin_dep_threshold = 1e-6;
  double conv_thr = 1e-8;
  int charge = 0;
  int multiplicity = 1;
  int max_points_per_box = 32; // runtime via --box-size=
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
          << "  --xyz=<file>          Molecule XYZ file          (default: "
          << cfg.xyz_file << ")\n"
          << "  --basis=<file>        Basis set file             (default: "
          << cfg.basis_file << ")\n"
          << "  --method=<string>     'hf' or 'dft'              (default: "
          << cfg.method << ")\n"
          << "  --alg=<string>        'dense' or 'sparse'        (default: "
          << cfg.algorithm << ")\n"
          << "  --box-size=<int>      Max quadrature points/box  (default: "
          << cfg.max_points_per_box << ")  [only used by --alg=sparse]\n"
          << "  --nrad=<int>          Radial grid points         (default: "
          << cfg.nrad << ")\n"
          << "  --nang=<int>          Angular grid points        (default: "
          << cfg.nang << ")\n"
          << "  --lin-dep=<float>     Linear dep. threshold      (default: "
          << cfg.lin_dep_threshold << ")\n"
          << "  --conv-thr=<float>    SCF convergence threshold  (default: "
          << cfg.conv_thr << ")\n"
          << "  --charge=<int>        Molecular charge           (default: "
          << cfg.charge << ")\n"
          << "  --multiplicity=<int>  Spin multiplicity          (default: "
          << cfg.multiplicity << ")\n"
          << "  --xfunc=<name>        Exchange functional (dft)  (default: "
          << cfg.xfunc << ")\n"
          << "  --cfunc=<name>        Correlation functional(dft)(default: "
          << cfg.cfunc << ")\n"
          << "\n"
          << "  Supported functional names (only relevant for --method=dft):\n"
          << "    LDA exchange:     lda_x\n"
          << "    LDA correlation:  lda_c_pw, lda_c_vwn\n"
          << "    GGA exchange:     gga_x_pbe, gga_x_b88, gga_x_pw91\n"
          << "    GGA correlation:  gga_c_pbe, gga_c_lyp, gga_c_pw91\n";
      std::exit(0);
    } else if (!parse_string("--xyz=", cfg.xyz_file) &&
               !parse_string("--basis=", cfg.basis_file) &&
               !parse_string("--method=", cfg.method) &&
               !parse_string("--alg=", cfg.algorithm) &&
               !parse_int("--box-size=", cfg.max_points_per_box) &&
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
  if (cfg.max_points_per_box <= 0)
    throw std::runtime_error("--box-size must be positive");
  if (cfg.method != "hf" && cfg.method != "dft")
    throw std::runtime_error("Unknown --method: " + cfg.method +
                             " (expected 'hf' or 'dft')");
  if (cfg.algorithm != "dense" && cfg.algorithm != "sparse")
    throw std::runtime_error("Unknown --alg: " + cfg.algorithm +
                             " (expected 'dense' or 'sparse')");
  return cfg;
}

auto repeat(const std::string &s, int n) {
  std::string r;
  for (int i = 0; i < n; ++i)
    r += s;
  return r;
}

void print_config(const Config &cfg) {
  size_t width =
      std::max({cfg.xyz_file.size(), cfg.basis_file.size(),
                cfg.xfunc.size() + cfg.cfunc.size() + 3, size_t(20)});
  std::string h = repeat("─", static_cast<int>(width) + 2);

  std::cout << "\n";
  std::cout << "┌───────────────────────" << h << "┐\n";
  std::cout << "│    " << (cfg.method == "dft" ? "DFT" : "HF")
            << " Configuration"
            << repeat(" ", width + (cfg.method == "dft" ? 5 : 6)) << "│\n";
  std::cout << "├───────────────────────" << h << "┤\n";
  std::cout << "│ Molecule file        │ " << std::setw(width) << std::left
            << cfg.xyz_file << " │\n";
  std::cout << "│ Basis file           │ " << std::setw(width) << std::left
            << cfg.basis_file << " │\n";
  std::cout << "│ Method               │ " << std::setw(width) << std::left
            << cfg.method << " │\n";
  std::cout << "│ Algorithm            │ " << std::setw(width) << std::left
            << cfg.algorithm << " │\n";
  std::cout << "│ Box size (pts/box)   │ " << std::setw(width) << std::left
            << cfg.max_points_per_box << " │\n";
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
  if (cfg.method == "dft") {
    std::cout << "│ Exchange functional  │ " << std::setw(width) << std::left
              << cfg.xfunc << " │\n";
    std::cout << "│ Correlation function │ " << std::setw(width) << std::left
              << cfg.cfunc << " │\n";
  }
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

int main(int argc, char *argv[]) {
  Config cfg;
  try {
    cfg = parse_args(argc, argv);
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
  print_config(cfg);

  Kokkos::Timer program_timer;
  TimingRegistry startup_timing;
  TimingRegistry fock_cumulative_timing;

  Kokkos::initialize();
  {
    const bool is_dft = (cfg.method == "dft");
    const bool is_sparse = (cfg.algorithm == "sparse");
    const bool need_dense_aux_colloc = !is_sparse; // dense (A|B) + J/K paths
    const bool need_basis_colloc = !is_sparse || is_dft;
    const bool need_grad_colloc = is_dft; // needed for GGA quadrature

    Molecule mol;
    {
      TIME_SCOPE(startup_timing, "Read XYZ");
      read_xyz(cfg.xyz_file, mol);
    }

    const double screening_tol = 1e-10;

    FlatGrid grid;
    {
      TIME_SCOPE(startup_timing, "Build molecular grid");
      grid = make_flat_grid<bk_type, ll_type>(mol, cfg.nrad, cfg.nang,
                                              screening_tol * 1e-5);
    }

    STOBasisSet basis;
    STOBasisSet basis_aux;
    {
      TIME_SCOPE(startup_timing, "Load basis sets");
      {
        TIME_SCOPE(startup_timing, "Load primary basis");
        basis = load_adf_basis(mol, cfg.basis_file, screening_tol);
      }
      {
        TIME_SCOPE(startup_timing, "Load auxiliary (fitting) basis");
        basis_aux =
            load_adf_basis(mol, cfg.basis_file, screening_tol, /*fit=*/true);
      }
    }

    // ---- Resolve functionals from runtime strings (DFT only) -------------
    FunctionalInfo x_info{};
    FunctionalInfo c_info{};
    xc_func_type func_x{};
    xc_func_type func_c{};
    bool has_separate_c = false;
    double a_exx = 0.0;

    if (is_dft) {
      TIME_SCOPE(startup_timing, "Resolve/init XC functionals");

      x_info = lookup_functional(cfg.xfunc);
      if (xc_func_init(&func_x, x_info.xc_id, XC_POLARIZED) != 0) {
        throw std::runtime_error(
            "Failed to initialize Libxc exchange functional '" +
            x_info.canonical_name + "'");
      }

      // Libxc defines: XC_EXCHANGE=0, XC_CORRELATION=1,
      // XC_EXCHANGE_CORRELATION=2
      const bool is_combined_xc =
          (func_x.info->kind == XC_EXCHANGE_CORRELATION);
      c_info = lookup_functional(cfg.cfunc);

      if (is_combined_xc) {
        // Combined functionals (B3LYP, M06, ...) already include correlation.
        has_separate_c = false;
      } else {
        if (xc_func_init(&func_c, c_info.xc_id, XC_POLARIZED) != 0) {
          xc_func_end(&func_x);
          throw std::runtime_error(
              "Failed to initialize Libxc correlation functional '" +
              c_info.canonical_name + "'");
        }
        has_separate_c = true;
      }

      a_exx = xc_hyb_exx_coef(&func_x);
    }

    // ---- Sizes -------------------------------------------------------------
    const int N_bf = basis.nbf();
    const int N_bf_aux = basis_aux.nbf();
    const int N_quad = grid.quad_points.extent(0);

    ExecSpace space;

    // ---- Neighbor lists (sparse path only) ---------------------------------
    // NOTE: create_bounding_boxes() PERMUTES grid.quad_points and grid.weights
    // in place (Morton ordering). It must therefore run BEFORE any collocation
    // is filled, otherwise the collocation columns refer to the pre-permutation
    // point ordering while grid.weights uses the post-permutation ordering,
    // which silently corrupts every subsequent quadrature

    NeighborList nl;     // primary basis neighbor list (K/J sparse)
    NeighborList nl_aux; // auxiliary basis neighbor list (aux overlap sparse)

    if (is_sparse) {
      TIME_SCOPE(startup_timing, "Build neighbor lists (sparse)");
      {
        TIME_SCOPE(startup_timing, "Bounding boxes + neighbor list (aux)");
        auto bb_aux = create_bounding_boxes(grid, cfg.max_points_per_box);
        build_neighbor_list(basis_aux, bb_aux, cfg.max_points_per_box, N_quad,
                            nl_aux);
      }
      {
        TIME_SCOPE(startup_timing, "Bounding boxes + neighbor list (primary)");
        auto bb = create_bounding_boxes(grid, cfg.max_points_per_box);
        build_neighbor_list(basis, bb, cfg.max_points_per_box, N_quad, nl);
      }
    }

    // ---- Collocations (dense path and/or DFT quadrature) ------------------
    DeviceView2DLeft basis_collocation;
    DeviceView2DLeft basis_aux_collocation;
    DeviceView2DLeft potential_collocation_scaled;
    DeviceView2DLeft basis_collocation_gx, basis_collocation_gy,
        basis_collocation_gz;

    DeviceView2DLeft aux_overlap("Aux overlap", N_bf_aux, N_bf_aux);
    DeviceView2DLeft aux_overlap_sym("Aux overlap sym", N_bf_aux, N_bf_aux);

    if (need_basis_colloc) {
      TIME_SCOPE(startup_timing, "Fill primary basis collocation");
      basis_collocation = DeviceView2DLeft("Basis collocation", N_bf, N_quad);
      fill_collocation(space, basis, grid.quad_points, basis_collocation);
    }

    if (need_grad_colloc) {
      TIME_SCOPE(startup_timing, "Fill primary basis gradient collocation");
      basis_collocation_gx =
          DeviceView2DLeft("Basis collocation dX", N_bf, N_quad);
      basis_collocation_gy =
          DeviceView2DLeft("Basis collocation dY", N_bf, N_quad);
      basis_collocation_gz =
          DeviceView2DLeft("Basis collocation dZ", N_bf, N_quad);
      fill_grad_collocation(space, basis, grid.quad_points,
                            basis_collocation_gx, basis_collocation_gy,
                            basis_collocation_gz);
    }

    if (need_dense_aux_colloc) {
      TIME_SCOPE(startup_timing,
                 "Fill auxiliary basis collocation + potential");
      basis_aux_collocation =
          DeviceView2DLeft("Auxillary Basis collocation", N_bf_aux, N_quad);
      potential_collocation_scaled =
          DeviceView2DLeft("Potential collocation", N_bf_aux, N_quad);
      fill_collocation(space, basis_aux, grid.quad_points,
                       basis_aux_collocation);
      sto_potential_collocation_scaled(space, basis_aux, grid,
                                       potential_collocation_scaled);
    }

    // ---- Auxiliary overlap (A|B), dense or sparse --------------------------
    if (is_sparse) {
      TIME_SCOPE(startup_timing, "Auxiliary Coulomb overlap (sparse)");
      aux_overlap_sym =
          coulomb_overlap_integral_sparse(space, basis_aux, grid, nl_aux);
    } else {
      TIME_SCOPE(startup_timing, "Auxiliary Coulomb overlap (dense GEMM)");
      KokkosBlas::gemm(space, "N", "T", 1.0, basis_aux_collocation,
                       potential_collocation_scaled, 0.0, aux_overlap);
      Kokkos::parallel_for(
          Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {N_bf_aux, N_bf_aux}),
          KOKKOS_LAMBDA(int i, int j) {
            aux_overlap_sym(i, j) =
                0.5 * (aux_overlap(i, j) + aux_overlap(j, i));
          });
    }

    DeviceView2DLeft half_inverse_X;
    {
      TIME_SCOPE(startup_timing, "Half-inverse of auxiliary overlap");
      half_inverse_X =
          compute_half_inverse(aux_overlap_sym, cfg.lin_dep_threshold);
    }

    // ---- Core Hamiltonian ---------------------------------------------------
    decltype(compute_core_hamiltonian(basis, grid)) hcore;
    {
      TIME_SCOPE(startup_timing, "Core Hamiltonian (T + V_ne + S)");
      hcore = compute_core_hamiltonian(basis, grid);
    }

    // ---- Orthogonalisation matrix X
    // ------------------------------------------
    DeviceView2DLeft X;
    {
      TIME_SCOPE(startup_timing, "Orthogonalization matrix X = S^-1/2");
      Diagonalizer diag(N_bf);
      X = diag.compute_transformation(hcore.overlap, cfg.lin_dep_threshold);
    }

    arma::mat h_core, X_arma, S_arma;
    {
      TIME_SCOPE(startup_timing, "Copy core matrices to host (arma)");
      h_core = kokkos_to_arma(hcore.hamiltonian);
      X_arma = kokkos_to_arma(X);
      S_arma = kokkos_to_arma(hcore.overlap);
    }

    // Derive electron counts from geometry + charge + multiplicity
    int n_elec = mol.Z_total - cfg.charge;
    std::cout << "Number of Electrons: " << n_elec << std::endl;
    std::cout << "Z_total: " << mol.Z_total << std::endl;
    if ((n_elec + cfg.multiplicity - 1) % 2 != 0)
      throw std::runtime_error(
          "Charge and multiplicity are inconsistent with the number of "
          "electrons");
    double n_alpha = (n_elec + (cfg.multiplicity - 1)) / 2.0;
    double n_beta = (n_elec - (cfg.multiplicity - 1)) / 2.0;

    // ---- OOO setup
    // ------------------------------------------------------------
    arma::uvec blocks_per_type = {1, 1};    // 2 spin blocks (alpha and beta)
    arma::vec max_occupations = {1.0, 1.0}; // max 1 electron per spin channel
    arma::vec number_of_particles = {n_alpha, n_beta};
    std::vector<std::string> block_descriptions = {"alpha", "beta"};

    double E_nuc;
    {
      TIME_SCOPE(startup_timing, "Nuclear repulsion energy");
      E_nuc = compute_nuclear_repulsion(mol);
    }

    // ---- GWH initial guess -----------------------------------------------
    arma::mat F_gwh_orth;
    {
      TIME_SCOPE(startup_timing, "GWH initial guess");
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
      F_gwh_orth = X_arma.t() * F_gwh * X_arma;
    }

    startup_timing.report("Startup Timing Breakdown");

    // ---- Fock builder (unified HF/DFT, dense/sparse) ------------------------
    // A shared call counter so we can label each SCF iteration's timing
    // report; the underlying int is shared across lambda copies.
    auto call_count = std::make_shared<int>(0);

    auto fock_builder =
        [space, grid, basis, basis_aux, nl, basis_collocation,
         basis_collocation_gx, basis_collocation_gy, basis_collocation_gz,
         basis_aux_collocation, potential_collocation_scaled, h_core, X_arma,
         half_inverse_X, N_bf, E_nuc, S_arma, cfg, func_c, func_x, x_info,
         c_info, has_separate_c, a_exx, is_dft = is_dft, is_sparse = is_sparse,
         call_count, &fock_cumulative_timing](
            const OpenOrbitalOptimizer::DensityMatrix<double, double> &dm)
        -> std::pair<double, OpenOrbitalOptimizer::FockMatrix<double>> {
      TimingRegistry fock_timing;
      ++(*call_count);

      const auto &orbitals = dm.first;     // vector<arma::mat>, one per block
      const auto &occupations = dm.second; // vector<arma::vec>, one per block

      arma::mat C_alpha, C_beta;
      arma::vec occ_alpha, occ_beta;
      arma::mat D_alpha, D_beta, D_tot;
      {
        TIME_SCOPE(fock_timing, "Build density matrices");
        C_alpha = X_arma * orbitals[0];
        C_beta = X_arma * orbitals[1];
        occ_alpha = occupations[0]; // 0 or 1
        occ_beta = occupations[1];  // 0 or 1

        D_alpha = C_alpha * arma::diagmat(occ_alpha) * C_alpha.t();
        D_beta = C_beta * arma::diagmat(occ_beta) * C_beta.t();
        D_tot = D_alpha + D_beta;
      }

#ifndef NDEBUG
      std::cout << "Tr[D_alpha * S] = " << arma::trace(D_alpha * S_arma)
                << "\n";
      std::cout << "Tr[D_beta  * S] = " << arma::trace(D_beta * S_arma) << "\n";
      std::cout << "Tr[D_tot   * S] = " << arma::trace(D_tot * S_arma) << "\n";
#endif

      // Combined alpha+beta orbitals/occupations, used for the total-density
      // Coulomb (J) build (same total density regardless of HF vs DFT).
      DeviceView2DLeft k_C_tot, k_C_alpha, k_C_beta;
      DeviceView1D k_occ_tot, k_occ_alpha, k_occ_beta;
      {
        TIME_SCOPE(fock_timing, "Host->device transfer of orbitals");
        arma::mat C_combined = arma::join_horiz(C_alpha, C_beta);
        arma::vec occ_combined = arma::join_vert(occ_alpha, occ_beta);

        k_C_tot = arma_to_kokkos(C_combined, "C_combined");
        k_occ_tot = arma_to_kokkos1d(occ_combined, "occ_combined");
        k_C_alpha = arma_to_kokkos(C_alpha, "C_alpha");
        k_C_beta = arma_to_kokkos(C_beta, "C_beta");
        k_occ_alpha = arma_to_kokkos1d(occ_alpha, "occ_alpha");
        k_occ_beta = arma_to_kokkos1d(occ_beta, "occ_beta");
      }

      // ---- Coulomb (J) build: same for HF and DFT, dense or sparse --------
      arma::mat J_arma;
      {
        TIME_SCOPE(fock_timing, "Coulomb (J) build");
        DeviceView2DLeft J;
        if (is_sparse) {
          J = compute_coulomb_tiled(space, k_C_tot, k_occ_tot, basis,
                                     basis_aux, grid, nl, half_inverse_X);
        } else {
          J = compute_coulomb(space, k_C_tot, k_occ_tot, basis_collocation,
                              basis_aux_collocation,
                              potential_collocation_scaled, half_inverse_X);
        }
        J_arma = kokkos_to_arma(J);
      }

      double E_exchange = 0.0;       // HF: exact exchange energy
                                     // DFT: exchange-functional energy (or
                                     // combined XC energy for combined funcs)
      double E_correlation = 0.0;    // DFT correlation-functional energy only
      double E_exact_exchange = 0.0; // DFT hybrid exact-exchange energy

      arma::mat K_alpha_arma = arma::zeros<arma::mat>(N_bf, N_bf);
      arma::mat K_beta_arma = arma::zeros<arma::mat>(N_bf, N_bf);
      arma::mat Vx_alpha = arma::zeros<arma::mat>(N_bf, N_bf);
      arma::mat Vx_beta = arma::zeros<arma::mat>(N_bf, N_bf);
      arma::mat Vc_alpha = arma::zeros<arma::mat>(N_bf, N_bf);
      arma::mat Vc_beta = arma::zeros<arma::mat>(N_bf, N_bf);

      if (!is_dft) {
        // ---- Pure HF exact exchange -----------------------------------
        TIME_SCOPE(fock_timing, "Exact exchange (K) build");
        DeviceView2DLeft K_alpha, K_beta;
        {
          TIME_SCOPE(fock_timing, "  K alpha");
          if (is_sparse) {
            K_alpha = compute_exact_exchange_tiled(
                space, k_C_alpha, k_occ_alpha, basis, basis_aux, grid, nl,
                half_inverse_X);
          } else {
            K_alpha = compute_exact_exchange(
                space, k_C_alpha, k_occ_alpha, basis_collocation,
                basis_aux_collocation, potential_collocation_scaled,
                half_inverse_X);
          }
        }
        {
          TIME_SCOPE(fock_timing, "  K beta");
          if (is_sparse) {
            K_beta = compute_exact_exchange_tiled(space, k_C_beta, k_occ_beta,
                                                   basis, basis_aux, grid, nl,
                                                   half_inverse_X);
          } else {
            K_beta = compute_exact_exchange(
                space, k_C_beta, k_occ_beta, basis_collocation,
                basis_aux_collocation, potential_collocation_scaled,
                half_inverse_X);
          }
        }
        K_alpha_arma = kokkos_to_arma(K_alpha);
        K_beta_arma = kokkos_to_arma(K_beta);
        E_exchange = -0.5 * arma::trace(D_alpha * K_alpha_arma) -
                     0.5 * arma::trace(D_beta * K_beta_arma);
      } else {
        // ---- DFT exchange-correlation quadrature -------------------------
        {
          TIME_SCOPE(fock_timing, "Exchange functional (Vx)");
          XC_result_polarized E_x = evaluate_functional(
              x_info, func_x, basis_collocation, basis_collocation_gx,
              basis_collocation_gy, basis_collocation_gz, grid.weights,
              k_C_alpha, k_occ_alpha, k_C_beta, k_occ_beta);
          E_exchange = E_x.energy; // holds total E_xc if combined functional
          Vx_alpha = kokkos_to_arma(E_x.potential_alpha);
          Vx_beta = kokkos_to_arma(E_x.potential_beta);
        }

        if (has_separate_c) {
          TIME_SCOPE(fock_timing, "Correlation functional (Vc)");
          XC_result_polarized E_c = evaluate_functional(
              c_info, func_c, basis_collocation, basis_collocation_gx,
              basis_collocation_gy, basis_collocation_gz, grid.weights,
              k_C_alpha, k_occ_alpha, k_C_beta, k_occ_beta);
          E_correlation = E_c.energy;
          Vc_alpha = kokkos_to_arma(E_c.potential_alpha);
          Vc_beta = kokkos_to_arma(E_c.potential_beta);
        }

        if (a_exx > 0) {
          TIME_SCOPE(fock_timing, "Exact exchange (hybrid) build");
          DeviceView2DLeft K_alpha, K_beta;
          {
            TIME_SCOPE(fock_timing, "  K alpha");
            if (is_sparse) {
              K_alpha = compute_exact_exchange_sparse(
                  space, k_C_alpha, k_occ_alpha, basis, basis_aux, grid, nl,
                  half_inverse_X);
            } else {
              K_alpha = compute_exact_exchange(
                  space, k_C_alpha, k_occ_alpha, basis_collocation,
                  basis_aux_collocation, potential_collocation_scaled,
                  half_inverse_X);
            }
          }
          {
            TIME_SCOPE(fock_timing, "  K beta");
            if (is_sparse) {
              K_beta = compute_exact_exchange_sparse(
                  space, k_C_beta, k_occ_beta, basis, basis_aux, grid, nl,
                  half_inverse_X);
            } else {
              K_beta = compute_exact_exchange(
                  space, k_C_beta, k_occ_beta, basis_collocation,
                  basis_aux_collocation, potential_collocation_scaled,
                  half_inverse_X);
            }
          }
          K_alpha_arma = kokkos_to_arma(K_alpha);
          K_beta_arma = kokkos_to_arma(K_beta);
          E_exact_exchange = -0.5 * a_exx *
                             (arma::trace(D_alpha * K_alpha_arma) +
                              arma::trace(D_beta * K_beta_arma));
        }
      }

      double Etot;
      double E_core, E_coulomb;
      {
        TIME_SCOPE(fock_timing, "Energy assembly");
        E_core = arma::trace(D_tot * h_core);
        E_coulomb = 0.5 * arma::trace(D_tot * J_arma);
        Etot = E_nuc + E_core + E_coulomb + E_exchange + E_correlation +
               E_exact_exchange;
      }

      std::cout << "Total energy        : " << std::setprecision(10) << Etot
                << "\n";
      std::cout << "--------------------------------------------- \n";
      std::cout << "Nuclear Repulsion   : " << std::setprecision(10) << E_nuc
                << "\n";
      std::cout << "Electronic   Energy : " << std::setprecision(10)
                << E_core + E_coulomb + E_exchange + E_correlation +
                       E_exact_exchange
                << "\n";
      std::cout << "One Electron Energy : " << std::setprecision(10) << E_core
                << "\n";
      std::cout << "Two Electron Energy : " << std::setprecision(10)
                << E_coulomb + E_exchange + E_correlation + E_exact_exchange
                << "\n\n";

      arma::mat F_alpha_orth, F_beta_orth;
      {
        TIME_SCOPE(fock_timing, "Fock assembly + orthogonalization");
        arma::mat F_alpha, F_beta;
        if (!is_dft) {
          F_alpha = h_core + J_arma - K_alpha_arma;
          F_beta = h_core + J_arma - K_beta_arma;
        } else {
          F_alpha =
              h_core + J_arma - a_exx * K_alpha_arma + Vx_alpha + Vc_alpha;
          F_beta = h_core + J_arma - a_exx * K_beta_arma + Vx_beta + Vc_beta;
        }
        F_alpha_orth = X_arma.t() * F_alpha * X_arma;
        F_beta_orth = X_arma.t() * F_beta * X_arma;
      }

      fock_timing.report("Fock Build #" + std::to_string(*call_count) +
                         " Timing Breakdown");

      // Record this call's real total (sum of its top-level timed sections)
      // into the cumulative registry, so the summary reflects actual
      // per-call cost rather than bookkeeping overhead.
      fock_cumulative_timing.record("Fock build #" +
                                        std::to_string(*call_count),
                                    fock_timing.total_top_level_seconds());

      std::vector<arma::mat> fock_arma;
      fock_arma.push_back(F_alpha_orth);
      fock_arma.push_back(F_beta_orth);

      return std::make_pair(Etot, fock_arma);
    };

    // ---- Construct and run SCF solver
    // ----------------------------------------
    OpenOrbitalOptimizer::SCFSolver<double, double> solver(
        blocks_per_type, max_occupations, number_of_particles, fock_builder,
        block_descriptions);

    solver.convergence_threshold(cfg.conv_thr);
    solver.verbosity(5);
    solver.initialize_with_fock({F_gwh_orth, F_gwh_orth});

    Kokkos::Timer scf_timer;
    solver.run();
    const double scf_seconds = scf_timer.seconds();

    if (is_dft) {
      xc_func_end(&func_x);
      if (has_separate_c)
        xc_func_end(&func_c);
    }

    // ---- Check SCF converged to a sensible energy
    // -----------------------------
    double E_scf = solver.get_fock_build().first;
    std::cout << "SCF total energy: " << E_scf << " Eh\n";
    std::cout << "SCF wall time   : " << std::fixed << std::setprecision(4)
              << scf_seconds << " s over " << *call_count
              << " Fock build(s) (avg "
              << (*call_count > 0 ? scf_seconds / *call_count : 0.0)
              << " s/build)\n";

    fock_cumulative_timing.report(
        "Cumulative Fock Build Timing (per-call totals)");

    // ---- HOMO-LUMO gap
    // ---------------------------------------------------------
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

  std::cout << "\nTotal program wall time: " << std::fixed
            << std::setprecision(4) << program_timer.seconds() << " s\n";
  return 0;
}
