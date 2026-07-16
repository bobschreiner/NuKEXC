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

#include "nukexc/nukexc_config.hpp"
#include "nukexc/octree.hpp"
#include <Kokkos_Core.hpp>

#include <Kokkos_Core_fwd.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <integratorxx/quadratures/radial/treutlerahlrichs.hpp>
#include <iomanip>
#include <iostream>

#include <nukexc/grid.hpp>
#include <nukexc/molecule.hpp>
#include <nukexc/stobasis.hpp>

#include <catch2/catch_all.hpp>
#include <catch2/catch_assertion_info.hpp>
#include <vector>

using namespace Nukexc;

struct Config {
  int benchmark = 0;
  std::string algorithm = "fasts";
  int nrad = 50;
  int nang = 20;
  double screening_tol = 1e-6;
  int max_points_per_box = 32;
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
          << "  --benchmark=<int> Benchmark? (0=No, 1=Yes) (default: "
          << cfg.benchmark << ")\n"

          << "  --alg=<string>    Algorithm(fasts,fastv,slows,slowv) (default: "
          << cfg.algorithm << ")\n"
          << "  --nrad=<int>              Radial grid points        (default: "
          << cfg.nrad << ")\n"
          << "  --nang=<int>              Angular grid points       (default: "
          << cfg.nang << ")\n"
          << "  --tol=<float>             Screening tolerance       (default: "
          << cfg.screening_tol << ")\n"
          << "  --box-size=<int>          Max points per box        (default: "
          << cfg.max_points_per_box << ")\n";
      std::exit(0);
    } else if (!parse_int("--benchmark=", cfg.benchmark) &&
               !parse_string("--alg=", cfg.algorithm) &&
               !parse_int("--nrad=", cfg.nrad) &&
               !parse_int("--nang=", cfg.nang) &&
               !parse_double("--tol=", cfg.screening_tol) &&
               !parse_int("--box-size=", cfg.max_points_per_box)) {
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
  int width = size_t(30);
  std::string h = repeat("─", width + 2);

  std::cout << "\n";
  std::cout << "┌───────────────────────" << h << "┐\n";
  std::cout << "│    SCF Benchmark Configuration" << repeat(" ", width - 6)
            << "│\n";
  std::cout << "├───────────────────────" << h << "┤\n";
  std::cout << "│ Benchmark            │ " << std::setw(width) << cfg.benchmark
            << " │\n";
  std::cout << "│ Algorithm            │ " << std::setw(width) << cfg.algorithm
            << " │\n";
  std::cout << "│ Radial points        │ " << std::setw(width) << cfg.nrad
            << " │\n";
  std::cout << "│ Angular order        │ " << std::setw(width) << cfg.nang
            << " │\n";
  std::cout << "│ Screening tolerance  │ " << std::setw(width)
            << cfg.screening_tol << " │\n";
  std::cout << "│ Max points per box   │ " << std::setw(width)
            << cfg.max_points_per_box << " │\n";
  std::cout << "└──────────────────────┴" << h << "┘\n\n";
  std::cout << std::flush;
}
TEST_CASE("H2O_thakkar", "[h20_thakkar]") {
  Molecule mol;
  read_xyz("input/water.xyz", mol);
  STOBasisSet basis = load_thakkar_basis(mol, "input/k99light/neutral");

  // On CPU we can print out the basis set
  // Copy to host device
  auto n_h = Kokkos::create_mirror_view(basis.n);
  auto l_h = Kokkos::create_mirror_view(basis.l);
  auto m_h = Kokkos::create_mirror_view(basis.m);
  auto norm_h = Kokkos::create_mirror_view(basis.norm);
  auto zeta_h = Kokkos::create_mirror_view(basis.zeta);
  auto O_h = Kokkos::create_mirror_view(basis.O);
  auto cutoff_h = Kokkos::create_mirror_view(basis.cutoff_radii);

  Kokkos::deep_copy(n_h, basis.n);
  Kokkos::deep_copy(l_h, basis.l);
  Kokkos::deep_copy(m_h, basis.m);
  Kokkos::deep_copy(zeta_h, basis.zeta);
  Kokkos::deep_copy(norm_h, basis.norm);
  Kokkos::deep_copy(O_h, basis.O);
  Kokkos::deep_copy(cutoff_h, basis.cutoff_radii);

#if 0
  std::cout << "Thakkar Basis" << std::endl;
  for (int i = 0; i < basis.nbf(); ++i) {
    std::cout << "Basis function " << i << std::endl;
    std::cout << "n " << n_h(i) << std::endl;
    std::cout << "l " << l_h(i) << std::endl;
    std::cout << "m " << m_h(i) << std::endl;
    std::cout << "zeta " << zeta_h(i) << std::endl;
    std::cout << "norm " << norm_h(i) << std::endl;
    std::cout << "cutoff " << cutoff_h(i) << std::endl;
    std::cout << "O_h " << O_h(i)[0] << " " << O_h(i)[0] << " " << O_h(i)[2]
              << " " << std::endl
              << std::endl;
  }
  std::cout << "--------------------------------------------------"
            << std::endl;
#endif
};

TEST_CASE("H2O_adf_regular", "[h20][adf]") {
  Molecule mol;
  read_xyz("input/water.xyz", mol);
  STOBasisSet basis = load_adf_basis(mol);

  // On CPU we can print out the basis set
  // Copy to host device
  auto n_h = Kokkos::create_mirror_view(basis.n);
  auto l_h = Kokkos::create_mirror_view(basis.l);
  auto m_h = Kokkos::create_mirror_view(basis.m);
  auto norm_h = Kokkos::create_mirror_view(basis.norm);
  auto zeta_h = Kokkos::create_mirror_view(basis.zeta);
  auto O_h = Kokkos::create_mirror_view(basis.O);
  auto cutoff_h = Kokkos::create_mirror_view(basis.cutoff_radii);

  Kokkos::deep_copy(n_h, basis.n);
  Kokkos::deep_copy(l_h, basis.l);
  Kokkos::deep_copy(m_h, basis.m);
  Kokkos::deep_copy(zeta_h, basis.zeta);
  Kokkos::deep_copy(norm_h, basis.norm);
  Kokkos::deep_copy(O_h, basis.O);
  Kokkos::deep_copy(cutoff_h, basis.cutoff_radii);

#if 0
  std::cout << "ADF TZP" << std::endl;
  for (int i = 0; i < basis.nbf(); ++i) {
    std::cout << "Basis function " << i << std::endl;
    std::cout << "n " << n_h(i) << std::endl;
    std::cout << "l " << l_h(i) << std::endl;
    std::cout << "m " << m_h(i) << std::endl;
    std::cout << "zeta " << zeta_h(i) << std::endl;
    std::cout << "norm " << norm_h(i) << std::endl;
    std::cout << "cutoff " << cutoff_h(i) << std::endl;
    std::cout << "O_h " << O_h(i)[0] << " " << O_h(i)[0] << " " << O_h(i)[2]
              << " " << std::endl
              << std::endl;
  }
  std::cout << "--------------------------------------------------"
            << std::endl;
#endif
};

TEST_CASE("H2O_adf_QZ4P", "[h20][adf]") {
  Molecule mol;
  read_xyz("input/water.xyz", mol);
  STOBasisSet basis = load_adf_basis(mol, "input/zorabasis/QZ4P");

  // On CPU we can print out the basis set
  // Copy to host device
  auto n_h = Kokkos::create_mirror_view(basis.n);
  auto l_h = Kokkos::create_mirror_view(basis.l);
  auto m_h = Kokkos::create_mirror_view(basis.m);
  auto norm_h = Kokkos::create_mirror_view(basis.norm);
  auto zeta_h = Kokkos::create_mirror_view(basis.zeta);
  auto O_h = Kokkos::create_mirror_view(basis.O);
  auto cutoff_h = Kokkos::create_mirror_view(basis.cutoff_radii);

  Kokkos::deep_copy(n_h, basis.n);
  Kokkos::deep_copy(l_h, basis.l);
  Kokkos::deep_copy(m_h, basis.m);
  Kokkos::deep_copy(zeta_h, basis.zeta);
  Kokkos::deep_copy(norm_h, basis.norm);
  Kokkos::deep_copy(O_h, basis.O);
  Kokkos::deep_copy(cutoff_h, basis.cutoff_radii);

#if 0 
  std::cout << "ADF QZ4P" << std::endl;

  for (int i = 0; i < basis.nbf(); ++i) {
    std::cout << "Basis function " << i << std::endl;
    std::cout << "n " << n_h(i) << std::endl;
    std::cout << "l " << l_h(i) << std::endl;
    std::cout << "m " << m_h(i) << std::endl;
    std::cout << "zeta " << zeta_h(i) << std::endl;
    std::cout << "norm " << norm_h(i) << std::endl;
    std::cout << "cutoff " << cutoff_h(i) << std::endl;
    std::cout << "O_h " << O_h(i)[0] << " " << O_h(i)[0] << " " << O_h(i)[2]
              << " " << std::endl
              << std::endl;
  }
  std::cout << "--------------------------------------------------"
            << std::endl;
#endif
};

TEST_CASE("H2O_adf_QZ4P_AUX", "[h20][adf][fit]") {
  Molecule mol;
  read_xyz("input/water.xyz", mol);
  const bool fit = true;
  const double tol = 1e-10;
  STOBasisSet basis = load_adf_basis(mol, "input/zorabasis/QZ4P", tol, fit);

  // On CPU we can print out the basis set
  // Copy to host device
  auto n_h = Kokkos::create_mirror_view(basis.n);
  auto l_h = Kokkos::create_mirror_view(basis.l);
  auto m_h = Kokkos::create_mirror_view(basis.m);
  auto norm_h = Kokkos::create_mirror_view(basis.norm);
  auto zeta_h = Kokkos::create_mirror_view(basis.zeta);
  auto O_h = Kokkos::create_mirror_view(basis.O);
  auto cutoff_h = Kokkos::create_mirror_view(basis.cutoff_radii);

  Kokkos::deep_copy(n_h, basis.n);
  Kokkos::deep_copy(l_h, basis.l);
  Kokkos::deep_copy(m_h, basis.m);
  Kokkos::deep_copy(zeta_h, basis.zeta);
  Kokkos::deep_copy(norm_h, basis.norm);
  Kokkos::deep_copy(O_h, basis.O);
  Kokkos::deep_copy(cutoff_h, basis.cutoff_radii);

#if 0
  std::cout << "--------------------------------------------------"
            << std::endl;

  std::cout << "Auxillary ADF QZ4P" << std::endl;

  for (int i = 0; i < basis.nbf(); ++i) {
    std::cout << "Basis function " << i << std::endl;
    std::cout << "n " << n_h(i) << std::endl;
    std::cout << "l " << l_h(i) << std::endl;
    std::cout << "m " << m_h(i) << std::endl;
    std::cout << "zeta " << zeta_h(i) << std::endl;
    std::cout << "norm " << norm_h(i) << std::endl;
    std::cout << "cutoff " << cutoff_h(i) << std::endl;
    std::cout << "O_h " << O_h(i)[0] << " " << O_h(i)[0] << " " << O_h(i)[2]
              << " " << std::endl
              << std::endl;
  }
  std::cout << "--------------------------------------------------"
            << std::endl;
#endif
};

TEST_CASE("H2O_adf_QZ4P_AUX_cholesky", "[h20][adf][fit][cholesky]") {
  Molecule mol;
  read_xyz("input/water.xyz", mol);
  const bool fit = true;
  const double tol = 1e-10;

  STOBasisSet basis = load_adf_basis(mol, "input/zorabasis/QZ4P", tol);
  STOBasisSet basis_cholesky =
      load_adf_basis(mol, "input/zorabasis_cholesky/QZ4P.cholesky", tol);
  STOBasisSet basis_aux = load_adf_basis(mol, "input/zorabasis/QZ4P", tol, fit);
  STOBasisSet basis_aux_cholesky =
      load_adf_basis(mol, "input/zorabasis_cholesky/QZ4P.cholesky", tol, fit);

  // On CPU we can print out the basis set
  // Copy to host device
  auto n_h = Kokkos::create_mirror_view(basis_aux_cholesky.n);
  auto l_h = Kokkos::create_mirror_view(basis_aux_cholesky.l);
  auto m_h = Kokkos::create_mirror_view(basis_aux_cholesky.m);
  auto norm_h = Kokkos::create_mirror_view(basis_aux_cholesky.norm);
  auto zeta_h = Kokkos::create_mirror_view(basis_aux_cholesky.zeta);
  auto O_h = Kokkos::create_mirror_view(basis_aux_cholesky.O);
  auto cutoff_h = Kokkos::create_mirror_view(basis_aux_cholesky.cutoff_radii);

  Kokkos::deep_copy(n_h, basis_aux_cholesky.n);
  Kokkos::deep_copy(l_h, basis_aux_cholesky.l);
  Kokkos::deep_copy(m_h, basis_aux_cholesky.m);
  Kokkos::deep_copy(zeta_h, basis_aux_cholesky.zeta);
  Kokkos::deep_copy(norm_h, basis_aux_cholesky.norm);
  Kokkos::deep_copy(O_h, basis_aux_cholesky.O);
  Kokkos::deep_copy(cutoff_h, basis_aux_cholesky.cutoff_radii);

#if 0
  std::cout << "--------------------------------------------------"
            << std::endl;

  std::cout << "Auxillary ADF QZ4P Cholesky" << std::endl;

  for (int i = 0; i < basis_aux_cholesky.nbf(); ++i) {
    std::cout << "Basis function " << i << std::endl;
    std::cout << "n " << n_h(i) << std::endl;
    std::cout << "l " << l_h(i) << std::endl;
    std::cout << "m " << m_h(i) << std::endl;
    std::cout << "zeta " << zeta_h(i) << std::endl;
    std::cout << "norm " << norm_h(i) << std::endl;
    std::cout << "cutoff " << cutoff_h(i) << std::endl;
    std::cout << "O_h " << O_h(i)[0] << " " << O_h(i)[0] << " " << O_h(i)[2]
              << " " << std::endl
              << std::endl;
  }
  std::cout << "--------------------------------------------------"
            << std::endl;
#endif

  // Check that both regular basis sets are the same
  REQUIRE(basis.nbf() == basis_cholesky.nbf());

  // Check that the auxillary basis set created with auto Aux is blarger then
  // the regular ADF auxillary basis
  REQUIRE(basis_aux.nbf() < basis_aux_cholesky.nbf());
};

TEST_CASE("Basis Cutoff", "[h20][cutoff]") {

  using bk_type = IntegratorXX::Becke<double, double>;
  using ta_type = IntegratorXX::TreutlerAhlrichs<double, double>;
  using ll_type = IntegratorXX::LebedevLaikov<double>;

  double cutoff_tol = 1e-15;
  Molecule mol;
  read_xyz("input/water.xyz", mol);
  STOBasisSet basis = load_adf_basis(mol, "input/zorabasis/QZ4P", cutoff_tol);

  FlatGrid grid = make_flat_grid<ta_type, ll_type>(mol, 100, 50);

  int N = basis.nbf();
  int G = grid.quad_points.extent(0);

  Kokkos::View<double **> grid_values("Values", N, G);
  Kokkos::View<double **> grid_r("Radii", N, G);

  Kokkos::parallel_for(
      "Evaluate basis", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {N, G}),
      KOKKOS_LAMBDA(const int i, const int g) {
        double r = dist(basis.O(i), grid.quad_points(g));
        basis_eval(basis, i, grid.quad_points(g)[0], grid.quad_points(g)[1],
                   grid.quad_points(g)[2], grid_values(i, g));
        grid_r(i, g) = r;
      });

  auto grid_values_h = Kokkos::create_mirror_view(grid_values);
  auto grid_r_h = Kokkos::create_mirror_view(grid_r);
  auto cutoff_radii_h = Kokkos::create_mirror_view(basis.cutoff_radii);

  Kokkos::fence();
  Kokkos::deep_copy(grid_values_h, grid_values);
  Kokkos::deep_copy(grid_r_h, grid_r);
  Kokkos::deep_copy(cutoff_radii_h, basis.cutoff_radii);

  int count = 0;
  for (unsigned int i = 0; i < N; ++i) {
    for (unsigned int g = 0; g < G; ++g) {
      if (cutoff_radii_h(i) < grid_r_h(i, g)) {
        ++count;
        REQUIRE_THAT(grid_values_h(i, g),
                     Catch::Matchers::WithinAbs(0., cutoff_tol));
      }
    }
  }
};

void benchmark_collocation(Config cfg) {

  using bk_type = IntegratorXX::Becke<double, double>;
  using ta_type = IntegratorXX::TreutlerAhlrichs<double, double>;
  using ll_type = IntegratorXX::LebedevLaikov<double>;

  double cutoff_tol = cfg.screening_tol;
  Molecule mol;
  read_xyz("input/water.xyz", mol);
  STOBasisSet basis = load_adf_basis(mol, "input/zorabasis/QZ4P", cutoff_tol);

  FlatGrid grid = make_flat_grid<bk_type, ll_type>(mol, cfg.nrad, cfg.nang);
  int N_bf = basis.nbf();
  int N_quad = grid.quad_points.extent(0);

  const int max_points_per_box = cfg.max_points_per_box;
  NeighborList nl;
  auto bounding_boxes = create_bounding_boxes(grid, max_points_per_box);
  build_neighbor_list(basis, bounding_boxes, max_points_per_box, N_quad, nl);

  const int num_boxes = nl.offsets.extent(0) - 1;
  Kokkos::View<double **> S("overlap", N_bf,
                            N_bf); // per box, or global-indexed
  ExecSpace space;

  Kokkos::TeamPolicy<ExecSpace> policy_boxes(space, num_boxes, Kokkos::AUTO(),
                                             Kokkos::AUTO());

  using member_type = Kokkos::TeamPolicy<ExecSpace>::member_type;
  typedef ExecSpace::scratch_memory_space ScratchSpace;
  typedef Kokkos::View<double *, ScratchSpace,
                       Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      shared_view_double;

  typedef Kokkos::View<Point *, ScratchSpace,
                       Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      shared_view_points;

  int scratch_size_team = shared_view_double::shmem_size(max_points_per_box) +
                          shared_view_points::shmem_size(max_points_per_box);

  policy_boxes.set_scratch_size(0, Kokkos::PerTeam(scratch_size_team));

  Kokkos::Timer time_vector, time_serial;
  time_vector.reset();
  time_serial.reset();

  if (cfg.algorithm == "fastv") {
    double start_vector = time_vector.seconds();
    Kokkos::parallel_for(
        "Benchmark [Team->Thread->Vector] [basis_eval_fast]", policy_boxes,
        KOKKOS_LAMBDA(const member_type &team_member) {
          const int box_idx = team_member.league_rank();
          // Compute number of points per box
          const int start_points = box_idx * max_points_per_box;
          const int end_points =
              Kokkos::min(start_points + max_points_per_box, N_quad);
          const int num_points = end_points - start_points;

          // Compute number of neighbors per box
          const int start_neighbors = nl.offsets(box_idx);
          const int end_neighbors = nl.offsets(box_idx + 1);
          const int num_neighbors = end_neighbors - start_neighbors;

          shared_view_double weights_scratch(team_member.team_scratch(0),
                                             num_points);

          shared_view_points points_scratch(team_member.team_scratch(0),
                                            num_points);

          Kokkos::parallel_for(Kokkos::TeamVectorRange(team_member, num_points),
                               [=](const int local_g) {
                                 const int global_g = start_points + local_g;
                                 weights_scratch(local_g) =
                                     grid.weights(global_g);
                                 points_scratch(local_g) =
                                     grid.quad_points(global_g);
                               });

          team_member.team_barrier();

          Kokkos::parallel_for(
              Kokkos::TeamThreadMDRange(team_member, num_neighbors,
                                        num_neighbors),
              [=](const int local_i, const int local_j) {
                const int global_i = nl.neighbors(start_neighbors + local_i);
                const int global_j = nl.neighbors(start_neighbors + local_j);

                const ShellParams shell_i = load_shell(basis, global_i);
                const ShellParams shell_j = load_shell(basis, global_j);

                double sum = 0;
                Kokkos::parallel_reduce(
                    Kokkos::ThreadVectorRange(team_member, num_points),
                    [=](const int local_g, double &local_sum) {
                      local_sum +=
                          basis_eval_fast(shell_i, points_scratch(local_g)[0],
                                          points_scratch(local_g)[1],
                                          points_scratch(local_g)[2]) *
                          basis_eval_fast(shell_j, points_scratch(local_g)[0],
                                          points_scratch(local_g)[1],
                                          points_scratch(local_g)[2]);
                    },
                    sum);
                S(local_i, local_j) = sum;
              });
        });

    Kokkos::fence();
    double end_vector = time_vector.seconds();
    Kokkos::printf(
        "Benchmark [Team->Thread->Vector] [basis_eval_fast] took %fs \n",
        end_vector - start_vector);
  } else if (cfg.algorithm == "fasts") {

    double start_serial = time_serial.seconds();
    Kokkos::parallel_for(
        "Benchmark [Team->Thread->Serial] [basis_eval_fast]", policy_boxes,
        KOKKOS_LAMBDA(const member_type &team_member) {
          const int box_idx = team_member.league_rank();
          // Compute number of points per box
          const int start_points = box_idx * max_points_per_box;
          const int end_points =
              Kokkos::min(start_points + max_points_per_box, N_quad);
          const int num_points = end_points - start_points;

          // Compute number of neighbors per box
          const int start_neighbors = nl.offsets(box_idx);
          const int end_neighbors = nl.offsets(box_idx + 1);
          const int num_neighbors = end_neighbors - start_neighbors;

          shared_view_double weights_scratch(team_member.team_scratch(0),
                                             num_points);

          shared_view_points points_scratch(team_member.team_scratch(0),
                                            num_points);

          Kokkos::parallel_for(Kokkos::TeamVectorRange(team_member, num_points),
                               [=](const int local_g) {
                                 const int global_g = start_points + local_g;
                                 weights_scratch(local_g) =
                                     grid.weights(global_g);
                                 points_scratch(local_g) =
                                     grid.quad_points(global_g);
                               });

          team_member.team_barrier();

          Kokkos::parallel_for(
              Kokkos::TeamThreadMDRange(team_member, num_neighbors,
                                        num_neighbors),
              [=](const int local_i, const int local_j) {
                const int global_i = nl.neighbors(start_neighbors + local_i);
                const int global_j = nl.neighbors(start_neighbors + local_j);

                const ShellParams shell_i = load_shell(basis, global_i);
                const ShellParams shell_j = load_shell(basis, global_j);

                double sum = 0;
                for (int local_g = 0; local_g < num_points; ++local_g) {
                  sum += basis_eval_fast(shell_i, points_scratch(local_g)[0],
                                         points_scratch(local_g)[1],
                                         points_scratch(local_g)[2]) *
                         basis_eval_fast(shell_j, points_scratch(local_g)[0],
                                         points_scratch(local_g)[1],
                                         points_scratch(local_g)[2]);
                }

                S(local_i, local_j) = sum;
              });
        });
    Kokkos::fence();

    double end_serial = time_serial.seconds();
    Kokkos::printf(
        "Benchmark [Team->Thread->Serial] [basis_eval_fast] took %fs \n",
        end_serial - start_serial);

  } else if (cfg.algorithm == "slowv") {
    double start_vector_slow = time_vector.seconds();
    Kokkos::parallel_for(
        "Benchmark [Team->Thread->Vector] [basis_eval]", policy_boxes,
        KOKKOS_LAMBDA(const member_type &team_member) {
          const int box_idx = team_member.league_rank();
          // Compute number of points per box
          const int start_points = box_idx * max_points_per_box;
          const int end_points =
              Kokkos::min(start_points + max_points_per_box, N_quad);
          const int num_points = end_points - start_points;

          // Compute number of neighbors per box
          const int start_neighbors = nl.offsets(box_idx);
          const int end_neighbors = nl.offsets(box_idx + 1);
          const int num_neighbors = end_neighbors - start_neighbors;

          shared_view_double weights_scratch(team_member.team_scratch(0),
                                             num_points);

          shared_view_points points_scratch(team_member.team_scratch(0),
                                            num_points);

          Kokkos::parallel_for(Kokkos::TeamVectorRange(team_member, num_points),
                               [=](const int local_g) {
                                 const int global_g = start_points + local_g;
                                 weights_scratch(local_g) =
                                     grid.weights(global_g);
                                 points_scratch(local_g) =
                                     grid.quad_points(global_g);
                               });

          team_member.team_barrier();

          Kokkos::parallel_for(
              Kokkos::TeamThreadMDRange(team_member, num_neighbors,
                                        num_neighbors),
              [=](const int local_i, const int local_j) {
                const int global_i = nl.neighbors(start_neighbors + local_i);
                const int global_j = nl.neighbors(start_neighbors + local_j);

                double sum = 0;
                Kokkos::parallel_reduce(
                    Kokkos::ThreadVectorRange(team_member, num_points),
                    [=](const int local_g, double &local_sum) {
                      double phi_i;
                      double phi_j;

                      basis_eval(basis, global_i, points_scratch(local_g)[0],
                                 points_scratch(local_g)[1],
                                 points_scratch(local_g)[2], phi_i);

                      basis_eval(basis, global_j, points_scratch(local_g)[0],
                                 points_scratch(local_g)[1],
                                 points_scratch(local_g)[2], phi_j);
                      local_sum += phi_i * phi_j;
                    },
                    sum);

                S(local_i, local_j) = sum;
              });
        });

    Kokkos::fence();
    double end_vector_slow = time_vector.seconds();
    Kokkos::printf("Benchmark [Team->Thread->Vector] [basis_eval] took %fs \n",
                   end_vector_slow - start_vector_slow);
  } else if (cfg.algorithm == "slows") {
    double start_serial_slow = time_serial.seconds();
    Kokkos::parallel_for(
        "Benchmark [Team->Thread->Serial] [basis_eval]", policy_boxes,
        KOKKOS_LAMBDA(const member_type &team_member) {
          const int box_idx = team_member.league_rank();
          // Compute number of points per box
          const int start_points = box_idx * max_points_per_box;
          const int end_points =
              Kokkos::min(start_points + max_points_per_box, N_quad);
          const int num_points = end_points - start_points;

          // Compute number of neighbors per box
          const int start_neighbors = nl.offsets(box_idx);
          const int end_neighbors = nl.offsets(box_idx + 1);
          const int num_neighbors = end_neighbors - start_neighbors;

          shared_view_double weights_scratch(team_member.team_scratch(0),
                                             num_points);

          shared_view_points points_scratch(team_member.team_scratch(0),
                                            num_points);

          Kokkos::parallel_for(Kokkos::TeamVectorRange(team_member, num_points),
                               [=](const int local_g) {
                                 const int global_g = start_points + local_g;
                                 weights_scratch(local_g) =
                                     grid.weights(global_g);
                                 points_scratch(local_g) =
                                     grid.quad_points(global_g);
                               });

          team_member.team_barrier();

          Kokkos::parallel_for(
              Kokkos::TeamThreadMDRange(team_member, num_neighbors,
                                        num_neighbors),
              [=](const int local_i, const int local_j) {
                const int global_i = nl.neighbors(start_neighbors + local_i);
                const int global_j = nl.neighbors(start_neighbors + local_j);

                double sum = 0;
                for (int local_g = 0; local_g < num_points; ++local_g) {
                  double phi_i;
                  double phi_j;

                  basis_eval(basis, global_i, points_scratch(local_g)[0],
                             points_scratch(local_g)[1],
                             points_scratch(local_g)[2], phi_i);

                  basis_eval(basis, global_j, points_scratch(local_g)[0],
                             points_scratch(local_g)[1],
                             points_scratch(local_g)[2], phi_j);
                  sum += phi_i * phi_j;
                }
                S(local_i, local_j) = sum;
              });
        });

    Kokkos::fence();
    double end_serial_slow = time_serial.seconds();
    Kokkos::printf("Benchmark [Team->Thread->Serial] [basis_eval] took %fs \n",
                   end_serial_slow - start_serial_slow);
  }
};

int main(int argc, char *argv[]) {

  Config cfg;
  try {
    cfg = parse_args(argc, argv);
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }

  Kokkos::initialize();
  {
    if (cfg.benchmark) {
      print_config(cfg);
      benchmark_collocation(cfg);
    } else {
      int result = Catch::Session().run();
    }
  }
  Kokkos::finalize();
  return 0;
}
