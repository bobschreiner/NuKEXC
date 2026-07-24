/*
 *    NuKEXC -- Numerical Kokkos Enhanced Exchange Correlation Integrator
 *    Copyright (C) 2026 Bob Schreiner
 *
 *    This program is free software: you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation, either version 3 of the License, or
 *    (at your option) any later version.
 */

// Minimal, reusable *dense* unrestricted Hartree-Fock SCF driver. It is the
// dense HF path of standalone.cxx, stripped of CLI parsing, DFT, sparse/tiled
// options and timing, exposed as a single function so convergence studies can
// obtain a converged SCF total energy for an arbitrary grid. Everything is
// pre-computed per call (collocations, (A|B) half-inverse, core Hamiltonian),
// which is exactly what a grid-convergence sweep wants: each grid is
// independent.

#pragma once

#include <Kokkos_Core.hpp>
#include <armadillo>

#include <nukexc/core_hamiltonian.hpp>
#include <nukexc/coulomb.hpp>
#include <nukexc/diagonalizer.hpp>
#include <nukexc/exact_exchange.hpp>
#include <nukexc/grid.hpp>
#include <nukexc/molecule.hpp>
#include <nukexc/nuclear_repulsion.hpp>
#include <nukexc/nukexc_config.hpp>
#include <nukexc/nukexc_utils.hpp>
#include <nukexc/stobasis.hpp>
#include <nukexc/stopotential.hpp>

#include <openorbitaloptimizer/scfsolver.hpp>

#include "standalone_helpers.hpp" // kokkos_to_arma, arma_to_kokkos(1d)

#include <string>
#include <utility>
#include <vector>

namespace Nukexc {

// Converged UHF energy decomposition, split into the individual grid-sensitive
// terms (all in Ha):
//   kinetic            = Tr[D T]
//   nuclear_attraction = Tr[D V_ne]
//   coulomb            = 1/2 Tr[D J]
//   exchange           = -1/2 (Tr[D_a K_a] + Tr[D_b K_b])
//   nuclear_repulsion  = E_nuc                       (grid-independent)
//   total              = sum of the above
// one_electron() = kinetic + nuclear_attraction; two_electron() = coulomb +
// exchange. Reporting the terms separately shows which one converges slowest
// and which cancels non-monotonely in the total as the grid is refined.
struct ScfEnergies {
  double total = 0.0;
  double kinetic = 0.0;
  double nuclear_attraction = 0.0;
  double coulomb = 0.0;
  double exchange = 0.0;
  double nuclear_repulsion = 0.0;
  double one_electron() const { return kinetic + nuclear_attraction; }
  double two_electron() const { return coulomb + exchange; }
};

// Run a dense UHF SCF on `grid` and return its converged energy decomposition.
// `basis` is the primary basis, `basis_aux` the RI fitting basis. Closed- or
// open-shell is selected via `charge`/`multiplicity` (water: 0 / 1).
inline ScfEnergies run_uhf_scf_energy(ExecSpace &space, const Molecule &mol,
                                 const STOBasisSet &basis,
                                 const STOBasisSet &basis_aux,
                                 const FlatGrid &grid, int charge = 0,
                                 int multiplicity = 1, double conv_thr = 1e-9,
                                 double lin_dep_threshold = 1e-5) {
  const int N_bf = basis.nbf();
  const int N_bf_aux = basis_aux.nbf();
  const int N_quad = grid.quad_points.extent(0);

  // ---- Dense collocations: Phi, Phi_aux, and the (weight-scaled) potential --
  DeviceView2DLeft basis_collocation("Basis collocation", N_bf, N_quad);
  fill_collocation(space, basis, grid.quad_points, basis_collocation);

  DeviceView2DLeft basis_aux_collocation("Aux basis collocation", N_bf_aux,
                                         N_quad);
  fill_collocation(space, basis_aux, grid.quad_points, basis_aux_collocation);

  DeviceView2DLeft potential_collocation_scaled("Potential collocation",
                                                N_bf_aux, N_quad);
  sto_potential_collocation_scaled(space, basis_aux, grid,
                                   potential_collocation_scaled);

  // ---- Auxiliary Coulomb metric (A|B) and its symmetric half-inverse -------
  DeviceView2DLeft aux_overlap("Aux overlap", N_bf_aux, N_bf_aux);
  KokkosBlas::gemm(space, "N", "T", 1.0, basis_aux_collocation,
                   potential_collocation_scaled, 0.0, aux_overlap);
  DeviceView2DLeft aux_overlap_sym("Aux overlap sym", N_bf_aux, N_bf_aux);
  Kokkos::parallel_for(
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {N_bf_aux, N_bf_aux}),
      KOKKOS_LAMBDA(int i, int j) {
        aux_overlap_sym(i, j) = 0.5 * (aux_overlap(i, j) + aux_overlap(j, i));
      });
  DeviceView2DLeft half_inverse_X =
      compute_half_inverse(aux_overlap_sym, lin_dep_threshold);

  // ---- Core Hamiltonian, overlap and orthogonaliser X = S^{-1/2} -----------
  auto hcore = compute_core_hamiltonian(basis, grid);
  Diagonalizer diag(N_bf);
  DeviceView2DLeft X =
      diag.compute_transformation(hcore.overlap, lin_dep_threshold);

  arma::mat h_core = kokkos_to_arma(hcore.hamiltonian);
  arma::mat T_arma = kokkos_to_arma(hcore.kinetic);  // kinetic
  arma::mat Vne_arma = kokkos_to_arma(hcore.nuclear); // nuclear attraction
  arma::mat X_arma = kokkos_to_arma(X);
  arma::mat S_arma = kokkos_to_arma(hcore.overlap);

  // ---- Electron counts / OOO block setup -----------------------------------
  const int n_elec = mol.Z_total - charge;
  const double n_alpha = (n_elec + (multiplicity - 1)) / 2.0;
  const double n_beta = (n_elec - (multiplicity - 1)) / 2.0;

  arma::uvec blocks_per_type = {1, 1};
  arma::vec max_occupations = {1.0, 1.0};
  arma::vec number_of_particles = {n_alpha, n_beta};
  std::vector<std::string> block_descriptions = {"alpha", "beta"};

  const double E_nuc = compute_nuclear_repulsion(mol);

  // ---- GWH initial guess ---------------------------------------------------
  arma::mat F_gwh(N_bf, N_bf, arma::fill::zeros);
  const double K_gwh = 1.75;
  for (int i = 0; i < N_bf; ++i)
    for (int j = 0; j < N_bf; ++j)
      F_gwh(i, j) =
          (i == j) ? h_core(i, i)
                   : 0.5 * K_gwh * (h_core(i, i) + h_core(j, j)) * S_arma(i, j);
  arma::mat F_gwh_orth = X_arma.t() * F_gwh * X_arma;

  // Filled by the last (converged) Fock build so the caller gets the energy
  // decomposition without a separate re-evaluation.
  ScfEnergies conv{};
  conv.nuclear_repulsion = E_nuc;

  // ---- Dense UHF Fock builder (called synchronously by solver.run()) -------
  auto fock_builder =
      [&, space](const OpenOrbitalOptimizer::DensityMatrix<double, double> &dm)
      -> std::pair<double, OpenOrbitalOptimizer::FockMatrix<double>> {
    const auto &orbitals = dm.first;
    const auto &occupations = dm.second;

    arma::mat C_alpha = X_arma * orbitals[0];
    arma::mat C_beta = X_arma * orbitals[1];
    arma::vec occ_alpha = occupations[0];
    arma::vec occ_beta = occupations[1];

    arma::mat D_alpha = C_alpha * arma::diagmat(occ_alpha) * C_alpha.t();
    arma::mat D_beta = C_beta * arma::diagmat(occ_beta) * C_beta.t();
    arma::mat D_tot = D_alpha + D_beta;

    arma::mat C_combined = arma::join_horiz(C_alpha, C_beta);
    arma::vec occ_combined = arma::join_vert(occ_alpha, occ_beta);
    DeviceView2DLeft k_C_tot = arma_to_kokkos(C_combined, "C_combined");
    DeviceView1D k_occ_tot = arma_to_kokkos1d(occ_combined, "occ_combined");
    DeviceView2DLeft k_C_alpha = arma_to_kokkos(C_alpha, "C_alpha");
    DeviceView2DLeft k_C_beta = arma_to_kokkos(C_beta, "C_beta");
    DeviceView1D k_occ_alpha = arma_to_kokkos1d(occ_alpha, "occ_alpha");
    DeviceView1D k_occ_beta = arma_to_kokkos1d(occ_beta, "occ_beta");

    // Coulomb (total density) and both exact-exchange spin channels.
    DeviceView2DLeft J = compute_coulomb(
        space, k_C_tot, k_occ_tot, basis_collocation, basis_aux_collocation,
        potential_collocation_scaled, half_inverse_X);
    arma::mat J_arma = kokkos_to_arma(J);

    DeviceView2DLeft K_alpha = compute_exact_exchange(
        space, k_C_alpha, k_occ_alpha, basis_collocation, basis_aux_collocation,
        potential_collocation_scaled, half_inverse_X);
    DeviceView2DLeft K_beta = compute_exact_exchange(
        space, k_C_beta, k_occ_beta, basis_collocation, basis_aux_collocation,
        potential_collocation_scaled, half_inverse_X);
    arma::mat K_alpha_arma = kokkos_to_arma(K_alpha);
    arma::mat K_beta_arma = kokkos_to_arma(K_beta);

    const double E_exchange = -0.5 * arma::trace(D_alpha * K_alpha_arma) -
                              0.5 * arma::trace(D_beta * K_beta_arma);
    const double E_kinetic = arma::trace(D_tot * T_arma);
    const double E_nuc_attr = arma::trace(D_tot * Vne_arma);
    const double E_core = E_kinetic + E_nuc_attr; // = Tr[D h_core]
    const double E_coulomb = 0.5 * arma::trace(D_tot * J_arma);
    const double Etot = E_nuc + E_core + E_coulomb + E_exchange;

    conv.total = Etot;
    conv.kinetic = E_kinetic;
    conv.nuclear_attraction = E_nuc_attr;
    conv.coulomb = E_coulomb;
    conv.exchange = E_exchange;

    arma::mat F_alpha = h_core + J_arma - K_alpha_arma;
    arma::mat F_beta = h_core + J_arma - K_beta_arma;
    std::vector<arma::mat> fock_arma = {X_arma.t() * F_alpha * X_arma,
                                        X_arma.t() * F_beta * X_arma};
    return std::make_pair(Etot, fock_arma);
  };

  // ---- Construct and run the SCF solver ------------------------------------
  OpenOrbitalOptimizer::SCFSolver<double, double> solver(
      blocks_per_type, max_occupations, number_of_particles, fock_builder,
      block_descriptions);
  solver.convergence_threshold(conv_thr);
  solver.verbosity(0);
  solver.initialize_with_fock({F_gwh_orth, F_gwh_orth});
  solver.run();

  return conv;
}

} // namespace Nukexc
