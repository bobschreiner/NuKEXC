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
#include <catch2/matchers/catch_matchers.hpp>

#include <integratorxx/composite_quadratures/spherical_quadrature.hpp>
#include <integratorxx/generators/radial_factory.hpp>
#include <integratorxx/generators/spherical_factory.hpp>
#include <integratorxx/quadratures/radial.hpp>
#include <integratorxx/quadratures/s2.hpp>

#include <nukexc/diagonaliser.hpp>
#include <nukexc/grid.hpp>
#include <nukexc/integration.hpp>
#include <nukexc/molecule.hpp>
#include <nukexc/partitioning.hpp>
#include <nukexc/stobasis.hpp>

#include <cmath>
#include <vector>

using namespace NuKEXC;

using bk_type = IntegratorXX::Becke<double, double>;
using ll_type = IntegratorXX::LebedevLaikov<double>;

// ============================================================
//  Helper: matrix symmetry check
// ============================================================

template <typename MirrorView>
void require_symmetric(const MirrorView &M, double tol = 1e-8) {
  for (int i = 0; i < (int)M.extent(0); ++i)
    for (int j = 0; j < (int)M.extent(1); ++j)
      REQUIRE_THAT(M(i, j), Catch::Matchers::WithinAbs(M(j, i), tol));
}

// ============================================================
//  TEST 1 — Hydrogen 1s (single STO, exact hydrogenic values)
// ============================================================
//
// For a normalized 1s STO with n=1, l=0, zeta=1:
//   S = 1,   T = zeta^2/2 = 0.5,   V = -Z*zeta = -1.0
// These are exact for the true hydrogen ground state.
// Virial theorem: 2T + V = 0 (equivalently E = T + V = -0.5)

TEST_CASE("hydrogen 1s -- normalization, eigenvalues, virial",
          "[hydrogen_1s]") {

  Molecule mol(std::vector<std::vector<double>>{{0., 0., 0.}},
               std::vector<unsigned>{1u});
  auto grid = make_flat_grid<bk_type, ll_type>(mol);
  STOBasisSet basis = load_thakkar_basis(mol);

  auto S = overlap_integral(basis, grid.quad_points, grid.weights);
  auto T = kinetic_integral(basis, grid.quad_points, grid.weights);
  auto V = nuclear_potential_integral(basis, grid.quad_points, grid.weights,
                                      grid.atom_centers, grid.Z);

  auto S_h = Kokkos::create_mirror_view(S);
  auto T_h = Kokkos::create_mirror_view(T);
  auto V_h = Kokkos::create_mirror_view(V);
  Kokkos::deep_copy(S_h, S);
  Kokkos::deep_copy(T_h, T);
  Kokkos::deep_copy(V_h, V);

  require_symmetric(S_h);
  require_symmetric(T_h);
  require_symmetric(V_h);

  REQUIRE_THAT(S_h(0, 0), Catch::Matchers::WithinRel(1.0, 1e-7));
  REQUIRE_THAT(T_h(0, 0), Catch::Matchers::WithinRel(0.5, 1e-7));
  REQUIRE_THAT(V_h(0, 0), Catch::Matchers::WithinRel(-1.0, 1e-7));

  // Virial theorem: 2T + V = 0
  REQUIRE_THAT(2.0 * T_h(0, 0) + V_h(0, 0),
               Catch::Matchers::WithinAbs(0.0, 1e-7));
  // Total energy
  REQUIRE_THAT(T_h(0, 0) + V_h(0, 0), Catch::Matchers::WithinRel(-0.5, 1e-7));
}

// ============================================================
//  TEST 2 — Single-center 1s + 2p (multi-shell)
// ============================================================
//
// Basis: exact hydrogenic 1s (n=1, l=0, zeta=1.0)
//      + all three 2p (n=2, l=1, zeta=0.5)  at the origin.
//
//  Exact values for 2p (n=2, l=1, zeta=0.5, Z=1) derived analytically:
//  ------------------------------------------------------------------
//  N^2 = (2*zeta)^{2n+1} / (2n)! = 1^5 / 24 = 1/24
//
//  Kinetic energy (integrate term by term):
//    T = N^2/2 * [-(n(n-1)-l(l+1)) * I(2n-2, 2z)
//                 + 2*zeta*n * I(2n-1, 2z) - zeta^2 * I(2n, 2z)]
//    For n=2, l=1, zeta=0.5:  all I evaluated at alpha=1 -> I(k,1)=k!
//    Term 1: -(2 - 2) * 2! = 0
//    Term 2: 2*0.5*2 * 3! = 12
//    Term 3: 0.25 * 4!    =  6
//    T = (1/24)/2 * (12 - 6) = 1/8
//
//  Nuclear potential (V = -Z * <1/r>):
//    <1/r> = N^2 * (2n-1)! / (2*zeta)^{2n} = (1/24)*6/1 = 1/4
//    V = -1/4
//
//  Orthogonality:
//    1s ⊥ 2p   -- exact by angular symmetry (different l), should hold
//                 to near machine precision rather than just quadrature noise.
//    2p_m ⊥ 2p_{m'} -- exact by angular symmetry (different m).
//
//  m-degeneracy:
//    T and V diagonal elements are identical for m = -1, 0, +1.

TEST_CASE("single-center 1s + 2p -- orthogonality, degeneracy, exact values",
          "[multi_shell]") {

  // Index map:  0 -> 1s,  1 -> 2p_{m=-1},  2 -> 2p_{m=0},  3 -> 2p_{m=+1}
  STOBasisSet basis = make_manual_basis({
      {1, 0, 0, 1.0, 0., 0., 0.},  // 1s
      {2, 1, -1, 0.5, 0., 0., 0.}, // 2p_{-1}
      {2, 1, 0, 0.5, 0., 0., 0.},  // 2p_0
      {2, 1, +1, 0.5, 0., 0., 0.}, // 2p_{+1}
  });

  Molecule mol(std::vector<std::vector<double>>{{0., 0., 0.}},
               std::vector<unsigned>{1u});
  auto grid = make_flat_grid<bk_type, ll_type>(mol);

  auto S = overlap_integral(basis, grid.quad_points, grid.weights);
  auto T = kinetic_integral(basis, grid.quad_points, grid.weights);
  auto V = nuclear_potential_integral(basis, grid.quad_points, grid.weights,
                                      grid.atom_centers, grid.Z);

  auto S_h = Kokkos::create_mirror_view(S);
  auto T_h = Kokkos::create_mirror_view(T);
  auto V_h = Kokkos::create_mirror_view(V);
  Kokkos::deep_copy(S_h, S);
  Kokkos::deep_copy(T_h, T);
  Kokkos::deep_copy(V_h, V);

  // All matrices must be symmetric
  require_symmetric(S_h);
  require_symmetric(T_h);
  require_symmetric(V_h);

  // ---- Normalization ----
  REQUIRE_THAT(S_h(0, 0), Catch::Matchers::WithinRel(1.0, 1e-7)); // 1s
  REQUIRE_THAT(S_h(1, 1), Catch::Matchers::WithinRel(1.0, 1e-7)); // 2p_{-1}
  REQUIRE_THAT(S_h(2, 2), Catch::Matchers::WithinRel(1.0, 1e-7)); // 2p_0
  REQUIRE_THAT(S_h(3, 3), Catch::Matchers::WithinRel(1.0, 1e-7)); // 2p_{+1}

  // ---- Angular orthogonality: 1s ⊥ all 2p ----
  // This is exact by symmetry; use a tight absolute tolerance.
  REQUIRE_THAT(S_h(0, 1), Catch::Matchers::WithinAbs(0.0, 1e-10));
  REQUIRE_THAT(S_h(0, 2), Catch::Matchers::WithinAbs(0.0, 1e-10));
  REQUIRE_THAT(S_h(0, 3), Catch::Matchers::WithinAbs(0.0, 1e-10));

  // ---- Angular orthogonality: 2p components mutually orthogonal ----
  REQUIRE_THAT(S_h(1, 2), Catch::Matchers::WithinAbs(0.0, 1e-10));
  REQUIRE_THAT(S_h(1, 3), Catch::Matchers::WithinAbs(0.0, 1e-10));
  REQUIRE_THAT(S_h(2, 3), Catch::Matchers::WithinAbs(0.0, 1e-10));

  // ---- Kinetic energy: exact analytical values ----
  REQUIRE_THAT(T_h(0, 0), Catch::Matchers::WithinRel(0.5, 1e-7));   // 1s
  REQUIRE_THAT(T_h(1, 1), Catch::Matchers::WithinRel(0.125, 1e-7)); // 2p_{-1}
  REQUIRE_THAT(T_h(2, 2), Catch::Matchers::WithinRel(0.125, 1e-7)); // 2p_0
  REQUIRE_THAT(T_h(3, 3), Catch::Matchers::WithinRel(0.125, 1e-7)); // 2p_{+1}

  // ---- Nuclear potential energy: exact analytical values ----
  REQUIRE_THAT(V_h(0, 0), Catch::Matchers::WithinRel(-1.0, 1e-7));  // 1s
  REQUIRE_THAT(V_h(1, 1), Catch::Matchers::WithinRel(-0.25, 1e-7)); // 2p_{-1}
  REQUIRE_THAT(V_h(2, 2), Catch::Matchers::WithinRel(-0.25, 1e-7)); // 2p_0
  REQUIRE_THAT(V_h(3, 3), Catch::Matchers::WithinRel(-0.25, 1e-7)); // 2p_{+1}

  // ---- m-degeneracy: all 2p states must give identical diagonal T and V ----
  // Using a tighter relative tolerance here than for the absolute values,
  // because residual asymmetry diagnoses a grid-symmetry or sign error.
  REQUIRE_THAT(T_h(1, 1), Catch::Matchers::WithinRel(T_h(2, 2), 1e-10));
  REQUIRE_THAT(T_h(1, 1), Catch::Matchers::WithinRel(T_h(3, 3), 1e-10));
  REQUIRE_THAT(V_h(1, 1), Catch::Matchers::WithinRel(V_h(2, 2), 1e-10));
  REQUIRE_THAT(V_h(1, 1), Catch::Matchers::WithinRel(V_h(3, 3), 1e-10));

  // ---- Virial theorem: 2T + V = 0 for both shells ----
  REQUIRE_THAT(2.0 * T_h(0, 0) + V_h(0, 0),
               Catch::Matchers::WithinAbs(0.0, 1e-7)); // 1s
  REQUIRE_THAT(2.0 * T_h(1, 1) + V_h(1, 1),
               Catch::Matchers::WithinAbs(0.0, 1e-7)); // 2p

  // ---- Off-diagonal T and V blocks are zero by angular symmetry ----
  // Different-l blocks (1s/2p) and different-m blocks within 2p
  for (int i = 1; i <= 3; ++i) {
    REQUIRE_THAT(T_h(0, i), Catch::Matchers::WithinAbs(0.0, 1e-8));
    REQUIRE_THAT(V_h(0, i), Catch::Matchers::WithinAbs(0.0, 1e-8));
  }
  for (int i = 1; i <= 3; ++i)
    for (int j = i + 1; j <= 3; ++j) {
      REQUIRE_THAT(T_h(i, j), Catch::Matchers::WithinAbs(0.0, 1e-8));
      REQUIRE_THAT(V_h(i, j), Catch::Matchers::WithinAbs(0.0, 1e-8));
    }

  // ---- Diagonalization Test ----
  int n_basis = 4;
  HostView2DLeft H_h("mo_coeffs", n_basis, n_basis);
  HostView2DLeft mo_coeffs_h("mo_coeffs", n_basis, n_basis);
  HostView1D mo_energies_h("mo_energies", n_basis);

  for (int i = 0; i < n_basis; ++i) {
    for (int j = 0; j < n_basis; ++j) {
      H_h(i, j) = T_h(i, j) + V_h(i, j);
    }
  }

  NuKEXC::Diagonalizer diagonalizer(n_basis);
  diagonalizer.compute_transformation(S_h);
  diagonalizer.solve(H_h, mo_coeffs_h, mo_energies_h);

  // Sort if necessary, though for H they should naturally fall into -0.5 and
  // -0.125 We expect one -0.5 (1s) and three -0.125 (2p)
  double e_1s = -0.5;
  double e_2p = -0.125;

  // Check 1s energy (usually the lowest)
  REQUIRE_THAT(mo_energies_h(0), Catch::Matchers::WithinRel(e_1s, 1e-7));

  // Check 2p degeneracy and values
  for (int i = 1; i < 4; ++i) {
    REQUIRE_THAT(mo_energies_h(i), Catch::Matchers::WithinRel(e_2p, 1e-7));
  }

  // 5. Verify Orthonormality of MO Coefficients: C^T * S * C = I
  for (int i = 0; i < n_basis; ++i) {
    for (int j = 0; j < n_basis; ++j) {
      double orthogonality_sum = 0.0;
      for (int a = 0; a < n_basis; ++a) {
        for (int b = 0; b < n_basis; ++b) {
          orthogonality_sum +=
              mo_coeffs_h(a, i) * S_h(a, b) * mo_coeffs_h(b, j);
        }
      }
      double expected = (i == j) ? 1.0 : 0.0;
      REQUIRE_THAT(orthogonality_sum,
                   Catch::Matchers::WithinAbs(expected, 1e-8));
    }
  }
}

// ============================================================
//  TEST 3 — H2+ overlap matrix
// ============================================================
//
// Two H atoms 1 bohr apart along x.  Basis: one 1s STO (zeta=1) per atom.
//
// The off-diagonal overlap integral is known exactly for unnormalized STOs
// and reduces to:
//
//   S_AB(zeta=1, R) = e^{-R} (1 + R + R^2/3)
//
// At R = 1 bohr:  S_AB = e^{-1} * 7/3 ≈ 0.85836...
//
// All matrices must be symmetric, and the on-diagonal elements are 1
// (each function is normalized by construction).
// T_AA = T_BB = 0.5 still holds for the single-center kinetic integrals.

TEST_CASE("H2+ overlap matrix -- symmetry and analytical off-diagonal",
          "[h2_plus]") {

  const double R = 1.0; // bond length in bohr

  Molecule mol(std::vector<std::vector<double>>{{0., 0., 0.}, {R, 0., 0.}},
               std::vector<unsigned>{1u, 1u});

  auto grid = make_flat_grid<bk_type, ll_type>(mol);

  // Build the basis explicitly so the zeta value is unambiguous.
  STOBasisSet basis = make_manual_basis({
      {1, 0, 0, 1.0, 0., 0., 0.}, // 1s on atom A
      {1, 0, 0, 1.0, R, 0., 0.},  // 1s on atom B
  });

  auto S = overlap_integral(basis, grid.quad_points, grid.weights);
  auto T = kinetic_integral(basis, grid.quad_points, grid.weights);
  auto V = nuclear_potential_integral(basis, grid.quad_points, grid.weights,
                                      grid.atom_centers, grid.Z);

  auto S_h = Kokkos::create_mirror_view(S);
  auto T_h = Kokkos::create_mirror_view(T);
  auto V_h = Kokkos::create_mirror_view(V);
  Kokkos::deep_copy(S_h, S);
  Kokkos::deep_copy(T_h, T);
  Kokkos::deep_copy(V_h, V);

  require_symmetric(S_h);
  require_symmetric(T_h);
  require_symmetric(V_h);

  // ---- Normalization ----
  REQUIRE_THAT(S_h(0, 0), Catch::Matchers::WithinRel(1.0, 1e-7));
  REQUIRE_THAT(S_h(1, 1), Catch::Matchers::WithinRel(1.0, 1e-7));

  // ---- Off-diagonal overlap: exact formula S_AB = e^{-R}(1 + R + R^2/3) ----
  const double S_exact = std::exp(-R) * (1.0 + R + R * R / 3.0);
  REQUIRE_THAT(S_h(0, 1), Catch::Matchers::WithinRel(S_exact, 1e-6));

  // ---- Single-center kinetic energy is unchanged by the second atom ----
  REQUIRE_THAT(T_h(0, 0), Catch::Matchers::WithinRel(0.5, 1e-7));
  REQUIRE_THAT(T_h(1, 1), Catch::Matchers::WithinRel(0.5, 1e-7));

  // ---- V has no closed form for the cross-nuclear terms, but the two  ----
  // ---- on-diagonal elements must be equal by the symmetry of the system ----
  REQUIRE_THAT(V_h(0, 0), Catch::Matchers::WithinRel(V_h(1, 1), 1e-6));
}

// ============================================================
//  TEST 4 — H2+ Energies
// ============================================================
//
// Two H atoms R bohr apart along x.  Basis: one 1s STO (zeta=1) per atom.
//
// The off-diagonal overlap integral is known exactly for unnormalized STOs
// and reduces to:
//
//   S_AB(zeta=1, R) = e^{-R} (1 + R + R^2/3)
//
// At R = 1 bohr:  S_AB = e^{-1} * 7/3 ≈ 0.85836...
//
// All matrices must be symmetric, and the on-diagonal elements are 1
// (each function is normalized by construction).
// T_AA = T_BB = 0.5 still holds for the single-center kinetic integrals.

TEST_CASE("H2+ Energies", "[h2_plus][energies]") {

  const double R = 1.0; // bond length in bohr

  Molecule mol(std::vector<std::vector<double>>{{0., 0., 0.}, {R, 0., 0.}},
               std::vector<unsigned>{1u, 1u});

  auto grid = make_flat_grid<bk_type, ll_type>(mol, 100, 50);

  // Build the basis explicitly so the zeta value is unambiguous.
  STOBasisSet basis = load_adf_basis(mol, "input/zorabasis/QZ4P");

  auto S = overlap_integral(basis, grid.quad_points, grid.weights);
  auto T = kinetic_integral(basis, grid.quad_points, grid.weights);
  auto V = nuclear_potential_integral(basis, grid.quad_points, grid.weights,
                                      grid.atom_centers, grid.Z);

  auto S_h = Kokkos::create_mirror_view(S);
  auto T_h = Kokkos::create_mirror_view(T);
  auto V_h = Kokkos::create_mirror_view(V);
  Kokkos::deep_copy(S_h, S);
  Kokkos::deep_copy(T_h, T);
  Kokkos::deep_copy(V_h, V);

  require_symmetric(S_h);
  require_symmetric(T_h);
  require_symmetric(V_h);

  // ---- Diagonalization Test ----
  int n_basis = basis.nbf();

  // 1. Prepare Batched Views on Host

  HostView2DLeft H_h("mo_coeffs", n_basis, n_basis);
  HostView2DLeft mo_coeffs_h("mo_coeffs", n_basis, n_basis);
  HostView1D mo_energies_h("mo_energies", n_basis);

  for (int i = 0; i < n_basis; ++i) {
    for (int j = 0; j < n_basis; ++j) {
      H_h(i, j) = T_h(i, j) + V_h(i, j);
    }
  }

  NuKEXC::Diagonalizer diagonalizer(n_basis);
  diagonalizer.compute_transformation(S_h);
  diagonalizer.solve(H_h, mo_coeffs_h, mo_energies_h);

  for (int i = 0; i < n_basis; ++i) {
    for (int j = 0; j < n_basis; ++j) {
      double orthogonality_sum = 0.0;
      for (int a = 0; a < n_basis; ++a) {
        for (int b = 0; b < n_basis; ++b) {
          orthogonality_sum +=
              mo_coeffs_h(a, i) * S_h(a, b) * mo_coeffs_h(b, j);
        }
      }
      double expected = (i == j) ? 1.0 : 0.0;
      // Use Abs tolerance because off-diagonals should be near zero
      REQUIRE_THAT(orthogonality_sum,
                   Catch::Matchers::WithinAbs(expected, 1e-8));
    }
  }

  // 6. Verify Energy Ordering (Ascending)
  for (int i = 0; i < n_basis - 1; ++i) {
    CHECK(mo_energies_h(i) <= mo_energies_h(i + 1));
  }

  // 7. Verify the Ground State Energy (sigma_g)
  // For H2+ at R=1.0 bohr, the exact electronic energy is roughly -1.45 au
  // Depending on your basis set quality, we check if it's in the ballpark.
  double e_ground = mo_energies_h(0);
  REQUIRE(e_ground < 0.0); // Must be bound

  // Optional: print out the spectrum for debugging
  std::cout << "H2+ Spectrum (R=" << R << ")" << std::endl;
  for (int i = 0; i < n_basis; ++i)
    std::cout << mo_energies_h(i) << std::endl;
  std::cout << std::endl;
}

// ============================================================
int main() {
  Kokkos::initialize();
  int result = Catch::Session().run();
  Kokkos::finalize();
  return result;
}
