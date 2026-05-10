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

#pragma once

#include <cctype>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "atomic_properties.hpp"
#include "molecule.hpp"
#include "nukexc_config.hpp"
#include "nukexc_utils.hpp"
#include "spherical_harmonics.hpp"

namespace NuKEXC {

struct STOBasisSet {

  Kokkos::View<int *> n;
  Kokkos::View<int *> l;
  Kokkos::View<int *> m;
  Kokkos::View<double *> norm;
  Kokkos::View<double *> zeta;
  Kokkos::View<double *[3]> O;
  Kokkos::View<double *> cutoff_radii;

  size_t nbf() const { return O.extent(0); };
};

// Map shell labels to l values
int label_to_l(char label) {
  label = std::toupper(label);
  static const std::string order =
      "SPDFGHIK"; // Standard spectroscopic notation (skipping J)
  size_t pos = order.find(label);
  return (pos != std::string::npos) ? static_cast<int>(pos) : -1;
}

// ============================================================
//  Manual STOBasisSet
// ============================================================
//
// Constructs an STOBasisSet directly from a plain list rather than
// reading from disk.  This decouples integration tests from the
// basis-file format and makes the quantum numbers explicit.
//
// Each entry is { n, l, m, zeta, ox, oy, oz }.

struct STOFunc {
  int n, l, m;
  double zeta;
  double ox, oy, oz;
};

STOBasisSet make_manual_basis(const std::vector<STOFunc> &funcs,
                              double cutoff_tol = 1e-10) {
  const size_t nbf = funcs.size();

  STOBasisSet basis;
  basis.n = Kokkos::View<int *>("n", nbf);
  basis.l = Kokkos::View<int *>("l", nbf);
  basis.m = Kokkos::View<int *>("m", nbf);
  basis.zeta = Kokkos::View<double *>("zeta", nbf);
  basis.norm = Kokkos::View<double *>("coeff", nbf);
  basis.O = Kokkos::View<double *[3]>("centers", nbf);
  basis.cutoff_radii = Kokkos::View<double *>("cutoff radii", nbf);

  auto n_h = Kokkos::create_mirror_view(basis.n);
  auto l_h = Kokkos::create_mirror_view(basis.l);
  auto m_h = Kokkos::create_mirror_view(basis.m);
  auto zeta_h = Kokkos::create_mirror_view(basis.zeta);
  auto norm_h = Kokkos::create_mirror_view(basis.norm);
  auto O_h = Kokkos::create_mirror_view(basis.O);
  auto cutoff_radii_h = Kokkos::create_mirror_view(basis.cutoff_radii);
  for (size_t i = 0; i < nbf; ++i) {
    const auto &f = funcs[i];
    // Matches the normalization used by load_sto_basis:
    //   N = (2*zeta)^{n+0.5} / sqrt((2n)!)
    const double norm = std::pow(2.0 * f.zeta, f.n + 0.5) /
                        std::sqrt(static_cast<double>(factorial(2 * f.n)));
    const double cutoff_guess = (std::log(norm / cutoff_tol) / f.zeta) * 1.2;
    double poly_factor = std::pow(cutoff_guess, f.n - 1);
    const double cutoff = std::log((norm * poly_factor) / cutoff_tol) / f.zeta;

    n_h(i) = f.n;
    l_h(i) = f.l;
    m_h(i) = f.m;
    zeta_h(i) = f.zeta;
    norm_h(i) = norm;
    O_h(i, 0) = f.ox;
    O_h(i, 1) = f.oy;
    O_h(i, 2) = f.oz;
    cutoff_radii_h(i) = cutoff;
  }

  Kokkos::deep_copy(basis.n, n_h);
  Kokkos::deep_copy(basis.l, l_h);
  Kokkos::deep_copy(basis.m, m_h);
  Kokkos::deep_copy(basis.zeta, zeta_h);
  Kokkos::deep_copy(basis.norm, norm_h);
  Kokkos::deep_copy(basis.O, O_h);

  return basis;
}

STOBasisSet load_adf_basis(const Molecule &mol,
                           const std::string &data_dir = "input/zorabasis/TZP",
                           double cutoff_tol = 1e-10) {

  std::vector<STOFunc> temp_basis;

  for (size_t i = 0; i < mol.natoms; ++i) {
    std::string element_symbol = detail::symbols[mol.Z(i)];
    element_symbol[0] = std::toupper(element_symbol[0]);

    std::string filename = data_dir + "/" + element_symbol;

    std::ifstream file(filename);
    if (!file.is_open())
      continue;

    std::string line;
    int current_l = 0;

    // Skip all lines until we find the BASIS keyword
    while (std::getline(file, line)) {
      if (line == "BASIS")
        break;
    }

    // Read line by line until we find END keyword
    while (std::getline(file, line)) {
      if (line == "END") {
        break;

        // Skip the line if it is empty
      } else if (line.empty()) {
        continue;

      } else {

        // momentum labels This handles S, P, D, F, G, H, I, K (l=0 to 7)

        //  Parse Basis rows: Label (e.g. 4F), zeta
        std::stringstream ss(line);
        std::string label;
        double zeta;

        if (ss >> label >> zeta) {

          int n = std::stoi(label.substr(0, 1));
          int l = label_to_l(label[1]);

          // 4. Expand for each m component (-l to +l)
          // This accounts for the degeneracy of higher l states
          for (int m = -l; m <= l; ++m) {
            temp_basis.push_back({n, l, m, zeta, mol.atom_centers(i, 0),
                                  mol.atom_centers(i, 1),
                                  mol.atom_centers(i, 2)});
          }
        }
      }
    }
  }
  return make_manual_basis(temp_basis);
}

STOBasisSet
load_thakkar_basis(const Molecule &mol,
                   const std::string &data_dir = "input/k99light/neutral",
                   const double cutoff_tol = 1e-10) {

  std::vector<STOFunc> temp_basis;

  for (size_t i = 0; i < mol.natoms; ++i) {
    std::string element_symbol = detail::symbols[mol.Z(i)];
    element_symbol[0] = std::tolower(element_symbol[0]);

    std::string filename = data_dir + "/" + element_symbol;

    std::ifstream file(filename);
    if (!file.is_open())
      continue;

    std::string line;
    int current_l = 0;

    while (std::getline(file, line)) {
      if (line.empty())
        continue;

      // 1. Peek at the first word of the line
      std::stringstream check_ss(line);
      std::string first_word;
      check_ss >> first_word;

      // 2. If the first word is a single letter (S, P, D, etc.)
      if (first_word.length() == 1) {
        int found_l = label_to_l(first_word[0]);
        if (found_l != -1) {
          current_l = found_l;
          // Don't 'continue' here!
          // The line might also contain the first basis function.
        }
      }

      // Detect any shell block by checking for single-letter angular
      // momentum labels This handles S, P, D, F, G, H, I, K (l=0 to 7)

      // Skip headers/metadata
      if (line.find("BASIS") != std::string::npos ||
          line.find("CUSP") != std::string::npos)
        continue;

      //  Parse Basis rows: Label (e.g. 4F), zeta, then coefficients
      std::stringstream ss(line);
      std::string label;
      double zeta;

      if (ss >> label >> zeta) {
        // Ensure the label starts with a digit (the n quantum number)
        if (!std::isdigit(label[0]))
          continue;

        int n = std::stoi(label.substr(0, 1));
        std::vector<double> coeffs;
        double c;
        while (ss >> c)
          coeffs.push_back(c);

        // 4. Expand for each m component (-l to +l)
        // This accounts for the degeneracy of higher l states
        for (int m = -current_l; m <= current_l; ++m) {
          temp_basis.push_back({n, current_l, m, zeta, mol.atom_centers(i, 0),
                                mol.atom_centers(i, 1),
                                mol.atom_centers(i, 2)});
        }
      }
    }
  }

  return make_manual_basis(temp_basis);
}

KOKKOS_INLINE_FUNCTION
double basis_eval(const STOBasisSet basis, const int basis_idx, const double x,
                  const double y, const double z) {

  const int n_val = basis.n(basis_idx);
  const int l_val = basis.l(basis_idx);
  const int m_val = basis.m(basis_idx);
  const double norm = basis.norm(basis_idx);
  const double zeta = basis.zeta(basis_idx);

  // radial part of the shell
  // radial_part = R_nl(r) = r^(n-1) * C_nl * exp(-⍺ * r))

  double dx = x - basis.O(basis_idx, 0);
  double dy = y - basis.O(basis_idx, 1);
  double dz = z - basis.O(basis_idx, 2);
  double r = Kokkos::sqrt(dx * dx + dy * dy + dz * dz) +
             epsilon_shift; // Avoid pow(0,0)

  double radial_part;

  radial_part =
      norm * Kokkos::pow(r, n_val - l_val - 1) * Kokkos::exp(-zeta * r);

  // Angular part of the shell
  // https://en.wikipedia.org/wiki/Spherical_harmonics
  double angular_part = real_solid_harmonic_cart(l_val, m_val, dx, dy, dz);

  return radial_part * angular_part;
}

KOKKOS_INLINE_FUNCTION
void basis_eval_grad(const STOBasisSet basis, const int basis_idx,
                     const double x, const double y, const double z, double &gx,
                     double &gy, double &gz) {

  const int n_val = basis.n(basis_idx);
  const int l_val = basis.l(basis_idx);
  const int m_val = basis.m(basis_idx);
  const double norm = basis.norm(basis_idx);
  const double zeta = basis.zeta(basis_idx);

  double dx = x - basis.O(basis_idx, 0);
  double dy = y - basis.O(basis_idx, 1);
  double dz = z - basis.O(basis_idx, 2);
  double r = Kokkos::sqrt(dx * dx + dy * dy + dz * dz) +
             epsilon_shift; // Avoid pow(0,0)

  // Angular part
  double S_val;
  S_val = real_solid_harmonic_cart(l_val, m_val, dx, dy, dz);

  double dS_dx;
  double dS_dy;
  double dS_dz;

  grad_real_solid_harmonic_cart(l_val, m_val, dx, dy, dz, dS_dx, dS_dy, dS_dz);

  // Radial part
  double pow_term = safe_pow(r, n_val - l_val - 1);
  double exp_term = Kokkos::exp(-zeta * r);
  double R_pre = norm * pow_term * exp_term;

  double dR_dr = ((n_val - l_val - 1) / r - zeta) * R_pre;

  double common_R = dR_dr / r;

  gx = R_pre * dS_dx + S_val * (dx * common_R);
  gy = R_pre * dS_dy + S_val * (dy * common_R);
  gz = R_pre * dS_dz + S_val * (dz * common_R);
}

template <typename PointsView, typename ValuesView>
void fill_collocation(
    ExecSpace &space, const STOBasisSet &basis, PointsView collocation_points,
    ValuesView collocation_values) // pre-allocated, written in place
{
  int N = basis.nbf();
  int G = collocation_points.extent(0);
  Kokkos::parallel_for(
      "Fill collocation",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>(space, {0, 0}, {N, G}),
      KOKKOS_LAMBDA(int i, int j) {
        // same math as before, written directly into col(i,j)
        const int n_val = basis.n(i);
        const int l_val = basis.l(i);
        const int m_val = basis.m(i);
        const double norm = basis.norm(i);
        const double zeta = basis.zeta(i);

        // radial part of the shell
        // radial_part = R_nl(r) = r^(n-1) * C_nl * exp(-⍺ * r))
        double dx = collocation_points(j, 0) - basis.O(i, 0);
        double dy = collocation_points(j, 1) - basis.O(i, 1);
        double dz = collocation_points(j, 2) - basis.O(i, 2);
        double r = Kokkos::sqrt(dx * dx + dy * dy + dz * dz) +
                   epsilon_shift; // Avoid pow(0,0)

        double radial_part;

        radial_part =
            norm * Kokkos::pow(r, n_val - l_val - 1) * Kokkos::exp(-zeta * r);

        // Angular part of the shell
        // https://en.wikipedia.org/wiki/Spherical_harmonics
        double angular_part =
            real_solid_harmonic_cart(l_val, m_val, dx, dy, dz);

        collocation_values(i, j) = radial_part * angular_part;
      });
}

template <typename PointsView, typename ValuesView>
void fill_grad_collocation(ExecSpace &space, const STOBasisSet &basis_set,
                           PointsView &collocation_points,
                           ValuesView &collocation_values) {

  size_t col_points = collocation_points.extent(0);
  size_t nbasis_functions = basis_set.nbf();

  Kokkos::MDRangePolicy<Kokkos::Rank<2>> md_policy(
      space, {0, 0}, {nbasis_functions, col_points});

  Kokkos::parallel_for(
      "Fill collocation grad", md_policy,
      KOKKOS_LAMBDA(const int &i, const int &j) {
        const int n_val = basis_set.n(i);
        const int l_val = basis_set.l(i);
        const int m_val = basis_set.m(i);
        const double norm = basis_set.norm(i);
        const double zeta = basis_set.zeta(i);

        double dx = collocation_points(j, 0) - basis_set.O(i, 0);
        double dy = collocation_points(j, 1) - basis_set.O(i, 1);
        double dz = collocation_points(j, 2) - basis_set.O(i, 2);
        double r = Kokkos::sqrt(dx * dx + dy * dy + dz * dz) +
                   epsilon_shift; // Avoid pow(0,0)

        // Angular part
        double S_val;
        S_val = real_solid_harmonic_cart(l_val, m_val, dx, dy, dz);

        double dS_dx;
        double dS_dy;
        double dS_dz;

        grad_real_solid_harmonic_cart(l_val, m_val, dx, dy, dz, dS_dx, dS_dy,
                                      dS_dz);

        // Radial part
        double pow_term = safe_pow(r, n_val - l_val - 1);
        double exp_term = Kokkos::exp(-zeta * r);
        double R_pre = norm * pow_term * exp_term;

        double dR_dr = ((n_val - l_val - 1) / r - zeta) * R_pre;
        double common_R = dR_dr / r;

        collocation_values(i, j, 0) = R_pre * dS_dx + S_val * (dx * common_R);
        collocation_values(i, j, 1) = R_pre * dS_dy + S_val * (dy * common_R);
        collocation_values(i, j, 2) = R_pre * dS_dz + S_val * (dz * common_R);
      });
}
} // namespace NuKEXC
