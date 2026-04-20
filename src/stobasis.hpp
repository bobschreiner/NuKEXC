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
#include "kokkos_config.hpp"
#include "molecule.hpp"
#include "nukexc_utils.hpp"
#include "spherical_harmonics.hpp"

namespace NuKEXC {

struct STOBasisSet {

  Kokkos::View<int *> n_;
  Kokkos::View<int *> l_;
  Kokkos::View<int *> m_;
  Kokkos::View<double *> norm_;
  Kokkos::View<double *> alpha_;
  Kokkos::View<double *[3]> O_;

  size_t nbf() const { return O_.extent(0); };
};

// Map shell labels to l values
int label_to_l(char label) {
  label = std::toupper(label);
  static const std::string order =
      "SPDFGHIK"; // Standard spectroscopic notation (skipping J)
  size_t pos = order.find(label);
  return (pos != std::string::npos) ? static_cast<int>(pos) : -1;
}

STOBasisSet load_sto_basis(const Molecule &mol, const std::string &data_dir) {
  struct RawFunc {
    int n, l, m;
    double alpha, norm;
    double x, y, z;
  };
  std::vector<RawFunc> temp_basis;

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

      //  Parse Basis rows: Label (e.g. 4F), Alpha, then coefficients
      std::stringstream ss(line);
      std::string label;
      double alpha;

      if (ss >> label >> alpha) {
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
          temp_basis.push_back({n, current_l, m, alpha, 1.,
                                mol.atom_centers(i, 0), mol.atom_centers(i, 1),
                                mol.atom_centers(i, 2)});
        }
      }
    }
  }

  // Allocate Kokkos Views based on total expanded size
  size_t total_nbf = temp_basis.size();
  STOBasisSet basis;
  basis.n_ = Kokkos::View<int *>("n", total_nbf);
  basis.l_ = Kokkos::View<int *>("l", total_nbf);
  basis.m_ = Kokkos::View<int *>("m", total_nbf);
  basis.alpha_ = Kokkos::View<double *>("alpha", total_nbf);
  basis.norm_ = Kokkos::View<double *>("normalization", total_nbf);
  basis.O_ = Kokkos::View<double *[3]>("centers", total_nbf);

  // Create Host Mirrors
  auto n_h = Kokkos::create_mirror_view(basis.n_);
  auto l_h = Kokkos::create_mirror_view(basis.l_);
  auto m_h = Kokkos::create_mirror_view(basis.m_);
  auto alpha_h = Kokkos::create_mirror_view(basis.alpha_);
  auto norm_h = Kokkos::create_mirror_view(basis.norm_);
  auto O_h = Kokkos::create_mirror_view(basis.O_);

  for (size_t i = 0; i < total_nbf; ++i) {
    int n = temp_basis[i].n;
    double alpha = temp_basis[i].alpha;
    double norm = std::pow(2.0 * alpha, n + 0.5) / std::sqrt(factorial(2 * n));

    n_h(i) = temp_basis[i].n;
    l_h(i) = temp_basis[i].l;
    m_h(i) = temp_basis[i].m;
    alpha_h(i) = temp_basis[i].alpha;
    norm_h(i) = norm;
    O_h(i, 0) = temp_basis[i].x;
    O_h(i, 1) = temp_basis[i].y;
    O_h(i, 2) = temp_basis[i].z;
  }

  Kokkos::deep_copy(basis.n_, n_h);
  Kokkos::deep_copy(basis.l_, l_h);
  Kokkos::deep_copy(basis.m_, m_h);
  Kokkos::deep_copy(basis.alpha_, alpha_h);
  Kokkos::deep_copy(basis.norm_, norm_h);
  Kokkos::deep_copy(basis.O_, O_h);

  return basis;
}

Kokkos::View<double **> evaluate_sto_basis_shells_on_collocation_points(
    const STOBasisSet &basis_set,
    Kokkos::View<double *[3]> &collocation_points) {

  size_t col_points = collocation_points.extent(0);
  size_t nbasis_functions = basis_set.nbf();

  Kokkos::View<double **> collocation_values("collocation values",
                                             nbasis_functions, col_points);

  Kokkos::MDRangePolicy<Kokkos::Rank<2>> md_policy(
      {0, 0}, {nbasis_functions, col_points});
  Kokkos::parallel_for(
      "Compute collocation of shells", md_policy,
      KOKKOS_LAMBDA(const int &i, const int &j) {
        const int n_val = basis_set.n_(i);
        const int l_val = basis_set.l_(i);
        const int m_val = basis_set.m_(i);
        const double norm = basis_set.norm_(i);
        const double a = basis_set.alpha_(i);

        // radial part of the shell
        // radial_part = R_nl(r) = r^(n-1) * C_nl * exp(-⍺ * r))

        double dx = collocation_points(j, 0) - basis_set.O_(i, 0);
        double dy = collocation_points(j, 1) - basis_set.O_(i, 1);
        double dz = collocation_points(j, 2) - basis_set.O_(i, 2);
        double r = Kokkos::sqrt(dx * dx + dy * dy + dz * dz); // Avoid pow(0,0)

        double radial_part;

        if (r > 0.0) {
          radial_part =
              norm * Kokkos::pow(r, n_val - l_val - 1) * Kokkos::exp(-a * r);
        } else {
          // At the nucleus (r=0):
          // Only n=1 has a non-zero value (r^0 = 1).
          // n > 1 are all 0.0 (r^1, r^2, etc.)
          radial_part = (n_val == 1) ? norm * Kokkos::pow(r, l_val - 1) : 0.0;
        }

        // Angular part of the shell
        // https://en.wikipedia.org/wiki/Spherical_harmonics
        double angular_part =
            real_solid_harmonic_cart(l_val, m_val, dx, dy, dz);

        collocation_values(i, j) = radial_part * angular_part;
      });
  return collocation_values;
}

} // namespace NuKEXC
