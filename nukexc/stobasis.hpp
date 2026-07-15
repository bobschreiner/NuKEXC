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

#include <Kokkos_Core.hpp>
#include <Kokkos_Macros.hpp>
#include <cctype>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "atomic_properties.hpp"
#include "molecule.hpp"
#include "nukexc_config.hpp"
#include "nukexc_utils.hpp"
#include "spherical_harmonics.hpp"

namespace Nukexc {

struct STOBasisSet {
  Kokkos::View<int *> n;
  Kokkos::View<int *> l;
  Kokkos::View<int *> m;
  Kokkos::View<double *> norm;
  Kokkos::View<double *> zeta;
  Kokkos::View<Point *> O;
  Kokkos::View<double *> cutoff_radii;

  size_t nbf() const { return O.extent(0); };
};

// Map shell labels to l values
int label_to_l(char label) {
  label = std::toupper(label);
  static const std::string order =
      "SPDFGHI"; // Standard spectroscopic notation (skipping J)
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
double compute_cutoff(STOFunc f, const double norm, const double cutoff_tol) {
  // Preconditions
  KOKKOS_ASSERT(f.n >= 1);
  KOKKOS_ASSERT(f.zeta > 0.0);
  KOKKOS_ASSERT(cutoff_tol > 0.0);

  // If the function is already everywhere below tolerance, cutoff is at origin
  if (norm <= cutoff_tol)
    return 0.0;

  const double ln_tol = Kokkos::log(cutoff_tol);
  const double ln_norm = Kokkos::log(norm);

  // g(r) = ln_norm + (n-1)*ln(r) - zeta*r - ln_tol = 0
  // g'(r) = (n-1)/r - zeta
  // The unique root we want is on the far side of the peak, i.e. r > peak,
  // where g'(r) < 0 so the function is strictly decreasing.
  const double peak = (f.n > 1) ? (f.n - 1) / f.zeta : 0.0;

  // Initial guess: asymptotically correct when the log(r) term is negligible.
  // Guaranteed to be >= peak for reasonable inputs (norm >> tol).
  double r = Kokkos::max(peak + (ln_norm - ln_tol) / f.zeta, peak + 1e-6);

  // Safe interval: we stay in (peak, +inf) where g is monotone decreasing,
  // so there is exactly one root and Newton steps are well-defined.
  // We track an upper bound via the interval [r_lo, r_hi] and fall back to
  // bisection whenever a Newton step would leave (peak, r_hi].
  double r_lo = peak + 1e-10;
  double r_hi = r;

  // Ensure r_hi is actually above the root (g(r_hi) <= 0)
  // If not, double r_hi until it is.
  for (int k = 0; k < 64; ++k) {
    double g_hi =
        ln_norm + (f.n - 1) * Kokkos::log(r_hi) - f.zeta * r_hi - ln_tol;
    if (g_hi <= 0.0)
      break;
    r_hi *= 2.0;
  }
  r = r_hi;

  for (int k = 0; k < 50; ++k) {
    double g = ln_norm + (f.n - 1) * Kokkos::log(r) - f.zeta * r - ln_tol;

    // Update bracket
    if (g > 0.0)
      r_lo = r;
    else
      r_hi = r;

    double g_prime = (f.n - 1) / r - f.zeta;

    // g_prime < 0 everywhere in (peak, +inf); if somehow we're at the peak
    // itself, fall back to bisection immediately.
    double r_next;
    if (g_prime >= 0.0) {
      r_next = 0.5 * (r_lo + r_hi); // bisection fallback
    } else {
      r_next = r - g / g_prime; // Newton step
      // Reject step if it leaves the safe interval
      if (r_next <= r_lo || r_next >= r_hi)
        r_next = 0.5 * (r_lo + r_hi); // bisection fallback
    }

    double delta = Kokkos::abs(r_next - r);
    r = r_next;

    if (delta < 1e-10 * r)
      break;

#ifndef NDEBUG
    // In debug builds, assert we haven't diverged
    KOKKOS_ASSERT(r > 0.0);
#endif
  }

#ifndef NDEBUG
  {
    double g_final = ln_norm + (f.n - 1) * Kokkos::log(r) - f.zeta * r - ln_tol;
    KOKKOS_ASSERT(Kokkos::abs(g_final) < 1e-6);
  }
#endif
  return r;
}

STOBasisSet make_manual_basis(const std::vector<STOFunc> &funcs,
                              double cutoff_tol = 1e-10) {
  const size_t nbf = funcs.size();

  STOBasisSet basis;
  basis.n = Kokkos::View<int *>("n", nbf);
  basis.l = Kokkos::View<int *>("l", nbf);
  basis.m = Kokkos::View<int *>("m", nbf);
  basis.zeta = Kokkos::View<double *>("zeta", nbf);
  basis.norm = Kokkos::View<double *>("coeff", nbf);
  basis.O = Kokkos::View<Point *>(" centers ", nbf);
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

    // Compute cutoff radius, with the Newton-Raphson method
    //
    n_h(i) = f.n;
    l_h(i) = f.l;
    m_h(i) = f.m;
    zeta_h(i) = f.zeta;
    norm_h(i) = norm;
    O_h(i)[0] = f.ox;
    O_h(i)[1] = f.oy;
    O_h(i)[2] = f.oz;
    cutoff_radii_h(i) = compute_cutoff(f, norm, cutoff_tol);
  }

  Kokkos::deep_copy(basis.n, n_h);
  Kokkos::deep_copy(basis.l, l_h);
  Kokkos::deep_copy(basis.m, m_h);
  Kokkos::deep_copy(basis.zeta, zeta_h);
  Kokkos::deep_copy(basis.norm, norm_h);
  Kokkos::deep_copy(basis.O, O_h);
  Kokkos::deep_copy(basis.cutoff_radii, cutoff_radii_h);

  return basis;
}

STOBasisSet load_adf_basis(const Molecule &mol,
                           const std::string &data_dir = "input/zorabasis/TZP",
                           const double cutoff_tol = 1e-10,
                           const bool fit = false) {

  std::vector<STOFunc> temp_basis;
  auto Z_h = Kokkos::create_mirror_view_and_copy(HostSpace{}, mol.Z);

  for (size_t i = 0; i < mol.natoms; ++i) {
    std::string element_symbol = detail::symbols[Z_h(i)];
    element_symbol[0] = std::toupper(element_symbol[0]);

    std::string filename = data_dir + "/" + element_symbol;

    std::ifstream file(filename);
    if (!file.is_open())
      continue;

    std::string line;

    // Skip all lines until we find the basis keyword
    std::string basis_keyword = fit ? "FIT" : "BASIS";
    while (std::getline(file, line)) {
      if (line.find(basis_keyword) != std::string::npos)
        break;
    }

    // Read line by line until we find END keyword
    while (std::getline(file, line)) {

      if (line.find("END") != std::string::npos) {
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

          auto atom_centers_h = Kokkos::create_mirror_view_and_copy(
              HostSpace{}, mol.atom_centers);

          // 4. Expand for each m component (-l to +l)
          // This accounts for the degeneracy of higher l states
          for (int m = -l; m <= l; ++m) {
            temp_basis.push_back({n, l, m, zeta, atom_centers_h(i)[0],
                                  atom_centers_h(i)[1], atom_centers_h(i)[2]});
          }
        }
      }
    }
  }
  return make_manual_basis(temp_basis, cutoff_tol);
}

STOBasisSet
load_thakkar_basis(const Molecule &mol,
                   const std::string &data_dir = "input/k99light/neutral",
                   const double cutoff_tol = 1e-10) {

  std::vector<STOFunc> temp_basis;

  auto Z_h = Kokkos::create_mirror_view_and_copy(HostSpace{}, mol.Z);
  for (size_t i = 0; i < mol.natoms; ++i) {
    std::string element_symbol = detail::symbols[Z_h(i)];
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

        auto atom_centers_h =
            Kokkos::create_mirror_view_and_copy(HostSpace{}, mol.atom_centers);
        // 4. Expand for each m component (-l to +l)
        // This accounts for the degeneracy of higher l states
        for (int m = -current_l; m <= current_l; ++m) {
          temp_basis.push_back({n, current_l, m, zeta, atom_centers_h(i)[0],
                                atom_centers_h(i)[1], atom_centers_h(i)[2]});
        }
      }
    }
  }

  return make_manual_basis(temp_basis, cutoff_tol);
}

struct ShellParams {
  int l, m, n;
  double zeta, norm, ox, oy, oz;
};

KOKKOS_INLINE_FUNCTION
ShellParams load_shell(const STOBasisSet &basis, int basis_idx) {
  return {basis.l(basis_idx),    basis.m(basis_idx),    basis.n(basis_idx),
          basis.zeta(basis_idx), basis.norm(basis_idx), basis.O(basis_idx)[0],
          basis.O(basis_idx)[1], basis.O(basis_idx)[2]};
}

KOKKOS_INLINE_FUNCTION
void basis_eval(const STOBasisSet basis, const int basis_idx, const double x,
                const double y, const double z, double &val) {

  const int l_val = basis.l(basis_idx);
  const int m_val = basis.m(basis_idx);

  const double dx = x - basis.O(basis_idx)[0];
  const double dy = y - basis.O(basis_idx)[1];
  const double dz = z - basis.O(basis_idx)[2];

  double radial_part;
  {
    // radial part of the shell
    // radial_part = R_nl(r) = r^(n-1) * C_nl * exp(-⍺ * r))
    const int n_val = basis.n(basis_idx);
    const double zeta = basis.zeta(basis_idx);
    const double norm = basis.norm(basis_idx);
    const double r = Kokkos::sqrt(dx * dx + dy * dy + dz * dz) +
                     epsilon_shift; // Avoid pow(0,0)

    radial_part =
        norm * int_pow(r, n_val - l_val - 1) * Kokkos::exp(-zeta * r);
  }

  // Angular part of the shell
  // https://en.wikipedia.org/wiki/Spherical_harmonics
  double angular_part;
  real_solid_harmonic_cart_precomputed(l_val, m_val, dx, dy, dz, angular_part);

  val = radial_part * angular_part;
}

KOKKOS_INLINE_FUNCTION
double basis_eval_fast(const ShellParams &sh, double x, double y, double z) {
  const double dx = x - sh.ox;
  const double dy = y - sh.oy;
  const double dz = z - sh.oz;
  const double r = Kokkos::sqrt(dx * dx + dy * dy + dz * dz) + epsilon_shift;

  const int k = sh.n - sh.l - 1;
  const double radial =
      (k == 0) ? sh.norm * Kokkos::exp(-sh.zeta * r)
               : sh.norm * int_pow(r, k) * Kokkos::exp(-sh.zeta * r);

  double angular_part;
  real_solid_harmonic_cart_precomputed(sh.l, sh.m, x, y, z, angular_part);
  return radial * angular_part;
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

  const double dx = x - basis.O(basis_idx)[0];
  const double dy = y - basis.O(basis_idx)[1];
  const double dz = z - basis.O(basis_idx)[2];
  const double r = Kokkos::sqrt(dx * dx + dy * dy + dz * dz) +
                   epsilon_shift; // Avoid pow(0,0)

  // Angular part
  double S_val;
  S_val = real_solid_harmonic_cart(l_val, m_val, dx, dy, dz);

  double dS_dx;
  double dS_dy;
  double dS_dz;

  grad_real_solid_harmonic_cart(l_val, m_val, dx, dy, dz, dS_dx, dS_dy, dS_dz);

  // Radial part
  double pow_term = int_pow(r, n_val - l_val - 1);
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
        double dx = collocation_points(j)[0] - basis.O(i)[0];
        double dy = collocation_points(j)[1] - basis.O(i)[1];
        double dz = collocation_points(j)[2] - basis.O(i)[2];

        double radial_part;
        {
          double r = Kokkos::sqrt(dx * dx + dy * dy + dz * dz) +
                     epsilon_shift; // Avoid pow(0,0)

          radial_part =
              norm * int_pow(r, n_val - l_val - 1) * Kokkos::exp(-zeta * r);
        }
        // Angular part of the shell
        // https://en.wikipedia.org/wiki/Spherical_harmonics
        double angular_part;
        real_solid_harmonic_cart_precomputed(l_val, m_val, dx, dy, dz,
                                             angular_part);

        collocation_values(i, j) = radial_part * angular_part;
      });
}

template <typename PointsView, typename ValuesView>
void fill_grad_collocation(ExecSpace &space, const STOBasisSet &basis_set,
                           PointsView &collocation_points,
                           ValuesView &collocation_gx,
                           ValuesView &collocation_gy,
                           ValuesView &collocation_gz) {

  size_t col_points = collocation_points.extent(0);
  size_t nbasis_functions = basis_set.nbf();

  Kokkos::MDRangePolicy<Kokkos::Rank<2>> md_policy(
      space, {0, 0}, {nbasis_functions, col_points});

  Kokkos::parallel_for(
      "Fill collocation grad", md_policy,
      KOKKOS_LAMBDA(const int &i, const int &g) {
        const int n_val = basis_set.n(i);
        const int l_val = basis_set.l(i);
        const int m_val = basis_set.m(i);
        const double norm = basis_set.norm(i);
        const double zeta = basis_set.zeta(i);

        double dx = collocation_points(g)[0] - basis_set.O(i)[0];
        double dy = collocation_points(g)[1] - basis_set.O(i)[1];
        double dz = collocation_points(g)[2] - basis_set.O(i)[2];
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

        collocation_gx(i, g) = R_pre * dS_dx + S_val * (dx * common_R);
        collocation_gy(i, g) = R_pre * dS_dy + S_val * (dy * common_R);
        collocation_gz(i, g) = R_pre * dS_dz + S_val * (dz * common_R);
      });
}

struct VG {
  double val; // Value
  double dx;  // d/dx
  double dy;  // d/dy
  double dz;  // d/dz
};

struct ScratchBasisParams {
  double zeta, norm;
  Point O;
  int n, l, m;
};

KOKKOS_INLINE_FUNCTION
void basis_eval_with_grad(const ScratchBasisParams &basis, const Point &p,
                          double &val, double &gradx, double &grady,
                          double &gradz) {

  const int n_val = basis.n;
  const int l_val = basis.l;
  const int m_val = basis.m;
  const double norm = basis.norm;
  const double zeta = basis.zeta;
  const Point origin = basis.O;

  double dx = p[0] - origin[0];
  double dy = p[1] - origin[1];
  double dz = p[2] - origin[2];
  double r = Kokkos::sqrt(dx * dx + dy * dy + dz * dz) +
             epsilon_shift; // Avoid pow(0,0)

  // Angular part
  double S_val;
  double dS_dx;
  double dS_dy;
  double dS_dz;

  // Completely precomputed coefficients
  real_solid_harmonic_cart_and_grad_precomputed(l_val, m_val, dx, dy, dz, S_val,
                                                dS_dx, dS_dy, dS_dz);

  // Radial part
  double pow_term = int_pow(r, n_val - l_val - 1);
  double exp_term = Kokkos::exp(-zeta * r);
  double R_pre = norm * pow_term * exp_term;

  double dR_dr = ((n_val - l_val - 1) / r - zeta) * R_pre;
  double common_R = dR_dr / r;

  val = R_pre * S_val;
  gradx = R_pre * dS_dx + S_val * (dx * common_R);
  grady = R_pre * dS_dy + S_val * (dy * common_R);
  gradz = R_pre * dS_dz + S_val * (dz * common_R);
}

} // namespace Nukexc
