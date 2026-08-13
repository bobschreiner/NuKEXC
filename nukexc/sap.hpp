/*
 *    NuKEXC -- Numerical Kokkos Enhanced Exchange Correlation Integrator
 *    Copyright (c) 2026, Bob Schreiner
 *    All rights reserved.
 *
 *    SPDX-License-Identifier: BSD-3-Clause
 *
 *    Redistribution and use in source and binary forms, with or without
 *    modification, are permitted provided that the following conditions are
 *    met:
 *
 *    1. Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *
 *    2. Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *
 *    3. Neither the name of the copyright holder nor the names of its
 *       contributors may be used to endorse or promote products derived from
 *       this software without specific prior written permission.
 *
 *    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 *    A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 *    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 *    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include "atomic_properties.hpp"
#include "nukexc_config.hpp"

#include <Kokkos_Core.hpp>
#include <Kokkos_MathematicalFunctions.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace Nukexc {

// Superposition of atomic potentials (SAP) initial guess.
//
//   V_SAP(r) = - sum_A Z_eff^A(|r - R_A|) / |r - R_A|
//
// The guess Fock matrix is T + V_SAP -- NOT H_core + V_SAP, since V_SAP
// already contains the -Z/r nuclear attraction.
//
// Data: S. Lehtola, "Assessment of Initial Guesses for Self-Consistent Field
// Calculations. Superposition of Atomic Potentials: Simple yet Efficient",
// J. Chem. Theory Comput. 15, 1593 (2019), DOI 10.1021/acs.jctc.8b01089.
//
// The v_<El>.dat files tabulate the *screening* charge q(r) = Z - Z_eff(r),
// which runs from q(0) = 0 (bare nucleus, nothing screened) up to q(inf) = Z
// (neutral atom, fully screened). The potential needs the complement, so the
// sign is flipped once here on load:
//
//     Z_eff(r) = Z - q(r)
//
// Z_eff(0) = Z reproduces the bare nuclear cusp exactly, so the r -> 0 limit
// of V_SAP is identical to the ordinary nuclear attraction.

// Radial tables for every atom in the molecule, in device memory.
//
// One row per *atom* rather than per element: duplicating rows for repeated
// elements costs n_atoms * n_r * 8 B (a few hundred kB for anything we run)
// and buys a direct tab(k, i) indexing off the atom loop counter, with no
// Z -> row indirection in the innermost loop.
struct SapPotentials {
  DeviceView2DRight zeff;   // (n_atoms, n_r) effective charge Z_eff
  DeviceView1D grid_a;      // (n_atoms) radial grid parameter a
  DeviceView1D grid_b;      // (n_atoms) radial grid parameter b
  Kokkos::View<int *> n_pt; // (n_atoms) tabulated points for this atom
};

namespace detail {

// One element's table plus the parameters of its radial grid.
struct SapRadialTable {
  std::vector<double> zeff;
  double a{1.0};
  double b{0.0};
};

// Every file in the distribution samples the GPAW radial grid
//
//     r_g = a * g / (1 - b * g),     g = 0 ... n-1
//
// which inverts in closed form to g = r / (a + b * r). That is what lets the
// device-side lookup skip the search: the bracketing index is two flops away.
// (a, b) are recovered by least squares on the linear form r = a*g + b*(r*g);
// over the shipped files this reproduces r to ~5e-10, i.e. to the precision
// the files are printed at.
inline void fit_gpaw_grid(const std::vector<double> &r, double &a, double &b) {
  double Suu = 0.0, Suv = 0.0, Svv = 0.0, Suy = 0.0, Svy = 0.0;
  for (size_t i = 0; i < r.size(); ++i) {
    const double u = static_cast<double>(i); // g
    const double v = r[i] * u;               // r * g
    Suu += u * u;
    Suv += u * v;
    Svv += v * v;
    Suy += u * r[i];
    Svy += v * r[i];
  }

  const double det = Suu * Svv - Suv * Suv;
  if (std::abs(det) < 1e-30)
    throw std::runtime_error("SAP: degenerate radial grid fit");

  a = (Svv * Suy - Suv * Svy) / det;
  b = (Suu * Svy - Suv * Suy) / det;

  // Fail loudly rather than silently interpolating at the wrong radii: if a
  // different potential set is ever dropped in, the closed-form inverse below
  // is only valid if it uses the same grid family.
  double max_err = 0.0;
  for (size_t i = 0; i < r.size(); ++i) {
    const double g = static_cast<double>(i);
    const double pred = a * g / (1.0 - b * g);
    max_err = std::max(max_err, std::abs(pred - r[i]));
  }
  if (max_err > 1e-6)
    throw std::runtime_error(
        "SAP: radial grid is not of the form r = a*g/(1 - b*g) (max deviation "
        + std::to_string(max_err) +
        " bohr). The analytic index inverse does not apply to this potential "
        "set; a bracketing search would be required instead.");
}

// Read v_<El>.dat and convert the screening charge to Z_eff.
inline SapRadialTable read_sap_file(const std::string &dir, unsigned Z) {
  SapRadialTable tab;

  // Ghost centres carry no potential.
  if (Z == 0)
    return tab;

  if (Z >= symbols.size())
    throw std::runtime_error("SAP: no element symbol known for Z = " +
                             std::to_string(Z));

  const std::string path = dir + "v_" + symbols[Z] + ".dat";
  std::ifstream file(path);
  if (!file)
    throw std::runtime_error("SAP: cannot open potential file '" + path +
                             "'. Set NUKEXC_SAP_DIR to the directory holding "
                             "the v_<El>.dat tables.");

  std::vector<double> r, q;
  double ri, qi;
  while (file >> ri >> qi) {
    r.push_back(ri);
    q.push_back(qi);
  }

  if (r.size() < 2)
    throw std::runtime_error("SAP: '" + path + "' holds fewer than 2 points");

  // q(r) = Z - Z_eff(r)  ->  Z_eff(r) = Z - q(r)
  tab.zeff.resize(r.size());
  for (size_t i = 0; i < r.size(); ++i)
    tab.zeff[i] = static_cast<double>(Z) - q[i];

  fit_gpaw_grid(r, tab.a, tab.b);
  return tab;
}

} // namespace detail

// Directory holding the v_<El>.dat tables. Relative paths only work when the
// process runs from the repository root, which is not the case on a compute
// node, so allow an override.
inline std::string sap_potential_dir(const std::string &fallback) {
  if (const char *env = std::getenv("NUKEXC_SAP_DIR")) {
    std::string dir(env);
    if (!dir.empty() && dir.back() != '/')
      dir.push_back('/');
    return dir;
  }
  return fallback;
}

// Build the device-side SAP tables for the atoms of a molecule.
//
// Each distinct element is read once and fanned out to the atoms carrying it,
// so a W4-11 species costs two or three file reads regardless of size.
inline SapPotentials load_sap_potentials(const Kokkos::View<unsigned *> &Z_dev,
                                         const std::string &dir) {
  const size_t n_atoms = Z_dev.extent(0);
  auto Z_h = Kokkos::create_mirror_view_and_copy(HostSpace{}, Z_dev);

  std::map<unsigned, detail::SapRadialTable> by_element;
  size_t n_r = 0;
  for (size_t k = 0; k < n_atoms; ++k) {
    const unsigned Zk = Z_h(k);
    if (by_element.find(Zk) == by_element.end())
      by_element.emplace(Zk, detail::read_sap_file(dir, Zk));
    n_r = std::max(n_r, by_element.at(Zk).zeff.size());
  }

  SapPotentials sap;
  sap.zeff = DeviceView2DRight("SAP Z_eff", n_atoms, std::max<size_t>(n_r, 1));
  sap.grid_a = DeviceView1D("SAP grid a", n_atoms);
  sap.grid_b = DeviceView1D("SAP grid b", n_atoms);
  sap.n_pt = Kokkos::View<int *>("SAP n_pt", n_atoms);

  auto zeff_h = Kokkos::create_mirror_view(sap.zeff);
  auto a_h = Kokkos::create_mirror_view(sap.grid_a);
  auto b_h = Kokkos::create_mirror_view(sap.grid_b);
  auto n_h = Kokkos::create_mirror_view(sap.n_pt);

  for (size_t k = 0; k < n_atoms; ++k) {
    const detail::SapRadialTable &tab = by_element.at(Z_h(k));
    a_h(k) = tab.a;
    b_h(k) = tab.b;
    n_h(k) = static_cast<int>(tab.zeff.size());
    for (size_t i = 0; i < tab.zeff.size(); ++i)
      zeff_h(k, i) = tab.zeff[i];
  }

  Kokkos::deep_copy(sap.zeff, zeff_h);
  Kokkos::deep_copy(sap.grid_a, a_h);
  Kokkos::deep_copy(sap.grid_b, b_h);
  Kokkos::deep_copy(sap.n_pt, n_h);

  return sap;
}

// Effective charge of atom k at distance r, by linear interpolation on the
// tabulated radial grid.
//
// The clamp at zero is not cosmetic: the tabulated tails carry numerical noise
// of order 1e-15 with either sign, and the caller feeds this into a sqrt (the
// nuclear term is accumulated as a rank update, V = -(sqrt(v) phi)(sqrt(v)
// phi)^T). A single negative sample would turn the whole guess into NaN.
KOKKOS_INLINE_FUNCTION
double sap_zeff(const SapPotentials &sap, const int k, const double r) {
  const int n = sap.n_pt(k);
  if (n < 2)
    return 0.0; // ghost centre

  // Invert r = a*g/(1 - b*g). r >= 0 and a, b > 0, so x >= 0 always, and x is
  // bounded above by 1/b -- no overflow for large r.
  const double x = r / (sap.grid_a(k) + sap.grid_b(k) * r);
  if (x >= static_cast<double>(n - 1))
    return 0.0; // past the tabulated range: neutral atom, fully screened

  const int i = static_cast<int>(x);
  const double f = x - static_cast<double>(i);
  return Kokkos::fmax(
      (1.0 - f) * sap.zeff(k, i) + f * sap.zeff(k, i + 1), 0.0);
}

} // namespace Nukexc
