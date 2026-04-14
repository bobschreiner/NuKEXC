#pragma once

#include "kokkos_config.hpp"
#include "nukexc_utils.hpp"
#include "shell.hpp"
#include "spherical_harmonics.hpp"

namespace NuKEXC {

struct GeneralShellDevice {

  // Shared Data Buffers
  Kokkos::View<double *> all_alphas; // Exponents (Alphas for GTO/STO)
  Kokkos::View<double *> all_coeffs; // Coefficients
  Kokkos::View<size_t *> offsets;    // Start index of each shell in the buffers

  // Shell Metadata
  Kokkos::View<double **> O;      // Origins (N_shells x 3)
  Kokkos::View<int *> l;          // Angular momentum
  Kokkos::View<int *> n_prim;     // Number of primitives/points per shell
  Kokkos::View<int *> shell_type; // 0: GTO, 1: STO, 2: NAO, etc.

  int maximal_l = 7;
  Kokkos::View<double **> harmonic_pre_factors;

  GeneralShellDevice(int n_shells, int total_prims)
      : all_alphas("alphas", total_prims), all_coeffs("coeffs", total_prims),
        offsets("offsets", n_shells), O("origins", n_shells, 3),
        l("l", n_shells), n_prim("n_prim", n_shells),
        shell_type("shell_type", n_shells) {

    // Initialize the harmonic pre_factors
    harmonic_pre_factors = Kokkos::View<double **>(
        "Harmonic pre-factors", maximal_l+1, maximal_l + 1);
    for (int l = 0; l < maximal_l+1; ++l) {
      detail::compute_prefactors_for_spherical_harmonics(l,
                                                         harmonic_pre_factors);
    }
  };
};

template <typename F>
void sync_general_shells(
    const std::vector<std::unique_ptr<NuKEXC::Shell<F>>> &host_shells,
    GeneralShellDevice &device) {

  auto h_alphas = Kokkos::create_mirror_view(device.all_alphas);
  auto h_coeffs = Kokkos::create_mirror_view(device.all_coeffs);
  auto h_off = Kokkos::create_mirror_view(device.offsets);
  auto h_nprim = Kokkos::create_mirror_view(device.n_prim);
  auto h_l = Kokkos::create_mirror_view(device.l);
  auto h_type = Kokkos::create_mirror_view(device.shell_type);
  auto h_O = Kokkos::create_mirror_view(device.O);

  size_t current_offset = 0;
  for (size_t i = 0; i < host_shells.size(); ++i) {
    const auto &s = host_shells[i];

    h_off(i) = current_offset;
    h_nprim(i) = s->n_prim();
    h_l(i) = s->l();
    h_type(i) = s->get_type(); // e.g., NAO=0, STO=1 , GTO=2

    for (int p = 0; p < s->n_prim(); ++p) {
      h_alphas(current_offset + p) = s->alpha(p);
      h_coeffs(current_offset + p) = s->coeff(p);
    }
    for (int d = 0; d < 3; ++d)
      h_O(i, d) = s->origin(d);

    current_offset += s->n_prim();
  }

  Kokkos::deep_copy(device.all_alphas, h_alphas);
  Kokkos::deep_copy(device.all_coeffs, h_coeffs);
  Kokkos::deep_copy(device.offsets, h_off);
  Kokkos::deep_copy(device.O, h_O);
  Kokkos::deep_copy(device.l, h_l);
  Kokkos::deep_copy(device.n_prim, h_nprim);
  Kokkos::deep_copy(device.shell_type, h_type);
}

void evaluate_gto_basis_shells_on_collocation_points(
    const GeneralShellDevice &device,
    const Kokkos::View<double **> &collocation_points,
    Kokkos::View<double ***> &collocation_values) {

  size_t n_shells = device.O.extent(0);
  size_t col_points = collocation_points.extent(0);

  Kokkos::MDRangePolicy<Kokkos::Rank<2>> md_policy({0, 0},
                                                   {n_shells, col_points});
  Kokkos::parallel_for(
      "Compute collocation of shells", md_policy,
      KOKKOS_LAMBDA(const int &i, const int &j) {
        const int start = device.offsets(i);
        const int n = device.n_prim(i);
        const int l_val = device.l(i);

        // radial part of the shell
        // radial_part = R_l(r) = r^l * (∑_i C_i * exp(-⍺_i * r^2))
        double radial_part = 0;

        double r = utils::rad_dist(
            Kokkos::subview(device.O, i, Kokkos::ALL()),
            Kokkos::subview(collocation_points, j, Kokkos::ALL()));

        double r2 = r * r;
        for (int p = 0; p < n; ++p) {
          double a = device.all_alphas(start + p);
          double coeff = device.all_coeffs(start + p);

          radial_part += coeff * Kokkos::exp(-a * r2);
        }
        radial_part *= Kokkos::pow(r, l_val);

        // Angular part of the shell
        // Angular part  = Y_lm = (-1)^m * sqrt{[(2l+1)/4π] * [(l-m)! / (l+m)!]}
        // P_ml(cos(θ)) * (cos(m*ϕ) + i*sin(m*ϕ))
        Kokkos::View<double *> angular_part("Spherical harmonic",
                                            2 * l_val + 1);
        for (int m_val = -l_val; m_val < l_val + 1; ++m_val) {
          double x = collocation_points(i, 0);
          double y = collocation_points(i, 1);
          double z = collocation_points(i, 2);
          double theta = Kokkos::acos(z / r);
          double phi = Kokkos::atan2(y, x);

          int m_abs = Kokkos::abs(m_val);
          angular_part(m_val) = NuKEXC::detail::real_spherical_harmonic(
              l_val, m_val, theta, phi,
              device.harmonic_pre_factors(l_val, m_abs));
        }


	//TODO:: write test for real_spherical_harmonic
	//TODO:: implement sum over something 
      });
}

} // namespace NuKEXC
