#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <iostream>
// #include <tuple>

#include "contiguous_container.hpp"
#include "gau_rad_eval.hpp"

namespace NuKEXC {

namespace detail {

static constexpr size_t shell_nprim_max = 32ul;

static constexpr std::array<int64_t, 31> df_Kminus1 = {{1LL,
                                                        1LL,
                                                        1LL,
                                                        2LL,
                                                        3LL,
                                                        8LL,
                                                        15LL,
                                                        48LL,
                                                        105LL,
                                                        384LL,
                                                        945LL,
                                                        3840LL,
                                                        10395LL,
                                                        46080LL,
                                                        135135LL,
                                                        645120LL,
                                                        2027025LL,
                                                        10321920LL,
                                                        34459425LL,
                                                        185794560LL,
                                                        654729075LL,
                                                        3715891200LL,
                                                        13749310575LL,
                                                        81749606400LL,
                                                        316234143225LL,
                                                        1961990553600LL,
                                                        7905853580625LL,
                                                        51011754393600LL,
                                                        213458046676875LL,
                                                        1428329123020800LL,
                                                        6190283353629375LL}};

static constexpr double default_shell_tolerance = 1e-10;

} // namespace detail

using PrimSize = int;
using AngularMomentum = int;
using SphericalType = int;

// Base class for all types of Shells
template <typename F> class alignas(256) Shell {

public:
  using prim_array = std::array<F, detail::shell_nprim_max>;
  using cart_array = std::array<double, 3>;

protected:
  cart_array O_;

  PrimSize nprim_;
  AngularMomentum l_;
  SphericalType pure_;

  double cutoff_radius_;
  double shell_tolerance_{detail::default_shell_tolerance};

  // double _pad_; // Pad to be a multiple of 16

  virtual void normalize() {}

  virtual void compute_shell_cutoff() { cutoff_radius_ = 0; }

public:
  Shell() : nprim_{0}, l_{0}, pure_{false} {};

  Shell(PrimSize nprim, AngularMomentum l, SphericalType pure, cart_array O,
        bool _normalize = true)
      : O_(O), nprim_(nprim), l_(l), pure_(pure) {

    if (_normalize) {
      normalize();
    }
    compute_shell_cutoff();
  }

  void set_shell_tolerance(double tol) {
    if (tol != shell_tolerance_) {
      shell_tolerance_ = tol;
      compute_shell_cutoff();
    }
  }

  ~Shell() noexcept = default;

  Shell(const Shell &) = default;
  Shell(Shell &&) noexcept = default;

  Shell &operator=(const Shell &) = default;
  Shell &operator=(Shell &&) noexcept = default;

  inline const double *O_data() const { return detail::contiguous_data(O_); }
  inline double *O_data() { return detail::contiguous_data(O_); }

  inline double cutoff_radius() const { return cutoff_radius_; }
  inline int cart_size() const { return (l_ + 1) * (l_ + 2) / 2; }
  inline int pure_size() const { return 2 * l_ + 1; }
  inline int size() const { return pure_ ? pure_size() : cart_size(); }

  inline const int &nprim() const { return nprim_; }
  inline const int &l() const { return l_; }
  inline const int &pure() const { return pure_; }
  inline const cart_array &O() const { return O_; }

  inline int &nprim() { return nprim_; }
  inline int &l() { return l_; }
  inline int &pure() { return pure_; }
  inline double &cutoff_radius() { return cutoff_radius_; }
  inline cart_array &O() { return O_; }

  inline void set_pure(bool p) { pure_ = p; }

  virtual inline int get_type() { return 0; }

  template <typename Archive> void serialize(Archive &ar) {
    ar(nprim_, l_, pure_, O_, cutoff_radius_, shell_tolerance_);
  }

  // TODO: improve this function
  virtual bool operator==(const Shell &other) const {
    if (other.nprim_ != nprim_)
      return false;
    if (other.l_ != l_)
      return false;
    if (other.pure_ != pure_)
      return false;
    if (other.O_ != O_)
      return false;

    return true;
  }
};

// Specialized GTO Shell class
template <typename F> class alignas(256) GTOShell : public Shell<F> {

public:
  using prim_array = std::array<F, detail::shell_nprim_max>;
  using cart_array = std::array<double, 3>;

private:
  prim_array alpha_;
  prim_array coeff_;

  // double _pad_; // Pad to be a multiple of 16

  // Shamelessly adapted from GAUXC which adapted from Libint...
protected:
  void normalize() {

    assert(l_ <= 15);

    constexpr auto sqrt_Pi_cubed = F{5.56832799683170784528481798212};

    const auto two_to_l = std::pow(2, this->l_);
    const auto df_term =
        two_to_l / sqrt_Pi_cubed / detail::df_Kminus1[2 * this->l_];

    for (int i = 0; i < this->nprim_; ++i) {
      assert(alpha_[i] >= 0.);
      if (alpha_[i] != 0.) {
        const auto two_alpha = 2 * alpha_[i];
        const auto two_alpha_to_am32 =
            std::pow(two_alpha, this->l_ + 1) * std::sqrt(two_alpha);
        const auto normalization_factor =
            std::sqrt(df_term * two_alpha_to_am32);

        coeff_[i] *= normalization_factor;
      }
    }

    double norm{0};
    for (int i = 0; i < this->nprim_; ++i) {
      for (int j = 0; j <= i; ++j) {
        const auto gamma = alpha_[i] + alpha_[j];
        const auto gamma_to_am32 =
            std::pow(gamma, this->l_ + 1) * std::sqrt(gamma);
        norm += (i == j ? 1 : 2) * coeff_[i] * coeff_[j] /
                (df_term * gamma_to_am32);
      }
    }

    auto normalization_factor = 1. / std::sqrt(norm);
    for (int i = 0; i < this->nprim_; ++i) {
      coeff_[i] *= normalization_factor;
    }
  }

  void compute_shell_cutoff() {
    this->cutoff_radius_ =
        util::gau_rad_cutoff(this->l_, this->nprim_, alpha_.data(),
                             coeff_.data(), this->shell_tolerance_);
  }

public:
  GTOShell() : Shell<F>() {};

  GTOShell(PrimSize nprim, AngularMomentum l, SphericalType pure,
           prim_array alpha, prim_array coeff, cart_array O,
           bool _normalize = true)
      : Shell<F>(nprim, l, pure, O, _normalize), alpha_(alpha), coeff_(coeff) {

    if (_normalize) {
      normalize();
    }
    compute_shell_cutoff();
  }

  ~GTOShell() noexcept = default;

  GTOShell(const GTOShell &) = default;
  GTOShell(GTOShell &&) noexcept = default;

  GTOShell &operator=(const GTOShell &) = default;
  GTOShell &operator=(GTOShell &&) noexcept = default;

  const F *alpha_data() const { return detail::contiguous_data(alpha_); }
  inline const F *coeff_data() const { return detail::contiguous_data(coeff_); }
  inline F *alpha_data() { return detail::contiguous_data(alpha_); }
  inline F *coeff_data() { return detail::contiguous_data(coeff_); }

  inline const prim_array &alpha() const { return alpha_; }
  inline const prim_array &coeff() const { return coeff_; }

  inline prim_array &alpha() { return alpha_; }
  inline prim_array &coeff() { return coeff_; }

  virtual inline int get_type() { return 2; }

  template <typename Archive> void serialize(Archive &ar) {
    ar(this->nprim_, this->l_, this->pure_, alpha_, coeff_, this->O_,
       this->cutoff_radius_, this->shell_tolerance_);
  }

  bool operator==(const Shell<F> &other) const { return false; };
  bool operator==(const GTOShell &other) const {
    if (other.nprim_ != this->nprim_)
      return false;
    if (other.l_ != this->l_)
      return false;
    if (other.pure_ != this->pure_)
      return false;
    if (other.O_ != this->O_)
      return false;

    for (auto i = 0; i < this->nprim_; ++i) {
      if (alpha_[i] != other.alpha_[i])
        return false;
      if (coeff_[i] != other.coeff_[i])
        return false;
    }

    return true;
  }
};

template <typename T>
inline std::ostream &operator<<(std::ostream &os, const Shell<T> &sh) {
  os << "NuKEXC::Shell:( O={" << sh.O()[0] << "," << sh.O()[1] << ","
     << sh.O()[2] << "}" << std::endl;
  os << "  ";
  os << " {l=" << sh.l() << ",sph=" << sh.pure() << "}";
  os << std::endl;

  return os;
}

template <typename T>
inline std::ostream &operator<<(std::ostream &os, const GTOShell<T> &sh) {
  os << "NuKEXC::Shell:( O={" << sh.O()[0] << "," << sh.O()[1] << ","
     << sh.O()[2] << "}" << std::endl;
  os << "  ";
  os << " {l=" << sh.l() << ",sph=" << sh.pure() << "}";
  os << std::endl;

  for (auto i = 0ul; i < sh.nprim(); ++i) {
    os << "  " << sh.alpha()[i];
    os << " " << sh.coeff().at(i);
    os << std::endl;
  }

  return os;
}
} // namespace NuKEXC
