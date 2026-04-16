/**
 * GauXC Copyright (c) 2020-2024, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy).
 *
 * (c) 2024-2025, Microsoft Corporation
 * (edited) 2026 Bob Schreiner
 *
 * All rights reserved.
 *
 * See https://github.com/wavefunction91/GauXC LICENSE.txt for details
 */

#pragma once

#include <numeric>
#include <type_traits>
#include <vector>

#include "shell.hpp"

namespace NuKEXC {

/**
 *  @brief A class to manage a Gaussian type orbital (GTO) basis set
 *
 *  Extends std::vector<GTOShell<F>>
 *
 *  @tparam F Datatype representing the internal basis set storage
 */
template <typename F> struct GTOBasisSet : public std::vector<GTOShell<F>> {
private:
  /// Tests if the base class can be constructed from @p Args
  template <typename... Args>
  static constexpr auto can_construct_base_v =
      std::is_constructible<std::vector<GTOShell<F>>, Args...>::value;

public:
  /**
   *  @brief Construct a GTOBasisSet object
   *
   *  Delegates to std::vector<GTOShell<F>>::vector
   *
   *  @tparam Args Parameter pack for arguments that are passed to
   *  base constructor
   *  @tparam <anonymous> Used to disable this method via SFINAE if the base
   *  class can not be constructed from @p Args
   */
  template <typename... Args,
            typename = std::enable_if_t<can_construct_base_v<Args...>>>
  explicit GTOBasisSet(Args &&...args)
      : std::vector<GTOShell<F>>(std::forward<Args>(args)...) {}

  /// Copy a GTOBasisSet object
  GTOBasisSet(const GTOBasisSet &) = default;

  /// Move a GTOBasisSet object
  GTOBasisSet(GTOBasisSet &&) noexcept = default;

  /// Copy-assign GTOBasisSet object
  GTOBasisSet &operator=(const GTOBasisSet &) = default;

  /// Move-assign GTOBasisSet object
  GTOBasisSet &operator=(GTOBasisSet &&) noexcept = default;

  /**
   *  @brief Return the number of GTO shells which comprise the GTOBasisSet
   * object
   *
   *  Delegates to std::vector<GTOShell<F>>::size
   *
   *  @returns the number of GTO shells which comprise the GTOBasisSet object
   */
  inline int nshells() const { return this->size(); };

  /**
   *  @brief Return the number of GTO basis functions which comprise the
   *  GTOBasisSet object.
   *
   *  This routine accumulates the shell sizes (accounting for Cart/Sph angular
   *  factors) for each shell in the basis set.
   *
   *  @returns the number of GTO basis functions which comprise the GTOBasisSet
   *  object.
   */
  inline int nbf() const {
    return std::accumulate(
        this->cbegin(), this->cend(), 0ul,
        [](const auto &a, const auto &b) { return a + b.size(); });
  };

  /**
   *  @brief Return the number of cartesian GTO basis functions which comprise
   * the GTOBasisSet object.
   *
   *  This routine accumulates the cartesian shell sizes for each shell in the
   * basis set.
   *
   *  @returns the number of cartesian GTO basis functions which comprise the
   * GTOBasisSet object.
   */
  inline int nbf_cart() const {
    return std::accumulate(
        this->cbegin(), this->cend(), 0ul,
        [](const auto &a, const auto &b) { return a + b.cart_size(); });
  };

  /**
   *  @brief Determine the number of basis functions contained in a
   *  specified subset of the GTOBasisSet object.
   *
   *  Performs the following operation:
   *    for( i in shell_list ) nbf += size of shell i
   *
   *  @tparam IntegralIterator Iterator type representing the list of
   *  shell indices.
   *
   *  @param[in] shell_list_begin Start iterator for shell list
   *  @param[in] shell_list_end   End iterator for shell_list
   *  @returns   Number of basis functions in the specified shell subset.
   */
  template <typename IntegralIterator>
  inline int nbf_subset(IntegralIterator shell_list_begin,
                        IntegralIterator shell_list_end) const {
    int _nbf = 0;
    for (auto it = shell_list_begin; it != shell_list_end; ++it)
      _nbf += std::vector<GTOShell<F>>::at(*it).size();
    return _nbf;
  }

  /**
   *  @brief Determine the number of cartesian basis functions contained in a
   *  specified subset of the GTOBasisSet object.
   *
   *  Performs the following operation:
   *    for( i in shell_list ) nbf += cartesian size of shell i
   *
   *  @tparam IntegralIterator Iterator type representing the list of
   *  shell indices.
   *
   *  @param[in] shell_list_begin Start iterator for shell list
   *  @param[in] shell_list_end   End iterator for shell_list
   *  @returns   Number of cartesian basis functions in the specified shell
   * subset.
   */
  template <typename IntegralIterator>
  inline int nbf_cart_subset(IntegralIterator shell_list_begin,
                             IntegralIterator shell_list_end) const {
    int _nbf = 0;
    for (auto it = shell_list_begin; it != shell_list_end; ++it)
      _nbf += std::vector<GTOShell<F>>::at(*it).cart_size();
    return _nbf;
  }

  inline int max_l() const {
    return std::max_element(
               this->cbegin(), this->cend(),
               [](const auto &a, const auto &b) { return a.l() < b.l(); })
        ->l();
  }

}; // class GTOBasisSet

} // namespace NuKEXC
