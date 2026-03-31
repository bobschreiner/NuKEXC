#pragma once

#include <vector>
#include <algorithm>

#include "atom.hpp"

namespace NuKEXC {


class Molecule : public std::vector<Atom> {
private:
  /// Tests if the base class can be constructed from @p Args
  template <typename... Args>
  static constexpr auto can_construct_base_v = 
    std::is_constructible<std::vector<Atom>, Args...>::value;

public:

  template <typename... Args, 
            typename = std::enable_if_t<can_construct_base_v<Args...>>>
  Molecule( Args&&... args ) :
    std::vector<Atom>( std::forward<Args>(args)... ) { }

  Molecule( const Molecule& )          = default;
  Molecule( Molecule&&      ) noexcept = default;

  Molecule& operator=( const Molecule& other ) = default;

  size_t natoms() const { return this->size(); }

  int maxZ() const {
    return std::max_element( this->cbegin(), this->cend(),
      []( const auto& a, const auto& b) {
        return a.Z < b.Z;
      })->Z;
  }

  bool operator==(const Molecule& other) {
    if(other.size() != this->size()) return false;
    for( auto i = 0ul; i < this->size(); ++i )
      if( other[i] != operator[](i) ) return false;
    return true;
  }
};

}
