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

#pragma once
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace NuKEXC {
namespace detail {

const double ang_to_bohr = 1.8897261246;
std::map<std::string, int> am_map = {{"S", 0}, {"P", 1}, {"D", 2}, {"F", 3},
                                     {"G", 4}, {"H", 5}, {"I", 6}, {"J", 7}};

unsigned get_atomic_number(const std::string &symbol) {
  static const std::unordered_map<std::string, unsigned> pt = {
      {"H", 1},   {"HE", 2},  {"LI", 3},  {"BE", 4},  {"B", 5},   {"C", 6},
      {"N", 7},   {"O", 8},   {"F", 9},   {"NE", 10}, {"NA", 11}, {"MG", 12},
      {"AL", 13}, {"SI", 14}, {"P", 15},  {"S", 16},  {"CL", 17}, {"AR", 18},
      {"K", 19},  {"CA", 20}, {"SC", 21}, {"TI", 22}, {"V", 23},  {"CR", 24},
      {"MN", 25}, {"FE", 26}, {"CO", 27}, {"NI", 28}, {"CU", 29}, {"ZN", 30},
      {"GA", 31}, {"GE", 32}, {"AS", 33}, {"SE", 34}, {"BR", 35}, {"KR", 36},
      {"RB", 37}, {"SR", 38}, {"Y", 39},  {"ZR", 40}, {"NB", 41}, {"MO", 42},
      {"TC", 43}, {"RU", 44}, {"RH", 45}, {"PD", 46}, {"AG", 47}, {"CD", 48},
      {"IN", 49}, {"SN", 50}, {"SB", 51}, {"TE", 52}, {"I", 53},  {"XE", 54},
      {"CS", 55}, {"BA", 56}, {"LA", 57}, {"CE", 58}, {"PR", 59}, {"ND", 60},
      {"PM", 61}, {"SM", 62}, {"EU", 63}, {"GD", 64}, {"TB", 65}, {"DY", 66},
      {"HO", 67}, {"ER", 68}, {"TM", 69}, {"YB", 70}, {"LU", 71}, {"HF", 72},
      {"TA", 73}, {"W", 74},  {"RE", 75}, {"OS", 76}, {"IR", 77}, {"PT", 78},
      {"AU", 79}, {"HG", 80}, {"TL", 81}, {"PB", 82}, {"BI", 83}, {"PO", 84},
      {"AT", 85}, {"RN", 86}, {"FR", 87}, {"RA", 88}, {"AC", 89}, {"TH", 90},
      {"PA", 91}, {"U", 92},  {"NP", 93}, {"PU", 94}, {"AM", 95}, {"CM", 96}};
  auto it = pt.find(symbol);
  return (it != pt.end()) ? it->second : 0;
}
} // namespace detail
} // namespace NuKEXC
