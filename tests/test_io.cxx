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

#include "test_io.hpp"

#include <cstddef>
#include <iomanip>
#include <iostream>

bool ArgParser::string_opt(const std::string &prefix, std::string &out) const {
  if (arg.rfind(prefix, 0) == 0) {
    out = arg.substr(prefix.size());
    return true;
  }
  return false;
}

bool ArgParser::int_opt(const std::string &prefix, int &out) const {
  if (arg.rfind(prefix, 0) == 0) {
    out = std::stoi(arg.substr(prefix.size()));
    return true;
  }
  return false;
}

bool ArgParser::double_opt(const std::string &prefix, double &out) const {
  if (arg.rfind(prefix, 0) == 0) {
    out = std::stod(arg.substr(prefix.size()));
    return true;
  }
  return false;
}

std::string repeat(const std::string &s, int n) {
  std::string r;
  for (int i = 0; i < n; ++i)
    r += s;
  return r;
}

void print_config_box(
    const std::string &title,
    const std::vector<std::pair<std::string, std::string>> &rows) {
  std::size_t label_w = 0;
  std::size_t value_w = 0;
  for (const auto &r : rows) {
    label_w = std::max(label_w, r.first.size());
    value_w = std::max(value_w, r.second.size());
  }
  value_w = std::max<std::size_t>(value_w, 20);

  const int label_cell = static_cast<int>(label_w) + 2; // one space each side
  const int value_cell = static_cast<int>(value_w) + 2;
  const int full = label_cell + 1 + value_cell; // + column divider

  // Centre the title across the full inner width.
  const int title_len = static_cast<int>(title.size());
  const int pad = full > title_len ? full - title_len : 0;
  const int pad_l = pad / 2;
  const int pad_r = pad - pad_l;

  std::cout << "\n";
  std::cout << "┌" << repeat("─", full) << "┐\n";
  std::cout << "│" << repeat(" ", pad_l) << title << repeat(" ", pad_r)
            << "│\n";
  std::cout << "├" << repeat("─", label_cell) << "┬" << repeat("─", value_cell)
            << "┤\n";
  for (const auto &r : rows) {
    std::cout << "│ " << std::left << std::setw(static_cast<int>(label_w))
              << r.first << " │ " << std::left
              << std::setw(static_cast<int>(value_w)) << r.second << " │\n";
  }
  std::cout << "└" << repeat("─", label_cell) << "┴" << repeat("─", value_cell)
            << "┘\n\n";
  std::cout << std::flush;
}
