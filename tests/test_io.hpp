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

// ===========================================================================
// Shared command-line I/O helpers for the test / benchmark drivers.
//
// Every driver that takes command-line arguments used to carry its own copy of
// the same three argument-parsing lambdas, a `repeat()` helper and a hand-drawn
// "config box". Each driver still owns its own Config struct (the fields differ
// from test to test), but the boilerplate machinery now lives here.
// ===========================================================================

#pragma once

#include <algorithm>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

// ---------------------------------------------------------------------------
// One argv token, with the "--prefix=<value>" matching logic each driver used
// to duplicate as local lambdas. A driver's parse_args() constructs one
// ArgParser per token and chains the *_opt() calls, e.g.:
//
//   for (int i = 1; i < argc; ++i) {
//     ArgParser p{argv[i]};
//     if (p.arg == "--help") { ...; std::exit(0); }
//     else if (!p.string_opt("--xyz=", cfg.xyz_file) &&
//              !p.int_opt("--nrad=", cfg.nrad) &&
//              ...) {
//       throw std::runtime_error("Unknown argument: " + p.arg);
//     }
//   }
// ---------------------------------------------------------------------------
struct ArgParser {
  std::string arg;

  // If `arg` starts with `prefix`, parse the remainder into `out` and return
  // true; otherwise leave `out` untouched and return false.
  bool string_opt(const std::string &prefix, std::string &out) const;
  bool int_opt(const std::string &prefix, int &out) const;
  bool double_opt(const std::string &prefix, double &out) const;
};

// Concatenate `n` copies of `s` (used to draw the box frame lines).
std::string repeat(const std::string &s, int n);

// Render any streamable value the way std::cout would, so it can be dropped
// into a print_config_box() cell. Keeps scientific notation (e.g. "1e-06")
// instead of std::to_string's "0.000001".
template <typename T> std::string cfg_val(const T &v) {
  std::ostringstream os;
  os << v;
  return os.str();
}

// Print a titled two-column configuration box:
//
//   ┌────────────────────────────────┐
//   │           <title>              │
//   ├──────────────────┬─────────────┤
//   │ <label>          │ <value>     │
//   └──────────────────┴─────────────┘
//
// Column widths are derived from the longest label/value.
void print_config_box(
    const std::string &title,
    const std::vector<std::pair<std::string, std::string>> &rows);
