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

#include "standards.hpp"
#include <catch2/catch_all.hpp>
#include <catch2/catch_test_macros.hpp>
#include <nukexc/nukexc_config.hpp>
#include <nukexc/octree.hpp>
#include <nukexc/stobasis.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

using namespace Nukexc;

// ── Helpers ──────────────────────────────────────────────────────────────────

// Brute-force reference: for each box, collect all basis indices whose
// cutoff sphere intersects the box.  A sphere intersects a box iff the
// distance from the sphere center to the nearest point in the box is less
// than the cutoff radius.
static double point_box_dist_sq(const Point &p, const Box &b) {
  double d2 = 0.0;
  for (int k = 0; k < 3; ++k) {
    double lo = b.minCorner()._coords[k];
    double hi = b.maxCorner()._coords[k];
    double c = p._coords[k];
    double e = std::max(lo - c, 0.0) + std::max(c - hi, 0.0);
    d2 += e * e;
  }
  return d2;
}

static std::vector<std::vector<int>>
brute_force_neighbors(const std::vector<Box> &boxes,
                      const std::vector<Point> &origins,
                      const std::vector<double> &radii) {
  std::vector<std::vector<int>> result(boxes.size());
  for (size_t b = 0; b < boxes.size(); ++b)
    for (size_t i = 0; i < origins.size(); ++i)
      if (point_box_dist_sq(origins[i], boxes[b]) < radii[i] * radii[i])
        result[b].push_back(static_cast<int>(i));
  return result;
}

void create_histogram(std::vector<double> percentages, int num_bins,
                      const std::string &output_stem = "histogram") {
  if (percentages.empty() || num_bins <= 0)
    return;

  // 1. Find data boundaries
  auto [min_it, max_it] =
      std::minmax_element(percentages.begin(), percentages.end());
  double min_val = *min_it;
  double max_val = *max_it;
  if (std::abs(max_val - min_val) < 1e-9)
    max_val += 1.0;

  // 2. Count frequencies into bins
  double bin_width = (max_val - min_val) / num_bins;
  std::vector<int> bins(num_bins, 0);
  for (double val : percentages) {
    int bin_idx = static_cast<int>((val - min_val) / bin_width);
    if (bin_idx >= num_bins)
      bin_idx = num_bins - 1;
    if (bin_idx < 0)
      bin_idx = 0;
    bins[bin_idx]++;
  }

  // 3. ASCII output (unchanged)
  int max_freq = *std::max_element(bins.begin(), bins.end());
  const int max_bar_width = 40;
  std::cout << "\n=== Histogram ===\n";
  for (int i = 0; i < num_bins; ++i) {
    double lo = min_val + i * bin_width;
    double hi = lo + bin_width;
    std::printf("[%6.2f%% - %6.2f%%]: ", lo, hi);
    int bar_length = (max_freq > max_bar_width)
                         ? (bins[i] * max_bar_width) / max_freq
                         : bins[i];
    std::cout << std::string(bar_length, '#') << " (" << bins[i] << ")\n";
  }
  std::cout << "=================\n";

  // 4. Write .dat file  (bin_center  count)
  //    pgfplots reads this with \addplot table
  const std::string dat_path = output_stem + ".dat";
  std::ofstream dat(dat_path);
  if (!dat) {
    std::cerr << "Could not open " << dat_path << "\n";
    return;
  }

  dat << "% bin_start bin_end bin_center count\n";
  for (int i = 0; i < num_bins; ++i) {
    double lo = min_val + i * bin_width;
    double hi = lo + bin_width;
    double mid = (lo + hi) / 2.0;
    dat << std::fixed << std::setprecision(6) << lo << " " << hi << " " << mid
        << " " << bins[i] << "\n";
  }
  dat.close();

  // 5. Write self-contained .tex snippet (no external .dat needed)
  const std::string tex_path = output_stem + ".tex";
  std::ofstream tex(tex_path);
  if (!tex) {
    std::cerr << "Could not open " << tex_path << "\n";
    return;
  }

  // Compute a clean tick spacing: aim for ~8 ticks across the range
  // Round up to a "nice" number (1, 2, 2.5, 5, 10, 20, ...)
  auto nice_step = [](double raw) -> double {
    double exp = std::pow(10.0, std::floor(std::log10(raw)));
    double f = raw / exp;
    if (f <= 1.0)
      return 1.0 * exp;
    else if (f <= 2.0)
      return 2.0 * exp;
    else if (f <= 2.5)
      return 2.5 * exp;
    else if (f <= 5.0)
      return 5.0 * exp;
    else
      return 10.0 * exp;
  };

  double tick_step = nice_step((max_val - min_val) / 8.0);
  // First tick: round min_val up to nearest multiple of tick_step
  double first_tick = std::ceil(min_val / tick_step) * tick_step;

  // Build the explicit tick list
  std::ostringstream tick_list;
  for (double t = first_tick; t <= max_val + 1e-9; t += tick_step)
    tick_list << std::fixed << std::setprecision(1) << t << ",";
  std::string ticks = tick_list.str();
  if (!ticks.empty())
    ticks.pop_back(); // remove trailing comma

  const double bar_width_frac = 0.92;
  tex << std::fixed << std::setprecision(4);
  tex << "% Auto-generated by create_histogram() -- do not edit by hand\n"
      << "\\begin{tikzpicture}\n"
      << "\\begin{axis}[\n"
      << "  ybar,\n"
      << "  bar width=" << bin_width * bar_width_frac << "pt,\n"
      << "  xlabel={Value (\\%)},\n"
      << "  ylabel={Count},\n"
      << "  xtick={" << ticks << "},\n" // <-- sparse, clean ticks
      << "  xticklabel={$\\pgfmathprintnumber"
         "[fixed,precision=1]{\\tick}\\%%$},\n" // <-- math mode, no rotation
      << "  ymin=0,\n"
      << "  enlarge x limits=0.05,\n"
      << "  grid=major,\n"
      << "  grid style={dashed, gray!40},\n"
      << "  width=0.85\\linewidth,\n"
      << "  height=6cm,\n"
      << "]\n"
      << "\\addplot[fill=blue!40, draw=blue!70] coordinates {\n";

    for (int i = 0; i < num_bins; ++i) {
    double lo = min_val + i * bin_width;
    double mid = lo + bin_width / 2.0;
    tex << "  (" << mid << ", " << bins[i] << ")\n";
  }

  tex << "};\n"
      << "\\end{axis}\n"
      << "\\end{tikzpicture}\n";
  tex.close();

  std::cout << "LaTeX snippet written: " << tex_path << "\n";
  std::cout << "LaTeX files written: " << dat_path << " and " << tex_path
            << "\n";
}

// ── Fixtures ─────────────────────────────────────────────────────────────────

// Place basis functions on a regular 3-D grid and build a small set of
// non-overlapping boxes.  This gives us full control over which intersections
// should exist.
struct SyntheticFixture {
  std::vector<Point> O_v;
  std::vector<double> cutoff_radii_v;
  std::vector<Box> boxes_v;

  Kokkos::View<Point *, ExecSpace> O;
  Kokkos::View<double *, ExecSpace> cutoff_radii;
  Kokkos::View<Box *, ExecSpace> boxes;

  SyntheticFixture() {
    // 3x3x3 basis origins spaced 2 bohr apart, all with radius 1.2 bohr.
    // Neighbouring origins are 2 bohr apart, so each sphere just misses
    // its immediate neighbour's box.
    for (int ix = 0; ix < 3; ++ix)
      for (int iy = 0; iy < 3; ++iy)
        for (int iz = 0; iz < 3; ++iz) {
          O_v.push_back({static_cast<double>(ix * 2),
                         static_cast<double>(iy * 2),
                         static_cast<double>(iz * 2)});
          cutoff_radii_v.push_back(1.2);
        }

    // Four non-overlapping unit boxes placed at predictable positions.
    //  box 0: [0,1]^3     — contains origin (0,0,0), sphere radius 1.2
    //  box 1: [3,4]^3     — midpoint between grid points, no origin inside
    //  box 2: [-2,-1]^3   — well outside all spheres
    //  box 3: [0.5,1.5]^3 — straddles origin (0,0,0) and (2,0,0) etc.
    auto make_box = [](double x0, double y0, double z0, double x1, double y1,
                       double z1) {
      return Box{Point{x0, y0, z0}, Point{x1, y1, z1}};
    };
    boxes_v.push_back(make_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0));
    boxes_v.push_back(make_box(3.0, 3.0, 3.0, 4.0, 4.0, 4.0));
    boxes_v.push_back(make_box(-2.0, -2.0, -2.0, -1.0, -1.0, -1.0));
    boxes_v.push_back(make_box(0.5, 0.5, 0.5, 1.5, 1.5, 1.5));

    // Upload to device
    O = Kokkos::View<Point *, ExecSpace>("origins", O_v.size());
    cutoff_radii =
        Kokkos::View<double *, ExecSpace>("radii", cutoff_radii_v.size());
    boxes = Kokkos::View<Box *, ExecSpace>("boxes", boxes_v.size());

    auto o_h = Kokkos::create_mirror_view(O);
    auto r_h = Kokkos::create_mirror_view(cutoff_radii);
    auto b_h = Kokkos::create_mirror_view(boxes);

    for (size_t i = 0; i < O_v.size(); ++i)
      o_h(i) = O_v[i];
    for (size_t i = 0; i < cutoff_radii_v.size(); ++i)
      r_h(i) = cutoff_radii_v[i];
    for (size_t i = 0; i < boxes_v.size(); ++i)
      b_h(i) = boxes_v[i];

    Kokkos::deep_copy(O, o_h);
    Kokkos::deep_copy(cutoff_radii, r_h);
    Kokkos::deep_copy(boxes, b_h);
  }
};

// ── Helper: unpack NeighborList to hot ─────────────────────────

static std::vector<std::vector<int>> unpack(const NeighborList &nl,
                                            int num_boxes) {
  auto neighbors_h =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, nl.neighbors);
  auto offsets_h =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, nl.offsets);

  std::vector<std::vector<int>> result(num_boxes);
  for (int b = 0; b < num_boxes; ++b) {
    for (int k = offsets_h(b); k < offsets_h(b + 1); ++k)
      result[b].push_back(neighbors_h(k));
    std::sort(result[b].begin(), result[b].end());
  }
  return result;
}

// ── Tests ───────────────────────────────────────────────────────

TEST_CASE("NeighborList: no false negatives", "[octree][neighborlist]") {
  SyntheticFixture f;
  auto ref = brute_force_neighbors(f.boxes_v, f.O_v, f.cutoff_radii_v);

  NeighborList nl;
  build_neighbor_list(f, f.boxes, 0, 0, nl);
  auto got = unpack(nl, static_cast<int>(f.boxes_v.size()));

  // Every index in the reference must appear in the ArborX result.
  // (ArborX is allowed to return conservative false positives.)
  for (size_t b = 0; b < f.boxes_v.size(); ++b) {
    for (int idx : ref[b]) {
      INFO("box " << b << " missing basis function " << idx);
      CHECK(std::binary_search(got[b].begin(), got[b].end(), idx));
    }
  }
}

TEST_CASE("NeighborList: empty box returns no neighbors",
          "[octree][neighborlist]") {
  SyntheticFixture f;

  // Box far from all origins — should have no intersecting spheres.
  Box far_box{Point{100.0, 100.0, 100.0}, Point{101.0, 101.0, 101.0}};
  Kokkos::View<Box *, ExecSpace> boxes_dev("boxes", 1);
  auto b_h = Kokkos::create_mirror_view(boxes_dev);
  b_h(0) = far_box;
  Kokkos::deep_copy(boxes_dev, b_h);

  NeighborList nl;
  build_neighbor_list(f, boxes_dev, 1, 2, nl);
  auto got = unpack(nl, 1);
  CHECK(got[0].empty());
}

TEST_CASE("NeighborList: offsets are well-formed CSR",
          "[octree][neighborlist]") {
  SyntheticFixture f;
  NeighborList nl;
  build_neighbor_list(f, f.boxes, 0, 0, nl);

  auto offsets_h =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, nl.offsets);

  const int num_boxes = static_cast<int>(f.boxes_v.size());

  // offsets has exactly num_boxes+1 entries
  CHECK(static_cast<int>(nl.offsets.extent(0)) == num_boxes + 1);
  // first entry is zero
  CHECK(offsets_h(0) == 0);
  // monotonically non-decreasing
  for (int b = 0; b < num_boxes; ++b)
    CHECK(offsets_h(b + 1) >= offsets_h(b));
  // last entry equals total neighbor count
  CHECK(offsets_h(num_boxes) == static_cast<int>(nl.neighbors.extent(0)));
}

TEST_CASE("NeighborList: benezene regression",
          "[octree][benezene][regression]") {
  // Checks that the neighbor list on a real molecule is strictly smaller
  // than the dense N*num_boxes product (i.e. screening is actually happening).
  using ta_type = IntegratorXX::TreutlerAhlrichs<double, double>;
  using ll_type = IntegratorXX::LebedevLaikov<double>;

  Molecule mol = make_benzene();
  STOBasisSet basis = load_adf_basis(mol, "input/zorabasis/QZ4P", 1e-6);
  FlatGrid grid = make_flat_grid<ta_type, ll_type>(mol, 30, 30);

  const int points_per_box = 8;

  //  auto bb = create_bounding_boxes(grid, points_per_box);
  auto bb = create_bounding_boxes(grid, points_per_box);

  NeighborList nl;
  build_neighbor_list(basis, bb, points_per_box, grid.quad_points.extent(0),
                      nl);

  auto offsets_h =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, nl.offsets);

  const int64_t num_boxes = static_cast<int64_t>(bb.extent(0));
  const int64_t N = static_cast<int64_t>(basis.nbf());
  const int64_t dense = num_boxes * N;
  const int64_t screened = offsets_h(num_boxes);
  const double sparsity = 1.0 - static_cast<double>(screened) / dense;

  std::cout << "[benzene regression]\n"
            << "  boxes    : " << num_boxes << '\n'
            << "  basis    : " << N << '\n'
            << "  dense    : " << dense << '\n'
            << "  screened : " << screened << '\n'
            << "  sparsity : " << std::fixed << std::setprecision(1)
            << sparsity * 100 << "%\n";

  std::vector<double> percentages;
  for (int b = 0; b < num_boxes; ++b) {
    CHECK(offsets_h(b + 1) >= offsets_h(b));
    percentages.push_back((offsets_h(b + 1) - offsets_h(b)) / (double)N * 100);
  }

  create_histogram(percentages, 50, "benzene_histogram");
}

TEST_CASE("NeighborList: taxol regression",
          "[octree][neighborlist][regression]") {
  // Checks that the neighbor list on a real molecule is strictly smaller
  // than the dense N*num_boxes product (i.e. screening is actually happening).
  using ta_type = IntegratorXX::TreutlerAhlrichs<double, double>;
  using ll_type = IntegratorXX::LebedevLaikov<double>;

  Molecule mol = make_taxol();
  STOBasisSet basis = load_adf_basis(mol, "input/zorabasis/QZ4P", 1e-6);
  FlatGrid grid = make_flat_grid<ta_type, ll_type>(mol, 30, 30);

  const int points_per_box = 8;

  auto bb = create_bounding_boxes(grid, points_per_box);
  //  auto bb = create_bounding_boxes(grid, points_per_box);
  NeighborList nl;
  build_neighbor_list(basis, bb, points_per_box, grid.quad_points.extent(0),
                      nl);

  auto offsets_h =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, nl.offsets);

  const int64_t num_boxes = static_cast<int64_t>(bb.extent(0));
  const int64_t N = static_cast<int64_t>(basis.nbf());
  const int64_t dense = num_boxes * N;
  const int64_t screened = offsets_h(num_boxes);
  const double sparsity = 1.0 - static_cast<double>(screened) / dense;

  std::cout << "[taxol regression]\n"
            << "  boxes    : " << num_boxes << '\n'
            << "  basis    : " << N << '\n'
            << "  dense    : " << dense << '\n'
            << "  screened : " << screened << '\n'
            << "  sparsity : " << std::fixed << std::setprecision(1)
            << sparsity * 100 << "%\n";

  // For a large molecule with 1e-8 cutoff we expect >50% sparsity.
  CHECK(sparsity > 0.3);

  std::vector<double> percentages;
  for (int b = 0; b < num_boxes; ++b) {
    CHECK(offsets_h(b + 1) >= offsets_h(b));
    percentages.push_back((offsets_h(b + 1) - offsets_h(b)) / (double)N * 100);
  }

  create_histogram(percentages, 50, "taxol_histogram");
}

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  {
    int result = Catch::Session().run(argc, argv);
  }
  Kokkos::finalize();
  return 0;
}
