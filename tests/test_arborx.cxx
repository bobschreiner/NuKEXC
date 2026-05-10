#include "standards.hpp"

#include <ArborX.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Macros.hpp>
#include <cfloat>
#include <detail/ArborX_SpaceFillingCurves.hpp>
#include <detail/ArborX_TreeVisualization.hpp>
#include <fstream>
#include <integratorxx/quadratures/s2/lebedev_laikov.hpp>
#include <iostream>
#include <nukexc/molecule.hpp>
#include <nukexc/nukexc_config.hpp>
#include <nukexc/octree.hpp>
#include <nukexc/stobasis.hpp>

using namespace NuKEXC;
using bk_type = IntegratorXX::Becke<double, double>;
using ll_type = IntegratorXX::LebedevLaikov<double>;

// ─── Visualization helpers ──────────────────────────────────────────

// visualize_bounding_boxes
// Writes all bounding boxes to "bounding_boxes.vtk" as hexahedral cells in
// VTK Legacy ASCII format.  Open the file in ParaView and colour by:
//   • tile_id — verify spatial coherence (ids should form contiguous bands)
//   • volume  — identify oversized or degenerate tiles
void visualize_bounding_boxes(Kokkos::View<Box *> bb_dev) {
  // Mirror from device to host so we can iterate in plain C++
  auto bb = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, bb_dev);
  const int n = static_cast<int>(bb.extent(0));

  const std::string path = "bounding_boxes.vtk";
  std::ofstream out(path);
  if (!out) {
    std::cerr << "[visualize_bounding_boxes] ERROR: cannot open " << path
              << '\n';
    return;
  }

  // ── VTK Legacy ASCII header ──────────────────────────────────────────────
  out << "# vtk DataFile Version 3.0\n"
      << "Bounding Boxes\n"
      << "ASCII\n"
      << "DATASET UNSTRUCTURED_GRID\n\n";

  // ── Points: 8 corners per hexahedron ────────────────────────────────────
  out << "POINTS " << n * 8 << " rdouble\n";
  for (int i = 0; i < n; ++i) {
    const double x0 = bb(i).minCorner()._coords[0];
    const double y0 = bb(i).minCorner()._coords[1];
    const double z0 = bb(i).minCorner()._coords[2];
    const double x1 = bb(i).maxCorner()._coords[0];
    const double y1 = bb(i).maxCorner()._coords[1];
    const double z1 = bb(i).maxCorner()._coords[2];

    // VTK_HEXAHEDRON corner order:
    //   bottom face (z=z0): 0–3 counter-clockwise when viewed from below
    //   top    face (z=z1): 4–7 directly above 0–3
    out << x0 << ' ' << y0 << ' ' << z0 << '\n'  // 0
        << x1 << ' ' << y0 << ' ' << z0 << '\n'  // 1
        << x1 << ' ' << y1 << ' ' << z0 << '\n'  // 2
        << x0 << ' ' << y1 << ' ' << z0 << '\n'  // 3
        << x0 << ' ' << y0 << ' ' << z1 << '\n'  // 4
        << x1 << ' ' << y0 << ' ' << z1 << '\n'  // 5
        << x1 << ' ' << y1 << ' ' << z1 << '\n'  // 6
        << x0 << ' ' << y1 << ' ' << z1 << '\n'; // 7
  }

  // ── Cells: one VTK_HEXAHEDRON (type 12) per tile ────────────────────────
  out << "\nCELLS " << n << ' ' << n * 9 << '\n'; // 9 ints: 1 count + 8 ids
  for (int i = 0; i < n; ++i) {
    const int b = i * 8;
    out << "8 " << b + 0 << ' ' << b + 1 << ' ' << b + 2 << ' ' << b + 3 << ' '
        << b + 4 << ' ' << b + 5 << ' ' << b + 6 << ' ' << b + 7 << '\n';
  }

  out << "\nCELL_TYPES " << n << '\n';
  for (int i = 0; i < n; ++i)
    out << "12\n"; // VTK_HEXAHEDRON

  // ── Cell scalar data ─────────────────────────────────────────────────────
  out << "\nCELL_DATA " << n << '\n';

  // tile_id: colour by index — spatially coherent tiling shows bands
  out << "SCALARS tile_id int 1\n"
      << "LOOKUP_TABLE default\n";
  for (int i = 0; i < n; ++i)
    out << i << '\n';

  // volume: large values reveal inefficient (sparse/oversized) tiles
  out << "\nSCALARS volume double 1\n"
      << "LOOKUP_TABLE default\n";
  double vol_min = DBL_MAX, vol_max = 0.f, vol_sum = 0.f;
  std::vector<double> volume_v;
  for (int i = 0; i < n; ++i) {
    const double dx =
        bb(i).maxCorner()._coords[0] - bb(i).minCorner()._coords[0];
    const double dy =
        bb(i).maxCorner()._coords[1] - bb(i).minCorner()._coords[1];
    const double dz =
        bb(i).maxCorner()._coords[2] - bb(i).minCorner()._coords[2];
    const double v = dx * dy * dz;
    volume_v.push_back(v);
    out << v << '\n';
    vol_min = std::min(vol_min, v);
    vol_max = std::max(vol_max, v);
    vol_sum += v;
  }

  size_t n_half = n / 2;
  std::nth_element(volume_v.begin(), volume_v.begin() + n_half, volume_v.end());
  double median = volume_v[n_half];
  out.flush();
  std::cout << "[visualize_bounding_boxes]\n"
            << "  Tiles  : " << n << '\n'
            << "  Volume : min=" << vol_min << "  max=" << vol_max
            << "  avg=" << vol_sum / n << " median=" << median << '\n'
            << "  Output : " << path << '\n'
            << "  Tip    : open in ParaView, colour by tile_id or volume.\n\n";
}

// visualize_points_with_tiles
// Writes the Morton-sorted quadrature points to "grid_points.vtk" as a VTK
// vertex cloud, coloured by tile index.  Load alongside bounding_boxes.vtk in
// the same ParaView session to confirm that every tile's points sit inside its
// box.
void visualize_points_with_tiles(Kokkos::View<Point *, ExecSpace> points_dev,
                                 int max_pts_per_tile) {
  auto pts =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, points_dev);
  const int total = static_cast<int>(pts.extent(0));

  const std::string path = "grid_points.vtk";
  std::ofstream out(path);
  if (!out) {
    std::cerr << "[visualize_points_with_tiles] ERROR: cannot open " << path
              << '\n';
    return;
  }

  out << "# vtk DataFile Version 3.0\n"
      << "Grid Points (Morton sorted)\n"
      << "ASCII\n"
      << "DATASET UNSTRUCTURED_GRID\n\n";

  out << "POINTS " << total << " double\n";
  for (int i = 0; i < total; ++i)
    out << pts(i)._coords[0] << ' ' << pts(i)._coords[1] << ' '
        << pts(i)._coords[2] << '\n';

  // Each point is its own VTK_VERTEX cell (type 1)
  out << "\nCELLS " << total << ' ' << total * 2 << '\n';
  for (int i = 0; i < total; ++i)
    out << "1 " << i << '\n';

  out << "\nCELL_TYPES " << total << '\n';
  for (int i = 0; i < total; ++i)
    out << "1\n";

  out << "\nCELL_DATA " << total << '\n'
      << "SCALARS tile_id int 1\n"
      << "LOOKUP_TABLE default\n";
  for (int i = 0; i < total; ++i)
    out << (i / max_pts_per_tile) << '\n';

  out.flush();
  std::cout << "[visualize_points_with_tiles]\n"
            << "  Points : " << total << '\n'
            << "  Output : " << path << '\n'
            << "  Tip    : load alongside bounding_boxes.vtk in ParaView;\n"
            << "           matching tile_id colours confirm points are\n"
            << "           correctly contained within their boxes.\n\n";
}
// ─── Main --------------------------------------
int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  {
    Molecule mol = make_benzene();
    FlatGrid grid = make_flat_grid<bk_type, ll_type>(mol);
    int max_points_per_bb = 512;
    // Create bounding boxes
    auto bb = create_bounding_boxes(grid, max_points_per_bb);
    // TODO remove
    Kokkos::View<Point *, ExecSpace> points;

    // ── Visualize ────────────────────────────────────────────────────────
    visualize_bounding_boxes(bb);
    visualize_points_with_tiles(points, max_points_per_bb);
  }
  Kokkos::finalize();
  return 0;
}
