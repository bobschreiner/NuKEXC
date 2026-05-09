#include <ArborX.hpp>
#include <Kokkos_Core.hpp>

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  {
    using ExecutionSpace = Kokkos::DefaultExecutionSpace;
    using MemorySpace = typename ExecutionSpace::memory_space;

    int num_points = 100000;
    int points_per_tile = 512;
    int num_tiles = (num_points + points_per_tile - 1) / points_per_tile;

    // 1. Setup your points in a Kokkos View
    Kokkos::View<ArborX::Point<3> *, MemorySpace> points("points", num_points);

    // 2. Sort points by Morton Code (Z-curve) to create spatial tiles
    Kokkos::View<int *, MemorySpace> permutation("permutation", num_points);
    ArborX::Details::MortonCodeSorter<MemorySpace> sorter(ExecutionSpace{},
                                                          points);
    sorter.sort(ExecutionSpace{}, points, permutation);

    // 3. Compute AABBs for each 512-point tile
    Kokkos::View<ArborX::Box *, MemorySpace> tile_boxes("tile_boxes",
                                                        num_tiles);
    Kokkos::parallel_for(
        "compute_tiles", Kokkos::RangePolicy<ExecutionSpace>(0, num_tiles),
        KOKKOS_LAMBDA(int i) {
          ArborX::Box box;
          for (int j = 0; j < points_per_tile; ++j) {
            int idx = i * points_per_tile + j;
            if (idx < num_points) {
              box.insert(points(permutation(idx)));
            }
          }
          tile_boxes(i) = box;
        });

    // 4. Build a BVH of the TILE BOXES
    ArborX::BVH<MemorySpace> bvh(ExecutionSpace{}, tile_boxes);

    // 5. Define a Sphere and Intersect Query
    ArborX::Sphere sphere{{0.5, 0.5, 0.5}, 0.1}; // Center and Radius
    auto query = ArborX::Experimental::make_intersects(sphere);

    // 6. Perform the search
    // indices: which tiles intersect; offset: used to parse results
    Kokkos::View<int *, MemorySpace> indices("indices", 0);
    Kokkos::View<int *, MemorySpace> offset("offset", 0);
    bvh.query(ExecutionSpace{},
              Kokkos::View<decltype(query) *, MemorySpace>("q", 1), indices,
              offset);

    printf("Found %d tiles intersecting the sphere.\n", indices.extent(0));
  }
  Kokkos::finalize();
  return 0;
}
