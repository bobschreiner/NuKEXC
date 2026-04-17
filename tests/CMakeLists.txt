include (nukexc-catch2)
include (nukexc-integratorxx)
include (nukexc-kokkos)

add_executable(test_grid standards.cxx test_grid.cxx )
add_executable(test_partition standards.cxx test_partition.cxx )
add_executable(test_spherical_harmonics test_spherical_harmonics.cxx)
add_executable(test_molecule test_molecule.cxx)
add_executable(benchmark_partition  standards.cxx benchmark_partition.cxx )


target_link_libraries(test_grid Catch2::Catch2 IntegratorXX::IntegratorXX Kokkos::kokkos)
target_link_libraries(test_partition Catch2::Catch2 IntegratorXX::IntegratorXX Kokkos::kokkos)
target_link_libraries(benchmark_partition Catch2::Catch2 IntegratorXX::IntegratorXX Kokkos::kokkos)
target_link_libraries(test_spherical_harmonics Catch2::Catch2 IntegratorXX::IntegratorXX Kokkos::kokkos)
target_link_libraries(test_molecule Kokkos::kokkos)

add_custom_command(TARGET test_molecule PRE_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy_directory
                       ${CMAKE_SOURCE_DIR}/input $<TARGET_FILE_DIR:test_molecule>/input)


	       

add_test( NAME GRID_TEST COMMAND $<TARGET_FILE:test_grid> )
add_test( NAME PARTITION_TEST COMMAND $<TARGET_FILE:test_partition> )
add_test( NAME HARMONICS_TEST COMMAND $<TARGET_FILE:test_spherical_harmonics> )
add_test( NAME MOLECULE_TEST COMMAND $<TARGET_FILE:test_molecule> )
