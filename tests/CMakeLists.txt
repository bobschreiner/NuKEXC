include (nukexc-catch2)
include (nukexc-integratorxx)
include (nukexc-kokkos)

add_executable(first-test test.cxx)
add_executable(test_basisset ../src/parse_basis.cxx standards.cxx test_basisset.cxx )
add_executable(test_grid ../src/parse_basis.cxx standards.cxx test_grid.cxx )
add_executable(test_partition ../src/parse_basis.cxx standards.cxx test_partition.cxx )
add_executable(benchmark_partition ../src/parse_basis.cxx standards.cxx benchmark_partition.cxx )

target_link_libraries(test_basisset Catch2::Catch2 )
target_link_libraries(test_grid Catch2::Catch2 IntegratorXX::IntegratorXX)
target_link_libraries(test_partition Catch2::Catch2 IntegratorXX::IntegratorXX Kokkos::kokkos)
target_link_libraries(benchmark_partition Catch2::Catch2 IntegratorXX::IntegratorXX Kokkos::kokkos)

add_custom_command(TARGET test_basisset PRE_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy_directory
                       ${CMAKE_SOURCE_DIR}/tests/input $<TARGET_FILE_DIR:test_basisset>/input)



add_test( NAME BASIS_TEST COMMAND $<TARGET_FILE:test_basisset> )
add_test( NAME GRID_TEST COMMAND $<TARGET_FILE:test_grid> )
add_test( NAME PARTITION_TEST COMMAND $<TARGET_FILE:test_partition> )

