include (nukexc-catch2)
include (nukexc-integratorxx)
include (nukexc-kokkos)

add_executable(test_grid standards.cxx test_grid.cxx )
add_executable(test_partition standards.cxx test_partition.cxx )
add_executable(test_spherical_harmonics test_spherical_harmonics.cxx)
add_executable(test_molecule test_molecule.cxx)
add_executable(test_basisset test_basisset.cxx)
add_executable(test_integration test_integration.cxx standards.cxx)
add_executable(test_hydrogen test_hydrogen.cxx)

add_executable(convergence_integration convergence_integration.cxx standards.cxx)
add_executable(convergence_h2plus convergence_h2plus.cxx)

add_executable(benchmark_partition  standards.cxx benchmark_partition.cxx )


# Create a list of all your test targets
set(ALL_TEST_TARGETS 
    test_grid 
    test_partition 
    test_spherical_harmonics 
    test_molecule 
    test_basisset 
    test_integration 
    test_hydrogen
    convergence_integration
    convergence_h2plus
    benchmark_partition

)

# Apply linking to all of them at once
foreach(tgt IN LISTS ALL_TEST_TARGETS)
    target_link_libraries(${tgt} PRIVATE libnukexc Catch2::Catch2)
endforeach()


add_custom_command(TARGET test_molecule PRE_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy_directory
                       ${CMAKE_SOURCE_DIR}/input $<TARGET_FILE_DIR:test_molecule>/input)


	       

add_test( NAME GRID_TEST COMMAND $<TARGET_FILE:test_grid> )
add_test( NAME PARTITION_TEST COMMAND $<TARGET_FILE:test_partition> )
add_test( NAME HARMONICS_TEST COMMAND $<TARGET_FILE:test_spherical_harmonics> )
add_test( NAME MOLECULE_TEST COMMAND $<TARGET_FILE:test_molecule> )
add_test( NAME BASISSET_TEST COMMAND $<TARGET_FILE:test_basisset> )
add_test( NAME INTEGRATION_TEST COMMAND $<TARGET_FILE:test_integration> )
add_test( NAME HYDROGEN_TEST COMMAND $<TARGET_FILE:test_hydrogen> )
