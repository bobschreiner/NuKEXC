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
add_executable(test_arborx test_arborx.cxx standards.cxx)

add_executable(convergence_integration convergence_integration.cxx standards.cxx)
add_executable(convergence_h2plus convergence_h2plus.cxx)
add_executable(convergence_coreH convergence_coreH.cxx)

add_executable(benchmark_partition  standards.cxx benchmark_partition.cxx )
add_executable(benchmark_scf standards.cxx benchmark_scf.cxx )
add_executable(benchmark_cutoff standards.cxx benchmark_cutoff.cxx )

add_custom_target(copy_test_inputs 
    COMMAND  ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_SOURCE_DIR}/input" "${CMAKE_CURRENT_BINARY_DIR}/input"
    COMMENT "Ensuring input directory is available in the build folder"
    )

# Create a list of all your test targets
set(ALL_TEST_TARGETS 
    test_grid 
    test_partition 
    test_spherical_harmonics 
    test_molecule 
    test_basisset 
    test_integration 
    test_hydrogen
    test_arborx
    convergence_integration
    convergence_h2plus
    convergence_coreH
    benchmark_partition
    benchmark_scf
    benchmark_cutoff

)

# Apply linking to all of them at once
foreach(tgt IN LISTS ALL_TEST_TARGETS)
    target_link_libraries(${tgt} PRIVATE libnukexc Catch2::Catch2)

    add_dependencies(${tgt} copy_test_inputs)
endforeach()

add_test( NAME GRID_TEST COMMAND $<TARGET_FILE:test_grid> )
add_test( NAME PARTITION_TEST COMMAND $<TARGET_FILE:test_partition> )
add_test( NAME HARMONICS_TEST COMMAND $<TARGET_FILE:test_spherical_harmonics> )
add_test( NAME MOLECULE_TEST COMMAND $<TARGET_FILE:test_molecule> )
add_test( NAME BASISSET_TEST COMMAND $<TARGET_FILE:test_basisset> )
add_test( NAME INTEGRATION_TEST COMMAND $<TARGET_FILE:test_integration> )
add_test( NAME HYDROGEN_TEST COMMAND $<TARGET_FILE:test_hydrogen> )
add_test( NAME H2PLUS_CONVERGENCE_TEST COMMAND $<TARGET_FILE:convergence_h2plus> )
add_test( NAME COREH_CONVERGENCE_TEST COMMAND $<TARGET_FILE:convergence_coreH> )

