include (nukexc-catch2)
include (nukexc-armadillo)
include (nukexc-openorbitaloptimizer)  

add_executable(test_grid standards.cxx test_grid.cxx )
add_executable(test_partition standards.cxx test_partition.cxx )
add_executable(test_spherical_harmonics test_spherical_harmonics.cxx)
add_executable(test_molecule test_molecule.cxx)
add_executable(test_basisset test_basisset.cxx)
add_executable(test_integration test_integration.cxx standards.cxx)
add_executable(test_hydrogen test_hydrogen.cxx)
add_executable(test_arborx test_arborx.cxx standards.cxx)
add_executable(test_neighborlist test_neighborlist.cxx standards.cxx)
add_executable(test_potential test_potential.cxx)
add_executable(test_density test_density.cxx)
add_executable(test_libxc test_libxc.cxx)
add_executable(test_helium test_helium.cxx)

add_executable(convergence_integration convergence_integration.cxx standards.cxx)
add_executable(convergence_h2plus convergence_h2plus.cxx)
add_executable(convergence_coreH convergence_coreH.cxx)

add_executable(benchmark_partition  standards.cxx benchmark_partition.cxx )
add_executable(benchmark_scf standards.cxx benchmark_scf.cxx )
add_executable(benchmark_coreH standards.cxx benchmark_coreH.cxx )
add_executable(benchmark_cutoff standards.cxx benchmark_cutoff.cxx )

add_executable(standalone_hf standards.cxx standalone_hf.cxx)
add_executable(standalone_dft standards.cxx standalone_dft.cxx)

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
    test_neighborlist
    test_potential
    test_density
    test_libxc
    test_helium

    convergence_integration
    convergence_h2plus
    convergence_coreH

    benchmark_partition
    benchmark_scf
    benchmark_coreH
    benchmark_cutoff

    standalone_hf 
    standalone_dft
)

# Apply linking to all of them at once
foreach(tgt IN LISTS ALL_TEST_TARGETS)
    target_link_libraries(${tgt} PRIVATE libnukexc Catch2::Catch2)

    add_dependencies(${tgt} copy_test_inputs)
endforeach()

target_link_libraries( standalone_hf PRIVATE OpenOrbitalOptimizer::OpenOrbitalOptimizer )

add_test( NAME GRID_TEST COMMAND $<TARGET_FILE:test_grid> )
add_test( NAME PARTITION_TEST COMMAND $<TARGET_FILE:test_partition> )
add_test( NAME HARMONICS_TEST COMMAND $<TARGET_FILE:test_spherical_harmonics> )
add_test( NAME MOLECULE_TEST COMMAND $<TARGET_FILE:test_molecule> )
add_test( NAME BASISSET_TEST COMMAND $<TARGET_FILE:test_basisset> )
add_test( NAME INTEGRATION_TEST COMMAND $<TARGET_FILE:test_integration> )
add_test( NAME HYDROGEN_TEST COMMAND $<TARGET_FILE:test_hydrogen> )
add_test( NAME NEIGHBORLIST_TEST COMMAND $<TARGET_FILE:test_neighborlist> )
add_test( NAME H2PLUS_CONVERGENCE_TEST COMMAND $<TARGET_FILE:convergence_h2plus> )
add_test( NAME COREH_CONVERGENCE_TEST COMMAND $<TARGET_FILE:convergence_coreH> )
add_test( NAME POTENTIAL_TEST COMMAND $<TARGET_FILE:test_potential> )
add_test( NAME DENSITY_TEST COMMAND $<TARGET_FILE:test_density> )
add_test( NAME LIBXC_TEST COMMAND $<TARGET_FILE:test_libxc> )
add_test( NAME HELIUM_TEST COMMAND $<TARGET_FILE:test_helium> )

