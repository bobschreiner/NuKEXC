# Grid-convergence studies. Each binary sweeps a grid parameter and writes a
# convergence_*.csv next to itself; the Python scripts in this directory turn
# those CSVs into figures (see convergence_studies/README.md).
#
# Catch2/Armadillo/OOO and NUKEXC_TEST_SUPPORT_DIR come from the top-level
# CMakeLists; the shared driver helpers (standards.*, test_io.*,
# standalone_helpers.hpp -- the last pulled in by scf_driver.hpp) still live in
# tests/ and are compiled/included from there.

set(SUPPORT ${NUKEXC_TEST_SUPPORT_DIR})

add_executable(convergence_integration convergence_integration.cxx ${SUPPORT}/standards.cxx ${SUPPORT}/test_io.cxx)
add_executable(convergence_h2plus      convergence_h2plus.cxx)
add_executable(convergence_coreH       convergence_coreH.cxx)
add_executable(convergence_2d          convergence_2d.cxx)
add_executable(convergence_radial_h    convergence_radial_h.cxx)
add_executable(convergence_adaptive    convergence_adaptive.cxx ${SUPPORT}/standards.cxx)
add_executable(convergence_pruning     convergence_pruning.cxx  ${SUPPORT}/standards.cxx)

set(ALL_CONVERGENCE_TARGETS
    convergence_integration
    convergence_h2plus
    convergence_coreH
    convergence_2d
    convergence_radial_h
    convergence_adaptive
    convergence_pruning
)

# The studies resolve their data with paths relative to the working directory
# ("input/zorabasis/QZ4P", ...), so this build directory needs its own copy of
# input/ -- both for manual runs and for the two ctest entries below, which
# run with WORKING_DIRECTORY = ${CMAKE_CURRENT_BINARY_DIR}.
add_custom_target(copy_convergence_inputs
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_SOURCE_DIR}/input" "${CMAKE_CURRENT_BINARY_DIR}/input"
    COMMENT "Ensuring input directory is available in the convergence_studies build folder"
    )

foreach(tgt IN LISTS ALL_CONVERGENCE_TARGETS)
    target_link_libraries(${tgt} PRIVATE libnukexc Catch2::Catch2)
    target_include_directories(${tgt} PRIVATE ${SUPPORT})
    add_dependencies(${tgt} copy_convergence_inputs)
endforeach()

nukexc_ptxas_verbose(${ALL_CONVERGENCE_TARGETS})

# The two studies that assert on their result are also regression tests.
add_test( NAME H2PLUS_CONVERGENCE_TEST COMMAND $<TARGET_FILE:convergence_h2plus> )
add_test( NAME COREH_CONVERGENCE_TEST  COMMAND $<TARGET_FILE:convergence_coreH> )
