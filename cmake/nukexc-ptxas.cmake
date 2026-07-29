# Opt-in: make ptxas print per-kernel register / shared-memory usage at compile
# time (CUDA / nvcc_wrapper only). Configure with -DNUKEXC_PTXAS_VERBOSE=ON.
# Applied per-target (not via CMAKE_CXX_FLAGS) so it never touches CMake's
# compiler-detection test program.
#
# Shared by tests/, benchmarking/ and convergence_studies/; each calls
# nukexc_ptxas_verbose() with its own list of targets.

option(NUKEXC_PTXAS_VERBOSE
       "Print ptxas per-kernel register/smem usage (CUDA nvcc_wrapper builds)"
       OFF)

function(nukexc_ptxas_verbose)
    if(NOT NUKEXC_PTXAS_VERBOSE)
        return()
    endif()
    if(NOT Kokkos_ENABLE_CUDA)
        message(WARNING
            "NUKEXC_PTXAS_VERBOSE requested but Kokkos_ENABLE_CUDA is OFF; "
            "-Xptxas=-v only works with the nvcc_wrapper CXX compiler.")
        return()
    endif()
    foreach(tgt IN LISTS ARGV)
        target_compile_options(${tgt} PRIVATE -Xptxas=-v)
    endforeach()
    message(STATUS "NUKEXC_PTXAS_VERBOSE: ptxas -v enabled on ${ARGC} target(s) "
                   "in ${CMAKE_CURRENT_SOURCE_DIR}")
endfunction()
