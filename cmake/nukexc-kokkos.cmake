find_package(Kokkos REQUIRED) # Try to find Kokkos externally

if(Kokkos_FOUND)
	
    message(STATUS "Found Kokkos: ${Kokkos_DIR} (version \"${Kokkos_VERSION}\")")

    # Check if OpenMP is one of the enabled backends
    if(Kokkos_ENABLE_OPENMP)
        message(STATUS "Kokkos has OpenMP enabled. Finding OpenMP...")
	find_package(OpenMP REQUIRED COMPONENTS CXX)
	target_link_libraries(libnukexc INTERFACE OpenMP::OpenMP_CXX)
    endif()

else()
    message(STATUS "Kokkos not found externally. Please install Kokkos first.")
endif()

target_link_libraries(libnukexc INTERFACE Kokkos::kokkos)
