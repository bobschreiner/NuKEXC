find_package(Kokkos REQUIRED) # Try to find Kokkos externally

if(Kokkos_FOUND)
	
    message(STATUS "Found Kokkos: ${Kokkos_DIR} (version \"${Kokkos_VERSION}\")")

    # Check if OpenMP is one of the enabled backends
    if(Kokkos_ENABLE_OPENMP)
        message(STATUS "Kokkos has OpenMP enabled. Finding OpenMP...")
	find_package(OpenMP REQUIRED COMPONENTS CXX)
	target_link_libraries(libnukexc INTERFACE OpenMP::OpenMP_CXX)
    endif()
    if(Kokkos_ENABLE_HIP)
		message(STATUS "Kokkos Kernels has ROCm SVD support. Linking ROCm libraries...")
                find_package(rocblas REQUIRED)
                find_package(rocsparse REQUIRED)
                find_package(rocsolver REQUIRED)
		target_link_libraries(libnukexc INTERFACE roc::rocblas roc::rocsparse roc::rocsolver)
    endif()
    

else()
    message(STATUS "Kokkos not found externally. Please install Kokkos first.")
endif()

target_link_libraries(libnukexc INTERFACE Kokkos::kokkos)
