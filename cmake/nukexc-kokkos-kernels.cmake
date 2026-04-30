find_package(KokkosKernels REQUIRED) # Try to find Kokkos externally

if(KokkosKernels_FOUND)
    message(STATUS "Found Kokkos-kernels: ${KokkosKernels_DIR} (version \"${KokkosKernels_VERSION}\")")
else()
    message(STATUS "Kokkos-kernels not found externally. Please install Kokkos-kernels first.")
endif()

find_package(LAPACK REQUIRED)
target_link_libraries(libnukexc INTERFACE Kokkos::kokkoskernels)
target_link_libraries(libnukexc INTERFACE LAPACK::LAPACK)
