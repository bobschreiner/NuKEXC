find_package(KokkosKernels REQUIRED) # Try to find Kokkos externally

if(KokkosKernels_FOUND)
    message(STATUS "Found Kokkos: ${KokkosKernels_DIR} (version \"${KokkosKernels_VERSION}\")")

target_link_libraries(libnukexc INTERFACE Kokkos::kokkoskernels)
