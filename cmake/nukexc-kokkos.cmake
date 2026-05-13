find_package(Kokkos REQUIRED)

if(Kokkos_FOUND)
    message(STATUS "Found Kokkos: ${Kokkos_DIR} (version \"${Kokkos_VERSION}\")")

    # Detect Cray Programming Environment — on LUMI/Cray, OpenMP is injected
    # automatically by the compiler wrappers and must not be found via CMake.
    if(DEFINED ENV{CRAY_PE_VERSION} OR DEFINED ENV{CRAYPE_VERSION})
        set(ON_CRAY_SYSTEM TRUE)
        message(STATUS "Cray Programming Environment detected — skipping CMake OpenMP search")
    else()
        set(ON_CRAY_SYSTEM FALSE)
    endif()

    if(Kokkos_ENABLE_OPENMP)
        if(ON_CRAY_SYSTEM)
            message(STATUS "Kokkos has OpenMP enabled. Cray wrappers handle OpenMP — not linking explicitly.")
        else()
            message(STATUS "Kokkos has OpenMP enabled. Finding OpenMP...")
            find_package(OpenMP REQUIRED COMPONENTS CXX)
            target_link_libraries(libnukexc INTERFACE OpenMP::OpenMP_CXX)
        endif()
    endif()

    if(Kokkos_ENABLE_HIP)
        message(STATUS "Kokkos has HIP enabled. Linking ROCm libraries...")
        find_package(rocblas REQUIRED)
        find_package(rocsparse REQUIRED)
        find_package(rocsolver REQUIRED)
        target_link_libraries(libnukexc INTERFACE roc::rocblas roc::rocsparse roc::rocsolver)
    endif()

else()
    message(STATUS "Kokkos not found externally. Please install Kokkos first.")
endif()

target_link_libraries(libnukexc INTERFACE Kokkos::kokkos)
