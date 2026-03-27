find_package(Kokkos CONFIG) # Try to find Kokkos externally

if(Kokkos_FOUND)
	
    message(STATUS "Found Kokkos: ${Kokkos_DIR} (version \"${Kokkos_VERSION}\")")

else()

    message(STATUS "Kokkos not found externally. Fetching via FetchContent.")
    include(FetchContent)

    FetchContent_Declare(
      Kokkos
      URL      https://github.com/kokkos/kokkos/releases/download/4.5.01/kokkos-4.5.01.tar.gz
      URL_HASH SHA256=52d003ffbbe05f30c89966e4009c017efb1662b02b2b73190670d3418719564c
    )

    FetchContent_MakeAvailable(Kokkos)

endif()

target_link_libraries(nukexc PUBLIC Kokkos::kokkos)
