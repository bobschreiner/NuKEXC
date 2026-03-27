include( nukexc-integratorxx )
include( nukexc-kokkos)

add_executable(main main.cxx)
target_link_libraries(main Kokkos::kokkos)
