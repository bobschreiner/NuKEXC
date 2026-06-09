include(FetchContent)

set(_arma_depends
    "${CMAKE_BINARY_DIR}/_deps/armadillo-build/ArmadilloLibraryDepends.cmake")

if(NOT EXISTS "${_arma_depends}")
  set(Armadillo_FOUND FALSE)
else()
  find_package(Armadillo CONFIG QUIET)
endif()

if(NOT Armadillo_FOUND)
  message(STATUS "Armadillo not found, fetching...")
  FetchContent_Declare(
    armadillo
    GIT_REPOSITORY https://gitlab.com/conradsnicta/armadillo-code.git
    GIT_TAG 14.6.3
  )
  FetchContent_MakeAvailable(armadillo)
endif()

target_link_libraries(libnukexc INTERFACE armadillo)


