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

  # Armadillo generates ArmadilloConfig.cmake during configure, but
  # ArmadilloLibraryDepends.cmake only exists after the build step.
  # Any downstream find_package(Armadillo) during this same configure
  # run (e.g. from OpenOrbitalOptimizer) will fail with a hard error.
  # Overwrite the generated config with a safe version that guards the
  # include so it is a no-op when the depends file is not yet present.
  set(_arma_config "${CMAKE_BINARY_DIR}/_deps/armadillo-build/ArmadilloConfig.cmake")
  file(READ "${_arma_config}" _arma_config_contents)
  string(REPLACE
    "include(\"\${CMAKE_CURRENT_LIST_DIR}/ArmadilloLibraryDepends.cmake\")"
    "if(EXISTS \"\${CMAKE_CURRENT_LIST_DIR}/ArmadilloLibraryDepends.cmake\")\n  include(\"\${CMAKE_CURRENT_LIST_DIR}/ArmadilloLibraryDepends.cmake\")\nendif()"
    _arma_config_contents_patched
    "${_arma_config_contents}"
  )
  file(WRITE "${_arma_config}" "${_arma_config_contents_patched}")
endif()

target_link_libraries(libnukexc INTERFACE armadillo)

