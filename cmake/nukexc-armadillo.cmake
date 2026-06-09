include(FetchContent)

find_package(Armadillo CONFIG QUIET)

# Normalize case
if(armadillo_FOUND AND NOT Armadillo_FOUND)
  set(Armadillo_FOUND  TRUE)
  set(Armadillo_VERSION ${armadillo_VERSION})
  set(Armadillo_DIR     ${armadillo_DIR})
endif()

if(NOT Armadillo_FOUND)
  message(STATUS "Armadillo not found, fetching...")
  FetchContent_Declare(
    armadillo
    GIT_REPOSITORY https://gitlab.com/conradsnicta/armadillo-code.git
    GIT_TAG        14.6.3
  )
  FetchContent_MakeAvailable(armadillo)

  # ArmadilloConfig.cmake is generated during configure but
  # ArmadilloLibraryDepends.cmake only exists after the build step.
  # Patch the config to guard the include so downstream find_package
  # calls (e.g. from OpenOrbitalOptimizer) don't hard-error.
  set(_arma_config "${CMAKE_BINARY_DIR}/_deps/armadillo-build/ArmadilloConfig.cmake")
  file(READ "${_arma_config}" _arma_config_contents)
  string(REGEX REPLACE
    "include[ \t]*\\([ \t]*\"[^\"]*ArmadilloLibraryDepends\\.cmake\"[ \t]*\\)"
    "if(EXISTS \"\\${CMAKE_CURRENT_LIST_DIR}/ArmadilloLibraryDepends.cmake\")\n  include(\"\\${CMAKE_CURRENT_LIST_DIR}/ArmadilloLibraryDepends.cmake\")\nendif()"
    _arma_config_patched
    "${_arma_config_contents}"
  )
  file(WRITE "${_arma_config}" "${_arma_config_patched}")

  # Point Armadillo_DIR at the build tree so any downstream find_package(Armadillo)
  # — including OpenOrbitalOptimizer's internal one — finds the patched config.
  set(Armadillo_DIR "${CMAKE_BINARY_DIR}/_deps/armadillo-build"
      CACHE PATH "Armadillo build-tree cmake dir" FORCE)

else()
  message(STATUS "Found Armadillo: ${Armadillo_DIR} (version \"${Armadillo_VERSION}\")")
endif()

# Bridge to namespaced target your code expects
if(TARGET armadillo)
  # FetchContent case — target already correct
elseif(TARGET Armadillo::Armadillo)
  add_library(armadillo ALIAS Armadillo::Armadillo)
elseif(DEFINED ARMADILLO_LIBRARIES)
  # Fallback for old Find-module style
  add_library(armadillo INTERFACE IMPORTED)
  target_link_libraries(armadillo INTERFACE ${ARMADILLO_LIBRARIES})
  if(DEFINED ARMADILLO_INCLUDE_DIRS)
    target_include_directories(armadillo INTERFACE ${ARMADILLO_INCLUDE_DIRS})
  endif()
else()
  message(FATAL_ERROR
    "Armadillo was found but no usable target or variable (armadillo, "
    "Armadillo::Armadillo, ARMADILLO_LIBRARIES) could be identified. "
    "Check your Armadillo installation.")
endif()

target_link_libraries(libnukexc INTERFACE armadillo)

