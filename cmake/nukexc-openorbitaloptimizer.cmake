find_package(OpenOrbitalOptimizer QUIET)

if(OpenOrbitalOptimizer_FOUND)
  message(STATUS "Found OpenOrbitalOptimizer: ${OpenOrbitalOptimizer_DIR} (version \"${OpenOrbitalOptimizer_VERSION}\")")
else()
  include(nukexc-dep-versions)
  message(STATUS "Could not find OpenOrbitalOptimizer... Building")
  message(STATUS "OPENORBITALOPTIMIZER REPO = ${NUKEXC_OPENORBITALOPTIMIZER_REPOSITORY}")
  message(STATUS "OPENORBITALOPTIMIZER REV  = ${NUKEXC_OPENORBITALOPTIMIZER_REVISION}")

  if(NOT TARGET armadillo)
    message(FATAL_ERROR
      "armadillo target not found. "
      "Include nukexc-armadillo.cmake before nukexc-openorbitaloptimizer.cmake.")
  endif()

  include(FetchContent)
  FetchContent_Declare(
    openorbitaloptimizer
    GIT_REPOSITORY ${NUKEXC_OPENORBITALOPTIMIZER_REPOSITORY}
    GIT_TAG        ${NUKEXC_OPENORBITALOPTIMIZER_REVISION}
  )
  FetchContent_MakeAvailable(openorbitaloptimizer)
endif()

if(TARGET OpenOrbitalOptimizer::OpenOrbitalOptimizer)
  # Already correct
elseif(TARGET openorbitaloptimizer)
  add_library(OpenOrbitalOptimizer::OpenOrbitalOptimizer ALIAS openorbitaloptimizer)
else()
  message(FATAL_ERROR
    "OpenOrbitalOptimizer was found but no usable target "
    "(OpenOrbitalOptimizer::OpenOrbitalOptimizer, openorbitaloptimizer) "
    "could be identified. Check your OpenOrbitalOptimizer installation.")
endif()

target_link_libraries(libnukexc INTERFACE OpenOrbitalOptimizer::OpenOrbitalOptimizer)

