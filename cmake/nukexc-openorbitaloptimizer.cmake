find_package( OpenOrbitalOptimizer QUIET )
if( NOT ${OpenOrbitalOptimizer_FOUND} )
  include( nukexc-dep-versions )
  message( STATUS "Could not find OpenOrbitalOptimizer... Building" )
  message( STATUS "OPENORBITALOPTIMIZER REPO = ${NUKEXC_OPENORBITALOPTIMIZER_REPOSITORY}" )
  message( STATUS "OPENORBITALOPTIMIZER REV  = ${NUKEXC_OPENORBITALOPTIMIZER_REVISION}"   )

  if( NOT TARGET armadillo )
    message( FATAL_ERROR
      "armadillo target not found. Include nukexc-armadillo.cmake before "
      "nukexc-openorbitaloptimizer.cmake." )
  endif()

  FetchContent_Declare(
    openorbitaloptimizer
    GIT_REPOSITORY ${NUKEXC_OPENORBITALOPTIMIZER_REPOSITORY}
    GIT_TAG        ${NUKEXC_OPENORBITALOPTIMIZER_REVISION}
  )
  FetchContent_MakeAvailable( openorbitaloptimizer )
endif()

target_link_libraries( libnukexc INTERFACE OpenOrbitalOptimizer::OpenOrbitalOptimizer )
