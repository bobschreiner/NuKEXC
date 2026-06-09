find_package( OpenOrbitalOptimizer QUIET )
if( NOT ${OpenOrbitalOptimizer_FOUND} )
  include( nukexc-dep-versions )
  message( STATUS "Could not find OpenOrbitalOptimizer... Building" )
  message( STATUS "OPENORBITALOPTIMIZER REPO = ${NUKEXC_OPENORBITALOPTIMIZER_REPOSITORY}" )
  message( STATUS "OPENORBITALOPTIMIZER REV  = ${NUKEXC_OPENORBITALOPTIMIZER_REVISION}"   )

  # OpenOrbitalOptimizer requires Armadillo — ensure it is available
  # before the subproject is configured, mirroring the Config logic
  if( NOT TARGET armadillo )
    find_package( Armadillo CONFIG REQUIRED )
    get_property( _ooo_iid TARGET armadillo
                  PROPERTY INTERFACE_INCLUDE_DIRECTORIES )
    if( NOT EXISTS "${_ooo_iid}" )
      if( "${CMAKE_SYSTEM_NAME}" STREQUAL "Windows" )
        if( DEFINED ENV{CONDA_PREFIX} )
          message( STATUS "Healing armadillo INTERFACE_INCLUDE_DIRECTORIES "
                          "to $ENV{CONDA_PREFIX}\\Library\\include" )
          set_property( TARGET armadillo PROPERTY
            INTERFACE_INCLUDE_DIRECTORIES
            "$ENV{CONDA_PREFIX}\\Library\\include" )
        endif()
      endif()
    endif()
  endif()

  FetchContent_Declare(
    openorbitaloptimizer
    GIT_REPOSITORY ${NUKEXC_OPENORBITALOPTIMIZER_REPOSITORY}
    GIT_TAG        ${NUKEXC_OPENORBITALOPTIMIZER_REVISION}
  )
  FetchContent_MakeAvailable( openorbitaloptimizer )
endif()

target_link_libraries( libnukexc INTERFACE OpenOrbitalOptimizer::OpenOrbitalOptimizer )
