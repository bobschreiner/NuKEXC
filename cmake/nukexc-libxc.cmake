find_package( Libxc QUIET )

if( NOT ${Libxc_FOUND} )
  include( nukexc-dep-versions )
  message( STATUS "Could not find Libxc... Building" )
  message( STATUS "LIBXC REPO = ${NUKEXC_LIBXC_REPOSITORY}" )
  message( STATUS "LIBXC REV  = ${NUKEXC_LIBXC_REVISION}"   )
  set( LIBXC_ENABLE_TESTS OFF CACHE BOOL "" )
  FetchContent_Declare(
    libxc
    GIT_REPOSITORY ${NUKEXC_LIBXC_REPOSITORY}
    GIT_TAG        ${NUKEXC_LIBXC_REVISION}
  )
  FetchContent_MakeAvailable( libxc )
  else()
	message(STATUS "Found Libxc: ${Libxc_DIR} (version \"${Libxc_VERSION}\")")
endif()
target_link_libraries( libnukexc INTERFACE Libxc::xc )
