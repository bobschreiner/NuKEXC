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
  # When built via FetchContent, libxc exports target 'xc', not 'Libxc::xc'.
  # Create an alias so the rest of the build uses a consistent name.
 else()
	message(STATUS "Found Libxc: ${Libxc_DIR} (version \"${Libxc_VERSION}\")")
endif()
  if( TARGET xc )
    target_include_directories( xc INTERFACE 
      $<BUILD_INTERFACE:${libxc_BINARY_DIR}/src>
      $<BUILD_INTERFACE:${libxc_SOURCE_DIR}/src>
      $<BUILD_INTERFACE:${libxc_BINARY_DIR}/include>
    )
    endif()
  if( NOT TARGET Libxc::xc )
    add_library( Libxc::xc ALIAS xc )
  endif()
 
target_link_libraries( libnukexc INTERFACE Libxc::xc )
