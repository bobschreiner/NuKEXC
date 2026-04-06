find_package( IntegratorXX QUIET)
if( NOT ${IntegratorXX_FOUND} )

  include( nukexc-dep-versions )

  message( STATUS "Could not find IntegratorXX... Building" )
  message( STATUS "INTEGRATORXX REPO = ${NUKEXC_INTEGRATORXX_REPOSITORY}" )
  message( STATUS "INTEGRATORXX REV  = ${NUKEXC_INTEGRATORXX_REVISION}"   )

  set( INTEGRATORXX_ENABLE_TESTS OFF CACHE BOOL "" )
  FetchContent_Declare(
    integratorxx
    GIT_REPOSITORY ${NUKEXC_INTEGRATORXX_REPOSITORY} 
    GIT_TAG        ${NUKEXC_INTEGRATORXX_REVISION} 
  )

  FetchContent_MakeAvailable( integratorxx )

endif()

target_link_libraries( nukexc PUBLIC IntegratorXX::IntegratorXX) 
