find_package( Catch2 CONFIG)

if(Catch2_FOUND)

	message(STATUS "Found Catch2: ${Catch2_DIR} (version \"${Catch2_VERSION}\")")
else()


  message(STATUS "Catch2 not found externally. Fetching via FetchContent.")
  include(FetchContent)
  FetchContent_Declare(
    catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.14.0
  )
  
  set(CATCH_BUILD_TESTING OFF CACHE BOOL "Build SelfTest project" FORCE)
  set(CATCH_INSTALL_DOCS OFF CACHE BOOL "Install documentation alongside library" FORCE)
  set(CATCH_INSTALL_HELPERS OFF CACHE BOOL "Install contrib alongside library" FORCE)

  FetchContent_MakeAvailable( catch2 )

endif()

target_link_libraries(nukexc PUBLIC Catch2::Catch2)
