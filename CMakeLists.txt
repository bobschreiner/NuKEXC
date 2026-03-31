cmake_minimum_required( VERSION 3.20 FATAL_ERROR )

include(FetchContent)
set( FETCHCONTENT_UPDATES_DISCONNECTED ON CACHE BOOL "Disable FC Updates" )

project( NuKEXC VERSION 0.0.1 LANGUAGES C CXX )

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# Place local modules in the path
list( PREPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake )

# Guard some options settings to only default when not a subproject
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  # Populate BUILD_TESTING prior to dependencies to avoid clash
  include(CTest)

  # Default the built type
  if( NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES )
    set( CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Choose the type of build" FORCE )
    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
      "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
  endif()
endif()

add_library(nukexc src/main.cxx)

add_subdirectory( src )
add_subdirectory( tests )

list(REMOVE_AT CMAKE_MODULE_PATH 0)
