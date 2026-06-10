find_package(Armadillo REQUIRED)

if(Armadillo_FOUND)
  message(STATUS "Found Armadillo: ${Armadillo_DIR} (version \"${Armadillo_VERSION}\")")
else()
  message(FATAL_ERROR "Armadillo not found externally. Please install Armadillo first.")
endif()

if(TARGET armadillo)
  # Already correct
elseif(TARGET Armadillo::Armadillo)
  add_library(armadillo ALIAS Armadillo::Armadillo)
elseif(DEFINED ARMADILLO_LIBRARIES)
  add_library(armadillo INTERFACE IMPORTED)
  target_link_libraries(armadillo INTERFACE ${ARMADILLO_LIBRARIES})
  if(DEFINED ARMADILLO_INCLUDE_DIRS)
    target_include_directories(armadillo INTERFACE ${ARMADILLO_INCLUDE_DIRS})
  endif()
else()
  message(FATAL_ERROR
    "Armadillo was found but no usable target or variable "
    "(armadillo, Armadillo::Armadillo, ARMADILLO_LIBRARIES) could be identified. "
    "Check your Armadillo installation.")
endif()

target_link_libraries(libnukexc INTERFACE armadillo)
