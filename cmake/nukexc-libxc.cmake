# Enforce that Libxc must be installed and found beforehand
find_package(Libxc REQUIRED)

if(Libxc_FOUND OR libxc_FOUND)
    # Standardize case checking across modern/legacy modules
    if(libxc_FOUND)
        set(Libxc_FOUND TRUE)
        set(Libxc_VERSION ${libxc_VERSION})
        set(Libxc_DIR ${libxc_DIR})
    endif()

    message(STATUS "Found Libxc: ${Libxc_DIR} (version \"${Libxc_VERSION}\")")

    # Bridge modern namespaced targets to your internal expectations
    if(TARGET libxc::xc AND NOT TARGET Libxc::xc)
        add_library(Libxc::xc ALIAS libxc::xc)
    elseif(TARGET xc AND NOT TARGET Libxc::xc)
        add_library(Libxc::xc ALIAS xc)
    elseif(NOT TARGET Libxc::xc AND DEFINED LIBXC_LIBRARIES)
        # Fallback raw path variable wrapper for older Find modules
        add_library(Libxc::xc INTERFACE IMPORTED)
        target_link_libraries(Libxc::xc INTERFACE ${LIBXC_LIBRARIES})
        if(DEFINED LIBXC_INCLUDE_DIRS)
            target_include_directories(Libxc::xc INTERFACE ${LIBXC_INCLUDE_DIRS})
        endif()
    endif()

else()
    # Explicit backup message in case REQUIRED isn't caught by some old CMake implementations
    message(FATAL_ERROR "Libxc not found externally. Please load an appropriate module or install Libxc first.")
endif()

# Uniformly bind to your core interface target
target_link_libraries(libnukexc INTERFACE Libxc::xc)
