find_package(ArborX QUIET)

if(ArborX_FOUND)
    message(STATUS "Found ArborX: ${ArborX_DIR} (version \"${ArborX_VERSION}\")")
else()
    message(STATUS "ArborX not found externally, fetching...")
    include(FetchContent)
    FetchContent_Declare(
        ArborX
        GIT_REPOSITORY https://github.com/arborx/ArborX.git
        GIT_TAG        v2.1
    )
    FetchContent_MakeAvailable(ArborX)
endif()

target_link_libraries(libnukexc INTERFACE ArborX::ArborX)
