include(FetchContent)

find_package(Armadillo CONFIG QUIET)
get_property(allTargets GLOBAL PROPERTY TARGETS)
message(STATUS "Targets=${allTargets}")
if(NOT Armadillo_FOUND)
    message(STATUS "Armadillo not found, fetching...")

    FetchContent_Declare(
        armadillo
        GIT_REPOSITORY https://gitlab.com/conradsnicta/armadillo-code.git
        GIT_TAG 14.6.3
    )

    FetchContent_MakeAvailable(armadillo)
endif()

target_link_libraries( libnukexc INTERFACE armadillo )
