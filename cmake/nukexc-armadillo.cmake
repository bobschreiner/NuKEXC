include(FetchContent)

find_package(Armadillo CONFIG QUIET)

if(NOT Armadillo_FOUND)
    message(STATUS "Armadillo not found, fetching...")

    FetchContent_Declare(
        armadillo
        GIT_REPOSITORY https://gitlab.com/conradsnicta/armadillo-code.git
        GIT_TAG 14.6.3
    )

    FetchContent_MakeAvailable(armadillo)

    find_package(Armadillo CONFIG REQUIRED)
endif()
