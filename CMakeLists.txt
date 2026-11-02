cmake_minimum_required(VERSION 3.15)
project(torchsky)

if(DEFINED ENV{TORCHSKY_DISABLE_CPP} AND "$ENV{TORCHSKY_DISABLE_CPP}" STREQUAL "1")
    message(STATUS "TORCHSKY_DISABLE_CPP=1: skipping native extension build")
    return()
endif()

# Add the actual CMake project as a subdirectory
add_subdirectory(src/torchsky/cpp_src)
