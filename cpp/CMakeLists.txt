# CMake version
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

# Declare project
project(OpenUnmixProj)

# Set c++ standard
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Thirdparties
set(libtorch_VERSION 1.9.0)
set(CMAKE_PREFIX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/lib/libtorch)
find_package(Torch ${libtorch_VERSION} EXACT QUIET CONFIG)
if(NOT Torch_FOUND)
  message(STATUS "libtorch ${libtorch_VERSION} - not found")
  message(STATUS "Fetching libtorch")
  include(FetchContent)
  FetchContent_Declare(
          libtorch
          URL https://download.pytorch.org/libtorch/cpu/libtorch-macos-${libtorch_VERSION}.zip
          SOURCE_DIR libtorch)
  FetchContent_GetProperties(libtorch)
  if(NOT libtorch_POPULATED)
    unset(FETCHCONTENT_QUIET CACHE)
    FetchContent_Populate(libtorch)
    list(APPEND CMAKE_PREFIX_PATH ${CMAKE_BINARY_DIR}/libtorch)
  endif()
  find_package(Torch ${libtorch_VERSION} EXACT CONFIG REQUIRED)
else()
  message(STATUS "libtorch ${libtorch_VERSION} - found")
endif()

add_subdirectory(Libraries/AudioFile)
add_subdirectory(Libraries/pffft)
include_directories(Libraries/AudioFile)
include_directories(Libraries/pffft)
include_directories(${TORCH_INCLUDE_DIRS})

# Includes
include_directories(Includes)

# Project module
add_subdirectory(Sources/OpenUnmix)
