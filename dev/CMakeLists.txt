cmake_minimum_required(VERSION 3.20)
project(azah)
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT azah)
set(LOCAL_DIR ${CMAKE_CURRENT_LIST_DIR})

set(SOURCE_DIR ${LOCAL_DIR}/src)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# For MSVC; gives a nice project layout.
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
 
if(MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /fp:fast /arch:AVX2 /bigobj")   
else()
  message(FATAL_ERROR "Only MSVC is supported for now.")
endif()
 
define_property(
  TARGET
  PROPERTY FOLDER
  INHERITED
  BRIEF_DOCS "Set the folder name."
  FULL_DOCS  "Use to organize targets in an IDE."
)

# Set up external deps below...

include(FetchContent)

FetchContent_Declare(
  gflags
  GIT_REPOSITORY https://github.com/gflags/gflags
  GIT_TAG        master
)

FetchContent_Declare(
  glog
  GIT_REPOSITORY https://github.com/google/glog
  GIT_TAG        master
)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest
  GIT_TAG        main
)

FetchContent_Declare(
  abseil
  GIT_REPOSITORY https://github.com/abseil/abseil-cpp
  GIT_TAG        master
)

FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen
  GIT_TAG        master
)

option(BUILD_gflags_LIB ON)
# We don't care about the included project tests.
option(BUILD_TESTING OFF)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "")
FetchContent_MakeAvailable(eigen glog abseil gflags googletest)

enable_testing()
option(BUILD_TESTING ON)

add_subdirectory(${SOURCE_DIR})
