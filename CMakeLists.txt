cmake_minimum_required( VERSION 2.8.6 FATAL_ERROR )

# We want C++11
set( GCC_FLAGS "-Wall -Wextra -Wpedantic" )
set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GCC_FLAGS} -std=c++11" CACHE STRING "Flags used by the compiler during all build types")

project( bagged-data )

option( BUILD_TESTING "Build tests" ON )

find_package(Eigen3 REQUIRED)
include_directories( SYSTEM ${EIGEN3_INCLUDE_DIR} )


include_directories( include )


if( BUILD_TESTING )
  enable_testing()
  add_subdirectory( test )
endif( BUILD_TESTING )

