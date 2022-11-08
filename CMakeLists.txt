cmake_minimum_required(VERSION 3.7)
project(~/work/sim_indirect_recip_stochastic)

set(CMAKE_CXX_STANDARD 14)

#set(CMAKE_VERBOSE_MAKEFILE 1)

find_package(OpenMP REQUIRED)
include_directories(${CMAKE_SOURCE_DIR}/icecream ${CMAKE_SOURCE_DIR}/json/include)

set(SOURCE_FILES Norm.hpp Game.hpp)

add_executable(main_CESS.out main_CESS.cpp ${SOURCE_FILES})
target_link_libraries(main_CESS.out PRIVATE OpenMP::OpenMP_CXX)

