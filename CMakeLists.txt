cmake_minimum_required(VERSION 3.7)
project(sim_indirect_recip_stochastic)

set(CMAKE_CXX_STANDARD 17)

#set(CMAKE_VERBOSE_MAKEFILE 1)

find_package(OpenMP REQUIRED)
find_package(Eigen3 REQUIRED)
include_directories(${CMAKE_SOURCE_DIR}/icecream-cpp)

set(SOURCE_FILES Norm.hpp Game.hpp)

add_executable(main_CESS.out main_CESS.cpp ${SOURCE_FILES})

add_executable(main_priv_sim.out main_priv_sim.cpp ${SOURCE_FILES})
target_link_libraries(main_priv_sim.out OpenMP::OpenMP_CXX Eigen3::Eigen)

add_executable(test_Norm test_Norm.cpp Norm.hpp)

add_executable(test_Game test_Game.cpp Game.hpp Norm.hpp)