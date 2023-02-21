cmake_minimum_required(VERSION 3.7)
project(sim_indirect_recip_stochastic)

set(CMAKE_CXX_STANDARD 17)

#set(CMAKE_VERBOSE_MAKEFILE 1)

include_directories(${CMAKE_SOURCE_DIR}/icecream-cpp)

add_executable(main_public_rep.out main_public_rep.cpp Norm.hpp PublicRepGame.hpp)

add_executable(test_Norm test_Norm.cpp Norm.hpp)

add_executable(test_PublicRepGame test_PublicRepGame.cpp PublicRepGame.hpp Norm.hpp)
