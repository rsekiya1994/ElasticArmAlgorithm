cmake_minimum_required(VERSION 3.14)
project(EAA_Example CXX)

set(CMAKE_CXX_STANDARD 14)

find_package(Eigen3 REQUIRED)
add_compile_options(-Wall)

add_executable(test_eaa.exe test_eaa.cxx)
target_include_directories(test_eaa.exe PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../)
target_link_libraries(test_eaa.exe Eigen3::Eigen)

add_executable(test_QN.exe test_QN.cxx)
target_include_directories(test_QN.exe PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../)
target_link_libraries(test_QN.exe Eigen3::Eigen)