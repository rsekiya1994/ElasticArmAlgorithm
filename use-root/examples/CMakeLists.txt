cmake_minimum_required(VERSION 3.14)
project(EAA_Example CXX)

set(CMAKE_CXX_STANDARD 14)

find_package(ROOT REQUIRED)

# add_executable(test_eaa.exe test_eaa.cxx)
# target_include_directories(test_eaa.exe PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../)
# target_link_libraries(test_eaa.exe Eigen3::Eigen)

add_executable(test_EAA.exe test_EAA.cxx)
target_include_directories(test_EAA.exe PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../ ${ROOT_INCLUDE_DIRS})
target_link_libraries(test_EAA.exe ${ROOT_LIBRARIES})