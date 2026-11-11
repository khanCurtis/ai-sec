cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(day01)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(day01 day01.cpp)
target_link_libraries(day01 "${TORCH_LIBRARIES}")
set_property(TARGET day01 PROPERTY CXX_STANDARD 20)
