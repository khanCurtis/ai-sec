cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(day02)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(day02 day02.cpp)
target_link_libraries(day02 "${TORCH_LIBRARIES}")
set_property(TARGET day02 PROPERTY CXX_STANDARD 20)
