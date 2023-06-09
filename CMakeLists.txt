cmake_minimum_required(VERSION 3.14)
project(CUDDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_EXPORT_COMPILECOMMANDS ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

add_subdirectory("1.HelloWorld")
add_subdirectory("2.Thread")
