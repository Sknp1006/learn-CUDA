cmake_minimum_required(VERSION 3.14)
project(CUDDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILECOMMANDS ON)
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install)

add_subdirectory("1.HelloWorld")
add_subdirectory("2.Thread")
