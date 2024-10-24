cmake_minimum_required(VERSION 3.14)
project(CUDDA)

enable_testing()  # 启用测试(必须在顶层)

set(CMAKE_CXX_STANDARD 17)  # C++17
set(CMAKE_CUDA_STANDARD 17)  # nvcc 
# set(CMAKE_BUILD_TYPE Release)  // build.bat已指定为Release
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install)
set(CMAKE_CUDA_ARCHITECTURES 75)  # Nvidia Geforce 2070s

set(3rdparty ${CMAKE_SOURCE_DIR}/3rdparty)

add_subdirectory("1.HelloWorld")
