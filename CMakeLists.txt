cmake_minimum_required(VERSION 3.14)
project(CUDDA)

set(CMAKE_CXX_STANDARD 17)  # C++17
set(CMAKE_CUDA_STANDARD 17)  # nvcc 
# set(CMAKE_BUILD_TYPE Release)  // build.bat已指定为Release
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install)
set(CMAKE_CUDA_ARCHITECTURES 75)  # Nvidia Geforce 2070s
set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF)  # 分离声明和定义，不建议开启，关闭可方便自动内联优化

add_subdirectory("1.HelloWorld")
