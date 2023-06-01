cmake_minimum_required(VERSION 3.14)
project(HelloWorld)

find_package(CUDA 11.6 REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
message(STATUS "CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")

cuda_add_executable(Hello helloworld.cu)