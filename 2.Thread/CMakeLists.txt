cmake_minimum_required(VERSION 3.14)
project(Thread)

find_package(CUDA 11.6 REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
message(STATUS "CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")

cuda_add_executable(${PROJECT_NAME} thread.cu)
