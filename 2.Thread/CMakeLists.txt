cmake_minimum_required(VERSION 3.14)
project(Thread LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
message(STATUS "CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")

add_executable(${PROJECT_NAME} thread.cu)
install(TARGETS ${PROJECT_NAME} DESTINATION bin)