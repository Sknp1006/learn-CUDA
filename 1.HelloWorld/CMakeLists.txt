cmake_minimum_required(VERSION 3.14)
project(HelloWorld LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)
message(STATUS "cuda version: " ${CUDA_VERSION_STRING})
include_directories(${CUDA_INCLUDE_DIRS})

# helloworld.cu
add_executable(00helloworld helloworld.cu)
target_link_libraries(00helloworld ${CUDA_LIBRARIES})
install(TARGETS 00helloworld DESTINATION bin)

# thread.cu
add_executable(01thread thread.cu)
target_link_libraries(01thread ${CUDA_LIBRARIES})
# kernel launch from __device__ or __global__ functions requires separate compilation mode
# 在kernel()中调用another()，需要开启分离编译,此kernel非彼kernel()
set_target_properties(01thread PROPERTIES CUDA_SEPARABLE_COMPILATION ON)  # 启用分离声明和定义
install(TARGETS 01thread DESTINATION bin)
