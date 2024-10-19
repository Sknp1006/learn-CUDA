cmake_minimum_required(VERSION 3.14)
project(HelloWorld LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

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
# set_property(TARGET 01thread PROPERTY CUDA_SEPARABLE_COMPILATION ON)  # 另一种写法
install(TARGETS 01thread DESTINATION bin)

# memory.cu
add_executable(02memory memory.cu)
target_include_directories(02memory PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(02memory ${CUDA_LIBRARIES})
install(TARGETS 02memory DESTINATION bin)