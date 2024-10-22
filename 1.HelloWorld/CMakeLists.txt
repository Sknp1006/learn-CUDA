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

# array.cu
add_executable(03array array.cu)
target_include_directories(03array PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(03array ${CUDA_LIBRARIES})
install(TARGETS 03array DESTINATION bin)

# sugar.cu
add_executable(04sugar sugar.cu)
target_include_directories(04sugar PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(04sugar ${CUDA_LIBRARIES})
target_compile_options(04sugar PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)  # 启用CUDA扩展lambda，允许lambda声明为__device__
install(TARGETS 04sugar DESTINATION bin)

# math.cu
add_executable(05math math.cu)
target_include_directories(05math PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(05math ${CUDA_LIBRARIES})
target_compile_options(05math PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
install(TARGETS 05math DESTINATION bin)

# thrust.cu
add_executable(06thrust thrust.cu)
target_include_directories(06thrust PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(06thrust ${CUDA_LIBRARIES})
target_compile_options(06thrust PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
install(TARGETS 06thrust DESTINATION bin)

# atomic.cu
add_executable(07atomic atomic.cu)
target_include_directories(07atomic PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(07atomic ${CUDA_LIBRARIES})
install(TARGETS 07atomic DESTINATION bin)