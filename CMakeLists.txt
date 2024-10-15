cmake_minimum_required(VERSION 3.14)
project(CUDDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install)

cmake_policy(SET CMP0146 NEW)
# set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6")
# set(CUDA_NVCC_EXECUTABLE ${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc.exe)
# set(CUDA_INCLUDE_DIRS ${CUDA_TOOLKIT_ROOT_DIR}/include)
# set(CUDA_CUDART_LIBRARY ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/cudart.lib)

# 添加子目录
add_subdirectory("1.HelloWorld")
add_subdirectory("2.Thread")