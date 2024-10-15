cmake_minimum_required(VERSION 3.14)
project(HelloWorld LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)
message(STATUS "cuda version: " ${CUDA_VERSION_STRING})
include_directories(${CUDA_INCLUDE_DIRS})

cuda_add_executable(${PROJECT_NAME} helloworld.cu)
target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})

install(TARGETS ${PROJECT_NAME} DESTINATION bin)