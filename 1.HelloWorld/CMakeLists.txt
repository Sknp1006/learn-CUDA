cmake_minimum_required(VERSION 3.14)
project(HelloWorld)

find_package(CUDA REQUIRED)

cuda_add_executable(Hello helloworld.cu)