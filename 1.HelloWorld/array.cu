#include <cstdio>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include <chrono>

// 一个线程处理32个数
__global__ void single_thread(int *arr, int n) {
    for (int i = 0; i < n; i++) {
        arr[i] = i;
    }
}

__global__ void multi_thread(int *arr, int n) {
    int i = threadIdx.x;
    arr[i] = i;
}

__global__ void grid_stride_loop(int *arr, int n) {
    /**
     * @brief 小技巧：网格跨步循环
     * 无论调用者指定了多少个线程（blockDim），都能根据给定的n区间循环，不会越界，也不会漏掉元素
     * 这样符合CPU上的parallel for的习惯，不需要考虑线程数和元素数的关系
     * 
     * 备忘：blockDim是thread个数，gridDim是block个数
     */
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        arr[i] = i;
    }
}

__global__ void grid_stride_loop_2(int *arr, int n) {
    /**
     * @brief 用这个方法无论指定多少线程和板块都能根据给定的n区间循环，不会越界，也不会漏掉元素
     * 
     */
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // 计算索引
    int stride = blockDim.x * gridDim.x;  // 计算步长
    for (int i = idx; i < n; i += stride) {
        arr[i] = i;
    }
}

__global__ void round_down_block(int *arr, int n) {
    /**
     * @brief 线程不是无限的，一个block中的线程数是有限的，所以需要多个block来处理大量数据
     * 
     */
    int i = blockDim.x * blockIdx.x + threadIdx.x;  // 获取线程在网格中的索引
    arr[i] = i;
}

__global__ void round_up_block(int *arr, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;  // 向上取整要防止越界
    arr[i] = i;
}

int main() {
    {
        auto start = std::chrono::high_resolution_clock::now();
        int n = 32;
        int *arr;
        checkCudaErrors(cudaMallocManaged(&arr, n * sizeof(int)));
        single_thread<<<1, 1>>>(arr, n);
        checkCudaErrors(cudaDeviceSynchronize());
        for (int i = 0; i < n; i++) {
            printf("arr[%d]: %d\n", i, arr[i]);
        }
        cudaFree(arr);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        printf("single_thread: %ld us\n", duration.count());
    }

    {
        auto start = std::chrono::high_resolution_clock::now();
        int n = 32;
        int *arr;
        checkCudaErrors(cudaMallocManaged(&arr, n * sizeof(int)));
        multi_thread<<<1, n>>>(arr, n);
        checkCudaErrors(cudaDeviceSynchronize());
        for (int i = 0; i < n; i++) {
            printf("arr[%d]: %d\n", i, arr[i]);
        }
        cudaFree(arr);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        printf("multi_thread: %ld us\n", duration.count());
    }

    {
        auto start = std::chrono::high_resolution_clock::now();
        int n = 7;
        int *arr;
        checkCudaErrors(cudaMallocManaged(&arr, n * sizeof(int)));
        grid_stride_loop<<<1, 4>>>(arr, n);
        checkCudaErrors(cudaDeviceSynchronize());
        for (int i = 0; i < n; i++) {
            printf("arr[%d]: %d\n", i, arr[i]);
        }
        cudaFree(arr);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        printf("grid_stride_loop: %ld us\n", duration.count());
    }

    {
        auto start = std::chrono::high_resolution_clock::now();
        int n = 65536;
        int *arr;
        checkCudaErrors(cudaMallocManaged(&arr, n * sizeof(int)));
        int nthreads = 128;
        int nblocks = n / nthreads;  // 向下取整，当n不能整除nthreads时，会丢掉一部分数据，解决办法是向上取整
        round_down_block<<<nblocks, nthreads>>>(arr, n);  // nblocks个block，每个block有nthreads个线程
        checkCudaErrors(cudaDeviceSynchronize());
        // for (int i = 0; i < n; i++) {
        //     printf("arr[%d]: %d\n", i, arr[i]);
        // }
        cudaFree(arr);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        printf("round_down_block: %ld us\n", duration.count());
    }

    {
        auto start = std::chrono::high_resolution_clock::now();
        int n = 65535;
        int *arr;
        checkCudaErrors(cudaMallocManaged(&arr, n * sizeof(int)));
        int nthreads = 128;
        int nblocks = (n + nthreads - 1) / nthreads;  // 向上取整
        round_up_block<<<nblocks, nthreads>>>(arr, n);  // nblocks个block，每个block有nthreads个线程
        checkCudaErrors(cudaDeviceSynchronize());
        // for (int i = 0; i < n; i++) {
        //     printf("arr[%d]: %d\n", i, arr[i]);
        // }
        cudaFree(arr);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        printf("round_up_block: %ld us\n", duration.count());
    }

    {
        auto start = std::chrono::high_resolution_clock::now();
        int n = 65536;
        int *arr;
        checkCudaErrors(cudaMallocManaged(&arr, n * sizeof(int)));
        grid_stride_loop_2<<<32, 128>>>(arr, n);  // 32个block，每个block有128个线程
        checkCudaErrors(cudaDeviceSynchronize());
        // for (int i = 0; i < n; i++) {
        //     printf("arr[%d]: %d\n", i, arr[i]);
        // }
        cudaFree(arr);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        printf("grid_stride_loop_2: %ld us\n", duration.count());
    }

    return 0;
}