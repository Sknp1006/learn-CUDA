#include <cstdio>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include <vector>
#include "CudaAllocator.h"
#include "ticktock.h"

__global__ void parallel_sum(int *sum, int const *arr, int n)
{
    for (int i = blockDim.x * blockIdx.x + threadIdx.x;
         i < n; i += blockDim.x * gridDim.x)
    {
        sum[0] += arr[i]; // 会有多个线程同时写入sum[0]，导致结果不确定
    }
}

__global__ void parallel_sum_atomic(int *sum, int const *arr, int n)
{
    for (int i = blockDim.x * blockIdx.x + threadIdx.x;
         i < n; i += blockDim.x * gridDim.x)
    {
        // atomicAdd返回值是原来的值
        // 相当于 old = *dst; *dst += src;
        // 利用此特性可以往全局的res数组中记录当前数组大小
        atomicAdd(&sum[0], arr[i]); // 原子操作，保证只有一个线程写入sum[0]
    }
}

/**
 * @brief 结果过滤器，利用了atomicAdd返回旧值的特性，将符合条件的元素放入res数组中
 * 
 * @param sum 
 * @param res 
 * @param arr 
 * @param n 
 * @return __global__ 
 */
__global__ void parallel_filter(int *sum, int *res, int const *arr, int n) {
    for (int i = blockDim.x * blockIdx.x + threadIdx.x;
         i < n; i += blockDim.x * gridDim.x) {
        if (arr[i] >= 2) {
            int loc = atomicAdd(&sum[0], 1);
            res[loc] = arr[i];
        }
    }
}

/// @brief TLS(Thread Local Storage)线程本地存储，以降低atomic操作的次数
/// 例如 parallel_sum_tls<<<n / 4096, 128>>> 每个线程将执行 8 次，而atomicAdd只执行 1 次
__global__ void parallel_sum_tls(int *sum, int const *arr, int n) {
    int local_sum = 0;
    for (int i = blockDim.x * blockIdx.x + threadIdx.x;
         i < n; i += blockDim.x * gridDim.x) {
        local_sum += arr[i];
    }
    atomicAdd(sum, local_sum);
}


int main()
{
    {
        int n = 65536;
        std::vector<int, CudaAllocator<int>> arr(n);
        std::vector<int, CudaAllocator<int>> sum(1);
        for (int i = 0; i < n; i++)
        {
            arr[i] = std::rand() % 4;  // 0-3
        }
        TICK(parallel_sum);
        parallel_sum<<<n / 128, 128>>>(sum.data(), arr.data(), n); // 每个线程被调用次数 = n / (blockDim.x * gridDim.x)
        checkCudaErrors(cudaDeviceSynchronize());
        TOCK(parallel_sum);
        printf("wrong result: %d\n", sum[0]);
    }

    {
        int n = 65536;
        std::vector<int, CudaAllocator<int>> arr(n);
        std::vector<int, CudaAllocator<int>> sum(1);
        for (int i = 0; i < n; i++)
        {
            arr[i] = std::rand() % 4;  // 0-3
            // arr[i] = 1;
        }
        TICK(parallel_sum_atomic);
        parallel_sum_atomic<<<n / 128, 128>>>(sum.data(), arr.data(), n); // 每个线程被调用次数 = n / (blockDim.x * gridDim.x)
        checkCudaErrors(cudaDeviceSynchronize());
        TOCK(parallel_sum_atomic);
        printf("correct result: %d\n", sum[0]);
    }

    {
        int n = 1<<24;  // 2^24
        std::vector<int, CudaAllocator<int>> arr(n);
        std::vector<int, CudaAllocator<int>> sum(1);
        std::vector<int, CudaAllocator<int>> res(n);
        for (int i = 0; i < n; i++) {
            arr[i] = std::rand() % 4;  // 0-3
        }
        TICK(parallel_filter);
        parallel_filter<<<n / 4096, 512>>>(sum.data(), res.data(), arr.data(), n);  // 每个线程有8次atomic
        checkCudaErrors(cudaDeviceSynchronize());
        TOCK(parallel_filter);
        for (int i = 0; i < sum[0]; i++) {
            if (res[i] < 2) {
                printf("Wrong At %d\n", i);
                return -1;
            }
        }
        printf("All Correct!\n");
    }

    return 0;
}
