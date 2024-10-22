#include <cstdio>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include <vector>
#include "CudaAllocator.h"
#include "ticktock.h"

/// @brief 复杂度为O(n)的并行求和算法
__global__ void parallel_sum(int *sum, int const *arr, int n) {
    for (int i = blockDim.x * blockIdx.x + threadIdx.x;
         i < n / 1024; i += blockDim.x * gridDim.x) {
        int local_sum = 0;
        for (int j = i * 1024; j < i * 1024 + 1024; j++) {
            local_sum += arr[j];
        }
        sum[i] = local_sum;
    }
}

/// @brief 线程局部数组，复杂度为O(log(n))的并行求和算法
__global__ void parallel_sum_local(int *sum, int const *arr, int n) {
    for (int i = blockDim.x * blockIdx.x + threadIdx.x;
         i < n / 1024; i += blockDim.x * gridDim.x) {
        int local_sum[1024];
        for (int j = 0; j < 1024; j++) {
            local_sum[j] = arr[i * 1024 + j];
        }
        for (int j = 0; j < 512; j++) {
            local_sum[j] += local_sum[j + 512];
        }
        for (int j = 0; j < 256; j++) {
            local_sum[j] += local_sum[j + 256];
        }
        for (int j = 0; j < 128; j++) {
            local_sum[j] += local_sum[j + 128];
        }
        for (int j = 0; j < 64; j++) {
            local_sum[j] += local_sum[j + 64];
        }
        for (int j = 0; j < 32; j++) {
            local_sum[j] += local_sum[j + 32];
        }
        for (int j = 0; j < 16; j++) {
            local_sum[j] += local_sum[j + 16];
        }
        for (int j = 0; j < 8; j++) {
            local_sum[j] += local_sum[j + 8];
        }
        for (int j = 0; j < 4; j++) {
            local_sum[j] += local_sum[j + 4];
        }
        for (int j = 0; j < 2; j++) {
            local_sum[j] += local_sum[j + 2];
        }
        for (int j = 0; j < 1; j++) {
            local_sum[j] += local_sum[j + 1];
        }
        sum[i] = local_sum[0];
    }
}

/// @brief 板块局部数组，复杂度为O(log(n))的并行求和算法
__global__ void parallel_sum_block(int *sum, int const *arr, int n) {
    __shared__ int local_sum[1024];  // 由板块内所有线程共享的局部数组
    int j = threadIdx.x;
    int i = blockIdx.x;
    local_sum[j] = arr[i * 1024 + j];
    if (j < 512) {
        local_sum[j] += local_sum[j + 512];
    }
    if (j < 256) {
        local_sum[j] += local_sum[j + 256];
    }
    if (j < 128) {
        local_sum[j] += local_sum[j + 128];
    }
    if (j < 64) {
        local_sum[j] += local_sum[j + 64];
    }
    if (j < 32) {
        local_sum[j] += local_sum[j + 32];
    }
    if (j < 16) {
        local_sum[j] += local_sum[j + 16];
    }
    if (j < 8) {
        local_sum[j] += local_sum[j + 8];
    }
    if (j < 4) {
        local_sum[j] += local_sum[j + 4];
    }
    if (j < 2) {
        local_sum[j] += local_sum[j + 2];
    }
    if (j == 0) {
        sum[i] = local_sum[0] + local_sum[1];
    }
}

__global__ void parallel_sum_block_sync(int *sum, int const *arr, int n) {
    __shared__ int local_sum[1024];
    int j = threadIdx.x;
    int i = blockIdx.x;
    local_sum[j] = arr[i * 1024 + j];
    __syncthreads();
    if (j < 512) {
        local_sum[j] += local_sum[j + 512];
    }
    __syncthreads();
    if (j < 256) {
        local_sum[j] += local_sum[j + 256];
    }
    __syncthreads();
    if (j < 128) {
        local_sum[j] += local_sum[j + 128];
    }
    __syncthreads();
    if (j < 64) {
        local_sum[j] += local_sum[j + 64];
    }
    __syncthreads();
    if (j < 32) {
        local_sum[j] += local_sum[j + 32];
    }
    __syncthreads();
    if (j < 16) {
        local_sum[j] += local_sum[j + 16];
    }
    __syncthreads();
    if (j < 8) {
        local_sum[j] += local_sum[j + 8];
    }
    __syncthreads();
    if (j < 4) {
        local_sum[j] += local_sum[j + 4];
    }
    __syncthreads();
    if (j < 2) {
        local_sum[j] += local_sum[j + 2];
    }
    __syncthreads();
    if (j == 0) {
        sum[i] = local_sum[0] + local_sum[1];
    }
}

__global__ void parallel_sum_block_wrap(int *sum, int const *arr, int n) {
    // 这里要使用volatile，否则编译器会优化掉
    __shared__ volatile int local_sum[1024];
    int j = threadIdx.x;
    int i = blockIdx.x;
    local_sum[j] = arr[i * 1024 + j];
    __syncthreads();
    if (j < 512) {
        local_sum[j] += local_sum[j + 512];
    }
    __syncthreads();
    if (j < 256) {
        local_sum[j] += local_sum[j + 256];
    }
    __syncthreads();
    if (j < 128) {
        local_sum[j] += local_sum[j + 128];
    }
    __syncthreads();
    if (j < 64) {
        local_sum[j] += local_sum[j + 64];
    }
    __syncthreads();
    if (j < 32) {
        local_sum[j] += local_sum[j + 32];
    }
    if (j < 16) {
        local_sum[j] += local_sum[j + 16];
    }
    if (j < 8) {
        local_sum[j] += local_sum[j + 8];
    }
    if (j < 4) {
        local_sum[j] += local_sum[j + 4];
    }
    if (j < 2) {
        local_sum[j] += local_sum[j + 2];
    }
    if (j == 0) {
        sum[i] = local_sum[0] + local_sum[1];
    }
}

/// @brief 避免线程组分歧的做法，让32个线程进入同一个分支
__global__ void parallel_sum_block_wrap_2(int *sum, int const *arr, int n) {
    __shared__ volatile int local_sum[1024];
    int j = threadIdx.x;
    int i = blockIdx.x;
    local_sum[j] = arr[i * 1024 + j];
    __syncthreads();
    if (j < 512) {
        local_sum[j] += local_sum[j + 512];
    }
    __syncthreads();
    if (j < 256) {
        local_sum[j] += local_sum[j + 256];
    }
    __syncthreads();
    if (j < 128) {
        local_sum[j] += local_sum[j + 128];
    }
    __syncthreads();
    if (j < 64) {
        local_sum[j] += local_sum[j + 64];
    }
    __syncthreads();
    if (j < 32) {
        local_sum[j] += local_sum[j + 32];
        local_sum[j] += local_sum[j + 16];
        local_sum[j] += local_sum[j + 8];
        local_sum[j] += local_sum[j + 4];
        local_sum[j] += local_sum[j + 2];
        if (j == 0) {
            sum[i] = local_sum[0] + local_sum[1];
        }
    }
}

__global__ void parallel_sum_grid_loop(int *sum, int const *arr, int n) {
    __shared__ volatile int local_sum[1024];
    int j = threadIdx.x;
    int i = blockIdx.x;
    int temp_sum = 0;
    for (int t = i * 1024 + j; t < n; t += 1024 * gridDim.x) {
        temp_sum += arr[t];
    }
    local_sum[j] = temp_sum;
    __syncthreads();
    if (j < 512) {
        local_sum[j] += local_sum[j + 512];
    }
    __syncthreads();
    if (j < 256) {
        local_sum[j] += local_sum[j + 256];
    }
    __syncthreads();
    if (j < 128) {
        local_sum[j] += local_sum[j + 128];
    }
    __syncthreads();
    if (j < 64) {
        local_sum[j] += local_sum[j + 64];
    }
    __syncthreads();
    if (j < 32) {
        local_sum[j] += local_sum[j + 32];
        local_sum[j] += local_sum[j + 16];
        local_sum[j] += local_sum[j + 8];
        local_sum[j] += local_sum[j + 4];
        local_sum[j] += local_sum[j + 2];
        if (j == 0) {
            sum[i] = local_sum[0] + local_sum[1];
        }
    }
}

/**
 * @brief 使用模板包装一下
 * 板块局部数组加速，BLS(block-local storage)
 */
template <int blockSize, class T>
__global__ void parallel_sum_kernel(T *sum, T const *arr, int n) {
    __shared__ volatile int local_sum[blockSize];
    int j = threadIdx.x;
    int i = blockIdx.x;
    T temp_sum = 0;
    for (int t = i * blockSize + j; t < n; t += blockSize * gridDim.x) {
        temp_sum += arr[t];
    }
    local_sum[j] = temp_sum;
    __syncthreads();
    if constexpr (blockSize >= 1024) {
        if (j < 512)
            local_sum[j] += local_sum[j + 512];
        __syncthreads();
    }
    if constexpr (blockSize >= 512) {
        if (j < 256)
            local_sum[j] += local_sum[j + 256];
        __syncthreads();
    }
    if constexpr (blockSize >= 256) {
        if (j < 128)
            local_sum[j] += local_sum[j + 128];
        __syncthreads();
    }
    if constexpr (blockSize >= 128) {
        if (j < 64)
            local_sum[j] += local_sum[j + 64];
        __syncthreads();
    }
    if (j < 32) {
        if constexpr (blockSize >= 64)
            local_sum[j] += local_sum[j + 32];
        if constexpr (blockSize >= 32)
            local_sum[j] += local_sum[j + 16];
        if constexpr (blockSize >= 16)
            local_sum[j] += local_sum[j + 8];
        if constexpr (blockSize >= 8)
            local_sum[j] += local_sum[j + 4];
        if constexpr (blockSize >= 4)
            local_sum[j] += local_sum[j + 2];
        if (j == 0) {
            sum[i] = local_sum[0] + local_sum[1];
        }
    }
}
template <int reduceScale = 4096, int blockSize = 256, class T>
int parallel_sum_temp(T const *arr, int n) {
    std::vector<int, CudaAllocator<int>> sum(n / reduceScale);
    parallel_sum_kernel<blockSize><<<n / reduceScale, blockSize>>>(sum.data(), arr, n);
    checkCudaErrors(cudaDeviceSynchronize());
    T final_sum = 0;
    for (int i = 0; i < n / reduceScale; i++) {
        final_sum += sum[i];
    }
    return final_sum;
}

/// @brief 对上一个模板包装的进一步优化，使用递归O(log(n))的并行求和算法
template <int blockSize, class T>
__global__ void parallel_sum_kernel_recu(T *sum, T const *arr, int n) {
    __shared__ volatile int local_sum[blockSize];
    int j = threadIdx.x;
    int i = blockIdx.x;
    T temp_sum = 0;
    for (int t = i * blockSize + j; t < n; t += blockSize * gridDim.x) {
        temp_sum += arr[t];
    }
    local_sum[j] = temp_sum;
    __syncthreads();
    if constexpr (blockSize >= 1024) {
        if (j < 512)
            local_sum[j] += local_sum[j + 512];
        __syncthreads();
    }
    if constexpr (blockSize >= 512) {
        if (j < 256)
            local_sum[j] += local_sum[j + 256];
        __syncthreads();
    }
    if constexpr (blockSize >= 256) {
        if (j < 128)
            local_sum[j] += local_sum[j + 128];
        __syncthreads();
    }
    if constexpr (blockSize >= 128) {
        if (j < 64)
            local_sum[j] += local_sum[j + 64];
        __syncthreads();
    }
    if (j < 32) {
        if constexpr (blockSize >= 64)
            local_sum[j] += local_sum[j + 32];
        if constexpr (blockSize >= 32)
            local_sum[j] += local_sum[j + 16];
        if constexpr (blockSize >= 16)
            local_sum[j] += local_sum[j + 8];
        if constexpr (blockSize >= 8)
            local_sum[j] += local_sum[j + 4];
        if constexpr (blockSize >= 4)
            local_sum[j] += local_sum[j + 2];
        if (j == 0) {
            sum[i] = local_sum[0] + local_sum[1];
        }
    }
}
template <int reduceScale = 4096, int blockSize = 256, int cutoffSize = reduceScale * 2, class T>
int parallel_sum_recu(T const *arr, int n) {
    if (n > cutoffSize) {  // 当递归到一定程度时，使用CPU串行求和
        std::vector<int, CudaAllocator<int>> sum(n / reduceScale);
        parallel_sum_kernel_recu<blockSize><<<n / reduceScale, blockSize>>>(sum.data(), arr, n);
        return parallel_sum_recu(sum.data(), n / reduceScale);
    } else {
        checkCudaErrors(cudaDeviceSynchronize());
        T final_sum = 0;
        for (int i = 0; i < n; i++) {
            final_sum += arr[i];
        }
        return final_sum;
    }
}

__global__ void parallel_sum_tls(int *sum, int const *arr, int n) {
    int local_sum = 0;
    for (int i = blockDim.x * blockIdx.x + threadIdx.x;
         i < n; i += blockDim.x * gridDim.x) {
        local_sum += arr[i];
    }
    atomicAdd(sum, local_sum);
}


int main() {
    {
        /**
         * @brief 无原子的解决方案：sum变成数组
         * 
         * 首先声明sum比原来小1024倍的数组
         * 然后在GPU上启动n/1024个线程，每个线程计算一个sum[i]
         * 因为每个线程都写入了不同的地址，所以不存在任何冲突
         * 
         * 最终可将缩小后的数组在CPU上计算
         */
        int n = 1<<24;
        std::vector<int, CudaAllocator<int>> arr(n);
        std::vector<int, CudaAllocator<int>> sum(n / 1024);
        for (int i = 0; i < n; i++) {
            arr[i] = std::rand() % 4;
        }
        TICK(parallel_sum);
        parallel_sum<<<n / 1024 / 128, 128>>>(sum.data(), arr.data(), n);
        checkCudaErrors(cudaDeviceSynchronize());
        int final_sum = 0;
        for (int i = 0; i < n / 1024; i++) {
            final_sum += sum[i];
        }
        TOCK(parallel_sum);
        printf("1.result: %d\n", final_sum);
    }

    {
        /**
         * @brief 第二种优化方法：先读取到局部数组，然后分步缩减
         * 
         * 上一个方法的问题：每个线程实际上是串行的过程，数据非常依赖 local_sum += arr[j]，下一个时刻的local_sum依赖上一个时刻的local_sum
         * 要消除这种依赖可以使用parallel_sum_local，这样每个for内部无依赖，从而并行计算
         * 对于CPU而言是SIMD指令集并行，虽然GPU没有(((
         */
        int n = 1<<24;
        std::vector<int, CudaAllocator<int>> arr(n);
        std::vector<int, CudaAllocator<int>> sum(n / 1024);
        for (int i = 0; i < n; i++) {
            arr[i] = std::rand() % 4;
        }
        TICK(parallel_sum_local);
        parallel_sum_local<<<n / 1024 / 128, 128>>>(sum.data(), arr.data(), n);
        checkCudaErrors(cudaDeviceSynchronize());
        int final_sum = 0;
        for (int i = 0; i < n / 1024; i++) {
            final_sum += sum[i];
        }
        TOCK(parallel_sum_local);
        printf("2.result: %d\n", final_sum);
    }

    {
        /**
         * @brief 板块内存共享，复杂度为O(log(n))的并行求和算法
         * 
         * 这个方法可能导致错误的值，有可能一个线程执行到if(j<32)而另一个线程还在if(j<64)的地方
         * 因为当发生内存数据等待的时候，SM会切换到另一个线程
         */
        int n = 1<<24;
        std::vector<int, CudaAllocator<int>> arr(n);
        std::vector<int, CudaAllocator<int>> sum(n / 1024);
        for (int i = 0; i < n; i++) {
            arr[i] = std::rand() % 4;
        }
        TICK(parallel_sum_block);
        parallel_sum_block<<<n / 1024, 1024>>>(sum.data(), arr.data(), n);
        checkCudaErrors(cudaDeviceSynchronize());
        int final_sum = 0;
        for (int i = 0; i < n / 1024; i++) {
            final_sum += sum[i];
        }
        TOCK(parallel_sum_block);
        printf("3.result: %d\n", final_sum);
    }

    {
        /**
         * @brief 对于上一个方法的改进：使用__syncthreads()同步线程
         * 这可以强制同步当前板块内的所有线程，保证local_sum写入成功
         * 
         * 注意：
         *  如果将__syncthreads()放在if之内，会发生未定义行为
         */
        int n = 1<<24;
        std::vector<int, CudaAllocator<int>> arr(n);
        std::vector<int, CudaAllocator<int>> sum(n / 1024);
        for (int i = 0; i < n; i++) {
            arr[i] = std::rand() % 4;
        }
        TICK(parallel_sum_block_sync);
        parallel_sum_block_sync<<<n / 1024, 1024>>>(sum.data(), arr.data(), n);  // 全并行
        checkCudaErrors(cudaDeviceSynchronize());
        int final_sum = 0;
        for (int i = 0; i < n / 1024; i++) {
            final_sum += sum[i];
        }
        TOCK(parallel_sum_block_sync);
        printf("4.result: %d\n", final_sum);
    }

    {
        /**
         * @brief 线程组(wrap)：32个线程为一组
         * 
         * SM对线程的调度是按照32个线程为一组的，也就是0~31为一组，32~63为一组，以此类推
         * 因此j<32之后就不需要再同步了
         * 
         * 同时为了防止编译器擅自优化，需要使用volatile关键字修饰local_sum
         */
        int n = 1<<24;
        std::vector<int, CudaAllocator<int>> arr(n);
        std::vector<int, CudaAllocator<int>> sum(n / 1024);
        for (int i = 0; i < n; i++) {
            arr[i] = std::rand() % 4;
        }
        TICK(parallel_sum_block_wrap);
        parallel_sum_block_wrap<<<n / 1024, 1024>>>(sum.data(), arr.data(), n);
        checkCudaErrors(cudaDeviceSynchronize());
        int final_sum = 0;
        for (int i = 0; i < n / 1024; i++) {
            final_sum += sum[i];
        }
        TOCK(parallel_sum_block_wrap);
        printf("5.result: %d\n", final_sum);
    }

    {
        /**
         * @brief 避免线程组产生线程分歧
         * 
         * 加if的初衷是为了节省不必要的计算，但是这会导致线程分歧
         * 因此可以把j<32的几个赋值合并到一起，反而更快
         */
        int n = 1<<24;
        std::vector<int, CudaAllocator<int>> arr(n);
        std::vector<int, CudaAllocator<int>> sum(n / 1024);
        for (int i = 0; i < n; i++) {
            arr[i] = std::rand() % 4;
        }
        TICK(parallel_sum_block_wrap_2);
        parallel_sum_block_wrap_2<<<n / 1024, 1024>>>(sum.data(), arr.data(), n);
        checkCudaErrors(cudaDeviceSynchronize());
        int final_sum = 0;
        for (int i = 0; i < n / 1024; i++) {
            final_sum += sum[i];
        }
        TOCK(parallel_sum_block_wrap_2);
        printf("6.result: %d\n", final_sum);
    }

    {
        /**
         * @brief 更进一步使用网格跨步循环
         * 
         * 因为共享内存的开销大，而全局内存arr的访问少，因此可以
         * 增大每个线程访问arr的次数，从而超过共享内存部分的时间
         */
        int n = 1<<24;
        std::vector<int, CudaAllocator<int>> arr(n);
        std::vector<int, CudaAllocator<int>> sum(n / 4096);
        for (int i = 0; i < n; i++) {
            arr[i] = std::rand() % 4;
        }
        TICK(parallel_sum_grid_loop);
        parallel_sum_grid_loop<<<n / 4096, 1024>>>(sum.data(), arr.data(), n);
        checkCudaErrors(cudaDeviceSynchronize());
        int final_sum = 0;
        for (int i = 0; i < n / 4096; i++) {
            final_sum += sum[i];
        }
        TOCK(parallel_sum_grid_loop);
        printf("7.result: %d\n", final_sum);
    }

    {
        /// 使用模板封装的版本
        int n = 1<<24;
        std::vector<int, CudaAllocator<int>> arr(n);
        std::vector<int, CudaAllocator<int>> sum(n / 4096);
        for (int i = 0; i < n; i++) {
            arr[i] = std::rand() % 4;
        }
        TICK(parallel_sum_temp);
        int final_sum = parallel_sum_temp(arr.data(), n);
        TOCK(parallel_sum_temp);
        printf("8.result: %d\n", final_sum);
    }

    {
        /// 使用递归的版本
        int n = 1<<24;
        std::vector<int, CudaAllocator<int>> arr(n);
        for (int i = 0; i < n; i++) {
            arr[i] = std::rand() % 4;
        }
        TICK(parallel_sum_recu);
        int final_sum = parallel_sum_recu(arr.data(), n);
        TOCK(parallel_sum_recu);
        printf("9.result: %d\n", final_sum);
    }

    {
        /// 使用线程局部存储的atomicAdd版本
        int n = 1<<24;
        std::vector<int, CudaAllocator<int>> arr(n);
        std::vector<int, CudaAllocator<int>> sum(1);
        for (int i = 0; i < n; i++) {
            arr[i] = std::rand() % 4;
        }
        TICK(parallel_sum_tls);
        parallel_sum_tls<<<n / 4096, 1024>>>(sum.data(), arr.data(), n);
        checkCudaErrors(cudaDeviceSynchronize());
        TOCK(parallel_sum_tls);
        printf("10.result: %d\n", sum[0]);
    }


    return 0;
}