#include <cstdio>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include <vector>
#include "CudaAllocator.h"
#include "ticktock.h"

/// 效率不高
template <class T>
__global__ void parallel_transpose(T *out, T const *in, int nx, int ny)
{
    int linearized = blockIdx.x * blockDim.x + threadIdx.x;
    int y = linearized / nx;
    int x = linearized % nx;
    if (x >= nx || y >= ny)
        return;
    out[y * nx + x] = in[x * nx + y];
}

/// 循环分块
template <class T>
__global__ void parallel_transpose_1(T *out, T const *in, int nx, int ny)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny)
        return;
    out[y * nx + x] = in[x * nx + y];
}

/// 采用共享内存
template <int blockSize, class T>
__global__ void parallel_transpose_shared(T *out, T const *in, int nx, int ny)
{
    int x = blockIdx.x * blockSize + threadIdx.x;
    int y = blockIdx.y * blockSize + threadIdx.y;
    if (x >= nx || y >= ny)
        return;
    __shared__ T tmp[blockSize * blockSize];
    int rx = blockIdx.y * blockSize + threadIdx.x;
    int ry = blockIdx.x * blockSize + threadIdx.y;
    tmp[threadIdx.y * blockSize + threadIdx.x] = in[ry * nx + rx];
    __syncthreads();
    out[y * nx + x] = tmp[threadIdx.x * blockSize + threadIdx.y];
}

/// 通过跨步+1避免区块冲突
template <int blockSize, calss T>
__global__ void parallel_transpose_shared_1(T *out, T const *in, int nx, int ny)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny)
        return;
    __shared__ T tmp[(blockSize + 1) * blockSize];
    int rx = blockIdx.y * blockSize + threadIdx.x;
    int ry = blockIdx.x * blockSize + threadIdx.y;
    tmp[threadIdx.y * (blockSize + 1) + threadIdx.x] = in[ry * nx + rx];
    __syncthreads();
    out[y * nx + x] = tmp[threadIdx.x * (blockSize + 1) + threadIdx.y];
}

int main()
{
    {
        // TICK(total)
        int nx = 1 << 14, ny = 1 << 14;
        std::vector<int, CudaAllocator<int>> in(nx * ny);
        std::vector<int, CudaAllocator<int>> out(nx * ny);
        for (int i = 0; i < nx * ny; i++)
        {
            in[i] = i;
        }
        TICK(parallel_transpose);
        parallel_transpose<<<nx * ny / 1024, 1024>>>(out.data(), in.data(), nx, ny);
        checkCudaErrors(cudaDeviceSynchronize());
        TOCK(parallel_transpose); // 0.71s
        // for (int y = 0; y < ny; y++) {
        //     for (int x = 0; x < nx; x++) {
        //         if (out[y * nx + x] != in[x * nx + y]) {
        //             printf("Wrong At x=%d,y=%d: %d != %d\n", x, y,
        //                 out[y * nx + x], in[x * nx + y]);
        //             return -1;
        //         }
        //     }
        // }
        // printf("All Correct!\n");
        // TOCK(total);  // 22s
    }

    {
        /// 采用循环分块的方式，提高缓存局域性
        int nx = 1 << 14, ny = 1 << 14;
        std::vector<int, CudaAllocator<int>> in(nx * ny);
        std::vector<int, CudaAllocator<int>> out(nx * ny);
        for (int i = 0; i < nx * ny; i++)
        {
            in[i] = i;
        }
        TICK(parallel_transpose_1);
        parallel_transpose_1<<<dim3(nx / 32, ny / 32, 1), dim3(32, 32, 1)>>>(out.data(), in.data(), nx, ny);
        checkCudaErrors(cudaDeviceSynchronize());
        TOCK(parallel_transpose_1); // 0.70s
        // for (int y = 0; y < ny; y++) {
        //     for (int x = 0; x < nx; x++) {
        //         if (out[y * nx + x] != in[x * nx + y]) {
        //             printf("Wrong At x=%d,y=%d: %d != %d\n", x, y,
        //                 out[y * nx + x], in[x * nx + y]);
        //             return -1;
        //         }
        //     }
        // }
        // printf("All Correct!\n");
    }

    {
        /**
         * @brief 使用共享内存
         *
         * 上一个方法对in的读取是存在跨步的，而GPU喜欢顺序读取，这样跨步就不高效了
         * 但是做矩阵转置必然有一个跨步
         * 可以把in分块，按跨步方式读取，而块内则是连续地读————从低效全局的内存读到高效的共享内存中，然后在共享内存中跨步地读，连续地写到out指向的低效全局内存中
         * 这样跨步发生在共享内存中，而不是全局内存中，因此更快
         */
        int nx = 1 << 14, ny = 1 << 14;
        std::vector<int, CudaAllocator<int>> in(nx * ny);
        std::vector<int, CudaAllocator<int>> out(nx * ny);
        for (int i = 0; i < nx * ny; i++)
        {
            in[i] = i;
        }
        TICK(parallel_transpose_shared);
        parallel_transpose_shared<32><<<dim3(nx / 32, ny / 32, 1), dim3(32, 32, 1)>>>(out.data(), in.data(), nx, ny);
        checkCudaErrors(cudaDeviceSynchronize());
        TOCK(parallel_transpose_shared);
        // for (int y = 0; y < ny; y++) {
        //     for (int x = 0; x < nx; x++) {
        //         if (out[y * nx + x] != in[x * nx + y]) {
        //             printf("Wrong At x=%d,y=%d: %d != %d\n", x, y,
        //                 out[y * nx + x], in[x * nx + y]);
        //             return -1;
        //         }
        //     }
        // }
        // printf("All Correct!\n");
    }

    return 0;
}

/// 不知道为什么，实验没达到教程预期的效果