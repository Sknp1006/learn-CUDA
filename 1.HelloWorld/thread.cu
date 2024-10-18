#include <cstdio>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void norm_thread() {
    printf("Block %d of %d, Thread %d of %d\n",
           blockIdx.x, gridDim.x, threadIdx.x, blockDim.x);
    // blockIdx.x 表示当前线程所在的 block 的索引
    // gridDim.x 表示当前 grid 中 block 的数量
    // threadIdx.x 表示当前线程在 block 中的索引
    // blockDim.x 表示当前 block 中线程的数量
}

__global__ void flat_thread() {
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int tnum = blockDim.x * gridDim.x;
    printf("Flattened Thread %d of %d\n", tid, tnum);
}

__global__ void _3d_thread() {
    printf("Block (%d,%d,%d) of (%d,%d,%d), Thread (%d,%d,%d) of (%d,%d,%d)\n",
        blockIdx.x, blockIdx.y, blockIdx.z,
        gridDim.x, gridDim.y, gridDim.z,
        threadIdx.x, threadIdx.y, threadIdx.z,
        blockDim.x, blockDim.y, blockDim.z);
}

__global__ void _2d_thread() {
    printf("Block (%d,%d) of (%d,%d), Thread (%d,%d) of (%d,%d)\n",
           blockIdx.x, blockIdx.y,
           gridDim.x, gridDim.y,
           threadIdx.x, threadIdx.y,
           blockDim.x, blockDim.y);
}

__global__ void another() {
    printf("another: Thread %d of %d\n", threadIdx.x, blockDim.x);
}

__global__ void kernel() {
    printf("kernel: Thread %d of %d\n", threadIdx.x, blockDim.x);
    int numthreads = threadIdx.x * threadIdx.x + 1;
    another<<<1, numthreads>>>();  // numthreads: 0,1,2
    printf("kernel: called another with %d threads\n", numthreads);
}

int main() {
    norm_thread<<<2, 3>>>();  // 2 blocks, 3 threads per block
    // Block 1 of 2, Thread 0 of 3
    // Block 1 of 2, Thread 1 of 3
    // Block 1 of 2, Thread 2 of 3
    // Block 0 of 2, Thread 0 of 3
    // Block 0 of 2, Thread 1 of 3
    // Block 0 of 2, Thread 2 of 3
    cudaDeviceSynchronize();

    flat_thread<<<2, 3>>>();
    // Flattened Thread 3 of 6
    // Flattened Thread 4 of 6
    // Flattened Thread 5 of 6
    // Flattened Thread 0 of 6
    // Flattened Thread 1 of 6
    // Flattened Thread 2 of 6
    cudaDeviceSynchronize();

    _3d_thread<<<dim3(2, 1, 1), dim3(2, 2, 2)>>>();  // 2 blocks, 8 threads per block
    // Block (1,0,0) of (2,1,1), Thread (0,0,0) of (2,2,2)
    // Block (1,0,0) of (2,1,1), Thread (1,0,0) of (2,2,2)
    // Block (1,0,0) of (2,1,1), Thread (0,1,0) of (2,2,2)
    // Block (1,0,0) of (2,1,1), Thread (1,1,0) of (2,2,2)
    // Block (1,0,0) of (2,1,1), Thread (0,0,1) of (2,2,2)
    // Block (1,0,0) of (2,1,1), Thread (1,0,1) of (2,2,2)
    // Block (1,0,0) of (2,1,1), Thread (0,1,1) of (2,2,2)
    // Block (1,0,0) of (2,1,1), Thread (1,1,1) of (2,2,2)
    // Block (0,0,0) of (2,1,1), Thread (0,0,0) of (2,2,2)
    // Block (0,0,0) of (2,1,1), Thread (1,0,0) of (2,2,2)
    // Block (0,0,0) of (2,1,1), Thread (0,1,0) of (2,2,2)
    // Block (0,0,0) of (2,1,1), Thread (1,1,0) of (2,2,2)
    // Block (0,0,0) of (2,1,1), Thread (0,0,1) of (2,2,2)
    // Block (0,0,0) of (2,1,1), Thread (1,0,1) of (2,2,2)
    // Block (0,0,0) of (2,1,1), Thread (0,1,1) of (2,2,2)
    // Block (0,0,0) of (2,1,1), Thread (1,1,1) of (2,2,2)
    cudaDeviceSynchronize();

    _2d_thread<<<dim3(2, 1, 1), dim3(3, 2, 1)>>>();  // 2 blocks, 6 threads per block
    // Block (1,0) of (2,1), Thread (0,0) of (3,2)
    // Block (1,0) of (2,1), Thread (1,0) of (3,2)
    // Block (1,0) of (2,1), Thread (2,0) of (3,2)
    // Block (1,0) of (2,1), Thread (0,1) of (3,2)
    // Block (1,0) of (2,1), Thread (1,1) of (3,2)
    // Block (1,0) of (2,1), Thread (2,1) of (3,2)
    // Block (0,0) of (2,1), Thread (0,0) of (3,2)
    // Block (0,0) of (2,1), Thread (1,0) of (3,2)
    // Block (0,0) of (2,1), Thread (2,0) of (3,2)
    // Block (0,0) of (2,1), Thread (0,1) of (3,2)
    // Block (0,0) of (2,1), Thread (1,1) of (3,2)
    // Block (0,0) of (2,1), Thread (2,1) of (3,2)
    cudaDeviceSynchronize();

    kernel<<<1, 3>>>();  // 1 block, 3 threads per block
    // kernel: Thread 0 of 3
    // kernel: Thread 1 of 3
    // kernel: Thread 2 of 3
    // kernel: called another with 1 threads
    // kernel: called another with 2 threads
    // kernel: called another with 5 threads
    // another: Thread 0 of 1
    // another: Thread 0 of 2
    // another: Thread 1 of 2
    // another: Thread 0 of 5
    // another: Thread 1 of 5
    // another: Thread 2 of 5
    // another: Thread 3 of 5
    // another: Thread 4 of 5
    cudaDeviceSynchronize();
    return 0;
}