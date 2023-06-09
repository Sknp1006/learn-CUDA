#include <stdio.h>
#include <stdlib.h>
#include <conio.h>

__global__ void what_is_my_id(unsigned int * const block,
    unsigned int * const thread,
    unsigned int * const warp,
    unsigned int * const calc_thread)
{
    // Thread id is block index * block size + thread offset into the block
    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;

    block[thread_idx] = blockIdx.x;
    thread[thread_idx] = threadIdx.x;

    // Calculate warp using built-in variable warpSize
    warp[thread_idx] = threadIdx.x / warpSize;
    calc_thread[thread_idx] = thread_idx;
}

#define ARRAY_SIZE 128
#define ARRAY_SIZE_IN_BYTES (sizeof(unsigned int) * (ARRAY_SIZE))

// Declare statically four arrays of ARRAY_SIZE each

unsigned int cpu_block[ARRAY_SIZE];
unsigned int cpu_thread[ARRAY_SIZE];
unsigned int cpu_warp[ARRAY_SIZE];
unsigned int cpu_calc_thread[ARRAY_SIZE];

int main()
{
    // Define total data element
    const unsigned int num_elem = 64;

    // Define grid and block size
    const unsigned int num_block = 2;
    const unsigned int num_thread = 32;
    char ch;

    // Declare pointers for GPU based params
    unsigned int * gpu_block;
    unsigned int * gpu_thread;
    unsigned int * gpu_warp;
    unsigned int * gpu_calc_thread;

    // Allocate GPU memory
    cudaMalloc((void**)&gpu_block, num_elem * sizeof(unsigned int));
    cudaMalloc((void**)&gpu_thread, num_elem * sizeof(unsigned int));
    cudaMalloc((void**)&gpu_warp, num_elem * sizeof(unsigned int));
    cudaMalloc((void**)&gpu_calc_thread, num_elem * sizeof(unsigned int));

    // Execute kernel
    what_is_my_id <<<num_block, num_thread >>> (gpu_block, gpu_thread, gpu_warp, gpu_calc_thread);

    // Copy back GPU results to CPU
    unsigned int cpu_block[num_elem];
    cudaMemcpy(cpu_block, gpu_block, num_elem * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    unsigned int cpu_thread[num_elem];
    cudaMemcpy(cpu_thread, gpu_thread, num_elem * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    unsigned int cpu_warp[num_elem];
    cudaMemcpy(cpu_warp, gpu_warp, num_elem * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    unsigned int cpu_calc_thread[num_elem];
    cudaMemcpy(cpu_calc_thread, gpu_calc_thread, num_elem * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(gpu_block);
    cudaFree(gpu_thread);
    cudaFree(gpu_warp);
    cudaFree(gpu_calc_thread);

    // Print results
    for (unsigned int i = 0; i < num_elem; ++i)
    {
        printf("Calculated Thread: %u - Block: %u - Warp %u - Thread %u\n",
            cpu_calc_thread[i], cpu_block[i], cpu_warp[i], cpu_thread[i]);
    }
    ch = getch();

    return 0;
}
