#include <cstdio>
#include <cuda_runtime.h>
#include "helper_cuda.h"

// a __global__ function must have a void return type 因为 GPU 是异步的,不会等待函数返回
// __global__ int return_42() {
//     return 42;
// }

// 使用指针传递参数也是令人失望的,会触发类似 CPU 的 segmentation fault
__global__ void pret_kernel(int *pret) { 
    *pret = 42;
}

int main() { 
    // 出现异常后,程序会终止,故对此注释
    // {
    //     int ret = 0;
    //     pret_kernel<<<1, 1>>>(&ret);
    //     cudaError_t err = cudaDeviceSynchronize();
    //     printf("error code = %d  error message = %s\n", err, cudaGetErrorString(err));
    //     //printf -> (error code = 700  error message = an illegal memory access was encountered)
    //     printf("ret = %d\n", ret);
    //     //printf -> (ret = 0)
    //     checkCudaErrors(cudaDeviceSynchronize());  // 更方便的捕获和打印异常
    //     //CUDA error at C:\Users\Administrator\Documents\GitHub\learn-CUDA\1.HelloWorld\memory.cu:24 code=700(cudaErrorIllegalAddress) "cudaDeviceSynchronize()"
    // }

    {
        int *pret;
        checkCudaErrors(cudaMalloc(&pret, sizeof(int)));  // 使用 cudaMalloc 分配显存
        pret_kernel<<<1, 1>>>(pret);
        checkCudaErrors(cudaDeviceSynchronize());
        // printf("result = %d\n", *pret);  // 依然不能获取正确的值,并导致崩溃,因为 pret 指向的是显存
        cudaFree(pret);
    }

    // 使用 cudaMalloc + cudaMemcpy 从显存拷贝到内存
    {
        int ret = 0;
        int *pret;
        checkCudaErrors(cudaMalloc(&pret, sizeof(int)));
        pret_kernel<<<1, 1>>>(pret);
        // checkCudaErrors(cudaDeviceSynchronize());  // cudaMemcpy 会自动同步,此处可以省略
        checkCudaErrors(cudaMemcpy(&ret, pret, sizeof(int), cudaMemcpyDeviceToHost));  // 使用 cudaMemcpy 从显存拷贝到内存
        printf("result = %d\n", ret);
        //printf -> (result = 42)
        cudaFree(pret);
    }

    // 使用 cudaMallocManaged 分配统一内存
    {
        int *pret;
        checkCudaErrors(cudaMallocManaged(&pret, sizeof(int)));
        pret_kernel<<<1, 1>>>(pret);
        checkCudaErrors(cudaDeviceSynchronize());  // 由于没有了 cudaMemcpy,需要显式同步
        printf("result = %d\n", *pret);
        //printf -> (result = 42)
        cudaFree(pret);
    }

    return 0; 
}