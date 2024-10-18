#include <cuda_runtime.h>
#include <cstdio>

__global__ void sayGlobalHello()
{
    printf("hello world!\n");
}

__device__ void sayDeviceHello()
{
    printf("Hello from GPU!\n");
}

__host__ __device__ void saySayHello()
{
#ifdef __CUDA_ARCH__ // 判断是否在 GPU 上运行
    printf("Hello from GPU! %d\n", __CUDA_ARCH__);
#else
    printf("Hello from CPU!\n");
#endif
}

__global__ void call_in_global()
{
    printf("call __host__ __device__ in __global__: ");
    saySayHello();  // 此时进入 __CUDA_ARCH__ 分支
}

__host__ void call_in_host()
{
    printf("call __host__ __device__ in __host__: ");
    saySayHello();  // 此时进入 else 分支
}

__global__ void callDevice()
{
    printf("call __device__ function in __global__: ");
    sayDeviceHello();
}

int main(int argc, char* argv[])
{
    sayGlobalHello<<<1, 1>>>();           // 直接调用 __global__ 运行在 GPU 上 
    //printf -> (Hello world!)
    cudaDeviceSynchronize();

    saySayHello();                  // 直接调用 __host__ __device__ 运行在 CPU 上 
    //printf -> (Hello from CPU!)
    callDevice<<<1, 1>>>();         // __global__ 正确调用同名的 __device__ 函数 运行在 GPU 上 
    //printf -> (call __device__ function in __global__: Hello from GPU!)
    cudaDeviceSynchronize();

    call_in_global<<<1, 1>>>();     // __global__ 调用 __host__ __device__ 函数, 运行在 GPU 上 
    //printf -> (call __host__ __device__ in __global__: Hello from GPU! 750)
    cudaDeviceSynchronize();
    call_in_host();                 // __host__ 调用 __host__ __device__ 函数, 运行在 CPU 上 
    //printf -> (call __host__ __device__ in __host__: Hello from CPU!)
    return 0;
}