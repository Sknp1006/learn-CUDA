#include <cuda_runtime.h>
#include <cstdio>
#include "helper_cuda.h"
#include "CudaAllocator.h"
#include <vector>
#include "ticktock.h"


template<class Func>
__global__ void parallel_for(int n, Func func) {
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        func(i);
    }
}

int main() {
    // 经典案例：计算sin函数
    {
        int n = 65536;
        std::vector<float, CudaAllocator<float>> arr(n);
        parallel_for<<<32, 128>>>(n, [arr = arr.data()] __device__ (int i) {
            arr[i] = sinf(i);  // sin是double类型，sinf是float类型，使用float提升性能
        });
        checkCudaErrors(cudaDeviceSynchronize());
        for (int i = 0; i < 10; i++) {
            printf("diff = %f\n", arr[i] - sinf(i));  // 检查与CPU计算的差值
        }
    }

    // 对比cpu和gpu的性能
    {
        int n = 1<<25;
        std::vector<float, CudaAllocator<float>> gpu(n);
        std::vector<float> cpu(n);

        TICK(cpu_sinf);
        for (int i = 0; i < n; i++) {
            cpu[i] = sinf(i);
        }
        TOCK(cpu_sinf);

        TICK(gpu_sinf);
        parallel_for<<<n / 512, 128>>>(n, [gpu = gpu.data()] __device__ (int i) {
            gpu[i] = sinf(i);
        });
        checkCudaErrors(cudaDeviceSynchronize());
        TOCK(gpu_sinf);  // 异步执行，需要等待结束
        // cpu_sinf: 0.180258s
        // gpu_sinf: 0.001435s
    }

    {
        int n = 65536;
        float a = 3.14f;
        std::vector<float, CudaAllocator<float>> x(n);
        std::vector<float, CudaAllocator<float>> y(n);
        // 初始化x和y
        for (int i = 0; i < n; i++) {
            x[i] = std::rand() * (1.f / RAND_MAX);  // 0~1之间的随机数
            y[i] = std::rand() * (1.f / RAND_MAX);  // 0~1之间的随机数
        }
        parallel_for<<<n / 512, 128>>>(n, [a, x = x.data(), y = y.data()] __device__ (int i) {
            x[i] = a * x[i] + y[i];  // x = a * x + y
        });
        checkCudaErrors(cudaDeviceSynchronize());
        for (int i = 0; i < 10; i++) {
            printf("SAXPY x[%d] = %f\n", i, x[i]);
        }
    }

    return 0;
}