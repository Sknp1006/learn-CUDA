#include <cstdio>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include <thrust/universal_vector.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <thrust/generate.h>

template <class Func>
__global__ void parallel_for(int n, Func func) {
    for (int i = blockDim.x * blockIdx.x + threadIdx.x;
         i < n; i += blockDim.x * gridDim.x) {
        func(i);
    }
}

int main() {
    {    
        int n = 65536;
        float a = 3.14f;
        thrust::universal_vector<float> x(n);  // universal_vector会自动在同一内存上分配
        thrust::universal_vector<float> y(n);  // 省去自己写CudaAllocator
        for (int i = 0; i < n; i++) {
            x[i] = std::rand() * (1.f / RAND_MAX);
            y[i] = std::rand() * (1.f / RAND_MAX);
        }
        parallel_for<<<n / 512, 128>>>(n, [a, x = x.data(), y = y.data()] __device__ (int i) {
            x[i] = a * x[i] + y[i];
        });
        checkCudaErrors(cudaDeviceSynchronize());
        // for (int i = 0; i < n; i++) {
        //     printf("x[%d] = %f\n", i, x[i]);
        // }
    }

    /**
     * 除了universal_vector，还有device_vector和host_vector
     * 
     * device_vector: 用于在设备上分配内存
     * host_vector: 用于在主机上分配内存
     * 
     * 通过赋值运算符，可以将host_vector和device_vector之间的数据拷贝（自动调用cudaMemcpy）
     * 
     */
    {
        int n = 65536;
        float a = 3.14f;
        thrust::host_vector<float> x_host(n);
        thrust::host_vector<float> y_host(n);
        for (int i = 0; i < n; i++) {
            x_host[i] = std::rand() * (1.f / RAND_MAX);
            y_host[i] = std::rand() * (1.f / RAND_MAX);
        }
        thrust::device_vector<float> x_dev = x_host;  // host_vector -> device_vector
        thrust::device_vector<float> y_dev = x_host;  // host_vector -> device_vector
        parallel_for<<<n / 512, 128>>>(n, [a, x_dev = x_dev.data(), y_dev = y_dev.data()] __device__ (int i) {
            x_dev[i] = a * x_dev[i] + y_dev[i];
        });
        x_host = x_dev;  // device_vector -> host_vector
        // for (int i = 0; i < n; i++) {
        //     printf("x[%d] = %f\n", i, x_host[i]);
        // }
    }

    /**
     * thrust::generate(begin, end, generator)对标准库的std::generate，用于生成随机数并写入[begin, end)区间
     * 前两个参数是迭代器，第三个参数是一个函数对象，用于生成随机数
     */
    {
        int n = 65536;
        float a = 3.14f;
        thrust::host_vector<float> x_host(n);
        thrust::host_vector<float> y_host(n);
        auto float_rand = [] {
            return std::rand() * (1.f / RAND_MAX);
        };
        thrust::generate(x_host.begin(), x_host.end(), float_rand);
        thrust::generate(y_host.begin(), y_host.end(), float_rand);
        thrust::device_vector<float> x_dev = x_host;
        thrust::device_vector<float> y_dev = x_host;
        parallel_for<<<n / 512, 128>>>(n, [a, x_dev = x_dev.data(), y_dev = y_dev.data()] __device__ (int i) {
            x_dev[i] = a * x_dev[i] + y_dev[i];
        });
        x_host = x_dev;
        // for (int i = 0; i < n; i++) {
        //     printf("x[%d] = %f\n", i, x_host[i]);
        // }
    }

    /**
     * thrust::for_each(begin, end, func)对标准库的std::for_each，用于对[begin, end)区间的元素进行操作
     * 
     * 此外还有thrust::reduce、thrust::sort、thrust::find_it、thrust::count_if、thrust::reverse、thrust::inclusive_scan等函数
     */
    {
        int n = 65536;
        float a = 3.14f;
        thrust::host_vector<float> x_host(n);
        thrust::host_vector<float> y_host(n);
        thrust::for_each(x_host.begin(), x_host.end(), [] (float &x) {
            x = std::rand() * (1.f / RAND_MAX);
        });
        thrust::for_each(y_host.begin(), y_host.end(), [] (float &y) {
            y = std::rand() * (1.f / RAND_MAX);
        });
        thrust::device_vector<float> x_dev = x_host;
        thrust::device_vector<float> y_dev = x_host;
        thrust::for_each(x_dev.begin(), x_dev.end(), [] __device__ (float &x) {
            x += 100.f;
        });
        thrust::for_each(x_dev.cbegin(), x_dev.cend(), [] __device__ (float const &x) {
            printf("%f\n", x);
        });
        parallel_for<<<n / 512, 128>>>(n, [a, x_dev = x_dev.data(), y_dev = y_dev.data()] __device__ (int i) {
            x_dev[i] = a * x_dev[i] + y_dev[i];
        });
        x_host = x_dev;
        for (int i = 0; i < n; i++) {
            printf("x[%d] = %f\n", i, x_host[i]);
        }
    }

    /**
     *  thrust::make_counting_iterator是计数迭代器，用于生成[begin, end)区间的整数序列
     * 
     *  thrust::for_each(
        thrust::make_counting_iterator(0), 
        thrust::make_counting_iterator(10), [] __device__ (int i) {
        printf("%d\n", i);
        });
     * 
     */

    /**
     * thrust::make_zip_iterator是zip迭代器，用于将多个迭代器打包成一个迭代器（thrust::tuple）
     * 
     *  thrust::for_each(
        thrust::make_zip_iterator(x_dev.begin(), y_dev.cbegin()),
        thrust::make_zip_iterator(x_dev.end(), y_dev.cend()),
        [a] __device__ (auto const &tup) {
        auto &x = thrust::get<0>(tup);
        auto const &y = thrust::get<1>(tup);
        x = a * x + y;
        });
     * 
     */

    return 0;
}
