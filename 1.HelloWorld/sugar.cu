// template<
//     class T,
//     class Allocator = std::allocator<T>
// > class vector;

// vector有两个参数，第一个是类型T，第二个是分配器Allocator。默认分配器是std::allocator<T>。
// 分配器负责分配和释放内存，初始化T对象等。它有两个成员函数：
// T* allocate(size_t n) 
// void deallocate(T* p, size_t n)。

#include <cstdio>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include <vector>

template<class T>
struct CudaAllocator {
    using value_type = T;
    T *allocate(size_t n) {
        T *ptr = nullptr;
        checkCudaErrors(cudaMallocManaged(&ptr, n * sizeof(T)));
        return ptr;
    }

    void deallocate(T *ptr, size_t n) {
        checkCudaErrors(cudaFree(ptr));
    }

    /**
     * @brief vector在初始化时会调用所有元素的无参构造函数，使用construct函数以跳过无参构造（避免初始化为0）
     * 这样可以避免在CPU上低效的零初始化，提高性能
     * 
     * is_pod_v 是一个C++17的特性，用于判断一个类型是否是POD类型（Plain Old Data）例如：int, float, char, struct A {int a; float b;}等
     */
    template<class... Args>
    void construct(T *ptr, Args &&... args) {
        if constexpr (!(sizeof...(Args) == 0 && std::is_pod_v<T>)) {
            // 无参且是POD类型的反，即有参或不是POD类型便调用构造函数
            ::new((void *)ptr) T(std::forward<Args>(args)...);
        }
    }
};

__global__ void kernel(int *arr, int n) {
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        arr[i] = i;
    }
}

/// kernel可以是一个模板函数，这体现了CUDA对C++的完全支持
template <int N, class T>
__global__ void t_kernel(T *arr) {
    for (int i = blockDim.x * blockIdx.x + threadIdx.x;
         i < N; i += blockDim.x * gridDim.x) {
        arr[i] = i;
    }
}


/**
 * @brief 核函数可以接受函子functor（仿函数），实现函数式编程
 * 注意：
 * 1. Func 不可以是Func const &，因为无法从GPU访问指向CPU的内存
 * 2. 做参数的函数必须有成员函数operator()，即functor类，而不能是独立的函数
 * 3. 这个函数必须标记为__device__，否则会在CPU上运行
 */
template <class Func>
__global__ void parallel_for(int n, Func func) {
    for (int i = blockDim.x * blockIdx.x + threadIdx.x;
         i < n; i += blockDim.x * gridDim.x) {
        func(i);
    }
}
struct MyFunctor {
    __device__ void operator()(int i) const {
        printf("number %d\n", i);
    }
};

int main() {
    {
        int n = 65536;
        std::vector<int, CudaAllocator<int>> arr(n);

        kernel<<<32, 128>>>(arr.data(), n);
        checkCudaErrors(cudaDeviceSynchronize());
        // for (int i = 0; i < n; i++) {
        //     printf("arr[%d] = %d\n",i, arr[i]);
        // }
    }

    {
        constexpr int n = 65536;  // n 需要是编译器常量
        std::vector<int, CudaAllocator<int>> arr(n);
        t_kernel<n><<<32, 128>>>(arr.data());
        checkCudaErrors(cudaDeviceSynchronize());
        // for (int i = 0; i < n; i++) {
        //     printf("arr[%d] = %d\n",i, arr[i]);
        // }
    }

    {
        int n = 65536;
        parallel_for<<<32, 128>>>(n, MyFunctor{});  // 用一个函数对象来代替lambda
        checkCudaErrors(cudaDeviceSynchronize());
    }

    {
        // 用lambda表达式来代替函子
        int n = 65536;
        parallel_for<<<32, 128>>>(n, [] __device__(int i) {
            printf("(lambda)number %d\n", i);
        });
        checkCudaErrors(cudaDeviceSynchronize());
    }

    {
        // 当lambda表达式使用[&]捕获变量时会发生错误，因为捕获到的是CPU堆栈上arr本身，而不是arr所指向的内存地址（GPU）
        int n = 65536;
        std::vector<int, CudaAllocator<int>> arr(n);  // arr本身在CPU上，而数据在GPU上
        // parallel_for<<<32, 128>>>(n, [&] __device__(int i) {arr[i] = i;});  // 错误
        // parallel_for<<<32, 128>>>(n, [=] __device__(int i) {arr[i] = i;});  // 也有问题，因为vector是深拷贝，这样会把arr拷贝到GPU上，而不是浅拷贝其起始地址指针
        parallel_for<<<32, 128>>>(n, [arr = arr.data()] __device__(int i) {arr[i] = i;});  // 正确，只捕获arr的指针
        checkCudaErrors(cudaDeviceSynchronize());
        // for (int i = 0; i < n; i++) {
        //     printf("(lambda)arr[%d] = %d\n",i, arr[i]);
        // }
    }

    return 0;
}