# 笔记

## 参考教程
- https://www.bilibili.com/video/BV16b4y1E74f 


## CUDA关键字
- `__global__` 函数修饰符, 表明该函数在设备上执行, 由主机调用;
  - 所有的 `__global__` 必须是void类型, 返回值不会被使用;
  - 即使使用指针参数也无法改写, 因为指针参数是在主机内存中;
- `__device__` 函数修饰符, 表明该函数在设备上执行, 由设备调用;
- `__host__` 函数修饰符, 表明该函数在主机 , 由主机调用;

- `__inline__` 函数修饰符, 声明一个函数为内联函数(weak符号), 不论是CPU还是GPU都可以使用;GCC编译器则是 `__attribute__(("inline"))` 
- `__forceinline__` 函数修饰符, 强制内联函数, 不论是CPU还是GPU都可以使用;GCC编译器则是 `__attribute__(("always_inline"))` 
- `__noinline__` 函数修饰符, 禁止内联函数; 

- `__shared__` 修饰符, 表明该变量在共享内存中, 只能在设备代码中使用

## 一些记录

- 实际上GPU的版块相当于CPU线程, GPU的线程相当于CPU的SIMD;
 
- 在核函数调用另一个核函数,这是从Kelper架构开始支持的,使用时需要 set(CUDA_SEPARABLE_COMPILATION ON) 选项;
  <p>kernel launch from __device__ or __global__ functions requires separate compilation mode</p>

- checkCudaErrors 函数是一个宏定义, 用于检查CUDA函数的返回值, 如果返回值不为 cudaSuccess, 则打印错误信息并退出程序;
  ```c
  #define checkCudaErrors(err) __checkCudaErrors (err, __FILE__, __LINE__)
  ```

- cudaMemcpy 会自动同步,可以省略 cudaDeviceSynchronize() 函数;

- cudaMemcpy 函数的第四个参数是枚举类型, 表示数据传输的方向:
  - cudaMemcpyHostToDevice: 从主机内存拷贝到设备内存;
  - cudaMemcpyDeviceToHost: 从设备内存拷贝到主机内存;
  - cudaMemcpyDeviceToDevice: 设备内存之间拷贝;
  - cudaMemcpyHostToHost: 主机内存之间拷贝;

- cudaMalloc 函数用于在设备上分配内存, cudaFree 用于释放设备上的内存;

- cudaMallocMangaed 函数用于在统一内存上分配内存:
  - 当从CPU访问时,会自动进行数据拷贝,无需手动调用cudaMemcpy,释放内存时使用cudaFree;
  - 统一内存有一定的开销,有条件尽量分离显存和内存;

- 除了sinf之外，还有__sinf低精度的函数，适合有性能要求的图形学任务;
- __fdividef(x, y)提供更快的浮点除法，精度相同，但在 2^126 < y < 2^128 时会得到错误结果;

- 一些编译器选项:
  - --use_fast_math: 自动将sinf替换为__sinf; 
  - --ftz=true: 将极小数（denormal）替换为0;
  - --prec-div=false: 降低除法精度，提高性能;
  - --prec-sqrt=false: 降低开方精度，提高性能;
  - --fmad: 默认开启，会自动把a*b+c替换为fma(a, b, c);
  - 开启 --use_fast_math 选项会自动开启以上所有选项;

### 关于CMP0146警告
```powershell
CMP0146
-------

.. versionadded:: 3.27

The ``FindCUDA`` module is removed.

The ``FindCUDA`` module has been deprecated since CMake 3.10.
CMake 3.27 and above prefer to not provide the module.
This policy provides compatibility for projects that have not been
ported away from it.

Projects using the ``FindCUDA`` module should be updated to use
CMake's first-class ``CUDA`` language support.  List ``CUDA`` among the
languages named in the top-level call to the ``project()`` command,
or call the ``enable_language()`` command with ``CUDA``.
Then one can add CUDA (``.cu``) sources directly to targets,
similar to other languages.

The ``OLD`` behavior of this policy is for ``find_package(CUDA)`` to
load the deprecated module.  The ``NEW`` behavior is for uses of the
module to fail as if it does not exist.

This policy was introduced in CMake version 3.27.  CMake version
3.28.3 warns when the policy is not set and uses ``OLD`` behavior.
Use the ``cmake_policy()`` command to set it to ``OLD`` or ``NEW``
explicitly.

.. note::
  The ``OLD`` behavior of a policy is
  ``deprecated by definition``
  and may be removed in a future version of CMake.
```
将 find_package(CUDA) 替换为 find_package(CUDAToolkit REQUIRED) 即可解决警告;

- 原子操作:
  - atomicAdd(dst, src): *dst += src;
  - atomicSub(dst, src): *dst -= src;
  - atomicOr(dst, src): *dst |= src;
  - atomicAnd(dst, src): *dst &= src;
  - atomicXor(dst, src): *dst ^= src;
  - atomicMax(dst, src): *dst = max(*dst, src);
  - atomicMin(dst, src): *dst = min(*dst, src);
  - atomicExch(dst, src): *dst = src;
  - atomicCAS(dst, compare, val): if (*dst == compare) *dst = val; return *dst;

- 用atomicCAS实现atomicAdd:
```c++
    __device__ __inline__ int my_atomic_add(int *dst, int src) {
    int old = *dst, expect;
    do {
        expect = old;
        old = atomicCAS(dst, expect, expect + src);  // 为什么要判断expect, 为了防止其他线程修改了dst的值;
    } while (expect != old);
    return old;
}

__global__ void parallel_sum(int *sum, int const *arr, int n) {
    int local_sum = 0;
    for (int i = blockDim.x * blockIdx.x + threadIdx.x;
         i < n; i += blockDim.x * gridDim.x) {
        local_sum += arr[i];
    }
    my_atomic_add(&sum[0], local_sum);
}
```

- atomicCAS可以实现任意原子操作:
> CAS非常影响性能，如无必要，尽量避免使用;
```c++
__device__ __inline__ int float_atomic_add(float *dst, float src) {
    int old = __float_as_int(*dst), expect;
    do {
        expect = old;
        old = atomicCAS((int *)dst, expect,
                    __float_as_int(__int_as_float(expect) + src));
    } while (expect != old);
    return old;
}
```

- SM(Streaming Multiprocess) 与板块(block)的关系
  - GPU是由多个流式处理器(SM)组成的, 每个SM可以处理一个或多个板块(block);
  - SM由多个SP(Streaming Processor)组成, 每个SP可以处理一个或多个线程;
  - 每个SM都有自己的共享内存;
  - 通常板块数量总是大于SM的数量;
  - GPU不会像CPU一样做时间片轮换，板块一旦被调度到一个SM就会一直执行，直到完成;没有保存和切换上下文的开销;
  - 一个SM可以同时执行多个板块，这时多个板块共用同一块共享内存;
  - 板块内部的每个线程则是被进一步调度到SM上的每个SP;

- 线程组分歧(wrap divergence)
  - GPU线程组中32个线程是绑在一起执行的，就像CPU的SIMD;
  - 因此出现分支if语句时，如果32个cond有真有假，则会导致两个分支都执行;
  - 不过在cond为假的线程在真分支会避免修改寄存器和访存，为了避免则会产生开销;
  - 因此建议GPU上的if尽可能32个线程都处于同一个分支，要么全为真，要么全为假，否则实际消耗两倍时间;
  - 避免修改寄存器和访存相当于CPU的SIMD指令_mm_blendv_ps和_mm_store_mask_ps，不过GPU这种SIMT的设计能自动处理分支和循环的分歧，这是门槛比CPU低的地方; ——王鑫磊

- BLS(block-local storage)
  - https://developer.download.nvidia.cn/assets/cuda/files/reduction.pdf

- block.cu的例子说明了atomicAdd的原理，实际上：
  - 编译器自动优化成了BLS数组求和，甚至比手写的更高效;
  - 其他原子操作同理;

- 板块中线程数量过多：寄存器打翻(register spill)
  - GPU线程的寄存器实际上也是一块较小而快的内存，称之为寄存器仓库(register file);
  - 板块内的所有线程都共用一个寄存器仓库，因此一个板块的线程数(blockDim)越多，每个线程可用的寄存器数就越少;
  - 当程序恰好用到了非常多的寄存器，超过了一个板块的寄存器数时，需要把一部分打翻到一级缓存中，这时对寄存器的访问就和访问一级缓存一样慢;
  - 若一级缓存依然不够，就会打翻到所有SM共用的二级缓存;
  - 此外，如果在线程局部分配一个数组，并通过动态下标访问，那无论如何都会打翻到一级缓存，因为寄存器不能动态寻址;
  - 对于Fermi架构，每个线程最多可以有63个寄存器，每个有4字节;

- 板块中线程数量过少：延迟隐藏(latency hiding)失效
  - 每个SM一次只能执行板块中的一个线程组(wrap)，也就是32个线程;
  - 当线程组陷入内存等待时，可以切换到另一个线程，这样一个wrap的内存延迟被多个wrap的计算隐藏了;
  - 因此当线程数量太少时，无法通过切换线程来隐藏延迟，导致低效;
  - 此外，最好让blockDim是32的倍数，这样可以充分利用GPU的SIMT架构;
  - 结论：对于使用寄存器较少、访存为主的核函数(如矢量加法)，使用大blockDim为宜，反之(光线追踪)使用较小blockDim为宜;

- GPU内存模型
 - per-thread local memory: 每个线程的本地内存(寄存器)，存储每个线程的局部变量;
 - per-block shared memory: 每个板块的共享内存，通过__shared__修饰符声明;
 - per-device global memory: 设备全局内存，在mian()中通过cudaMalloc分配;

- 共享内存：什么是区块(bank)
  - GPU的共享内存实际上是32块内存条并联而成(类比CPU的双通道);
  - 每个bank都可以独立的访问，他们每个时钟周期都可以读取一个int;
  - 然后他们把地址空间32分，第i根内存条负责addr%32==i的几个int存储，这样交错存储可以保证随机访问时，访存尽量分摊到32个区块上，速度提升32倍;
  - 比如：__shared__ int arr[1024];那么arr[0]在bank0，arr[1]在bank1，arr[32]在bank0，arr[33]在bank1，以此类推;
  
- 区块冲突(bank conflict)
  - bank的设计有一个问题，如果多个线程同时访问同一个bank，就会产生冲突，需要排队等待;
  - 例如：arr[0]和arr[32]在同一个bank，那么同时访问arr[0]和arr[32]的线程就会冲突;
  - 为了避免冲突，可以把跨步从32改为33，故意不对齐，这样线程0访问的是bank0，线程1访问arr[33]位于bank1，线程2访问arr[66]位于bank2，这样就避免了冲突;

- 优化手法总结：
  - 线程组分歧(wrap divergence): 尽量保证32个线程都处于同一个分支;
  - 延迟隐藏(latency hiding): 需要有足够的blockDim提供SM在陷入内存等待时调度到其他线程组;
  - 寄存器打翻(register spill): 如果核函数用到很多局部变量，可以考虑减少blockDim;
  - 共享内存(shared memory): 全局内存比较低效，多次使用的数据可以放到共享内存中;
  - 跨步访问(coalesced access): 建议先顺序读到共享内存，让高带宽的共享内存来承受跨步访问的开销;
  - 区块冲突(bank conflict): 避免多个线程同时访问同一个bank，可以通过故意不对齐来避免冲突;
  - Nvidia显卡的wrap大小是32，而AMD则是64;

