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