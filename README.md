# 学习CUDA

## 1.准备环境
- [x] 操作系统：Windows11
- [x] CUDA：12.6
- [x] cuDNN：8.8.0
- [x] 最新的Nvidia显卡驱动

### 1.1.安装CUDA与cuDNN

<!-- - 推荐在windows上使用 [**scoop**](https://scoop.sh) ：

> Scoop是一个Windows下的命令行包管理器，类似于Linux下的apt-get或yum。使用Scoop，你可以方便地安装、升级和卸载各种软件包，而无需手动下载和安装。Scoop的包管理方式类似于Homebrew，它使用GitHub上的存储库来管理软件包。Scoop支持自动更新和版本控制，可以轻松地管理多个版本的软件包。Scoop还支持自定义存储库和桶，可以方便地添加自己的软件包。Scoop是一个开源项目，可以在GitHub上找到它的源代码和文档。

```powershell
scoop bucket add versions
scoop install versions/cuda11.6
```

- 下载 [`cuDNN`](https://pan.baidu.com/s/1tcxxyhBh1wl5_toUFbcMGw?pwd=2013) 并解压到cuda安装目录的根目录，默认安装路径为 `C:\Users\用户名\scoop\apps\cuda11.6\11.6.2` 

- current文件夹相当于快捷方式，可使用 `scoop reset` 切换版本：

```powershell
scoop reset cuda11.6
``` -->

- 资源获取 [CUDA12.6 && cuDNN8.8](https://pan.baidu.com/s/1paZ-MSXFU-ubKce14qWDsQ?pwd=qgyz)
- 安装过程：略
- 安装完成务必重启电脑！！！

### 1.2 文件结构
```
1.HelloWorld：新手村
```

## 2.编译
```powershell
.\build.bat
```

## 3.参考资料
- 官方：
    - [CUDA官方文档](https://docs.nvidia.com/cuda/) 
    - [NVIDIA/cuda-samples](https://github.com/NVIDIA/cuda-samples) 
    - [CUDA Code Samples](https://developer.nvidia.com/cuda-code-samples) 
    - [CUDA Zone](https://developer.nvidia.com/cuda-zone) 
    - [基于CUDA的应用程序](https://developer.nvidia.com/cuda-action-research-apps) 
- 电子书：
    - [《CUDA并行程序设计》](https://pan.baidu.com/s/16Q-lNmrZIrXqYjTBeBArnQ?pwd=pj27) 
    - [CUDA Professional](https://github.com/brucefan1983/CUDA-Programming) 
    - [更多...](https://pan.baidu.com/s/1paZ-MSXFU-ubKce14qWDsQ?pwd=qgyz) 
- 相关视频：
    - https://www.bilibili.com/video/BV16b4y1E74f 
    - https://www.bilibili.com/video/BV1vJ411D73S 
- 其他：
    - [OpenCL](https://www.khronos.org/opencl) 
    - [C++ Accelerated Massive Parallelism](https://learn.microsoft.com/zh-cn/cpp/parallel/amp/cpp-amp-cpp-accelerated-massive-parallelism?view=msvc-170) 
    - [Algorithms for Modern Hardware](https://en.algorithmica.org/hpc/) 


## Commit 规范
> [https://fe-notes.yunyoujun.cn/common/dev/#commit-message](https://fe-notes.yunyoujun.cn/common/dev/#commit-message)

- **build** ：对构建系统或者外部依赖项进行了修改
- **ci** ：对CI配置文件或脚本进行了修改
- **docs** ：对文档进行了修改
- **feat** ：增加新的特征
- **fix** ：修复bug
- **pref** ：提高性能的代码更改
- **refactor** ：既不是修复bug也不是添加特征的代码重构
- **style** ：不影响代码含义的修改，比如空格、格式化、缺失的分号等
- **test** ：增加确实的测试或者矫正已存在的测试