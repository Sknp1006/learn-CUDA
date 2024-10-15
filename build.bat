@echo off
setlocal

:: 设置 CUDA 和 Visual Studio 环境变量
set CUDA_PATH=C:\Users\Administrator\scoop\apps\cuda11.6\current
set PATH=%CUDA_PATH%\bin;%PATH%
@REM call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"

:: 创建并进入构建目录
if not exist build (
    mkdir build
)
else (
    rmdir /s /q build
    mkdir build
)
cd build

:: 运行 CMake 配置命令
cmake -G "Visual Studio 17 2022" -A x64 ..

:: 运行 CMake 构建命令
cmake --build . --config Release --target install

endlocal
pause