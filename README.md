# 学习CUDA

## 1.准备环境
- [x] 操作系统：Windows10
- [x] CUDA：11.6
- [x] cuDNN：8.9
- [x] 最新的Nvidia显卡驱动

### 1.1.安装CUDA与cuDNN

- 推荐在windows上使用 [**scoop**](https://scoop.sh) ：

> Scoop是一个Windows下的命令行包管理器，类似于Linux下的apt-get或yum。使用Scoop，你可以方便地安装、升级和卸载各种软件包，而无需手动下载和安装。Scoop的包管理方式类似于Homebrew，它使用GitHub上的存储库来管理软件包。Scoop支持自动更新和版本控制，可以轻松地管理多个版本的软件包。Scoop还支持自定义存储库和桶，可以方便地添加自己的软件包。Scoop是一个开源项目，可以在GitHub上找到它的源代码和文档。

```powershell
scoop bucket add versions
scoop install versions/cuda11.6
```

- 下载 [`cuDNN`](链接：https://pan.baidu.com/s/1ZeVygsBDv3qrlr_mXLQ38Q?pwd=kgaa) 并解压到cuda安装目录的根目录，默认安装路径为 `C:\Users\用户名\scoop\apps\cuda11.6\11.6.2` 

- current文件夹相当于快捷方式，可使用 `scoop reset` 切换版本：

```powershell
scoop reset cuda11.6
```

### 1.2 编辑器环境

- 推荐使用vscode

- 插件列表

```txt
C/C++ Extension Pack:
	C/C++
	C/C++ Themes
	CMake
	CMake Tools
	
Darcula Theme
```

- 工作区配置

```json
{    
	"C_Cpp.errorSquiggles": "disabled",
    "C_Cpp.codeAnalysis.clangTidy.enabled": true,
    "C_Cpp.autocomplete": "default",
    "C_Cpp.default.intelliSenseMode": "windows-msvc-x64",
    "C_Cpp.inlayHints.autoDeclarationTypes.enabled": true,
    "C_Cpp.inlayHints.autoDeclarationTypes.showOnLeft": true,
    "C_Cpp.intelliSenseEngine": "default",
    "C_Cpp.intelliSenseEngineFallback": "enabled",
    "C_Cpp.experimentalFeatures": "enabled",
    "C_Cpp.default.configurationProvider": "ms-vscode.cmake-tools"
}
```

