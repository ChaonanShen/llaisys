# LLAISYS 作业报告

## 一、作业概述

本次作业完成了 LLAISYS 从 Tensor、CPU 算子、Qwen2 模型推理到 NVIDIA CUDA 后端的主要实现。底层计算使用 C/C++ 和 CUDA 实现，Python 部分主要负责接口封装、模型权重加载和测试。

## 二、实现内容

### 1. Tensor

- 通过 Runtime API 实现从 Host 加载数据。
- 实现 Tensor 连续性判断。
- 实现 `view`、`permute` 和 `slice`，操作过程中共享底层 Storage，不进行不必要的数据复制。
- 对 shape、stride、offset 和主要参数进行了合法性检查。

### 2. CPU 算子

完成了以下 CPU 算子：

- `add`
- `argmax`
- `embedding`
- `linear`
- `rearrange`
- `rms_norm`
- `rope`
- `self_attention`
- `swiglu`

主要浮点计算路径支持 Float32、Float16 和 BFloat16。

### 3. Qwen2 模型推理

- 实现 Qwen2 模型的创建、销毁、权重访问、状态重置和推理接口。
- 通过 Python 封装加载 safetensors 模型权重。
- 实现贪心解码和 KV Cache 复用。
- 支持在 CPU 和 NVIDIA GPU 设备上运行。

### 4. NVIDIA CUDA 后端

- 实现设备管理、Stream、显存分配和内存复制等 CUDA Runtime API。
- 为全部作业算子增加 NVIDIA CUDA 实现。
- 使用 `--nv-gpu=y` 控制 CUDA 后端的编译。
- CUDA 默认关闭，因此没有 CUDA 环境时仍可正常构建和测试 CPU 版本。

## 三、复现方法

### 1. CPU 构建与测试

环境要求：Xmake、支持 C++17 的编译器、Python 3.9 或更高版本。

```bash
xmake f --nv-gpu=n -c
xmake
xmake install
pip install ./python

python test/test_runtime.py --device cpu
python test/test_tensor.py
python test/ops/add.py
python test/ops/argmax.py
python test/ops/embedding.py
python test/ops/linear.py
python test/ops/rms_norm.py
python test/ops/rope.py
python test/ops/self_attention.py
python test/ops/swiglu.py
python test/test_infer.py --test
```

如果没有通过 `--model` 指定本地模型路径，推理测试会自动下载 `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`。使用本地模型的命令如下：

```bash
python test/test_infer.py --model /path/to/model --test
```

### 2. NVIDIA CUDA 构建与测试

环境要求：上述 CPU 环境、NVIDIA GPU、CUDA Toolkit，以及支持 CUDA 的 PyTorch。

```bash
xmake f --nv-gpu=y -cv
xmake
xmake install
pip install ./python

python test/test_runtime.py --device nvidia
python test/ops/add.py --device nvidia
python test/ops/argmax.py --device nvidia
python test/ops/embedding.py --device nvidia
python test/ops/linear.py --device nvidia
python test/ops/rms_norm.py --device nvidia
python test/ops/rope.py --device nvidia
python test/ops/self_attention.py --device nvidia
python test/ops/swiglu.py --device nvidia
python test/test_infer.py --model /path/to/model --test --device nvidia
```

## 四、测试结果与平台状态

| 平台 | 配置 | 验证结果 |
| --- | --- | --- |
| Windows x64 | CPU、MSVC、默认构建 | GitHub Actions 构建及 Assignment 0～3 测试通过 |
| Ubuntu x64 | CPU、GCC、默认构建 | GitHub Actions 构建及 Assignment 0～3 测试通过 |
| macOS x64 | CPU、Clang、默认构建 | 本地构建、Runtime、Tensor 和全部 CPU 算子测试通过 |
| Linux + NVIDIA Tesla V100 | CUDA 构建，启用 `--nv-gpu=y` | CUDA 后端及第四部分相关测试验证通过 |

Windows 和 Ubuntu 的结果由提交 `fb2fec8` 对应的 GitHub Actions [`Build and test` 工作流](https://github.com/ChaonanShen/llaisys/actions/runs/31611376655)验证。

NVIDIA CUDA 后端已经在 **Linux + NVIDIA Tesla V100** 环境下完成构建和测试，验证通过。
