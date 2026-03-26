---
layout: post
title: Introduction to CUDA & GPU Architecture
date: 2026-03-25 00:00:00
description: 《CUDA编程：基础与实践》阅读笔记
tags: CUDA
categories: ai-infra
tabs: true
mermaid:
  enabled: true
  zoomable: true
thumbnail:
toc:
  beginning: true
  sidebar: left
pretty_table: true
tikzjax: true
pseudocode: true
---

[official doc - CUDA](https://docs.nvidia.com/cuda/) 包括：

- installation guides
- programming guides
  - [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
  - [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
  - 针对最近几个 GPU 架构进行优化的指南：[Volta](https://docs.nvidia.com/cuda/volta-tuning-guide/) [Ampere](https://docs.nvidia.com/cuda/ampere-tuning-guide/) [Hopper](https://docs.nvidia.com/cuda/hopper-tuning-guide/) [Blackwell](https://docs.nvidia.com/cuda/blackwell-tuning-guide/)
- CUDA API References
  - [CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/)
  - [CUDA Driver API](https://docs.nvidia.com/cuda/cuda-driver-api/)
  - [CUDA Math API](https://docs.nvidia.com/cuda/cuda-math-api/)
  - ...

# 1. GPU 简介

GPU 版本号 X.Y 表征**计算能力**，即 GPU 支持哪些硬件特性和指令集；

浮点数运算峰值（FLOPS）、显存（即 GPU 内存）带宽和显存容量决定**计算性能**。

计算能力 ≠ 计算性能。

CUDA 提供了两层 API：CUDA driver API 和 CUDA runtime API。后者更高级，可读性更好。

CUDA 开发环境组件：

- 应用程序层：开发者主要工作区域，写 CUDA kernel 等
- CUDA Runtime API：cudaMalloc、cudaMemcpy、流管理、事件同步等高层封装
- CUDA 库：cuBLAS、cuDNN、cuFFT 等
- CUDA Driver API：底层控制、上下文管理等

![](/assets/img/13.jpg)

CUDA 版本与 GPU 版本不同。GPU 版本本质上表示 GPU 硬件架构的版本，而 CUDA 版本是 GPU 软件开发平台的版本

nvidia-smi：[official doc](https://developer.nvidia.com/system-management-interface)
- 第一行有 Nvidia Driver 版本和 CUDA 工具箱的版本
- 仅有一个 GPU，型号为 Tesla T4，设备号为 0
  - 可以在运行 CUDA 程序之前设置环境变量 CUDA_VISIBLE_DEVICES 来选定一个 CPU
- Performance State = P8：性能档位，范围为 P0~P12，P8 属于低功耗，因为此时没有任何计算任务在跑
- Persistence Mode = Off：驱动不会在没有任务时保持常驻
- Display Active = Off：没有连接显示器
- Memory Usage 显存占用 = 2MiB，因为此时没有计算任务在跑
- Uncorr ECC = 0：ECC 内存
- Compute Mode = Default：允许多个进程同时共享该 GPU。对应的其他模式还有 Exclusive Process（独占进程）和 Prohibited（禁止计算）
- MIG Mode = N/A：Tesla T4 不支持 Multi-Instance GPU 功能

``` bash
Wed Mar 25 16:19:08 2026       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.288.01             Driver Version: 535.288.01   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  Tesla T4                       Off | 00000000:00:07.0 Off |                    0 |
| N/A   32C    P8               9W /  70W |      2MiB / 15360MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|  No running processes found                                                           |
+---------------------------------------------------------------------------------------+
```

nvcc：[official doc](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/)

- 是编译器驱动，不直接编译代码，而是在背后协调多个真正的编译器来完成工作。nvcc 的工作：
  1. 拆分 `.cu` 文件为 GPU 部分和 CPU 部分
  2. 对于 GPU 部分，调用 cicc 生成 PTX 伪汇编代码，再调用 ptxas 生成二进制的 cubin 目标代码
      - 源代码 -> PTX 代码：`-arch=compute_XY` 指定虚拟架构的计算能力
      - PTX -> cubin：`-code=sm_ZW` 指定真实架构（Turing、Ampere 等等）的计算能力（Z.W），对应的可执行文件只能在主版本号为 Z、次版本号 >= W 的 GPU 中运行
  3. 对于 CPU 部分，调用 g++(Linux) 或 cl.exe(Windows) 编译
  4. 链接两部分，得到最终可执行文件

一个编译好的 CUDA 可执行文件，内部可以同时包含：

- 针对不同 GPU 型号预先编译好的 cubin（直接能跑）
- 一份通用的 PTX 中间代码（需要运行时再编译）

运行时驱动会检测当前 GPU，优先找匹配的 cubin 直接用，找不到就拿 PTX 即时编译一份。

``` bash
-gencode arch=compute_35,code=sm_35   # 给 3.5 算力的 GPU 用
-gencode arch=compute_50,code=sm_50   # 给 5.0 算力的 GPU 用
-gencode arch=compute_60,code=sm_60   # 给 6.0 算力的 GPU 用
-gencode arch=compute_60,code=compute_60   # `code=compute_60` 表示不生成 cubin，而是把 PTX 代码嵌入可执行文件。当用户的 GPU 算力高于 6.0（比如 7.5 的 T4），找不到匹配的 cubin，驱动就拿这份 PTX 即时编译成适合当前 GPU 的 cubin 来跑。
# 因为这份 PTX 是按 compute_60 的特性集生成的，不包含 7.0 以后才有的新指令（比如 Tensor Core）。虽然能跑，但新硬件的高级特性用不上。
```

``` bash
-arch=sm_XY
等价于
-gencode arch=compute_XY,code=sm_XY
-gencode arch=compute_XY,code=compute_XY
```

核函数中不支持 C++ 的 iostream。

# 2. CUDA Thread

一块 GPU 中有很多计算核心，可以支持多个 thread。

三括号表明启动线程数：<<<线程块个数 grid size，每个线程块中的线程数 block size>>>

- 核函数内部，用两个内建变量来保存 grid size 和 block size：
  - gridDim.x = grid size
  - blockDim.x = block size
- 核函数中预定义了如下标识线程的内建变量：
  - blockIdx.x：一个 thread 在一个 grid 中的线程块指标，[0, gridDim.x - 1]
  - threadIdx.x：一个 thread 在一个 block 中的线程指标，[0, blockDim.x - 1]
- dim3 下的唯一线程标识和线程索引：
  - 线程索引：
    - nx = blockDim.x * blockIdx.x + threadIdx.x
    - ny = ...
    - nz = ...
  - 线程标识：
    - 线程在块内的 id：tid = (threadIdx.z * blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x
    - 线程在 grid 内的 block id：bid = (blockIdx.z * gridDim.x * gridDim.y) + (blockIdx.y * gridDim.x) + blockIdx.x
- grid size 在 x、y、z 三个方向的最大允许值分别为 $2^{31} - 1, 65535, 65535$
- block size 在 x、y、z 三个方向上的最大允许值为 $1024, 1024, 64$，且线程块总的大小（即三者乘积）不能大于 1024

线程束（thread warp）：
- 一个线程束是同一个线程块中相邻的 warpSize 个线程。warpSize 是内建变量

cudaDeviceSynchronize：这个 Runtime API 的功能是同步 host 和 device，从而 flush buffer（调用输出函数时，输出流是先存放在 buffer 的）

启动线程数大于计算核心数时才能更充分地利用 GPU 中的计算资源，因为线程数远多于核心数时，GPU 调度器善用零开销切换 warp 来延迟隐藏，核心几乎没有空闲时间，GPU 利用率高。
- 只要启动线程足够多，SM 驻留线程（某一时刻 SM 上正在活跃的线程）就名额能被被填满，Occupancy（= SM 驻留 warp 数 / SM 最大 warp 数）才能达到上限
- Occupancy 是为 memory-bound 场景设计的优化手段，是延迟隐藏能力的前提条件。Occupancy 不是越高越好（e.g. 为了启动线程数足够多，使用 `__launch_bounds__` 控制每个线程分配的寄存器上限，导致 register spilling，访存延迟增加，延迟隐藏也变差）