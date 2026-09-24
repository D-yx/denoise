# Hybrid语音增强方法的实验设置与结果

## 1. 实验目的

本实验在 Edinburgh VoiceBank+DEMAND 和 MS-SNSD 两个数据集上训练并评估 Hybrid 实时语音增强方法。实现基于 Xiph.Org RNNoise 官方主分支的 PyTorch 训练管线（代码版本 `70f1d256acd4b34a572f999a05c87bf00b67730d`），并使用官方 DSP 前端提取频带特征和生成监督目标。两个数据集分别训练独立模型，不共享参数。

## 2. 数据集与预处理

### 2.1 Edinburgh VoiceBank+DEMAND

Edinburgh 数据集采用配对的干净—含噪语音。训练集包含 11,572 对语音，测试集包含 824 对语音。测试集覆盖 bus、cafe、living、office 和 psquare 五种噪声，信噪比为 2.5、7.5、12.5 和 17.5 dB。

所有音频首先转换为 48 kHz、单声道信号。训练噪声由配对信号通过

\[
n = \operatorname{clip}(y_{\mathrm{noisy}}-y_{\mathrm{clean}},-1,1)
\]

提取，再交替写入 background-noise 和 foreground-noise 数据流。处理后的干净语音流时长约为 9.391 h，两路噪声流的时间轴长度也均为 9.391 h（其中非对应路段由零填充）。

### 2.2 MS-SNSD

MS-SNSD 数据集包含 23,075 条训练干净语音、128 条训练噪声、1,100 条测试干净语音和 51 条测试噪声。训练阶段直接将 `clean_train` 与 `noise_train` 转换为 48 kHz、单声道、16-bit PCM 数据流。噪声文件按顺序交替划分为 background 和 foreground 两路。处理后的干净语音流约为 19.031 h，background 和 foreground 噪声流分别约为 1.398 h 和 1.893 h。

测试对使用固定随机种子 0 生成。对每条干净语音随机选取噪声片段，并从 \(\{0,10,20,30,40\}\) dB 中等概率选取目标 SNR。合成信号若出现峰值越界，则统一缩放至有效幅度范围。合成在 16 kHz 下完成，Hybrid 推理前再重采样至 48 kHz。

### 2.3 RNNoise 训练特征

两个数据集均使用官方 `dump_features` 生成 10,000 个训练序列，每个序列包含 2,000 帧，即每个数据集共生成 20,000,000 帧。在 48 kHz 采样率下，处理帧移为 480 samples（10 ms），分析窗长为 960 samples（20 ms），因而训练帧的累计时间尺度约为 55.56 h。

每帧包含 98 个 32-bit 浮点数：65 维模型输入特征、32 维频带增益目标和 1 维 VAD 目标。每个数据集生成的 `features.f32` 大小为 7.84 GB（十进制）。

## 3. 模型与训练细节

### 3.1 模型结构

模型输入为 65 维 DSP 特征，首先经过两个核宽为 3 的一维卷积层，然后输入三层级联 GRU。条件特征尺寸为 128，GRU 隐藏尺寸为 384。输出头分别预测 32 个频带增益和 1 个 VAD 概率。模型共包含 2,884,769 个可训练参数。本次实验未启用结构化 GRU 稀疏化，最终部署模型使用官方脚本进行量化并编译至 C 推理程序。

### 3.2 优化设置

| 设置 | 数值 |
|---|---:|
| Epochs | 60 |
| Batch size | 8 |
| 序列长度 | 2,000 帧（20 s） |
| 每轮更新次数 | 1,250 |
| 总更新次数 | 75,000 |
| 优化器 | AdamW |
| 初始学习率 | \(1\times10^{-3}\) |
| Adam \(\beta\) | \((0.8,0.98)\) |
| Adam \(\epsilon\) | \(1\times10^{-8}\) |
| Weight decay | 0.01（PyTorch AdamW 默认值） |
| 学习率调度 | \(\mathrm{lr}_t=10^{-3}/(1+5\times10^{-5}t)\) |
| 末次更新学习率 | 约 \(2.105\times10^{-4}\) |
| 感知增益指数 \(\gamma\) | 0.25 |
| DataLoader workers | 0 |

损失函数由频带增益损失和 VAD 损失构成：

\[
\mathcal{L}=\mathcal{L}_{\mathrm{gain}}+10^{-3}\mathcal{L}_{\mathrm{VAD}}.
\]

其中增益损失在 \(\gamma=0.25\) 的感知域中计算，并对语音活动帧施加更高权重；VAD 分支使用加权二元交叉熵。学习率调度器在每次参数更新后执行一次，而非按 epoch 调整。

### 3.3 训练平台与耗时

训练在一台搭载 NVIDIA GeForce RTX 4050 Laptop GPU（6 GB 显存）的笔记本电脑上完成。主机处理器为 Intel Core i7-13700H（14 核 20 线程），内存为 31.6 GiB。软件环境为 Windows 11、Python 3.11.16、PyTorch 2.8.0+cu128、CUDA 12.8、NumPy 1.26.4 和 SciPy 1.17.1。

| 数据集 | 特征生成耗时 | 60 epochs 训练耗时 | 平均每轮 | 选中检查点 | 检查点损失 |
|---|---:|---:|---:|---:|---:|
| Edinburgh | 约 1 h 36 min | 约 7 h 39 min | 约 7.65 min | Epoch 60 | 0.003090 |
| MS-SNSD | 约 1 h 19 min | 约 8 h 59 min | 约 8.98 min | Epoch 57 | 0.042947 |

耗时由特征文件和 checkpoint 的文件时间戳恢复，精度约为 1 min。Edinburgh 的训练时间窗为 2026-09-22 21:16 至 2026-09-23 04:55；MS-SNSD 为 2026-09-23 14:15 至 23:14。该时间为包含数据读取、每轮 checkpoint 保存在内的墙钟时间，不是单独的 CUDA kernel 计时。

每轮训练后保存 checkpoint。最终对所有 checkpoint 的完整 epoch 平均训练损失进行比较，导出损失最低的模型。由于 RNNoise 官方训练器未配置独立验证集，此处“最佳 checkpoint”是指训练损失最低，而非验证集性能最优。

## 4. 评价方法

测试时使用量化后的 C 模型对 48 kHz 音频逐帧推理。RNNoise 引入约一帧的固定处理延迟，因此先在 \(\pm100\) ms 范围内通过互相关估计增强语音与干净参考之间的偏移，再裁剪两者的共同有效区间。两个数据集的平均检测延迟均约为 480 samples（10 ms）。

报告以下五项指标：全局信噪比（SNR）、分段信噪比（SSNR，30 ms 帧长和 15 ms 帧移，单帧结果裁剪至 \([-10,35]\) dB）、尺度不变信噪失真比（SI-SDR）、宽带 PESQ 和短时客观可懂度（STOI）。PESQ 在 16 kHz 下计算；其余波形指标在对齐后的信号上计算。

## 5. 实验结果

| 训练/测试数据集 | 测试文件数 | SNR (dB) | SSNR (dB) | SI-SDR (dB) | PESQ | STOI | 平均延迟 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Edinburgh | 824 | **12.641** | **6.028** | **12.310** | **2.106** | **0.889** | 9.994 |
| MS-SNSD | 1,100 | **12.596** | **8.775** | **12.327** | **2.177** | **0.920** | 9.998 |

MS-SNSD 上的 SSNR、PESQ 和 STOI 分别比 Edinburgh 高 2.747 dB、0.071 和 0.031，而两者的全局 SNR 与 SI-SDR 非常接近。这表明两个独立训练的模型均能保持稳定的波形重建能力，而 MS-SNSD 合成测试条件下的分帧噪声抑制和可懂度更高。但由于两个测试集的噪声类型、SNR 分布和录音条件不同，不应将数据集间的指标差异直接解释为模型结构优劣。

## 6. 讨论与局限

1. 实验使用的是现行 RNNoise 官方主分支的 PyTorch 架构，而非 2018 年初始 RNNoise 发布版的原始小型网络，因此不能在不加说明的情况下与早期 RNNoise 参数量或耗时直接等同。
2. checkpoint 选择依据是训练损失，尚缺少独立验证集。更严格的模型选择应从训练语料中划分说话人互斥的验证集，并按验证损失或验证 PESQ/STOI 选择 checkpoint。
3. 当前结果是增强语音的绝对指标，未在结果表中同时给出每个测试集的 noisy-input 基线。若用于正式投稿，应使用完全相同的对齐和指标实现计算 noisy 基线，并报告 \(\Delta\mathrm{SNR}\)、\(\Delta\mathrm{PESQ}\) 和 \(\Delta\mathrm{STOI}\)。
4. MS-SNSD 的测试语音由固定随机种子合成，可保证本项目内的重复性；但与其他工作比较时，必须确认对方使用了相同的噪声抽样、SNR 集合和峰值缩放方式。

## 7. 可复现性

两个实验的入口命令如下：

```powershell
C:\Users\Administrator\miniconda3\envs\rnnoise-cuda\python.exe hybrid_edinburgh.py
C:\Users\Administrator\miniconda3\envs\rnnoise-cuda\python.exe hybrid_ms_snsd.py
```

完整管线依次执行数据预处理、98 维特征生成、CUDA 训练、最佳 checkpoint 选择、权重量化、C 推理程序编译、测试、时延对齐和指标汇总。数值结果保存于 `results/hybrid_edinburgh/` 和 `results/hybrid_ms-snsd/`。
