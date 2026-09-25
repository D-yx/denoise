# Wave-U-Net 语音增强方法的实验设置与结果

## 1. 实验目的与复现范围

本实验在 Edinburgh VoiceBank+DEMAND 和 MS-SNSD 两个数据集上复现时域 Wave-U-Net 语音增强方法。网络实现来自项目内 `WAVE_U_NET/src/Wave-U-Net-Pytorch`，训练与评测由根目录的 `run_denoising.py` 统一管理；Edinburgh 使用 3 张 GPU，MS-SNSD 使用 4 张 GPU，并通过 PyTorch DistributedDataParallel（DDP）进行单机多卡训练。两个数据集分别训练独立模型，不共享参数。

Edinburgh 与 MS-SNSD 实验均已完成训练和全量测试。MS-SNSD 最终采用同域、说话人与噪声文件互斥的 v5 数据配方；该结果用于检验 Wave-U-Net 本身，不等同于官方 `clean_test/noise_test` OOD benchmark。

## 2. 复现环境

### 2.1 硬件环境

| 项目 | 配置 |
|---|---|
| GPU | 5 × NVIDIA GeForce RTX 4090 |
| 单卡显存 | 24,564 MiB |
| Edinburgh 训练用卡数 | 3 |
| MS-SNSD long-form 训练用卡数 | 4 |
| CPU | 2 × Intel Xeon Silver 4310 @ 2.10 GHz |
| CPU 核心/线程 | 24 核、48 线程 |
| NUMA 节点 | 2 |
| 内存 | 188 GiB |
| NVIDIA 驱动 | 580.173.02 |

训练启动脚本在启动前连续采样 GPU 利用率与显存占用，按两者之和选择当前负载最低的指定数量 GPU。该机制只负责选卡，不设置利用率或显存阈值；实际训练仍可能受到服务器上其他任务的资源竞争影响。

### 2.2 软件环境

实验使用 Conda 环境 `denoise`，环境定义保存在 `environment.yml`。

| 软件 | 版本 |
|---|---:|
| Linux | 服务器原生环境 |
| Python | 3.11.16 |
| PyTorch | 2.5.1 |
| PyTorch CUDA runtime | 12.4 |
| cuDNN | 9.1.0 |
| NumPy | 1.26.4 |
| SciPy | 1.17.1 |
| SoundFile | 0.14.0 |
| PESQ | 0.0.4 |
| pystoi | 0.4.1 |

## 3. 数据集与预处理

### 3.1 Edinburgh VoiceBank+DEMAND

Edinburgh 数据集已提供文件名一一对应的 clean/noisy 语音：训练集 11,572 对，测试集 824 对。所有音频在预处理阶段读取为单声道 `float32`，由 48 kHz 使用多相滤波重采样至 16 kHz，并以 `.npy` 保存，避免每个 epoch 重复读取 48 kHz WAV 和重复重采样，也避免额外的 PCM 量化。

预处理后数据位于：

```text
.prepared/wave-u-net-edinburgh-16000hz/
├── train/clean
├── train/noisy
├── test/clean
└── test/noisy
```

缓存共占用约 4.4 GB。实测预处理时间约 56 s，其中训练集约 22 s，测试集约 33 s，使用 8 个 CPU worker。缓存数据与原在线重采样结果经过逐样本比较，结果完全一致。当前训练阶段读取 train 缓存；测试阶段仍从 Edinburgh 原始配对 WAV 读取并在线重采样，因而测试缓存目前不参与指标计算。

训练前使用固定随机种子 0 打乱 11,572 对训练数据，然后按 90%/10% 划分：约 10,414 对用于训练，约 1,158 对用于验证。该划分是文件级随机划分，并非说话人互斥划分。

### 3.2 MS-SNSD同域 long-form 预处理

MS-SNSD 原始 `clean_train/noise_train` 包含 23,075 条干净语音和128条噪声。早期实验直接使用官方 `clean_test/noise_test`，但其文件组织、电平、峰值和静音统计与训练池存在明显差异。最终 v5 配方从同一原始池按说话人和噪声文件互斥划分 train/validation/test，再使用完全一致的合成器生成 long-form 波形对：

1. 按说话人组织短 utterance，并拼接为至少 10 s 的连续语音；
2. 相邻 utterance 之间插入 0.2 s 静音；
3. clean 与 noise 分别归一化至 -25 dBFS RMS，并独立施加 0.7–1.3 增益；
4. 训练中每个 long-form clean 覆盖 \(\{0,10,20,30,40\}\) dB，测试集均衡覆盖五档；
5. 若混合峰值超过 0.99，则对 clean 和 noisy 同比缩放，保证监督目标与输入保持严格幅度对应；
6. 使用固定随机种子 0，确保噪声选择、截取位置和 SNR 可复现；
7. 47/5/6名说话人分别进入训练、验证、测试，集合互斥；
8. 102/13/13个噪声文件分别用于训练、验证、测试，集合互斥。

预处理后的数据统计如下：

| 划分 | 数量 | 最短时长 | 中位时长 | 最长时长 |
|---|---:|---:|---:|---:|
| long-form 训练与验证 mixture | 27,295 | ≥10 s | 约11.6 s | — |
| 实际训练 | 24,510 | ≥10 s | — | — |
| 验证 | 2,785 | ≥10 s | — | — |
| 同域测试 | 682 | ≥10 s | 约11.7 s | — |

缓存位于 `.prepared/ms-snsd-long/`，完成标记为 `recipe=in-domain-speaker-noise-disjoint-v5`。训练与测试的 clean RMS、静音比例、时长和峰值统计接近；训练脚本会拒绝旧配方缓存，checkpoint 也记录数据配方以防误恢复。

### 3.3 两个数据集采用不同长度处理策略的原因

Wave-U-Net 当前配置每次需要 159,717 samples 输入，即约 9.98 s，但只输出约 1.03 s 波形。因此，原始文件长度决定了模型在一次前向传播中看到多少真实上下文，以及输入两端需要补多少零。两个数据集的实测时长分布如下：

| 数据集划分 | 文件数 | 中位时长 | 小于 9.98 s 的比例 |
|---|---:|---:|---:|
| Edinburgh 训练 | 11,572 | 2.68 s | 99.78% |
| Edinburgh 测试 | 824 | 2.33 s | 100% |
| MS-SNSD 原始训练 utterance | 23,075 | 2.75 s | 99.75% |
| MS-SNSD v5 同域测试 | 682 | 约11.7 s | 0% |

Edinburgh 的训练和测试文件都以 2–4 s 短语音为主。二者进入 Wave-U-Net 时都会在感受野两端产生相似的大范围零填充，因此训练和测试具有一致的上下文及边界条件。Edinburgh 还直接提供同一数据制作流程下、时间严格对齐的 noisy/clean 文件对，不需要在项目中重新确定噪声片段、电平或 SNR。基于这一点，本实验保留每条 Edinburgh utterance 的独立性，不进行跨文件拼接。

MS-SNSD 的原始训练语音同样较短，但其测试语音全部超过 10 s，若直接按原始 utterance 训练，训练和测试将形成不同的上下文结构。与此同时，MS-SNSD 只提供独立的 clean 和 noise 素材，需要由合成器构造监督对。数据集自带配置明确指定 `audio_length=10` s 和 `silence_length=0.2` s，即先拼接短 utterance，再生成 long-form clean/noisy 对。因此，本实验对 MS-SNSD 采用至少 10 s 的拼接策略，使训练、验证和测试的时间上下文、电平归一化及 SNR 生成过程保持一致。

这两种处理方式并不表示 Wave-U-Net 在所有 Edinburgh 类数据上都无需拼接。决定因素是训练集与测试集的文件长度及边界条件是否一致：如果将 Edinburgh 测试语音改为 10–16 s 的连续长录音，也应同步为训练集构造匹配的 long-form 样本；反之，如果 MS-SNSD 的训练和测试均由相同分布的短文件组成，则可采用与 Edinburgh 相同的逐文件处理方式。

## 4. Wave-U-Net 模型

本实验保持项目中的 Wave-U-Net 算法主体不变。模型直接接收单声道时域波形并输出增强后的单声道波形，不使用 STFT 或手工声学特征。

| 设置 | 数值 |
|---|---:|
| 下采样/上采样层级 | 12 |
| 初始通道数 | 24 |
| 第 \(i\) 层通道数 | \(24(i+1)\) |
| 卷积核宽度 | 15 |
| Stride | 2 |
| 每层卷积深度 | 1 |
| 卷积归一化 | normal（无 BN/GN） |
| 重采样 | fixed sinc-based filtering |
| 输出模式 | 单模型输出 clean |
| 可训练参数 | 24,297,913 |
| 目标输出长度 | 16,000 samples |
| 实际几何输出长度 | 16,413 samples（约 1.026 s） |
| 实际输入长度 | 159,717 samples（约 9.982 s） |
| 左/右上下文 | 各 71,652 samples（约 4.478 s） |

实际输出长度由 Wave-U-Net 的 valid convolution 和上下采样几何共同确定，因此略大于配置中的 16,000 samples。

## 5. 训练设置

### 5.1 优化设置

| 设置 | 数值 |
|---|---:|
| Epochs | 100 |
| 每卡 batch size | 4 |
| Edinburgh GPU 数 | 3 |
| Edinburgh 全局 batch size | 12 |
| MS-SNSD GPU 数 | 4 |
| MS-SNSD 全局 batch size | 16 |
| 优化器 | Adam |
| 学习率 | \(1\times10^{-4}\) |
| 损失函数 | 波形 L1 loss |
| 学习率调度 | 无 |
| Early stopping patience | 20 epochs |
| DataLoader workers | 0/进程 |
| 随机种子 | 0（各 DDP rank 使用 `seed + rank`） |

每个训练样本随机选择一个输出起点，截取 16,413 samples 的 clean 目标，并按模型左右上下文截取或补零得到 159,717 samples 的 noisy 输入。验证集使用确定性起点。模型选择依据为完整验证集的平均 L1 loss；`best.pt` 保存验证损失最低的模型，`last.pt` 保存最近一轮状态。

### 5.2 多卡训练

多卡训练使用 NCCL 后端和 PyTorch DDP，每张 GPU 对应一个进程。`DistributedSampler` 将训练和验证数据分片；每轮开始调用 `set_epoch(epoch)` 保证各轮 shuffle 不同。训练和验证损失通过 `all_reduce` 汇总，只有 rank 0 保存 checkpoint。

| 数据集 | 每轮训练样本 | GPU | 每卡 batch | 每轮约更新次数 | 每轮验证 batch |
|---|---:|---:|---:|---:|---:|
| Edinburgh | 约 10,414 | 3 | 4 | 868 | 97 |
| MS-SNSD v5 | 24,510 | 4 | 4 | 1,532 | 175 |

### 5.3 实测训练耗时

Edinburgh 的 `last.pt` 首次创建于 2026-09-24 12:48:40，epoch 100 写入时间为 16:30:50，对应包含训练、验证和 checkpoint 写入的墙钟窗口约 3 h 42 min。最后 11 个 epoch 从最佳模型写入到最终模型共耗时约 22 min 22 s，即稳定阶段约 2.03 min/epoch；按稳定速度折算，纯 100 轮训练约 3 h 23 min。服务器共享负载会使总墙钟时间高于该折算值。

MS-SNSD v5 使用4张RTX 4090训练，最佳模型来自 epoch 77，验证 L1 为0.002729。训练在 epoch 97 因连续20轮未改善而正常早停；最终测试只加载 epoch 77 的 `best.pt`。

| 数据集 | GPU | 状态 | 最佳 epoch | 最佳验证 L1 | 训练耗时 |
|---|---:|---|---:|---:|---:|
| Edinburgh | 3 | 已完成 100 epochs | 89 | 0.005983 | 墙钟窗口约 3 h 42 min |
| MS-SNSD v5 | 4 | epoch 97早停 | 77 | 0.002729 | 约5小时 |

## 6. 测试流程与评价指标

测试只加载对应数据集的 `best.pt`。Edinburgh checkpoint 为 `.prepared/wave-u-net-edinburgh/best.pt`；MS-SNSD checkpoint 为 `.prepared/wave-u-net-ms-snsd-long/best.pt`。

长音频按模型实际输出长度 16,413 samples 依次处理。每个输出片段前后各提供 71,652 samples 上下文；音频边界处使用零填充。模型输出按时间顺序直接拼接并裁剪回输入长度。测试为单 GPU、`eval()` 和 `torch.no_grad()` 模式。

评价前在 \(\pm100\) ms 范围通过归一化互相关估计增强语音与 clean 参考的固定延迟，然后裁剪共同有效区间。报告：

- 全局 SNR；
- SSNR：30 ms 帧长、15 ms 帧移，单帧裁剪至 \([-10,35]\) dB；
- SI-SDR；
- 16 kHz 宽带 PESQ；
- STOI；
- 平均检测延迟。

Edinburgh 全量结果写入 `results/wave-u-net_edinburgh/`；MS-SNSD v5 的682条同域测试结果写入 `results/wave-u-net_ms-snsd-indomain/`。增强 WAV 可重新生成，不纳入 Git。

## 7. 实验结果

### 7.1 已完成的 Edinburgh 结果

| 训练/测试数据集 | 测试文件数 | SNR (dB) | SSNR (dB) | SI-SDR (dB) | PESQ | STOI | 平均延迟 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Edinburgh | 824 | **19.592** | **10.611** | **19.539** | **2.522** | **0.935** | 0 ms |

最佳模型来自 epoch 89；其验证 L1 为 0.005983。epoch 100 的验证 L1 为 0.006085，因此全量测试正确使用 epoch 89，而非最后一轮模型。

### 7.2 MS-SNSD v5最终结果

| 训练/测试数据集 | 测试文件数 | SNR | SSNR | SI-SDR | PESQ | STOI | 平均延迟 |
|---|---:|---:|---:|---:|---:|---:|---:|
| MS-SNSD v5同域测试 | 682 | **25.704** | **17.680** | **25.647** | **3.018** | **0.872** | 0 ms |

MS-SNSD 输入基线为 SNR 19.967、SSNR 13.785、SI-SDR 19.968、PESQ 2.450、STOI 0.841；对应增益分别为 +5.738 dB、+3.896 dB、+5.679 dB、+0.567 和 +0.031。0/10/20/30 dB 档均显著改善，40 dB 档 SI-SDR 轻微下降0.245 dB但 PESQ/STOI仍提升。全部6名测试说话人均取得正的平均 SI-SDR、PESQ 和 STOI 增益。

## 8. Wave-U-Net 与 RNNoise/Hybrid 的数据处理差异

| 方面 | Wave-U-Net | RNNoise/Hybrid |
|---|---|---|
| 输入域 | 16 kHz 原始时域波形 | 48 kHz DSP 帧特征 |
| 输入表示 | 159,717 samples 波形窗口 | 每帧 65 维输入特征 |
| 监督目标 | 16,413 samples clean 波形 | 32 维频带增益 + 1 维 VAD |
| 数据预处理 | 重采样、配对、长段组织和 SNR 混合 | 构造连续 speech/background/foreground PCM 流，再由 `dump_features` 提取 98 维帧记录 |
| Edinburgh 噪声处理 | 直接使用官方 noisy/clean 对 | 用 `noisy-clean` 估计噪声，再拆成 background/foreground 流 |
| MS-SNSD 噪声处理 | 按目标 SNR直接生成 noisy/clean 波形对 | clean 和 noise 分别写入连续 PCM，由 RNNoise 特征生成器在线组合并产生频带监督 |
| 中间数据 | 16 kHz `.npy` 或 WAV 对；Edinburgh 缓存约 4.4 GB，MS-SNSD 缓存需本地生成 | 每数据集 7.84 GB `features.f32`，共 20,000,000 帧 |
| 时间上下文 | 单次输入约 9.98 s，输出约 1.03 s | 20 ms 分析窗、10 ms 帧移；训练序列为 2,000 帧（20 s） |
| 时延处理 | 非因果窗口，边界需要约 4.48 s 左右上下文；离线分段推理 | 面向实时因果处理，实测约 10 ms 固定延迟 |
| 数据长度敏感性 | 高；训练/测试窗口及补零模式必须一致 | 较低；统一转换为连续帧流后抽取固定长度序列 |
| 输出部署 | PyTorch 波形模型 | 量化权重并编译成 C 推理程序 |

Wave-U-Net 的输入感受野较长，因此预处理必须保证训练、验证与测试具有一致的时间上下文和边界条件。RNNoise 先把语音和噪声整理为长 PCM 流，再统一抽取 20 s 特征序列，因此天然弱化了原始文件长度差异。相对地，RNNoise 的数据准备更依赖其专用 DSP 前端和特征格式，而 Wave-U-Net 的监督信号保持为直观的波形对。

## 9. 可复现命令

### 9.1 Edinburgh

```bash
./preprocess_wave_u_net_edinburgh.sh
GPU_COUNT=3 ./train_wave_u_net_edinburgh.sh
conda run --no-capture-output -n denoise \
  python wave_u_net_edinburgh.py --stage test
```

### 9.2 MS-SNSD long-form

```bash
./preprocess_wave_u_net_ms_snsd.sh --force
GPU_COUNT=4 ./train_wave_u_net_ms_snsd.sh --no-resume --wave-epochs 100 --wave-batch-size 4 --workers 4
conda run --no-capture-output -n denoise \
  python wave_u_net_ms_snsd.py --stage test
```

若训练中断，默认会从对应 `last.pt` 恢复；使用 `--no-resume` 可从头开始。预处理使用完成标记检查文件数，训练不会接受部分生成的数据。

## 10. 局限与后续补充

1. Edinburgh 当前使用文件级随机验证划分，而 MS-SNSD long-form 使用说话人互斥划分，两者验证损失不应直接横向比较。
2. 服务器 GPU 为共享资源，动态选卡只能降低启动时冲突，无法保证训练全程独占；耗时应视为本服务器本次运行的墙钟实测。
3. MS-SNSD v5 为同域实验，不能与采用官方 `clean_test/noise_test` 的论文结果直接横向比较；官方目录应作为单独的 OOD benchmark。
4. MS-SNSD 结果同时报告 noisy-input 基线和 `Delta_*` 增益；逐文件结果可进一步按 SNR 和说话人统计。
5. Wave-U-Net 为非因果离线模型，不能将其时延与实时 RNNoise 的约 10 ms 算法延迟直接等同。
6. Edinburgh 与 MS-SNSD 的长度处理策略是根据各自训练/测试分布确定的实验设计，不能脱离文件时长、语音来源和合成流程直接互换。
