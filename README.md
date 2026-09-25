# Speech Denoising Reproduction

本仓库复现并比较两类语音降噪模型：

- **Hybrid / RNNoise**：Xiph.Org RNNoise 的 DSP 前端与 PyTorch GRU 模型；
- **Wave-U-Net**：基于一维时域卷积网络的语音增强实现。

两种方法均可在 Edinburgh VoiceBank+DEMAND 与 MS-SNSD 上执行预处理、训练、推理和指标评估。仓库保留关键源码、自动化脚本以及数值实验结果；数据集、checkpoint、生成特征、增强音频与本地安装包因体积或许可原因不上传。

## 已有实验结果

| 方法 | 数据集 | 文件数 | SNR (dB) | SSNR (dB) | SI-SDR (dB) | PESQ | STOI |
|---|---|---:|---:|---:|---:|---:|---:|
| Hybrid | Edinburgh | 824 | 12.641 | 6.028 | 12.310 | 2.106 | 0.889 |
| Hybrid | MS-SNSD | 1,100 | 12.596 | 8.775 | 12.327 | 2.177 | 0.920 |
| Wave-U-Net | Edinburgh | 824 | 19.592 | 10.611 | 19.539 | 2.522 | 0.935 |
| Wave-U-Net | MS-SNSD（同域、说话人与噪声互斥） | 682 | 25.704 | 17.680 | 25.647 | 3.018 | 0.872 |

完整汇总和逐文件指标位于 [`results/`](results/)；增强 WAV 未纳入版本控制，可运行测试阶段重新生成。实验设置、训练耗时和局限性分别见 [`HYBRID_EXPERIMENTS.md`](HYBRID_EXPERIMENTS.md) 和 [`WAVE_U_NET_EXPERIMENTS.md`](WAVE_U_NET_EXPERIMENTS.md)。

## 目录结构

```text
.
├── Hybrid/                 # RNNoise/Hybrid 源码与 PyTorch 训练代码
├── WAVE_U_NET/              # Wave-U-Net 源码、环境和辅助脚本
├── dataset/                 # 数据集下载与目录说明（不含数据）
├── results/                 # summary 与逐文件 CSV
├── run_denoising.py         # 四个实验共用的完整流水线
├── hybrid_edinburgh.py      # Hybrid + Edinburgh 入口
├── hybrid_ms_snsd.py        # Hybrid + MS-SNSD 入口
├── wave_u_net_edinburgh.py  # Wave-U-Net + Edinburgh 入口
└── wave_u_net_ms_snsd.py    # Wave-U-Net + MS-SNSD 入口
```

## 环境准备

RNNoise/Hybrid 的原实验环境为 Windows 11、Python 3.11、PyTorch 2.8.0 + CUDA 12.8，并需要 CMake、Ninja 和 MinGW-w64。Wave-U-Net 最终实验在 Linux、Python 3.11、PyTorch 2.5.1 + CUDA 12.4 和 5×RTX 4090 服务器上完成；Linux 环境见根目录 `environment.yml`。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

PyTorch/CUDA 的具体安装命令请按显卡驱动从 [PyTorch 官方安装页](https://pytorch.org/get-started/locally/) 选择。Hybrid 在 Windows 上也可使用：

```powershell
powershell -ExecutionPolicy Bypass -File .\Hybrid\setup_windows.ps1
```

随后按 [`dataset/README.md`](dataset/README.md) 下载并摆放数据。

## 运行实验

所有命令均在仓库根目录执行。省略 `--stage` 时依次执行预处理、训练和测试。

```powershell
python hybrid_edinburgh.py
python hybrid_ms_snsd.py
python wave_u_net_edinburgh.py --cuda
python wave_u_net_ms_snsd.py --cuda
```

也可以分阶段运行，便于检查产物或恢复训练：

```powershell
python hybrid_edinburgh.py --stage preprocess
python hybrid_edinburgh.py --stage train
python hybrid_edinburgh.py --stage test
```

快速验证整条流水线：

```powershell
python hybrid_edinburgh.py --sequences 2 --epochs 1 --batch-size 1 --max-test 2 --force
```

默认正式实验使用 10,000 个序列（每序列 2,000 帧）、60 epochs、batch size 8。生成文件写入 `.prepared/`，测试结果写入 `results/`。更完整的参数说明和命令见 [`RUN_EXPERIMENTS.md`](RUN_EXPERIMENTS.md)。

## 上游实现与许可

- RNNoise/Hybrid 基于 [xiph/rnnoise](https://github.com/xiph/rnnoise)，实验记录所用提交为 `70f1d256acd4b34a572f999a05c87bf00b67730d`，许可见 [`Hybrid/COPYING`](Hybrid/COPYING)。
- PyTorch Wave-U-Net 基于 [f90/Wave-U-Net-Pytorch](https://github.com/f90/Wave-U-Net-Pytorch) 的 `86c113da51540c94b01a91fbf17227f0e9c3c274`；TensorFlow 参考实现基于 [f90/Wave-U-Net](https://github.com/f90/Wave-U-Net) 的 `75eeae69953d6688ae7ec863c69146171accf08c`。代码目录保留各自的 README 与 LICENSE。
- 数据集不随仓库分发，其许可与引用要求以各数据集官网为准。

## 可复现性说明

随机种子默认为 `0`。MS-SNSD 的 Wave-U-Net v5 实验从 `clean_train/noise_train` 单一原始池按说话人和噪声文件互斥划分 train/validation/test，再用完全相同的 long-form 合成器生成 `{0, 10, 20, 30, 40}` dB 条件，从而排除原先官方 train/test 文件组织与电平分布不一致的影响。该结果属于同域实验，不应冒充官方 `clean_test/noise_test` OOD benchmark。指标包括输入基线、SNR、SSNR、SI-SDR、宽带 PESQ、STOI 及相对增益。checkpoint、预处理缓存和增强 WAV 未上传，需要本地重新训练生成；GPU、依赖版本和底层算子差异可能造成轻微数值差异。

## 项目文档

- [`experiments_section.tex`](experiments_section.tex)：基于《计算机学报》LaTeX 模板编写的完整实验章节；
- [`语音去噪算法复现软件用户手册_GBT8567-2006.docx`](语音去噪算法复现软件用户手册_GBT8567-2006.docx)：按 GB/T 8567—2006 软件用户手册结构编制的环境搭建、数据准备、训练与测试说明；
- [`RUN_EXPERIMENTS.md`](RUN_EXPERIMENTS.md)：命令行实验运行速查；
- [`HYBRID_EXPERIMENTS.md`](HYBRID_EXPERIMENTS.md)：RNNoise 实验设置及现有结果记录。
- [`WAVE_U_NET_EXPERIMENTS.md`](WAVE_U_NET_EXPERIMENTS.md)：Wave-U-Net 数据配方、训练记录及最终结果。
