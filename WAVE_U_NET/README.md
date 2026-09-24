# Wave-U-Net 论文复现

本项目复现 Stoller、Ewert、Dixon 在 ISMIR 2018 发表的 **Wave-U-Net: A Multi-Scale Neural Network for End-to-End Audio Source Separation**。模型直接处理时域波形，通过多尺度下采样、上采样和跳跃连接分离音乐中的 `bass / drums / other / vocals`。

## 项目状态

- 论文：`WAVE-U-NET.pdf`
- 作者原始 TensorFlow 代码：`src/Wave-U-Net-TensorFlow-original`，固定提交 `75eeae69953d6688ae7ec863c69146171accf08c`
- 作者 PyTorch 改进版（本项目实际训练入口）：`src/Wave-U-Net-Pytorch`，固定提交 `86c113da51540c94b01a91fbf17227f0e9c3c274`
- 独立 Conda 环境：`waveunet-repro`，Python 3.11、PyTorch 2.8/CUDA 12.8
- 数据集：MUSDB18-HQ，150 首、四音轨、约 22.66 GB 压缩包

当前机器已完成源码和 CUDA 环境配置，CPU/GPU smoke test 均通过。MUSDB18-HQ 已建立可续传下载（当前约完成 1.3 GiB，`data/musdb18hq.zip.aria2` 保存精确进度）；由于当前到 Zenodo 的实际速度仅约 2–3 MiB/s，尚未完成 22.66 GB 下载及解压。作者权重的 Dropbox 域名在当前网络超时，也尚未取得。重新执行第 2 节和第 4 节的下载命令即可续接，不会从头开始。

原论文代码依赖 Python 3.6、TensorFlow 1.8 和 CUDA 9，无法合理使用当前 RTX 4050。为保证当前机器可训练，本项目使用同一作者的 PyTorch Wave-U-Net，并仅修复现代 NumPy、librosa、PyTorch checkpoint 与 CUDA 推理兼容问题。原 TensorFlow 源码完整保留，用于核对论文 M1–M7 配置；PyTorch 版属于作者后续改进实现，并非逐位等价的 TensorFlow 1.x 重跑。

## 目录

```text
WAVE_U_NET/
├─ environment.yml                 # 可复建的 Conda 环境
├─ scripts/                        # 环境、数据、训练、推理、评测入口
├─ data/MUSDB18-HQ/                # 数据集（下载后生成，不纳入版本控制）
├─ outputs/                        # 推理音频与评测结果
├─ src/Wave-U-Net-Pytorch/         # 当前可运行实现
├─ src/Wave-U-Net-TensorFlow-original/
└─ WAVE-U-NET.pdf
```

## 1. 环境

本机 Miniconda 安装在 `C:\Users\Administrator\miniconda3`，未加入系统 PATH，不会污染系统 Python。新机器可先安装 Miniconda，再在项目根目录执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_env.ps1
```

验证 CPU 和 GPU：

```powershell
C:\Users\Administrator\miniconda3\Scripts\conda.exe run -n waveunet-repro python .\src\Wave-U-Net-Pytorch\smoke_test.py
C:\Users\Administrator\miniconda3\Scripts\conda.exe run -n waveunet-repro python .\src\Wave-U-Net-Pytorch\smoke_test.py --cuda
```

无需全局执行 `conda init`；本文命令统一使用 `conda run`，不会改变当前 PowerShell 的 Python 环境。

## 2. 下载数据

在项目根目录执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\download_musdb18hq.ps1 -Extract
```

脚本从 Zenodo 官方记录 `3338373` 下载，支持断点续传，并在解压前校验：

```text
文件：musdb18hq.zip
大小：22656664047 bytes
MD5：12d4f2ecd55245a4688754dd76363103
```

最终结构必须是：

```text
data/MUSDB18-HQ/
├─ train/       # 100 tracks
└─ test/        # 50 tracks
```

若压缩包解压后额外套了一层目录，把含 `train` 和 `test` 的目录传给训练脚本的 `-Dataset` 即可。首次训练会在 `src/Wave-U-Net-Pytorch/hdf` 生成 HDF5 缓存，需要额外磁盘空间；不要中断首次预处理。

## 3. 启动训练

RTX 4050 只有 6 GB 显存，默认脚本将 batch size 设为 1：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\train.ps1
```

自定义数据位置或 CPU 训练：

```powershell
.\scripts\train.ps1 -Dataset D:\datasets\MUSDB18-HQ -BatchSize 1 -Workers 0
.\scripts\train.ps1 -Cpu -BatchSize 1
```

Windows 上建议 `Workers=0`，最稳定。训练会把 100 首训练歌曲固定按随机种子 1337 分为 75 首训练、25 首验证；50 首官方 test 仅用于最终评测。默认使用四个独立 Wave-U-Net、44.1 kHz 立体声、L1 损失、Adam、循环学习率和早停。输出位置：

- checkpoint：`src/Wave-U-Net-Pytorch/checkpoints/waveunet/checkpoint_*`
- TensorBoard：`src/Wave-U-Net-Pytorch/logs/waveunet`
- 最终指标：checkpoint 目录下的 `results.pkl`

查看训练曲线：

```powershell
C:\Users\Administrator\miniconda3\Scripts\conda.exe run -n waveunet-repro tensorboard --logdir .\src\Wave-U-Net-Pytorch\logs --port 6006
```

然后浏览器访问 `http://localhost:6006`。完整训练耗时较长，建议先确认 smoke test 与 HDF 预处理正常，再进行长时间训练。

续训时直接调用底层入口并提供 checkpoint：

```powershell
cd .\src\Wave-U-Net-Pytorch
C:\Users\Administrator\miniconda3\Scripts\conda.exe run -n waveunet-repro python train.py --cuda --dataset_dir ..\..\data\MUSDB18-HQ --batch_size 1 --num_workers 0 --load_model checkpoints\waveunet\checkpoint_XXXX
```

## 4. 测试与推理

作者预训练模型下载：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\download_pretrained.ps1
```

其官方权重托管于 Dropbox；若所在网络不能访问 Dropbox，请在浏览器打开作者 README 中的链接，手动将 `models.7z` 解压到 `src/Wave-U-Net-Pytorch/checkpoints`，确保文件 `checkpoints/waveunet/model` 存在。

对仓库示例音频推理：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\predict.ps1
```

对自己的音频推理：

```powershell
.\scripts\predict.ps1 -InputFile D:\audio\song.wav -Checkpoint .\src\Wave-U-Net-Pytorch\checkpoints\waveunet\model -Output .\outputs\song
```

输出为四个 WAV 文件。显存不足时加 `-Cpu`。

在 MUSDB18-HQ 官方 test split 上计算 SDR/ISR/SIR/SAR：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\evaluate.ps1 -Checkpoint .\src\Wave-U-Net-Pytorch\checkpoints\waveunet\model
```

评测结果写入 `outputs/evaluation/results.pkl` 和便于阅读的 `summary.json`。该评测会逐首完整推理，耗时明显长于单曲测试。

## 5. 复现实验说明

论文主要报告的是原 TensorFlow 实现的 M1–M7。精确核对这些配置可查看 `src/Wave-U-Net-TensorFlow-original/README.md` 与 `Config.py`；其中 M4 是带有效输入上下文和立体声 I/O 的最佳论文人声模型，M6 是多乐器分离模型。本项目的现代训练入口采用作者后续 PyTorch 改进版，默认一次训练四个独立源模型，因此结果应与该 PyTorch 实现比较，不能把数值差异简单视为论文复现失败。

建议记录：GPU/驱动、源码提交、`environment.yml`、随机种子、训练步数、最佳验证 checkpoint，以及 test split 的 median SDR。MUSDB18-HQ 音频有独立许可；请勿把 `data/` 中音频提交到代码仓库。

## 6. 常见问题

- `CUDA out of memory`：保持 `BatchSize=1`；关闭占用 GPU 的程序；仍不足时用 `-Cpu` 验证流程。
- 找不到 checkpoint：确认路径指向无扩展名的 `...\checkpoints\waveunet\model` 或实际 `checkpoint_XXXX` 文件。
- 下载中断：重新运行数据或权重下载脚本，会从已有文件继续。
- HDF 参数不一致：修改采样率、声道或乐器列表后，删除对应的 `hdf/train.hdf5`、`val.hdf5`、`test.hdf5` 再预处理。
- Conda 中文路径解码错误：在 PowerShell 先执行 `$env:PYTHONUTF8='1'`；项目的 `setup_env.ps1` 已自动设置。

## 引用

```bibtex
@inproceedings{stoller2018waveunet,
  title={Wave-U-Net: A Multi-Scale Neural Network for End-to-End Audio Source Separation},
  author={Stoller, Daniel and Ewert, Sebastian and Dixon, Simon},
  booktitle={Proceedings of ISMIR},
  year={2018}
}
```
