# 语音去噪实验运行说明

项目提供四个入口：`hybrid_edinburgh.py`、`hybrid_ms_snsd.py`、
`wave_u_net_edinburgh.py` 和 `wave_u_net_ms_snsd.py`。所有命令均在项目
根目录运行。不指定 `--stage` 时，默认完成 **预处理 → 特征生成 → 训练 →
模型导出 → 测试 → 指标汇总**。

## 1. Hybrid / 官方 RNNoise 环境

当前 `Hybrid` 使用Xiph RNNoise官方PyTorch/CUDA主分支。首次运行前执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\Hybrid\setup_windows.ps1
```

验证GPU：

```powershell
& 'C:\Users\Administrator\miniconda3\envs\rnnoise-cuda\python.exe' -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

正常情况下应显示 `True` 和NVIDIA GPU名称。只有诊断时才使用 `--cpu`。

## 2. Hybrid完整流程

Edinburgh完整运行：

```powershell
& 'C:\Users\Administrator\miniconda3\envs\rnnoise-cuda\python.exe' hybrid_edinburgh.py
```

MS-SNSD完整运行：

```powershell
& 'C:\Users\Administrator\miniconda3\envs\rnnoise-cuda\python.exe' hybrid_ms_snsd.py
```

默认配置针对本机6 GiB RTX 4050：10,000个训练序列、每序列2,000帧、
batch size 8、60 epochs，共约75,000次参数更新。每帧包含65维输入特征、
32维频带增益和1维VAD，共98个浮点数。

### 2.1 数据预处理

```powershell
& 'C:\Users\Administrator\miniconda3\envs\rnnoise-cuda\python.exe' hybrid_edinburgh.py --stage preprocess
& 'C:\Users\Administrator\miniconda3\envs\rnnoise-cuda\python.exe' hybrid_ms_snsd.py --stage preprocess
```

Edinburgh使用发布的 noisy-clean 配对，通过 `noise = noisy - clean` 提取噪声，
再划分为背景和前景噪声。MS-SNSD读取 `clean_train` 和 `noise_train`，并将
噪声划分成背景、前景两路。所有音频转换为48 kHz、单声道、16-bit PCM：

```text
.prepared/hybrid-数据集/speech.pcm
.prepared/hybrid-数据集/background_noise.pcm
.prepared/hybrid-数据集/foreground_noise.pcm
```

### 2.2 特征生成和CUDA训练

```powershell
& 'C:\Users\Administrator\miniconda3\envs\rnnoise-cuda\python.exe' hybrid_edinburgh.py --stage train
& 'C:\Users\Administrator\miniconda3\envs\rnnoise-cuda\python.exe' hybrid_ms_snsd.py --stage train
```

该阶段自动执行：

1. 编译官方 `dump_features.exe`；
2. 生成98维 `features.f32`；
3. 使用PyTorch在CUDA上训练；
4. 每个epoch保存 `.pth` checkpoint；
5. 将最后一个checkpoint量化导出为 `rnnoise_data.c/h`；
6. 编译包含新权重的 `rnnoise_demo.exe`。

主要产物：

```text
.prepared/hybrid-数据集/features.f32
.prepared/hybrid-数据集/training/checkpoints/rnnoise_N.pth
.prepared/hybrid-数据集/c_model/rnnoise_data.c
.prepared/hybrid-数据集/c_model/rnnoise_data.h
Hybrid/build/rnnoise_demo.exe
```

训练中断后，重新运行相同命令会自动从编号最大的checkpoint继续。忽略已有
checkpoint并重新训练可加 `--no-resume`：

```powershell
& 'C:\Users\Administrator\miniconda3\envs\rnnoise-cuda\python.exe' hybrid_edinburgh.py --stage train --no-resume
```

注意：`--no-resume` 不删除旧checkpoint。需要独立新实验时，应先备份或更换
`.prepared/hybrid-数据集/training` 目录。

### 2.3 测试和指标

```powershell
& 'C:\Users\Administrator\miniconda3\envs\rnnoise-cuda\python.exe' hybrid_edinburgh.py --stage test
& 'C:\Users\Administrator\miniconda3\envs\rnnoise-cuda\python.exe' hybrid_ms_snsd.py --stage test
```

测试使用刚训练并重新编译的官方C推理程序。MS-SNSD测试集以固定随机种子，
按照0、10、20、30、40 dB SNR合成 noisy-clean 测试对。

输出目录为：

```text
results/hybrid_edinburgh/
results/hybrid_ms-snsd/
```

每个目录包含：

- `enhanced/*.wav`：增强语音；
- `per_file.csv`：逐文件结果；
- `summary.json`：机器可读汇总；
- `summary.txt`：文本汇总。

评价指标包括 SNR、SSNR、SI-SDR、PESQ 和 STOI。

## 3. 快速检查与常用参数

用少量数据检查完整管线：

```powershell
& 'C:\Users\Administrator\miniconda3\envs\rnnoise-cuda\python.exe' hybrid_edinburgh.py `
    --sequences 2 --epochs 1 --batch-size 1 --max-test 2 --force
```

常用参数：

```text
--stage preprocess|train|test|all
--sequences N       训练序列数，每个序列固定2,000帧
--epochs N          总训练轮数
--batch-size N      6 GiB显存建议从8开始
--workers N         Windows默认0最稳定
--max-test N        仅测试前N条音频
--force             重新生成预处理和特征
--no-resume         不加载已有checkpoint
--sparse            启用官方GRU结构化稀疏化
--cpu               禁用CUDA，仅用于诊断
```

## 4. Wave-U-Net完整流程

```powershell
conda run -n waveunet-repro python wave_u_net_edinburgh.py --cuda
conda run -n waveunet-repro python wave_u_net_ms_snsd.py --cuda
```

也可以分阶段执行：

```powershell
conda run -n waveunet-repro python wave_u_net_edinburgh.py --stage preprocess
conda run -n waveunet-repro python wave_u_net_edinburgh.py --stage train --cuda
conda run -n waveunet-repro python wave_u_net_edinburgh.py --stage test --cuda
```

Wave-U-Net使用16 kHz单声道 noisy-clean 配对、12层网络、24个初始通道、
15点卷积核、stride 2、L1损失和Adam。结果写入：

```text
results/wave-u-net_edinburgh/
results/wave-u-net_ms-snsd/
```

## 5. 推荐的正式实验顺序

正式实验建议分阶段运行，便于检查产物与恢复训练：

```powershell
# 1. 数据预处理
& 'C:\Users\Administrator\miniconda3\envs\rnnoise-cuda\python.exe' hybrid_edinburgh.py --stage preprocess

# 2. 特征生成、CUDA训练、权重导出和C程序编译
& 'C:\Users\Administrator\miniconda3\envs\rnnoise-cuda\python.exe' hybrid_edinburgh.py --stage train

# 3. 测试和指标汇总
& 'C:\Users\Administrator\miniconda3\envs\rnnoise-cuda\python.exe' hybrid_edinburgh.py --stage test
```

确认一套数据集完整运行后，再换成另一数据集入口。
