# 数据集下载与目录结构

本目录不提交音频数据。请自行接受相应许可、下载并解压到以下固定路径；流水线按这些目录名读取数据。

## Edinburgh VoiceBank+DEMAND

- 官方发布页：[University of Edinburgh DataShare](https://datashare.ed.ac.uk/handle/10283/2791)
- 数据集论文：C. Valentini-Botinhao 等，*Noisy speech database for training speech enhancement algorithms and TTS models*。

至少需要形成以下结构：

```text
dataset/Edinburgh_VoiceBank_DEMAND/
├── clean_trainset_28spk_wav/
├── noisy_trainset_28spk_wav/
├── clean_testset_wav/
└── noisy_testset_wav/
```

同一 split 中 clean/noisy WAV 文件名必须一一对应。

## MS-SNSD

- 官方仓库及下载说明：[microsoft/MS-SNSD](https://github.com/microsoft/MS-SNSD)

下载干净语音与噪声后整理为：

```text
dataset/MS-SNSD/
├── clean_train/
├── noise_train/
├── clean_test/
└── noise_test/
```

项目会使用固定随机种子将 clean 与 noise 合成为训练/测试配对，生成内容位于 `.prepared/ms-snsd/`，无需手工创建。

## 检查

放置数据后可用少量样本执行快速检查：

```powershell
python hybrid_edinburgh.py --sequences 2 --epochs 1 --batch-size 1 --max-test 2 --force
python hybrid_ms_snsd.py --sequences 2 --epochs 1 --batch-size 1 --max-test 2 --force
```
