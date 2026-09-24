# Official RNNoise CUDA pipeline on Windows

This directory tracks the official Xiph RNNoise main branch at commit
`70f1d256acd4b34a572f999a05c87bf00b67730d`.  The upstream model uses
98-float records (65 input features, 32 gains, one VAD target) and PyTorch.

Create the native Windows CUDA environment and feature extractor:

```powershell
powershell -ExecutionPolicy Bypass -File .\Hybrid\setup_windows.ps1
```

Then use either root entry point.  Both default to the complete pipeline and
automatically select CUDA; pass `--cpu` only for diagnostics.

```powershell
& 'C:\Users\Administrator\miniconda3\envs\rnnoise-cuda\python.exe' hybrid_edinburgh.py
& 'C:\Users\Administrator\miniconda3\envs\rnnoise-cuda\python.exe' hybrid_ms_snsd.py
```

Training writes one `.pth` checkpoint per epoch. Re-running training resumes
from the latest checkpoint unless `--no-resume` is supplied.

The project defaults to 10,000 sequences, batch 8 and 60 epochs: about 75,000
optimizer updates as recommended upstream, while fitting the local 6 GiB RTX 4050.
