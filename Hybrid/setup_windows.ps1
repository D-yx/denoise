param([string]$Conda = "C:\Users\Administrator\miniconda3\Scripts\conda.exe")
$ErrorActionPreference = "Stop"
$python = "C:\Users\Administrator\miniconda3\envs\rnnoise-cuda\python.exe"
if (-not (Test-Path $python)) {
    & $Conda create -y -n rnnoise-cuda --override-channels -c conda-forge `
        python=3.11 "cmake>=3.25" ninja m2w64-gcc numpy=1.26 scipy libsndfile pip
}
$localTorch = Join-Path (Split-Path $PSScriptRoot) "WAVE_U_NET\data\torch-2.8.0+cu128-cp311-cp311-win_amd64.whl"
if (Test-Path $localTorch) {
    & $python -m pip install $localTorch soundfile==0.12.1 tqdm==4.66.5 pesq==0.0.4 pystoi==0.4.1
} else {
    & $python -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 `
        torch==2.8.0 soundfile==0.12.1 tqdm==4.66.5 pesq==0.0.4 pystoi==0.4.1
}
& $Conda run -n rnnoise-cuda python -c "import torch; print('PyTorch', torch.__version__, 'CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'unavailable')"
& $Conda run -n rnnoise-cuda cmake -S $PSScriptRoot -B "$PSScriptRoot\build" -G Ninja -DCMAKE_C_COMPILER=x86_64-w64-mingw32-gcc
& $Conda run -n rnnoise-cuda cmake --build "$PSScriptRoot\build" --target dump_features
