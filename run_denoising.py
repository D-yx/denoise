"""One-command RNNoise (Hybrid) and Wave-U-Net speech-denoising experiments.

The four small entry scripts in the project root call this module.  Run with
``--stage preprocess|train|test|all`` (``all`` is the default).
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import correlate, correlation_lags, resample_poly

ROOT = Path(__file__).resolve().parent
DATASETS = ROOT / "dataset"
RESULTS = ROOT / "results"

# Calling a Conda environment's python.exe directly on Windows does not run
# activation hooks.  Make its native tools and DLLs available to subprocesses
# so the documented one-command entry points can find CMake, Ninja and MinGW.
if os.name == "nt":
    env_root = Path(sys.prefix)
    env_bins = [env_root, env_root/"Library"/"bin", env_root/"Scripts", env_root/"bin"]
    existing = [str(path) for path in env_bins if path.exists()]
    os.environ["PATH"] = os.pathsep.join(existing + [os.environ.get("PATH", "")])


def log(message: str) -> None:
    print(time.strftime("[%H:%M:%S]"), message, flush=True)


def audio(path: Path, sr: int) -> np.ndarray:
    x, old_sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(1)
    if old_sr != sr:
        g = math.gcd(old_sr, sr)
        x = resample_poly(x, sr // g, old_sr // g).astype(np.float32)
    return np.nan_to_num(x).clip(-1, 1)


def cached_audio(path: Path, sr: int) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path, allow_pickle=False).astype(np.float32, copy=False)
    return audio(path, sr)


def write(path: Path, x: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.asarray(x).clip(-1, 1), sr, subtype="PCM_16")


def mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr: float, rng: np.random.Generator) -> np.ndarray:
    if len(noise) < len(clean):
        noise = np.tile(noise, math.ceil(len(clean) / max(len(noise), 1)))
    start = int(rng.integers(0, max(1, len(noise) - len(clean) + 1)))
    noise = noise[start:start + len(clean)]
    scale = np.sqrt(np.mean(clean ** 2) + 1e-12) / (np.sqrt(np.mean(noise ** 2) + 1e-12) * 10 ** (snr / 20))
    mixed = clean + scale * noise
    peak = np.max(np.abs(mixed))
    return (mixed / max(1.0, peak)).astype(np.float32)


def paired_files(dataset: str, split: str, prepared_sr: int | None = None) -> list[tuple[Path, Path]]:
    base = DATASETS / ("Edinburgh_VoiceBank_DEMAND" if dataset == "edinburgh" else "MS-SNSD")
    if dataset == "edinburgh":
        if prepared_sr is not None:
            made = ROOT / ".prepared" / f"wave-u-net-{dataset}-{prepared_sr}hz" / split
            return [(made / "noisy" / p.name, p) for p in sorted((made / "clean").glob("*.npy"))
                    if (made / "noisy" / p.name).exists()]
        clean_dir = base / ("clean_trainset_28spk_wav" if split == "train" else "clean_testset_wav")
        noisy_dir = base / ("noisy_trainset_28spk_wav" if split == "train" else "noisy_testset_wav")
        return [(noisy_dir / p.name, p) for p in sorted(clean_dir.glob("*.wav")) if (noisy_dir / p.name).exists()]
    # MS-SNSD pairs are made by preprocess_ms().
    made = ROOT / ".prepared" / "ms-snsd-long" / split
    return [(made / "noisy" / p.name, p) for p in sorted((made / "clean").glob("*.wav"))
            if (made / "noisy" / p.name).exists()]


def _prepare_ms_pair(task: tuple[list[Path], Path, Path, Path, int, float, bool, bool]) -> None:
    clean_paths, noise_path, clean_dst, noisy_dst, seed, snr, augment, force = task
    if clean_dst.exists() and noisy_dst.exists() and not force: return
    rng = np.random.default_rng(seed); silence = np.zeros(3200, dtype=np.float32)
    pieces = []
    for i, path in enumerate(clean_paths):
        if i: pieces.append(silence)
        pieces.append(audio(path, 16000))
    clean = np.concatenate(pieces).astype(np.float32)
    noise = audio(noise_path, 16000)
    if len(noise) < len(clean): noise = np.tile(noise, math.ceil(len(clean) / max(1, len(noise))))
    offset = int(rng.integers(0, max(1, len(noise) - len(clean) + 1)))
    noise = noise[offset:offset + len(clean)]
    clean_rms = float(np.sqrt(np.mean(clean * clean) + 1e-12))
    noise_rms = float(np.sqrt(np.mean(noise * noise) + 1e-12))
    # Follow the MS-SNSD recipe: normalize both sources to -25 dBFS, then set SNR.
    target_rms = 10 ** (-25 / 20)
    clean = clean * (target_rms / clean_rms)
    noise = noise * (target_rms / noise_rms) * (10 ** (-snr / 20))
    if augment:
        # Match the upstream Wave-U-Net random_amplify recipe without touching
        # the training loop: independently vary target speech and residual
        # noise gains.  This broadens both absolute level and effective SNR.
        clean *= float(rng.uniform(.7, 1.3))
        noise *= float(rng.uniform(.7, 1.3))
    noisy = clean + noise
    peak = float(np.max(np.abs(noisy)))
    if peak > .99:
        scale = .99 / peak; clean *= scale; noisy *= scale
    write(clean_dst, clean, 16000); write(noisy_dst, noisy, 16000)


def preprocess_ms(seed: int, force: bool, workers: int = 0) -> None:
    base, out = DATASETS / "MS-SNSD", ROOT / ".prepared" / "ms-snsd-long"
    workers = workers or min(8, os.cpu_count() or 1); minimum_frames = 160000
    all_cleans = sorted((base / "clean_train").glob("*.wav"))
    all_noises = sorted((base / "noise_train").glob("*.wav"))
    if not all_cleans or len(all_noises) < 3:
        raise FileNotFoundError("MS-SNSD clean_train/noise_train is incomplete")

    # Build train/validation/test from one source domain.  The official
    # clean_test/noise_test material has measurably different level, silence,
    # and file-construction statistics, so it is an OOD benchmark rather than
    # a domain-matched test set.  Keep speakers and noise files disjoint while
    # applying the exact same synthesizer to all three logical partitions.
    by_speaker: dict[str, list[Path]] = {}
    for path in all_cleans:
        by_speaker.setdefault(path.stem.split("_", 1)[0], []).append(path)
    speakers = sorted(by_speaker)
    split_rng = random.Random(seed); split_rng.shuffle(speakers)
    test_count = max(1, round(len(speakers) * .1))
    test_speakers = set(speakers[:test_count])
    train_pool_speakers = sorted(set(speakers) - test_speakers)
    # Reproduce train_wave()'s validation-speaker selection exactly.
    val_order = train_pool_speakers.copy(); random.Random(seed).shuffle(val_order)
    val_speakers = set(val_order[:max(1, round(len(val_order) * .1))])

    noise_order = all_noises.copy(); split_rng.shuffle(noise_order)
    noise_test_count = max(1, round(len(noise_order) * .1))
    noise_val_count = max(1, round(len(noise_order) * .1))
    test_noises = noise_order[:noise_test_count]
    val_noises = noise_order[noise_test_count:noise_test_count + noise_val_count]
    train_noises = noise_order[noise_test_count + noise_val_count:]

    def make_groups(selected: list[str]) -> list[tuple[str, list[Path]]]:
        groups: list[tuple[str, list[Path]]] = []
        for speaker in sorted(selected):
            paths = sorted(by_speaker[speaker]); current: list[Path] = []
            frames = sequence = 0
            for path in paths:
                current.append(path); frames += sf.info(path).frames + (3200 if len(current) > 1 else 0)
                if frames >= minimum_frames:
                    groups.append((f"{speaker}_seq{sequence:04d}", current)); current, frames, sequence = [], 0, sequence + 1
            if current:
                fill = iter(paths)
                while frames < minimum_frames:
                    path = next(fill, paths[0]); current.append(path); frames += sf.info(path).frames + 3200
                groups.append((f"{speaker}_seq{sequence:04d}", current))
        return groups

    for split in ("train", "test"):
        selected_speakers = train_pool_speakers if split == "train" else sorted(test_speakers)
        groups = make_groups(selected_speakers)
        # The upstream MS-SNSD synthesizer emits every configured SNR for each
        # constructed clean/noise utterance.  Keeping only one randomly chosen
        # SNR per utterance made the 24 M parameter Wave-U-Net memorize a small
        # set of fixed mixtures and generalize very poorly to the earlier OOD
        # test. Use the full SNR sweep for training; retain one balanced
        # condition per held-out long-form test utterance for per-SNR reporting.
        snr_levels = np.array([0., 10., 20., 30., 40.])
        test_snrs = np.resize(snr_levels, len(groups))
        np.random.default_rng(seed + (1 if split == "test" else 0)).shuffle(test_snrs)
        tasks = []
        for i, (name, paths) in enumerate(groups):
            levels = snr_levels if split == "train" else (test_snrs[i],)
            for j, snr in enumerate(levels):
                speaker = name.split("_seq", 1)[0]
                noise_pool = (test_noises if split == "test" else
                              val_noises if speaker in val_speakers else train_noises)
                task_rng = np.random.default_rng(seed * 10_000_019 + i * len(snr_levels) + j)
                noise = noise_pool[int(task_rng.integers(len(noise_pool)))]; dst = out / split
                suffix = f"_snr{int(snr):02d}" if split == "train" else ""
                pair_name = f"{name}{suffix}.wav"
                tasks.append((paths, noise, dst / "clean" / pair_name, dst / "noisy" / pair_name,
                              seed * 1_000_003 + i * len(snr_levels) + j, float(snr), True, force))
        # Remove files belonging to an older recipe.  Without this narrow,
        # recipe-owned cleanup, paired_files() would silently include the old
        # one-SNR training pairs alongside the new sweep.
        expected_paths = {p for task in tasks for p in (task[2], task[3])}
        for kind in ("clean", "noisy"):
            for stale in (out / split / kind).glob("*.wav"):
                if stale not in expected_paths:
                    stale.unlink()
        log(f"MS-SNSD {split}: synthesising {len(tasks)} >=10-second pairs with {workers} workers")
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            for i, _ in enumerate(pool.map(_prepare_ms_pair, tasks, chunksize=4), 1):
                if i % 250 == 0 or i == len(tasks): log(f"  {i}/{len(tasks)}")
        complete_pairs = paired_files("ms-snsd", split)
        if len(complete_pairs) != len(tasks): raise RuntimeError(f"MS-SNSD {split} preparation is incomplete ({len(complete_pairs)}/{len(tasks)})")
        (out / split / ".complete").write_text(
            f"recipe=in-domain-speaker-noise-disjoint-v5\npairs={len(tasks)}\nsample_rate=16000\nminimum_seconds=10\nsilence_seconds=0.2\nseed={seed}\n"
            f"train_speakers={len(set(train_pool_speakers)-val_speakers)}\nval_speakers={len(val_speakers)}\ntest_speakers={len(test_speakers)}\n"
            f"train_noises={len(train_noises)}\nval_noises={len(val_noises)}\ntest_noises={len(test_noises)}\n",
            encoding="utf-8")


def _prepare_wave_pair(task: tuple[Path, Path, Path, Path, int, bool]) -> None:
    noisy_src, clean_src, noisy_dst, clean_dst, sr, force = task
    for src, dst in ((noisy_src, noisy_dst), (clean_src, clean_dst)):
        if dst.exists() and not force:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(dst.suffix + ".tmp")
        with tmp.open("wb") as handle:
            np.save(handle, audio(src, sr), allow_pickle=False)
        os.replace(tmp, dst)


def preprocess_wave(dataset: str, args: argparse.Namespace) -> None:
    if dataset != "edinburgh":
        preprocess_ms(args.seed, args.force, args.preprocess_workers)
        return
    workers = args.preprocess_workers or min(8, os.cpu_count() or 1)
    for split in ("train", "test"):
        source_pairs = paired_files(dataset, split)
        out = ROOT / ".prepared" / f"wave-u-net-{dataset}-{args.wave_sr}hz" / split
        tasks = [(noisy, clean, out / "noisy" / f"{noisy.stem}.npy",
                  out / "clean" / f"{clean.stem}.npy", args.wave_sr, args.force)
                 for noisy, clean in source_pairs]
        if not tasks: raise RuntimeError(f"No {dataset} {split} pairs found")
        log(f"Wave-U-Net {split}: caching {len(tasks)} pairs at {args.wave_sr} Hz with {workers} workers")
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            for i, _ in enumerate(pool.map(_prepare_wave_pair, tasks, chunksize=8), 1):
                if i % 500 == 0 or i == len(tasks): log(f"  {i}/{len(tasks)}")
        (out / ".complete").write_text(f"pairs={len(tasks)}\nsample_rate={args.wave_sr}\n", encoding="utf-8")


def prepare_hybrid(dataset: str, args: argparse.Namespace) -> None:
    out = ROOT / ".prepared" / f"hybrid-{dataset}"
    speech_pcm = out / "speech.pcm"
    background_pcm = out / "background_noise.pcm"
    foreground_pcm = out / "foreground_noise.pcm"
    if not args.force and all(p.exists() for p in (speech_pcm, background_pcm, foreground_pcm)):
        log("Official RNNoise PCM streams already exist; use --force to regenerate")
        return
    out.mkdir(parents=True, exist_ok=True)
    if dataset == "edinburgh":
        pairs = paired_files(dataset, "train")
        if not pairs: raise RuntimeError("No Edinburgh training pairs found")
        log(f"Official RNNoise: extracting speech/background/foreground from {len(pairs)} pairs")
        with speech_pcm.open("wb") as fs, background_pcm.open("wb") as fb, foreground_pcm.open("wb") as ff:
            for i, (npth, cp) in enumerate(pairs, 1):
                c, n = audio(cp, 48000), audio(npth, 48000)
                length=min(len(c),len(n)); c,n=c[:length],(n[:length]-c[:length]).clip(-1,1)
                (c*32767).astype("<i2").tofile(fs)
                ((n if i%2 else np.zeros_like(n))*32767).astype("<i2").tofile(fb)
                ((n if not i%2 else np.zeros_like(n))*32767).astype("<i2").tofile(ff)
                if i%250==0 or i==len(pairs): log(f"  {i}/{len(pairs)}")
    else:
        base=DATASETS/"MS-SNSD"; cleans=sorted((base/"clean_train").glob("*.wav")); noises=sorted((base/"noise_train").glob("*.wav"))
        if not cleans or len(noises)<2: raise RuntimeError("MS-SNSD clean_train/noise_train is incomplete")
        log(f"Official RNNoise: converting {len(cleans)} speech and {len(noises)} noise files")
        with speech_pcm.open("wb") as fs:
            for i,p in enumerate(cleans,1):
                (audio(p,48000)*32767).astype("<i2").tofile(fs)
                if i%500==0 or i==len(cleans): log(f"  speech {i}/{len(cleans)}")
        with background_pcm.open("wb") as fb, foreground_pcm.open("wb") as ff:
            for i,p in enumerate(noises):
                target=fb if i%2==0 else ff; (audio(p,48000)*32767).astype("<i2").tofile(target)
        log("  noise streams ready")


def run(cmd: list[str], cwd: Path, env: dict | None = None) -> None:
    log("$ " + subprocess.list2cmdline([str(x) for x in cmd]))
    subprocess.run([str(x) for x in cmd], cwd=cwd, env=env, check=True)


def best_rnnoise_checkpoint(checkpoints: list[Path]) -> tuple[Path, float]:
    """Return the checkpoint with the lowest recorded epoch training loss."""
    if not checkpoints:
        raise RuntimeError("No RNNoise checkpoints found")
    # Keep torch's Intel OpenMP runtime out of the pipeline process.  Importing
    # torch here and scipy/pystoi later can load two OpenMP runtimes on Windows.
    code = (
        "from pathlib import Path; import math,sys,torch; "
        "ps=sorted(Path(sys.argv[1]).glob('rnnoise_*.pth'),"
        "key=lambda p:int(p.stem.rsplit('_',1)[-1])); "
        "xs=[(float(s.get('loss',float('inf'))),p) for p in ps "
        "for s in [torch.load(p,map_location='cpu',weights_only=False)]]; "
        "xs=[x for x in xs if math.isfinite(x[0])]; "
        "assert xs,'RNNoise checkpoints contain no finite loss values'; "
        "loss,path=min(xs,key=lambda x:(x[0],x[1].name)); "
        "print(str(path.resolve())+'\\t'+repr(loss))"
    )
    result = subprocess.run([sys.executable, "-c", code, str(checkpoints[0].parent)],
                            text=True, capture_output=True, check=True)
    path_text, loss_text = result.stdout.strip().rsplit("\t", 1)
    return Path(path_text), float(loss_text)


def train_hybrid(dataset: str, args: argparse.Namespace) -> None:
    base, prep = ROOT / "Hybrid", ROOT / ".prepared" / f"hybrid-{dataset}"
    required=[prep/"speech.pcm",prep/"background_noise.pcm",prep/"foreground_noise.pcm"]
    if not all(p.exists() for p in required): raise FileNotFoundError("Run --stage preprocess before training")
    build=base/"build"; extractor=build/"dump_features.exe"
    if not extractor.exists():
        run(["cmake","-S",base,"-B",build,"-G","Ninja","-DCMAKE_C_COMPILER=x86_64-w64-mingw32-gcc"],ROOT)
        run(["cmake","--build",build,"--target","dump_features"],ROOT)
    features=prep/"features.f32"
    sequences=args.sequences if args.frames is None else max(1,math.ceil(args.frames/2000))
    expected=sequences*2000*98*4
    if args.force or not features.exists() or features.stat().st_size!=expected:
        log(f"Generating official 98-D features: {sequences:,} sequences ({sequences*2000:,} frames)")
        run([extractor,*required,features,str(sequences)],base)
    else: log(f"Reusing complete feature file: {features}")
    output=prep/"training"; checkpoints=output/"checkpoints"; checkpoints.mkdir(parents=True,exist_ok=True)
    ckpts=sorted(checkpoints.glob("rnnoise_*.pth"),key=lambda p:int(p.stem.rsplit('_',1)[-1]))
    cmd=[sys.executable,"train_rnnoise.py",features,output,"--epochs",str(args.epochs),
         "--batch-size",str(args.batch_size),"--sequence-length","2000","--num-workers",str(args.workers)]
    if args.sparse: cmd.append("--sparse")
    if ckpts and not args.no_resume: cmd += ["--initial-checkpoint",ckpts[-1]]; log(f"Resuming {ckpts[-1].name}")
    env=os.environ.copy()
    if args.cpu: env["CUDA_VISIBLE_DEVICES"]="-1"
    log("CUDA training requested" if not args.cpu else "CPU training requested")
    log("$ "+subprocess.list2cmdline([str(x) for x in cmd])); subprocess.run([str(x) for x in cmd],cwd=base/"torch"/"rnnoise",env=env,check=True)
    ckpts=sorted(checkpoints.glob("rnnoise_*.pth"),key=lambda p:int(p.stem.rsplit('_',1)[-1]))
    if not ckpts: raise RuntimeError("Official trainer produced no checkpoint")
    selected, selected_loss = best_rnnoise_checkpoint(ckpts)
    log(f"Exporting best checkpoint {selected.name} (loss={selected_loss:.8f})")
    export=prep/"c_model"
    run([sys.executable,"dump_rnnoise_weights.py","--quantize",selected,export],base/"torch"/"rnnoise")
    shutil.copy2(export/"rnnoise_data.c",base/"src"/"rnnoise_data.c")
    shutil.copy2(export/"rnnoise_data.h",base/"src"/"rnnoise_data.h")
    run(["cmake","-S",base,"-B",build,"-G","Ninja","-DCMAKE_C_COMPILER=x86_64-w64-mingw32-gcc"],ROOT)
    run(["cmake","--build",build,"--target","rnnoise_demo"],ROOT)
    # Keep a dataset-specific executable: the shared Hybrid build is replaced
    # whenever another dataset exports and recompiles its weights.
    shutil.copy2(build/"rnnoise_demo.exe", prep/"rnnoise_demo.exe")
    log(f"Dataset-specific inference executable: {prep/'rnnoise_demo.exe'}")


def wave_imports():
    src = ROOT / "WAVE_U_NET" / "src" / "Wave-U-Net-Pytorch"
    sys.path.insert(0, str(src))
    import torch
    from model.waveunet import Waveunet
    return torch, Waveunet


def make_wave_model(torch, Waveunet, sr: int):
    # Paper M4/M5 family: 12 levels, 24 initial filters, 15-sample kernels, stride 2.
    channels = [24 * (i + 1) for i in range(12)]
    return Waveunet(1, channels, 1, ["clean"], kernel_size=15,
                    target_output_size=sr, conv_type="normal", res="fixed",
                    separate=False, depth=1, strides=2)


class PairDataset:
    def __init__(self, pairs, model, sr, random_crop):
        self.pairs, self.model, self.sr, self.random_crop = pairs, model, sr, random_crop
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        noisy_p, clean_p = self.pairs[idx]; x, y = cached_audio(noisy_p, self.sr), cached_audio(clean_p, self.sr)
        length = min(len(x), len(y)); x, y = x[:length], y[:length]
        out_n, in_n = self.model.output_size, self.model.input_size
        start = random.randint(0, max(0, length - out_n)) if self.random_crop else (idx * out_n) % max(1, length)
        left = self.model.shapes["output_start_frame"]; in_start = start - left
        def segment(a, pos, n):
            lo, hi = max(0, pos), min(len(a), pos+n); z = a[lo:hi]
            return np.pad(z, (max(0, -pos), max(0, pos+n-len(a))))[:n].astype(np.float32)
        return segment(x, in_start, in_n)[None], segment(y, start, out_n)[None]


def train_wave(dataset: str, args: argparse.Namespace) -> None:
    data_recipe = None
    prepared_sr = args.wave_sr if dataset == "edinburgh" else None
    pairs = paired_files(dataset, "train", prepared_sr)
    if not pairs:
        raise FileNotFoundError("Prepared Wave-U-Net data not found; run the preprocessing script first")
    if dataset == "edinburgh":
        cache_root = ROOT / ".prepared" / f"wave-u-net-{dataset}-{args.wave_sr}hz" / "train"
        expected = len(paired_files(dataset, "train"))
    else:
        cache_root = ROOT / ".prepared" / "ms-snsd-long" / "train"
        marker = cache_root / ".complete"
        if not marker.exists(): raise RuntimeError("Prepared MS-SNSD long-form training data is missing; rerun preprocessing")
        metadata = dict(line.split("=", 1) for line in marker.read_text().splitlines() if "=" in line)
        if metadata.get("recipe") != "in-domain-speaker-noise-disjoint-v5":
            raise RuntimeError("Prepared MS-SNSD cache uses an obsolete recipe; rerun preprocessing with --force")
        data_recipe = metadata["recipe"]
        expected = int(metadata["pairs"])
    if not (cache_root / ".complete").exists() or len(pairs) != expected:
        raise RuntimeError(f"Prepared training data is incomplete ({len(pairs)}/{expected}); rerun preprocessing")

    torch, Waveunet = wave_imports()
    # Fixed input geometry benefits from cuDNN algorithm autotuning.  TF32 is
    # substantially faster on Ampere/Ada GPUs while keeping model parameters,
    # optimizer state, and the L1 training objective in float32.
    if torch.cuda.is_available() and not args.cpu:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from torch.utils.data import DataLoader, DistributedSampler
    world_size = int(os.environ.get("WORLD_SIZE", "1")); rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0")); distributed = world_size > 1
    if distributed:
        if args.cpu: backend, device = "gloo", torch.device("cpu")
        else:
            if not torch.cuda.is_available(): raise RuntimeError("DDP requested CUDA training, but CUDA is unavailable")
            torch.cuda.set_device(local_rank); backend, device = "nccl", torch.device("cuda", local_rank)
        dist.init_process_group(backend=backend, init_method="env://")
    else:
        device = torch.device("cuda" if not args.cpu and torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed + rank); random.seed(args.seed + rank)
    if device.type == "cuda": torch.cuda.manual_seed_all(args.seed + rank)
    if rank == 0:
        gpu = f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""
        log(f"Wave-U-Net training: device={device}{gpu}, world_size={world_size}")
    model = make_wave_model(torch, Waveunet, args.wave_sr).to(device)
    rng = random.Random(args.seed)
    if dataset == "ms-snsd":
        speakers = sorted({clean.stem.split("_seq", 1)[0] for _, clean in pairs}); rng.shuffle(speakers)
        val_speakers = set(speakers[:max(1, round(len(speakers) * .1))])
        train = [pair for pair in pairs if pair[1].stem.split("_seq", 1)[0] not in val_speakers]
        val = [pair for pair in pairs if pair[1].stem.split("_seq", 1)[0] in val_speakers]
        if rank == 0: log(f"Speaker-disjoint split: train={len(train)}, val={len(val)}, val_speakers={len(val_speakers)}")
    else:
        rng.shuffle(pairs); cut = max(1, int(len(pairs) * .9)); train, val = pairs[:cut], pairs[cut:] or pairs[-1:]
    train_data, val_data = PairDataset(train, model, args.wave_sr, True), PairDataset(val, model, args.wave_sr, False)
    train_sampler = DistributedSampler(train_data, shuffle=True, seed=args.seed) if distributed else None
    val_sampler = DistributedSampler(val_data, shuffle=False) if distributed else None
    loader_args = {"batch_size": args.wave_batch_size, "num_workers": args.workers,
                   "pin_memory": device.type == "cuda"}
    if args.workers > 0:
        loader_args.update({"persistent_workers": True, "prefetch_factor": 2})
    tr = DataLoader(train_data, shuffle=train_sampler is None, sampler=train_sampler, **loader_args)
    va = DataLoader(val_data, sampler=val_sampler, **loader_args)
    run_name = "wave-u-net-ms-snsd-long" if dataset == "ms-snsd" else f"wave-u-net-{dataset}"
    out = ROOT / ".prepared" / run_name; out.mkdir(parents=True, exist_ok=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4); loss_fn = torch.nn.L1Loss()
    best, bad, start_epoch = float("inf"), 0, 1
    last_path = out / "last.pt"
    if last_path.exists() and not args.no_resume:
        state = torch.load(last_path, map_location=device, weights_only=False)
        if dataset == "ms-snsd" and state.get("data_recipe") != data_recipe:
            raise RuntimeError("Checkpoint was trained with a different MS-SNSD recipe; use --no-resume")
        model.load_state_dict(state["model"])
        if "optimizer" in state: opt.load_state_dict(state["optimizer"])
        start_epoch = int(state.get("epoch", 0)) + 1
        best = float(state.get("best_val_loss", state.get("val_loss", float("inf"))))
        bad = int(state.get("bad_epochs", 0))
        if rank == 0: log(f"Resuming Wave-U-Net from epoch {start_epoch-1}; best_val_loss={best:.6f}")
    if distributed: model = DistributedDataParallel(model, device_ids=[local_rank] if device.type == "cuda" else None)
    if start_epoch > args.wave_epochs:
        if rank == 0: log(f"Wave-U-Net training already reached {args.wave_epochs} epochs")
        if distributed: dist.barrier(); dist.destroy_process_group()
        return
    if bad >= args.patience:
        if rank == 0: log(f"Wave-U-Net already stopped after {bad} unimproved epochs")
        if distributed: dist.barrier(); dist.destroy_process_group()
        return
    for epoch in range(start_epoch, args.wave_epochs + 1):
        if train_sampler is not None: train_sampler.set_epoch(epoch)
        model.train(); total = 0.0
        for step, (x, y) in enumerate(tr, 1):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True); opt.zero_grad(set_to_none=True); pred = model(x)["clean"]
            loss = loss_fn(pred, y); loss.backward(); opt.step(); total += loss.item()
            if rank == 0 and (step % args.log_interval == 0 or step == len(tr)): log(f"epoch {epoch} {step}/{len(tr)} loss={total/step:.6f}")
        model.eval(); v = 0.0
        with torch.no_grad():
            for x, y in va: v += loss_fn(model(x.to(device, non_blocking=True))["clean"], y.to(device, non_blocking=True)).item()
        totals = torch.tensor([total, len(tr), v, len(va)], dtype=torch.float64, device=device)
        if distributed: dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        train_loss = (totals[0] / totals[1].clamp_min(1)).item(); v = (totals[2] / totals[3].clamp_min(1)).item()
        if rank == 0: log(f"epoch {epoch}: train_loss={train_loss:.6f} val_loss={v:.6f}")
        improved = v < best
        if improved: best, bad = v, 0
        else: bad += 1
        raw_model = model.module if distributed else model
        state = {"model": raw_model.state_dict(), "optimizer": opt.state_dict(), "epoch": epoch,
                 "val_loss": v, "best_val_loss": best, "bad_epochs": bad, "sr": args.wave_sr,
                 "data_recipe": data_recipe}
        if rank == 0:
            torch.save(state, last_path)
            if improved: torch.save(state, out / "best.pt")
        if distributed: dist.barrier()
        if bad >= args.patience:
            if rank == 0: log(f"Early stopping after {bad} unimproved epochs")
            break
    if distributed: dist.destroy_process_group()


def enhance_wave(path: Path, checkpoint: Path, sr: int, cuda: bool,
                 expected_recipe: str | None = None) -> tuple[np.ndarray, int]:
    torch, Waveunet = wave_imports(); device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")
    if not checkpoint.exists(): raise FileNotFoundError(f"Best Wave-U-Net checkpoint not found: {checkpoint}; run --stage train first")
    cache = getattr(enhance_wave, "_cache", {})
    key = (str(checkpoint.resolve()), sr, str(device))
    if key not in cache:
        model = make_wave_model(torch, Waveunet, sr); state = torch.load(checkpoint, map_location=device, weights_only=False)
        if expected_recipe is not None and state.get("data_recipe") != expected_recipe:
            raise RuntimeError("Checkpoint does not match the prepared MS-SNSD test recipe; retrain with --no-resume")
        model.load_state_dict(state["model"]); model.to(device).eval(); cache[key] = model
        enhance_wave._cache = cache
        log(f"Loaded Wave-U-Net best checkpoint: {checkpoint.name} (epoch={state.get('epoch', 'unknown')}, val_loss={state.get('val_loss', float('nan')):.6f})")
    model = cache[key]; x = audio(path, sr)
    left, out_n, in_n = model.shapes["output_start_frame"], model.output_size, model.input_size
    pieces = []
    with torch.no_grad():
        for start in range(0, len(x), out_n):
            z = np.pad(x[max(0,start-left):min(len(x),start-left+in_n)],
                       (max(0,left-start), max(0,start-left+in_n-len(x))))[:in_n]
            p = model(torch.from_numpy(z[None,None].astype(np.float32)).to(device))["clean"][0,0].cpu().numpy()
            pieces.append(p)
    return np.concatenate(pieces)[:len(x)], sr


def align_audio(clean: np.ndarray, enhanced: np.ndarray, sr: int,
                max_delay_seconds: float = 0.1) -> tuple[np.ndarray, np.ndarray, int]:
    """Estimate a small fixed delay and return the common aligned interval.

    A positive delay means that ``enhanced`` starts later than ``clean``.  The
    coarse correlation is downsampled for speed, then refined at full rate.
    """
    length = min(len(clean), len(enhanced))
    if length < 2:
        return clean[:length], enhanced[:length], 0
    c, e = clean[:length].astype(np.float64), enhanced[:length].astype(np.float64)
    c -= np.mean(c); e -= np.mean(e)
    max_delay = min(int(round(max_delay_seconds * sr)), length - 1)

    def search(lo: int, hi: int, step: int, stride: int) -> int:
        best_delay, best_score = lo, -float("inf")
        for delay in range(lo, hi + 1, step):
            if delay >= 0:
                cc, ee = c[:length-delay:stride], e[delay:length:stride]
            else:
                cc, ee = c[-delay:length:stride], e[:length+delay:stride]
            # Elementwise sums avoid repeatedly dispatching tiny BLAS dot jobs.
            denom = math.sqrt(float(np.sum(cc*cc) * np.sum(ee*ee))) + 1e-12
            score = float(np.sum(cc*ee)) / denom
            if score > best_score:
                best_delay, best_score = delay, score
        return best_delay

    # Locate the peak cheaply at 8 kHz, then refine only the neighbouring
    # full-rate samples with normalized correlation.
    factor = max(1, sr // 8000)
    coarse_c, coarse_e = c[::factor], e[::factor]
    corr = correlate(coarse_e, coarse_c, mode="full", method="fft")
    lags = correlation_lags(len(coarse_e), len(coarse_c), mode="full")
    keep = np.abs(lags) <= math.ceil(max_delay / factor)
    coarse_delay = int(lags[keep][np.argmax(corr[keep])]) * factor
    best_delay = search(max(-max_delay, coarse_delay-factor),
                        min(max_delay, coarse_delay+factor), 1, 1)

    if best_delay >= 0:
        return clean[:length-best_delay], enhanced[best_delay:length], best_delay
    return clean[-best_delay:length], enhanced[:length+best_delay], best_delay


def metric_values(clean: np.ndarray, enhanced: np.ndarray, sr: int) -> dict[str, float]:
    c, e, delay = align_audio(clean, enhanced, sr); length = len(c)
    err = c-e; snr = 10*np.log10((np.sum(c*c)+1e-12)/(np.sum(err*err)+1e-12))
    frame, hop, vals = int(.03*sr), int(.015*sr), []
    for i in range(0, max(1, length-frame+1), max(1, hop)):
        cc, ee = c[i:i+frame], err[i:i+frame]
        if len(cc): vals.append(np.clip(10*np.log10((np.sum(cc*cc)+1e-12)/(np.sum(ee*ee)+1e-12)), -10, 35))
    try:
        from pystoi import stoi
        stoi_v = float(stoi(c, e, sr, extended=False))
    except ImportError as ex: raise RuntimeError("Install evaluation dependencies: pip install pesq pystoi") from ex
    from pesq import pesq
    if sr not in (8000,16000): c16, e16 = audio_array_resample(c,sr,16000), audio_array_resample(e,sr,16000); psr=16000
    else: c16,e16,psr=c,e,sr
    # Scale-invariant SDR is the source-separation paper's central metric.
    c0, e0 = c-np.mean(c), e-np.mean(e)
    target = np.sum(e0*c0)*c0/(np.sum(c0*c0)+1e-12)
    sisdr = 10*np.log10((np.sum(target*target)+1e-12)/(np.sum((e0-target)**2)+1e-12))
    return {"Delay_samples": delay, "Delay_ms": 1000.0*delay/sr, "Aligned_samples": length,
            "SNR_dB": float(snr), "SSNR_dB": float(np.mean(vals)), "SI_SDR_dB": float(sisdr),
            "PESQ": float(pesq(psr,c16,e16,"wb" if psr==16000 else "nb")), "STOI": stoi_v}


def audio_array_resample(x, old, new):
    g=math.gcd(old,new); return resample_poly(x,new//g,old//g).astype(np.float32)


def test(method: str, dataset: str, args: argparse.Namespace) -> None:
    if dataset == "ms-snsd" and not paired_files(dataset, "test"): preprocess_ms(args.seed, args.force)
    pairs = paired_files(dataset, "test")
    result_name = "wave-u-net_ms-snsd-indomain" if method == "wave-u-net" and dataset == "ms-snsd" else f"{method}_{dataset}"
    out = RESULTS / result_name
    audio_out = out / "enhanced"; audio_out.mkdir(parents=True, exist_ok=True); rows=[]
    if args.max_test: pairs = pairs[:args.max_test]
    for i,(noisy,clean_p) in enumerate(pairs,1):
        if method == "hybrid":
            sr=48000; tmp_in=out/"_input.pcm"; tmp_out=out/"_output.pcm"
            (audio(noisy,sr)*32767).astype("<i2").tofile(tmp_in)
            trained=ROOT/".prepared"/f"hybrid-{dataset}"/"rnnoise_demo.exe"
            if not trained.exists(): raise FileNotFoundError(f"Trained executable not found: {trained}; run --stage train first")
            run([trained,tmp_in,tmp_out],ROOT/"Hybrid")
            enhanced=np.fromfile(tmp_out,"<i2").astype(np.float32)/32768; tmp_in.unlink(); tmp_out.unlink()
        else:
            checkpoint_name = "wave-u-net-ms-snsd-long" if dataset == "ms-snsd" else f"wave-u-net-{dataset}"
            expected_recipe = "in-domain-speaker-noise-disjoint-v5" if dataset == "ms-snsd" else None
            enhanced,sr=enhance_wave(noisy,ROOT/".prepared"/checkpoint_name/"best.pt",args.wave_sr,not args.cpu,
                                     expected_recipe=expected_recipe)
        clean=audio(clean_p,sr); values=metric_values(clean,enhanced,sr)
        baseline=metric_values(clean,audio(noisy,sr),sr)
        values.update({f"Input_{key}": baseline[key] for key in ("SNR_dB","SSNR_dB","SI_SDR_dB","PESQ","STOI")})
        values["file"]=clean_p.name; rows.append(values)
        write(audio_out/clean_p.name,enhanced,sr); log(f"test {i}/{len(pairs)} {clean_p.name}: "+" ".join(f"{k}={v:.4f}" for k,v in values.items() if k!='file'))
    if not rows: raise RuntimeError("No test pairs found")
    fields=["file","Delay_samples","Delay_ms","Aligned_samples","SNR_dB","SSNR_dB","SI_SDR_dB","PESQ","STOI",
            "Input_SNR_dB","Input_SSNR_dB","Input_SI_SDR_dB","Input_PESQ","Input_STOI"]
    with (out/"per_file.csv").open("w",newline="",encoding="utf-8-sig") as f: w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
    metric_fields=["SNR_dB","SSNR_dB","SI_SDR_dB","PESQ","STOI"]
    summary={k:float(np.mean([r[k] for r in rows])) for k in metric_fields}
    summary.update({f"Input_{k}":float(np.mean([r[f'Input_{k}'] for r in rows])) for k in metric_fields})
    summary.update({f"Delta_{k}":summary[k]-summary[f"Input_{k}"] for k in metric_fields})
    summary["mean_delay_samples"]=float(np.mean([r["Delay_samples"] for r in rows]))
    summary["mean_delay_ms"]=float(np.mean([r["Delay_ms"] for r in rows])); summary["files"]=len(rows)
    (out/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")
    with (out/"summary.txt").open("w",encoding="utf-8") as f:
        f.write(f"method={method}\ndataset={dataset}\nfiles={len(rows)}\n"); f.writelines(f"{k}: {v:.6f}\n" for k,v in summary.items() if k!='files')
    log(f"Results written to {out}")


def main(fixed_method: str | None=None, fixed_dataset: str | None=None) -> None:
    p=argparse.ArgumentParser(description="Speech denoising train/test pipeline")
    p.add_argument("--stage",choices=["preprocess","train","test","all"],default="all")
    p.add_argument("--seed",type=int,default=0);p.add_argument("--force",action="store_true")
    p.add_argument("--sequences",type=int,default=10_000,help="Official RNNoise minimum; each sequence is 2,000 frames")
    p.add_argument("--frames",type=int,default=None,help="Compatibility alias; rounded up to 2,000-frame sequences")
    # 10k sequences / batch 8 / 60 epochs = 75k updates, matching upstream guidance;
    # batch 8 is deliberately sized for this machine's 6 GiB RTX 4050.
    p.add_argument("--epochs",type=int,default=60);p.add_argument("--batch-size",type=int,default=8)
    p.add_argument("--cpu",action="store_true",help="Disable CUDA training/inference")
    p.add_argument("--no-resume",action="store_true",help="Start training without resuming an existing checkpoint")
    p.add_argument("--sparse",action="store_true",help="Enable official structured GRU sparsification")
    p.add_argument("--wave-epochs",type=int,default=100);p.add_argument("--wave-batch-size",type=int,default=4);p.add_argument("--wave-sr",type=int,default=16000)
    p.add_argument("--patience",type=int,default=20);p.add_argument("--workers",type=int,default=0);p.add_argument("--log-interval",type=int,default=10)
    p.add_argument("--preprocess-workers",type=int,default=0,help="Wave cache workers; 0 selects up to 8 CPU cores")
    p.add_argument("--cuda",action="store_true",help="Compatibility flag; Wave-U-Net now uses CUDA automatically")
    p.add_argument("--max-test",type=int,default=None)
    args=p.parse_args(); method,dataset=fixed_method,fixed_dataset
    if not method or not dataset: p.error("Use one of the four dataset-specific entry scripts")
    log(f"Pipeline: {method} + {dataset}; stage={args.stage}")
    if args.stage in ("preprocess","all"):
        (prepare_hybrid(dataset,args) if method=="hybrid" else preprocess_wave(dataset,args))
    if args.stage in ("train","all"): (train_hybrid(dataset,args) if method=="hybrid" else train_wave(dataset,args))
    if args.stage in ("test","all"): test(method,dataset,args)
