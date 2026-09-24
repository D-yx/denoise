"""Evaluate one checkpoint on the official MUSDB18-HQ test split."""
import argparse
import json
import os
import pickle

import numpy as np
import torch

import model.utils as model_utils
from data.musdb import get_musdb_folds
from model.waveunet import Waveunet
from test import evaluate


def main(args):
    if args.cuda and not torch.cuda.is_available():
        raise RuntimeError("--cuda was requested, but CUDA is not available")
    features = ([args.features * i for i in range(1, args.levels + 1)]
                if args.feature_growth == "add"
                else [args.features * 2 ** i for i in range(args.levels)])
    model = Waveunet(args.channels, features, args.channels, args.instruments,
                     kernel_size=args.kernel_size,
                     target_output_size=int(args.output_size * args.sr),
                     depth=args.depth, strides=args.strides,
                     conv_type=args.conv_type, res=args.res,
                     separate=args.separate)
    if args.cuda:
        model = model_utils.DataParallel(model).cuda()
    model_utils.load_model(model, None, args.checkpoint, args.cuda)
    test_tracks = get_musdb_folds(args.dataset_dir)["test"]
    metrics = evaluate(args, test_tracks, model, args.instruments)
    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, "results.pkl"), "wb") as handle:
        pickle.dump(metrics, handle)
    summary = {
        instrument: {
            metric: float(np.nanmedian(np.concatenate([
                song[instrument][metric] for song in metrics
            ])))
            for metric in ("SDR", "ISR", "SIR", "SAR")
        }
        for instrument in args.instruments
    }
    with open(os.path.join(args.output, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="evaluation")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--instruments", nargs="+", default=["bass", "drums", "other", "vocals"])
    parser.add_argument("--features", type=int, default=32)
    parser.add_argument("--levels", type=int, default=6)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--sr", type=int, default=44100)
    parser.add_argument("--channels", type=int, default=2)
    parser.add_argument("--kernel_size", type=int, default=5)
    parser.add_argument("--output_size", type=float, default=2.0)
    parser.add_argument("--strides", type=int, default=4)
    parser.add_argument("--conv_type", default="gn", choices=["normal", "bn", "gn"])
    parser.add_argument("--res", default="fixed", choices=["fixed", "learned"])
    parser.add_argument("--separate", type=int, default=1)
    parser.add_argument("--feature_growth", default="double", choices=["add", "double"])
    main(parser.parse_args())

