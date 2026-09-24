"""Fast environment/model check; does not require MUSDB18-HQ or a checkpoint."""
import argparse
import torch

from model.waveunet import Waveunet


parser = argparse.ArgumentParser()
parser.add_argument("--cuda", action="store_true")
args = parser.parse_args()

if args.cuda and not torch.cuda.is_available():
    raise RuntimeError("--cuda was requested, but CUDA is not available")
device = torch.device("cuda" if args.cuda else "cpu")
model = Waveunet(2, [8, 16, 32], 2, ["vocals"], kernel_size=5,
                 target_output_size=2048, depth=1, strides=2,
                 conv_type="gn", res="fixed", separate=1).to(device)
x = torch.randn(1, 2, model.shapes["input_frames"], device=device)
with torch.no_grad():
    y = model(x, "vocals")["vocals"]
assert y.shape == (1, 2, model.shapes["output_frames"]), y.shape
print(f"OK device={device}, torch={torch.__version__}, output={tuple(y.shape)}")
