
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

try:
    import litert_torch
except ImportError:
    print("Error: litert_torch is not installed. Run: pip install litert-torch", file=sys.stderr)
    raise


class DSCNN(nn.Module):
    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            self._ds_block(32, 32),
            self._ds_block(32, 32),
            self._ds_block(32, 32),
            self._ds_block(32, 32),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, num_classes),
        )

    def _ds_block(self, in_ch: int, out_ch: int):
        return nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(),

            nn.Conv2d(in_ch, out_ch, kernel_size=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.model(x)


def load_checkpoint(model: nn.Module, checkpoint_path: str):
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    # Support either a raw state_dict or a wrapped checkpoint
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    # Strip a leading "module." if the model was saved with DataParallel
    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            cleaned[k[len("module."):]] = v
        else:
            cleaned[k] = v

    model.load_state_dict(cleaned, strict=True)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint")
    parser.add_argument("--output", required=True, help="Path to output .tflite file")
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--n-mels", type=int, default=64)
    parser.add_argument("--time-frames", type=int, default=101)
    parser.add_argument("--verify", action="store_true", help="Run a numeric sanity check after conversion")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    output = Path(args.output)

    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    # Build and load model
    model = DSCNN(num_classes=args.num_classes)
    model = load_checkpoint(model, str(checkpoint))
    model.eval()

    # Example input in NCHW format: [batch, channel, mel, time]
    sample_input = torch.randn(1, 1, args.n_mels, args.time_frames, dtype=torch.float32)
    sample_inputs = (sample_input,)

    # Run PyTorch once
    with torch.no_grad():
        torch_out = model(*sample_inputs).detach().cpu().numpy()

    # Convert to LiteRT / TFLite
    edge_model = litert_torch.convert(model, sample_inputs)

    # Optional sanity check
    if args.verify:
        edge_out = edge_model(*sample_inputs)
        if np.allclose(torch_out, edge_out, atol=1e-4, rtol=1e-4):
            print("Conversion check passed: PyTorch and TFLite outputs are close.")
        else:
            max_abs = np.max(np.abs(torch_out - edge_out))
            print(f"Warning: outputs differ. max_abs_diff={max_abs:.6f}")

    # Export .tflite
    edge_model.export(str(output))
    print(f"Saved TFLite model to: {output}")


if __name__ == "__main__":
    main()