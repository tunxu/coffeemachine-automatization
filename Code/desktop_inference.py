#!/usr/bin/env python3
"""
Real-time keyword spotting inference from microphone using the model from fernando_classifier.ipynb

Preprocessing matches notebook:
- 16 kHz, mono
- normalize loudness: waveform / max(abs(waveform))
- pad/trim to 1 second (16000 samples)
- MelSpectrogram(sample_rate=16000, n_mels=64, n_fft=1024, hop_length=512)
Model matches notebook: KeywordCNN with 3 conv blocks + global avg pool.

Run:
  pip install torch torchaudio sounddevice numpy
  python kws_mic_infer.py --ckpt checkpoints/small_kws.pt --labels labels.txt

labels.txt: one class name per line in the SAME order as training
(= sorted(os.listdir(root_dir)) used in the notebook).
"""

import argparse
import time
from collections import deque
from pathlib import Path
import numpy as np
import sounddevice as sd
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T


# ----------------------------
# Model: EXACT copy from notebook
# ----------------------------
class KeywordCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)

        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        # x: [batch, 1, n_mels, time]
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = x.mean(dim=[2, 3])  # global average pooling -> [batch, channels]
        x = self.dropout(x)
        x = self.fc(x)
        return x


def load_labels(args) -> list[str]:
    if args.labels:
        lines = Path(args.labels).read_text(encoding="utf-8").splitlines()
        labels = [ln.strip() for ln in lines if ln.strip()]
        if not labels:
            raise ValueError("labels.txt is empty.")
        return labels

    if args.data_root:
        root = Path(args.data_root)
        if not root.exists():
            raise FileNotFoundError(f"--data-root not found: {root}")
        labels = sorted([p.name for p in root.iterdir() if p.is_dir()])
        if not labels:
            raise ValueError(f"No class folders found in: {root}")
        return labels

    # Fallback: user didn't provide labels
    raise ValueError("Provide --labels labels.txt or --data-root TRAIN_DATA_DIR to determine class order.")


def pick_input_device(device_query: str | None) -> int | None:
    """
    If device_query is None: let sounddevice pick default.
    If provided: match substring against input device names.
    Returns device index or None.
    """
    if not device_query:
        return None

    device_query_l = device_query.lower()
    devices = sd.query_devices()
    candidates = []
    for idx, d in enumerate(devices):
        if d["max_input_channels"] > 0 and device_query_l in d["name"].lower():
            candidates.append((idx, d["name"]))

    if not candidates:
        names = [f"{i}: {d['name']}" for i, d in enumerate(devices) if d["max_input_channels"] > 0]
        raise ValueError(
            f"No input device matched '{device_query}'. Available input devices:\n" + "\n".join(names)
        )

    # pick first match
    idx, name = candidates[0]
    print(f"[audio] Using input device {idx}: {name}")
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to .pt state_dict (e.g. checkpoints/small_kws.pt)")
    ap.add_argument("--labels", default=None, help="labels.txt (one label per line, in training order)")
    ap.add_argument("--data-root", default=None, help="Training data root to infer labels = sorted(subfolders)")
    ap.add_argument("--device", default=None, help="Substring to match microphone device name (optional)")

    ap.add_argument("--sample-rate", type=int, default=16000)
    ap.add_argument("--window-ms", type=int, default=1000)
    ap.add_argument("--hop-ms", type=int, default=200)

    ap.add_argument("--threshold", type=float, default=0.80, help="Confidence threshold for keyword hit")
    ap.add_argument("--require-n", type=int, default=3, help="Require N hits...")
    ap.add_argument("--in-last", type=int, default=5, help="...within last M frames")
    ap.add_argument("--cooldown", type=float, default=1.0, help="Seconds after trigger to ignore new triggers")

    ap.add_argument("--topk", type=int, default=3, help="Print top-k classes each frame")
    ap.add_argument("--target", default=None, help="Optional: label name to treat as the 'keyword' for triggering")
    args = ap.parse_args()

    labels = load_labels(args)
    num_classes = len(labels)

    # target keyword index (optional)
    target_idx = None
    if args.target is not None:
        if args.target not in labels:
            raise ValueError(f"--target '{args.target}' not in labels: {labels}")
        target_idx = labels.index(args.target)
        print(f"[kws] Target keyword: '{args.target}' (index {target_idx})")

    # Torch device
    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[torch] Using device: {torch_device}")

    # Load model
    model = KeywordCNN(num_classes=num_classes).to(torch_device).eval()
    state = torch.load(args.ckpt, map_location=torch_device)
    model.load_state_dict(state)
    print(f"[model] Loaded checkpoint: {args.ckpt}")

    # Mel transform EXACT params from notebook
    mel = T.MelSpectrogram(
        sample_rate=args.sample_rate,
        n_mels=64,
        n_fft=1024,
        hop_length=512,
    ).to(torch_device)

    # Audio buffering
    window_samples = int(args.sample_rate * args.window_ms / 1000)
    hop_samples = int(args.sample_rate * args.hop_ms / 1000)
    ring = deque(maxlen=window_samples)

    # Trigger smoothing
    recent_hits = deque(maxlen=args.in_last)
    last_fire = 0.0

    # Pick mic device if requested
    in_dev_idx = pick_input_device(args.device)
    if in_dev_idx is not None:
        sd.default.device = (in_dev_idx, None)

    def callback(indata, frames, time_info, status):
        if status:
            print(status)
        # indata is float32 in [-1,1], shape [frames, channels]
        ring.extend(indata[:, 0].astype(np.float32))

    print("[audio] Starting microphone stream... Ctrl+C to stop.")
    with sd.InputStream(
        samplerate=args.sample_rate,
        channels=1,
        dtype="float32",
        blocksize=hop_samples,
        callback=callback,
    ):
        # wait until ring fills first time
        while len(ring) < window_samples:
            time.sleep(0.01)

        while True:
            if len(ring) < window_samples:
                continue

            # 1) Get latest 1s window
            x = np.array(ring, dtype=np.float32)  # shape [T]

            # 2) Normalize loudness like training (waveform / max(abs))
            mx = float(np.max(np.abs(x))) if x.size else 0.0
            if mx > 0:
                x = x / mx

            # 3) Ensure exact length (should already be 1s due to ring maxlen,
            #    but keep for safety)
            if x.shape[0] < window_samples:
                x = np.pad(x, (0, window_samples - x.shape[0]))
            else:
                x = x[:window_samples]

            # 4) To torch: [1, 1, T]
            wav = torch.from_numpy(x).to(torch_device).unsqueeze(0).unsqueeze(0)

            # 5) Mel: torchaudio expects [batch, time] OR [channels, time] typically.
            #    In your training, you used transform(waveform) with waveform [1, T]
            wav_for_mel = wav.squeeze(0)   # [1, T]
            feats = mel(wav_for_mel)       # [1, n_mels, frames]
            feats = feats.unsqueeze(0)     # [1, 1, n_mels, frames]  (batch=1, channel=1)

            with torch.no_grad():
                logits = model(feats)
                probs = torch.softmax(logits, dim=-1).squeeze(0)  # [C]

            # Print top-k
            topk = min(args.topk, num_classes)
            vals, idxs = torch.topk(probs, k=topk)
            top_str = " | ".join([f"{labels[i]}: {vals[j].item():.2f}" for j, i in enumerate(idxs.tolist())])
            print(f"\r{top_str}   ", end="")

            # Trigger logic (only if target provided)
            if target_idx is not None:
                conf = probs[target_idx].item()
                hit = conf >= args.threshold
                recent_hits.append(1 if hit else 0)

                now = time.time()
                if sum(recent_hits) >= args.require_n and (now - last_fire) > args.cooldown:
                    print(f"\nDETECTED '{labels[target_idx]}' (conf={conf:.2f})")
                    last_fire = now

            time.sleep(args.hop_ms / 1000.0)


if __name__ == "__main__":
    main()
