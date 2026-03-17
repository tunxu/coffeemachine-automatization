import soundfile as sf
import numpy as np
import wave
import struct
from pathlib import Path

BACKGROUND_NOISE_DIR = Path("_background_noise_")
OUT_DIR = Path("data/noise")
SAMPLE_RATE = 16000
DURATION = 3
TARGET_SAMPLES = SAMPLE_RATE * DURATION

OUT_DIR.mkdir(parents=True, exist_ok=True)


def write_wav(path, samples):
    samples = (samples * 32767).astype(np.int16)
    with wave.open(str(path), "w") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SAMPLE_RATE)
        f.writeframes(struct.pack(f"<{len(samples)}h", *samples.tolist()))


noise_files = list(BACKGROUND_NOISE_DIR.glob("*.wav"))
print(f"Found {len(noise_files)} background noise files\n")

count = 0
for noise_file in noise_files:
    print(f"Processing: {noise_file.name}")
    audio, sr = sf.read(str(noise_file), dtype="float32", always_2d=True)

    # Mix down to mono
    if audio.shape[1] > 1:
        audio = audio.mean(axis=1)
    else:
        audio = audio[:, 0]

    # Resample to 16kHz if needed
    if sr != SAMPLE_RATE:
        print(f"  Resampling from {sr}Hz to {SAMPLE_RATE}Hz")
        ratio = SAMPLE_RATE / sr
        new_len = int(len(audio) * ratio)
        indices = np.linspace(0, len(audio) - 1, new_len)
        lo = np.floor(indices).astype(int)
        hi = np.minimum(lo + 1, len(audio) - 1)
        frac = indices - lo
        audio = audio[lo] * (1 - frac) + audio[hi] * frac

    # Slice into chunks
    n_slices = len(audio) // TARGET_SAMPLES
    print(f"  {len(audio)} samples → {n_slices} slices of {DURATION}s")

    for i in range(n_slices):
        chunk = audio[i * TARGET_SAMPLES:(i + 1) * TARGET_SAMPLES]
        write_wav(OUT_DIR / f"noise_{count:05d}.wav", chunk)
        count += 1

print(f"\nDone! Created {count} noise samples in {OUT_DIR}")