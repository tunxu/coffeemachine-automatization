import numpy as np
import wave
import struct
from pathlib import Path

OUT_DIR = Path("data/noise")
SAMPLE_RATE = 16000
DURATION = 3
TARGET_SAMPLES = SAMPLE_RATE * DURATION
TARGET_COUNT = 2250

OUT_DIR.mkdir(parents=True, exist_ok=True)

# Count existing noise samples
existing = list(OUT_DIR.glob("*.wav"))
current_count = len(existing)
needed = TARGET_COUNT - current_count
print(f"Existing: {current_count} | Need to generate: {needed}")


def write_wav(path, samples):
    samples = (samples * 32767).astype(np.int16)
    with wave.open(str(path), "w") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SAMPLE_RATE)
        f.writeframes(struct.pack(f"<{len(samples)}h", *samples.tolist()))


for i in range(needed):
    choice = np.random.randint(0, 4)

    if choice == 0:
        # Pure silence
        audio = np.zeros(TARGET_SAMPLES, dtype=np.float32)

    elif choice == 1:
        # White noise at low volume
        audio = np.random.normal(0, 0.02, TARGET_SAMPLES).astype(np.float32)

    elif choice == 2:
        # Pink-ish noise (low frequency heavy)
        white = np.random.randn(TARGET_SAMPLES)
        b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
        a = [1, -2.494956002, 2.017265875, -0.522189400]
        from scipy.signal import lfilter
        audio = lfilter(b, a, white).astype(np.float32)
        audio = (audio / np.max(np.abs(audio)) * 0.05).astype(np.float32)

    elif choice == 3:
        # Random amplitude bursts (simulates distant sounds)
        audio = np.zeros(TARGET_SAMPLES, dtype=np.float32)
        n_bursts = np.random.randint(1, 5)
        for _ in range(n_bursts):
            start = np.random.randint(0, TARGET_SAMPLES - SAMPLE_RATE)
            length = np.random.randint(SAMPLE_RATE // 4, SAMPLE_RATE)
            burst = np.random.normal(0, np.random.uniform(0.01, 0.05), length)
            audio[start:start + length] += burst
        audio = np.clip(audio, -1.0, 1.0)

    out_path = OUT_DIR / f"noise_{current_count + i:05d}.wav"
    write_wav(out_path, audio)

print(f"\nDone! Noise folder now has {TARGET_COUNT} samples.")