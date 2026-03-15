import wave
import random
import struct
from pathlib import Path
from piper import PiperVoice, SynthesisConfig

TEXTS = [
    "Ok Fernando",
    "Kaffee groß",
    "Kaffee klein",
]

VOICES_DIR = Path("piper-voices-de")
OUT_DIR = Path("data")
SAMPLES_PER_VOICE = 250
TARGET_RATE = 16000
TARGET_SAMPLES = TARGET_RATE*3  # 3 seconds


def random_config():
    return SynthesisConfig(
        length_scale=round(random.uniform(0.5, 2.0), 3),
        noise_scale=round(random.uniform(0.0, 1.0), 3),
        noise_w_scale=round(random.uniform(0.0, 1.0), 3),
        volume=round(random.uniform(0.5, 1.0), 3),
    )


def resample_linear(samples, src_rate, dst_rate):
    if src_rate == dst_rate:
        return samples
    ratio = src_rate / dst_rate
    out_len = int(len(samples) / ratio)
    output = []
    for i in range(out_len):
        src_pos = i * ratio
        lo = int(src_pos)
        hi = min(lo + 1, len(samples) - 1)
        frac = src_pos - lo
        output.append(int(samples[lo] * (1 - frac) + samples[hi] * frac))
    return output


def fit_to_duration(samples, target_len):
    if len(samples) >= target_len:
        return samples[:target_len]
    return samples + [0] * (target_len - len(samples))


def synthesize_to_samples(voice, text, syn_config):
    buf = []
    src_rate = [TARGET_RATE]

    class CapturingWave:
        def __init__(self):
            self._framerate = TARGET_RATE
            self._nchannels = 1
            self._sampwidth = 2

        def setnchannels(self, n): self._nchannels = n
        def setsampwidth(self, w): self._sampwidth = w
        def setframerate(self, r):
            self._framerate = r
            src_rate[0] = r
        def writeframes(self, data):
            for i in range(0, len(data), 2):
                buf.append(struct.unpack_from("<h", data, i)[0])
        def close(self): pass

    voice.synthesize_wav(text, CapturingWave(), syn_config=syn_config)

    resampled = resample_linear(buf, src_rate[0], TARGET_RATE)
    return fit_to_duration(resampled, TARGET_SAMPLES)


def write_wav(path, samples):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "w") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(TARGET_RATE)
        f.writeframes(struct.pack(f"<{len(samples)}h", *samples))


model_paths = sorted(VOICES_DIR.glob("**/*.onnx"))
print(f"Found {len(model_paths)} voices\n")

for text in TEXTS:
    safe_text = text.lower().replace(" ", "_").replace("ß", "ss")
    print(f"[{safe_text}]")

    count = 1
    for model_path in model_paths:
        voice_name = model_path.stem
        print(f"  {voice_name} ({SAMPLES_PER_VOICE} samples)")
        voice = PiperVoice.load(str(model_path))

        for _ in range(SAMPLES_PER_VOICE):
            cfg = random_config()
            samples = synthesize_to_samples(voice, text, cfg)
            out_path = OUT_DIR / safe_text / f"{voice_name}_{count}.wav"
            write_wav(out_path, samples)
            count += 1

print("\nDone!")