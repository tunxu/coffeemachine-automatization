"""Generate spoken WAV samples using Piper TTS voices.

This script synthesizes a set of utterances for every voice model found under
`piper-voices-de/`. It creates _N_ variations per prompt by randomizing the
Piper `SynthesisConfig` and saves the results as 16kHz mono WAV files.

Usage:
    python generate_samples.py

The output is written into a structure like:
    data/<prompt>/<voice_name>_<index>.wav

Dependencies:
  - piper (https://github.com/rhasspy/piper)
  - soundfile / libsndfile (indirectly used by other tools)

"""

import argparse
import random
import struct
import wave
from pathlib import Path
from typing import List, Optional

from piper import PiperVoice, SynthesisConfig

# Text prompts to synthesize.
TEXTS = [
    "Ok Fernando",
    "Kaffee groß",
    "Kaffee klein",
]

# Input voices directory (contains .onnx voice models).
VOICES_DIR = Path("piper-voices-de")

# Output folder for generated WAVs.
OUT_DIR = Path("data")

# How many samples to generate per voice.
SAMPLES_PER_VOICE = 250

# Target sample rate (16kHz is typical for keyword spotting).
TARGET_RATE = 16000

# Target length (in samples) per output file.
TARGET_SAMPLES = TARGET_RATE * 3  # 3 seconds


def random_config() -> SynthesisConfig:
    """Create a randomized Piper `SynthesisConfig`.

    The randomization helps generate subtle variations in pitch, speed, and noise.

    Returns:
        A configured `SynthesisConfig` instance.
    """

    return SynthesisConfig(
        length_scale=round(random.uniform(0.5, 2.0), 3),
        noise_scale=round(random.uniform(0.0, 1.0), 3),
        noise_w_scale=round(random.uniform(0.0, 1.0), 3),
        volume=round(random.uniform(0.5, 1.0), 3),
    )


def resample_linear(samples: List[int], src_rate: int, dst_rate: int) -> List[int]:
    """Resample a PCM waveform using simple linear interpolation.

    This function is intentionally lightweight and avoids dependencies (e.g. scipy).
    It's good enough for small TTS-generated datasets where the sample rates are
    close (e.g., 16kHz <-> 24kHz).

    Args:
        samples: Input PCM samples (16-bit signed ints).
        src_rate: Original sample rate of `samples`.
        dst_rate: Desired output sample rate.

    Returns:
        Resampled PCM samples at `dst_rate`.
    """

    if src_rate == dst_rate:
        return samples

    ratio = src_rate / dst_rate
    out_len = int(len(samples) / ratio)
    output: List[int] = []

    for i in range(out_len):
        src_pos = i * ratio
        lo = int(src_pos)
        hi = min(lo + 1, len(samples) - 1)
        frac = src_pos - lo
        output.append(int(samples[lo] * (1 - frac) + samples[hi] * frac))

    return output


def fit_to_duration(samples: List[int], target_len: int) -> List[int]:
    """Pad or trim samples so output is exactly `target_len`.

    Any extra samples are discarded. Missing samples are padded with silence.
    """

    if len(samples) >= target_len:
        return samples[:target_len]
    return samples + [0] * (target_len - len(samples))


def synthesize_to_samples(
    voice: PiperVoice,
    text: str,
    syn_config: SynthesisConfig,
    target_rate: int = TARGET_RATE,
    target_samples: int = TARGET_SAMPLES,
) -> List[int]:
    """Synthesize text with a Piper voice into raw PCM samples.

    Piper writes WAV bytes to a file-like object. This helper captures that output
    into a Python list of ints, resamples it to `target_rate`, and forces a fixed
    duration of `target_samples`.

    Args:
        voice: Loaded `PiperVoice` instance.
        text: Text to synthesize.
        syn_config: Piper synthesis config.
        target_rate: Desired output sample rate.
        target_samples: Desired output length in samples.

    Returns:
        A list of 16-bit PCM samples.
    """

    buf: List[int] = []
    src_rate: List[int] = [target_rate]

    class CapturingWave:
        def __init__(self) -> None:
            self._framerate = target_rate
            self._nchannels = 1
            self._sampwidth = 2

        def setnchannels(self, n: int) -> None:  # noqa: D401
            self._nchannels = n

        def setsampwidth(self, w: int) -> None:  # noqa: D401
            self._sampwidth = w

        def setframerate(self, r: int) -> None:  # noqa: D401
            self._framerate = r
            src_rate[0] = r

        def writeframes(self, data: bytes) -> None:  # noqa: D401
            # Piper produces 16-bit little-endian PCM.
            for i in range(0, len(data), 2):
                buf.append(struct.unpack_from("<h", data, i)[0])

        def close(self) -> None:
            pass

    voice.synthesize_wav(text, CapturingWave(), syn_config=syn_config)

    resampled = resample_linear(buf, src_rate[0], target_rate)
    return fit_to_duration(resampled, target_samples)


def write_wav(path: Path, samples: List[int], sample_rate: int = TARGET_RATE) -> None:
    """Write out a list of 16-bit PCM samples as a WAV file.

    Args:
        path: Output file path.
        samples: PCM samples (16-bit signed ints).
        sample_rate: Sample rate to store in the WAV header.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "w") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(struct.pack(f"<{len(samples)}h", *samples))


def _safe_text(text: str) -> str:
    """Create a filesystem-safe folder name from a prompt."""

    return text.lower().replace(" ", "_").replace("ß", "ss")


def generate_samples(
    voices_dir: Path,
    out_dir: Path,
    texts: List[str],
    samples_per_voice: int,
    seed: Optional[int] = None,
) -> None:
    """Generate WAV samples for every voice model in `voices_dir`."""

    if seed is not None:
        random.seed(seed)

    model_paths = sorted(voices_dir.glob("**/*.onnx"))
    print(f"Found {len(model_paths)} voices in {voices_dir}")

    if not model_paths:
        raise SystemExit(f"No voice models found in {voices_dir}.")

    for text in texts:
        safe_text = _safe_text(text)
        print(f"[{safe_text}]")

        count = 1
        for model_path in model_paths:
            voice_name = model_path.stem
            print(f"  {voice_name} ({samples_per_voice} samples)")
            voice = PiperVoice.load(str(model_path))

            for _ in range(samples_per_voice):
                cfg = random_config()
                samples = synthesize_to_samples(voice, text, cfg)
                out_path = out_dir / safe_text / f"{voice_name}_{count}.wav"
                write_wav(out_path, samples)
                count += 1

    print("\nDone!")
# Legacy module entry point (replaced by `generate_samples()` + CLI)

for text in TEXTS:
    safe_text = text.lower().replace(" ", "_").replace("ß", "ss")
    print(f"[{safe_text}]")

    count = 1
    for model_path in model_path:
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