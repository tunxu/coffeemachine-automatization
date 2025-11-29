import os
import torch
import torchaudio
import soundfile as sf

noise_waveform, noise_sr = torchaudio.load("data/noise.wav")

# splitting the tensor into 1 second segments and saving them
os.makedirs("data/noise", exist_ok=True)
for i in range(0, noise_waveform.shape[1], noise_sr):
    segment = noise_waveform[:, i:i+noise_sr]
    if segment.shape[1] < noise_sr:
        break
    out_path = os.path.join("data/noise", f"noise_{i//noise_sr:03d}.wav")
    sf.write(out_path, segment.squeeze().cpu().numpy(), noise_sr)
    print(f"Saved: {out_path} ({segment.shape[1] / noise_sr:.2f}s)")
