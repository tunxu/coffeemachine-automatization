import os
import torch
import torchaudio

mutated_audio = []

# Load list of previously mutated files
if os.path.exists("mutated.txt"):
    with open("mutated.txt", "rt") as file:
        for line in file:
            mutated_audio.append(line.strip())

with open("mutated.txt", "w") as mutated_file:
    print("Successfully opened mutated.txt")

    for root, dirs, files in os.walk("data", topdown=False):
        for name in files:

            # Ignore previously mutated or generated files
            if (
                name in mutated_audio
                or name.endswith("_pitch_up.wav")
                or name.endswith("_pitch_down.wav")
                or name.endswith("_noise.wav")
                or name.endswith("_volshift.wav")
            ):
                continue

            full_path = os.path.join(root, name)
            print(f"Processing: {full_path}")

            # Load audio
            audio_tensor, sample_rate = torchaudio.load(full_path)

            # Convert to mono if needed
            if audio_tensor.shape[0] > 1:
                audio_tensor = torch.mean(audio_tensor, dim=0, keepdim=True)
                print(" -> Converted to mono")

            # ------------------------------------
            # PITCH UP
            # ------------------------------------
            pitch_up = torchaudio.transforms.PitchShift(sample_rate, n_steps=2)
            pitched_up = pitch_up(audio_tensor)
            pitch_up_name = f"{name[:-4]}_pitch_up.wav"
            torchaudio.save(os.path.join(root, pitch_up_name), pitched_up, sample_rate)
            print(f" -> Created {pitch_up_name}")

            # ------------------------------------
            # PITCH DOWN
            # ------------------------------------
            pitch_down = torchaudio.transforms.PitchShift(sample_rate, n_steps=-2)
            pitched_down = pitch_down(audio_tensor)
            pitch_down_name = f"{name[:-4]}_pitch_down.wav"
            torchaudio.save(os.path.join(root, pitch_down_name), pitched_down, sample_rate)
            print(f" -> Created {pitch_down_name}")

            # ------------------------------------
            # ADD NOISE
            # ------------------------------------
            noise_strength = 0.015  # adjust noise amount here
            noise = torch.randn_like(audio_tensor) * noise_strength
            noisy_audio = torch.clamp(audio_tensor + noise, -1.0, 1.0)

            noise_name = f"{name[:-4]}_noise.wav"
            torchaudio.save(os.path.join(root, noise_name), noisy_audio, sample_rate)
            print(f" -> Created {noise_name}")

            # ------------------------------------
            # VOLUME SHIFT (random)
            # ------------------------------------
            # Example: multiply waveform by a random factor between 0.7 and 1.3
            volume_factor = torch.empty(1).uniform_(0.7, 1.3).item()
            volshift_audio = torch.clamp(audio_tensor * volume_factor, -1.0, 1.0)

            volshift_name = f"{name[:-4]}_volshift.wav"
            torchaudio.save(os.path.join(root, volshift_name), volshift_audio, sample_rate)
            print(f" -> Volume shifted by factor {volume_factor:.2f} → {volshift_name}")

            # ------------------------------------
            # Record all processed files
            # ------------------------------------
            mutated_file.write(f"{name}\n")
            mutated_file.write(f"{pitch_up_name}\n")
            mutated_file.write(f"{pitch_down_name}\n")
            mutated_file.write(f"{noise_name}\n")
            mutated_file.write(f"{volshift_name}\n")
