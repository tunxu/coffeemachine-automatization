import os
import torch
import torchaudio
import soundfile as sf


def split_audio_silero_vad(
    input_path,
    output_dir,
    threshold=0.5,
    min_speech_ms=200,
    pad_ms=150,
    merge_speech_gap_ms=150,
    device="cpu",
):
    """
    Split an audio file into speech segments using Silero VAD (torch.hub).
    """

    # --- Ensure output directory exists ---
    os.makedirs(output_dir, exist_ok=True)

    # --- Load audio ---
    wav, sr = torchaudio.load(input_path)
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)  # convert to mono
    wav = wav.to(device)

    # --- Load Silero VAD from torch.hub ---
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                  model='silero_vad',
                                  force_reload=False,
                                  onnx=False)
    (get_speech_timestamps,
     save_audio,
     read_audio,
     VADIterator,
     collect_chunks) = utils

    # --- Detect speech segments ---
    speech_timestamps = get_speech_timestamps(
        wav, model, sampling_rate=sr,
        threshold=threshold, min_speech_duration_ms=min_speech_ms,
        min_silence_duration_ms=merge_speech_gap_ms,
        window_size_samples=1536
    )

    print(f"Detected {len(speech_timestamps)} speech regions")

    # --- Export each speech segment ---
    pad_samples = int((pad_ms / 1000) * sr)

    for i, ts in enumerate(speech_timestamps, 1):
        start = max(0, ts['start'] - pad_samples)
        end = min(wav.shape[-1], ts['end'] + pad_samples)
        segment = wav[:, start:end]

        if segment.shape[-1] < int(sr * (min_speech_ms / 1000)):
            continue  # skip too short

        out_path = os.path.join(output_dir, f"kaffee_tus_{i:03d}.wav")
        sf.write(out_path, segment.squeeze().cpu().numpy(), sr)
        print(f"Saved: {out_path} ({(end - start) / sr:.2f}s)")

    print(f"✅ Done. Exported {len(speech_timestamps)} segments to '{output_dir}'.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Split audio into speech segments using Silero VAD (torchaudio + torch.hub).")
    parser.add_argument("input_audio", help="Path to input audio file (WAV/MP3/FLAC/etc.)")
    parser.add_argument("output_dir", help="Output directory for speech segments")
    parser.add_argument("--threshold", type=float, default=0.5, help="Speech detection threshold (0–1, lower = more sensitive)")
    parser.add_argument("--min_speech_ms", type=int, default=200, help="Minimum speech duration in milliseconds")
    parser.add_argument("--pad_ms", type=int, default=150, help="Padding before/after detected speech in milliseconds")
    parser.add_argument("--merge_gap_ms", type=int, default=150, help="Merge segments separated by less than this silence gap (ms)")
    args = parser.parse_args()

    split_audio_silero_vad(
        args.input_audio,
        args.output_dir,
        threshold=args.threshold,
        min_speech_ms=args.min_speech_ms,
        pad_ms=args.pad_ms,
        merge_speech_gap_ms=args.merge_gap_ms
    )
