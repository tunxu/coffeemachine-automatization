from torch.utils.data import Dataset
import torchaudio
import torchaudio.functional as F
import os
import torch

class MyAudioDataset(Dataset):
    def __init__(self, root_dir, sample_rate=16000, transform=None):
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.transform = transform

        # Collect all file paths and labels
        self.file_paths = []
        self.labels = []
        self.label_names = sorted(os.listdir(root_dir))

        for label_idx, label_name in enumerate(self.label_names):
            label_folder = os.path.join(root_dir, label_name)
            for file_name in os.listdir(label_folder):
                if file_name.endswith(".wav"):
                    self.file_paths.append(os.path.join(label_folder, file_name))
                    self.labels.append(label_idx)


    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]

        waveform, sr = torchaudio.load(file_path)

        # Resample if needed
        if sr != self.sample_rate:
            waveform =  F.resample(
            waveform,
            sr,
            16000,
            lowpass_filter_width=64,
            rolloff=0.9475937167399596,
            resampling_method="sinc_interp_kaiser",
            beta=14.769656459379492,
        )

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Normalize loudness
        waveform = waveform / torch.max(torch.abs(waveform))

        # Pad shorter clips, trim longer clips
        max_len = self.sample_rate  # 1 second = 16000 samples
        if waveform.shape[1] < max_len:
            waveform = torch.nn.functional.pad(waveform, (0, max_len - waveform.shape[1]))
        else:
            waveform = waveform[:, :max_len]

        # Apply transforms (e.g. MelSpectrogram)
        if self.transform:
            features = self.transform(waveform)
        else:
            features = waveform
            
        return features, label

