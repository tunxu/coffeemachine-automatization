from torch.utils.data import Dataset
import soundfile as sf
import torch
import torch.nn.functional as F
import os

class CoffeeSpeechDataset(Dataset):
    def __init__(self, root_dir, sample_rate=16000, transform=None):
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.transform = transform

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

        data, sr = sf.read(file_path, dtype="float32")
        waveform = torch.from_numpy(data).unsqueeze(0)  # [1, samples]

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Normalize loudness
        max_val = torch.max(torch.abs(waveform))
        if max_val > 0:
            waveform = waveform / max_val

        # Pad or trim to 3 seconds
        max_len = self.sample_rate
        if waveform.shape[1] < max_len:
            waveform = F.pad(waveform, (0, max_len - waveform.shape[1]))
        else:
            waveform = waveform[:, :max_len]

        if self.transform:
            features = self.transform(waveform)
            features = torch.log(1 + features)  # Apply log for better dynamic range
        else:
            features = waveform

        return features, label