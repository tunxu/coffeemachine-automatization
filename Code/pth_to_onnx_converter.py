import os
import torch
import torchaudio
import torch.nn as nn
import torchaudio.functional as F
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from model_file import KeywordCNN, KeywordDSCNN, KeywordMLP
import tensorflow as tf
import numpy as np
import subprocess


class KeywordCNNv2(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        def conv_block(in_ch, out_ch, kernel=3, stride=1, pool=True):
            layers = [
                nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=kernel//2, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ]
            if pool:
                layers.append(nn.MaxPool2d(2, 2))
            return nn.Sequential(*layers)

        # Feature extractor
        self.layer1 = conv_block(1, 32)
        self.layer2 = conv_block(32, 64)
        self.layer3 = conv_block(64, 128)

        # Depthwise separable conv (for temporal modeling)
        self.dw_conv = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3,1), padding=(1,0), groups=128),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # Squeeze-and-Excitation (light attention)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(128, 32, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 128, 1),
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # Depthwise temporal conv
        x = self.dw_conv(x)

        # Squeeze-and-Excitation
        se = self.se(x)
        x = x * se

        # Global average pooling
        x = x.mean(dim=[2, 3])
        x = self.dropout(x)
        x = self.fc(x)
        return x
    
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
    
transform = T.MelSpectrogram(
    sample_rate=16000,
    n_mels=64,
    n_fft=1024,
    hop_length=512
)


full_dataset = MyAudioDataset("data", sample_rate=16000, transform=transform)

def representative_dataset_gen(dataset, num_samples=200, target_time=32):
    """
    Yields [input] where input is NHWC float32 with shape (1, 64, 32, 1).
    This is used ONLY for calibration; TFLite will quantize internally.
    """
    for i in range(min(num_samples, len(dataset))):
        features, _ = dataset[i]  # torch.Tensor, e.g. [1, 64, T]
        x = features.detach().cpu().numpy().astype(np.float32)

        # Ensure shape [1, 64, T]
        if x.ndim == 2:           # [64, T]
            x = x[None, ...]      # [1, 64, T]
        if x.ndim != 3 or x.shape[0] != 1 or x.shape[1] != 64:
            raise ValueError(f"Unexpected feature shape from dataset: {x.shape}")

        # Pad/crop time to 32
        T = x.shape[2]
        if T > target_time:
            x = x[:, :, :target_time]
        elif T < target_time:
            pad = target_time - T
            x = np.pad(x, ((0, 0), (0, 0), (0, pad)), mode="constant")

        # NCHW-ish [1,64,32] -> NHWC [1,64,32,1]
        x = np.transpose(x, (1, 2, 0))  # [64,32,1]
        x = x[None, ...]                # [1,64,32,1]

        yield [x]

def convert_pth_to_onnx(pth_model_path, onnx_model_path):
    model = KeywordMLP(num_classes=5)
    state_dict = torch.load(pth_model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    os.makedirs(os.path.dirname(onnx_model_path), exist_ok=True)

    dummy_input = torch.randn(1, 1, 64, 32)  # batch=1 is fine

    torch.onnx.export(
        model,
        dummy_input,
        onnx_model_path,
        opset_version=18,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )

    print(f"Saved ONNX model to: {onnx_model_path}")

def convert_onnx_to_tflite(onnx_model_path, tflite_model_path):
    os.makedirs(os.path.dirname(tflite_model_path), exist_ok=True)

    out_dir = os.path.splitext(tflite_model_path)[0] + "_tf"
    os.makedirs(out_dir, exist_ok=True)

    # 1) ONNX -> TensorFlow SavedModel
    subprocess.check_call([
        "onnx2tf",
        "-i", onnx_model_path,
        "-o", out_dir,
    ])

    saved_model_dir = out_dir  # onnx2tf writes SavedModel directly to out_dir

    # 2) SavedModel -> FULL INT8 TFLite
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset_gen(full_dataset, num_samples=200, target_time=32)

    # Force integer-only kernels (good for MCUs) and int8 I/O
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    with open(tflite_model_path, "wb") as f:
        f.write(tflite_model)

    print(f"Saved INT8 TFLite model to: {tflite_model_path}")
if __name__ == "__main__":
    print(os.getcwd())
    convert_pth_to_onnx(
        pth_model_path="checkpoints/MLP/kws_model.pt",
        onnx_model_path="kws_onnx/kws_model_MLP.onnx",
    )
    convert_onnx_to_tflite(
        onnx_model_path="kws_onnx/kws_model_MLP.onnx",
        tflite_model_path="kws_tflite/int8/kws_model_int8_MLP.tflite",
    )
