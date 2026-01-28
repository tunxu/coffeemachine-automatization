import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from tensorflow.lite.python.interpreter import Interpreter
from dataset_file import MyAudioDataset

# -----------------------
# CONFIG
# -----------------------
TFLITE_PATH = "kws_tflite/int8/kws_model_int8_MLP.tflite"

SAMPLE_RATE = 16000
N_MELS = 64
TARGET_TIME = 32
N_FFT = 1024
HOP_LENGTH = 512

# If your training used dB/log mel, enable ONE of these:
USE_DB = False
USE_LOG = False

# -----------------------
# Preprocessing helpers
# -----------------------
import torchaudio.transforms as T

_mel_transform = T.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_mels=N_MELS,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
)

_db_transform = T.AmplitudeToDB()  # used only if USE_DB=True


def to_mel_1x64x32(feat: torch.Tensor) -> torch.Tensor:
    """
    Convert dataset output to a mel tensor shaped [1, 64, 32] (no batch dim).
    Handles:
      - waveform [T] or [1, T]
      - mel-like [64, T], [1,64,T], [1,1,64,T]
    """
    x = feat

    # Waveform heuristic: 1D or 2D with long last dim (e.g. 16000)
    is_wave = x.ndim == 1 or (x.ndim == 2 and x.shape[-1] > 512 and x.shape[0] in (1,))
    if is_wave:
        if x.ndim == 1:
            x = x.unsqueeze(0)  # [1, T]
        mel = _mel_transform(x)  # [1, 64, time]

        if USE_DB:
            mel = _db_transform(mel)
        elif USE_LOG:
            mel = torch.log(torch.clamp(mel, min=1e-6))

        # Force time dimension to TARGET_TIME
        t = mel.shape[-1]
        if t > TARGET_TIME:
            mel = mel[..., :TARGET_TIME]
        elif t < TARGET_TIME:
            mel = F.pad(mel, (0, TARGET_TIME - t))

        return mel  # [1, 64, 32]

    # Otherwise assume it's already mel-like and just normalize shape
    if x.ndim == 2:            # [64, T]
        x = x.unsqueeze(0)     # [1, 64, T]
    elif x.ndim == 3:
        # could be [1,64,T] or [C,64,T]
        # If it's [1,64,T], keep. If it's [C,64,T] with C!=1, that's unexpected.
        pass
    elif x.ndim == 4:
        # [1,1,64,T] -> [1,64,T]
        if x.shape[0] == 1 and x.shape[1] == 1:
            x = x[0]
        else:
            raise ValueError(f"Unexpected 4D feature shape: {x.shape}")
    else:
        raise ValueError(f"Unexpected feature shape: {x.shape}")

    # Now x should be [1, 64, T]
    if x.shape[0] != 1 or x.shape[1] != N_MELS:
        raise ValueError(f"Expected mel shape [1, {N_MELS}, T], got {x.shape}")

    t = x.shape[-1]
    if t > TARGET_TIME:
        x = x[..., :TARGET_TIME]
    elif t < TARGET_TIME:
        x = F.pad(x, (0, TARGET_TIME - t))

    return x  # [1, 64, 32]


def mel_to_tflite_nhwc(mel_1x64x32: torch.Tensor) -> np.ndarray:
    """
    mel_1x64x32: torch tensor [1,64,32]
    returns: np float32 [1,64,32,1] (NHWC)
    """
    mel = mel_1x64x32.float().cpu().numpy()          # [1,64,32]
    mel = mel[..., None]                             # [1,64,32,1]
    return mel.astype(np.float32)

def quantize_int8(x_float: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    q = np.round(x_float / scale + zero_point)
    q = np.clip(q, -128, 127)
    return q.astype(np.int8)

def dequantize_int8(x_q: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    return (x_q.astype(np.float32) - zero_point) * scale


# -----------------------
# TFLite eval
# -----------------------
def run_eval_tflite(test_loader, label_names=None, save_cm_path="test_results/confusion_matrix_int8_MLP.png"):
    if not os.path.exists(TFLITE_PATH):
        raise FileNotFoundError(f"TFLite model not found: {TFLITE_PATH} (cwd={os.getcwd()})")

    interp = Interpreter(model_path=TFLITE_PATH)
    interp.allocate_tensors()

    in_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]
    in_idx = in_det["index"]
    out_idx = out_det["index"]

    print("TFLite input:", in_det["shape"], in_det["dtype"], "quant:", in_det.get("quantization"))
    print("TFLite output:", out_det["shape"], out_det["dtype"], "quant:", out_det.get("quantization"))

    # Quant params
    in_scale, in_zero = in_det.get("quantization", (0.0, 0))
    out_scale, out_zero = out_det.get("quantization", (0.0, 0))

    # Expect fixed input [1,64,32,1]
    expected = tuple(in_det["shape"])
    if expected != (1, N_MELS, TARGET_TIME, 1):
        print("WARNING: Model input shape is", expected, "but script assumes", (1, N_MELS, TARGET_TIME, 1))

    all_preds, all_labels = [], []

    for inputs, labels in test_loader:
        for i in range(len(labels)):
            feat = inputs[i]
            lab = int(labels[i])

            mel = to_mel_1x64x32(feat)   # torch [1,64,32]
            x1 = mel_to_tflite_nhwc(mel) # np float32 [1,64,32,1]

            if x1.shape != (1, N_MELS, TARGET_TIME, 1):
                raise ValueError(f"TFLite input wrong shape {x1.shape}, expected {(1, N_MELS, TARGET_TIME, 1)}")

            # ---- IMPORTANT: quantize input if model expects int8/uint8 ----
            if in_det["dtype"] == np.int8:
                if in_scale == 0:
                    raise ValueError("Input is int8 but quantization scale is 0.0 (bad model metadata).")
                x_in = quantize_int8(x1, in_scale, int(in_zero))  # int8
            elif in_det["dtype"] == np.uint8:
                # If you ever end up with uint8 models
                q = np.round(x1 / in_scale + in_zero)
                q = np.clip(q, 0, 255).astype(np.uint8)
                x_in = q
            else:
                x_in = x1.astype(in_det["dtype"], copy=False)

            interp.set_tensor(in_idx, x_in)
            interp.invoke()

            out = interp.get_tensor(out_idx)

            # For classification, argmax works directly on quantized outputs too.
            pred = int(np.argmax(out, axis=-1)[0])

            all_preds.append(pred)
            all_labels.append(lab)

    all_preds = np.array(all_preds, dtype=np.int64)
    all_labels = np.array(all_labels, dtype=np.int64)

    acc = float((all_preds == all_labels).mean())
    print(f"Accuracy: {acc*100:.2f}% ({len(all_labels)} samples)")

    from collections import Counter
    print("Label distribution:", Counter(all_labels.tolist()))
    print("Pred distribution :", Counter(all_preds.tolist()))

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=label_names)
    disp.plot(xticks_rotation=45)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_cm_path), exist_ok=True)
    plt.savefig(save_cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix to: {save_cm_path}")

    return acc, cm


if __name__ == "__main__":
    test_dataset = MyAudioDataset(root_dir="data", sample_rate=SAMPLE_RATE)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

    label_names = getattr(test_dataset, "label_names", None)
    run_eval_tflite(test_loader, label_names=label_names)
