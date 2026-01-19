import torch
import onnx

def convert_pth_to_onnx(pth_model_path, onnx_model_path, input_size=()):
    # Load the PyTorch model
    model = torch.load(pth_model_path)
    input_tensor = torch.randn(input_size)
    # Export the model to ONNX format
    torch.onnx.export(model, input_tensor, onnx_model_path, export_params=True)