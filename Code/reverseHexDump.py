import re

with open("model.cc", "r", encoding="utf-8") as f:
    text = f.read()

hex_bytes = re.findall(r'0x([0-9a-fA-F]{2})', text)
data = bytes(int(b, 16) for b in hex_bytes)

with open("model_DSCNNv1.tflite", "wb") as f:
    f.write(data)

print("Wrote", len(data), "bytes")