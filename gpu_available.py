import torch

print("✅ CUDA Available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("🔥 GPU Device Count:", torch.cuda.device_count())
    print("💻 GPU Name:", torch.cuda.get_device_name(0))
else:
    print("⚠️ No GPU detected by PyTorch.")
