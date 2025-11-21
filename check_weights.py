import torch
from safetensors.torch import load_file
import os

adapter_path = "models/phi3-finetuned/adapter_model.safetensors"

if os.path.exists(adapter_path):
    print(f"Checking {adapter_path}...")
    tensors = load_file(adapter_path)
    has_nan = False
    for name, tensor in tensors.items():
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            print(f"Found NaN/Inf in {name}")
            has_nan = True
    
    if not has_nan:
        print("No NaNs or Infs found in adapter weights.")
else:
    print("Adapter file not found.")
