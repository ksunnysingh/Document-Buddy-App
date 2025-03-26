import torch

print("🔍 Checking if MPS (Apple Metal GPU) is available in PyTorch...")
print("MPS Available:", torch.backends.mps.is_available()) # True = MPS supported
print("MPS Built:", torch.backends.mps.is_built()) # True = your PyTorch build has MPS

