import torch

# Check if CUDA is available
print(f"CUDA available: {torch.cuda.is_available()}")

# If CUDA is available, print more details
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name()}")
    print(f"Device count: {torch.cuda.device_count()}")