import torch

try:
    if torch.cuda.is_available():
        print(f"CUDA is available. GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available.")
except Exception as e:
    print(f"An error occurred: {e}")