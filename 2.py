import pynvml
pynvml.nvmlInit()
print("Driver:", pynvml.nvmlSystemGetDriverVersion())