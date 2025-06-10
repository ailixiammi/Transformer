import torch
import torchvision
import torchaudio
import numpy
import pandas
import spacy
import altair

print("PyTorch version:", torch.__version__)
print("TorchVision version:", torchvision.__version__)
print("TorchAudio version:", torchaudio.__version__)
print("NumPy version:", numpy.__version__)
print("Pandas version:", pandas.__version__)
print("Spacy version:", spacy.__version__)
print("Altair version:", altair.__version__)

print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("Device count:", torch.cuda.device_count())
print("Current device:", torch.cuda.current_device())
print("Device name:", torch.cuda.get_device_name(0))