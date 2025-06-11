import torch
import spacy

print("PyTorch version:", torch.__version__)
print("spaCy version:", spacy.__version__)

# 测试 spaCy 分词器
spacy_en = spacy.load("en_core_web_sm")
spacy_de = spacy.load("de_core_news_sm")
print("spaCy tokenizer loaded successfully.")

print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("Device count:", torch.cuda.device_count())