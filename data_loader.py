import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import spacy
import torch
from collections import Counter

# 加载分词器
spacy_en = spacy.load("en_core_web_sm")
spacy_de = spacy.load("de_core_news_sm")

# 分词函数
def tokenize_en(text):
    return [token.text.lower() for token in spacy_en(text)]

def tokenize_de(text):
    return [token.text.lower() for token in spacy_de(text)]

# 特殊标记
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
special_tokens = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

# 构建词汇表
def build_vocab(texts, tokenizer, min_freq=3):
    counter = Counter()
    for text in texts:
        counter.update(tokenizer(text))
    vocab = {token: idx for idx, (token, freq) in enumerate(counter.items()) if freq >= min_freq}
    for token in special_tokens:
        vocab[token] = len(vocab)
    return vocab

# 数据集类
class TranslationDataset(Dataset):
    def __init__(self, src_texts, trg_texts, src_vocab, trg_vocab, tokenizer):
        self.src_texts = src_texts
        self.trg_texts = trg_texts
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.src_texts)

    def __getitem__(self, idx):
        src_tokens = [SOS_TOKEN] + self.tokenizer(self.src_texts[idx]) + [EOS_TOKEN]
        trg_tokens = [SOS_TOKEN] + self.tokenizer(self.trg_texts[idx]) + [EOS_TOKEN]
        src_indices = [self.src_vocab.get(token, self.src_vocab[UNK_TOKEN]) for token in src_tokens]
        trg_indices = [self.trg_vocab.get(token, self.trg_vocab[UNK_TOKEN]) for token in trg_tokens]
        return torch.tensor(src_indices, dtype=torch.long), torch.tensor(trg_indices, dtype=torch.long)

# 数据处理函数
def collate_batch(batch, src_vocab, trg_vocab, pad_token_src, pad_token_trg):
    src_list, trg_list = [], []
    for src, trg in batch:
        src_list.append(src)
        trg_list.append(trg)
    src_padded = pad_sequence(src_list, padding_value=pad_token_src)
    trg_padded = pad_sequence(trg_list, padding_value=pad_token_trg)
    return src_padded.transpose(0, 1), trg_padded.transpose(0, 1)

# 加载数据
def load_data(data_dir, batch_size):
    train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    valid_df = pd.read_csv(os.path.join(data_dir, 'valid.csv'))
    test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))

    src_vocab = build_vocab(train_df['src'], tokenize_en, min_freq=3)
    trg_vocab = build_vocab(train_df['trg'], tokenize_de, min_freq=3)

    train_dataset = TranslationDataset(train_df['src'], train_df['trg'], src_vocab, trg_vocab, tokenize_en)
    valid_dataset = TranslationDataset(valid_df['src'], valid_df['trg'], src_vocab, trg_vocab, tokenize_en)
    test_dataset = TranslationDataset(test_df['src'], test_df['trg'], src_vocab, trg_vocab, tokenize_en)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda batch: collate_batch(batch, src_vocab, trg_vocab, src_vocab[PAD_TOKEN], trg_vocab[PAD_TOKEN]))
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda batch: collate_batch(batch, src_vocab, trg_vocab, src_vocab[PAD_TOKEN], trg_vocab[PAD_TOKEN]))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda batch: collate_batch(batch, src_vocab, trg_vocab, src_vocab[PAD_TOKEN], trg_vocab[PAD_TOKEN]))

    return train_loader, valid_loader, test_loader, src_vocab, trg_vocab