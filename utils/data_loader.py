import torch
from torchtext.datasets import Multi30k
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

# 定义语言和分词器
SRC_LANGUAGE = "de"  # 源语言改为德语(原数据集不支持中文)
TGT_LANGUAGE = "en"  # 目标语言英语

# 获取分词器
tokenizer_src = get_tokenizer("spacy", language="de_core_news_sm")
tokenizer_tgt = get_tokenizer("spacy", language="en_core_web_sm")


# 定义数据预处理函数
def yield_tokens(data_iter, tokenizer, index):
    for data in data_iter:
        yield tokenizer(data[index])


# 加载数据集 - 注意Multi30k返回的是(src, tgt)对
train_iter = Multi30k(split="train", language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
valid_iter = Multi30k(split="valid", language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
test_iter = Multi30k(split="test", language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))

# 构建词汇表
SRC_VOCAB = build_vocab_from_iterator(
    yield_tokens(train_iter, tokenizer_src, 0),
    min_freq=2,
    specials=["<unk>", "<pad>", "<bos>", "<eos>"]
)
TGT_VOCAB = build_vocab_from_iterator(
    yield_tokens(train_iter, tokenizer_tgt, 1),
    min_freq=2,
    specials=["<unk>", "<pad>", "<bos>", "<eos>"]
)

# 设置默认索引
SRC_VOCAB.set_default_index(SRC_VOCAB["<unk>"])
TGT_VOCAB.set_default_index(TGT_VOCAB["<unk>"])


# 定义数据集类
class TranslationDataset(Dataset):
    def __init__(self, data_iter, src_vocab, tgt_vocab, tokenizer_src, tokenizer_tgt):
        self.data = list(data_iter)
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_text, tgt_text = self.data[idx]
        src_tokens = [SRC_VOCAB["<bos>"]] + self.tokenizer_src(src_text) + [SRC_VOCAB["<eos>"]]
        tgt_tokens = [TGT_VOCAB["<bos>"]] + self.tokenizer_tgt(tgt_text) + [TGT_VOCAB["<eos>"]]
        return torch.tensor(src_tokens), torch.tensor(tgt_tokens)


# 定义collate函数
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src, tgt in batch:
        src_batch.append(src)
        tgt_batch.append(tgt)

    src_batch = pad_sequence(src_batch, padding_value=SRC_VOCAB["<pad>"])
    tgt_batch = pad_sequence(tgt_batch, padding_value=TGT_VOCAB["<pad>"])
    return src_batch, tgt_batch


# 创建数据集和数据加载器
train_dataset = TranslationDataset(train_iter, SRC_VOCAB, TGT_VOCAB, tokenizer_src, tokenizer_tgt)
valid_dataset = TranslationDataset(valid_iter, SRC_VOCAB, TGT_VOCAB, tokenizer_src, tokenizer_tgt)
test_dataset = TranslationDataset(test_iter, SRC_VOCAB, TGT_VOCAB, tokenizer_src, tokenizer_tgt)


def load_data(batch_size=32):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return train_loader, valid_loader, test_loader, SRC_VOCAB, TGT_VOCAB


# 测试
if __name__ == "__main__":
    train_loader, valid_loader, test_loader, SRC_VOCAB, TGT_VOCAB = load_data(batch_size=32)
    for src, tgt in train_loader:
        print(f"Source shape: {src.shape}, Target shape: {tgt.shape}")
        break