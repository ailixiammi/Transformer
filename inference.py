import torch
import spacy
from model.transformer import Transformer
from config import config

# 加载分词器
spacy_en = spacy.load('en_core_web_sm')

# 加载词汇表
src_vocab = torch.load("src_vocab.pth")
trg_vocab = torch.load("trg_vocab.pth")

# 加载模型
model = Transformer(len(src_vocab), len(trg_vocab), config).to(config.device)
model.load_state_dict(torch.load(config.model_path))
model.eval()

# 翻译函数
def translate(sentence, max_len=50):
    tokens = [token.text.lower() for token in spacy_en(sentence)]
    tokens = ['<sos>'] + tokens + ['<eos>']
    src_indexes = [src_vocab.get(token, src_vocab['<unk>']) for token in tokens]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(config.device)

    trg_indexes = [trg_vocab['<sos>']]

    for i in range(max_len):
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(config.device)

        with torch.no_grad():
            output = model(src_tensor, trg_tensor)
            pred_token = output.argmax(2)[:, -1].item()

        trg_indexes.append(pred_token)
        if pred_token == trg_vocab['<eos>']:
            break

    trg_tokens = [trg_vocab.get_itos()[i] for i in trg_indexes]
    return trg_tokens[1:]

# 测试
while True:
    sentence = input("Enter English sentence (type 'exit' to quit): ")
    if sentence.lower() == 'exit':
        break
    translation = translate(sentence)
    print("German translation:", ' '.join(translation))