import torch
from models.transformer import Transformer
from utils.data_loader import load_data
from utils.loss import LabelSmoothingLoss

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32

    # 加载数据
    train_iterator, valid_iterator, test_iterator, SRC, TGT = load_data(batch_size=batch_size, device=device)

    # 加载模型
    model = Transformer(len(SRC.vocab), len(TGT.vocab), N=6, d_model=512, d_ff=2048, h=8, dropout=0.1)
    model.load_state_dict(torch.load("model.pth"))
    model.to(device)
    model.eval()

    # 定义损失函数
    criterion = LabelSmoothingLoss(len(TGT.vocab), padding_idx=TGT.vocab.stoi["<pad>"], smoothing=0.1)

    total_loss = 0
    with torch.no_grad():
        for batch in test_iterator:
            src = batch.src.to(device)
            tgt = batch.trg.to(device)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            src_mask = (src != SRC.vocab.stoi["<pad>"]).unsqueeze(-2)
            tgt_mask = (tgt_input != TGT.vocab.stoi["<pad>"]).unsqueeze(-2)
            tgt_mask = tgt_mask & subsequent_mask(tgt_input.size(-1)).to(device)

            output = model(src, tgt_input, src_mask, tgt_mask)
            loss = criterion(output.transpose(1, 2), tgt_output)
            total_loss += loss.item()

    print(f"Test Loss: {total_loss / len(test_iterator)}")

def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return subsequent_mask == 0

if __name__ == "__main__":
    evaluate()