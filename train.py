import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from model.transformer import Transformer
from data_loader import load_data, PAD_TOKEN
from config import config
import time
import matplotlib.pyplot as plt

# 加载数据
train_loader, valid_loader, src_vocab, trg_vocab = load_data(config.data_dir, config.batch_size)

# 初始化模型
model = Transformer(len(src_vocab), len(trg_vocab), config).to(config.device)

# 损失函数（忽略填充部分）
criterion = nn.CrossEntropyLoss(ignore_index=trg_vocab[PAD_TOKEN])

# 优化器
optimizer = optim.Adam(model.parameters(), lr=config.lr, betas=config.betas, eps=config.eps, weight_decay=config.weight_decay)

# 学习率调度器（带预热）
scheduler = LambdaLR(optimizer, lr_lambda=lambda step: min((step + 1) ** -0.5, (step + 1) * (config.warmup_steps ** -1.5)))

# 训练函数
def train_epoch(model, loader, optimizer, scheduler, criterion):
    model.train()
    total_loss = 0
    for i, (src, trg) in enumerate(loader):
        optimizer.zero_grad()

        # 目标输入和输出（用于teacher forcing）
        output = model(src, trg[:, :-1])
        output_dim = output.shape[-1]

        # 计算损失
        loss = criterion(output.reshape(-1, output_dim), trg[:, 1:].reshape(-1))
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_norm)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        if i % 100 == 0:
            print(f"Batch {i}, Loss: {loss.item()}")

    return total_loss / len(loader)

# 验证函数
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for src, trg in loader:
            output = model(src, trg[:, :-1])
            output_dim = output.shape[-1]
            loss = criterion(output.reshape(-1, output_dim), trg[:, 1:].reshape(-1))
            total_loss += loss.item()
    return total_loss / len(loader)

# 训练循环
train_losses = []
valid_losses = []

start_time = time.time()
for epoch in range(config.epochs):
    train_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion)
    valid_loss = evaluate(model, valid_loader, criterion)

    train_losses.append(train_loss)
    valid_losses.append(valid_loss)

    print(f"Epoch {epoch + 1}/{config.epochs}")
    print(f"Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")

    # 保存模型
    if valid_loss == min(valid_losses):
        torch.save(model.state_dict(), config.model_path)
        print("Model saved!")

print(f"Training completed in {time.time() - start_time:.2f} seconds")

# 绘制损失曲线
plt.plot(train_losses, label='Train')
plt.plot(valid_losses, label='Validation')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('loss_curve.png')
plt.show()