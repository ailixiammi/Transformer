import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR


def get_optimizer(model, lr=0.0001):
    """
    获取优化器

    参数:
        model (nn.Module): 模型
        lr (float, optional): 学习率。默认为 0.0001

    返回:
        torch.optim.Optimizer: 优化器
    """
    return optim.Adam(model.parameters(), lr=lr)


def get_scheduler(optimizer, warmup_steps=4000):
    """
    获取学习率调度器

    参数:
        optimizer (torch.optim.Optimizer): 优化器
        warmup_steps (int, optional): 预热步数。默认为 4000

    返回:
        torch.optim.lr_scheduler.LambdaLR: 学习率调度器
    """
    return LambdaLR(optimizer, lambda step: min(step ** -0.5, step * warmup_steps ** -1.5))