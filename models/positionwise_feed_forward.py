import torch
import torch.nn as nn

class PositionwiseFeedForward(nn.Module):
    """
    逐位置前馈网络
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        初始化逐位置前馈网络

        参数:
            d_model (int): 模型的维度
            d_ff (int): 前馈网络的维度
            dropout (float, optional): Dropout 概率。默认为 0.1
        """
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)  # 第一层全连接
        self.w_2 = nn.Linear(d_ff, d_model)  # 第二层全连接
        self.dropout = nn.Dropout(dropout)  # Dropout 层

    def forward(self, x):
        """
        前向传播函数

        参数:
            x (Tensor): 输入序列

        返回:
            Tensor: 前馈网络的输出
        """
        return self.w_2(self.dropout(torch.relu(self.w_1(x))))  # ReLU 激活 + Dropout + 第二层全连接