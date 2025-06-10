import torch.nn as nn

class SublayerConnection(nn.Module):
    """
    子层连接：残差连接和层归一化
    """

    def __init__(self, size, dropout):
        """
        初始化子层连接

        参数:
            size (int): 模型的维度
            dropout (float): Dropout 概率
        """
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)  # 层归一化
        self.dropout = nn.Dropout(dropout)  # Dropout 层

    def forward(self, x, sublayer):
        """
        前向传播函数

        参数:
            x (Tensor): 输入序列
            sublayer (callable): 子层函数

        返回:
            Tensor: 添加残差连接和层归一化后的序列
        """
        return x + self.dropout(sublayer(self.norm(x)))  # 残差连接 + Dropout + 层归一化