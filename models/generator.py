import torch.nn as nn

class Generator(nn.Module):
    """
    生成器：将解码器的输出转换为最终的预测结果
    """

    def __init__(self, d_model, vocab):
        """
        初始化生成器

        参数:
            d_model (int): 模型的维度
            vocab (int): 目标语言词汇表的大小
        """
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)  # 线性层

    def forward(self, x):
        """
        前向传播函数

        参数:
            x (Tensor): 解码器的输出

        返回:
            Tensor: 生成的预测结果（log-softmax 分布）
        """
        return nn.functional.log_softmax(self.proj(x), dim=-1)  # 应用线性层和 log-softmax