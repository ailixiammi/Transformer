import math
import torch.nn as nn
from models.positional_encoding import PositionalEncoding

class Embeddings(nn.Module):
    """
    嵌入层：将输入的词汇索引转换为向量表示
    """
    def __init__(self, d_model, vocab, dropout=0.1, max_len=50000):
        """
        d_model (int): 模型的维度
        vocab (int): 词汇表的大小
        dropout (float): Dropout 概率
        max_len (int): 最大序列长度
        """
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model) # 创建嵌入层
        self.d_model = d_model  # 模型维度
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_len)  # 创建位置编码层

    def forward(self, x):
        """
        前向传播函数
        参数: x (Tensor): 输入的词汇索引
        返回: Tensor: 嵌入后的向量表示
        """
        embedded = self.lut(x) * math.sqrt(self.d_model)  # 缩放嵌入向量
        return self.positional_encoding(embedded)