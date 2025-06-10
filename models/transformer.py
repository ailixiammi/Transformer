import torch.nn as nn
from models.encoder import Encoder
from models.decoder import Decoder
from models.embeddings import Embeddings
from models.generator import Generator
from models.multi_headed_attention import MultiHeadedAttention
from models.positionwise_feed_forward import PositionwiseFeedForward
from models.decoder_layer import DecoderLayer
from models.encoder_layer import EncoderLayer
from models.positional_encoding import PositionalEncoding

class Transformer(nn.Module):
    """
    Transformer 模型：包含编码器和解码器
    """

    def __init__(self, src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
        """
        初始化 Transformer 模型

        参数:
            src_vocab (int): 源语言词汇表大小
            tgt_vocab (int): 目标语言词汇表大小
            N (int, optional): 编码器和解码器的层数。默认为 6
            d_model (int, optional): 模型的维度。默认为 512
            d_ff (int, optional): 前馈网络的维度。默认为 2048
            h (int, optional): 注意力头的数量。默认为 8
            dropout (float, optional): Dropout 概率。默认为 0.1
        """
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            EncoderLayer(
                d_model,
                MultiHeadedAttention(h, d_model),
                PositionwiseFeedForward(d_model, d_ff, dropout),
                dropout
            ),
            N
        )
        self.decoder = Decoder(
            DecoderLayer(
                d_model,
                MultiHeadedAttention(h, d_model),
                MultiHeadedAttention(h, d_model),
                PositionwiseFeedForward(d_model, d_ff, dropout),
                dropout
            ),
            N
        )
        self.src_embed = nn.Sequential(
            Embeddings(d_model, src_vocab),
            PositionalEncoding(d_model, dropout)
        )
        self.tgt_embed = nn.Sequential(
            Embeddings(d_model, tgt_vocab),
            PositionalEncoding(d_model, dropout)
        )
        self.generator = Generator(d_model, tgt_vocab)

    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        前向传播函数。

        参数:
            src (Tensor): 源语言输入序列
            tgt (Tensor): 目标语言输入序列
            src_mask (Tensor): 源语言掩码
            tgt_mask (Tensor): 目标语言掩码

        返回:
            Tensor: Transformer 模型的输出
        """
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        """
        编码器的前向传播函数

        参数:
            src (Tensor): 源语言输入序列
            src_mask (Tensor): 源语言掩码

        返回:
            Tensor: 编码器的输出
        """
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        """
        解码器的前向传播函数

        参数:
            memory (Tensor): 编码器的输出
            src_mask (Tensor): 源语言掩码
            tgt (Tensor): 目标语言输入序列
            tgt_mask (Tensor): 目标语言掩码

        返回:
            Tensor: 解码器的输出
        """
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)