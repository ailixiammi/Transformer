import torch.nn as nn
from model.encoder import EncoderLayer
from model.decoder import DecoderLayer
from model.embeddings import PositionalEncoding


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, config):
        super().__init__()
        self.d_model = config.d_model

        # 嵌入层
        self.src_embed = nn.Embedding(src_vocab_size, config.d_model)
        self.trg_embed = nn.Embedding(trg_vocab_size, config.d_model)

        # 位置编码
        self.pos_encoder = PositionalEncoding(config.d_model, config.dropout)

        # 编码器
        self.encoder = nn.ModuleList([
            EncoderLayer(
                config.d_model,
                config.n_heads,
                config.dim_feedforward,
                config.dropout
            ) for _ in range(config.num_encoder_layers)
        ])

        # 解码器
        self.decoder = nn.ModuleList([
            DecoderLayer(
                config.d_model,
                config.n_heads,
                config.dim_feedforward,
                config.dropout
            ) for _ in range(config.num_decoder_layers)
        ])

        # 输出层
        self.fc_out = nn.Linear(config.d_model, trg_vocab_size)

        # 层归一化
        self.encoder_norm = nn.LayerNorm(config.d_model)
        self.decoder_norm = nn.LayerNorm(config.d_model)

    def forward(self, src, trg, src_mask=None, trg_mask=None):
        # 源序列嵌入 + 位置编码
        src_emb = self.src_embed(src) * (self.d_model ** 0.5)
        src_emb = self.pos_encoder(src_emb)

        # 目标序列嵌入 + 位置编码
        trg_emb = self.trg_embed(trg) * (self.d_model ** 0.5)
        trg_emb = self.pos_encoder(trg_emb)

        # 编码器处理
        memory = src_emb
        for layer in self.encoder:
            memory = layer(memory, src_mask)
        memory = self.encoder_norm(memory)

        # 解码器处理
        output = trg_emb
        for layer in self.decoder:
            output = layer(output, memory, src_mask, trg_mask)
        output = self.decoder_norm(output)

        # 输出层
        logits = self.fc_out(output)
        return logits