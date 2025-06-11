import torch


class Config:
    # 数据参数
    data_dir = "./data"
    batch_size = 128
    max_seq_len = 100
    min_freq = 3  # 词汇表最小词频

    # 模型参数
    src_lang = "en"
    tgt_lang = "de"
    d_model = 512
    n_heads = 8
    num_encoder_layers = 6
    num_decoder_layers = 6
    dim_feedforward = 2048
    dropout = 0.1

    # 训练参数
    epochs = 30
    lr = 0.0001
    betas = (0.9, 0.98)
    eps = 1e-9
    weight_decay = 0.0001
    clip_norm = 1.0
    warmup_steps = 4000

    # 设备
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 保存路径
    model_path = "transformer_model.pth"


config = Config()