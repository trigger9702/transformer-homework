import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 设备配置（自动检测GPU/CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 1. 位置编码（绝对位置编码，sin/cos形式）
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # 生成位置编码矩阵 (max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pos_enc = torch.zeros(max_seq_len, d_model)
        pos_enc[:, 0::2] = torch.sin(position * div_term)  # 偶数维用sin
        pos_enc[:, 1::2] = torch.cos(position * div_term)  # 奇数维用cos
        self.register_buffer("pos_enc", pos_enc)  # 不参与梯度更新

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        x = x + self.pos_enc[:x.size(1), :].to(x.device)
        return self.dropout(x)


# 2. Multi-Head Self-Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        assert d_model % nhead == 0, "d_model必须是nhead的整数倍"
        self.d_k = d_model // nhead  # 每个head的维度
        self.nhead = nhead

        # 线性投影层（Q, K, V分开投影）
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32)).to(device)

    def split_heads(self, x):
        # x: (batch_size, seq_len, d_model) → (batch_size, nhead, seq_len, d_k)
        batch_size = x.size(0)
        return x.view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)

    def merge_heads(self, x):
        # x: (batch_size, nhead, seq_len, d_k) → (batch_size, seq_len, d_model)
        batch_size = x.size(0)
        return x.transpose(1, 2).contiguous().view(batch_size, -1, self.nhead * self.d_k)

    def forward(self, q, k, v, mask=None):
        # q/k/v: (batch_size, seq_len_q/seq_len_k/seq_len_v, d_model)
        # mask: (batch_size, seq_len_q, seq_len_k) 或 (batch_size, 1, seq_len_k)（广播）

        # 线性投影 + 分head
        q = self.split_heads(self.w_q(q))  # (batch_size, nhead, seq_len_q, d_k)
        k = self.split_heads(self.w_k(k))  # (batch_size, nhead, seq_len_k, d_k)
        v = self.split_heads(self.w_v(v))  # (batch_size, nhead, seq_len_v, d_k)

        # 计算注意力分数：(Q @ K^T) / sqrt(d_k)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (batch_size, nhead, seq_len_q, seq_len_k)

        # 应用mask（-inf会被softmax归一化为0）
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # 注意力权重 + dropout
        attn_weights = F.softmax(attn_scores, dim=-1)  # (batch_size, nhead, seq_len_q, seq_len_k)
        attn_weights = self.dropout(attn_weights)

        # 加权求和 + 合并head + 输出投影
        output = torch.matmul(attn_weights, v)  # (batch_size, nhead, seq_len_q, d_k)
        output = self.merge_heads(output)  # (batch_size, seq_len_q, d_model)
        output = self.w_o(output)  # (batch_size, seq_len_q, d_model)

        return output, attn_weights


# 3. Position-wise Feed-Forward Network
class PositionWiseFFN(nn.Module):
    def __init__(self, d_model, d_ff=512, dropout=0.1):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        # x: (batch_size, seq_len, d_model) → 输出同shape
        return self.ffn(x)


# 4. 残差连接 + LayerNorm（Pre-LN结构，更稳定）
class ResidualNorm(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # sublayer: 子层（如Attention/FFN），输入x，输出同shape
        return x + self.dropout(sublayer(self.norm(x)))


# 5. Encoder Block
class EncoderBlock(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.ffn = PositionWiseFFN(d_model, d_ff, dropout)
        self.residual_norm1 = ResidualNorm(d_model, dropout)
        self.residual_norm2 = ResidualNorm(d_model, dropout)

    def forward(self, x, src_mask=None):
        # x: (batch_size, seq_len, d_model)
        # 自注意力层（显式传递q=k=v=x，mask=src_mask）
        attn_output, attn_weights = self.self_attn(x, x, x, mask=src_mask)
        x = self.residual_norm1(x, lambda y: attn_output)  # 残差+Norm
        # FFN层
        ffn_output = self.ffn(x)
        x = self.residual_norm2(x, lambda y: ffn_output)
        return x, attn_weights  # 输出格式固定：(output, 注意力权重)


# 6. Decoder Block（加分项：Encoder-Decoder架构支持）
class DecoderBlock(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)  # 掩码自注意力（防止看未来）
        self.cross_attn = MultiHeadAttention(d_model, nhead, dropout)  # Encoder-Decoder注意力
        self.ffn = PositionWiseFFN(d_model, d_ff, dropout)
        self.residual_norm1 = ResidualNorm(d_model, dropout)
        self.residual_norm2 = ResidualNorm(d_model, dropout)
        self.residual_norm3 = ResidualNorm(d_model, dropout)

    def forward(self, x, enc_output, tgt_mask=None, src_tgt_mask=None):
        # x: Decoder输入 (batch_size, tgt_seq_len, d_model)
        # enc_output: Encoder输出 (batch_size, src_seq_len, d_model)

        # 1. 掩码自注意力
        attn1, attn_weights1 = self.self_attn(x, x, x, mask=tgt_mask)
        x = self.residual_norm1(x, lambda y: attn1)

        # 2. Encoder-Decoder注意力（Q来自Decoder，K/V来自Encoder）
        attn2, attn_weights2 = self.cross_attn(x, enc_output, enc_output, mask=src_tgt_mask)
        x = self.residual_norm2(x, lambda y: attn2)

        # 3. FFN层
        ffn_output = self.ffn(x)
        x = self.residual_norm3(x, lambda y: ffn_output)

        return x, (attn_weights1, attn_weights2)


# 7. 完整Transformer模型（支持Encoder-only和Encoder-Decoder）
class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_encoder_layers=2,
                 num_decoder_layers=0, d_ff=512, max_seq_len=512, dropout=0.1, is_encoder_only=True):
        super().__init__()
        self.is_encoder_only = is_encoder_only
        self.d_model = d_model

        # 嵌入层（字符→d_model维度）
        self.embedding = nn.Embedding(vocab_size, d_model)
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        # Encoder（使用统一接口的EncoderBlock）
        self.encoder_layers = nn.ModuleList([
            EncoderBlock(d_model, nhead, d_ff, dropout) for _ in range(num_encoder_layers)
        ])

        # Decoder（可选）
        if not is_encoder_only:
            self.decoder_layers = nn.ModuleList([
                DecoderBlock(d_model, nhead, d_ff, dropout) for _ in range(num_decoder_layers)
            ])

        # 输出层（d_model→vocab_size，预测下一个字符）
        self.fc_out = nn.Linear(d_model, vocab_size)

        # 初始化参数
        self._init_weights()

    def _init_weights(self):
        # 初始化嵌入层和线性层
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)

    def generate_look_ahead_mask(self, seq_len):
        # 生成掩码：(seq_len, seq_len)，下三角为1（允许看过去，禁止看未来）
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask  # (seq_len, seq_len)

    def forward(self, src, tgt=None):
        # src: (batch_size, seq_len) → Encoder输入
        # tgt: (batch_size, tgt_seq_len) → Decoder输入（Encoder-Decoder模式下）

        # 1. 嵌入 + 位置编码
        src_emb = self.embedding(src) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32)).to(device)
        src_emb = self.pos_encoding(src_emb)  # (batch_size, seq_len, d_model)

        # 2. Encoder前向传播（调用接口统一的EncoderBlock）
        enc_output = src_emb
        enc_attn_weights = []
        src_mask = None  # LM任务无需src_mask（无padding）
        for enc_layer in self.encoder_layers:
            enc_output, attn_w = enc_layer(enc_output, src_mask)  # 接口：x, src_mask=None
            enc_attn_weights.append(attn_w)

        # 3. Encoder-only模式（LM任务）：直接用Encoder输出预测
        if self.is_encoder_only:
            output = self.fc_out(enc_output)  # (batch_size, seq_len, vocab_size)
            return output, enc_attn_weights

        # 4. Encoder-Decoder模式（如翻译，需tgt输入）
        if tgt is None:
            raise ValueError("Encoder-Decoder模式需要tgt输入")

        # Decoder嵌入 + 位置编码
        tgt_emb = self.embedding(tgt) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32)).to(device)
        tgt_emb = self.pos_encoding(tgt_emb)  # (batch_size, tgt_seq_len, d_model)

        # 生成Decoder掩码（look-ahead mask）
        tgt_seq_len = tgt.size(1)
        tgt_mask = self.generate_look_ahead_mask(tgt_seq_len).unsqueeze(0)  # (1, tgt_seq_len, tgt_seq_len)

        # Decoder前向传播
        dec_output = tgt_emb
        dec_attn_weights = []
        for dec_layer in self.decoder_layers:
            dec_output, attn_ws = dec_layer(dec_output, enc_output, tgt_mask)
            dec_attn_weights.append(attn_ws)

        output = self.fc_out(dec_output)  # (batch_size, tgt_seq_len, vocab_size)
        return output, (enc_attn_weights, dec_attn_weights)


# 模型参数统计函数
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)