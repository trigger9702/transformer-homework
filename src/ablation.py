import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from tqdm import tqdm
from collections import defaultdict

# 导入自定义模块
from model import TransformerLM, MultiHeadAttention, PositionWiseFFN, EncoderBlock
from data import load_data
from train import train_epoch, val_epoch, WarmupCosineLR

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 消融实验配置：4组对比
ablation_configs = {
    "full_model": {  # 完整模型（基准）
        "d_model": 128, "nhead": 4, "num_encoder_layers": 2,
        "use_pos_encoding": True, "use_multi_head": True, "use_residual": True
    },
    "no_pos_encoding": {  # 无位置编码
        "d_model": 128, "nhead": 4, "num_encoder_layers": 2,
        "use_pos_encoding": False, "use_multi_head": True, "use_residual": True
    },
    "single_head": {  # 单head注意力
        "d_model": 128, "nhead": 1, "num_encoder_layers": 2,
        "use_pos_encoding": True, "use_multi_head": False, "use_residual": True
    },
    "no_residual": {  # 无残差连接（修复接口错误）
        "d_model": 128, "nhead": 4, "num_encoder_layers": 2,
        "use_pos_encoding": True, "use_multi_head": True, "use_residual": False
    }
}


# 自定义无残差EncoderBlock（接口与原EncoderBlock一致）
class NoResidualEncoderBlock(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = PositionWiseFFN(d_model, d_ff, dropout)
        self.dropout2 = nn.Dropout(dropout)

    # 接口统一：(x, src_mask=None)，输出(output, attn_weights)
    def forward(self, x, src_mask=None):
        # 自注意力层：显式传递q=k=v=x
        x_norm = self.norm1(x)
        attn_output, attn_weights = self.self_attn(x_norm, x_norm, x_norm, mask=src_mask)
        # 无残差连接：直接用注意力输出
        x = self.dropout1(attn_output)

        # FFN层：无残差连接
        x_norm2 = self.norm2(x)
        ffn_output = self.ffn(x_norm2)
        x = self.dropout2(ffn_output)

        return x, attn_weights


# 支持消融的Transformer模型
class AblationTransformerLM(TransformerLM):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_encoder_layers=2,
                 use_pos_encoding=True, use_multi_head=True, use_residual=True, **kwargs):
        super().__init__(vocab_size, d_model, nhead, num_encoder_layers, **kwargs)
        self.use_pos_encoding = use_pos_encoding
        self.use_residual = use_residual

        # 无残差时替换EncoderBlock
        if not use_residual:
            self.encoder_layers = nn.ModuleList([
                NoResidualEncoderBlock(d_model, nhead, 512, 0.1)
                for _ in range(num_encoder_layers)
            ])

    def forward(self, src, tgt=None):
        # 嵌入层
        src_emb = self.embedding(src) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32)).to(device)

        # 可选关闭位置编码
        if self.use_pos_encoding:
            src_emb = self.pos_encoding(src_emb)

        # Encoder前向传播
        enc_output = src_emb
        enc_attn_weights = []
        src_mask = None
        for enc_layer in self.encoder_layers:
            enc_output, attn_w = enc_layer(enc_output, src_mask)
            enc_attn_weights.append(attn_w)

        # 输出层
        output = self.fc_out(enc_output)
        return output, enc_attn_weights


# 消融实验结果可视化
def plot_ablation_results(ablation_results, save_path="results/ablation_study.png"):
    os.makedirs("results", exist_ok=True)
    plt.figure(figsize=(14, 6))
    colors = ["#2E8B57", "#FF6347", "#4169E1", "#FF4500"]

    # 子图1：验证损失曲线
    plt.subplot(1, 2, 1)
    for idx, (name, res) in enumerate(ablation_results.items()):
        epochs = range(1, len(res["val_losses"]) + 1)
        plt.plot(epochs, res["val_losses"], label=name, color=colors[idx], linewidth=2)
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Validation Loss", fontsize=11)
    plt.title("Ablation Study: Validation Loss Curves", fontsize=12, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    # 子图2：最终困惑度柱状图
    plt.subplot(1, 2, 2)
    names = list(ablation_results.keys())
    perps = [ablation_results[name]["final_val_perp"] for name in names]
    bars = plt.bar(names, perps, color=colors, alpha=0.8)
    plt.xlabel("Model Variant", fontsize=11)
    plt.ylabel("Final Validation Perplexity", fontsize=11)
    plt.title("Ablation Study: Final Perplexity", fontsize=12, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.grid(True, alpha=0.3, axis="y")

    # 标注数值
    for bar, perp in zip(bars, perps):
        plt.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.5,
                 f"{perp:.1f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"消融实验图表已保存至：{save_path}")


# 打印消融实验结果表格
def print_ablation_table(ablation_results):
    print("\n==================================================")
    print("消融实验结果汇总表")
    print("==================================================")
    print(f"{'模型变体':<20} {'最终验证损失':<15} {'最终验证困惑度':<15}")
    print("-" * 50)
    for name, res in ablation_results.items():
        print(f"{name:<20} {res['final_val_loss']:.4f} {'':<5} {res['final_val_perp']:.2f}")
    print("==================================================")


# 主函数：运行消融实验
def main():
    # 实验配置
    seed = 42
    epochs_ablation = 20
    batch_size = 32
    seq_len = 64
    lr = 1e-3

    # 设置随机种子（可复现）
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    print(f"消融实验随机种子：{seed}")

    # 加载数据（与主训练一致）
    train_loader, val_loader, _, tokenizer = load_data(seq_len=seq_len, batch_size=batch_size)
    vocab_size = tokenizer.vocab_size

    # 存储实验结果
    ablation_results = defaultdict(dict)

    # 运行4组消融实验
    print(f"\n开始消融实验（共4组模型，每组{epochs_ablation}轮）...")
    for name, config in ablation_configs.items():
        print(f"\n==================================================")
        print(f"消融实验：{name}")
        print(f"配置：{config}")
        print(f"==================================================")

        # 初始化模型
        model_ablation = AblationTransformerLM(
            vocab_size=vocab_size,
            d_model=config["d_model"],
            nhead=config["nhead"],
            num_encoder_layers=config["num_encoder_layers"],
            use_pos_encoding=config["use_pos_encoding"],
            use_multi_head=config["use_multi_head"],
            use_residual=config["use_residual"],
            is_encoder_only=True,
            max_seq_len=seq_len,
            dropout=0.1
        ).to(device)

        # 优化器和调度器
        optimizer_ablation = optim.AdamW(
            model_ablation.parameters(),
            lr=lr,
            weight_decay=1e-4,
            betas=(0.9, 0.98)
        )
        total_steps_ablation = epochs_ablation * len(train_loader)
        scheduler_ablation = WarmupCosineLR(
            optimizer_ablation,
            warmup_steps=100,
            total_steps=total_steps_ablation,
            eta_min=1e-5
        )

        # 损失函数
        criterion = nn.CrossEntropyLoss()

        # 训练记录
        train_losses = []
        val_losses = []

        # 训练循环
        for epoch in range(epochs_ablation):
            print(f"\nEpoch {epoch + 1}/{epochs_ablation}")
            train_loss = train_epoch(model_ablation, train_loader, optimizer_ablation, scheduler_ablation, criterion)
            val_loss = val_epoch(model_ablation, val_loader, criterion)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            # 打印日志
            train_perp = np.exp(train_loss)
            val_perp = np.exp(val_loss)
            print(f"Train Loss: {train_loss:.4f} | Train Perp: {train_perp:.2f}")
            print(f"Val Loss: {val_loss:.4f} | Val Perp: {val_perp:.2f}")

        # 保存结果
        ablation_results[name] = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "final_val_loss": val_losses[-1],
            "final_val_perp": np.exp(val_losses[-1]),
            "config": config
        }

    # 可视化结果
    plot_ablation_results(ablation_results)

    # 打印表格
    print_ablation_table(ablation_results)

    # 保存结果到文件
    with open("results/ablation_results.json", "w", encoding="utf-8") as f:
        json.dump(ablation_results, f, indent=4, ensure_ascii=False)
    print(f"\n消融实验结果已保存至：results/ablation_results.json")


if __name__ == "__main__":
    main()