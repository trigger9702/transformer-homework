import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from collections import defaultdict

# 导入自定义模块
from model import TransformerLM, count_params
from data import load_data

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 1. 学习率调度器（Linear Warmup + Cosine Annealing）
class WarmupCosineLR(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, eta_min=0, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Warmup阶段：线性增长
            lr = self.base_lrs[0] * (self.last_epoch + 1) / self.warmup_steps
        else:
            # Cosine阶段：余弦衰减
            progress = (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.eta_min + (self.base_lrs[0] - self.eta_min) * (1 + np.cos(np.pi * progress)) / 2
        return [lr] * len(self.optimizer.param_groups)


# 2. 训练曲线可视化
def plot_training_curves(results, save_path="results/training_curves.png"):
    os.makedirs("results", exist_ok=True)
    epochs = range(1, len(results["train_loss"]) + 1)

    plt.figure(figsize=(12, 4))
    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, results["train_loss"], label="Train Loss")
    plt.plot(epochs, results["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True)

    # 困惑度曲线（Perplexity = exp(loss)）
    plt.subplot(1, 2, 2)
    train_perp = [np.exp(loss) for loss in results["train_loss"]]
    val_perp = [np.exp(loss) for loss in results["val_loss"]]
    plt.plot(epochs, train_perp, label="Train Perplexity")
    plt.plot(epochs, val_perp, label="Val Perplexity")
    plt.xlabel("Epoch")
    plt.ylabel("Perplexity")
    plt.title("Training & Validation Perplexity")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"训练曲线已保存至：{save_path}")


# 3. 模型保存与加载
def save_model(model, optimizer, scheduler, results, epoch, save_path="models/transformer_lm.pth"):
    os.makedirs("models", exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "results": results,
    }, save_path)
    print(f"模型保存至：{save_path}")


def load_model(model, optimizer, scheduler, load_path="models/transformer_lm.pth"):
    checkpoint = torch.load(load_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    results = checkpoint["results"]
    start_epoch = checkpoint["epoch"] + 1
    print(f"模型加载自：{load_path}，从第{start_epoch}轮继续训练")
    return model, optimizer, scheduler, results, start_epoch


# 4. 单轮训练函数
def train_epoch(model, loader, optimizer, scheduler, criterion):
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc="Training")
    for batch_idx, (src, tgt) in enumerate(pbar):
        src, tgt = src.to(device), tgt.to(device)

        # 前向传播
        output, _ = model(src)  # (batch_size, seq_len, vocab_size)

        # 计算损失（展平为(batch_size*seq_len, vocab_size) vs (batch_size*seq_len,)）
        loss = criterion(output.reshape(-1, output.size(-1)), tgt.reshape(-1))

        # 反向传播 + 梯度裁剪
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
        optimizer.step()
        scheduler.step()  # 学习率更新

        # 累计损失
        total_loss += loss.item()
        pbar.set_postfix({"Loss": loss.item()})

    avg_loss = total_loss / len(loader)
    return avg_loss


# 5. 单轮验证函数
def val_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        pbar = tqdm(loader, desc="Validation")
        for batch_idx, (src, tgt) in enumerate(pbar):
            src, tgt = src.to(device), tgt.to(device)

            output, _ = model(src)
            loss = criterion(output.reshape(-1, output.size(-1)), tgt.reshape(-1))
            total_loss += loss.item()
            pbar.set_postfix({"Loss": loss.item()})

    avg_loss = total_loss / len(loader)
    return avg_loss


# 6. 主训练函数
def main():
    # 训练配置（可通过命令行参数修改，这里固定为作业要求的可复现参数）
    seed = 42
    epochs = 30
    batch_size = 32
    seq_len = 64
    lr = 1e-3
    warmup_steps = 100

    # 设置随机种子（可复现）
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"随机种子：{seed}")

    # 加载数据
    train_loader, val_loader, test_loader, tokenizer = load_data(seq_len=seq_len, batch_size=batch_size)
    vocab_size = tokenizer.vocab_size

    # 初始化模型
    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=128,
        nhead=4,
        num_encoder_layers=2,
        d_ff=512,
        max_seq_len=seq_len,
        is_encoder_only=True
    ).to(device)
    print(f"模型参数量：{count_params(model):,}")

    # 损失函数、优化器、调度器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4,
        betas=(0.9, 0.98)
    )
    total_steps = epochs * len(train_loader)
    scheduler = WarmupCosineLR(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        eta_min=1e-5
    )

    # 记录训练结果
    results = {
        "train_loss": [],
        "val_loss": []
    }

    # 开始训练
    print(f"\n开始训练（共{epochs}轮，设备：{device}）...")
    for epoch in range(epochs):
        print(f"\n===== Epoch {epoch + 1}/{epochs} =====")

        train_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion)
        val_loss = val_epoch(model, val_loader, criterion)

        # 记录结果
        results["train_loss"].append(train_loss)
        results["val_loss"].append(val_loss)

        # 打印日志
        train_perp = np.exp(train_loss)
        val_perp = np.exp(val_loss)
        print(f"Train Loss: {train_loss:.4f} | Train Perp: {train_perp:.2f}")
        print(f"Val Loss: {val_loss:.4f} | Val Perp: {val_perp:.2f}")

        # 每5轮保存模型
        if (epoch + 1) % 5 == 0:
            save_model(model, optimizer, scheduler, results, epoch, f"models/transformer_lm_epoch_{epoch + 1}.pth")

    # 最终保存
    save_model(model, optimizer, scheduler, results, epochs - 1, "models/transformer_lm_final.pth")

    # 可视化训练曲线
    plot_training_curves(results)

    # 测试集评估
    print(f"\n===== 测试集评估 =====")
    test_loss = val_epoch(model, test_loader, criterion)
    test_perp = np.exp(test_loss)
    print(f"Test Loss: {test_loss:.4f} | Test Perp: {test_perp:.2f}")

    # 保存测试结果
    results["test_loss"] = test_loss
    results["test_perp"] = test_perp
    with open("results/training_results.json", "w", encoding="utf-8") as f:
        import json
        json.dump(results, f, indent=4)
    print(f"训练结果已保存至：results/training_results.json")


if __name__ == "__main__":
    main()