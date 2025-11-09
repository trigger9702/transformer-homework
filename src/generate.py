import torch
import torch.nn.functional as F
import re
import os

# 导入自定义模块
from model import TransformerLM, PositionalEncoding
from train import load_model
from data import load_data

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 优化版文本生成函数（过滤乱码+抑制重复）
def generate_text(
        model, tokenizer, start_text, max_len=200, temperature=0.6,
        top_k=30, top_p=0.8, repetition_penalty=1.4, filter_special_chars=True
):
    model.eval()

    # 预处理起始文本（过滤无效字符）
    if filter_special_chars:
        valid_chars = set(tokenizer.chars)
        start_text = "".join([c for c in start_text if c in valid_chars])
        if not start_text:
            start_text = "To be"

    # 动态扩展位置编码（支持长序列）
    original_pos_enc = model.pos_encoding
    model.pos_encoding = PositionalEncoding(
        d_model=model.d_model, max_seq_len=512, dropout=0.1
    ).to(device)

    # 编码起始文本
    input_ids = torch.tensor(tokenizer.encode(start_text), dtype=torch.long).unsqueeze(0).to(device)
    generated_ids = input_ids.clone()

    # 定义有效字符ID（过滤乱码）
    valid_token_ids = []
    if filter_special_chars:
        for char in tokenizer.chars:
            if char.isalpha() or char in ".,;:!?'()- " or char == "\n":
                valid_token_ids.append(tokenizer.char_to_idx[char])
        valid_token_ids = torch.tensor(valid_token_ids, dtype=torch.long).to(device)

    with torch.no_grad():
        for _ in range(max_len):
            # 限制序列长度
            if input_ids.size(1) > 511:
                input_ids = input_ids[:, -511:]

            # 前向传播
            output, _ = model(input_ids)
            next_token_logits = output[:, -1, :]

            # 过滤无效字符
            if filter_special_chars:
                invalid_mask = torch.ones_like(next_token_logits, dtype=torch.bool)
                invalid_mask[:, valid_token_ids] = False
                next_token_logits[invalid_mask] = -float("inf")

            # 重复惩罚
            for id in generated_ids.squeeze(0):
                if next_token_logits[0, id] > 0:
                    next_token_logits[0, id] /= repetition_penalty * 1.1
                else:
                    next_token_logits[0, id] *= repetition_penalty * 1.1

            # 温度调节
            next_token_logits = next_token_logits / temperature

            # Top-k过滤
            if top_k > 0:
                top_k_values, top_k_indices = torch.topk(next_token_logits, top_k)
                next_token_logits = torch.full_like(next_token_logits, -float("inf"))
                next_token_logits.scatter_(1, top_k_indices, top_k_values)

            # Top-p过滤
            if top_p > 0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                keep_mask = cumulative_probs <= top_p
                keep_mask[0, 0] = True
                sorted_logits = sorted_logits[keep_mask]
                sorted_indices = sorted_indices[keep_mask]

            # 采样
            probs = F.softmax(sorted_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)
            next_token_id = sorted_indices.gather(0, next_token_id).unsqueeze(0)

            # 拼接序列
            input_ids = torch.cat([input_ids, next_token_id], dim=-1)
            generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)

    # 恢复原始位置编码
    model.pos_encoding = original_pos_enc

    # 后处理：去除连续重复字符
    generated_text = tokenizer.decode(generated_ids.squeeze(0).cpu().numpy())
    if filter_special_chars:
        generated_text = re.sub(r"(.)\1{2,}", r"\1\1", generated_text)  # 最多保留2个重复
        generated_text = re.sub(r"\s{2,}", " ", generated_text)

    return generated_text


# 主函数：加载模型并生成文本
def main():
    # 加载数据（获取tokenizer）
    _, _, _, tokenizer = load_data(seq_len=64, batch_size=32)
    vocab_size = tokenizer.vocab_size

    # 初始化模型并加载训练好的权重
    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=128,
        nhead=4,
        num_encoder_layers=2,
        d_ff=512,
        is_encoder_only=True
    ).to(device)

    # 加载训练好的模型（优先加载final模型）
    model_path = "models/transformer_lm_final.pth"
    if os.path.exists(model_path):
        from train import load_model
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
        model, _, _, _, _ = load_model(model, optimizer, scheduler, model_path)
        print(f"已加载训练好的模型：{model_path}")
    else:
        print("未找到训练好的模型，使用随机初始化模型（生成效果较差）")

    # 生成文本的起始句子
    start_texts = [
        "To be or not to be,",
        "King Henry said:",
        "Romeo and Juliet ",
        "Once upon a time,"
    ]

    # 生成文本
    print("===== 文本生成结果 =====")
    generated_examples = []
    for start in start_texts:
        generated = generate_text(
            model, tokenizer, start_text=start,
            max_len=200,
            temperature=0.6,
            top_k=25,
            top_p=0.75,
            repetition_penalty=1.5,
            filter_special_chars=True
        )
        print(f"\n【起始文本】: {start}")
        print(f"【生成文本】: {generated}")
        print("-" * 100)
        generated_examples.append({
            "start_text": start,
            "generated_text": generated
        })

    # 保存生成结果到文件
    os.makedirs("results", exist_ok=True)
    with open("results/generation_examples.json", "w", encoding="utf-8") as f:
        import json
        json.dump(generated_examples, f, indent=4, ensure_ascii=False)
    print(f"\n生成结果已保存至：results/generation_examples.json")


if __name__ == "__main__":
    main()