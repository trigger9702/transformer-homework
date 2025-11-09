import torch
import numpy as np
import os
import json

def count_params(model):
    """统计模型可训练参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def set_seed(seed=42):
    """设置随机种子，保证实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"随机种子已设置为：{seed}")

def create_dirs(dirs_list):
    """创建多个目录（如果不存在）"""
    for dir in dirs_list:
        if not os.path.exists(dir):
            os.makedirs(dir)
            print(f"创建目录：{dir}")

def save_json(data, save_path):
    """保存数据到JSON文件"""
    create_dirs([os.path.dirname(save_path)])
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"数据已保存至：{save_path}")

def load_json(load_path):
    """从JSON文件加载数据"""
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"文件不存在：{load_path}")
    with open(load_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"数据已加载自：{load_path}")
    return data

def print_model_summary(model):
    """打印模型结构和参数统计"""
    print("="*50)
    print("模型结构摘要")
    print("="*50)
    print(model)
    print(f"\n可训练参数总数：{count_params(model):,}")
    print("="*50)