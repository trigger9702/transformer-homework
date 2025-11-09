# 手工实现Transformer用于小规模文本建模任务

本项目手工实现Transformer核心组件，构建Encoder-only架构语言模型，在Tiny Shakespeare数据集上完成字符级语言建模与消融实验。

## 项目简介
- 核心目标：实现Transformer关键组件（多头自注意力、位置编码、残差连接等），验证模型收敛性与组件有效性
- 数据集：Tiny Shakespeare（~1MB字符级文本，词汇表大小65）
- 模型架构：2层Encoder，特征维度128，4头注意力
- 实验内容：基础模型训练（20轮）+ 4组消融实验（每组10轮）
- 核心结果：基础模型验证困惑度1.03，位置编码对性能影响最显著

## 环境配置
### 依赖包清单
torch>=1.18.0          # 核心深度学习框架
numpy>=1.26.0          # 数值计算
matplotlib>=3.8.0      # 实验图表绘制
tqdm>=4.66.0           # 进度条显示
requests>=2.31.0       # 数据集下载

### 快速安装
pip install torch numpy matplotlib tqdm requests

## 快速开始
### 克隆项目（可选）
git clone https://github.com/your-username/transformer-text-modeling.git
cd transformer-text-modeling

### 数据集下载
运行以下代码自动下载 Tiny Shakespeare 数据集到 data/ 目录：
import requests
import os

# 创建数据目录
os.makedirs("data", exist_ok=True)
# 下载数据集
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
response = requests.get(url)
with open("data/input.txt", "w", encoding="utf-8") as f:
    f.write(response.text)
print("数据集下载完成！")

### 模型训练
#### 基础模型训练（20 轮）
python train.py \
    --data_path ./data/input.txt \
    --epochs 20 \
    --batch_size 32 \
    --seq_len 64 \
    --d_model 128 \
    --nhead 4 \
    --num_layers 2 \
    --d_ff 512 \
    --dropout 0.1 \
    --lr 1e-3 \
    --save_path ./checkpoints/transformer_lm_final.pth \
    --log_path ./logs/training.log

#### 消融实验训练（4 组）
# 1. 无位置编码
python train.py --data_path ./data/input.txt --epochs 10 --no_pos_encoding --save_path ./checkpoints/no_pos.pth

# 2. 单头注意力
python train.py --data_path ./data/input.txt --epochs 10 --single_head --save_path ./checkpoints/single_head.pth

# 3. 无残差连接
python train.py --data_path ./data/input.txt --epochs 10 --no_residual --save_path ./checkpoints/no_residual.pth

# 4. 基准模型（10轮，用于对比）
python train.py --data_path ./data/input.txt --epochs 10 --save_path ./checkpoints/baseline_10ep.pth

### 文本生成
python generate.py \
    --model_path ./checkpoints/transformer_lm_final.pth \
    --start_text "To be or not to be," \
    --generate_len 200 \
    --temperature 0.7 \
    --save_result ./results/generated_text.txt

### 实验结果可视化
python plot_results.py \
    --log_path ./logs/training.log \
    --ablation_logs ./logs/ablation_logs/ \
    --save_dir ./results/

## 实验说明
### 数据集详情
- 规模：~1MB 文本，约 40 万字符
- 分割比例：训练集 90% / 验证集 5% / 测试集 5%
- 编码方式：字符级编码，词汇表大小 = 65（包含大小写字母、标点、空格等）

### 模型参数配置
参数                取值
特征维度（d_model）  128
注意力头数（nhead）  4
Encoder 层数        2
FFN 隐藏层维度（d_ff） 512
Dropout 率          0.1
序列长度            64
批大小              32
可训练参数量        ~1.2M

### 消融实验设计
模型变体           核心修改               训练轮数
full_model（基准） 完整模型               10
no_pos_encoding    去除位置编码           10
single_head        单头注意力（替换多头） 10
no_residual        去除残差连接           10

## 文件结构
transformer-homework/  # 仓库根目录
├── src/  
│   ├── model.py        # Transformer核心组件实现
│   ├── data.py         # 数据加载与预处理
│   ├── train.py        # 模型训练主脚本
│   ├── ablation.py     # 消融实验辅助脚本
│   ├── generate.py     # 文本生成脚本
│   └── utils.py        # 工具函数（日志、指标计算等）
├── results/  
│   ├── training_curves_final.png  # 训练曲线图表
│   ├── ablation_study_fixed.png   # 消融实验对比图
│   ├── ablation_results_fixed.json # 消融实验数值结果
│   └── generation_examples.txt    # 文本生成示例
├── datasets/  
│   └── tiny_shakespeare.zip       # 数据集压缩包（备用）
├── requirements.txt                # 依赖包清单
├── README.md                       # 项目说明文档
└── scripts/
    └── run.sh                      # 一键运行脚本（可选）

## 核心结果
### 基础模型性能
指标         训练集  验证集
Loss         0.0287  0.0275
Perplexity（困惑度） 1.03  1.03

### 消融实验性能对比
模型变体           验证 Loss  验证 Perplexity
full_model（基准） 0.0272     1.03
no_pos_encoding    1.9885     7.30
single_head        0.0295     1.03
no_residual        0.0312     1.03

## 注意事项
1. 训练前确保数据集路径正确，若自动下载失败可手动下载数据集放入 data/input.txt
2. 模型训练默认使用 GPU（CUDA），无 GPU 时自动切换 CPU（训练速度较慢）
3. 随机种子固定为 42，确保实验可复现
4. 生成文本时可调整 temperature 参数（0.1-1.0），值越小生成越确定，值越大越随机
5. 实验图表会自动保存到 results/ 目录，可直接用于 LaTeX 报告

## 扩展方向
1. 增大模型规模（d_model=256、Encoder 层数=3）提升语义表达能力
2. 优化生成策略（Top-k/Top-p 采样、重复惩罚）减少乱码
3. 实现 Encoder-Decoder 架构，适配机器翻译、文本摘要任务
4. 迁移至 WikiText-2 等大规模数据集验证模型泛化能力