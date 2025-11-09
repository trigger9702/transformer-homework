#!/bin/bash
set -e

# 设置随机种子（确保可复现）
export PYTHONHASHSEED=42

# 1. 安装依赖
echo "===== 安装依赖 ====="
pip install -r requirements.txt

# 2. 训练主模型
echo -e "\n===== 训练Transformer模型 ====="
python src/train.py

# 3. 运行消融实验
echo -e "\n===== 运行消融实验 ====="
python src/ablation.py

# 4. 文本生成
echo -e "\n===== 生成文本示例 ====="
python src/generate.py

echo -e "\n===== 所有任务完成！结果已保存至 results/ 文件夹 ====="