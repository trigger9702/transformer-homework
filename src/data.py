import torch
from torch.utils.data import Dataset, DataLoader
import urllib.request
import os

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 1. 下载Tiny Shakespeare数据集（绕过Hugging Face缓存）
def download_tiny_shakespeare(local_path="datasets/tinyshakespeare.txt"):
    """下载Tiny Shakespeare数据集，若下载失败则使用备用文本"""
    os.makedirs("datasets", exist_ok=True)
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

    try:
        urllib.request.urlretrieve(url, local_path)
        print(f"数据集下载成功：{local_path}")
    except Exception as e:
        print(f"下载失败，使用备用文本：{e}")
        # 备用文本（莎士比亚风格片段）
        fallback_text = """First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?

All:
Resolved. resolved.

First Citizen:
First, you know Caius Marcius is chief enemy to the people.

All:
We know't, we know't.

First Citizen:
Let us kill him, and we'll have corn at our own price.
Is't a verdict?

All:
No more talking on't; let it be done: away, away!

Second Citizen:
One word, good citizens.

First Citizen:
We are accounted poor citizens, the patricians good.
What authority surfeits on would relieve us: if they
would yield us but the superfluity, while it were
wholesome, we might guess they relieved us humanely;
but they think we are too dear: the leanness that
afflicts us, the object of our misery, is as an
inventory to particularise their abundance; our
sufferance is a gain to them. Let us revenge this with
our pikes, ere we become rakes: for the gods know I
speak this in hunger for bread, not in thirst for revenge.
"""
        with open(local_path, "w", encoding="utf-8") as f:
            f.write(fallback_text)
        print(f"备用文本已保存：{local_path}")

    return local_path


# 2. 字符级Tokenizer
class CharTokenizer:
    def __init__(self, text):
        self.chars = sorted(list(set(text)))  # 所有唯一字符
        self.vocab_size = len(self.chars)
        self.char_to_idx = {c: i for i, c in enumerate(self.chars)}
        self.idx_to_char = {i: c for i, c in enumerate(self.chars)}

    def encode(self, text):
        return [self.char_to_idx[c] for c in text]  # 文本→索引

    def decode(self, idx):
        return "".join([self.idx_to_char[i] for i in idx])  # 索引→文本


# 3. 数据集类（LM任务：input→target右移一位）
class LMDataset(Dataset):
    def __init__(self, text, tokenizer, seq_len=64):
        self.data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        self.seq_len = seq_len
        self.total_samples = len(self.data) - self.seq_len  # 滑动窗口生成样本

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        input_seq = self.data[idx:idx + self.seq_len]
        target_seq = self.data[idx + 1:idx + self.seq_len + 1]  # target右移一位
        return input_seq, target_seq


# 4. 数据加载主函数（供其他脚本调用）
def load_data(seq_len=64, batch_size=32):
    """加载训练/验证/测试数据，返回dataloader和tokenizer"""
    # 下载数据集
    data_path = download_tiny_shakespeare()

    # 读取文本并分割训练/验证/测试集（90%/5%/5%）
    with open(data_path, "r", encoding="utf-8") as f:
        full_text = f.read()

    total_len = len(full_text)
    train_len = int(total_len * 0.9)
    val_len = int(total_len * 0.05)

    train_text = full_text[:train_len]
    val_text = full_text[train_len:train_len + val_len]
    test_text = full_text[train_len + val_len:]

    # 初始化Tokenizer
    tokenizer = CharTokenizer(train_text)
    print(f"词汇表大小：{tokenizer.vocab_size}")

    # 生成Dataset和DataLoader
    train_dataset = LMDataset(train_text, tokenizer, seq_len)
    val_dataset = LMDataset(val_text, tokenizer, seq_len)
    test_dataset = LMDataset(test_text, tokenizer, seq_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    # 验证数据加载
    x, y = next(iter(train_loader))
    print(f"Input shape: {x.shape} (batch_size, seq_len)")
    print(f"Target shape: {y.shape} (batch_size, seq_len)")

    return train_loader, val_loader, test_loader, tokenizer