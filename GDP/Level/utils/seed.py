import os
import random
import numpy as np
import torch

def set_seed(seed=42):
    """设置所有相关模块的随机种子，确保训练可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # CUDA 的确定性设置（可选）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 环境变量（控制 hash 随机性）
    os.environ["PYTHONHASHSEED"] = str(seed)
