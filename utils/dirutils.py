# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 17:53:07 2024

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %% imports
from pathlib import Path
import yaml


# %% load path
def load_path_config(project_dir):
    path_config_path = project_dir / '.path_config.yaml'
    with path_config_path.open('r') as file:
        path_config = yaml.safe_load(file)
    return path_config


# %%
def find_common_bid_ask_files(path: Path):
    """
    搜索路径中所有Bid_*.parquet和Ask_*.parquet文件，提取文件名中*部分，并返回交集。

    Args:
        path (Path): 输入路径，类型为pathlib.Path。

    Returns:
        set: Bid和Ask文件名中提取出的*部分的交集。
    """
    # 找出所有符合Bid_*.parquet的文件
    bid_files = {file.stem[4:] for file in path.glob("Bid_*.parquet")}
    
    # 找出所有符合Ask_*.parquet的文件
    ask_files = {file.stem[4:] for file in path.glob("Ask_*.parquet")}
    
    # 返回两者的交集
    return sorted(bid_files & ask_files)

# =============================================================================
# # 示例调用
# path = Path(r'D:\CNIndexFutures\timeseries\factor_factory\sample_data\indicators\test')
# common_files = find_common_bid_ask_files(path)
# print(common_files)
# =============================================================================


def list_parquet_files(path: Path):
    """
    读取指定路径下所有的parquet文件，并返回不带扩展名的文件名列表。
    
    Args:
        path (Path): 输入路径，类型为pathlib.Path。
    
    Returns:
        list: 所有parquet文件的文件名（不包含扩展名）列表。
    """
    # 找出所有符合*.parquet的文件
    parquet_files = [file.stem for file in path.glob("*.parquet")]
    
    # 返回排序后的文件名列表
    return sorted(parquet_files)