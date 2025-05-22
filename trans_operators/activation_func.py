# -*- coding: utf-8 -*-
"""
Created on Thu May 22 10:24:37 2025

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %%
import pandas as pd
import numpy as np


# %%
def power(df: pd.DataFrame, gamma: float, direction: str = "pos") -> pd.DataFrame:
    """
    对 0~1 区间的数据应用幂函数变换：
    - direction = "pos"：压缩小值（靠近 0 的值更小）
    - direction = "neg"：压缩大值（靠近 1 的值更小）

    参数:
        df (pd.DataFrame): 输入 0~1 区间的 DataFrame。
        gamma (float): 幂指数，必须为正数。
        direction (str): "pos" 表示压缩小值；"neg" 表示压缩大值。

    返回:
        pd.DataFrame: 变换后的 DataFrame。
    """
    if gamma <= 0:
        raise ValueError("gamma 必须为正数")
    if direction not in {"pos", "neg"}:
        raise ValueError("direction 必须为 'pos' 或 'neg'")

    def transform(x):
        if not (0 <= x <= 1):
            return np.nan
        if direction == "pos":
            return np.power(x, gamma)
        else:  # direction == "neg"
            return np.power(1 - x, gamma)

    return df.applymap(transform)

