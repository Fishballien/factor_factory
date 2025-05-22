# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 16:42:18 2024

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %% imports
import pandas as pd
import numpy as np
from joblib import Parallel, delayed


from speedutils import timeit


# %%
# hs300 五年 约30s
@timeit
def get_mean_by_row(df):
    return df.mean(axis=1)


# 几乎不能用
@timeit
def get_mean_by_row(df, n_jobs=256):
    """
    使用 joblib 并行计算每行均值。
    
    参数:
    df (pd.DataFrame): 输入的 DataFrame。
    n_jobs (int): 并行作业的核数。
    
    返回:
    np.ndarray: 每行的均值。
    """
    # 将 DataFrame 拆分为 n_jobs 份
    chunks = np.array_split(df, n_jobs)

    # 并行计算每行均值
    results = Parallel(n_jobs=n_jobs)(
        delayed(lambda x: x.mean(axis=1).to_numpy())(chunk) for chunk in chunks
    )
    
    # 合并结果
    row_means = np.concatenate(results)

    # 转回 Series，保持索引一致，不添加 name
    return pd.Series(row_means, index=df.index)


# hs300 五年 约5s
@timeit
def get_mean_by_row(df):
    """
    使用 NumPy 矩阵计算 DataFrame 每行的平均值。
    
    参数:
    df (pd.DataFrame): 输入的 DataFrame。
    
    返回:
    pd.Series: 每行的平均值，索引与原始 DataFrame 一致。
    """
    # 转为 NumPy 矩阵并计算行均值
    row_means = np.nanmean(df.to_numpy(), axis=1)
    
    # 转回 Pandas Series，保持索引一致
    return pd.Series(row_means, index=df.index)