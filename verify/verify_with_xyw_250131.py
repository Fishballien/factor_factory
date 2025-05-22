# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 21:23:49 2025

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %% imports
import os
import pandas as pd
from pathlib import Path


# %%
def extract_filenames(folder_path: str, extension: str = ".parquet"):
    """
    从指定文件夹提取所有指定后缀的文件的文件名（不包含后缀）。
    
    :param folder_path: 目标文件夹路径
    :param extension: 需要提取的文件后缀，默认为 ".parquet"
    :return: 文件名（不包含后缀）的列表
    """
    filenames = []
    
    if not os.path.isdir(folder_path):
        raise ValueError(f"{folder_path} 不是一个有效的文件夹路径")
    
    for file in os.listdir(folder_path):
        if file.endswith(extension):
            filenames.append(os.path.splitext(file)[0])
    
    return filenames


# %%
xyw_stock_fac_path = '/mnt/30.133_stock_data_raid0/verify/cn_vtdoa/20240426'
zxt_stock_fac_dir = '/mnt/data1/xintang/lob_indicators/Batch10_fix_best_241218_selected_f64/cs'
zxt_fac_name = 'Bid_ValueTimeDecayOrderAmount_p0.1_v40000_d0.1'
ts = '2024-04-26 09:31:00'


# %%
stocks = extract_filenames(xyw_stock_fac_path)
xyw_fac = pd.Series(index=stocks)
for stock in stocks:
    xyw_fac_stock = pd.read_parquet(Path(xyw_stock_fac_path) / f'{stock}.parquet')
    xyw_fac.loc[stock] = xyw_fac_stock.iloc[0][stock]
                                    
    
# %%
zxt_fac_path = Path(zxt_stock_fac_dir) / f'{zxt_fac_name}.parquet'
zxt_fac = pd.read_parquet(zxt_fac_path)
zxt_fac_series = zxt_fac.loc[ts]

breakpoint()
