# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 10:02:30 2025

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %%
import pandas as pd
from pathlib import Path


from trans_operators.ts_intraday import *


# %%
raw_path = 'D:/mnt/data1/factors/basis_fr_hxl/IC_basis_pct.parquet'
target_dir = Path('D:/mnt/data1/factors/basis_pct')
target_dir.mkdir(parents=True, exist_ok=True)


# %%
raw_basis = pd.read_parquet(raw_path)
raw_basis['z2_z1_diff'] = raw_basis['z2'] - raw_basis['z1']

for col in raw_basis.columns:
    # Create a DataFrame with just this column (preserves column name)
    col_df = raw_basis[[col]]
    # Save as parquet
    col_df.to_parquet(target_dir / f'IC_{col}.parquet')
    
    
# %% intraday diff
diff_df = calculate_time_differences(raw_basis, '09:31:00')

for col in diff_df.columns:
    # Create a DataFrame with just this column (preserves column name)
    col_df = diff_df[[col]]
    # Save as parquet
    col_df.to_parquet(target_dir / f'IC_{col}.parquet')