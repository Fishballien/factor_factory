# -*- coding: utf-8 -*-
"""
Created on Thu May  8 14:29:37 2025

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
import numpy as np


# %%
# path = 'D:/mnt/idx_opt_processed/oi/oi_159915_old.csv'
# # path = 'D:/mnt/idx_opt_processed/oi_159915_raw/oi_159915_2025.csv'
# target_dir = Path('D:/mnt/idx_opt_processed/oi_510500')


# %%
path = 'D:/mnt/idx_opt_processed/oi_159915_raw/oi_159915_old.csv'
# path = 'D:/mnt/idx_opt_processed/oi_159915_raw/oi_159915_2025.csv'
target_dir = Path('D:/mnt/idx_opt_processed/oi_159915')


# %%
target_dir.mkdir(parents=True, exist_ok=True)
# 读取CSV文件，直接作为数字读取
oi_data = pd.read_csv(path)

# %%
# Assuming your dataframe is called oi_data
# First, convert the dt column to datetime type
oi_data['dt'] = pd.to_datetime(oi_data['dt']).dt.tz_localize(None)

# Set dt as the index
oi_data = oi_data.set_index('dt') #.astype(np.float64)

# Select only the columns you want
oi_data = oi_data[['call_oi_sum', 'put_oi_sum', 'pc']]

# Remove commas from numeric columns if they exist
oi_data['call_oi_sum'] = oi_data['call_oi_sum'].astype(str).str.replace(',', '').astype(float)
oi_data['put_oi_sum'] = oi_data['put_oi_sum'].astype(str).str.replace(',', '').astype(float)
oi_data['pc'] = oi_data['pc'].astype(str).str.replace(',', '').astype(float)
oi_data['oi_imb01'] = (oi_data['call_oi_sum'] - oi_data['put_oi_sum']) / (oi_data['call_oi_sum'] + oi_data['put_oi_sum'])

oi_data = oi_data.resample('1min', closed='right', label='right').mean()


# %%
# 创建交易时段的过滤器
def is_trading_hours(timestamp):
    # 获取小时和分钟
    hour = timestamp.hour
    minute = timestamp.minute
    
    # 上午交易时段：9:30-11:30
    morning_session = (hour == 9 and minute >= 30) or (hour == 10) or (hour == 11 and minute <= 30)
    
    # 下午交易时段：13:00-15:00
    afternoon_session = (hour >= 13 and hour <= 14) or (hour == 15 and minute == 0)
    
    return morning_session or afternoon_session

# 应用过滤器来筛选交易时段的数据
oi_data = oi_data[oi_data.index.map(is_trading_hours)]

for col in oi_data.columns:
    # Create a DataFrame with just this column (preserves column name)
    col_df = oi_data[[col]]
    # Save as parquet
    col_df.to_parquet(target_dir / f'{col}.parquet')
    
    
# %%
# =============================================================================
# # 复制原始数据
# df = oi_data.copy()
# 
# # 确保索引是datetime格式
# if not isinstance(df.index, pd.DatetimeIndex):
#     df.index = pd.to_datetime(df.index)
# 
# # 创建一个新列，只有9:30的行保留原值，其他设为NA
# for col in ['call_oi_sum', 'put_oi_sum', 'pc', 'oi_imb01']:
#     df[f'{col}_930'] = np.where(df.index.time == pd.to_datetime('09:30:00').time(), 
#                                 df[col], 
#                                 np.nan)
# 
# # 按日期分组，对每天的数据向后填充
# df = df.groupby(df.index.date).apply(lambda x: x.fillna(method='ffill'))
# # 重置索引（移除多级索引）
# df = df.reset_index(level=0, drop=True)
# 
# # 计算每个时间点与当天9:30的差值
# for col in ['call_oi_sum', 'put_oi_sum', 'pc', 'oi_imb01']:
#     df[f'{col}_diff'] = df[col] - df[f'{col}_930']
# 
# # 只保留原始列和差值列
# result = df[['call_oi_sum', 'put_oi_sum', 'pc', 'oi_imb01'] + 
#            [f'{col}_diff' for col in ['call_oi_sum', 'put_oi_sum', 'pc', 'oi_imb01']]]
# 
# # 显示结果的前几行
# print(result.head(10))
# 
# for col in result.columns:
#     # Create a DataFrame with just this column (preserves column name)
#     col_df = result[[col]]
#     # Save as parquet
#     col_df.to_parquet(target_dir / f'{col}.parquet')
# 
# =============================================================================
