# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 16:37:42 2024

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %% imports
from pathlib import Path
import pandas as pd


from utils.market import index_mapping


# %%
for index_name in index_mapping:
    index_code = index_mapping[index_name]
    path = rf'D:/CNIndexFutures/timeseries/factor_factory/sample_data/stockindex_weights/{index_name}_index_weight.csv'
    target_dir = Path(r'D:/CNIndexFutures/timeseries/factor_factory/sample_data/weights_matrix')
    target_dir.mkdir(parents=True, exist_ok=True)
    
    
    # %%
    weights = pd.read_csv(path)
    
    
    # %%
    # 假设你的 DataFrame 叫做 weights
    # 将 trade_date 转换为日期格式
    weights['trade_date'] = pd.to_datetime(weights['trade_date'], format='%Y%m%d')
    
    # 创建每日日期范围
    daily_index = pd.date_range(start=weights['trade_date'].min(), end=weights['trade_date'].max(), freq='D')
    # 获取所有股票代码，提前定义列名
    all_con_codes = weights['con_code'].unique()
    # 初始化 DataFrame，设定索引为每日日期，列为所有股票代码
    daily_weights = pd.DataFrame(index=daily_index, columns=all_con_codes)
    
    trade_dates = weights['trade_date'].unique()
    
    
    # 对每只股票逐列填充数据
    for con_code, group in weights.groupby('con_code'):
        # 将 trade_date 设置为索引并向前填充
        stock_weights = group.set_index('trade_date')['weight'].reindex(trade_dates).fillna(0).reindex(daily_index, method='ffill')
        # 填充 daily_weights 中对应的列
        daily_weights.loc[:, con_code] = stock_weights

    # 检查结果
    daily_weights.to_parquet(target_dir / f'{index_code}.parquet')