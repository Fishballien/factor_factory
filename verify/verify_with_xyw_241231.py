# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 11:28:59 2024

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
from pathlib import Path
from datetime import timedelta


from utils.datautils import replace_column_suffixes


# %%
def reindex_with_union(df1, df2):
    """
    获取两个 DataFrame 的列名并集，并按并集重新索引两个 DataFrame。

    参数:
    df1 (pd.DataFrame): 第一个 DataFrame。
    df2 (pd.DataFrame): 第二个 DataFrame。

    返回:
    tuple: (df1_reindexed, df2_reindexed) 按列名并集重新索引后的两个 DataFrame。
    """
    # 获取两个 DataFrame 列名的并集，保持顺序
    combined_columns = sorted(set(df1.columns).union(set(df2.columns)))

    # 按并集中的列重新索引两个 DataFrame
    df1_reindexed = df1.reindex(columns=combined_columns)
    df2_reindexed = df2.reindex(columns=combined_columns)

    return df1_reindexed, df2_reindexed


# %% zxt
a_path = r'D:/CNIndexFutures/timeseries/verify/verify/vtdoa_p1.0_v40000_d0.1_a.parquet'
b_path = r'D:/CNIndexFutures/timeseries/verify/verify/vtdoa_p1.0_v40000_d0.1_b.parquet'
f_path = r'D:/CNIndexFutures/timeseries/verify/verify/vtdoa_p1.0_v40000_d0.1.parquet'
sz50_path = 'D:/CNIndexFutures/timeseries/verify/verify/ValueTimeDecayOrderAmount_p1.0_v40000_d0.1-imb01_wavg-org.parquet'
sz50wgt_path = r'D:\CNIndexFutures\timeseries\factor_factory\sample_data\stockindex_weights\zz1000_index_weight.csv'


# %% xyw
b_xyw_path = r'D:/CNIndexFutures/timeseries/verify/verify/cn_vtdoa/merged_t0d0s0.parquet'
a_xyw_path = r'D:/CNIndexFutures/timeseries/verify/verify/cn_vtdoa/merged_t0d0s1.parquet'
a_xyw_dir = Path(r'D:/CNIndexFutures/timeseries/verify/verify/cn_vtdoa/20240426')
f_xyw_path = r'D:/CNIndexFutures/timeseries/verify/verify/cn_imbwavg/20240426/000852.XSHG.parquet'


# %%
a = replace_column_suffixes(pd.read_parquet(a_path)).astype(np.float32)
b = replace_column_suffixes(pd.read_parquet(b_path)).astype(np.float32)
f = pd.read_parquet(f_path)
sz50 = pd.read_parquet(sz50_path)
sz50_s = sz50.loc['20240426':'20240427'].rename(columns={'399001': '000016'})
sz50_weight = pd.read_csv(sz50wgt_path)
sz50_weight_pivot = sz50_weight.pivot(index='trade_date', columns='con_code', values='weight')
target_weight = sz50_weight_pivot.loc[20240329]
sz50_index = sz50_weight_pivot.columns[sz50_weight_pivot.loc[20240329].notna()].values
for i in range(10):
    a_50 = a.loc['20240426':'20240427', sz50_index].iloc[i]
    b_50 = b.loc['20240426':'20240427', sz50_index].iloc[i]
    imb_wavg = np.nanmean((b_50 - a_50) / (b_50 + a_50) * sz50_weight_pivot.loc[20240329])
    print(imb_wavg)


# %%
a_xyw = replace_column_suffixes(pd.read_parquet(a_xyw_path))
# b_xyw = replace_column_suffixes(pd.read_parquet(b_xyw_path))
f_xyw = pd.read_parquet(f_xyw_path)

# stock_codes = [code.replace('.SH', '.XSHG').replace('.SZ', '.XSHE') for code in sz50_index]
# res_list = []
# for stock in stock_codes:
#     stock_data = pd.read_parquet(a_xyw_dir / f'{stock}.parquet').set_index('timestamp')
#     stock_data = stock_data[~stock_data.index.duplicated(keep='first')]
#     res_list.append(stock_data['t0d0s1'].rename(stock))
# a_xyw = pd.concat(res_list, axis=1)
# a_xyw = replace_column_suffixes(a_xyw)
# a_xyw.index = a_xyw.index + timedelta(hours=8)


# %%
a, a_xyw = reindex_with_union(a, a_xyw)
diff_a = a-a_xyw
diff_a_s = diff_a[sz50_index]
diff_a_s.to_parquet(r'D:/CNIndexFutures/timeseries/verify/verify/diff_a.parquet')
diff_summary = diff_a_s.abs().iloc[:15].describe()
diff_summary.loc['max'].describe()
max_idx = diff_summary.loc['max'].argmax()
max_stock = diff_summary.columns[max_idx]
max_stock_diff = diff_a_s[max_stock]