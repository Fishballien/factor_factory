# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 16:50:12 2024

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
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed


from trans_operators import imb01
from utils.market import index_mapping
from utils.speedutils import timeit


# %%
sides = ('bid', 'ask')


# %%
def replace_column_suffixes(data, old_suffix1=".XSHE", new_suffix1=".SZ", old_suffix2=".XSHG", new_suffix2=".SH"):
    """
    替换 DataFrame 列名的后缀：
    - 将指定的 old_suffix1 替换为 new_suffix1
    - 将指定的 old_suffix2 替换为 new_suffix2

    参数:
    data (pd.DataFrame): 包含股票代码的 DataFrame，列名中含有需要替换的后缀。
    old_suffix1 (str): 要替换的第一个旧后缀，默认为 ".XSHE"。
    new_suffix1 (str): 替换的第一个新后缀，默认为 ".SZ"。
    old_suffix2 (str): 要替换的第二个旧后缀，默认为 ".XSHG"。
    new_suffix2 (str): 替换的第二个新后缀，默认为 ".SH"。

    返回:
    pd.DataFrame: 列名后缀已替换的新 DataFrame。
    """
    # 替换指定的列名后缀
    data.columns = data.columns.str.replace(old_suffix1, new_suffix1).str.replace(old_suffix2, new_suffix2)
    return data


# =============================================================================
# @timeit
# def apply_daily_weights_to_timeseries(data, daily_weights):
#     """
#     将每日权重应用到更高频的时间序列数据（例如分钟、秒级等），
#     根据日期将每日权重扩展到目标时间频率数据。
#     对于 daily_weights 中没有的股票代码列，填充权重为 0。
# 
#     参数:
#     data (pd.DataFrame): 高频时间序列数据，行是时间戳，列是股票代码。
#     daily_weights (pd.DataFrame): 每日权重数据，行是日期，列是股票代码。
# 
#     返回:
#     pd.DataFrame: 调整后的高频时间序列数据。
#     """
#     # 确保 daily_weights 的索引是日期格式
#     daily_weights.index = pd.to_datetime(daily_weights.index)
#     
#     # 提取 data 的日期部分用于匹配
#     data_dates = data.index.normalize()
#     
#     # 扩展 daily_weights 列，使其包含 data 中的所有列，不存在的列填充 0
#     expanded_weights = daily_weights.reindex(columns=data.columns, fill_value=0)
#     
#     # 将每日权重扩展到目标时间频率的数据
#     expanded_weights = expanded_weights.reindex(data_dates, method='ffill')
#     
#     # 逐元素相乘，应用权重
#     adjusted_data = data.mul(expanded_weights.values, axis=0)
#     
#     return adjusted_data
# =============================================================================


# =============================================================================
# @timeit
# def apply_daily_weights_to_timeseries(data, daily_weights):
#     """
#     使用 NumPy 优化方法，将每日权重直接应用到高频时间序列数据。
# 
#     参数:
#     data (pd.DataFrame): 高频时间序列数据，行是时间戳，列是股票代码。
#     daily_weights (pd.DataFrame): 每日权重数据，行是日期，列是股票代码。
# 
#     返回:
#     pd.DataFrame: 调整后的高频时间序列数据。
#     """
#     # 确保 daily_weights 索引为日期格式
#     daily_weights.index = pd.to_datetime(daily_weights.index)
#     
#     # 提取 data 的日期部分并找到对应权重
#     data_dates = data.index.normalize()
#     unique_dates = np.unique(data_dates)
#     
#     # 扩展权重到所有日期
#     expanded_weights = daily_weights.reindex(unique_dates, fill_value=0).ffill().reindex(columns=data.columns, fill_value=0)
# 
#     # 转为 NumPy 数组操作
#     data_array = data.to_numpy()
#     weights_array = expanded_weights.loc[data_dates].to_numpy()
# 
#     # 应用权重
#     adjusted_array = data_array * weights_array
# 
#     # 转回 DataFrame
#     return pd.DataFrame(adjusted_array, index=data.index, columns=data.columns)
# =============================================================================


@timeit
def apply_daily_weights_to_timeseries(data, daily_weights):
    """
    将每日权重应用到更高频的时间序列数据（例如分钟、秒级等），
    根据日期将每日权重扩展到目标时间频率数据。
    对于 daily_weights 中没有的股票代码列，填充权重为 0。

    参数:
    data (pd.DataFrame): 高频时间序列数据，行是时间戳，列是股票代码。
    daily_weights (pd.DataFrame): 每日权重数据，行是日期，列是股票代码。

    返回:
    pd.DataFrame: 调整后的高频时间序列数据。
    """
    daily_weights.index = pd.to_datetime(daily_weights.index)
    data_dates = data.index.normalize()
    data = data.reindex(columns=daily_weights.columns, fill_value=0)
    expanded_weights = daily_weights.reindex(data_dates, method='ffill')
    adjusted_data = data.mul(expanded_weights.values, axis=0)
    return adjusted_data


# =============================================================================
# @timeit
# def get_mean_by_row(df):
#     return df.mean(axis=1)
# =============================================================================


# =============================================================================
# @timeit
# def get_mean_by_row(df, n_jobs=256):
#     """
#     使用 joblib 并行计算每行均值。
#     
#     参数:
#     df (pd.DataFrame): 输入的 DataFrame。
#     n_jobs (int): 并行作业的核数。
#     
#     返回:
#     np.ndarray: 每行的均值。
#     """
#     # 将 DataFrame 拆分为 n_jobs 份
#     chunks = np.array_split(df, n_jobs)
# 
#     # 并行计算每行均值
#     results = Parallel(n_jobs=n_jobs)(
#         delayed(lambda x: x.mean(axis=1).to_numpy())(chunk) for chunk in chunks
#     )
#     
#     # 合并结果
#     row_means = np.concatenate(results)
# 
#     # 转回 Series，保持索引一致，不添加 name
#     return pd.Series(row_means, index=df.index)
# =============================================================================


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


# %%
ind_dir = Path(r'D:\CNIndexFutures\timeseries\factor_factory\sample_data\indicators/test')
daily_weights_dir = Path(r'D:/CNIndexFutures/timeseries/factor_factory/sample_data/weights_matrix_fr_sql')
factor_dir = Path(r'D:/CNIndexFutures/timeseries/factor_factory/sample_data/factors/test')
# ind_dir = Path(r'/mnt/data1/xintang/lob_indicators/indv0_total_amount/cs')
# daily_weights_dir = Path(r'/mnt/data1/xintang/factor_factory/sample_data/weights_matrix')
# factor_dir = Path(r'/mnt/data1/xintang/factor_factory/sample_data/factors')
ind_name = 'l_amount'
index_list = ['hs300'] #list(index_mapping.keys())


# %%
ind_sides = {side: replace_column_suffixes(pd.read_parquet(ind_dir / f'{side}_{ind_name}.parquet')).resample('1min').asfreq()
             for side in sides}


# %%
daily_weights = {}
for index_name in index_list:
    index_code = index_mapping[index_name]
    daily_weights[index_code] = pd.read_parquet(daily_weights_dir / f'{index_code}.parquet')


# %% ver 1: wavg amount -> imb
imb_name1 = 'wavg_imb01'
res_ver1 = {}
# for index_name, daily_weight in tqdm(daily_weights.items(), desc='processing indexes'):
for index_name, daily_weight in daily_weights.items():
    wadj_sides = {side: apply_daily_weights_to_timeseries(ind_sides[side], daily_weight) for side in sides}
    wadj_mean_sides = {side: get_mean_by_row(wadj_sides[side]) for side in wadj_sides}
    res_ver1[index_name] = imb01(wadj_mean_sides['bid'], wadj_mean_sides['ask'])
df_ver1 = pd.DataFrame(res_ver1)
df_ver1.to_parquet(factor_dir / f'{ind_name}_{imb_name1}.parquet')


# %% ver 2: imb -> wavg
imb_name2 = 'imb01_wavg'
res_ver2 = {}
# for index_name, daily_weight in tqdm(daily_weights.items(), desc='processing indexes'):
for index_name, daily_weight in daily_weights.items():
    imb = imb01(ind_sides['bid'], ind_sides['ask'])
    wadj_imb = apply_daily_weights_to_timeseries(imb, daily_weight)
    res_ver2[index_name] = get_mean_by_row(wadj_imb)
df_ver2 = pd.DataFrame(res_ver2)
df_ver2.to_parquet(factor_dir / f'{ind_name}_{imb_name2}.parquet')
