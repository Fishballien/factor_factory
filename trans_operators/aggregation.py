# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 16:58:15 2024

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %%
'''
新增:
    - 自等权
    - 自加权
    - 下沉等权（可选下沉阶数）
    - 下沉各自加权取平均（可选下沉阶数）
'''
# %%
__all__ = ['avg_side', 'avg_imb', 'wavg_imb', 'imb_wavg', 'imb_csf']


# %% imports
import pandas as pd
import numpy as np
from tqdm import tqdm


# %%
# index_seq = ['000300', '000905', '000852', '932000']


# %%
def zero_to_nan(df):
    """
    Convert all zero values in a DataFrame to NaN.

    Parameters:
    df (pd.DataFrame): Input DataFrame.

    Returns:
    pd.DataFrame: DataFrame with zeros replaced by NaN.
    """
    return df.where(df != 0, np.nan)


def apply_daily_weights_to_timeseries(data, daily_weights):
    """
    将每日权重应用到更高频的时间序列数据（例如分钟、秒级等），
    根据日期将每日权重扩展到目标时间频率数据。
    对于 daily_weights 中没有的股票代码列，填充权重为 nan。

    参数:
    data (pd.DataFrame): 高频时间序列数据，行是时间戳，列是股票代码。
    daily_weights (pd.DataFrame): 每日权重数据，行是日期，列是股票代码。

    返回:
    pd.DataFrame: 调整后的高频时间序列数据。
    """
    daily_weights.index = pd.to_datetime(daily_weights.index)
    data_dates = data.index.normalize()
    data = data.reindex(columns=daily_weights.columns, fill_value=0)
    expanded_weights = daily_weights.reindex(data_dates) # , method='ffill'
    expanded_weights = zero_to_nan(expanded_weights)
    adjusted_data = data.mul(expanded_weights.values, axis=0)
    return adjusted_data


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


def normalize_daily_weights(df):
    """
    Normalize daily weights DataFrame with faster computation. Set positive values to 1,
    zeros or NaNs to 0, and normalize each row such that the sum of each row equals 1.

    Args:
        df (pd.DataFrame): Input DataFrame with daily weights.

    Returns:
        pd.DataFrame: Normalized DataFrame with rows summing to 1.
    """
    # Convert to NumPy array for fast computation
    array = df.to_numpy()
    
    # Create a binary mask: 1 for positive values, 0 otherwise (NaN treated as 0)
    binary_mask = np.where(array > 0, 1, np.nan)
    
    # Calculate row sums
    # row_sums = binary_mask.sum(axis=1, keepdims=True)
    row_sums = np.nansum(binary_mask, axis=1, keepdims=True)
    
    # Avoid division by zero: mask rows where row_sums == 0
    row_sums[row_sums == 0] = 1  # Temporarily set zero sums to 1 to avoid NaN
    
    # Normalize rows
    normalized_array = binary_mask / row_sums
    
    # Convert back to DataFrame
    normalized_df = pd.DataFrame(normalized_array, index=df.index, columns=df.columns)
    
    return normalized_df


def merge_index_weights_to_binary(dfs):
    """
    Merge a list of DataFrames containing index constituent weights by taking the union of rows and columns.
    Assign 1 to cells where any weight > 0 and NaN elsewhere.

    Parameters:
    dfs (list of pd.DataFrame): List of DataFrames with stock weights. Rows are timestamps, columns are stocks.

    Returns:
    pd.DataFrame: A DataFrame with merged rows and columns, binarized as described.
    """
    # Get the union of all rows and columns efficiently
    all_rows = pd.Index(sorted(set.union(*(set(df.index) for df in dfs))))
    all_columns = pd.Index(sorted(set.union(*(set(df.columns) for df in dfs))))

    # Create a combined DataFrame with NaN
    combined_df = pd.DataFrame(np.nan, index=all_rows, columns=all_columns)

    # Update the combined DataFrame with binary values
    for df in dfs:
        binary_df = (df > 0).astype(float)
        binary_df[binary_df <= 0] = np.nan  # Explicitly set all non-positive values to NaN
        combined_df = combined_df.combine_first(binary_df)

    # Final cleanup to ensure only 1.0 and NaN
    combined_df[combined_df <= 0] = np.nan

    return combined_df


def get_downscale_indexes(index_name, index_seq, downscale_depth):
    self_index_rank = index_seq.index(index_name)
    downscale_indexes = [index_seq[rank] for rank in range(self_index_rank, self_index_rank+downscale_depth+1)
                         if rank < len(index_seq)]
    return downscale_indexes


def get_complement_indexes(index_name, index_seq):
    self_index_rank = index_seq.index(index_name)
    complement_indexes = [index_seq[rank] for rank in range(len(index_seq))
                         if rank < self_index_rank]
    return complement_indexes


def get_complement_weights(index_name, index_all, index_seq, daily_weights):
    complement_indexes = get_complement_indexes(index_name, index_seq)
    complement_index_weights = [daily_weights[complement_index]
                                for complement_index in complement_indexes]
    weight = subtract_weights(daily_weights[index_all], complement_index_weights)
    return weight


def get_merged_binary_weight_by_depth(norm_daily_weights, index_name, index_all, index_seq, downscale_depth):
    if downscale_depth == 0:
        weight = norm_daily_weights[index_name]
    elif isinstance(downscale_depth, int):
        downscale_indexes = get_downscale_indexes(index_name, index_seq, downscale_depth)
        depth_daily_weights = [norm_daily_weights[downscale_index]
                               for downscale_index in downscale_indexes]
        weight = merge_index_weights_to_binary(depth_daily_weights)
    elif downscale_depth == 'all':
        weight = get_complement_weights(index_name, index_all, index_seq, norm_daily_weights)
    else:
        raise NotImplementedError()
    return weight


def wavg_imb_by_single_index(index_name, ind_sides, imb_func, daily_weights, cs_func):
    wadj_sides = {side: apply_daily_weights_to_timeseries(ind_sides[side], daily_weights[index_name]) for side in ind_sides}
    wadj_mean_sides = {side: cs_func(wadj_sides[side]) for side in wadj_sides}
    return imb_func(wadj_mean_sides['Bid'], wadj_mean_sides['Ask'])


def imb_wavg_by_single_index(index_name, imb, daily_weights, cs_func):
    wadj_imb = apply_daily_weights_to_timeseries(imb, daily_weights[index_name])
    return cs_func(wadj_imb)


def mask_df_by_positive_values(df1, df2_list):
    """
    使用 df2_list 中的多个 DataFrame 累积掩码 df1，其中只要 df2 中的值大于 0，就会掩码 df1。
    
    参数:
    df1 (pd.DataFrame): 被掩码的 DataFrame
    df2_list (list): 包含多个用于生成掩码的 DataFrame 的列表
    
    返回:
    pd.DataFrame: 应用掩码后的 DataFrame
    """
    # 初始化一个与 df1 形状相同的掩码，所有值为 False
    combined_mask = pd.DataFrame(False, index=df1.index, columns=df1.columns)
    
    # 遍历 df2_list，对每个 df2 累积掩码
    for df2 in df2_list:
        # 对齐当前 df2 的索引和列到 df1
        aligned_df2 = df2.reindex_like(df1).fillna(0)
        
        # 生成当前 df2 的掩码：大于 0 的位置为 True
        current_mask = aligned_df2 > 0
        
        # 累积掩码
        combined_mask |= current_mask
    
    # 使用 combined_mask 将 df1 中对应位置设置为 NaN
    masked_df1 = df1.where(~combined_mask, np.nan)
    
    return masked_df1


def scale_rows_to_mean(df, target_mean=1.0):
    """
    对 DataFrame 的每一行进行线性缩放，使每行的均值调整为目标值。
    
    参数:
    df (pd.DataFrame): 待缩放的 DataFrame
    target_mean (float): 缩放后每行的目标均值
    
    返回:
    pd.DataFrame: 每行均值调整后的 DataFrame
    """
    # 计算每行当前的均值
    row_means = df.mean(axis=1)
    
    # 缩放因子 = 目标均值 / 当前均值
    scale_factors = target_mean / row_means
    
    # 对每行进行缩放
    scaled_df = df.mul(scale_factors, axis=0)
    
    return scaled_df


def subtract_weights(weight_board, weight_narrow_list):
    weight_board_masked = mask_df_by_positive_values(weight_board, weight_narrow_list)
    return scale_rows_to_mean(weight_board_masked)


# %%
def avg_side(ind_sides, target_indexes, daily_weights, index_all, index_seq, downscale_depth, 
             imb_func, ts_func_with_pr, cs_func, n_workers=1):
    res = {'Bid': {}, 'Ask': {}}
    norm_daily_weights = {index_code: normalize_daily_weights(daily_weight) 
                          for index_code, daily_weight in daily_weights.items()}
    iter_ = tqdm(target_indexes, desc='avg_imb by indexes') if n_workers == 1 else target_indexes
    for index_name in iter_:
        weight = get_merged_binary_weight_by_depth(norm_daily_weights, index_name, 
                                                   index_all, index_seq, downscale_depth)
        adj_sides = {side: apply_daily_weights_to_timeseries(ind_sides[side], weight) for side in ind_sides}
        adj_mean_sides = {side: cs_func(adj_sides[side]) for side in adj_sides}
        for side in ('Bid', 'Ask'):
            res[side][index_name] = adj_mean_sides[side]
    return {side: pd.DataFrame(res[side]) for side in res}


def avg_imb(ind_sides, target_indexes, daily_weights, index_all, index_seq, downscale_depth, 
            imb_func, ts_func_with_pr, cs_func, n_workers=1):
    res = {}
    norm_daily_weights = {index_code: normalize_daily_weights(daily_weight) 
                          for index_code, daily_weight in daily_weights.items()}
    iter_ = tqdm(target_indexes, desc='avg_imb by indexes') if n_workers == 1 else target_indexes
    for index_name in iter_:
        weight = get_merged_binary_weight_by_depth(norm_daily_weights, index_name, 
                                                   index_all, index_seq, downscale_depth)
        adj_sides = {side: apply_daily_weights_to_timeseries(ind_sides[side], weight) for side in ind_sides}
        adj_mean_sides = {side: cs_func(adj_sides[side]) for side in adj_sides}
        res[index_name] = imb_func(adj_mean_sides['Bid'], adj_mean_sides['Ask'])
    return pd.DataFrame(res)


def wavg_imb(ind_sides, target_indexes, daily_weights, index_all, index_seq, downscale_depth, 
             imb_func, ts_func_with_pr, cs_func, n_workers=1):
    res = {}
    res_by_index = {}
    iter_ = tqdm(target_indexes, desc='wavg_imb by indexes') if n_workers == 1 else target_indexes
    if isinstance(downscale_depth, int) and downscale_depth >= 0:
        for index_name in index_seq:
            res_by_index[index_name] = wavg_imb_by_single_index(index_name, ind_sides, imb_func, daily_weights, cs_func)
        for index_name in iter_:
            downscale_indexes = get_downscale_indexes(index_name, index_seq, downscale_depth)
            res[index_name] = pd.concat([res_by_index[downscale_index]
                                         for downscale_index in downscale_indexes], axis=1).mean(axis=1)
    elif isinstance(downscale_depth, str) and downscale_depth == 'all':
        new_weights = {}
        for index_name in iter_:
            new_weights[index_name] = get_complement_weights(index_name, index_all, index_seq, daily_weights)
        for index_name in iter_:
            res[index_name] = wavg_imb_by_single_index(index_name, ind_sides, imb_func, new_weights, cs_func)
    return pd.DataFrame(res)


def imb_wavg(ind_sides, target_indexes, daily_weights, index_all, index_seq, downscale_depth, 
             imb_func, ts_func_with_pr, cs_func, n_workers=1):
    res = {}
    res_by_index = {}
    imb = imb_func(ind_sides['Bid'], ind_sides['Ask'])
    if ts_func_with_pr is not None:
        imb = ts_func_with_pr(imb)
    iter_ = tqdm(target_indexes, desc='imb_wavg by indexes') if n_workers == 1 else target_indexes
    if isinstance(downscale_depth, int) and downscale_depth >= 0:
        for index_name in index_seq:
            res_by_index[index_name] = imb_wavg_by_single_index(index_name, imb, daily_weights, cs_func)
        for index_name in iter_:
            downscale_indexes = get_downscale_indexes(index_name, index_seq, downscale_depth)
            res[index_name] = pd.concat([res_by_index[downscale_index]
                                       for downscale_index in downscale_indexes], axis=1).mean(axis=1)
    elif isinstance(downscale_depth, str) and downscale_depth == 'all':
        new_weights = {}
        for index_name in iter_:
            new_weights[index_name] = get_complement_weights(index_name, index_all, index_seq, daily_weights)
        for index_name in iter_:
            res[index_name] = imb_wavg_by_single_index(index_name, imb, new_weights, cs_func)
    return pd.DataFrame(res)


def imb_avg(ind_sides, target_indexes, daily_weights, index_all, index_seq, downscale_depth, 
             imb_func, ts_func_with_pr, cs_func, n_workers=1):
    res = {}
    imb = imb_func(ind_sides['Bid'], ind_sides['Ask'])
    if ts_func_with_pr is not None:
        imb = ts_func_with_pr(imb)
    norm_daily_weights = {index_code: normalize_daily_weights(daily_weight) 
                          for index_code, daily_weight in daily_weights.items()}
    iter_ = tqdm(target_indexes, desc='imb_avg by indexes') if n_workers == 1 else target_indexes
    for index_name in iter_:
        weight = get_merged_binary_weight_by_depth(norm_daily_weights, index_name, 
                                                   index_all, index_seq, downscale_depth)
        res[index_name] = imb_wavg_by_single_index(index_name, imb, weight, cs_func)
    return pd.DataFrame(res)



def imb_csf(ind_sides, daily_weights, imb_func, cs_func, n_workers=1):
    res = {}
    iter_ = tqdm(daily_weights.items(), desc='imb_csf by indexes') if n_workers == 1 else daily_weights.items()
    for index_name, daily_weight in iter_:
        imb = imb_func(ind_sides['Bid'], ind_sides['Ask'])
        wadj_imb = apply_daily_weights_to_timeseries(imb, daily_weight)
        res[index_name] = cs_func(wadj_imb)
    return pd.DataFrame(res)