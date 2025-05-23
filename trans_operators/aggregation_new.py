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
__all__ = ['avg_side', 'avg_imb', 'wavg_imb', 'imb_wavg', 'imb_csf', 'avg_scale_imb', 'selfwavg_imb', 'selfavg_imb',
           'subset_wavg_imb', 'norm_wavg_imb']


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


def apply_norm_daily_weights_to_timeseries(data, daily_weights):
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
    try:
        data_dates = data.index.normalize()
    except:
        print(data)
    data = data.reindex(columns=daily_weights.columns, fill_value=np.nan)
    expanded_weights = daily_weights.reindex(data_dates) # , method='ffill'
    expanded_weights = zero_to_nan(expanded_weights)
    adjusted_data = data.mul(expanded_weights.values, axis=0)
    return adjusted_data


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
    # 确保日期格式正确
    daily_weights.index = pd.to_datetime(daily_weights.index)
    
    # 提取日期部分
    data_dates = data.index.normalize()
    
    # 使用daily_weights的列重建data，缺失值填充为np.nan
    data = data.reindex(columns=daily_weights.columns, fill_value=np.nan)
    
    # 将日权重扩展到高频数据的时间索引（基于日期部分）
    # 创建一个临时的映射 DataFrame，其索引为原始的高频时间戳
    expanded_weights_values = daily_weights.reindex(data_dates).values
    
    # 创建expanded_weights，与data形状一致但值来自daily_weights对应的日期
    expanded_weights = pd.DataFrame(
        expanded_weights_values,
        index=data.index,  # 保持原始高频时间戳索引
        columns=daily_weights.columns
    )
    
    # 将0值转换为NaN
    expanded_weights = zero_to_nan(expanded_weights)
    
    # 如果data中某位置是NaN，则expanded_weights对应位置也设为NaN
    mask = data.isna()
    expanded_weights = expanded_weights.mask(mask, np.nan)
    
    # 计算每行非NaN值的数量和权重总和
    row_counts = expanded_weights.notna().sum(axis=1)
    weight_sums = expanded_weights.sum(axis=1)
    
    # 对每行进行归一化处理：权重除以总和再乘以非NaN值的数量
    # 使用广播机制进行批量计算
    # 处理weight_sums为0的情况
    weight_sums = weight_sums.replace(0, np.nan)
    normalized_weights = expanded_weights.div(weight_sums, axis=0).mul(row_counts, axis=0)
    
    # 计算调整后的数据
    adjusted_data = data.mul(normalized_weights)
    
    return adjusted_data


def apply_minute_weights_to_timeseries(data, weights):
    """
    将每日级别的权重扩展并应用到分钟或更高频率的时间序列数据中。

    参数:
    data (pd.DataFrame): 高频时间序列数据，索引为时间戳，列为股票代码，表示各时间点的观测值。
    weights (pd.DataFrame): 每日权重数据，索引为日期（应与data的索引可对齐），列为股票代码，表示每只股票在该日的权重。

    返回:
    pd.DataFrame: 应用权重调整后的高频时间序列数据。未提供权重的股票列将返回NaN。
    """

    # 将权重对齐到高频数据的形状（按时间戳和股票代码对齐）
    weights = weights.reindex(index=data.index, columns=data.columns)

    # 将权重中的0值处理为NaN，表示不参与计算
    weights = zero_to_nan(weights)

    # 对于原始数据中为NaN的位置，将权重同步设为NaN，保持遮罩一致性
    weights = weights.mask(data.isna(), np.nan)

    # 统计每行（即每个时间点）非NaN的权重个数与总和
    row_counts = weights.notna().sum(axis=1)
    weight_sums = weights.sum(axis=1)

    # 避免除以0，将权重总和中的0替换为NaN
    weight_sums = weight_sums.replace(0, np.nan)

    # 对每行进行归一化，使得权重和为1后再乘以有效股票数量，实现等权调整
    normalized_weights = weights.div(weight_sums, axis=0).mul(row_counts, axis=0)

    # 应用归一化后的权重调整原始数据
    adjusted_data = data.mul(normalized_weights)

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


# =============================================================================
# def wavg_imb_by_single_index(index_name, ind_sides, imb_func, daily_weights, cs_func):
#     wadj_sides = {side: apply_daily_weights_to_timeseries(ind_sides[side], daily_weights[index_name]) for side in ind_sides}
#     wadj_mean_sides = {side: cs_func(wadj_sides[side]) for side in wadj_sides}
#     return imb_func(wadj_mean_sides['Bid'], wadj_mean_sides['Ask'])
# =============================================================================


def wavg_imb_by_single_index(index_name, ind_sides, imb_func, daily_weights, cs_func, ts_func_with_pr=None):
    wadj_sides = {side: apply_daily_weights_to_timeseries(ind_sides[side], daily_weights[index_name]) for side in ind_sides}
    wadj_mean_sides = {side: cs_func(wadj_sides[side]) for side in wadj_sides}
    
    # 新增: 应用 ts_func_with_pr 到每个 side
    if ts_func_with_pr is not None:
        wadj_mean_sides = {side: ts_func_with_pr(wadj_mean_sides[side]) for side in wadj_mean_sides}
        
    return imb_func(wadj_mean_sides['Bid'], wadj_mean_sides['Ask'])


def selfwavg_imb_by_single_index(index_name, ind_sides, imb_func, weights, cs_func, ts_func_with_pr=None):
    wadj_sides = {side: apply_minute_weights_to_timeseries(ind_sides[side], weights[index_name]) for side in ind_sides}
    wadj_mean_sides = {side: cs_func(wadj_sides[side]) for side in wadj_sides}
    
    # 新增: 应用 ts_func_with_pr 到每个 side
    if ts_func_with_pr is not None:
        wadj_mean_sides = {side: ts_func_with_pr(wadj_mean_sides[side]) for side in wadj_mean_sides}
        
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


def subtract_weights(weight_board, weight_narrow_list):
    weight_board_masked = mask_df_by_positive_values(weight_board, weight_narrow_list)
    return weight_board_masked


def safe_series_exp(x):
    """
    对 Series 求 exp，保留原 index。
    如果传入的不是 Series，会尝试转换为 Series。
    """
    if not isinstance(x, pd.Series):
        x = pd.Series(x)
    return pd.Series(np.exp(x.values), index=x.index)


def filter_weights_by_exchange(daily_weights, exchange='all'):
    """
    根据交易所过滤权重数据。
    
    参数:
    daily_weights (dict): 字典，键为指数代码，值为权重DataFrame。
    exchange (str): 交易所选择，可选值为 'all'、'SH'（上海）、'SZ'（深圳）。
    
    返回:
    dict: 过滤后的权重字典。
    """
    if exchange == 'all':
        return daily_weights
    
    filtered_weights = {}
    for index_code, weight_df in daily_weights.items():
        filtered_df = weight_df.copy()
        
        # 根据股票代码前缀过滤
        if exchange == 'SH':
            # 保留上交所股票（以'6'或'5'开头）
            for col in filtered_df.columns:
                if not (col.startswith('6') or col.startswith('5')):
                    filtered_df[col] = np.nan
        elif exchange == 'SZ':
            # 保留深交所股票（不是以'6'或'5'开头）
            for col in filtered_df.columns:
                if col.startswith('6') or col.startswith('5'):
                    filtered_df[col] = np.nan
        
        filtered_weights[index_code] = filtered_df
    
    return filtered_weights


# %%
def get_range_weight_by_depth(index_name, downscale_depth, daily_weights, index_all, index_seq, exchange='all'):
    """
    根据downscale_depth获取范围权重
    
    参数:
    index_name (str): 指数名称
    downscale_depth: 下沉深度，可以是整数或'all'
    daily_weights (dict): 日权重字典
    index_all (str): 全市场指数名称
    index_seq (list): 指数序列
    exchange (str): 交易所过滤选项
    
    返回:
    pd.DataFrame: 范围权重
    """
    # 根据交易所过滤权重
    filtered_weights = filter_weights_by_exchange(daily_weights, exchange)
    
    if downscale_depth == 'all':
        range_weight = get_complement_weights(index_name, index_all, index_seq, filtered_weights)
    elif isinstance(downscale_depth, int) and downscale_depth >= 0:
        norm_daily_weights = {index_code: normalize_daily_weights(daily_weight) 
                              for index_code, daily_weight in filtered_weights.items()}
        range_weight = get_merged_binary_weight_by_depth(norm_daily_weights, index_name, 
                                                         index_all, index_seq, downscale_depth)
    else:
        raise NotImplementedError(f"Unsupported downscale_depth: {downscale_depth}")
    
    return range_weight


def apply_range_mask_to_self_weights(range_weight, self_weight):
    """
    使用范围权重对自定义权重进行mask，只保留范围内有值且大于0的部分
    
    参数:
    range_weight (pd.DataFrame): 日级别的范围权重，需要扩展到分钟级别
    self_weight (pd.DataFrame): 分钟级别的自定义权重
    
    返回:
    pd.DataFrame: 经过mask处理的自定义权重
    """
    # 1. 将日级别范围权重扩展到分钟级别（类似 apply_norm_daily_weights_to_timeseries）
    range_weight.index = pd.to_datetime(range_weight.index)
    self_weight_dates = self_weight.index.normalize()
    
    # 将范围权重按日期扩展到高频时间戳
    expanded_range_values = range_weight.reindex(self_weight_dates).values
    range_weight_expanded = pd.DataFrame(
        expanded_range_values,
        index=self_weight.index,  # 保持原始高频时间戳索引
        columns=range_weight.columns
    )
    
    # 2. 将自定义权重对齐到范围权重的列
    self_weight_aligned = self_weight.reindex(columns=range_weight.columns, fill_value=np.nan)
    
    # 3. 创建mask：范围权重大于0的位置
    range_mask = (range_weight_expanded > 0) & range_weight_expanded.notna()
    
    # 4. 对自定义权重应用mask
    masked_self_weight = self_weight_aligned.where(range_mask, np.nan)
    
    return masked_self_weight


def get_masked_index_all_weight(index_name, downscale_depth, daily_weights, index_all, index_seq, exchange='all'):
    """
    获取masked后的index_all权重，保留原始权重值，仅对不在downscale范围内的股票进行mask
    
    Parameters:
    index_name (str): 指数名称
    downscale_depth: 下沉深度，可以是整数或'all'
    daily_weights (dict): 日权重字典
    index_all (str): 全市场指数名称
    index_seq (list): 指数序列
    exchange (str): 交易所过滤选项
    
    Returns:
    pd.DataFrame: masked后的权重，保留index_all的原始权重值
    """
    # 根据交易所过滤权重
    filtered_weights = filter_weights_by_exchange(daily_weights, exchange)
    
    # 获取index_all的原始权重
    index_all_weight = filtered_weights[index_all].copy()
    
    # 获取用于mask的范围权重（binary）
    if downscale_depth == 'all':
        # 如果是'all'，使用complement权重作为mask
        mask_weight = get_complement_weights(index_name, index_all, index_seq, filtered_weights)
    elif isinstance(downscale_depth, int) and downscale_depth >= 0:
        # 如果是整数，获取binary范围权重作为mask
        norm_daily_weights = {index_code: normalize_daily_weights(daily_weight) 
                              for index_code, daily_weight in filtered_weights.items()}
        mask_weight = get_merged_binary_weight_by_depth(norm_daily_weights, index_name, 
                                                        index_all, index_seq, downscale_depth)
    else:
        raise NotImplementedError(f"Unsupported downscale_depth: {downscale_depth}")
    
    # 创建mask：mask_weight中大于0的位置
    mask = (mask_weight > 0) & mask_weight.notna()
    
    # 对index_all_weight应用mask，保留原始权重值
    masked_weight = index_all_weight.where(mask, np.nan)
    
    return masked_weight



# %%
def avg_side(ind_sides, target_indexes, daily_weights, index_all, index_seq, downscale_depth, 
             imb_func, ts_func_with_pr, cs_func, n_workers=1, exchange='all'):
    res = {'Bid': {}, 'Ask': {}}
    norm_daily_weights = {index_code: normalize_daily_weights(daily_weight) 
                          for index_code, daily_weight in daily_weights.items()}
    iter_ = tqdm(target_indexes, desc='avg_imb by indexes') if n_workers == 1 else target_indexes
    for index_name in iter_:
        weight = get_merged_binary_weight_by_depth(norm_daily_weights, index_name, 
                                                   index_all, index_seq, downscale_depth)
        adj_sides = {side: apply_norm_daily_weights_to_timeseries(ind_sides[side], weight) for side in ind_sides}
        adj_mean_sides = {side: cs_func(adj_sides[side]) for side in adj_sides}
        for side in ('Bid', 'Ask'):
            res[side][index_name] = adj_mean_sides[side]
    return {side: pd.DataFrame(res[side]) for side in res}


# =============================================================================
# def avg_imb(ind_sides, target_indexes, daily_weights, index_all, index_seq, downscale_depth, 
#             imb_func, ts_func_with_pr, cs_func, n_workers=1):
#     res = {}
#     norm_daily_weights = {index_code: normalize_daily_weights(daily_weight) 
#                           for index_code, daily_weight in daily_weights.items()}
#     iter_ = tqdm(target_indexes, desc='avg_imb by indexes') if n_workers == 1 else target_indexes
#     for index_name in iter_:
#         weight = get_merged_binary_weight_by_depth(norm_daily_weights, index_name, 
#                                                    index_all, index_seq, downscale_depth)
#         adj_sides = {side: apply_norm_daily_weights_to_timeseries(ind_sides[side], weight) for side in ind_sides}
#         adj_mean_sides = {side: cs_func(adj_sides[side]) for side in adj_sides}
#         res[index_name] = imb_func(adj_mean_sides['Bid'], adj_mean_sides['Ask'])
#     return pd.DataFrame(res)
# 
# 
# def wavg_imb(ind_sides, target_indexes, daily_weights, index_all, index_seq, downscale_depth, 
#              imb_func, ts_func_with_pr, cs_func, n_workers=1):
#     res = {}
#     res_by_index = {}
#     iter_ = tqdm(target_indexes, desc='wavg_imb by indexes') if n_workers == 1 else target_indexes
#     if isinstance(downscale_depth, int) and downscale_depth >= 0:
#         for index_name in index_seq:
#             res_by_index[index_name] = wavg_imb_by_single_index(index_name, ind_sides, imb_func, daily_weights, cs_func)
#         for index_name in iter_:
#             downscale_indexes = get_downscale_indexes(index_name, index_seq, downscale_depth)
#             res[index_name] = pd.concat([res_by_index[downscale_index]
#                                          for downscale_index in downscale_indexes], axis=1).mean(axis=1)
#     elif isinstance(downscale_depth, str) and downscale_depth == 'all':
#         new_weights = {}
#         for index_name in iter_:
#             new_weights[index_name] = get_complement_weights(index_name, index_all, index_seq, daily_weights)
#         for index_name in iter_:
#             res[index_name] = wavg_imb_by_single_index(index_name, ind_sides, imb_func, new_weights, cs_func)
#     return pd.DataFrame(res)
# 
# 
# def imb_wavg(ind_sides, target_indexes, daily_weights, index_all, index_seq, downscale_depth, 
#              imb_func, ts_func_with_pr, cs_func, n_workers=1):
#     res = {}
#     res_by_index = {}
#     imb = imb_func(ind_sides['Bid'], ind_sides['Ask'])
#     if ts_func_with_pr is not None:
#         imb = ts_func_with_pr(imb)
#     iter_ = tqdm(target_indexes, desc='imb_wavg by indexes') if n_workers == 1 else target_indexes
#     if isinstance(downscale_depth, int) and downscale_depth >= 0:
#         for index_name in index_seq:
#             res_by_index[index_name] = imb_wavg_by_single_index(index_name, imb, daily_weights, cs_func)
#         for index_name in iter_:
#             downscale_indexes = get_downscale_indexes(index_name, index_seq, downscale_depth)
#             res[index_name] = pd.concat([res_by_index[downscale_index]
#                                        for downscale_index in downscale_indexes], axis=1).mean(axis=1)
#     elif isinstance(downscale_depth, str) and downscale_depth == 'all':
#         new_weights = {}
#         for index_name in iter_:
#             new_weights[index_name] = get_complement_weights(index_name, index_all, index_seq, daily_weights)
#         for index_name in iter_:
#             res[index_name] = imb_wavg_by_single_index(index_name, imb, new_weights, cs_func)
#     return pd.DataFrame(res)
# 
# =============================================================================

# =============================================================================
# def avg_imb(ind_sides, target_indexes, daily_weights, index_all, index_seq, downscale_depth, 
#             imb_func, ts_func_with_pr, cs_func, n_workers=1, exchange='all'):
#     # 根据交易所过滤权重
#     filtered_weights = filter_weights_by_exchange(daily_weights, exchange)
#     
#     res = {}
#     norm_daily_weights = {index_code: normalize_daily_weights(daily_weight) 
#                           for index_code, daily_weight in filtered_weights.items()}
#     
#     iter_ = tqdm(target_indexes, desc=f'avg_imb by indexes ({exchange})') if n_workers == 1 else target_indexes
#     for index_name in iter_:
#         weight = get_merged_binary_weight_by_depth(norm_daily_weights, index_name, 
#                                                    index_all, index_seq, downscale_depth)
#         adj_sides = {side: apply_norm_daily_weights_to_timeseries(ind_sides[side], weight) for side in ind_sides}
#         adj_mean_sides = {side: cs_func(adj_sides[side]) for side in adj_sides}
#         res[index_name] = imb_func(adj_mean_sides['Bid'], adj_mean_sides['Ask'])
#     return pd.DataFrame(res)
# 
# def wavg_imb(ind_sides, target_indexes, daily_weights, index_all, index_seq, downscale_depth, 
#              imb_func, ts_func_with_pr, cs_func, n_workers=1, exchange='all'):
#     # 根据交易所过滤权重
#     filtered_weights = filter_weights_by_exchange(daily_weights, exchange)
#     
#     res = {}
#     res_by_index = {}
#     iter_ = tqdm(target_indexes, desc=f'wavg_imb by indexes ({exchange})') if n_workers == 1 else target_indexes
#     
#     if isinstance(downscale_depth, int) and downscale_depth >= 0:
#         for index_name in index_seq:
#             res_by_index[index_name] = wavg_imb_by_single_index(index_name, ind_sides, imb_func, filtered_weights, cs_func)
#         for index_name in iter_:
#             downscale_indexes = get_downscale_indexes(index_name, index_seq, downscale_depth)
#             res[index_name] = pd.concat([res_by_index[downscale_index]
#                                          for downscale_index in downscale_indexes], axis=1).mean(axis=1)
#     elif isinstance(downscale_depth, str) and downscale_depth == 'all':
#         new_weights = {}
#         for index_name in iter_:
#             new_weights[index_name] = get_complement_weights(index_name, index_all, index_seq, filtered_weights)
#         for index_name in iter_:
#             res[index_name] = wavg_imb_by_single_index(index_name, ind_sides, imb_func, new_weights, cs_func)
#     
#     return pd.DataFrame(res)
# =============================================================================

def avg_imb(ind_sides, target_indexes, daily_weights, index_all, index_seq, downscale_depth, 
            imb_func, ts_func_with_pr, cs_func, n_workers=1, exchange='all'):
    # 根据交易所过滤权重
    filtered_weights = filter_weights_by_exchange(daily_weights, exchange)
    
    res = {}
    norm_daily_weights = {index_code: normalize_daily_weights(daily_weight) 
                          for index_code, daily_weight in filtered_weights.items()}
    
    iter_ = tqdm(target_indexes, desc=f'avg_imb by indexes ({exchange})') if n_workers == 1 else target_indexes
    for index_name in iter_:
        weight = get_merged_binary_weight_by_depth(norm_daily_weights, index_name, 
                                                   index_all, index_seq, downscale_depth)
        adj_sides = {side: apply_norm_daily_weights_to_timeseries(ind_sides[side], weight) for side in ind_sides}
        adj_mean_sides = {side: cs_func(adj_sides[side]) for side in adj_sides}
        
        # 新增: 应用 ts_func_with_pr 到每个 side
        if ts_func_with_pr is not None:
            adj_mean_sides = {side: ts_func_with_pr(adj_mean_sides[side]) for side in adj_mean_sides}
            
        res[index_name] = imb_func(adj_mean_sides['Bid'], adj_mean_sides['Ask'])
    return pd.DataFrame(res)

def wavg_imb(ind_sides, target_indexes, daily_weights, index_all, index_seq, downscale_depth, 
             imb_func, ts_func_with_pr, cs_func, n_workers=1, exchange='all'):
    # 根据交易所过滤权重
    filtered_weights = filter_weights_by_exchange(daily_weights, exchange)
    
    res = {}
    res_by_index = {}
    iter_ = tqdm(target_indexes, desc=f'wavg_imb by indexes ({exchange})') if n_workers == 1 else target_indexes
    
    if isinstance(downscale_depth, int) and downscale_depth >= 0:
        for index_name in index_seq:
            # 传入 ts_func_with_pr 到 wavg_imb_by_single_index
            res_by_index[index_name] = wavg_imb_by_single_index(index_name, ind_sides, imb_func, filtered_weights, cs_func, ts_func_with_pr)
        for index_name in iter_:
            downscale_indexes = get_downscale_indexes(index_name, index_seq, downscale_depth)
            res[index_name] = pd.concat([res_by_index[downscale_index]
                                         for downscale_index in downscale_indexes], axis=1).mean(axis=1)
    elif isinstance(downscale_depth, str) and downscale_depth == 'all':
        new_weights = {}
        for index_name in iter_:
            new_weights[index_name] = get_complement_weights(index_name, index_all, index_seq, filtered_weights)
        for index_name in iter_:
            # 传入 ts_func_with_pr 到 wavg_imb_by_single_index
            res[index_name] = wavg_imb_by_single_index(index_name, ind_sides, imb_func, new_weights, cs_func, ts_func_with_pr)
    
    return pd.DataFrame(res)

def imb_wavg(ind_sides, target_indexes, daily_weights, index_all, index_seq, downscale_depth, 
             imb_func, ts_func_with_pr, cs_func, n_workers=1, exchange='all'):
    # 根据交易所过滤权重
    filtered_weights = filter_weights_by_exchange(daily_weights, exchange)
    
    res = {}
    res_by_index = {}
    imb = imb_func(ind_sides['Bid'], ind_sides['Ask'])
    if ts_func_with_pr is not None:
        imb = ts_func_with_pr(imb)
    
    iter_ = tqdm(target_indexes, desc=f'imb_wavg by indexes ({exchange})') if n_workers == 1 else target_indexes
    
    if isinstance(downscale_depth, int) and downscale_depth >= 0:
        for index_name in index_seq:
            res_by_index[index_name] = imb_wavg_by_single_index(index_name, imb, filtered_weights, cs_func)
        for index_name in iter_:
            downscale_indexes = get_downscale_indexes(index_name, index_seq, downscale_depth)
            res[index_name] = pd.concat([res_by_index[downscale_index]
                                         for downscale_index in downscale_indexes], axis=1).mean(axis=1)
    elif isinstance(downscale_depth, str) and downscale_depth == 'all':
        new_weights = {}
        for index_name in iter_:
            new_weights[index_name] = get_complement_weights(index_name, index_all, index_seq, filtered_weights)
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


def avg_scale_imb(ind_sides, target_indexes, daily_weights, index_all, index_seq, downscale_depth, 
            imb_func, ts_func_with_pr, cs_func, n_workers=1):
    res = {}
    norm_daily_weights = {index_code: normalize_daily_weights(daily_weight) 
                          for index_code, daily_weight in daily_weights.items()}
    iter_ = tqdm(target_indexes, desc='avg_scale_imb by indexes') if n_workers == 1 else target_indexes
    for index_name in iter_:
        weight = get_merged_binary_weight_by_depth(norm_daily_weights, index_name, 
                                                   index_all, index_seq, downscale_depth)
        adj_sides = {side: apply_norm_daily_weights_to_timeseries(ind_sides[side], weight) for side in ind_sides}
        adj_mean_sides = {side: cs_func(adj_sides[side]) for side in adj_sides}
        adj_mean_scaled_sides = {side: ts_func_with_pr(adj_mean_sides[side]) for side in adj_mean_sides}
        res[index_name] = imb_func(adj_mean_scaled_sides['Bid'], adj_mean_scaled_sides['Ask'])
    return pd.DataFrame(res)


# %%
def selfwavg_imb(ind_sides, target_indexes, daily_weights, index_all, index_seq, downscale_depth, 
                 imb_func, ts_func_with_pr, cs_func, n_workers=1, exchange='all', selfdefined_weights=None):
    
    res = {}
    iter_ = tqdm(target_indexes, desc=f'selfwavg_imb by indexes ({exchange})') if n_workers == 1 else target_indexes

    for index_name in iter_:
        # 检查是否有downscale_depth参数
        if downscale_depth is not None:
            # 新做法：使用范围权重进行mask
            # 1. 获取范围权重
            range_weight = get_range_weight_by_depth(index_name, downscale_depth, daily_weights, 
                                                     index_all, index_seq, exchange)
            
            # 2. 使用范围权重对自定义权重进行mask
            self_weight = selfdefined_weights[index_name]
            masked_self_weight = apply_range_mask_to_self_weights(range_weight, self_weight)
            
            # 3. 使用masked后的自定义权重进行聚合
            res[index_name] = selfwavg_imb_by_single_index(index_name, ind_sides, imb_func, 
                                                           {index_name: masked_self_weight}, 
                                                           cs_func, ts_func_with_pr)
        else:
            # 原来的做法：直接使用自定义权重计算
            res[index_name] = selfwavg_imb_by_single_index(index_name, ind_sides, imb_func, 
                                                           selfdefined_weights, cs_func, ts_func_with_pr)
    
    return pd.DataFrame(res)


def selfavg_imb(ind_sides, target_indexes, daily_weights, index_all, index_seq, downscale_depth, 
                imb_func, ts_func_with_pr, cs_func, n_workers=1, exchange='all', selfdefined_weights=None):
    
    res = {}
    iter_ = tqdm(target_indexes, desc=f'selfavg_imb by indexes ({exchange})') if n_workers == 1 else target_indexes

    for index_name in iter_:
        # 检查是否有downscale_depth参数
        if downscale_depth is not None:
            # 新做法：使用范围权重进行mask
            # 1. 获取范围权重
            range_weight = get_range_weight_by_depth(index_name, downscale_depth, daily_weights, 
                                                     index_all, index_seq, exchange)
            
            # 2. 使用范围权重对自定义权重进行mask
            self_weight = selfdefined_weights[index_name]
            masked_self_weight = apply_range_mask_to_self_weights(range_weight, self_weight)
            
            # 3. 对masked后的权重进行归一化处理
            normalized_masked_weight = normalize_daily_weights(masked_self_weight)
            
            # 4. 使用normalized masked权重进行聚合
            res[index_name] = selfwavg_imb_by_single_index(index_name, ind_sides, imb_func, 
                                                           {index_name: normalized_masked_weight}, 
                                                           cs_func, ts_func_with_pr)
        else:
            # 原来的做法：先归一化再计算
            self_weight = selfdefined_weights[index_name]
            normalized_self_weight = normalize_daily_weights(self_weight)
            res[index_name] = selfwavg_imb_by_single_index(index_name, ind_sides, imb_func, 
                                                           {index_name: normalized_self_weight}, 
                                                           cs_func, ts_func_with_pr)
    
    return pd.DataFrame(res)


# %%
def subset_wavg_imb(ind_sides, target_indexes, daily_weights, index_all, index_seq, downscale_depth, 
                    imb_func, ts_func_with_pr, cs_func, n_workers=1, exchange='all', selfdefined_weights=None):
    """
    计算子集加权的imbalance，其中：
    - 分子：使用selfdefined_weights和daily_weights共同过滤的normalized_masked_weight聚合的bid和ask差值
    - 分母：使用仅daily_weights聚合的bid和ask总和
    
    这样可以观察在指数权重内所有股票量做归一化的情况下，
    你筛选出的特定股票子集的买卖差异，当子集权重较低时该值会被压低。
    
    参数:
    ind_sides: 包含'Bid'和'Ask'的字典，每个值为DataFrame
    target_indexes: 目标指数列表
    daily_weights: 日权重字典
    index_all: 全市场指数名称
    index_seq: 指数序列
    downscale_depth: 下沉深度
    imb_func: imbalance计算函数
    ts_func_with_pr: 时序函数（带参数）
    cs_func: 截面聚合函数
    n_workers: 工作进程数
    exchange: 交易所过滤选项
    selfdefined_weights: 自定义权重字典（可以是任何你筛选的股票权重）
    
    返回:
    pd.DataFrame: 子集加权的imbalance结果
    """
    
    res = {}
    iter_ = tqdm(target_indexes, desc=f'subset_wavg_imb by indexes ({exchange})') if n_workers == 1 else target_indexes

    for index_name in iter_:
        # 1. 获取范围权重（用于确定股票范围）
        range_weight = get_range_weight_by_depth(index_name, downscale_depth, daily_weights, 
                                                 index_all, index_seq, exchange)
        
        # 2. 准备分子权重：使用selfdefined_weights和daily_weights共同过滤
        self_weight = selfdefined_weights[index_name]
        masked_self_weight = apply_range_mask_to_self_weights(range_weight, self_weight)
        normalized_masked_weight = normalize_daily_weights(masked_self_weight)
        
        # 3. 准备分母权重：仅使用daily_weights
        filtered_weights = filter_weights_by_exchange(daily_weights, exchange)
        denom_weight = get_range_weight_by_depth(index_name, downscale_depth, filtered_weights, 
                                                index_all, index_seq, exchange)
        
        # 4. 分别计算分子和分母的聚合值
        # 分子：使用normalized_masked_weight
        numer_adj_sides = {side: apply_minute_weights_to_timeseries(ind_sides[side], normalized_masked_weight) 
                          for side in ind_sides}
        numer_adj_mean_sides = {side: cs_func(numer_adj_sides[side]) for side in numer_adj_sides}
        
        # 分母：使用denom_weight
        denom_adj_sides = {side: apply_norm_daily_weights_to_timeseries(ind_sides[side], denom_weight) 
                          for side in ind_sides}
        denom_adj_mean_sides = {side: cs_func(denom_adj_sides[side]) for side in denom_adj_sides}
        
        # 5. 应用时序函数（如果有）
        if ts_func_with_pr is not None:
            numer_adj_mean_sides = {side: ts_func_with_pr(numer_adj_mean_sides[side]) 
                                   for side in numer_adj_mean_sides}
            denom_adj_mean_sides = {side: ts_func_with_pr(denom_adj_mean_sides[side]) 
                                   for side in denom_adj_mean_sides}
        
        # 6. 计算子集加权的imbalance
        res[index_name] = imb_func(numer_adj_mean_sides['Bid'], numer_adj_mean_sides['Ask'],
                                  denom_adj_mean_sides['Bid'], denom_adj_mean_sides['Ask'])
    
    return pd.DataFrame(res)


# %%
def norm_wavg_imb(ind_sides, target_indexes, daily_weights, index_all, index_seq, downscale_depth, 
                  imb_func, ts_func_with_pr, cs_func, n_workers=1, exchange='all'):
    """
    新的聚合函数：
    1. 使用downscale_depth在index_all上选择指定范围，mask掉不在范围内的股票，但保留index_all的原始权重值
    2. 对输入的bid和ask因子先做归一化：bid/(bid+ask), ask/(bid+ask)
    3. 对归一化后的因子分别乘以masked的index_all权重，然后做cs_func
    4. 最后对两个结果计算imbalance
    
    Parameters:
    ind_sides (dict): 包含'Bid'和'Ask'的字典，值为DataFrame
    target_indexes (list): 目标指数列表
    daily_weights (dict): 日权重字典
    index_all (str): 全市场指数名称
    index_seq (list): 指数序列
    downscale_depth: 下沉深度
    imb_func: imbalance计算函数
    ts_func_with_pr: 时序函数（如果需要）
    cs_func: 截面函数
    n_workers (int): 工作进程数
    exchange (str): 交易所选择
    
    Returns:
    pd.DataFrame: 计算结果
    """
    res = {}
    iter_ = tqdm(target_indexes, desc=f'norm_wavg_imb by indexes ({exchange})') if n_workers == 1 else target_indexes
    
    for index_name in iter_:
        # 1. 获取masked后的index_all权重（保留原始权重值）
        masked_weight = get_masked_index_all_weight(index_name, downscale_depth, daily_weights, 
                                                   index_all, index_seq, exchange)
        
        # 2. 对输入因子进行归一化处理：a/(a+b), b/(a+b)
        bid_factor = ind_sides['Bid']
        ask_factor = ind_sides['Ask']
        
        # 计算 bid + ask，避免除零
        sum_factors = bid_factor + ask_factor
        sum_factors = sum_factors.replace(0, np.nan)  # 将0替换为NaN避免除零
        
        # 归一化
        norm_bid = bid_factor / sum_factors
        norm_ask = ask_factor / sum_factors
        
        # 3. 应用masked权重到归一化后的因子
        weighted_norm_bid = apply_daily_weights_to_timeseries(norm_bid, masked_weight)
        weighted_norm_ask = apply_daily_weights_to_timeseries(norm_ask, masked_weight)
        
        # 4. 对加权后的因子分别做cross-sectional function
        cs_bid = cs_func(weighted_norm_bid)
        cs_ask = cs_func(weighted_norm_ask)
        
        # 5. 应用时序函数（如果有）
        if ts_func_with_pr is not None:
            cs_bid = ts_func_with_pr(cs_bid)
            cs_ask = ts_func_with_pr(cs_ask)
        
        # 6. 计算imbalance
        res[index_name] = imb_func(cs_bid, cs_ask)
    
    return pd.DataFrame(res)