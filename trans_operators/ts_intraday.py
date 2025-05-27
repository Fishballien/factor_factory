# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 17:58:03 2025

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
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


# %%
def OAD(df, reference_time='0930', columns=None):
    """
    Calculate differences between each time point and a reference time for specified columns.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with datetime index
    reference_time : str, default '09:30:00'
        Reference time in format 'HH:MM:SS' to compare against
    columns : list or None, default None
        List of columns to calculate differences for. If None, uses all columns in df.
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing only the difference columns
    """
    
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Ensure index is datetime format
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # If no columns specified, use all columns in the DataFrame
    if columns is None:
        columns = df.columns.tolist()
    
    # Convert reference time string to time object
    ref_time = pd.to_datetime(f'{reference_time[:2]}:{reference_time[2:]}').time()
    
    # Create reference columns for each specified column
    for col in columns:
        # Create a new column that only keeps values at reference time
        ref_col_name = f"{col}_{reference_time.replace(':', '')}"
        df[ref_col_name] = np.where(df.index.time == ref_time, df[col], np.nan)
    
    # Group by date and forward fill the reference values
    df = df.groupby(df.index.date).apply(lambda x: x.fillna(method='ffill'))
    
    # Reset index (remove multi-level index)
    df = df.reset_index(level=0, drop=True)
    
    # Calculate differences
    diff_columns = []
    for col in columns:
        ref_col_name = f"{col}_{reference_time.replace(':', '')}"
        diff_col_name = f"{col}_diff"
        df[diff_col_name] = df[col] - df[ref_col_name]
        diff_columns.append(diff_col_name)
    
    # Return only the difference columns
    return df[diff_columns]

# Example usage:
# Assuming 'df' is your DataFrame with datetime index
# result = calculate_time_differences(df, reference_time='09:31:00', 
#                                    columns=['call_oi_sum', 'put_oi_sum', 'pc', 'oi_imb01'])


# %% ma
def intraSma(data, window: int):
    """
    计算日内简单滑动窗口均值，确保每天的计算仅使用当天的数据。
    
    Args:
        data: 时间序列数据，可以是DataFrame或Series，index为时间戳。
        window (int): 滑动窗口的大小。
        
    Returns:
        与输入相同类型的日内滑动均值结果，结构与输入一致。
    """
    # 判断输入是DataFrame还是Series
    is_series = isinstance(data, pd.Series)
    
    # 如果是Series，转换为DataFrame处理，方便统一逻辑
    if is_series:
        df = data.to_frame()
    else:
        df = data.copy()
    
    # 创建一个与输入相同结构的结果DataFrame
    result = pd.DataFrame(index=df.index, columns=df.columns)
    
    # 根据日期对数据进行分组
    grouped = df.groupby(df.index.date)
    
    # 对每一天的数据单独计算滑动平均
    for date, group in grouped:
        # 对当天的数据计算滑动平均
        day_result = group.rolling(window=window, min_periods=1).mean()
        
        # 将当天的结果填入总结果中
        result.loc[group.index] = day_result
    
    # 如果输入是Series，则返回Series，否则返回DataFrame
    if is_series:
        return result.iloc[:, 0]
    else:
        return result

def intraEwma(data, span: int):
    """
    计算日内指数加权移动平均(EWMA)，确保每天的计算仅使用当天的数据。
    
    Args:
        data: 时间序列数据，可以是DataFrame或Series，index为时间戳。
        span (int): 指数加权的周期数，类似于半衰期。
        
    Returns:
        与输入相同类型的日内指数加权移动平均结果，结构与输入一致。
    """
    # 判断输入是DataFrame还是Series
    is_series = isinstance(data, pd.Series)
    
    # 如果是Series，转换为DataFrame处理，方便统一逻辑
    if is_series:
        df = data.to_frame()
    else:
        df = data.copy()
    
    # 创建一个与输入相同结构的结果DataFrame
    result = pd.DataFrame(index=df.index, columns=df.columns)
    
    # 根据日期对数据进行分组
    grouped = df.groupby(df.index.date)
    
    # 对每一天的数据单独计算EWMA
    for date, group in grouped:
        # 对当天的数据计算EWMA，adjust=True确保更准确的指数权重
        day_result = group.ewm(span=span, min_periods=1, adjust=True).mean()
        
        # 将当天的结果填入总结果中
        result.loc[group.index] = day_result
    
    # 如果输入是Series，则返回Series，否则返回DataFrame
    if is_series:
        return result.iloc[:, 0]
    else:
        return result
    
    
# %%
def intraSum(data, window: int):
    """
    计算日内滑动窗口累计求和，确保每天的计算仅使用当天的数据。
    
    Args:
        data: 时间序列数据，可以是DataFrame或Series，index为时间戳。
        window (int): 滑动窗口的大小。
        
    Returns:
        与输入相同类型的日内滑动求和结果，结构与输入一致。
    """
    # 判断输入是DataFrame还是Series
    is_series = isinstance(data, pd.Series)
    
    # 如果是Series，转换为DataFrame处理，方便统一逻辑
    if is_series:
        df = data.to_frame()
    else:
        df = data.copy()
    
    # 创建一个与输入相同结构的结果DataFrame
    result = pd.DataFrame(index=df.index, columns=df.columns)
    
    # 根据日期对数据进行分组
    grouped = df.groupby(df.index.date)
    
    # 对每一天的数据单独计算滑动求和
    for date, group in grouped:
        # 对当天的数据计算滑动求和
        day_result = group.rolling(window=window, min_periods=1).sum()
        
        # 将当天的结果填入总结果中
        result.loc[group.index] = day_result
    
    # 如果输入是Series，则返回Series，否则返回DataFrame
    if is_series:
        return result.iloc[:, 0]
    else:
        return result
    
    
# %%
def process_intraCumSum_block(df_block, block_idx):
    """
    处理 intraCumSum 的单个数据块
    """
    # 创建一个与输入相同结构的结果DataFrame
    result = pd.DataFrame(index=df_block.index, columns=df_block.columns)
    
    # 根据日期对数据进行分组
    grouped = df_block.groupby(df_block.index.date)
    
    # 对每一天的数据单独计算累计求和
    for date, group in grouped:
        # 对当天的数据计算累计求和
        day_result = group.cumsum()
        
        # 将当天的结果填入总结果中
        result.loc[group.index] = day_result
    
    return block_idx, result


def intraCumSum_parallel(data, n_jobs: int = 150, block_size: int = 5):
    """
    intraCumSum 的并行加速版本
    
    Args:
        data: 时间序列数据，可以是DataFrame或Series，index为时间戳。
        n_jobs (int): 并行进程数，默认值为 150。
        block_size (int): 每个数据块的列数，默认值为 5。
        
    Returns:
        与输入相同类型的日内累计求和结果，结构与输入一致。
    """
    # 判断输入是DataFrame还是Series
    is_series = isinstance(data, pd.Series)
    
    # 如果是Series，转换为DataFrame处理，方便统一逻辑
    if is_series:
        df = data.to_frame()
    else:
        df = data.copy()
    
    # 将数据按列分块
    col_blocks = [df.columns[i:i+block_size] for i in range(0, len(df.columns), block_size)]
    result = pd.DataFrame(index=df.index, columns=df.columns)
    total_blocks = len(col_blocks)

    print(f"[intraCumSum_parallel] Launching {total_blocks} blocks with {n_jobs} processes...")

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        future_to_idx = {}
        for block_idx, cols in enumerate(col_blocks):
            df_block = df[cols]
            future = executor.submit(process_intraCumSum_block, df_block, block_idx)
            future_to_idx[future] = (block_idx, cols)

        with tqdm(total=total_blocks, desc="intraCumSum Progress") as pbar:
            for future in as_completed(future_to_idx):
                block_idx, cols = future_to_idx[future]
                _, block_result = future.result()
                for col in cols:
                    result[col] = block_result[col]
                pbar.update(1)

    print("[intraCumSum_parallel] All blocks completed.")
    
    # 如果输入是Series，则返回Series，否则返回DataFrame
    if is_series:
        return result.iloc[:, 0]
    else:
        return result


def intraCumSum(data):
    """
    计算日内累计求和，确保每天的计算仅使用当天的数据，每天重新开始累积。
    
    Args:
        data: 时间序列数据，可以是DataFrame或Series，index为时间戳。
        
    Returns:
        与输入相同类型的日内累计求和结果，结构与输入一致。
    """
    # 判断输入是DataFrame还是Series
    is_series = isinstance(data, pd.Series)
    
    # 如果是Series，转换为DataFrame处理，方便统一逻辑
    if is_series:
        df = data.to_frame()
    else:
        df = data.copy()
    
    # 创建一个与输入相同结构的结果DataFrame
    result = pd.DataFrame(index=df.index, columns=df.columns)
    
    # 根据日期对数据进行分组
    grouped = df.groupby(df.index.date)
    
    # 对每一天的数据单独计算累计求和
    for date, group in grouped:
        # 对当天的数据计算累计求和
        day_result = group.cumsum()
        
        # 将当天的结果填入总结果中
        result.loc[group.index] = day_result
    
    # 如果输入是Series，则返回Series，否则返回DataFrame
    if is_series:
        return result.iloc[:, 0]
    else:
        return result