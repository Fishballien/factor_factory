# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 21:22:28 2025

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


from trans_operators.ts import rollingMinuteQuantileScale, rollingRemoveIntradayEffect, zscore


# %%
def rollingMinuteQuantileScaleLatest(
    df: pd.DataFrame,
    window: str = "30d",
    min_periods: int = 1,
    quantile: float = 0.05
) -> pd.Series:
    """
    仅计算最新一行的分位数归一化，按指定窗口进行计算。

    参数:
        df (pd.DataFrame): 时间序列数据，index 必须为 DatetimeIndex。
        window (str, optional): 滚动窗口大小，支持 Pandas 时间窗口（如 "30d" 表示30天）。默认值为 "30d"。
        min_periods (int, optional): 窗口中要求的最少观察数，默认值为 1。
        quantile (float, optional): 用于归一化的分位数，默认值为 0.05。

    返回:
        pd.Series: 最新一行归一化后的值。
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame 的 index 必须是 DatetimeIndex 类型")

    if df.empty:
        raise ValueError("DataFrame 不能为空")

    # 获取最新一行的时间点和数据
    latest_row = df.iloc[-1]
    latest_time = df.index[-1]
    latest_minute = latest_time.time()

    # 筛选出对应窗口范围的数据
    start_time = latest_time - pd.Timedelta(window)
    window_data = df[(df.index > start_time) & (df.index <= latest_time)]

    # 筛选出对应分钟的历史数据
    mask = window_data.index.time == latest_minute
    group = window_data[mask]

    if group.empty:
        raise ValueError("指定窗口内没有足够的数据计算分位数")

    # 计算窗口的分位数
    lower_bound = group.quantile(quantile)
    upper_bound = group.quantile(1 - quantile)

    # 归一化计算
    scaled = (latest_row - lower_bound) / (upper_bound - lower_bound).replace(0, np.nan)

    # 裁剪结果在 0 和 1 之间
    return scaled.clip(lower=0, upper=1)


def rollingRemoveIntradayEffectLatest(
    df: pd.DataFrame, 
    window: str = '30d', 
) -> pd.Series:
    """
    仅对最新一行进行滚动去除因子中的日内效应计算，每个时间点回看窗口内的均值。

    参数：
        df (pd.DataFrame): 包含时间序列的DataFrame，index为DatetimeIndex。
        window (int): 回看滚动窗口大小（天数）。
        time_col (str): 每天的具体时间点列名（如 '09:30'）。

    返回：
        pd.Series: 去除滚动窗口内日内效应后的最新一行数据。
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame的index必须是DatetimeIndex类型")

    if df.empty:
        raise ValueError("DataFrame不能为空")

    # 获取最新一行的时间点和数据
    latest_row = df.iloc[-1]
    latest_time = df.index[-1]
    latest_minute = latest_time.time()

    # 筛选窗口范围内的数据
    start_time = latest_time - pd.Timedelta(window)
    window_data = df[(df.index > start_time) & (df.index <= latest_time)]

    # 筛选与最新时间点相同分钟的数据
    mask = window_data.index.time == latest_minute
    group = window_data[mask]

    if group.empty:
        raise ValueError("指定窗口内没有足够的数据计算日内均值")

    # 计算滚动窗口内的均值
    rolling_mean = group.mean()

    # 去除均值的影响
    result = latest_row - rolling_mean

    return result


def zscoreLatest(df, period_str):
    """
    仅计算最新一行的 z-score，即 (当前值 - 过去一段时间的均值) / 过去一段时间的标准差。

    参数：
    df (pd.DataFrame): datetime 为 index 的数据框。
    period_str (str): 过去时间段的长度，可以是小时（如 '4h'）或天（如 '3d'）。

    返回：
    pd.Series: 最新一行的 z-score 值。
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame 的 index 必须是 DatetimeIndex 类型")

    if df.empty:
        raise ValueError("DataFrame 不能为空")

    # 获取最新一行的时间点和数据
    latest_row = df.iloc[-1]
    latest_time = df.index[-1]

    # 筛选窗口范围内的数据
    start_time = latest_time - pd.Timedelta(period_str)
    window_data = df[(df.index > start_time) & (df.index <= latest_time)]

    if window_data.empty:
        raise ValueError("指定窗口内没有足够的数据计算 z-score")

    # 计算均值和标准差
    mean = window_data.mean()
    std = window_data.std()

    # 计算 z-score
    zscore = (latest_row - mean) / std

    return zscore



# %%
path = 'D:/CNIndexFutures/timeseries/lob_indicators/sample_data/batch10/ValueTimeDecayOrderAmount_p1.0_v40000_d0.1-wavg_imb04-org.parquet'


# %%
factor_org = pd.read_parquet(path)


# %%
# factor_trans = rollingMinuteQuantileScale(factor_org.iloc[:-129])
# factor_latest = rollingMinuteQuantileScaleLatest(factor_org.iloc[:-129])


# %%
# factor_trans = rollingRemoveIntradayEffect(factor_org.iloc[:-129])
# factor_latest = rollingRemoveIntradayEffectLatest(factor_org.iloc[:-129])


# %%
factor_trans = zscore(factor_org.iloc[:-129], '30d')
factor_latest = zscoreLatest(factor_org.iloc[:-129], '30d')