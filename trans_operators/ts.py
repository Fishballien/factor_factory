 # -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 16:52:45 2024

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
from numba import jit
from numba import njit, prange


from utils.timeutils import get_num_of_bars


# %%
def dod(df, n):
    """
    计算每个时间戳上的值为当前时间戳数据除以前n天同一时间戳的数据。

    参数：
    df (pd.DataFrame): datetime为index的数据框。
    n (int): 间隔天数。

    返回：
    pd.DataFrame: 计算后的数据框。
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame的index必须是DatetimeIndex类型")
    
    result = (df / df.shift(n, freq='D')).reindex(df.index)
    
    return result


def dod_three_minutes(df, n):
    """
    计算每个时间戳上的值为当前时间戳数据除以前一天相同时间前后共三分钟的均值。

    参数：
    df (pd.DataFrame): datetime为index的数据框。

    返回：
    pd.DataFrame: 计算后的数据框。
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame的index必须是DatetimeIndex类型")
    
    # 获取前一天相同时间前后共三分钟的数据
    prev_day = df.shift(n, freq='D')
    offsets = [-1, 0, 1]
    
    # 计算前一天相同时间前后共三分钟的均值
    prev_mean = prev_day.copy()
    prev_mean_values = []
    for offset in offsets:
        prev_mean_values.append(prev_day.shift(offset))
    prev_mean = sum(prev_mean_values) / len(prev_mean_values)
    
    # 计算比值
    result = df / prev_mean
    result = result.reindex(df.index)
    
    return result


def zscore(df, period_str):
    """
    计算每个时间戳上的值为 (当前值 - 过去一段时间的均值) / 过去一段时间的标准差。

    参数：
    df (pd.DataFrame): datetime为index的数据框。
    period_str (str): 过去时间段的长度，可以是小时（如'4h'）或天（如'3d'）。

    返回：
    pd.DataFrame: 计算后的数据框。
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame的index必须是DatetimeIndex类型")
    
    mean = df.rolling(window=period_str, min_periods=1).mean()
    std = df.rolling(window=period_str, min_periods=1).std()
    result = (df - mean) / std

    return result


def zscore_fxwd(df, period_str, freq='1min'):
    """
    计算每个时间戳上的值为 (当前值 - 过去一段时间的均值) / 过去一段时间的标准差。
    当标准差为 0 时，返回 NaN。

    参数：
    df (pd.DataFrame): datetime 为 index 的数据框。
    period_str (str): 过去时间段的长度，如 "4h"、"3d2h30min" 等。
    freq (str): 数据的时间间隔，如 "1min"、"5min"、"10s"。

    返回：
    pd.DataFrame: 计算后的数据框。
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame的index必须是DatetimeIndex类型")
    
    # 计算窗口大小
    window_size = get_num_of_bars(period_str, freq)

    if window_size < 1:
        raise ValueError("计算的窗口大小小于 1，检查 period_str 和 freq 是否匹配")

    mean = df.rolling(window=window_size, min_periods=1).mean()
    std = df.rolling(window=window_size, min_periods=1).std()
    
    # 防止除以零：将标准差为0的地方替换为NaN
    std_safe = std.replace(0, np.nan)
    
    result = (df - mean) / std_safe

    return result


def zsc(df, period_str, freq='1min'):
    """
    计算每个时间戳上的值为 (当前值 - 过去一段时间的均值) / 过去一段时间的标准差。
    当标准差为 0 时，返回 NaN。

    参数：
    df (pd.DataFrame): datetime 为 index 的数据框。
    period_str (str): 过去时间段的长度，如 "4h"、"3d2h30min" 等。
    freq (str): 数据的时间间隔，如 "1min"、"5min"、"10s"。

    返回：
    pd.DataFrame: 计算后的数据框。
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame的index必须是DatetimeIndex类型")
    
    # 计算窗口大小
    window_size = get_num_of_bars(period_str, freq)

    if window_size < 1:
        raise ValueError("计算的窗口大小小于 1，检查 period_str 和 freq 是否匹配")

    mean = df.rolling(window=window_size, min_periods=1).mean()
    std = df.rolling(window=window_size, min_periods=1).std()
    
    # 防止除以零：将标准差为0的地方替换为NaN
    std_safe = std.replace(0, np.nan)
    
    result = (df - mean) / std_safe

    return result


# %%
# =============================================================================
# def slope(df: pd.DataFrame, window: int | str) -> pd.DataFrame:
#     """
#     对 DataFrame 中每列计算滑动窗口斜率（线性拟合的斜率），支持时间窗口。
#     
#     Args:
#         df (pd.DataFrame): 时间序列数据（支持多列）。
#         window (int | str): 滑动窗口大小，可以是整数或字符串时间窗口（如 '5min'）。
#         
#     Returns:
#         pd.DataFrame: 每列的滚动窗口斜率，结构与输入一致。
#     """
#     def compute_slope(y):
#         # 动态生成 x 序列
#         n = len(y)
#         if n < 2:  # 如果窗口大小不足，返回 NaN
#             return np.nan
#         x = np.arange(n)
#         x_mean = x.mean()
#         x_squared_mean = (x**2).mean()
#         denominator = x_squared_mean - x_mean**2
#         
#         # 计算斜率
#         y_mean = y.mean()
#         numerator = np.dot(y - y_mean, x - x_mean)
#         return numerator / denominator
# 
#     # 应用滚动窗口计算
#     slopes = df.rolling(window=window).apply(compute_slope, raw=False)
#     return slopes
# =============================================================================


@jit(nopython=True)
def compute_slope(y):
    """
    使用 Numba 加速的斜率计算。
    
    Args:
        y (np.ndarray): 窗口内的数值。
        
    Returns:
        float: 计算得到的斜率值。
    """
    n = len(y)
    if n < 2:  # 如果窗口大小不足，返回 NaN
        return np.nan
    
    x = np.arange(n)
    x_mean = x.mean()
    y_mean = y.mean()
    x_squared_mean = (x**2).mean()
    denominator = x_squared_mean - x_mean**2
    numerator = np.dot(y - y_mean, x - x_mean)
    return numerator / denominator


def slope(df: pd.DataFrame, window: int | str) -> pd.DataFrame:
    """
    对 DataFrame 中每列计算滑动窗口斜率（线性拟合的斜率），支持时间窗口，并使用 Numba 加速。
    
    Args:
        df (pd.DataFrame): 时间序列数据（支持多列）。
        window (int | str): 滑动窗口大小，可以是整数或字符串时间窗口（如 '5min'）。
        
    Returns:
        pd.DataFrame: 每列的滚动窗口斜率，结构与输入一致。
    """
    # 应用滚动窗口计算
    slopes = df.rolling(window=window, min_periods=1).apply(compute_slope, raw=True)
    return slopes


# %% f
def lowFreqEnergyRatio(df: pd.DataFrame, window, cutoff_freq: float, sampling_interval: float = 1) -> pd.DataFrame:
    """
    对 DataFrame 中每列计算滚动窗口的低频能量占比。
    
    Args:
        df (pd.DataFrame): 时间序列数据（支持多列）。
        window_size (int): 滚动窗口的长度（数据点数）。
        cutoff_freq (float): 截止频率，保留低于该频率的分量（单位：cycles per minute）。
        sampling_interval (float): 采样间隔（单位：分钟，默认1分钟）。
        
    Returns:
        pd.DataFrame: 滚动窗口的低频能量占比，结构与输入一致。
    """
    def compute_energy_ratio(window_data):
        # 计算傅里叶变换
        fft_result = np.fft.fft(window_data)
        freqs = np.fft.fftfreq(len(window_data), d=sampling_interval)
        
        # 计算总能量和低频能量
        total_energy = np.sum(np.abs(fft_result)**2)
        low_freq_energy = np.sum(np.abs(fft_result[np.abs(freqs) <= cutoff_freq])**2)
        
        # 返回低频能量占比
        return low_freq_energy / total_energy
    
    # 对 DataFrame 每列应用滚动计算
    result = df.rolling(window=window, min_periods=1).apply(compute_energy_ratio, raw=True)
    return result


def freq_to_minutes(cutoff_freq: float) -> str:
    """
    将截止频率 (cutoff_freq) 转换为实际分钟数，返回格式为 xxmin。
    
    Args:
        cutoff_freq (float): 截止频率（单位：周期每分钟，cycles per minute）。
        
    Returns:
        str: 实际分钟数的字符串表示，格式为 "xxmin"。
    """
    if cutoff_freq <= 0:
        raise ValueError("cutoff_freq must be a positive value.")
    
    # 计算对应的分钟数
    minutes = round(1 / cutoff_freq)
    
    # 返回格式化字符串
    return f"{minutes}min"


def lowFreqEnergyRatio_x_slope(data, window, cutoff_freq, sampling_interval=1):
    energy_ratio = lowFreqEnergyRatio(data, window, cutoff_freq, sampling_interval)
    slp = slope(data, freq_to_minutes(cutoff_freq))
    return energy_ratio * slp


def lowFreqEnergyRatio_x_slpsign(data, window, cutoff_freq, sampling_interval=1):
    energy_ratio = lowFreqEnergyRatio(data, window, cutoff_freq, sampling_interval)
    slp = slope(data, freq_to_minutes(cutoff_freq))
    slp_sign = slp.apply(np.sign)  # 使用 numpy.sign 对每列计算符号
    return energy_ratio * slp_sign


def phaseTimesMagnitude(df: pd.DataFrame, window, cutoff_freq: float, sampling_interval: float = 1) -> pd.DataFrame:
    """
    对 DataFrame 中每列计算滚动窗口的低频相位 * 幅值结果。
    
    Args:
        df (pd.DataFrame): 时间序列数据（支持多列）。
        window (int | str): 滚动窗口的长度（数据点数或时间窗口）。
        cutoff_freq (float): 截止频率，仅保留低于该频率的分量。
        sampling_interval (float): 采样间隔（单位：分钟，默认 1 分钟）。
        
    Returns:
        pd.DataFrame: 每列的滚动窗口低频相位 * 幅值结果的累加值，结构与输入一致。
    """
    def compute_phase_times_magnitude(window_data):
        # 如果窗口数据不足，返回 NaN
        if len(window_data) < 2:
            return np.nan
        
        # 计算傅里叶变换
        fft_result = np.fft.fft(window_data)
        freqs = np.fft.fftfreq(len(window_data), d=sampling_interval)
        
        # 筛选低频分量
        low_freq_mask = np.abs(freqs) <= cutoff_freq
        fft_result_low_freq = fft_result[low_freq_mask]
        
        # 计算幅值和相位
        magnitudes = np.abs(fft_result_low_freq)
        phases = np.angle(fft_result_low_freq)
        
        # 相位 * 幅值
        phase_magnitude = phases * magnitudes
        
        # 返回结果的累加值
        return np.sum(phase_magnitude)
    
    # 对 DataFrame 每列应用滚动计算
    result = df.rolling(window=window, min_periods=1).apply(compute_phase_times_magnitude, raw=True)
    return result


# %%
def mean(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    计算滑动窗口内的均值。

    Args:
        df (pd.DataFrame): 时间序列数据（支持多列）。
        window (int): 滑动窗口的大小。

    Returns:
        pd.DataFrame: 滑动均值结果，结构与输入一致。
    """
    return df.rolling(window, min_periods=1).mean()


def std(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    计算滑动窗口内的标准差。

    Args:
        df (pd.DataFrame): 时间序列数据（支持多列）。
        window (int): 滑动窗口的大小。

    Returns:
        pd.DataFrame: 滑动标准差结果，结构与输入一致。
    """
    return df.rolling(window, min_periods=1).std()


def skew(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    计算滑动窗口内的偏度。

    Args:
        df (pd.DataFrame): 时间序列数据（支持多列）。
        window (int): 滑动窗口的大小。

    Returns:
        pd.DataFrame: 滑动偏度结果，结构与输入一致。
    """
    return df.rolling(window, min_periods=1).skew()


def kurt(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    计算滑动窗口内的峰度。

    Args:
        df (pd.DataFrame): 时间序列数据（支持多列）。
        window (int): 滑动窗口的大小。

    Returns:
        pd.DataFrame: 滑动峰度结果，结构与输入一致。
    """
    return df.rolling(window, min_periods=1).kurt()


def rmin(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    计算滑动窗口内的最小值。

    Args:
        df (pd.DataFrame): 时间序列数据（支持多列）。
        window (int): 滑动窗口的大小。

    Returns:
        pd.DataFrame: 滑动最小值结果，结构与输入一致。
    """
    return df.rolling(window, min_periods=1).min()


def rmax(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    计算滑动窗口内的最大值。

    Args:
        df (pd.DataFrame): 时间序列数据（支持多列）。
        window (int): 滑动窗口的大小。

    Returns:
        pd.DataFrame: 滑动最大值结果，结构与输入一致。
    """
    return df.rolling(window, min_periods=1).max()


def iqr(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    计算滑动窗口内的四分位距（75分位数 - 25分位数）。
    
    Args:
        df (pd.DataFrame): 时间序列数据（支持多列）。
        window (int): 滑动窗口的大小。
        
    Returns:
        pd.DataFrame: 滑动窗口内四分位距的结果，结构与输入一致。
    """
    q75 = df.rolling(window, min_periods=1).quantile(0.75)
    q25 = df.rolling(window, min_periods=1).quantile(0.25)
    return q75 - q25


def ewma(df: pd.DataFrame, span: int) -> pd.DataFrame:
    """
    计算连续的指数加权移动平均(EWMA)，允许跨日计算，不会在日期边界重置。
    
    Args:
        df (pd.DataFrame): 时间序列数据（支持多列），index为时间戳。
        span (int): 指数加权的周期数，类似于半衰期。
        
    Returns:
        pd.DataFrame: 连续指数加权移动平均结果，结构与输入一致。
    """
    # 直接对整个时间序列应用ewm方法，不按日期分组
    result = df.ewm(span=span, min_periods=1, adjust=True).mean()
    
    return result


# %% gpt - trend
def rsi(df: pd.DataFrame, period: str | int) -> pd.DataFrame: # relativeStrengthIndex
    """
    计算相对强弱指数 (RSI)。
    
    参数：
    df (pd.DataFrame): 时间序列数据，datetime 为 index。
    period (int): 计算 RSI 的时间步长。
    
    返回：
    pd.DataFrame: 每列的 RSI 值。
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame的index必须是DatetimeIndex类型")
    
    delta = df.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def trendContinuation(df: pd.DataFrame, window: str | int) -> pd.DataFrame:
    """
    计算因子在滑动窗口内的趋势方向延续率（包括方向信息）。
    
    参数：
    df (pd.DataFrame): datetime为index的数据框。
    window (int): 滑动窗口大小（数据点数）。
    
    返回：
    pd.DataFrame: 每列的趋势方向延续率（带方向，范围为-1到1）。
    """
    def continuation(series):
        # 如果窗口数据不足，返回 NaN
        if len(series) < 2:
            return np.nan
        diff = series.diff().dropna()
        if len(diff) < 2:
            return np.nan
        direction = np.sign(diff)
        continuation_rate = (direction == direction.iloc[0]).mean()
        return continuation_rate * direction.iloc[0]
    
    result = df.rolling(window=window, min_periods=1).apply(continuation, raw=False)
    return result


def trendReversalRate(df: pd.DataFrame, window: str | int) -> pd.DataFrame:
    """
    计算因子在滑动窗口内的趋势反转率（带方向信息）。
    
    参数：
    df (pd.DataFrame): datetime为index的数据框。
    window (int): 滑动窗口大小（数据点数）。
    
    返回：
    pd.DataFrame: 每列的趋势反转率（范围为 -1 到 1）。
    """
    def reversal(series):
        # 如果窗口数据不足，返回 NaN
        if len(series) < 2:
            return np.nan
        diff = series.diff().dropna()
        if len(diff) < 2:
            return np.nan
        direction = np.sign(diff)
        reversals = (direction != direction.shift(1)).sum()
        total_changes = len(direction) - 1
        reversal_rate = reversals / total_changes if total_changes > 0 else 0
        return -reversal_rate if direction.iloc[-1] < 0 else reversal_rate
    
    result = df.rolling(window=window, min_periods=1).apply(reversal, raw=False)
    return result


def meanTrendDirectionConsistency(df: pd.DataFrame, window: str | int) -> pd.DataFrame:
    """
    计算滑动窗口内的平均趋势方向一致性。
    
    参数：
    df (pd.DataFrame): datetime为index的数据框。
    window (int): 滑动窗口大小（数据点数）。
    
    返回：
    pd.DataFrame: 每列的平均趋势方向一致性（范围 -1 到 1）。
    """
    def mean_consistency(series):
        # 如果窗口数据不足，返回 NaN
        if len(series) < 2:
            return np.nan
        diff = series.diff().dropna()
        if len(diff) < 2:
            return np.nan
        direction = np.sign(diff)
        return direction.mean()
    
    result = df.rolling(window=window, min_periods=1).apply(mean_consistency, raw=False)
    return result


def trendDirectionEntropy(df: pd.DataFrame, window: str | int) -> pd.DataFrame:
    """
    计算滑动窗口内因子趋势方向的熵值，衡量方向变化的随机性。
    
    参数：
    df (pd.DataFrame): datetime为index的数据框。
    window (int): 滑动窗口大小（数据点数）。
    
    返回：
    pd.DataFrame: 每列趋势方向的熵值（范围为 0 到 1）。
    """
    def entropy(series):
        # 如果窗口数据不足，返回 NaN
        if len(series) < 2:
            return np.nan
        diff = series.diff().dropna()
        if len(diff) < 2:
            return np.nan
        direction = np.sign(diff)
        counts = direction.value_counts(normalize=True)
        return -np.sum(counts * np.log2(counts + 1e-9))
    
    result = df.rolling(window=window, min_periods=1).apply(entropy, raw=False)
    return result


def trendStrengthRatio(df: pd.DataFrame, window: str | int) -> pd.DataFrame:
    """
    计算滑动窗口内因子趋势的强度比率（趋势振幅与噪声的比值）。
    
    参数：
    df (pd.DataFrame): datetime为index的数据框。
    window (int): 滑动窗口大小（数据点数）。
    
    返回：
    pd.DataFrame: 每列趋势强度比率（正负值，方向与趋势一致）。
    """
    def strength_ratio(series):
        # 如果窗口数据不足，返回 NaN
        if len(series) < 2:
            return np.nan
        diff = series.diff().dropna()
        if len(diff) < 2:
            return np.nan
        trend_amplitude = series.max() - series.min()
        noise = np.std(diff)
        return trend_amplitude / (noise + 1e-9) * np.sign(diff.mean())
    
    result = df.rolling(window=window, min_periods=1).apply(strength_ratio, raw=False)
    return result


# %% gpt - percentile
def relativePercentileChange(df: pd.DataFrame, short_window: int, long_window: int) -> pd.DataFrame:
    """
    计算因子在短期趋势上的相对位置变化率，相对于长期窗口内的分布。
    
    参数：
    df (pd.DataFrame): datetime为index的数据框。
    short_window (int): 短期窗口大小，用于计算当前趋势值。
    long_window (int): 长期窗口大小，用于定义相对位置的分布。
    
    返回：
    pd.DataFrame: 每列的相对位置变化率（百分位变化）。
    """
    def percentile_change(series):
        # 如果窗口数据不足，返回 NaN
        if len(series) < 2:
            return np.nan
        short_value = series.iloc[-short_window:].mean()
        long_window_values = series.iloc[-long_window:]
        rank = (long_window_values < short_value).sum() / len(long_window_values)
        return rank
    
    # 计算短期相对长期的百分位
    result = df.rolling(long_window, min_periods=1).apply(percentile_change, raw=False).diff()
    return result


def layeredPercentileTrend(df: pd.DataFrame, short_window: int, long_window: int) -> pd.DataFrame:
    """
    分层计算因子在长期分布中的相对百分位，以及短期趋势。
    
    参数：
    df (pd.DataFrame): datetime为index的数据框。
    short_window (int): 短期窗口大小，用于计算趋势。
    long_window (int): 长期窗口大小，用于计算相对位置。
    
    返回：
    pd.DataFrame: 每列的分层百分位趋势，包含方向信息。
    """
    def layered_trend(series):
        # 如果窗口数据不足，返回 NaN
        if len(series) < 2:
            return np.nan
        short_value = series.iloc[-short_window:].mean()
        long_window_values = series.iloc[-long_window:]
        rank = (long_window_values < short_value).sum() / len(long_window_values)
        short_trend = series.diff(short_window).iloc[-1]
        return rank * np.sign(short_trend)
    
    result = df.rolling(long_window, min_periods=1).apply(layered_trend, raw=False)
    return result


def relativeExtremeTrend(df: pd.DataFrame, short_window: int, long_window: int) -> pd.DataFrame:
    """
    计算因子值相对于长期窗口的极值位置，以及短期趋势。
    
    参数：
    df (pd.DataFrame): datetime为index的数据框。
    short_window (int): 短期窗口大小，用于计算趋势。
    long_window (int): 长期窗口大小，用于计算相对极值。
    
    返回：
    pd.DataFrame: 每列的极值相对位置与短期趋势的结合值。
    """
    def extreme_trend(series):
        # 如果窗口数据不足，返回 NaN
        if len(series) < 2:
            return np.nan
        short_value = series.iloc[-short_window:].mean()
        long_max = series.iloc[-long_window:].max()
        long_min = series.iloc[-long_window:].min()
        relative_position = (short_value - long_min) / (long_max - long_min + 1e-9)
        short_trend = series.diff(short_window).iloc[-1]
        return relative_position * np.sign(short_trend)
    
    result = df.rolling(long_window, min_periods=1).apply(extreme_trend, raw=False)
    return result


def shortLongTrendInteraction(df: pd.DataFrame, short_window: int, long_window: int) -> pd.DataFrame:
    """
    计算短期趋势强度与长期趋势强度的交互关系。
    
    参数：
    df (pd.DataFrame): datetime为index的数据框。
    short_window (int): 短期窗口大小，用于计算短期趋势。
    long_window (int): 长期窗口大小，用于计算长期趋势。
    
    返回：
    pd.DataFrame: 每列的短长期趋势交互强度。
    """
    def interaction(series):
        # 如果窗口数据不足，返回 NaN
        if len(series) < 2:
            return np.nan
        short_trend = series.diff(short_window).mean()
        long_trend = series.diff(long_window).mean()
        return short_trend * long_trend
    
    result = df.rolling(long_window, min_periods=1).apply(interaction, raw=False)
    return result


def percentileTrendAcceleration(df: pd.DataFrame, short_window: int, long_window: int) -> pd.DataFrame:
    """
    计算因子在长期分布中的分位点变化加速度。
    
    参数：
    df (pd.DataFrame): datetime为index的数据框。
    short_window (int): 短期窗口大小，用于计算趋势加速度。
    long_window (int): 长期窗口大小，用于计算分位点分布。
    
    返回：
    pd.DataFrame: 每列的分位点趋势加速度。
    """
    def percentile_acceleration(series):
        # 如果窗口数据不足，返回 NaN
        if len(series) < 2:
            return np.nan
        short_value = series.iloc[-short_window:].mean()
        long_window_values = series.iloc[-long_window:]
        rank = (long_window_values < short_value).sum() / len(long_window_values)
        rank_diff = rank - (long_window_values < series.iloc[-short_window-1:-1].mean()).sum() / len(long_window_values)
        return rank_diff
    
    result = df.rolling(long_window, min_periods=1).apply(percentile_acceleration, raw=False)
    return result


# %% gpt - ft
def lowFreqWithPosTrend(df: pd.DataFrame, fft_window: str | int, pos_window: str | int, 
                        freq_cutoff: float, sampling_interval: float) -> pd.DataFrame:
    """
    提取短期傅里叶变换的低频幅度，并根据因子值在长期窗口内的相对位置判断趋势方向。
    
    参数：
    df (pd.DataFrame): datetime为index的数据框。
    fft_window (str | int): 用于傅里叶变换的短期窗口大小。
    pos_window (str | int): 用于计算长期相对位置的窗口大小。
    freq_cutoff (float): 低频分量的截止频率。
    sampling_interval (float): 采样间隔。
    
    返回：
    pd.DataFrame: 每列的短期低频幅度与基于长期相对位置的趋势方向交互结果。
    """
    def compute_low_freq_amplitude(series):
        # 如果窗口数据不足，返回 NaN
        if len(series) < 2:
            return np.nan
        fft_result = np.fft.fft(series)
        freqs = np.fft.fftfreq(len(series), d=sampling_interval)
        low_freq_amplitude = np.sum(np.abs(fft_result[np.abs(freqs) <= freq_cutoff]))
        return low_freq_amplitude

    def compute_pos_trend(series):
        # 如果窗口数据不足，返回 NaN
        if len(series) < 2:
            return np.nan
        rank = (series < series.iloc[-1]).sum() / len(series)
        return 1 if rank > 0.5 else -1

    low_freq_amplitude = df.rolling(window=fft_window, min_periods=1).apply(compute_low_freq_amplitude, raw=False)
    pos_trend = df.rolling(window=pos_window, min_periods=1).apply(compute_pos_trend, raw=False)

    return low_freq_amplitude * pos_trend


# %% gpt - intraday effect
def rollingRemoveIntradayEffect(df: pd.DataFrame, window: str = '30d', time_col: str = "time") -> pd.DataFrame:
    """
    滚动去除因子中的日内效应，每个时间点回看窗口内的均值。

    参数：
        df (pd.DataFrame): 包含时间序列的DataFrame，index为DatetimeIndex。
        window (int): 回看滚动窗口大小（天数）。
        time_col (str): 每天的具体时间点列名（如 '09:30'）。

    返回：
        pd.DataFrame: 去除滚动窗口内日内效应后的数据。
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame的index必须是DatetimeIndex类型")
    
    # 2. 提取每天的时间点，单独存储，避免污染数据
    time_points = df.index.time
    
    # 3. 滚动计算日内均值
    rolling_result = df.groupby(time_points, group_keys=False).apply(
        lambda x: x.rolling(window=window, min_periods=1).mean()
    )
    
    # 4. 去除滚动均值的影响
    result = df - rolling_result
    
    return result


def rollingMinuteQuantileScale(
    df: pd.DataFrame,
    window: str = "30d",
    min_periods: int = 1,
    quantile: float = 0.05
) -> pd.DataFrame:
    """
    每分钟时间点的滚动窗口分位数归一化。
    
    参数:
        df (pd.DataFrame): 时间序列数据，index 必须为 DatetimeIndex。
        window (str, optional): 滚动窗口大小，支持 Pandas 时间窗口（如 "30d" 表示30天）。默认值为 "30d"。
        min_periods (int, optional): 窗口中要求的最少观察数，默认值为 10。
        quantile (float, optional): 用于归一化的分位数，默认值为 0.05。
        
    返回:
        pd.DataFrame: 归一化后的数据。
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame 的 index 必须是 DatetimeIndex 类型")
    
    # 定义分位数
    lower_quantile = quantile
    upper_quantile = 1 - quantile

    # 提取每分钟的时间点
    minutes = df.index.time  # 使用索引时间提取分钟信息，而不添加列

    # 初始化存储结果
    result = pd.DataFrame(index=df.index, columns=df.columns)

    # 按分钟分组
    unique_minutes = np.unique(minutes)
    for minute in unique_minutes:
        # 找到当前分钟对应的索引
        mask = minutes == minute
        group = df[mask]

        # 对每组按滚动窗口计算分位数
        rolling_lower = group.rolling(window=window, min_periods=min_periods).quantile(lower_quantile)
        rolling_upper = group.rolling(window=window, min_periods=min_periods).quantile(upper_quantile)

        # 归一化：基于分位数进行缩放
        scaled = (group - rolling_lower) / (rolling_upper - rolling_lower).replace(0, np.nan)
        
        # 裁剪结果在 0 和 1 之间
        result.loc[group.index] = scaled.clip(lower=0, upper=1)

    return result


# =============================================================================
# def rollingAggMinuteMinMaxScale(
#     df: pd.DataFrame,
#     window: str = "30d",
#     min_periods: int = 1,
#     quantile: float = 0.05,
#     interval: int = 5
# ) -> pd.DataFrame:
#     """
#     每分钟时间点的滚动窗口分位数归一化，同时聚合到整 interval 分钟。
# 
#     参数:
#         df (pd.DataFrame): 时间序列数据，index 必须为 DatetimeIndex。
#         window (str, optional): 滚动窗口大小，支持 Pandas 时间窗口（如 "30d" 表示30天）。默认值为 "30d"。
#         min_periods (int, optional): 窗口中要求的最少观察数，默认值为 1。
#         quantile (float, optional): 用于归一化的分位数，默认值为 0.05。
#         interval (int, optional): 时间分组的间隔（以分钟为单位），默认值为 5。
# 
#     返回:
#         pd.DataFrame: 归一化后的数据。
#     """
#     if not isinstance(df.index, pd.DatetimeIndex):
#         raise ValueError("DataFrame 的 index 必须是 DatetimeIndex 类型")
# 
#     # 提取分钟并分组到最近的整 interval 分钟
#     grouped_minutes = (df.index.minute // interval) * interval
#     group_labels = df.index.hour * 60 + grouped_minutes
# 
#     # 初始化存储结果
#     result = pd.DataFrame(index=df.index, columns=df.columns)
# 
#     for group_id in np.unique(group_labels):
#         # 提取当前组
#         group_mask = group_labels == group_id
#         group = df[group_mask]
# 
#         # 对每组按滚动窗口计算分位数
#         rolling_lower = group.rolling(window=window, min_periods=min_periods).quantile(quantile)
#         rolling_upper = group.rolling(window=window, min_periods=min_periods).quantile(1 - quantile)
# 
#         # 归一化：基于分位数进行缩放
#         scaled = (group - rolling_lower) / (rolling_upper - rolling_lower).replace(0, np.nan)
# 
#         # 裁剪结果在 0 和 1 之间
#         result.loc[group.index] = scaled.clip(lower=0, upper=1)
# 
#     return result
# 
# 
# def rollingAggMinutePercentile(
#     df: pd.DataFrame,
#     window: str = "30d",
#     min_periods: int = 1,
#     interval: int = 5
# ) -> pd.DataFrame:
#     """
#     每分钟时间点的滚动窗口百分位计算，并聚合到整 interval 分钟。
# 
#     参数:
#         df (pd.DataFrame): 时间序列数据，index 必须为 DatetimeIndex。
#         window (str, optional): 滚动窗口大小，支持 Pandas 时间窗口（如 "30d" 表示30天）。默认值为 "30d"。
#         min_periods (int, optional): 窗口中要求的最少观察数，默认值为 1。
#         interval (int, optional): 时间分组的间隔（以分钟为单位），默认值为 5。
# 
#     返回:
#         pd.DataFrame: 百分位归一化后的数据。
#     """
#     if not isinstance(df.index, pd.DatetimeIndex):
#         raise ValueError("DataFrame 的 index 必须是 DatetimeIndex 类型")
# 
#     # 提取分钟并分组到最近的整 interval 分钟
#     grouped_minutes = (df.index.minute // interval) * interval
#     group_labels = df.index.hour * 60 + grouped_minutes
# 
#     # 初始化存储结果
#     result = pd.DataFrame(index=df.index, columns=df.columns)
# 
#     for group_id in np.unique(group_labels):
#         # 提取当前组
#         group_mask = group_labels == group_id
#         group = df[group_mask]
# 
#         # 对每组按滚动窗口计算百分位数
#         rolling_percentile = group.rolling(window=window, min_periods=min_periods).apply(
#             lambda x: 100 * (x.iloc[-1] <= x).sum() / len(x), raw=False
#         )
# 
#         # 存储结果
#         result.loc[group.index] = rolling_percentile
# 
#     return result
# =============================================================================


def rollingAggMinuteMinMaxScale(
    data,
    window: str = "30d",
    min_periods: int = 1,
    quantile: float = 0.05,
    interval: int = 5
):
    """
    每分钟时间点的滚动窗口分位数归一化，同时聚合到整 interval 分钟。
    参数:
        data: 时间序列数据，可以是 DataFrame 或 Series，index 必须为 DatetimeIndex。
        window (str, optional): 滚动窗口大小，支持 Pandas 时间窗口（如 "30d" 表示30天）。默认值为 "30d"。
        min_periods (int, optional): 窗口中要求的最少观察数，默认值为 1。
        quantile (float, optional): 用于归一化的分位数，默认值为 0.05。
        interval (int, optional): 时间分组的间隔（以分钟为单位），默认值为 5。
    返回:
        与输入相同类型的归一化后的数据（DataFrame 或 Series）。
    """
    # 将 Series 转换为 DataFrame 进行处理
    is_series = isinstance(data, pd.Series)
    df = data.to_frame() if is_series else data.copy()
    
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("输入数据的 index 必须是 DatetimeIndex 类型")
    
    # 提取分钟并分组到最近的整 interval 分钟
    grouped_minutes = (df.index.minute // interval) * interval
    group_labels = df.index.hour * 60 + grouped_minutes
    
    # 初始化存储结果
    result = pd.DataFrame(index=df.index, columns=df.columns)
    
    for group_id in np.unique(group_labels):
        # 提取当前组
        group_mask = group_labels == group_id
        group = df[group_mask]
        
        # 对每组按滚动窗口计算分位数
        rolling_lower = group.rolling(window=window, min_periods=min_periods).quantile(quantile)
        rolling_upper = group.rolling(window=window, min_periods=min_periods).quantile(1 - quantile)
        
        # 归一化：基于分位数进行缩放
        scaled = (group - rolling_lower) / (rolling_upper - rolling_lower).replace(0, np.nan)
        
        # 裁剪结果在 0 和 1 之间
        result.loc[group.index] = scaled.clip(lower=0, upper=1)
    
    # 如果输入是 Series，则返回 Series
    if is_series:
        return result.iloc[:, 0]
    return result


def aggMinmax(
    data,
    window: str = "30d",
    min_periods: int = 1,
    quantile: float = 0.05,
    interval: int = 5
):
    """
    每分钟时间点的滚动窗口分位数归一化，同时聚合到整 interval 分钟。
    参数:
        data: 时间序列数据，可以是 DataFrame 或 Series，index 必须为 DatetimeIndex。
        window (str, optional): 滚动窗口大小，支持 Pandas 时间窗口（如 "30d" 表示30天）。默认值为 "30d"。
        min_periods (int, optional): 窗口中要求的最少观察数，默认值为 1。
        quantile (float, optional): 用于归一化的分位数，默认值为 0.05。
        interval (int, optional): 时间分组的间隔（以分钟为单位），默认值为 5。
    返回:
        与输入相同类型的归一化后的数据（DataFrame 或 Series）。
    """
    # 将 Series 转换为 DataFrame 进行处理
    is_series = isinstance(data, pd.Series)
    df = data.to_frame() if is_series else data.copy()
    
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("输入数据的 index 必须是 DatetimeIndex 类型")
    
    # 提取分钟并分组到最近的整 interval 分钟
    grouped_minutes = (df.index.minute // interval) * interval
    group_labels = df.index.hour * 60 + grouped_minutes
    
    # 初始化存储结果
    result = pd.DataFrame(index=df.index, columns=df.columns)
    
    for group_id in np.unique(group_labels):
        # 提取当前组
        group_mask = group_labels == group_id
        group = df[group_mask]
        
        # 对每组按滚动窗口计算分位数
        rolling_lower = group.rolling(window=window, min_periods=min_periods).quantile(quantile)
        rolling_upper = group.rolling(window=window, min_periods=min_periods).quantile(1 - quantile)
        
        # 归一化：基于分位数进行缩放
        scaled = (group - rolling_lower) / (rolling_upper - rolling_lower).replace(0, np.nan)
        
        # 裁剪结果在 0 和 1 之间
        result.loc[group.index] = scaled.clip(lower=0, upper=1)
    
    # 如果输入是 Series，则返回 Series
    if is_series:
        return result.iloc[:, 0]
    return result


def rollingAggMinutePercentile(
    data,
    window: str = "30d",
    min_periods: int = 1,
    interval: int = 5
):
    """
    每分钟时间点的滚动窗口百分位计算，并聚合到整 interval 分钟。
    参数:
        data: 时间序列数据，可以是 DataFrame 或 Series，index 必须为 DatetimeIndex。
        window (str, optional): 滚动窗口大小，支持 Pandas 时间窗口（如 "30d" 表示30天）。默认值为 "30d"。
        min_periods (int, optional): 窗口中要求的最少观察数，默认值为 1。
        interval (int, optional): 时间分组的间隔（以分钟为单位），默认值为 5。
    返回:
        与输入相同类型的百分位归一化后的数据（DataFrame 或 Series）。
    """
    # 将 Series 转换为 DataFrame 进行处理
    is_series = isinstance(data, pd.Series)
    df = data.to_frame() if is_series else data.copy()
    
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("输入数据的 index 必须是 DatetimeIndex 类型")
    
    # 提取分钟并分组到最近的整 interval 分钟
    grouped_minutes = (df.index.minute // interval) * interval
    group_labels = df.index.hour * 60 + grouped_minutes
    
    # 初始化存储结果
    result = pd.DataFrame(index=df.index, columns=df.columns)
    
    for group_id in np.unique(group_labels):
        # 提取当前组
        group_mask = group_labels == group_id
        group = df[group_mask]
        
        # 对每组按滚动窗口计算百分位数
        rolling_percentile = group.rolling(window=window, min_periods=min_periods).apply(
            lambda x: 100 * (x.iloc[-1] <= x).sum() / len(x), raw=False
        )
        
        # 存储结果
        result.loc[group.index] = rolling_percentile
    
    # 如果输入是 Series，则返回 Series
    if is_series:
        return result.iloc[:, 0]
    return result


def rollingAggMinutePctl( # 修正正负号
    data,
    window: str = "30d",
    min_periods: int = 1,
    interval: int = 5
):
    """
    每分钟时间点的滚动窗口百分位计算，并聚合到整 interval 分钟。
    参数:
        data: 时间序列数据，可以是 DataFrame 或 Series，index 必须为 DatetimeIndex。
        window (str, optional): 滚动窗口大小，支持 Pandas 时间窗口（如 "30d" 表示30天）。默认值为 "30d"。
        min_periods (int, optional): 窗口中要求的最少观察数，默认值为 1。
        interval (int, optional): 时间分组的间隔（以分钟为单位），默认值为 5。
    返回:
        与输入相同类型的百分位归一化后的数据（DataFrame 或 Series）。
    """
    # 将 Series 转换为 DataFrame 进行处理
    is_series = isinstance(data, pd.Series)
    df = data.to_frame() if is_series else data.copy()
    
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("输入数据的 index 必须是 DatetimeIndex 类型")
    
    # 提取分钟并分组到最近的整 interval 分钟
    grouped_minutes = (df.index.minute // interval) * interval
    group_labels = df.index.hour * 60 + grouped_minutes
    
    # 初始化存储结果
    result = pd.DataFrame(index=df.index, columns=df.columns)
    
    for group_id in np.unique(group_labels):
        # 提取当前组
        group_mask = group_labels == group_id
        group = df[group_mask]
        
        # 对每组按滚动窗口计算百分位数
        rolling_percentile = group.rolling(window=window, min_periods=min_periods).apply(
            lambda x: (x.iloc[-1] >= x).sum() / len(x), raw=False
        )
        
        # 存储结果
        result.loc[group.index] = rolling_percentile
    
    # 如果输入是 Series，则返回 Series
    if is_series:
        return result.iloc[:, 0]
    return result


def aggPctl( # 修正正负号
    data,
    window: str = "30d",
    min_periods: int = 1,
    interval: int = 5
):
    """
    每分钟时间点的滚动窗口百分位计算，并聚合到整 interval 分钟。
    参数:
        data: 时间序列数据，可以是 DataFrame 或 Series，index 必须为 DatetimeIndex。
        window (str, optional): 滚动窗口大小，支持 Pandas 时间窗口（如 "30d" 表示30天）。默认值为 "30d"。
        min_periods (int, optional): 窗口中要求的最少观察数，默认值为 1。
        interval (int, optional): 时间分组的间隔（以分钟为单位），默认值为 5。
    返回:
        与输入相同类型的百分位归一化后的数据（DataFrame 或 Series）。
    """
    # 将 Series 转换为 DataFrame 进行处理
    is_series = isinstance(data, pd.Series)
    df = data.to_frame() if is_series else data.copy()
    
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("输入数据的 index 必须是 DatetimeIndex 类型")
    
    # 提取分钟并分组到最近的整 interval 分钟
    grouped_minutes = (df.index.minute // interval) * interval
    group_labels = df.index.hour * 60 + grouped_minutes
    
    # 初始化存储结果
    result = pd.DataFrame(index=df.index, columns=df.columns)
    
    for group_id in np.unique(group_labels):
        # 提取当前组
        group_mask = group_labels == group_id
        group = df[group_mask]
        
        # 对每组按滚动窗口计算百分位数
        rolling_percentile = group.rolling(window=window, min_periods=min_periods).apply(
            lambda x: (x.iloc[-1] >= x).sum() / len(x), raw=False
        )
        
        # 存储结果
        result.loc[group.index] = rolling_percentile
    
    # 如果输入是 Series，则返回 Series
    if is_series:
        return result.iloc[:, 0]
    return result


def rollingMinuteZScore(
    data,
    window: str = "30d",
    min_periods: int = 1,
    interval: int = 5
):
    """
    计算每个分钟时间点在过去同一分组分钟（同比）中的z-score，并聚合到整interval分钟。
    
    参数:
        data: 时间序列数据，可以是 DataFrame 或 Series，index 必须为 DatetimeIndex。
        window (str, optional): 滚动窗口大小，支持 Pandas 时间窗口（如 "30d" 表示30天）。默认值为 "30d"。
        min_periods (int, optional): 窗口中要求的最少观察数，默认值为 1。
        interval (int, optional): 时间分组的间隔（以分钟为单位），默认值为 5。
    
    返回:
        与输入相同类型的z-score标准化后的数据（DataFrame 或 Series）。
    """
    import pandas as pd
    import numpy as np
    
    # 将 Series 转换为 DataFrame 进行处理
    is_series = isinstance(data, pd.Series)
    df = data.to_frame() if is_series else data.copy()
    
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("输入数据的 index 必须是 DatetimeIndex 类型")
    
    # 提取分钟并分组到最近的整 interval 分钟
    grouped_minutes = (df.index.minute // interval) * interval
    # 创建每天中的分钟标识符（0-1439，代表一天中的每一分钟）
    group_labels = df.index.hour * 60 + grouped_minutes
    
    # 初始化存储结果
    result = pd.DataFrame(index=df.index, columns=df.columns)
    
    for group_id in np.unique(group_labels):
        # 提取当前组（同一分钟组的所有历史数据）
        group_mask = group_labels == group_id
        group = df[group_mask]
        
        if group.empty:
            continue
            
        # 按时间排序确保正确的历史计算
        group = group.sort_index()
        
        # 对每组按滚动窗口计算均值和标准差
        rolling_mean = group.rolling(window=window, min_periods=min_periods).mean()
        rolling_std = group.rolling(window=window, min_periods=min_periods).std()
        
        # 计算z-score: (x - mean) / std
        # 处理标准差为0的情况
        z_scores = (group - rolling_mean) / rolling_std.replace(0, np.nan)
        
        # 填充结果到主DataFrame
        result.loc[group.index] = z_scores
    
    # 如果输入是 Series，则返回 Series
    if is_series:
        return result.iloc[:, 0]
    
    return result


def aggZsc(
    data,
    window: str = "30d",
    min_periods: int = 1,
    interval: int = 5
):
    """
    计算每个分钟时间点在过去同一分组分钟（同比）中的z-score，并聚合到整interval分钟。
    
    参数:
        data: 时间序列数据，可以是 DataFrame 或 Series，index 必须为 DatetimeIndex。
        window (str, optional): 滚动窗口大小，支持 Pandas 时间窗口（如 "30d" 表示30天）。默认值为 "30d"。
        min_periods (int, optional): 窗口中要求的最少观察数，默认值为 1。
        interval (int, optional): 时间分组的间隔（以分钟为单位），默认值为 5。
    
    返回:
        与输入相同类型的z-score标准化后的数据（DataFrame 或 Series）。
    """
    import pandas as pd
    import numpy as np
    
    # 将 Series 转换为 DataFrame 进行处理
    is_series = isinstance(data, pd.Series)
    df = data.to_frame() if is_series else data.copy()
    
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("输入数据的 index 必须是 DatetimeIndex 类型")
    
    # 提取分钟并分组到最近的整 interval 分钟
    grouped_minutes = (df.index.minute // interval) * interval
    # 创建每天中的分钟标识符（0-1439，代表一天中的每一分钟）
    group_labels = df.index.hour * 60 + grouped_minutes
    
    # 初始化存储结果
    result = pd.DataFrame(index=df.index, columns=df.columns)
    
    for group_id in np.unique(group_labels):
        # 提取当前组（同一分钟组的所有历史数据）
        group_mask = group_labels == group_id
        group = df[group_mask]
        
        if group.empty:
            continue
            
        # 按时间排序确保正确的历史计算
        group = group.sort_index()
        
        # 对每组按滚动窗口计算均值和标准差
        rolling_mean = group.rolling(window=window, min_periods=min_periods).mean()
        rolling_std = group.rolling(window=window, min_periods=min_periods).std()
        
        # 计算z-score: (x - mean) / std
        # 处理标准差为0的情况
        z_scores = (group - rolling_mean) / rolling_std.replace(0, np.nan)
        
        # 填充结果到主DataFrame
        result.loc[group.index] = z_scores
    
    # 如果输入是 Series，则返回 Series
    if is_series:
        return result.iloc[:, 0]
    
    return result


def rollingMinuteDeMean(
    data,
    window: str = "30d",
    min_periods: int = 1,
    interval: int = 5
):
    """
    计算每个分钟时间点在过去同一分组分钟（同比）中的去均值结果，并聚合到整interval分钟。
    只执行中心化处理（减去均值），不进行标准差归一化。
    
    参数:
        data: 时间序列数据，可以是 DataFrame 或 Series，index 必须为 DatetimeIndex。
        window (str, optional): 滚动窗口大小，支持 Pandas 时间窗口（如 "30d" 表示30天）。默认值为 "30d"。
        min_periods (int, optional): 窗口中要求的最少观察数，默认值为 1。
        interval (int, optional): 时间分组的间隔（以分钟为单位），默认值为 5。
    
    返回:
        与输入相同类型的去均值后的数据（DataFrame 或 Series）。
    """
    import pandas as pd
    import numpy as np
    
    # 将 Series 转换为 DataFrame 进行处理
    is_series = isinstance(data, pd.Series)
    df = data.to_frame() if is_series else data.copy()
    
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("输入数据的 index 必须是 DatetimeIndex 类型")
    
    # 提取分钟并分组到最近的整 interval 分钟
    grouped_minutes = (df.index.minute // interval) * interval
    # 创建每天中的分钟标识符（0-1439，代表一天中的每一分钟）
    group_labels = df.index.hour * 60 + grouped_minutes
    
    # 初始化存储结果
    result = pd.DataFrame(index=df.index, columns=df.columns)
    
    for group_id in np.unique(group_labels):
        # 提取当前组（同一分钟组的所有历史数据）
        group_mask = group_labels == group_id
        group = df[group_mask]
        
        if group.empty:
            continue
            
        # 按时间排序确保正确的历史计算
        group = group.sort_index()
        
        # 对每组按滚动窗口计算均值
        rolling_mean = group.rolling(window=window, min_periods=min_periods).mean()
        
        # 只执行去均值操作: x - mean
        de_meaned = group - rolling_mean
        
        # 填充结果到主DataFrame
        result.loc[group.index] = de_meaned
    
    # 如果输入是 Series，则返回 Series
    if is_series:
        return result.iloc[:, 0]
    
    return result


def aggDmean(
    data,
    window: str = "30d",
    min_periods: int = 1,
    interval: int = 5
):
    """
    计算每个分钟时间点在过去同一分组分钟（同比）中的去均值结果，并聚合到整interval分钟。
    只执行中心化处理（减去均值），不进行标准差归一化。
    
    参数:
        data: 时间序列数据，可以是 DataFrame 或 Series，index 必须为 DatetimeIndex。
        window (str, optional): 滚动窗口大小，支持 Pandas 时间窗口（如 "30d" 表示30天）。默认值为 "30d"。
        min_periods (int, optional): 窗口中要求的最少观察数，默认值为 1。
        interval (int, optional): 时间分组的间隔（以分钟为单位），默认值为 5。
    
    返回:
        与输入相同类型的去均值后的数据（DataFrame 或 Series）。
    """
    import pandas as pd
    import numpy as np
    
    # 将 Series 转换为 DataFrame 进行处理
    is_series = isinstance(data, pd.Series)
    df = data.to_frame() if is_series else data.copy()
    
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("输入数据的 index 必须是 DatetimeIndex 类型")
    
    # 提取分钟并分组到最近的整 interval 分钟
    grouped_minutes = (df.index.minute // interval) * interval
    # 创建每天中的分钟标识符（0-1439，代表一天中的每一分钟）
    group_labels = df.index.hour * 60 + grouped_minutes
    
    # 初始化存储结果
    result = pd.DataFrame(index=df.index, columns=df.columns)
    
    for group_id in np.unique(group_labels):
        # 提取当前组（同一分钟组的所有历史数据）
        group_mask = group_labels == group_id
        group = df[group_mask]
        
        if group.empty:
            continue
            
        # 按时间排序确保正确的历史计算
        group = group.sort_index()
        
        # 对每组按滚动窗口计算均值
        rolling_mean = group.rolling(window=window, min_periods=min_periods).mean()
        
        # 只执行去均值操作: x - mean
        de_meaned = group - rolling_mean
        
        # 填充结果到主DataFrame
        result.loc[group.index] = de_meaned
    
    # 如果输入是 Series，则返回 Series
    if is_series:
        return result.iloc[:, 0]
    
    return result


# %%
def minmax_scale(df: pd.DataFrame, window=20, min_periods=10, quantile=0.05) -> pd.DataFrame:
    """
    滚动窗口分位数归一化。

    参数:
        df (pd.DataFrame): 用于计算滚动分位数归一化的变量
        step (int, optional): 计算归一化时的步长。默认值为1。
        window (int, optional): 滚动窗口大小。默认值为20。
        min_periods (int, optional): 窗口中要求的最少观察数。默认值为10。
        quantile (float, optional): 用于归一化的分位数。默认值为0.05，即5%。

    返回:
        pd.DataFrame: 归一化后的DataFrame
    """
    
    # 定义5%和95%分位数
    lower_quantile = quantile
    upper_quantile = 1 - quantile
    
    # 计算滚动窗口的分位数
    df_lower = df.rolling(window=window, min_periods=min_periods).quantile(lower_quantile)
    df_upper = df.rolling(window=window, min_periods=min_periods).quantile(upper_quantile)
    
    # 进行分位数归一化
    df_quantile_scaled = (df - df_lower) / (df_upper - df_lower).replace(0, np.nan)
    
    # 裁剪结果使其在 0 和 1 之间
    df_quantile_scaled = df_quantile_scaled.clip(lower=0, upper=1)
    
    return df_quantile_scaled


def minmax(df: pd.DataFrame, window=20, min_periods=10, quantile=0.05) -> pd.DataFrame:
    """
    滚动窗口分位数归一化。

    参数:
        df (pd.DataFrame): 用于计算滚动分位数归一化的变量
        step (int, optional): 计算归一化时的步长。默认值为1。
        window (int, optional): 滚动窗口大小。默认值为20。
        min_periods (int, optional): 窗口中要求的最少观察数。默认值为10。
        quantile (float, optional): 用于归一化的分位数。默认值为0.05，即5%。

    返回:
        pd.DataFrame: 归一化后的DataFrame
    """
    
    # 定义5%和95%分位数
    lower_quantile = quantile
    upper_quantile = 1 - quantile
    
    # 计算滚动窗口的分位数
    df_lower = df.rolling(window=window, min_periods=min_periods).quantile(lower_quantile)
    df_upper = df.rolling(window=window, min_periods=min_periods).quantile(upper_quantile)
    
    # 进行分位数归一化
    df_quantile_scaled = (df - df_lower) / (df_upper - df_lower).replace(0, np.nan)
    
    # 裁剪结果使其在 0 和 1 之间
    df_quantile_scaled = df_quantile_scaled.clip(lower=0, upper=1)
    
    return df_quantile_scaled


# =============================================================================
# # Numba 并行滚动分位归一化实现（修复 window 边界问题）
# @njit(parallel=True)
# def rolling_quantile_vectorized(data, window, lower_q, upper_q):
#     n_rows, n_cols = data.shape
#     result = np.empty((n_rows, n_cols), dtype=np.float32)
# 
#     for j in prange(n_cols):  # 并行每列
#         col = data[:, j]
#         lower = np.full(n_rows, np.nan, dtype=np.float32)
#         upper = np.full(n_rows, np.nan, dtype=np.float32)
# 
#         for i in range(n_rows):
#             start = max(0, i - window + 1)
#             window_vals = col[start:i + 1]
#             lower[i] = np.quantile(window_vals, lower_q)
#             upper[i] = np.quantile(window_vals, upper_q)
# 
#         for i in range(n_rows):
#             denom = upper[i] - lower[i]
#             if np.isnan(denom) or denom == 0:
#                 result[i, j] = np.nan
#             else:
#                 val = (col[i] - lower[i]) / denom
#                 result[i, j] = min(1.0, max(0.0, val))
# 
#     return result
# 
# 
# def minmax_scale_numba_parallel(df: pd.DataFrame, window=20, quantile=0.05) -> pd.DataFrame:
#     df_filled = df.fillna(0)  # 先填充 NaN
#     data = df_filled.values.astype(np.float32)
#     scaled = rolling_quantile_vectorized(data, window, quantile, 1 - quantile)
#     return pd.DataFrame(scaled, index=df.index, columns=df.columns)
# 
# =============================================================================


# =============================================================================
# # 对单列做 rolling quantile
# @njit(parallel=True)
# def rolling_quantile_numba(arr, window, quantile):
#     n = len(arr)
#     result = np.full(n, np.nan)
# 
#     for i in prange(window - 1, n):
#         start = max(0, i - window + 1)
#         window_vals = arr[start:i + 1]
#         result[i] = np.quantile(window_vals, quantile)
# 
#     return result
# 
# # 对整个 DataFrame 做分位数归一化，并打印进度
# def minmax_scale_numba_parallel(df: pd.DataFrame, window=20, quantile=0.05) -> pd.DataFrame:
#     lower_quantile = quantile
#     upper_quantile = 1 - quantile
#     result = pd.DataFrame(index=df.index, columns=df.columns, dtype=np.float32)
# 
#     total_cols = len(df.columns)
# 
#     for idx, col in enumerate(df.columns):
#         breakpoint()
#         series = df[col].fillna(0).values.astype(np.float32)
#         lower = rolling_quantile_numba(series, window, lower_quantile)
#         upper = rolling_quantile_numba(series, window, upper_quantile)
#         scaled = (series - lower) / (upper - lower + 1e-12)
#         scaled = np.clip(scaled, 0, 1)
#         result[col] = scaled
# 
#         # 打印进度（每处理完50列更新一次，或最后一列）
#         if (idx + 1) % 10 == 0 or idx == total_cols - 1:
#             print(f"[minmax_scale_numba] Processed {idx + 1}/{total_cols} columns")
# 
#     return result
# =============================================================================


import numpy as np
import pandas as pd
from numba import njit, prange, set_num_threads
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# =============================================================================
# @njit(parallel=True)
# def rolling_quantile_numba(arr, window, quantile):
#     n = len(arr)
#     result = np.full(n, np.nan, dtype=np.float32)
#     for i in prange(window - 1, n):
#         start = max(0, i - window + 1)
#         window_vals = arr[start:i + 1]
#         result[i] = np.quantile(window_vals, quantile)
#     return result
# 
# def process_column_block(df_block: pd.DataFrame, window: int, quantile: float, threads_per_worker: int, block_idx: int):
#     set_num_threads(threads_per_worker)
#     lower_q = quantile
#     upper_q = 1 - quantile
#     result_dict = {}
# 
#     total_cols = len(df_block.columns)
#     for idx, col in enumerate(df_block.columns):
#         series = df_block[col].values
#         lower = rolling_quantile_numba(series, window, lower_q)
#         upper = rolling_quantile_numba(series, window, upper_q)
#         scaled = (series - lower) / (upper - lower + 1e-12)
#         result_dict[col] = np.clip(scaled, 0, 1)
# 
#         # if (idx + 1) % 50 == 0 or idx == total_cols - 1:
#         #     print(f"[Worker-{block_idx}] Processed {idx + 1}/{total_cols} columns")
# 
#     return block_idx, result_dict
# =============================================================================

def process_column_block(df_block, window, quantile, block_idx):
    lower = df_block.rolling(window=window, min_periods=10).quantile(quantile)
    upper = df_block.rolling(window=window, min_periods=10).quantile(1 - quantile)
    scaled = (df_block - lower) / (upper - lower + 1e-12)
    scaled = scaled.clip(lower=0, upper=1)
    # print(f"[Worker-{block_idx}] Done with pandas.")
    return block_idx, scaled


def minmax_scale_numba_parallel(df: pd.DataFrame, window=20, quantile=0.05,
                                 n_jobs=150, block_size=5, threads_per_worker=10):
    df = df.fillna(0).astype(np.float32)
    col_blocks = [df.columns[i:i+block_size] for i in range(0, len(df.columns), block_size)]
    result = pd.DataFrame(index=df.index, columns=df.columns, dtype=np.float32)
    total_blocks = len(col_blocks)

    print(f"[Main] Launching {total_blocks} blocks with {n_jobs} processes...")

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        future_to_idx = {}
        for block_idx, cols in enumerate(col_blocks):
            df_block = df[cols]
            future = executor.submit(process_column_block, df_block, window, quantile, block_idx) 
            future_to_idx[future] = (block_idx, cols)

        with tqdm(total=total_blocks, desc="Progress") as pbar:
            for future in as_completed(future_to_idx):
                block_idx, cols = future_to_idx[future]
                _, block_result = future.result()
                for col in cols:
                    result[col] = block_result[col]
                pbar.update(1)

    print("[Main] All blocks completed and written into result DataFrame.")
    return result


def rlPctl(data_df, window):
    """
    计算每个位置的当前值在滚动窗口历史值中的分位数。

    参数：
    data_df (pd.DataFrame): 输入的数据 DataFrame。
    window (int): 滚动窗口的大小。

    返回：
    pd.DataFrame: 每个位置的当前值在滚动窗口中的分位数。
    """
    def calc_percentile(series):
        current_value = series.iloc[-1]
        rolling_window = series.iloc[:-1].dropna()
        if len(rolling_window) == 0:
            return np.nan
        return (rolling_window <= current_value).mean()

    return data_df.rolling(window, min_periods=1).apply(calc_percentile, raw=False)


def minmaxSep(df: pd.DataFrame, window=20, min_periods=10, quantile=0.05) -> pd.DataFrame:
    """
    滚动窗口分位数归一化，正值和负值分别归一化。

    参数:
        df (pd.DataFrame): 用于计算滚动分位数归一化的变量
        step (int, optional): 计算归一化时的步长。默认值为1。
        window (int, optional): 滚动窗口大小。默认值为20。
        min_periods (int, optional): 窗口中要求的最少观察数。默认值为10。
        quantile (float, optional): 用于归一化的分位数。默认值为0.05，即5%。

    返回:
        pd.DataFrame: 归一化后的DataFrame
    """
    # 定义5%和95%分位数
    lower_quantile = quantile
    upper_quantile = 1 - quantile
    
    # 创建正值和负值的掩码
    positive_mask = df > 0
    negative_mask = df < 0
    
    # 计算正值部分的滚动窗口分位数
    df_positive = df[positive_mask]
    df_positive_lower = df_positive.rolling(window=window, min_periods=min_periods).quantile(lower_quantile)
    df_positive_upper = df_positive.rolling(window=window, min_periods=min_periods).quantile(upper_quantile)
    
    # 计算负值部分的滚动窗口分位数
    df_negative = df[negative_mask]
    df_negative_lower = df_negative.rolling(window=window, min_periods=min_periods).quantile(lower_quantile)
    df_negative_upper = df_negative.rolling(window=window, min_periods=min_periods).quantile(upper_quantile)
    
    # 初始化归一化后的DataFrame
    df_quantile_scaled = pd.DataFrame(index=df.index, columns=df.columns)
    
    # 对正值部分进行归一化，使其在 0.5 到 1 之间
    df_quantile_scaled[positive_mask] = 0.5 + 0.5 * ((df_positive - df_positive_lower) / (df_positive_upper - df_positive_lower).replace(0, np.nan))
    
    # 对负值部分进行归一化，使其在 0 到 0.5 之间
    df_quantile_scaled[negative_mask] = 0.5 * ((df_negative - df_negative_upper) / (df_negative_lower - df_negative_upper).replace(0, np.nan))
    
    # 裁剪结果使其在 0 和 1 之间
    df_quantile_scaled = df_quantile_scaled.clip(lower=0, upper=1)
    
    return df_quantile_scaled


# %% Non-Linear
def bimodal_sin(df, power=2):
    """
    对整个 DataFrame 应用双峰变换（元素值需已归一化）：y = 1 - (sin(pi * x)) ** power

    参数:
    - df: 输入 DataFrame（每个元素都应在 [0, 1] 之间）
    - power: 幂次数，控制两头高、中间低的程度

    返回:
    - 一个新的 DataFrame，结构和原 df 相同
    """
    return 1 - (np.sin(np.pi * df)) ** power


def absv(df):
    return df.abs()
