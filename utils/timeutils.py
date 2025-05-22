# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 20:27:54 2024

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %%
import re
from datetime import datetime, timedelta
import numpy as np
import pandas as pd


# %%
def parse_time_string(time_string):
    """
    解析格式为 "xxdxxhxxminxxs" 的时间字符串并转换为总秒数。

    参数:
    time_string (str): 表示时间间隔的字符串，如 "1d2h30min45s"。

    返回:
    int: 转换后的总秒数。

    异常:
    ValueError: 如果时间字符串格式无效。
    """
    # 正则模式支持 d（天），h（小时），min（分钟），s（秒）
    pattern = re.compile(r'(?:(\d+)d)?(?:(\d+)h)?(?:(\d+)min)?(?:(\d+)s)?')
    match = pattern.fullmatch(time_string)
    
    if not match:
        raise ValueError("Invalid time string format")
    
    # 将天、小时、分钟、秒提取并转换为整数
    days = int(match.group(1)) if match.group(1) else 0
    hours = int(match.group(2)) if match.group(2) else 0
    mins = int(match.group(3)) if match.group(3) else 0
    secs = int(match.group(4)) if match.group(4) else 0
    
    # 转换为总秒数
    total_seconds = days * 4 * 60 * 60 + hours * 60 * 60 + mins * 60 + secs
    return total_seconds


def get_num_of_bars(period, org_bar):
    """
    计算 period 中包含多少个 org_bar 的整除部分。

    参数:
    period (str): 时间跨度字符串，如 "2h"。
    org_bar (str): 时间周期字符串，如 "30min"。

    返回:
    int: period 中包含 org_bar 的整除倍数。
    """
    # 将 period 和 org_bar 转换为秒数
    period_seconds = parse_time_string(period)
    org_bar_seconds = parse_time_string(org_bar)
    
    # 计算整除部分
    if org_bar_seconds == 0:
        raise ValueError("The org_bar string represents zero duration.")
    
    return period_seconds // org_bar_seconds


# %%
def get_date_based_on_timestamp(ts):
    ts = pd.Timestamp(ts)  # pandas.Timestamp 兼容 datetime64 和 datetime
    return ts.strftime('%Y%m%d')
    
    
def get_curr_date():
    """
    获取当前 UTC 日期，并根据指定的逻辑返回日期。

    此函数首先获取当前的 UTC 时间（datetime.utcnow()），
    然后调用 get_date_based_on_timestamp 函数来根据时间戳确定
    返回的具体日期格式（如 yyyymmdd）。
    
    返回:
    str: 当前 UTC 时间对应的日期，格式为 yyyymmdd。
    """
    now = datetime.now()
    return get_date_based_on_timestamp(now)


# %%
def round_up_timestamp(timestamp, interval_seconds=3):
    """
    将时间戳向后取整到指定秒数的倍数。
    
    :param timestamp: int, 时间戳，单位为毫秒。
    :param interval_seconds: int, 时间间隔，单位为秒，默认为3秒。
    :return: int, 向后取整后的时间戳，单位为毫秒。
    """
    interval_ms = interval_seconds * 1000  # 转换为毫秒
    remainder = timestamp % interval_ms
    if remainder == 0:
        return timestamp
    return timestamp + (interval_ms - remainder)


# %%
def get_a_share_intraday_time_series(date: datetime, params):
    """
    生成A股市场交易时间内的等间隔时间序列。
    
    :param date: 日期 (datetime对象)
    :param params: 时间间隔参数 (字典形式，如 {'seconds': 1} 或 {'minutes': 1})
    :return: numpy数组，包含当天交易时间内的时间戳序列 (毫秒级)
    """
    # 定义A股市场交易时段
    morning_start = datetime(date.year, date.month, date.day, 9, 30)
    morning_end = datetime(date.year, date.month, date.day, 11, 30)
    afternoon_start = datetime(date.year, date.month, date.day, 13, 0)
    afternoon_end = datetime(date.year, date.month, date.day, 14, 55)

    interval = timedelta(**params)
    
    # 生成上午交易时间序列
    morning_series = np.arange(morning_start + interval, morning_end + interval, 
                               interval).astype('i8') // 1e3
    
    # 生成下午交易时间序列
    afternoon_series = np.arange(afternoon_start + interval, afternoon_end + interval, 
                                 interval).astype('i8') // 1e3
    
    # 合并上午和下午时间序列
    time_series = np.concatenate([morning_series, afternoon_series]).astype(np.int64)
    
    return time_series


# %%
if __name__=='__main__':
    test_date = datetime(2024, 1, 15)
    params = {'minutes': 1}  # 以1分钟为间隔
    ts_int = get_a_share_intraday_time_series(test_date, params)
    ts = ts_int.view('M8[ms]')
