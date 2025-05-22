# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 16:47:14 2024

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""

import pandas as pd

def calculate_ratio(df, n):
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
    
    result = df.copy()
    result['ratio'] = result['value'] / result['value'].shift(n, freq='D')
    
    return result

# 示例用法
# 创建一个示例 DataFrame
data = {
    'value': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
}
dates = pd.to_datetime(['2023-01-01 00:00', '2023-01-01 00:01', '2023-01-01 00:03', '2023-01-01 00:07', '2023-01-02 00:00',
                        '2023-01-02 00:01', '2023-01-02 00:02', '2023-01-03 00:00', '2023-01-03 00:01', '2023-01-04 00:00'])
df = pd.DataFrame(data, index=dates)

# 计算每个时间戳上的值 = 当前时间戳数据 / 前3天同一时间戳的数据
n = 3
result_df = calculate_ratio(df, n)
print(result_df)
