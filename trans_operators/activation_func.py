# -*- coding: utf-8 -*-
"""
Created on Thu May 22 10:24:37 2025

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


# %%
def power(df: pd.DataFrame, gamma: float, direction: str = "pos") -> pd.DataFrame:
    """
    对 0~1 区间的数据应用幂函数变换：
    - direction = "pos"：压缩小值（靠近 0 的值更小）
    - direction = "neg"：压缩大值（靠近 1 的值更小）

    参数:
        df (pd.DataFrame): 输入 0~1 区间的 DataFrame。
        gamma (float): 幂指数，必须为正数。
        direction (str): "pos" 表示压缩小值；"neg" 表示压缩大值。

    返回:
        pd.DataFrame: 变换后的 DataFrame。
    """
    if gamma <= 0:
        raise ValueError("gamma 必须为正数")
    if direction not in {"pos", "neg"}:
        raise ValueError("direction 必须为 'pos' 或 'neg'")

    def transform(x):
        if not (0 <= x <= 1):
            return np.nan
        if direction == "pos":
            return np.power(x, gamma)
        else:  # direction == "neg"
            return np.power(1 - x, gamma)

    return df.applymap(transform)


# =============================================================================
# # 生成输入数据
# x = np.linspace(0, 1, 200)
# df_input = pd.DataFrame({"x": x})
# 
# # gamma 值列表
# gamma_list = [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 1.0]
# 
# # 画图：direction="pos"
# plt.figure(figsize=(10, 5))
# for gamma in gamma_list:
#     transformed = power(df_input, gamma, direction="pos")
#     plt.plot(x, transformed["x"], label=f"gamma={gamma}")
# plt.title("Power Transform - direction='pos'")
# plt.xlabel("x")
# plt.ylabel("Transformed x")
# plt.legend()
# plt.grid(True)
# plt.show()
# 
# # 画图：direction="neg"
# plt.figure(figsize=(10, 5))
# for gamma in gamma_list:
#     transformed = power(df_input, gamma, direction="neg")
#     plt.plot(x, transformed["x"], label=f"gamma={gamma}")
# plt.title("Power Transform - direction='neg'")
# plt.xlabel("x")
# plt.ylabel("Transformed x")
# plt.legend()
# plt.grid(True)
# plt.show()
# =============================================================================


# =============================================================================
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# 
# 
# # 生成输入数据
# x = np.linspace(0, 1, 200)
# df_input = pd.DataFrame({"x": x})
# 
# # 分别计算 pos 和 neg 变换
# df_pos = power(df_input, gamma=0.1, direction="pos")
# df_neg = power(df_input, gamma=0.05, direction="neg")
# 
# # 相乘
# df_combined = df_pos * df_neg
# 
# # 画图
# plt.figure(figsize=(10, 5))
# plt.plot(x, df_pos["x"], label="pos(gamma=0.1)", linestyle="--")
# plt.plot(x, df_neg["x"], label="neg(gamma=0.05)", linestyle="--")
# plt.plot(x, df_combined["x"], label="combined = pos * neg", linewidth=2)
# plt.title("Combined Transform: pos(0.1) * neg(0.05)")
# plt.xlabel("x")
# plt.ylabel("Transformed x")
# plt.legend()
# plt.grid(True)
# plt.show()
# 
# =============================================================================
