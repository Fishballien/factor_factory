# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 11:08:02 2025

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
import numpy as np
import matplotlib.pyplot as plt

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

# 创建输入数据范围从0到1
x = np.linspace(0, 1, 1000)

# 为不同的power值计算函数值
powers = [2, 5, 10, 20]
plt.figure(figsize=(10, 6))

for power in powers:
    y = bimodal_sin(x, power)
    plt.plot(x, y, label=f'power = {power}')

# 设置图表属性
plt.title('Bimodal Sin Function with Different Power Values')
plt.xlabel('x (input values)')
plt.ylabel('y = 1 - (sin(π*x))^power')
plt.grid(True, alpha=0.3)
plt.legend()
plt.ylim(0, 1.1)  # 限制y轴范围，使图表更清晰

# 显示图表
plt.tight_layout()
plt.show()