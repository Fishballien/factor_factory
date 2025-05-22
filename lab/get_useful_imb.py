# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 14:46:25 2024

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %% imports
import pandas as pd
import re
from collections import Counter


# %%
path = r'D:/crypto/multi_factor/factor_test_by_alpha/results/cluster/agg_241113_double3m/cluster_info_221201_241201.csv'


# %%
df = pd.read_csv(path)

# 使用正则表达式提取包含id的完整id名
df['id_extracted'] = df['factor'].apply(lambda x: re.search(r'id\d+_\d+', x).group() if re.search(r'id\d+_\d+', x) else None)

# 筛选出非空的id
id_list = df['id_extracted'].dropna().tolist()

c = Counter(id_list)