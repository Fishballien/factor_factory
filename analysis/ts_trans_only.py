# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 14:58:11 2025

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %%
from pathlib import Path


# %%
ind_dir = Path
# %%
# 处理第一层时序变换
ts_mapping = {}
ts_name_groups = {}  # 存储按基础转换函数名分组的ts_pr_names

for ts_name, ts_prs in ts_info.items():
    ts_pr_names = generate_factor_names(ts_name, ts_prs)
    ts_pr_list = para_allocation(ts_prs)
    ts_func = globals()[ts_name]
    
    # 按基础函数名（如'stdz', 'ma'等）分组存储所有参数组合
    ts_name_groups[ts_name] = []
    
    for ts_pr_name, ts_pr in zip(ts_pr_names, ts_pr_list):
        full_name = f"{ts_name}_{ts_pr_name}" if not ts_pr_name.startswith(f"{ts_name}_") else ts_pr_name
        ts_mapping[full_name] = partial(ts_func, **ts_pr)
        ts_name_groups[ts_name].append(full_name)