# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 17:57:20 2025

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %%
import sys
from pathlib import Path


# %% add sys path
file_path = Path(__file__).resolve()
file_dir = file_path.parents[0]
project_dir = file_path.parents[1]
sys.path.append(str(project_dir))


# %%
from utils.dirutils import load_path_config


# %%
ind_cate = ''
trans_path_list = [
    ['oad_0931', 'dod_all'],
    ['oad_0931', 'dod_all', 'scale_all'],
    ['oad_0931', 'dod_01out', 'bimodal_sin', 'scale_all'],
    ['oad_0931', 'dod_not01out', 'absv', 'scale_all'],
    ['oad_0931', 'scale_all'],
    ['oad_0931', 'scale_01out', 'bimodal_sin', 'scale_all'],
    ['oad_0931', 'scale_not01out', 'absv', 'scale_all'],
    ['dod_all'],
    ['dod_all', 'scale_all'],
    ['dod_01out', 'bimodal_sin', 'scale_all'],
    ['dod_not01out', 'absv', 'scale_all'],
    ['scale_all'],
    ['scale_01out', 'bimodal_sin', 'scale_all'],
    ['scale_not01out', 'absv', 'scale_all'],
    ]


# %%
factors = []
final_path = []
for trans_path in trans_path_list:
    current_level = ''
    for g_name in trans_path:
        trans_combo = {
            'generate_name': g_name,
            'org_name': current_level,
            }
        if trans_combo not in factors:
            factors.append(trans_combo)
        current_level = f'{current_level}_TS_{g_name}'
    final_path.append(current_level)
    