# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 10:26:01 2024

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %% imports
import sys
from pathlib import Path


# %% add sys path
file_path = Path(__file__).resolve()
file_dir = file_path.parents[0]
project_dir = file_path.parents[1]
sys.path.append(str(project_dir))


# %% import self_defined
from core.generate_factors import GenerateFactors


# %%
if __name__ == "__main__":
    args = {
            'generate_name': 'verify_complement_avg_side',
            'ind_cate': 'Batch10_fix_best_241218_selected',
            'ind_list': ['ValueTimeDecayOrderAmount_p1.0_v40000_d0.1'],
            }

    g = GenerateFactors(**args)
    g.run()

