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
import argparse


# %% add sys path
file_path = Path(__file__).resolve()
file_dir = file_path.parents[0]
project_dir = file_path.parents[1]
sys.path.append(str(project_dir))


# %% import self_defined
from core.generate_ts_trans_by_path import generate_by_path_batch_multi_factors


# %%
def main():
    '''read args'''
    parser = argparse.ArgumentParser(description='Generate factors by path batch with multiple factor configurations.')
    parser.add_argument('-b', '--batch_path_name', type=str, required=True, help='Batch path name')
    parser.add_argument('-m', '--multi_factor_name', type=str, required=True, help='Multi factor configuration name')
    
    args = parser.parse_args()
    
    generate_by_path_batch_multi_factors(
        batch_path_name=args.batch_path_name,
        multi_factor_name=args.multi_factor_name
    )
    
    
# %%
if __name__ == "__main__":
    main()