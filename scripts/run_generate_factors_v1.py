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
from core.generate_factors_v1 import GenerateFactors


# %%
def main():
    '''read args'''
    parser = argparse.ArgumentParser(description='To initialize Processor.')
    parser.add_argument('-g', '--generate_name', type=str, help='Generate Method Name')
    parser.add_argument('-i', '--ind_cate', type=str, help='Indicator Category Name')
    parser.add_argument('-dir', '--ind_dir', type=str, help='Indicator Directory')
    parser.add_argument('-wkr', '--n_workers', type=int, help='Number of workers', default=1)

    args = parser.parse_args()
    args = vars(args)

    g = GenerateFactors(**args)
    g.run()


# %%
if __name__ == "__main__":
    main()
