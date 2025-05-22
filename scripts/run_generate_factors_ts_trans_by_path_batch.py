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
from core.generate_ts_trans_by_path import GenerateByPathBatch


# %%
def main():
    '''read args'''
    parser = argparse.ArgumentParser(description='Generate factors by path batch.')
    parser.add_argument('-b', '--batch_path_name', type=str, required=True, help='Batch path name')
    parser.add_argument('-i', '--ind_cate', type=str, required=True, help='Industry category')
    parser.add_argument('-o', '--org_name', type=str, required=True, help='Original name')
    parser.add_argument('-f', '--factor_list', type=str, nargs='+', help='Factor list')
    parser.add_argument('-n', '--n_workers', type=int, default=1, help='Number of workers')
    parser.add_argument('-m', '--mode', type=str, default='init', help='Mode: init or update')
    
    args = parser.parse_args()
    
    g = GenerateByPathBatch(
        batch_path_name=args.batch_path_name,
        ind_cate=args.ind_cate,
        org_name=args.org_name,
        factor_list=args.factor_list,
        n_workers=args.n_workers,
        mode=args.mode
    )
    g.run()
    
    
# %%
if __name__ == "__main__":
    main()