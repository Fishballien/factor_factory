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
from core.generate_batch_from_selected_basic_fac import generate_ts_batch_from_selected_basic_fac


# %%
def main():
    '''Parse command line arguments and generate TS batch'''
    parser = argparse.ArgumentParser(description='Generate TS batch from selected basic factors')
    parser.add_argument('-b', '--batch_name', type=str, required=True, help='TS batch name')
    parser.add_argument('--no-run', action='store_true', help='Generate config only without running batch')
    args = parser.parse_args()
    
    # Run the generation function
    run_batch = not args.no_run
    config_path = generate_ts_batch_from_selected_basic_fac(args.batch_name, run_batch)
    
    if not run_batch:
        print(f"TS batch config generated at: {config_path}")
        print("Batch generation was not run (--no-run flag was used)")
        

# %%
if __name__ == "__main__":
    main()