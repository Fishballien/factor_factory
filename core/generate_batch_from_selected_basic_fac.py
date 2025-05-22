# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 14:02:32 2025

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
import json
import toml
from typing import Optional
import os

# Set up paths
file_path = Path(__file__).resolve()
file_dir = file_path.parents[0]
project_dir = file_path.parents[1]
sys.path.append(str(project_dir))

# Import utilities
from utils.dirutils import load_path_config
from utils.fsutils import parallel_copy_files
from core.generate_ts_trans_by_batch import BatchGenerate


# %%
def generate_ts_batch_from_selected_basic_fac(ts_batch_name: str, run_batch_generate: bool = True):
    """
    Generate a TS batch configuration and optionally run batch generation.
    The function reads configuration from {ts_batch_name}.toml in the gen_batch_config_dir.
    
    Args:
        ts_batch_name: Name of the TS batch
        run_batch_generate: Whether to run the batch generation
    """
    
    # Load path configuration
    path_config = load_path_config(project_dir)
    test_result_dir = Path(path_config['test_results'])
    param_dir = Path(path_config['param'])
    select_dir = test_result_dir / 'select_basic_features'
    gen_batch_config_dir = param_dir / 'generate_batch_config'
    ts_batch_config_dir = param_dir / 'ts_batch'
    
    # Read parameters from config file
    config_path = gen_batch_config_dir / f'{ts_batch_name}.toml'
    with open(config_path, 'r') as f:
        config = toml.load(f)
    
    select_name = config['select_name']
    ind_cate = config['ind_cate']
    org_fac_name = config['org_fac_name']
    n_workers = config['n_workers']
    mode = config['mode']
    ts_trans_list = config['ts_trans_list']
    
    # Load final factors
    final_factors_path = select_dir / select_name / org_fac_name / 'all_final_factors.json'
    with open(final_factors_path, 'r') as f:
        final_factors = json.load(f)
    
    # Copy files in parallel
    copy_list = []
    for factor_paths in final_factors:
        factor_org_path = Path(f'{os.path.join(*factor_paths)}.parquet')
        factor_target_path = Path(factor_paths[0]) / ind_cate / org_fac_name / factor_paths[2] / 'org' / f'{factor_paths[2]}.parquet'
        copy_list.append((factor_org_path, factor_target_path))
    
    parallel_copy_files(copy_list)
    
    # Generate batch TS config
    new_ind_cate = f'{ind_cate}/{org_fac_name}'
    factors = []
    for factor_paths in final_factors:
        basic_fac_name = factor_paths[2]
        for ts_trans_name in ts_trans_list:
            factors.append({
                'generate_name': ts_trans_name,
                'org_name': 'org',
                'ind_cate': f'{new_ind_cate}/{basic_fac_name}'
            })
    
    ts_batch_config = {
        'n_workers': n_workers,
        'mode': mode,
        'factors': factors,
    }
    
    config_path = ts_batch_config_dir / f'{ts_batch_name}.toml'
    with open(config_path, 'w') as f:
        toml.dump(ts_batch_config, f)
    
    # Run batch generate if requested
    if run_batch_generate:
        g = BatchGenerate(ts_batch_name)
        g.run()
    
    return config_path


# %%
if __name__ == "__main__":
    # Example usage
    ts_batch_name = 'basis_pct_250416_org_batch_250419_batch_test_v1_s2_ts_v1'
    run_batch_generate = True
    generate_ts_batch_from_selected_basic_fac(ts_batch_name, run_batch_generate)