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


# %% add sys path
file_path = Path(__file__).resolve()
file_dir = file_path.parents[0]
project_dir = file_path.parents[1]
sys.path.append(str(project_dir))


# %%
from utils.dirutils import load_path_config
from utils.fsutils import parallel_copy_files
from core.generate_ts_trans_by_batch import BatchGenerate


# %%
ts_batch_name = 'basis_pct_250416_org_batch_250419_batch_test_v1_s2_ts_v1'
run_batch_generate = True


# %% param: 可从配置文件读出
# =============================================================================
# select_name = 'basis_pct_250416_org_batch_250419_batch_test_v1_s2'
# ind_cate = 'basis_pct_250416'
# org_fac_name = 'IC_z2'
# 
# n_workers = 100
# mode = 'init'
# ts_trans_list = ['ts_shirley_v1']
# =============================================================================


# %%
path_config = load_path_config(project_dir)
test_result_dir = Path(path_config['test_results'])
param_dir = Path(path_config['param'])
gen_batch_config_dir = param_dir / 'generate_batch_config'
ts_batch_config_dir = param_dir / 'ts_batch'


# %% copy files
final_factors_path = test_result_dir / select_name / org_fac_name / 'all_final_factors.json'
with open(final_factors_path, 'r') as f:
    final_factors = json.load(f)

copy_list = []
for factor_paths in final_factors:
    factor_org_path = factor_paths.join('/')
    factor_target_path = factor_paths[0] / ind_cate / org_fac_name / factor_paths[2] / 'org'
    copy_list.append((factor_org_path, factor_target_path))
    
parallel_copy_files(copy_list)


# %% generate batch ts config
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
    'n_workers':n_workers,
    'mode': mode,
    'factors': factors,
    }

config_path = ts_batch_config_dir / f'{ts_batch_name}.toml'
with open(config_path, 'w') as f:
    toml.dump(ts_batch_config, f)
    
    
# %% run batch generate
g = BatchGenerate(batch_name)
g.run()
    