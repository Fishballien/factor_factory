# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 11:32:05 2024

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %%
import os
from pathlib import Path
import yaml
import toml
import json


# %% import self_defined
from core.generate_factors_ts_trans import GenerateFactors
from utils.dirutils import load_path_config
from utils.logutils import FishStyleLogger


# %%
class GenerateByPathBatch:
    
    def __init__(self, batch_path_name, ind_cate, org_name, factor_list=None, n_workers=1, mode='init'):
        self.batch_path_name = batch_path_name
        self.ind_cate = ind_cate
        self.org_name = org_name
        self.factor_list = factor_list
        self.n_workers = n_workers
        self.mode = mode
        
        self._load_path_config()
        self._init_dir()
        self._load_params()
        self._init_log()
        
    def _load_path_config(self):
        file_path = Path(__file__).resolve()
        project_dir = file_path.parents[1]
        self.path_config = load_path_config(project_dir)
        
    def _init_dir(self):
        self.param_dir = Path(self.path_config['param'])
        self.final_path_dir = self.param_dir / 'final_ts_path'
        self.final_path_dir.mkdir(parents=True, exist_ok=True)
        
    def _load_params(self):
        self.params = yaml.safe_load(open(self.param_dir / 'ts_path_batch' / f'{self.batch_path_name}.yaml'))
        
    def _init_log(self):
        self.log = FishStyleLogger()
        
    def run(self):
        params = self.params
        trans_path_list = params['trans_path_list']
        
        factors = []
        final_path = []
        for trans_path in trans_path_list:
            current_level = self.org_name
            for g_name in trans_path:
                trans_combo = {
                    'generate_name': g_name,
                    'org_name': current_level,
                    }
                if trans_combo not in factors:
                    factors.append(trans_combo)
                current_level = f'{current_level}_TS_{g_name}'
            final_path.append(current_level)
            
        final_path_name = self.batch_path_name
        
        # Save finalpath to JSON in self.final_path_dir
        final_path_file = self.final_path_dir / f'{self.org_name}_{final_path_name}.json'
        # if not os.path.exists(final_path_file):
        with open(final_path_file, 'w') as f:
            json.dump({'final_path': final_path}, f, indent=4)
        
        self.log.info(f'Final path saved to {final_path_file}')
        
        for factor_info in factors:
            generate_name = factor_info['generate_name']
            org_name = factor_info['org_name']
            
            self.log.info(f'Generate Start: {generate_name} - {org_name} - {self.ind_cate}')
            g = GenerateFactors(generate_name=generate_name,
                                org_name=org_name,
                                ind_cate=self.ind_cate,
                                factor_list=self.factor_list,
                                n_workers=self.n_workers,
                                mode=self.mode)
            g.run()
            self.log.info(f'Generate Finished: {generate_name} - {org_name} - {self.ind_cate}')
            
            
def generate_by_path_batch_multi_factors(batch_path_name, multi_factor_name):
    
    file_path = Path(__file__).resolve()
    project_dir = file_path.parents[1]
    path_config = load_path_config(project_dir)
    
    log = FishStyleLogger()
    
    param_dir = Path(path_config['param'])
    params = toml.load(param_dir / 'multi_factors' / f'{multi_factor_name}.toml')
    
    n_workers = params.get('n_workers', 1)
    mode = params.get('mode', 'init')
    
    org_factors = params['org_factors']
    for org_factor_info in org_factors:
        ind_cate = org_factor_info['ind_cate']
        org_name = org_factor_info['org_name']
        factor_list = org_factor_info.get('factor_list')
        
        log.info(f'Org Factor Start: {org_name} - {ind_cate}')
        
        g = GenerateByPathBatch(
            batch_path_name=batch_path_name,
            ind_cate=ind_cate,
            org_name=org_name,
            factor_list=factor_list,
            n_workers=n_workers,
            mode=mode,
        )
        g.run()
        
        log.info(f'Org Factor Finished: {org_name} - {ind_cate}')