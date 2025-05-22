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
from pathlib import Path
import toml


# %% import self_defined
from core.generate_factors import GenerateFactors
from utils.dirutils import load_path_config
from utils.logutils import FishStyleLogger


# %%
class BatchGenerate:
    
    def __init__(self, batch_name):
        self.batch_name = batch_name
        
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
        
    def _load_params(self):
        self.params = toml.load(self.param_dir / 'batch' / f'{self.batch_name}.toml')
        
    def _init_log(self):
        self.log = FishStyleLogger()

    def run(self):
        params = self.params
        factors = params['factors']
        n_workers = params['n_workers']
        mode = params['mode']
        
        for factor_info in factors:
            generate_name = factor_info['generate_name']
            ind_cate = factor_info['ind_cate']
            ind_list = factor_info.get('ind_list')
            ind_dir = factor_info.get('ind_dir')
            
            self.log.info(f'Generate Start: {generate_name} - {ind_cate}')
            g = GenerateFactors(generate_name=generate_name,
                                ind_cate=ind_cate,
                                ind_list=ind_list,
                                ind_dir=ind_dir,
                                n_workers=n_workers,
                                mode=mode)
            g.run()
            self.log.info(f'Generate Finished: {generate_name} - {ind_cate}')