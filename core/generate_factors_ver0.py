# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 17:14:02 2024

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
import pandas as pd
import numpy as np
from functools import partial
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import yaml
import traceback


from utils.datautils import replace_column_suffixes, add_dataframe_to_dataframe_reindex
from utils.dirutils import load_path_config, find_common_bid_ask_files
from utils.market import index_mapping, INDEX_SEQ, INDEX_ALL
from trans_operators import *
from utils.param import para_allocation
from utils.naming import generate_factor_names


# %%
def generate_one(ind_name, ind_dir, target_dir, aggregation_mapping, ts_mapping, mode='init'):
    try:
        ind_sides = {side: pd.read_parquet(ind_dir / f'{side}_{ind_name}.parquet') # replace_column_suffixes
                     for side in ('Bid', 'Ask')}
    except:
        ind_sides = {'Bid': pd.DataFrame(), 'Ask': pd.DataFrame()}
    
    for agg_name, agg_func in aggregation_mapping.items():
        possible_factor_names = {
            'single': {'single': f'{ind_name}-{agg_name}'}, 
            'combo': {'Bid': f'{ind_name}-{agg_name}_Bid', 'Ask': f'{ind_name}-{agg_name}_Ask'}
            }
        possible_factor_paths = {output_type: {tag: target_dir / f'{factor_name}-org.parquet'
                                               for tag, factor_name in factor_names.items()}
                                 for output_type, factor_names in possible_factor_names.items()}
        
        res = None
        for output_type, path_dict in possible_factor_paths.items():
            if all([os.path.exists(path) for path in path_dict.values()]):
                if output_type == 'single':
                    res = pd.read_parquet(path_dict['single'])
                elif output_type == 'combo':
                    res = {tag: pd.read_parquet(factor_path) for tag, factor_path in path_dict.items()}
                else:
                    NotImplementedError()
        if res is None:
            res = agg_func(ind_sides)
        if mode == 'update':
            res_inc = agg_func(ind_sides)
            if isinstance(res, pd.DataFrame):
                res = add_dataframe_to_dataframe_reindex(res, to_float32(res_inc))
            elif isinstance(res, dict):
                res = {tag: add_dataframe_to_dataframe_reindex(res[tag], to_float32(res_inc[tag])) for tag in list(res.keys())}
            
        if isinstance(res, pd.DataFrame):
            factor_name = f'{ind_name}-{agg_name}'
            factor = to_float32(res).dropna(how='all')
            factor.to_parquet(target_dir / f'{factor_name}-org.parquet')
            process_ts(factor, factor_name, ts_mapping, target_dir)
        elif isinstance(res, dict):
            for res_name, factor in res.items():
                factor_name = f'{ind_name}-{agg_name}_{res_name}'
                factor = to_float32(factor).dropna(how='all')
                factor.to_parquet(target_dir / f'{factor_name}-org.parquet')
                process_ts(factor, factor_name, ts_mapping, target_dir)
    return 0


# =============================================================================
# def process_ts(factor, factor_name, ts_mapping, target_dir):
#     for ts_name, ts_func in ts_mapping.items():
#         factor_ts = ts_func(factor)
#         factor_ts.to_parquet(target_dir / f'{factor_name}-{ts_name}.parquet')
# =============================================================================
        
        
def process_ts_task(ts_name, factor, factor_name, target_dir, ts_func):
    """单个任务的处理函数"""
    try:
        factor_ts = ts_func(factor)
        factor_ts.to_parquet(target_dir / f'{factor_name}-{ts_name}.parquet')
    except:
        traceback.print_exc()
        print(ts_name)


def process_ts(factor, factor_name, ts_mapping, target_dir):
    """
    使用多进程处理 ts_mapping 中的任务。
    
    Args:
        factor: 输入因子数据。
        factor_name: 因子名称。
        ts_mapping: 时间序列映射，键为名称，值为处理函数。
        target_dir: 输出目录。
    """
    # 动态分配进程数
    max_workers = min(len(ts_mapping), 30)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for ts_name, ts_func in ts_mapping.items():
            # 使用 partial 生成新函数，绑定 ts_func
            task_func = partial(process_ts_task, ts_name, factor, factor_name, target_dir, ts_func)
            # 提交任务到进程池
            futures.append(executor.submit(task_func))

        # 等待所有任务完成
        for future in futures:
            try:
                future.result()  # 捕获异常
            except Exception as e:
                print(f"Error processing {future}: {e}")
                
                
# %%
def preprocess_daily_weights(daily_weights):
    daily_weights = daily_weights.drop(columns=[col for col in daily_weights.columns if col.endswith('.XBEI')])
    daily_weights = daily_weights.replace(0, np.nan)
    return daily_weights
    

# %%
class GenerateFactors:
    
    target_folder = {
        'init': 'cs',
        'update': 'incremental_cs',
        }
    
    def __init__(self, generate_name, ind_cate, ind_list=None, ind_dir=None, n_workers=1, mode='init'):
        self.generate_name = generate_name
        self.ind_cate = ind_cate
        self.ind_list = ind_list
        self.ind_dir = Path(ind_dir) if ind_dir is not None else None
        self.n_workers = n_workers
        self.mode = mode
        
        self._load_path_config()
        self._init_dir()
        self._load_param()
        self._load_daily_weights()
        self._get_aggregation_mapping()
        self._get_ts_mapping()
        self._get_generate_one_func()
        self._get_ind_list()
        
    def _load_path_config(self):
        project_dir = Path(__file__).resolve().parents[1]
        path_config = load_path_config(project_dir)
        
        self.lob_ind_dir = Path(path_config['lob_indicators'])
        self.factor_dir = Path(path_config['factors'])
        self.daily_weights_dir = Path(path_config['daily_weights'])
        self.param_dir = Path(path_config['param'])
        
    def _init_dir(self):
        self.ind_cate_dir = self.ind_dir or self.lob_ind_dir / self.ind_cate / self.target_folder[self.mode]
        self.target_dir = self.factor_dir / self.ind_cate / self.generate_name
        self.target_dir.mkdir(exist_ok=True, parents=True)
        
    def _load_param(self):
        with open(self.param_dir / f'{self.generate_name}.yaml', "r") as file:
            self.params = yaml.safe_load(file)
    
    def _load_daily_weights(self):
        index_list = self.params['index_weights_to_load']
        
        daily_weights = {}
        for index_name in index_list:
            index_code = index_mapping[index_name]
            daily_weight = pd.read_parquet(self.daily_weights_dir / f'{index_code}.parquet')
            daily_weights[index_code] = preprocess_daily_weights(daily_weight)
        self.daily_weights = daily_weights
        
    def _get_aggregation_mapping(self):
        aggregation_info = self.params['aggregation']
        target_indexes = [index_mapping[index_name] for index_name in self.params.get('target_indexes', INDEX_SEQ)]
        index_all = index_mapping[self.params.get('index_all', INDEX_ALL)]
        index_seq = [index_mapping[index_name] for index_name in self.params.get('index_seq', INDEX_SEQ)]
        
        aggregation_mapping = {}
        for agg_name, agg_info in aggregation_info.items():
            combined_name = agg_info['aggregation']
            agg_func = globals()[combined_name]
            downscale_depth = agg_info.get('downscale_depth', 0)
            imb_name = agg_info.get('imb')
            imb_func = globals()[imb_name] if imb_name else None
            cs_name = agg_info.get('cs', 'csmean')
            cs_func = globals()[cs_name] if cs_name else None
            aggregation_mapping[agg_name] = partial(agg_func, daily_weights=self.daily_weights, 
                                                    target_indexes=target_indexes, index_all=index_all, 
                                                    index_seq=index_seq, downscale_depth=downscale_depth,
                                                    imb_func=imb_func, cs_func=cs_func,
                                                    n_workers=self.n_workers)
        self.aggregation_mapping = aggregation_mapping
        
    def _get_ts_mapping(self):
        ts_info = self.params['ts']
        
        ts_mapping = {}
        for ts_name, ts_prs in ts_info.items():
            ts_pr_names = generate_factor_names(ts_name, ts_prs)
            ts_pr_list = para_allocation(ts_prs)
            ts_func = globals()[ts_name]
            for ts_pr_name, ts_pr in zip(ts_pr_names, ts_pr_list):
                ts_mapping[ts_pr_name] = partial(ts_func, **ts_pr)
        self.ts_mapping = ts_mapping
        
    def _get_generate_one_func(self):
        self.generate_one = partial(generate_one, ind_dir=self.ind_cate_dir, target_dir=self.target_dir,
                                    aggregation_mapping=self.aggregation_mapping,
                                    ts_mapping=self.ts_mapping, mode=self.mode)
        
    def _get_ind_list(self):
        self.ind_list = self.ind_list or find_common_bid_ask_files(self.ind_cate_dir)
        
    def run(self):
        if self.n_workers is None or self.n_workers == 1:
            for ind_name in self.ind_list:
                self.generate_one(ind_name)
        else:
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                all_tasks = [executor.submit(self.generate_one, ind_name)
                             for ind_name in self.ind_list]
                num_of_success = 0
                for task in tqdm(as_completed(all_tasks), total=len(all_tasks), desc=f'{self.ind_cate}'):
                    res = task.result()
                    if res == 0:
                        num_of_success += 1
                print(f'num_of_success: {num_of_success}, num_of_failed: {len(self.ind_list)-num_of_success}')