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
os.environ["NUMBA_NUM_THREADS"] = "100"


# %%
from pathlib import Path
import pandas as pd
import numpy as np
from functools import partial
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import yaml
import traceback


from utils.datautils import replace_column_suffixes, add_dataframe_to_dataframe_reindex, check_dataframe_consistency
from utils.dirutils import load_path_config, find_common_bid_ask_files
from utils.market import index_mapping, INDEX_SEQ, INDEX_ALL
from trans_operators import *
from utils.param import para_allocation
from utils.naming import generate_factor_names
from utils.timeutils import parse_time_string


# %%
def preprocess_daily_weights(daily_weights):
    daily_weights = daily_weights.drop(columns=[col for col in daily_weights.columns if col.endswith('.XBEI')])
    daily_weights = daily_weights.replace(0, np.nan).shift(1)
    return daily_weights


def preprocess_corr_weights_v0(weights):
    # 1. 前向填充缺失值，限制最多填240个连续NaN
    weights = weights.fillna(method='ffill', limit=240)
    
    # 2. 将小于0的权重设置为0
    weights = weights.where(weights >= 0, 0)
    
    # 3. 将所有0的值设为NaN
    weights = weights.replace(0, np.nan)
    
    return weights


def preprocess_corr_weights_v1(weights):
    # 1. 前向填充缺失值，限制最多填240个连续NaN
    weights = weights.fillna(method='ffill', limit=240)
    
    # 2. 将小于0的权重设置为0
    weights = weights.where(weights >= 0.1, 0)
    
    # 3. 将所有0的值设为NaN
    weights = weights.replace(0, np.nan)
    
    return weights


def preprocess_corr_weights_v2(weights):
    # 1. 前向填充缺失值，限制最多填240个连续NaN
    weights = weights.fillna(method='ffill', limit=240)
    
    # 2. 将小于0的权重设置为0
    weights = weights.where(weights >= 0.2, 0)
    
    # 3. 将所有0的值设为NaN
    weights = weights.replace(0, np.nan)
    
    return weights


def preprocess_corr_weights_ma_v0(weights):
    # 1. 前向填充缺失值，限制最多填240个连续NaN
    weights = weights.fillna(method='ffill', limit=240)
    
    weights = weights.rolling(window=15).mean()
    
    # 2. 将小于0的权重设置为0
    weights = weights.where(weights >= 0, 0)
    
    # 3. 将所有0的值设为NaN
    weights = weights.replace(0, np.nan)
    
    return weights


# %%
def process_ts_task(ts_name, factor, factor_name, target_dir, ts_func, save_first_layer=True):
    """单个任务的处理函数，支持保存第一层结果的选项"""
    try:
        factor_ts = ts_func(factor)
        if save_first_layer:
            filename = factor_name if ts_name is None else f'{factor_name}-{ts_name}'
            factor_ts.to_parquet(target_dir / f'{filename}.parquet')
        return factor_ts
    except:
        traceback.print_exc()
        print(f"Error in processing {ts_name}")
        return None


def process_ts(factor, factor_name, ts_mapping, target_dir, ts_second_layer_mapping=None, save_first_layer=True):
    """
    使用多进程处理 ts_mapping 中的任务，并支持第二层时序变换。
    
    Args:
        factor: 输入因子数据。
        factor_name: 因子名称。
        ts_mapping: 第一层时间序列映射，键为名称，值为处理函数。
        target_dir: 输出目录。
        ts_second_layer_mapping: 第二层时间序列映射，键为名称，值为字典，包含 'base_ts' 和 'func'。
        save_first_layer: 是否保存第一层转换结果。
    """
    # 动态分配进程数
    max_workers = min(len(ts_mapping) if ts_mapping else 1, 30)
    first_layer_results = {}
    
    # 存储原始因子，以便对其应用第二层变换
    first_layer_results['org'] = factor
    
    # 处理第一层变换
    if ts_mapping:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            # 提交第一层任务
            for ts_name, ts_func in ts_mapping.items():
                task_func = partial(process_ts_task, ts_name, factor, factor_name, target_dir, ts_func, save_first_layer)
                futures[ts_name] = executor.submit(task_func)

            # 获取第一层任务结果
            for ts_name, future in tqdm(futures.items(), desc="Processing first layer"):
                try:
                    result = future.result()
                    if result is not None:
                        first_layer_results[ts_name] = result
                except Exception as e:
                    print(f"Error processing {ts_name}: {e}")
    
    # 如果有第二层变换，则处理
    if ts_second_layer_mapping:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            second_layer_futures = []
            
            for second_ts_name, config in ts_second_layer_mapping.items():
                base_ts_name = config['base_ts']
                second_ts_pr_name = config['second_ts_pr_name']
                second_ts_func = config['func']
                
                # 确保基础时序存在
                if base_ts_name in first_layer_results:
                    base_factor = first_layer_results[base_ts_name]
                    # 对于原始因子，使用原始因子名称；对于第一层变换，使用复合名称
                    if base_ts_name == 'org':
                        first_factor_name = factor_name
                    else:
                        first_factor_name = f"{factor_name}-{base_ts_name}"
                    
                    # 创建第二层任务
                    task = partial(process_ts_task, second_ts_pr_name, base_factor, first_factor_name, 
                                  target_dir, second_ts_func, True)  # 第二层结果总是保存
                    second_layer_futures.append(executor.submit(task))
            
            # 等待所有第二层任务完成
            for future in tqdm(as_completed(second_layer_futures), total=len(second_layer_futures), desc="Processing second layer"):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error in second layer processing: {e}")


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
        self._load_param()
        self._init_dir()
        self._load_daily_weights()
        self._load_selfdefined_weights()
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
        # 创建专门的org数据目录
        org_name = self.params.get('org_name', 'org')
        self.org_dir = self.factor_dir / self.ind_cate / org_name
        self.org_dir.mkdir(exist_ok=True, parents=True)
        self.debug_dir = self.factor_dir / 'debug' / self.ind_cate / self.generate_name
        self.debug_dir.mkdir(exist_ok=True, parents=True)
        
    def _load_param(self):
        with open(self.param_dir / 'from_cs' / f'{self.generate_name}.yaml', "r") as file:
            self.params = yaml.safe_load(file)
        
        # 获取数据加载位置相关配置
        self.load_from_org = self.params.get('load_from_org', True)
        self.load_from_target = self.params.get('load_from_target', True)
        
        # 获取数据保存位置相关配置
        self.save_to_org = self.params.get('save_to_org', True)
        self.save_org_to_target = self.params.get('save_org_to_target', True)
        
    def _load_daily_weights(self):
        index_list = self.params['index_weights_to_load']
        
        daily_weights = {}
        for index_name in index_list:
            index_code = index_mapping.get(index_name) or index_name
            daily_weight = pd.read_parquet(self.daily_weights_dir / f'{index_code}.parquet')
            daily_weights[index_code] = preprocess_daily_weights(daily_weight)
        self.daily_weights = daily_weights
        
    def _load_selfdefined_weights(self):
        selfdefined_weights_params = self.params.get('selfdefined_weights_params')
        weight_dir = Path(selfdefined_weights_params['weight_dir'])
        index_list = selfdefined_weights_params['index_list']
        suffix = selfdefined_weights_params['suffix']
        process_weight_func = globals()[selfdefined_weights_params['process_weight_func']]
        weights = {}
        for index_name in index_list:
            index_code = index_mapping.get(index_name) or index_name
            weight = pd.read_parquet(weight_dir / f'{index_code}_{suffix}.parquet')
            weights[index_code] = process_weight_func(weight)
        self.selfdefined_weights = weights
        
    def _get_aggregation_mapping(self):
        aggregation_info = self.params['aggregation']
        target_indexes = [index_mapping.get(index_name, index_name) for index_name in self.params.get('target_indexes', INDEX_SEQ)]
        index_all = index_mapping[self.params.get('index_all', INDEX_ALL)]
        index_seq = [index_mapping[index_name] for index_name in self.params.get('index_seq', [])]
        if len(index_seq) == 0:
            index_seq = target_indexes
        
        aggregation_mapping = {}
        for agg_name, agg_info in aggregation_info.items():
            combined_name = agg_info['aggregation']
            agg_func = globals()[combined_name]
            downscale_depth = agg_info.get('downscale_depth', None)
            imb_name = agg_info.get('imb')
            imb_func = globals()[imb_name] if imb_name else None
            ts_name = agg_info.get('ts')
            ts_func = globals()[ts_name] if ts_name else None
            ts_prs = agg_info.get('ts_prs')
            if ts_prs is not None:
                for ts_pr in ts_prs:
                    if 'window' in ts_pr and isinstance(ts_prs[ts_pr], str):
                        ts_prs[ts_pr] = int(parse_time_string(ts_prs[ts_pr]) / 60) # !!!: 待优化成由数据频率决定
            exchange = agg_info.get('exchange', 'all')
            ts_func_with_pr = partial(ts_func, **ts_prs) if ts_func is not None else None
            cs_name = agg_info.get('cs', 'csmean')
            cs_func = globals()[cs_name] if cs_name else None
            # 将参数整理成字典
            params = {
                "daily_weights": self.daily_weights,
                "target_indexes": target_indexes,
                "index_all": index_all,
                "index_seq": index_seq,
                "downscale_depth": downscale_depth,
                "imb_func": imb_func,
                "ts_func_with_pr": ts_func_with_pr,
                "cs_func": cs_func,
                "n_workers": self.n_workers,
                "exchange": exchange
            }
            
            # 检查self是否有selfdefined_weights属性，如果有则添加到params字典中
            if hasattr(self, 'selfdefined_weights'):
                params["selfdefined_weights"] = self.selfdefined_weights
            
            # 使用 ** 解包字典，将所有参数传递给 partial
            aggregation_mapping[agg_name] = partial(agg_func, **params)
        self.aggregation_mapping = aggregation_mapping
        
    def _get_ts_mapping(self):
        ts_info = self.params['ts']
        
        # 处理第一层时序变换
        ts_mapping = {}
        ts_name_groups = {}  # 存储按基础转换函数名分组的ts_pr_names
        
        for ts_name, ts_prs in ts_info.items():
            ts_pr_names = generate_factor_names(ts_name, ts_prs)
            ts_pr_list = para_allocation(ts_prs)
            ts_func = globals()[ts_name]
            
            # 按基础函数名（如'stdz', 'ma'等）分组存储所有参数组合
            ts_name_groups[ts_name] = []
            
            for ts_pr_name, ts_pr in zip(ts_pr_names, ts_pr_list):
                full_name = f"{ts_name}_{ts_pr_name}" if not ts_pr_name.startswith(f"{ts_name}_") else ts_pr_name
                ts_mapping[full_name] = partial(ts_func, **ts_pr)
                ts_name_groups[ts_name].append(full_name)
        
        self.ts_mapping = ts_mapping
        self.ts_name_groups = ts_name_groups
        
        # 处理第二层时序变换（如果存在）
        self.ts_second_layer_mapping = {}
        ts_second_layer_info = self.params.get('ts_second_layer', {})
        
        for second_ts_name, second_ts_config in ts_second_layer_info.items():
            base_ts_names = second_ts_config.get('base_ts', [])
            second_ts_prs = second_ts_config.get('params', {})
            
            # 直接使用 second_ts_name 作为函数名
            second_ts_func = globals()[second_ts_name]
            
            # 生成参数名和参数列表
            second_ts_pr_names = generate_factor_names(second_ts_name, second_ts_prs)
            second_ts_pr_list = para_allocation(second_ts_prs)
            
            # 确定基础时序变换列表
            effective_base_ts = []
            
            # 确保 'org' 总是包含在基础变换列表中以支持对原始因子应用第二层变换
            effective_base_ts.append('org')
            
            # 如果base_ts为空，则使用所有第一层变换
            if not base_ts_names:
                effective_base_ts.extend(list(ts_mapping.keys()))
            else:
                # 遍历每个base_ts，检查是否是基础函数名
                for base_name in base_ts_names:
                    if base_name in ts_name_groups:
                        # 如果是基础函数名，添加该函数的所有参数组合
                        effective_base_ts.extend(ts_name_groups[base_name])
                    elif base_name in ts_mapping:
                        # 如果是完整函数名（带参数的），直接添加
                        effective_base_ts.append(base_name)
            
            # 为每个有效的基础变换生成第二层变换
            for base_ts_name in effective_base_ts:
                for second_ts_pr_name, second_ts_pr in zip(second_ts_pr_names, second_ts_pr_list):
                    # 构建唯一标识符，包含基础变换和第二层变换的信息
                    full_name = f'{base_ts_name}-{second_ts_name}_{second_ts_pr_name}'
                    self.ts_second_layer_mapping[full_name] = {
                        'base_ts': base_ts_name,
                        'second_ts_pr_name': second_ts_pr_name,
                        'func': partial(second_ts_func, **second_ts_pr)
                    }
        
        # 获取是否保存第一层结果的配置
        self.save_first_layer = self.params.get('save_first_layer', True)
        
    def _get_generate_one_func(self):
        self.generate_one = partial(generate_one, ind_dir=self.ind_cate_dir, 
                                    target_dir=self.target_dir,
                                    org_dir=self.org_dir,
                                    aggregation_mapping=self.aggregation_mapping,
                                    ts_mapping=self.ts_mapping,
                                    ts_second_layer_mapping=self.ts_second_layer_mapping,
                                    save_first_layer=self.save_first_layer,
                                    mode=self.mode,
                                    load_from_org=self.load_from_org, load_from_target=self.load_from_target,
                                    save_to_org=self.save_to_org, save_org_to_target=self.save_org_to_target,
                                    debug_dir=self.debug_dir)
        
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


def generate_one(ind_name, ind_dir, target_dir, org_dir, aggregation_mapping, ts_mapping, 
                mode='init', ts_second_layer_mapping=None, save_first_layer=True, 
                load_from_org=True, load_from_target=True,
                save_to_org=True, save_org_to_target=True,
                debug_dir=Path('./debug')):
    """
    生成指标及其时序变换，支持精确控制数据加载和保存位置
    
    参数:
    - ind_name: 指标名称
    - ind_dir: 指标数据目录
    - target_dir: 目标输出目录
    - org_dir: 原始因子存储目录
    - aggregation_mapping: 聚合函数映射
    - ts_mapping: 时序变换映射
    - mode: 模式，'init'或'update'
    - ts_second_layer_mapping: 二级时序变换映射
    - save_first_layer: 是否保存一级时序变换结果
    - load_from_org: 是否从org_dir加载数据
    - load_from_target: 是否从target_dir加载数据
    - save_to_org: 是否保存数据到org_dir
    - save_to_target: 是否保存数据到target_dir
    - debug_dir: 调试信息目录
    """
    try:
        ind_sides = {side: pd.read_parquet(ind_dir / f'{side}_{ind_name}.parquet') # replace_column_suffixes
                     for side in ('Bid', 'Ask')}
    except:
        ind_sides = {'Bid': pd.DataFrame(), 'Ask': pd.DataFrame()}
    
    for agg_name, agg_func in aggregation_mapping.items():
        # 构建可能的因子名称
        possible_factor_names = {
            'single': {'single': f'{ind_name}-{agg_name}'}, 
            'combo': {'Bid': f'{ind_name}-{agg_name}_Bid', 'Ask': f'{ind_name}-{agg_name}_Ask'}
        }
        
        # 尝试加载现有数据
        res = None
        search_locations = []
        
        # 根据加载选项确定搜索位置
        if load_from_target:
            search_locations.append(('target', target_dir))
        if load_from_org:
            search_locations.append(('org', org_dir))
            
        # 按优先级搜索数据
        for location_name, location_dir in search_locations:
            for output_type, factor_names in possible_factor_names.items():
                path_dict = {tag: location_dir / f'{factor_name}-org.parquet'
                             for tag, factor_name in factor_names.items()}
                
                if all([os.path.exists(path) for path in path_dict.values()]):
                    if output_type == 'single':
                        res = pd.read_parquet(path_dict['single'])
                        print(f"Loading factor from {path_dict['single']} ({location_name})")
                        break
                    elif output_type == 'combo':
                        res = {tag: pd.read_parquet(factor_path) for tag, factor_path in path_dict.items()}
                        print(f"Loading factors from {', '.join(str(p) for p in path_dict.values())} ({location_name})")
                        break
            if res is not None:
                break
        
        # 如果没有找到现有数据，则计算新数据
        if res is None:
            print(f"Computing factor for {ind_name}-{agg_name}")
            res = agg_func(ind_sides)
            
        # 处理更新模式
        if mode == 'update':
            res_inc = agg_func(ind_sides)
            
            if isinstance(res, pd.DataFrame):
                # 单因子情况
                factor_name = f'{ind_name}-{agg_name}'
                status, info = check_dataframe_consistency(res, to_float32(res_inc))
                
                if status == "INCONSISTENT":
                    # 将不一致的新数据保存到debug目录
                    debug_path = debug_dir / f'{factor_name}-inconsistent.parquet'
                    to_float32(res_inc).to_parquet(debug_path)
                    
                    # 报错并退出
                    error_msg = (f"Data inconsistency detected for factor {factor_name}: "
                                f"At index {info['index']}, column {info['column']}, "
                                f"original value: {info['original_value']}, new value: {info['new_value']}. "
                                f"Total {info['inconsistent_count']} inconsistencies found. "
                                f"Debug data saved to {debug_path}")
                    raise ValueError(error_msg)
            
            elif isinstance(res, dict):
                # 组合因子情况
                for tag in list(res.keys()):
                    factor_name = f'{ind_name}-{agg_name}_{tag}'
                    status, info = check_dataframe_consistency(res[tag], to_float32(res_inc[tag]))
                    
                    if status == "INCONSISTENT":
                        # 将不一致的新数据保存到debug目录
                        debug_path = debug_dir / f'{factor_name}-inconsistent.parquet'
                        to_float32(res_inc[tag]).to_parquet(debug_path)
                        
                        # 报错并退出
                        error_msg = (f"Data inconsistency detected for factor {factor_name}: "
                                    f"At index {info['index']}, column {info['column']}, "
                                    f"original value: {info['original_value']}, new value: {info['new_value']}. "
                                    f"Total {info['inconsistent_count']} inconsistencies found. "
                                    f"Debug data saved to {debug_path}")
                        raise ValueError(error_msg)
            
            # 如果一致性检查通过或没有进行检查，则继续更新
            if isinstance(res, pd.DataFrame):
                res = add_dataframe_to_dataframe_reindex(res, to_float32(res_inc))
            elif isinstance(res, dict):
                res = {tag: add_dataframe_to_dataframe_reindex(res[tag], to_float32(res_inc[tag])) for tag in list(res.keys())}
        
        # 处理单因子或组合因子
        if isinstance(res, pd.DataFrame):
            factor_name = f'{ind_name}-{agg_name}'
            factor = to_float32(res).dropna(how='all')
            
            # 根据保存选项保存org因子
            save_locations = []
            if save_to_org:
                save_locations.append(('org', org_dir))
            if save_org_to_target:
                save_locations.append(('target', target_dir))
                
            for location_name, location_dir in save_locations:
                factor.to_parquet(location_dir / f'{factor_name}-org.parquet')
                print(f"Saved factor to {location_dir / f'{factor_name}-org.parquet'} ({location_name})")
                
            # 处理时序变换
            process_ts(factor, factor_name, ts_mapping, target_dir, ts_second_layer_mapping, save_first_layer)
        elif isinstance(res, dict):
            for res_name, factor in res.items():
                factor_name = f'{ind_name}-{agg_name}_{res_name}'
                factor = to_float32(factor).dropna(how='all')
                
                # 根据保存选项保存org因子
                save_locations = []
                if save_to_org:
                    save_locations.append(('org', org_dir))
                if save_org_to_target:
                    save_locations.append(('target', target_dir))
                    
                for location_name, location_dir in save_locations:
                    factor.to_parquet(location_dir / f'{factor_name}-org.parquet')
                    print(f"Saved factor to {location_dir / f'{factor_name}-org.parquet'} ({location_name})")
                
                # 处理时序变换
                process_ts(factor, factor_name, ts_mapping, target_dir, ts_second_layer_mapping, save_first_layer)
    return 0