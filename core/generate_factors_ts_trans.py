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
from functools import partial
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import yaml
import traceback
import warnings
warnings.filterwarnings("ignore")


from utils.datautils import add_dataframe_to_dataframe_reindex, check_dataframe_consistency
from utils.dirutils import load_path_config, list_parquet_files
from trans_operators import *
from utils.param import para_allocation
from utils.naming import generate_factor_names


# %%
class GenerateFactors:
    
    def __init__(self, generate_name, org_name, ind_cate, factor_list=None, org_dir=None, target_dir=None, n_workers=1, mode='init'):
        self.generate_name = generate_name
        self.org_name = org_name
        self.ind_cate = ind_cate
        self.factor_list = factor_list
        self.n_workers = n_workers
        self.mode = mode
        self.org_dir_input = org_dir
        self.target_dir_input = target_dir
        
        self._load_path_config()
        self._load_param()
        self._init_dir()
        self._get_ts_mapping()
        self._get_factor_list()
        self._get_generate_one_func()
        
    def _load_path_config(self):
        project_dir = Path(__file__).resolve().parents[1]
        path_config = load_path_config(project_dir)
        
        self.factor_dir = Path(path_config['factors'])
        self.param_dir = Path(path_config['param'])
        
    def _init_dir(self):
        # 如果指定了org_dir和target_dir则使用，否则使用默认值
        if self.org_dir_input is not None:
            self.org_dir = Path(self.org_dir_input)
        else:
            self.org_dir = self.factor_dir / self.ind_cate / self.org_name
            
        if self.target_dir_input is not None:
            self.target_dir = Path(self.target_dir_input)
        else:
            self.target_dir = self.factor_dir / self.ind_cate / f'{self.org_name}_TS_{self.generate_name}'
            
        self.target_dir.mkdir(exist_ok=True, parents=True)
        self.debug_dir = self.target_dir / 'debug'
        self.debug_dir.mkdir(exist_ok=True, parents=True)
        
    def _load_param(self):
        with open(self.param_dir / 'ts' / f'{self.generate_name}.yaml', "r") as file:
            self.params = yaml.safe_load(file)

    def _get_ts_mapping(self):
        ts_info = self.params['ts']
        
        # 处理第一层时序变换
        ts_mapping = {}
        ts_name_groups = {}  # 存储按基础转换函数名分组的ts_pr_names
        
        for ts_name, ts_prs in ts_info.items():
            ts_pr_names = generate_factor_names(ts_name, ts_prs) if len(ts_prs) > 0 else [ts_name]
            ts_pr_list = para_allocation(ts_prs) if len(ts_prs) > 0 else [{}]
            ts_func = globals()[ts_name]
            
            # 按基础函数名（如'stdz', 'ma'等）分组存储所有参数组合
            ts_name_groups[ts_name] = []
            
            for ts_pr_name, ts_pr in zip(ts_pr_names, ts_pr_list):
                full_name = ts_pr_name
                ts_mapping[full_name] = partial(ts_func, **ts_pr)
                ts_name_groups[ts_name].append(full_name)
        
        self.ts_mapping = ts_mapping
        
    def _get_generate_one_func(self):
        self.generate_one = partial(generate_ts_transform,
                                    org_dir=self.org_dir,
                                    target_dir=self.target_dir,
                                    ts_mapping=self.ts_mapping,
                                    debug_dir=self.debug_dir)
        
    def _get_factor_list(self):
        self.factor_list = self.factor_list or list_parquet_files(self.org_dir)
        
    def run(self):
        if self.n_workers is None or self.n_workers == 1:
            for factor_name in self.factor_list:
                self.generate_one(factor_name)
        else:
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                all_tasks = [executor.submit(self.generate_one, factor_name)
                             for factor_name in self.factor_list]
                num_of_success = 0
                for task in tqdm(as_completed(all_tasks), total=len(all_tasks), desc=f'{self.org_name}_TS_{self.generate_name}'):
                    res = task.result()
                    if res == 0:
                        num_of_success += 1
                print(f'num_of_success: {num_of_success}, num_of_failed: {len(self.factor_list)-num_of_success}')


# %%
def generate_ts_transform(org_factor_name, org_dir, target_dir, ts_mapping, debug_dir=Path('./debug')):
    """
    只对原始因子进行时序变换，不做聚合操作
    
    参数:
    - org_factor_name: 原始因子名称
    - org_dir: 原始因子存储目录
    - target_dir: 目标输出目录(时序变换结果保存位置)
    - ts_mapping: 时序变换映射
    - debug_dir: 调试信息目录
    """
    try:
        # 从org_dir加载指定的原始因子
        org_factor_path = org_dir / f'{org_factor_name}.parquet'
        factor = pd.read_parquet(org_factor_path)
        # print(f"Loading factor from {org_factor_path}")
    except Exception as e:
        print(f"Error loading factor {org_factor_name}: {e}")
        return 1
    
    # 确保数据类型为float32并删除全NA的行
    factor = to_float32(factor).dropna(how='all')
    
    # 直接处理时序变换，只做一层变换
    process_ts(factor, org_factor_name, ts_mapping, target_dir, debug_dir)
    
    return 0


def process_ts_task(ts_name, factor, factor_name, target_dir, ts_func, debug_dir=Path('./debug')):
    """
    单个时序变换任务的处理函数，添加一致性检查
    
    参数:
    - ts_name: 时序变换名称
    - factor: 原始因子数据
    - factor_name: 因子名称
    - target_dir: 目标输出目录
    - ts_func: 时序变换函数
    - debug_dir: 调试信息目录
    """
    try:
        # 计算时序变换结果
        factor_ts = ts_func(factor)
        
        # 定义输出路径
        output_path = target_dir / f'{factor_name}-{ts_name}.parquet'
        
        # 检查是否存在已有的因子数据
        if os.path.exists(output_path):
            # 读取已有的因子数据
            existing_factor_ts = pd.read_parquet(output_path)
            
            # 检查数据一致性
            status, info = check_dataframe_consistency(existing_factor_ts, factor_ts)
            
            if status == "INCONSISTENT":
                # 将不一致的新数据保存到debug目录
                debug_path = debug_dir / f'{factor_name}-{ts_name}-inconsistent.parquet'
                factor_ts.to_parquet(debug_path)
                
                # 报错
                error_msg = (f"Data inconsistency detected for factor {factor_name}-{ts_name}: "
                            f"At index {info['index']}, column {info['column']}, "
                            f"original value: {info['original_value']}, new value: {info['new_value']}. "
                            f"Total {info['inconsistent_count']} inconsistencies found. "
                            f"Debug data saved to {debug_path}")
                print(error_msg)
                return False  # 返回失败状态
            else:
                # 数据一致，进行更新（添加新数据）
                factor_ts = add_dataframe_to_dataframe_reindex(existing_factor_ts, factor_ts)
        
        # 保存结果
        factor_ts.to_parquet(output_path)
        return True  # 返回成功状态
        
    except Exception as e:
        traceback.print_exc()
        print(f"Error processing {ts_name} for {factor_name}: {e}")
        return False  # 返回失败状态


def process_ts(factor, factor_name, ts_mapping, target_dir, debug_dir):
    """
    使用多进程处理 ts_mapping 中的任务。
    
    Args:
        factor: 输入因子数据。
        factor_name: 因子名称。
        ts_mapping: 时间序列映射，键为名称，值为处理函数。
        target_dir: 输出目录。
    """
    # 动态分配进程数
    max_workers = min(len(ts_mapping), 200)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for ts_name, ts_func in ts_mapping.items():
            # 使用 partial 生成新函数，绑定 ts_func
            task_func = partial(process_ts_task, ts_name, factor, factor_name, target_dir, ts_func, debug_dir)
            # 提交任务到进程池
            futures.append(executor.submit(task_func))

        # 等待所有任务完成
        for future in futures:
            try:
                future.result()  # 捕获异常
            except Exception as e:
                print(f"Error processing {future}: {e}")

