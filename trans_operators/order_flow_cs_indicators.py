# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 16:58:15 2024

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %%
'''
新增:
    - 自等权
    - 自加权
    - 下沉等权（可选下沉阶数）
    - 下沉各自加权取平均（可选下沉阶数）
'''
# %%
__all__ = ['fund_flow_metrics']


# %% imports
import pandas as pd
from tqdm import tqdm


# %%
# index_seq = ['000300', '000905', '000852', '932000']


# %%
def fund_flow_metrics(ind_sides, target_indexes, daily_weights, index_all, index_seq, downscale_depth, 
                      imb_func, ts_func_with_pr, cs_func, n_workers=1, exchange='all'):
    """
    统一的资金流指标计算函数
    
    Parameters:
    ind_sides (dict): 包含'Bid'和'Ask'的字典，值为DataFrame
    target_indexes (list): 目标指数列表
    daily_weights (dict): 日权重字典
    index_all (str): 全市场指数名称
    index_seq (list): 指数序列
    downscale_depth: 下沉深度
    imb_func: imbalance计算函数（推荐使用imb06：ask-bid）
    ts_func_with_pr: 时序函数（如果需要）
    cs_func: 截面函数（选择具体的资金流指标函数）
    n_workers (int): 工作进程数
    exchange (str): 交易所选择
    
    Returns:
    pd.DataFrame: 指标计算结果
    """
    # 导入必要的函数
    from .aggregation_new import (filter_weights_by_exchange, normalize_daily_weights, 
                                 get_merged_binary_weight_by_depth, apply_norm_daily_weights_to_timeseries)
    
    # 根据交易所过滤权重
    filtered_weights = filter_weights_by_exchange(daily_weights, exchange)
    
    res = {}
    norm_daily_weights = {index_code: normalize_daily_weights(daily_weight) 
                          for index_code, daily_weight in filtered_weights.items()}
    
    iter_ = tqdm(target_indexes, desc=f'fund_flow_metrics by indexes ({exchange})') if n_workers == 1 else target_indexes
    for index_name in iter_:
        weight = get_merged_binary_weight_by_depth(norm_daily_weights, index_name, 
                                                   index_all, index_seq, downscale_depth)
        
        # 计算资金流差值 (通常是 ask - bid)
        fund_flow = imb_func(ind_sides['Bid'], ind_sides['Ask'])
        
        # 应用权重
        adj_fund_flow = apply_norm_daily_weights_to_timeseries(fund_flow, weight)
        
        # 应用时序函数（如果有）
        if ts_func_with_pr is not None:
            adj_fund_flow = ts_func_with_pr(adj_fund_flow)
        
        # 应用截面函数计算指标
        res[index_name] = cs_func(adj_fund_flow)
    
    return pd.DataFrame(res)