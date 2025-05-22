# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 17:16:52 2024

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %% imports
import numpy as np


# %%
def replace_column_suffixes(data, old_suffix1=".XSHE", new_suffix1=".SZ", old_suffix2=".XSHG", new_suffix2=".SH"):
    """
    替换 DataFrame 列名的后缀：
    - 将指定的 old_suffix1 替换为 new_suffix1
    - 将指定的 old_suffix2 替换为 new_suffix2

    参数:
    data (pd.DataFrame): 包含股票代码的 DataFrame，列名中含有需要替换的后缀。
    old_suffix1 (str): 要替换的第一个旧后缀，默认为 ".XSHE"。
    new_suffix1 (str): 替换的第一个新后缀，默认为 ".SZ"。
    old_suffix2 (str): 要替换的第二个旧后缀，默认为 ".XSHG"。
    new_suffix2 (str): 替换的第二个新后缀，默认为 ".SH"。

    返回:
    pd.DataFrame: 列名后缀已替换的新 DataFrame。
    """
    # 替换指定的列名后缀
    data.columns = data.columns.str.replace(old_suffix1, new_suffix1).str.replace(old_suffix2, new_suffix2)
    return data


# %%
def align_and_sort_columns(df_list):
    """
    对齐多个 DataFrame 的共同列，并按列名字母顺序重新排列。

    参数:
        df_list (list of pd.DataFrame): 包含多个 DataFrame 的列表。

    返回:
        list of pd.DataFrame: 对齐并重新排列列顺序后的 DataFrame 列表。
    """
    # 找出所有 DataFrame 的共同列
    common_cols = sorted(set.intersection(*(set(df.columns) for df in df_list)))
    
    # 按共同列重新索引每个 DataFrame
    aligned_dfs = [df[common_cols] for df in df_list]
    
    return aligned_dfs


# %%
def add_dataframe_to_dataframe_reindex(df, new_data):
    """
    使用 reindex 将新 DataFrame 的数据添加到目标 DataFrame 中，支持动态扩展列和行，原先没有值的地方填充 NaN。

    参数:
    df (pd.DataFrame): 目标 DataFrame。
    new_data (pd.DataFrame): 要添加的新 DataFrame。

    返回值:
    df (pd.DataFrame): 更新后的 DataFrame。
    """
    # 同时扩展行和列，并确保未填充的空值为 NaN，按排序
    df = df.reindex(index=df.index.union(new_data.index, sort=True),
                    columns=df.columns.union(new_data.columns, sort=True),
                    fill_value=np.nan)
    
    # 使用 loc 添加新数据
    df.loc[new_data.index, new_data.columns] = new_data

    return df


def check_dataframe_consistency(df, new_data):
    """
    使用矩阵运算检查两个DataFrame在重叠的索引部分和合并后的列上是否完全一致。
    完全一致的定义:
    - 两个值都是非NA且相等
    - 两个值都是NA
    - 如果一个值是NA而另一个不是，则视为不一致
    
    参数:
    df (pd.DataFrame): 目标 DataFrame。
    new_data (pd.DataFrame): 要检查的新 DataFrame。
    
    返回值:
    tuple: (status, info)
        - status (str): 'CONSISTENT' 表示数据一致或没有重叠；'INCONSISTENT' 表示存在不一致
        - info (dict): 当status为'INCONSISTENT'时，包含不一致的详细信息；否则为空字典
    """
    # 获取重叠的索引
    overlapping_indices = df.index.intersection(new_data.index)
    
    # 如果没有重叠的索引，直接返回一致状态
    if len(overlapping_indices) == 0:
        return "CONSISTENT", {}
    
    # 获取要检查的列（仅检查new_data中存在的列）
    columns_to_check = df.columns.intersection(new_data.columns)
    
    # 如果没有重叠的列，直接返回一致状态
    if len(columns_to_check) == 0:
        return "CONSISTENT", {}
    
    # 提取重叠部分的数据
    df_overlap = df.loc[overlapping_indices, columns_to_check]
    new_data_overlap = new_data.loc[overlapping_indices, columns_to_check]
    
    # 检查NA的一致性
    df_is_na = df_overlap.isna()
    new_is_na = new_data_overlap.isna()
    
    # NA状态应该一致（都是NA或都不是NA）
    na_inconsistent = (df_is_na != new_is_na)
    
    # 检查非NA值的一致性
    values_inconsistent = (df_overlap != new_data_overlap) & (~df_is_na) & (~new_is_na)
    
    # 合并两种不一致情况
    inconsistent_mask = na_inconsistent | values_inconsistent
    
    # 如果有不一致的元素
    if inconsistent_mask.any().any():
        # 找到第一个不一致的位置
        inconsistent_positions = [(idx, col) for idx, col in zip(
            *np.where(inconsistent_mask.values)
        )]
        
        # 获取第一个不一致的位置和值
        first_pos = inconsistent_positions[0]
        first_idx = overlapping_indices[first_pos[0]]
        first_col = columns_to_check[first_pos[1]]
        
        # 获取不一致的值
        df_value = df.loc[first_idx, first_col]
        new_value = new_data.loc[first_idx, first_col]
        
        # 创建详细信息字典
        info = {
            "index": first_idx,
            "column": first_col,
            "original_value": df_value,
            "new_value": new_value,
            "inconsistent_count": inconsistent_mask.sum().sum()
        }
        
        return "INCONSISTENT", info
    
    # 如果代码执行到这里，说明所有重叠部分都是一致的
    return "CONSISTENT", {}