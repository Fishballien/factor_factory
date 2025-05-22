# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 09:57:55 2024

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %% imports
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, entropy


# %%
def csmean(df):
    """
    使用 NumPy 矩阵计算 DataFrame 每行的平均值。
    """
    row_means = np.nanmean(df.to_numpy(), axis=1)
    return pd.Series(row_means, index=df.index)


# =============================================================================
# def csskew(df):
#     """
#     计算每行的偏度。
#     """
#     def row_skewness(row):
#         return pd.Series(row).skew(skipna=True)
# 
#     return df.apply(row_skewness, axis=1)
# 
# 
# def cssci(df):
#     """
#     计算符号协同性指数（SCI）。
#     """
#     def row_sci(row):
#         pos_ratio = np.sum(row > 0) / len(row)
#         neg_ratio = np.sum(row < 0) / len(row)
#         return pos_ratio - neg_ratio
# 
#     return df.apply(row_sci, axis=1)
# 
# 
# def cswvar(df):
#     """
#     计算符号加权的方差。
#     """
#     def row_weighted_variance(row):
#         row = row.dropna()
#         weights = np.sign(row) * np.abs(row)
#         mean_weighted = np.mean(weights)
#         return np.mean((weights - mean_weighted) ** 2)
# 
#     return df.apply(row_weighted_variance, axis=1)
# 
# 
# def csaci(df):
#     """
#     计算聚合性指数（ACI）。
#     """
#     def row_aci(row):
#         pairs = [(row[i], row[j]) for i in range(len(row)) for j in range(i + 1, len(row))]
#         concordance = np.sum(np.sign(x[0] * x[1]) for x in pairs) / len(pairs)
#         return concordance
# 
#     return df.apply(row_aci, axis=1)
# 
# 
# def csgci(df, alpha=1, beta=1, gamma=1):
#     """
#     计算全局协同性指数（GCI）。
#     """
#     mean_diff = csmean(df)
#     sci = cssci(df)
#     weighted_var = cswvar(df)
# 
#     return alpha * mean_diff + beta * sci + gamma * weighted_var
# 
# 
# def csswsm(df):
#     """
#     计算符号加权的二阶矩特征。
#     
#     参数:
#     df (pd.DataFrame): 输入数据框。
#     
#     返回:
#     pd.Series: 每行的符号加权二阶矩特征值。
#     """
#     def row_sign_weighted_second_moment(row):
#         row = row.dropna()
#         weights = np.sign(row) * row**2
#         total_weight = np.sum(np.abs(row)**2)
#         if total_weight == 0:
#             return 0
#         return np.sum(weights) / total_weight
# 
#     return df.apply(row_sign_weighted_second_moment, axis=1)
# =============================================================================


# %%
def cs_adjusted_cv(df):
    """
    计算相对于0.5的调整变异系数
    返回：Series，值越高表示一致性越强
    """
    # 计算与0.5的距离
    distance_from_mid = df.apply(lambda x: np.abs(x - 0.5), axis=1)
    mean_distance = distance_from_mid.mean(axis=1)
    # 计算这些距离的标准差
    std_distance = distance_from_mid.apply(lambda x: np.nanstd(x), axis=1)
    # 调整的变异系数
    return 1 - (std_distance / (mean_distance + 1e-10))  # 加小值避免除零

def cs_direction_consistency(df):
    """
    计算截面一致性方向指标
    返回：Series，正值表示一致偏多(>0.5)，负值表示一致偏空(<0.5)，范围[-1,1]
    """
    # 计算高于0.5的比例
    above_ratio = df.apply(lambda x: np.mean(x > 0.5), axis=1)
    # 计算低于0.5的比例
    below_ratio = df.apply(lambda x: np.mean(x < 0.5), axis=1)
    # 方向一致性：从-1(全部<0.5)到1(全部>0.5)
    return above_ratio - below_ratio

def cs_polarization(df):
    """
    计算截面极化指标
    返回：Series，值越高表示数据越远离0.5，范围[0,0.5]
    """
    # 计算与0.5的距离
    distance_from_mid = df.apply(lambda x: np.abs(x - 0.5), axis=1)
    # 返回平均距离，最大为0.5
    return distance_from_mid.mean(axis=1)

def cs_concentration(df):
    """
    计算截面集中度指标
    返回：Series，值越高表示一致性越强，范围[0,1]
    """
    # 计算与均值的平均绝对偏差
    mad = df.apply(lambda x: np.nanmean(np.abs(x - np.nanmean(x))), axis=1)
    # 集中度（1减去归一化的MAD）
    return 1 - (mad / 0.5)

def cs_direction(df):
    """
    计算截面方向指标
    返回：Series，正表示偏多，负表示偏空，范围[-1,1]
    """
    means = csmean(df)
    # 方向（相对于0.5的符号距离）
    return (means - 0.5) * 2  # 缩放到[-1,1]

def cs_consensus_strength(df):
    """
    计算一致性强度
    返回：Series，正值表示一致偏多，负值表示一致偏空，绝对值越大表示一致性越强，范围[-1,1]
    """
    # 与0.5的平均偏差(带符号)
    mean_deviation = df.apply(lambda x: np.mean(x - 0.5), axis=1)
    # 最大可能偏差为0.5，所以乘以2使范围为[-1,1]
    return mean_deviation * 2

def cs_skewness(df):
    """
    计算截面偏度
    返回：Series，正值表示分布偏向1，负值表示分布偏向0
    """
    return df.apply(lambda x: skew(x.dropna()), axis=1)

def cs_kurtosis(df):
    """
    计算截面峰度
    返回：Series，高值表示分布更集中
    """
    return df.apply(lambda x: kurtosis(x.dropna(), fisher=False), axis=1)

def cs_entropy(df):
    """
    计算截面熵
    返回：Series，值越低表示一致性越高
    """
    def calc_entropy(x):
        x = x.dropna()
        # 对于0-1值，先进行分箱
        hist, _ = np.histogram(x, bins=10, range=(0, 1), density=True)
        hist = hist / hist.sum()  # 归一化
        return entropy(hist + 1e-10)  # 添加小值避免log(0)
    
    return df.apply(calc_entropy, axis=1)

def cs_gini(df):
    """
    计算截面基尼系数
    返回：Series，值越高表示不平等程度越高，范围[0,1]
    """
    def gini(x):
        x = np.sort(x.dropna())
        n = len(x)
        cumx = np.cumsum(x)
        return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n if n > 0 and cumx[-1] > 0 else 0
    
    return df.apply(gini, axis=1)

def cs_iqr(df):
    """
    计算截面四分位差
    返回：Series，值越低表示一致性越高，范围[0,1]
    """
    q3 = df.apply(lambda x: np.nanquantile(x, 0.75), axis=1)
    q1 = df.apply(lambda x: np.nanquantile(x, 0.25), axis=1)
    return q3 - q1

def cs_mad(df):
    """
    计算截面平均绝对偏差
    返回：Series，值越低表示一致性越高，范围[0,0.5]
    """
    return df.apply(lambda x: np.nanmean(np.abs(x - np.nanmean(x))), axis=1)

def cs_consensus_index(df):
    """
    计算综合一致性指标
    返回：Series，正值接近1：强烈一致偏多，负值接近-1：强烈一致偏空，接近0：无明显一致性，范围[-1,1]
    """
    # 计算平均值相对于0.5的位置
    mean_position = csmean(df) - 0.5
    # 计算平均绝对偏差（越小表示一致性越高）
    mad = df.apply(lambda x: np.nanmean(np.abs(x - np.nanmean(x))), axis=1)
    # 计算一致性强度（1减去归一化MAD）
    consensus_strength = 1 - (mad / 0.5)
    # 综合指标：方向 * 强度
    return (mean_position / 0.5) * consensus_strength

def cs_above_threshold(df, threshold=0.5):
    """
    计算截面中高于阈值的比例
    返回：Series，表示高于阈值的比例，范围[0,1]
    """
    return df.apply(lambda x: np.mean(x > threshold), axis=1)

def cs_below_threshold(df, threshold=0.5):
    """
    计算截面中低于阈值的比例
    返回：Series，表示低于阈值的比例，范围[0,1]
    """
    return df.apply(lambda x: np.mean(x < threshold), axis=1)